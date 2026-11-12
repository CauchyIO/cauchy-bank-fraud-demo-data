"""Validation: structure (counts, keys, FKs) and story (each event visible with one query)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from bankgen.generate import TABLE_ORDER
from bankgen.profiles import get_profile


@dataclass(frozen=True)
class Check:
    name: str
    sql: str  # must return one row: (ok BOOLEAN, detail VARCHAR)
    kind: str = "structure"


FK = [
    ("customers", "region_id", "regions"),
    ("customers", "partner_id", "partners"),
    ("accounts", "customer_id", "customers"),
    ("loans", "customer_id", "customers"),
    ("loans", "partner_id", "partners"),
    ("insurance_policies", "customer_id", "customers"),
    ("transactions", "account_id", "accounts"),
    ("transactions", "customer_id", "customers"),
    ("transactions", "merchant_id", "merchants"),
    ("transactions", "fraud_case_id", "fraud_cases"),
    ("fraud_cases", "account_id", "accounts"),
    ("account_balances_monthly", "account_id", "accounts"),
    ("loan_balances_monthly", "loan_id", "loans"),
    ("partners", "region_id", "regions"),
    ("merchants", "region_id", "regions"),
]
PK = {
    "customers": "customer_id",
    "accounts": "account_id",
    "loans": "loan_id",
    "insurance_policies": "policy_id",
    "fraud_cases": "case_id",
    "transactions": "txn_id",
    "partners": "partner_id",
    "merchants": "merchant_id",
    "regions": "region_id",
}


def _fk_checks() -> list[Check]:
    out = []
    for child, col, parent in FK:
        pk = PK[parent]
        out.append(
            Check(
                f"fk {child}.{col} -> {parent}",
                f"SELECT COUNT(*) = 0, 'orphans: ' || COUNT(*) FROM {child} c "
                f"LEFT JOIN {parent} p ON c.{col} = p.{pk} WHERE c.{col} <> '' AND p.{pk} IS NULL",
            )
        )
    for t, pk in PK.items():
        out.append(
            Check(
                f"pk {t}.{pk} unique",
                f"SELECT COUNT(*) = COUNT(DISTINCT {pk}), 'rows ' || COUNT(*) FROM {t}",
            )
        )
    return out


COMMON_STRUCTURE = [
    Check(
        "every customer holds an account",
        "SELECT COUNT(*) = 0, 'without: ' || COUNT(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM accounts a WHERE a.customer_id = c.customer_id)",
    ),
    Check(
        "ledger reconciles to monthly balances",
        "SELECT ABS(SUM(x)) < 1.0, 'abs diff ' || ROUND(ABS(SUM(x)), 2) FROM (SELECT b.account_id, b.month, b.end_balance - b.opening_balance - COALESCE(t.s, 0) AS x FROM account_balances_monthly b LEFT JOIN (SELECT account_id, month, SUM(amount) s FROM transactions GROUP BY 1, 2) t USING (account_id, month))",
    ),
    Check(
        # a closed account's balance runs to zero in its closing month, and it has no row after it
        "a closed account ends at zero in its closing month",
        "SELECT COUNT(*) FILTER (WHERE b.end_balance <> 0) = 0 AND COUNT(*) FILTER (WHERE b.account_id IS NULL) = 0"
        " AND (SELECT COUNT(*) FROM account_balances_monthly x JOIN accounts y USING (account_id)"
        " WHERE x.month > substr(CAST(y.closed_date AS STRING), 1, 7)) = 0,"
        " 'closed ' || COUNT(*) || ', not zero ' || COUNT(*) FILTER (WHERE b.end_balance <> 0)"
        " || ', positive ' || COUNT(*) FILTER (WHERE b.end_balance > 0)"
        " || ', no closing row ' || COUNT(*) FILTER (WHERE b.account_id IS NULL)"
        " FROM accounts a LEFT JOIN account_balances_monthly b"
        " ON b.account_id = a.account_id AND b.month = substr(CAST(a.closed_date AS STRING), 1, 7)"
        " WHERE a.closed_date IS NOT NULL",
    ),
    Check(
        "no transaction outside the window",
        "SELECT COUNT(*) = 0, 'outside: ' || COUNT(*) FROM transactions WHERE ts < '2024-10-01' OR ts >= '2026-10-01'",
    ),
    Check(
        "no transaction before its account opened",
        "SELECT COUNT(*) = 0, 'early: ' || COUNT(*) FROM transactions t JOIN accounts a USING (account_id) WHERE CAST(t.ts AS DATE) < a.opened_date - INTERVAL 1 DAY",
    ),
    Check(
        "no transaction after its account closed",
        "SELECT COUNT(*) = 0, 'late: ' || COUNT(*) FROM transactions t JOIN accounts a USING (account_id) WHERE CAST(t.ts AS DATE) > a.closed_date",
    ),
    Check(
        "segments are skewed, not flat",
        "SELECT MAX(n) > 2.5 * MIN(n), 'max/min ' || ROUND(MAX(n) / MIN(n), 1) FROM (SELECT COUNT(*) n FROM customers GROUP BY segment)",
    ),
    Check(
        "amounts are long-tailed",
        "SELECT p99 > 8 * med, 'p99/median ' || ROUND(p99 / med, 1) FROM (SELECT quantile_cont(ABS(amount), 0.5) med, quantile_cont(ABS(amount), 0.99) p99 FROM transactions WHERE category NOT IN ('interest'))",
    ),
    Check(
        "December spending peaks",
        "SELECT dec_n > 1.08 * avg_other, 'dec/other ' || ROUND(dec_n / avg_other, 2) FROM (SELECT AVG(CASE WHEN month LIKE '%-12' THEN n END) AS dec_n, AVG(CASE WHEN month NOT LIKE '%-12' THEN n END) avg_other FROM (SELECT month, COUNT(*) n FROM transactions t JOIN merchants m USING (merchant_id) GROUP BY 1))",
    ),
    Check(
        "fraud losses tie to fraud transactions",
        "SELECT ABS(COALESCE(SUM(f.amount) + SUM(t.a), 0)) < 1, 'diff ' || ROUND(COALESCE(SUM(f.amount) + SUM(t.a), 0), 2) FROM fraud_cases f JOIN (SELECT fraud_case_id, SUM(amount) a FROM transactions WHERE is_fraud GROUP BY 1) t ON t.fraud_case_id = f.case_id",
    ),
]

STORY = {
    "cauchy": [
        Check(
            "E1 payment accounts exist only after launch and reach ~20%",
            "SELECT MIN(opened_date) >= DATE '2025-05-01' AND COUNT(DISTINCT customer_id) BETWEEN 0.15 * (SELECT COUNT(*) FROM customers) AND 0.26 * (SELECT COUNT(*) FROM customers), 'adopters ' || COUNT(DISTINCT customer_id) || ' first ' || MIN(opened_date) FROM accounts WHERE product = 'payment'",
            "story",
        ),
        Check(
            # the payment account counted out: adopters hold no more other products but churn about half as often
            "E1 adopters churn less and hold no more other products",
            "SELECT a_churn < 0.7 * o_churn AND a_prod <= o_prod, 'adopters ' || ROUND(a_prod, 2) || ' other products, churn ' || ROUND(a_churn, 3) || ' vs others ' || ROUND(o_prod, 2) || ', ' || ROUND(o_churn, 3) FROM (SELECT AVG(CASE WHEN payment_opened IS NOT NULL THEN other END) a_prod, AVG(CASE WHEN payment_opened IS NULL THEN other END) o_prod, AVG(CASE WHEN payment_opened IS NOT NULL THEN churn END) a_churn, AVG(CASE WHEN payment_opened IS NULL THEN churn END) o_churn FROM (SELECT payment_opened, other, CASE WHEN churned_date IS NOT NULL AND (payment_opened IS NULL OR churned_date > payment_opened) THEN 1.0 ELSE 0.0 END churn FROM (SELECT c.churned_date, (SELECT MIN(a.opened_date) FROM accounts a WHERE a.customer_id = c.customer_id AND a.product = 'payment') payment_opened, (SELECT COUNT(DISTINCT a.product) FROM accounts a WHERE a.customer_id = c.customer_id AND a.product <> 'payment') + (SELECT COUNT(DISTINCT l.product) FROM loans l WHERE l.customer_id = c.customer_id) + (SELECT COUNT(DISTINCT i.product) FROM insurance_policies i WHERE i.customer_id = c.customer_id AND i.status = 'active') other FROM customers c)))",
            "story",
        ),
        Check(
            "E2 savings outflow in Nov 2025 to Jan 2026",
            "SELECT ev < -3 * base, 'event net flow ' || ROUND(ev / 1e6, 2) || 'M vs base ' || ROUND(base / 1e6, 2) || 'M per month' FROM (SELECT AVG(CASE WHEN month BETWEEN '2025-11' AND '2026-01' THEN f END) ev, AVG(CASE WHEN month < '2025-11' THEN f END) base FROM (SELECT b.month, SUM(b.net_flow) f FROM account_balances_monthly b JOIN accounts a USING (account_id) WHERE a.product = 'savings' GROUP BY 1))",
            "story",
        ),
        Check(
            "E2 the rate gap opens in Nov 2025",
            "SELECT MAX(gap) >= 0.5, 'max competitor gap ' || MAX(gap) || ' pts' FROM (SELECT m.rate_pct - s.rate_pct gap FROM interest_rates s JOIN interest_rates m USING (month) WHERE s.rate_key = 'savings' AND m.rate_key = 'market_savings')",
            "story",
        ),
        Check(
            "E3 interest-only cohort is a quarter to a third of mortgages",
            "SELECT s_share BETWEEN 0.25 AND 0.35 AND seniors > 0, 'io share ' || ROUND(s_share, 3) || ', over-55 io maturing by 2032: ' || seniors FROM (SELECT AVG(is_interest_only::INT) AS s_share, SUM(CASE WHEN l.is_interest_only AND c.age >= 55 AND l.maturity_date <= DATE '2032-12-31' THEN 1 ELSE 0 END) seniors FROM loans l JOIN customers c USING (customer_id) WHERE product = 'mortgage')",
            "story",
        ),
        Check(
            "E3 interest-only mortgages maturing past pension age all mature by 2040",
            "SELECT COUNT(*) > 0 AND MAX(maturity_date) <= DATE '2040-12-31', 'last maturity ' || MAX(maturity_date) || ' of ' || COUNT(*) FROM loans WHERE product = 'mortgage' AND is_interest_only AND age_at_maturity >= 67",
            "story",
        ),
        Check(
            # sixfold on the targeted accounts; 5.3x to 8.3x over all cases across ten seeds
            "E4 fraud spikes in May-Jun 2026, over 4.5x the base",
            "SELECT ev > 4.5 * base, 'event ' || ROUND(ev, 1) || ' vs base ' || ROUND(base, 1) || ' cases/month' FROM (SELECT AVG(CASE WHEN month IN ('2026-05', '2026-06') THEN n END) ev, AVG(CASE WHEN month NOT IN ('2026-05', '2026-06') THEN n END) base FROM (SELECT month, COUNT(*) n FROM fraud_cases GROUP BY 1))",
            "story",
        ),
        Check(
            "E4 most wave scam cases on payment accounts hit accounts open six months or more",
            "SELECT older > 0.5 * total, older || ' of ' || total || ' scam cases on payment accounts in May-Jun 2026 on accounts open six months or more' FROM (SELECT COUNT(*) total, SUM(CASE WHEN a.opened_date <= CAST(f.month || '-01' AS DATE) - INTERVAL 6 MONTH THEN 1 ELSE 0 END) older FROM fraud_cases f JOIN accounts a ON a.account_id = f.account_id WHERE a.product = 'payment' AND f.month IN ('2026-05', '2026-06') AND f.fraud_type IN ('phishing', 'bank_helpdesk_scam', 'ideal_invoice_scam', 'investment_scam'))",
            "story",
        ),
        Check(
            "E5 energy label A+ is about a quarter of the book",
            "SELECT s_share BETWEEN 0.22 AND 0.30, 'A share ' || ROUND(s_share, 3) FROM (SELECT AVG((energy_label = 'A')::INT) AS s_share FROM loans WHERE product = 'mortgage')",
            "story",
        ),
        Check(
            "E6 the ECB deposit facility is on the published path from Jun 2026",
            "SELECT COUNT(*) = 4 AND BOOL_AND(rate_pct = CASE WHEN month = '2026-09' THEN 2.50 ELSE 2.25 END), STRING_AGG(month || ' ' || rate_pct, ', ' ORDER BY month) FROM interest_rates WHERE rate_key = 'ecb_dfr' AND month >= '2026-06'",
            "story",
        ),
    ],
}


def _count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    if row is None:
        raise RuntimeError(f"COUNT(*) on {table} returned no row")
    return int(row[0])


def run_checks(con: duckdb.DuckDBPyConnection, checks: list[Check]) -> list[tuple[Check, bool, str]]:
    out = []
    for c in checks:
        try:
            row = con.execute(c.sql).fetchone()
            ok, detail = (bool(row[0]), str(row[1])) if row else (False, "no row")
        except duckdb.Error as exc:
            ok, detail = False, f"query error: {exc}"
        out.append((c, ok, detail))
    return out


def validate(profile_key: str, out_root: Path, *, db_path: Path | None = None, quiet: bool = False) -> bool:
    get_profile(profile_key)
    db = db_path or out_root / f"{profile_key}.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    try:
        counts = {t: _count(con, t) for t in [*TABLE_ORDER, "transactions"]}
        results = run_checks(con, [*_fk_checks(), *COMMON_STRUCTURE, *STORY[profile_key]])
    finally:
        con.close()
    if not quiet:
        print(f"== validate {profile_key} ({db})")
        for t, n in counts.items():
            print(f"   {t:<26} {n:>12,}")
        for c, ok, detail in results:
            print(f"   [{'PASS' if ok else 'FAIL'}] {c.kind:<9} {c.name}: {detail}")
    failed = [c.name for c, ok, _ in results if not ok]
    if not quiet:
        print(f"   {len(results) - len(failed)}/{len(results)} checks passed")
    return not failed
