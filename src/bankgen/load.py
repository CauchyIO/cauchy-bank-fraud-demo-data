"""Load a profile's Parquet into one DuckDB file per bank. The DuckDB file is the local warehouse, described by
the data dictionary (`bankgen.dictionary`): every table, view and column carries its comment."""

from __future__ import annotations

from pathlib import Path

import duckdb

from bankgen.dictionary import comment_statements, describe
from bankgen.generate import TABLE_ORDER

VIEWS = {
    # convenience views for the questions most analyses start with
    "v_customer_products": """
        SELECT c.customer_id,
               COUNT(DISTINCT a.account_id)  AS deposit_accounts,
               COUNT(DISTINCT l.loan_id)     AS loans,
               COUNT(DISTINCT i.policy_id)   AS policies,
               COUNT(DISTINCT a.product) + COUNT(DISTINCT l.product) + COUNT(DISTINCT i.product) AS distinct_products
        FROM customers c
        LEFT JOIN accounts a ON a.customer_id = c.customer_id
        LEFT JOIN loans l ON l.customer_id = c.customer_id
        LEFT JOIN insurance_policies i ON i.customer_id = c.customer_id AND i.status = 'active'
        GROUP BY c.customer_id""",
    "v_monthly_deposits": """
        SELECT b.month, b.month_index, a.product, c.region_id, c.segment, a.origin,
               SUM(b.end_balance) AS end_balance, SUM(b.interest_paid) AS interest_paid,
               COUNT(*) AS accounts
        FROM account_balances_monthly b
        JOIN accounts a USING (account_id)
        JOIN customers c ON c.customer_id = a.customer_id
        GROUP BY ALL""",
    "v_monthly_loans": """
        SELECT m.month, m.month_index, l.product, l.collateral_type, l.origin, l.rate_type,
               c.region_id,
               SUM(m.outstanding_balance) AS outstanding_balance,
               SUM(m.interest_income) AS interest_income,
               SUM(CASE WHEN m.days_past_due >= 30 THEN m.outstanding_balance ELSE 0 END) AS delinquent_balance,
               SUM(CASE WHEN m.status = 'charged_off' THEN m.outstanding_balance ELSE 0 END) AS charged_off_balance,
               COUNT(*) AS loans
        FROM loan_balances_monthly m
        JOIN loans l USING (loan_id)
        JOIN customers c ON c.customer_id = l.customer_id
        GROUP BY ALL""",
}


def load(profile_key: str, out_root: Path, *, db_path: Path | None = None) -> Path:
    src = out_root / profile_key
    db = db_path or out_root / f"{profile_key}.duckdb"
    if db.exists():
        db.unlink()
    con = duckdb.connect(str(db))
    try:
        for name in TABLE_ORDER:
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{src / name}.parquet')")
        con.execute(
            f"CREATE TABLE transactions AS SELECT * FROM read_parquet('{src / 'transactions'}/*.parquet') ORDER BY ts"
        )
        for name, sql in VIEWS.items():
            con.execute(f"CREATE VIEW {name} AS {sql}")
        described = describe(profile_key)
        for relation in described.tables:
            for statement in comment_statements(described, relation):
                con.execute(statement)
    finally:
        con.close()
    return db
