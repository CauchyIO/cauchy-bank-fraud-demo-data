"""
The as-of rule: the book is observed at the window end, so nothing dated after it exists and no date is
moved onto it. These tests fail if any dated event piles up on the window's last day, or if a table builder
caps a date to the window end. The rule covers what became of a loan and a fraud case: no date, timestamp or
month column holds a value after the window end except the terms a contract schedules, and the loans table
agrees with the monthly schedule. A closed account's balance runs to zero in its closing month through
transactions dated on or before the close, and it has no row after that month.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import duckdb
import pytest

from bankgen.calendar import N_MONTHS, WINDOW_END, month_end, month_start
from bankgen.generate import generate
from bankgen.load import load
from bankgen.tables.loans import MONTH_STATUSES
from bankgen.tables.transactions import CLOSING_COUNTERPARTY

pytestmark = pytest.mark.unit

# enough customers that a daily count means something; the shared 120-customer cut is too small
CUSTOMERS = 2500

# every dated event a table records: (table, column)
EVENTS = [
    ("accounts", "opened_date"),
    ("accounts", "closed_date"),
    ("customers", "churned_date"),
    ("loans", "origination_date"),
    ("insurance_policies", "start_date"),
    ("insurance_policies", "end_date"),
    ("fraud_cases", "opened_ts"),
    ("transactions", "ts"),
]

TABLES = Path(__file__).resolve().parents[2] / "src" / "bankgen" / "tables"

# the only columns that may lie after the window end: terms a contract schedules, and the pension date, which
# are known at the as-of date; any other date, timestamp or 'YYYY-MM' column ends by the window end
SCHEDULED = {
    ("accounts", "maturity_date"),
    ("customers", "aow_date"),
    ("loans", "fixed_rate_period_end"),
    ("loans", "maturity_date"),
}


def last_day_ratio(daily: dict[date, int], month_index: int) -> float:
    """Events on the month's last day against the mean of its other days (0 when the month is empty)."""
    start, end = month_start(month_index), month_end(month_index)
    days = [date.fromordinal(o) for o in range(start.toordinal(), end.toordinal() + 1)]
    others = [daily.get(d, 0) for d in days[:-1]]
    mean = sum(others) / len(others)
    return daily.get(end, 0) / max(mean, 1.0)


def spikes_on_window_end(daily: dict[date, int]) -> bool:
    """True when the window's last day is far above its month, beyond what month-ends usually show.

    Some events belong on a month-end by design (interest postings, statement runs), so the window's
    last day is judged against the eleven month-ends before it, not against a flat day."""
    now = last_day_ratio(daily, N_MONTHS - 1)
    before = max(last_day_ratio(daily, m) for m in range(N_MONTHS - 12, N_MONTHS - 1))
    return daily.get(WINDOW_END, 0) >= 10 and now > 3 * max(before, 1.0)


def test_the_detector_catches_the_old_clamp() -> None:
    """The shape the clamp left: about four openings a day in September, then 600 on the 30th."""
    flat = {date.fromordinal(o): 4 for o in range(month_start(N_MONTHS - 12).toordinal(), WINDOW_END.toordinal() + 1)}
    assert not spikes_on_window_end(flat)
    assert spikes_on_window_end({**flat, WINDOW_END: 600})


@pytest.fixture(scope="module", params=["cauchy"])
def book(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    key = str(request.param)
    out = tmp_path_factory.mktemp(f"asof-{key}")
    generate(key, out, seed=7, n_customers=CUSTOMERS, log=False)
    return key, load(key, out)


@pytest.mark.parametrize(("table", "column"), EVENTS, ids=[f"{t}.{c}" for t, c in EVENTS])
def test_no_dated_event_piles_up_on_the_window_end(book: tuple[str, Path], table: str, column: str) -> None:
    key, db = book
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(
            f"SELECT CAST({column} AS DATE) d, COUNT(*) FROM {table} WHERE {column} IS NOT NULL GROUP BY 1"
        ).fetchall()
    daily = {d: int(c) for d, c in rows}
    assert not spikes_on_window_end(daily), f"{key} {table}.{column}: {daily.get(WINDOW_END, 0)} on {WINDOW_END}"


def dated_columns(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    """Every date, timestamp and month column of the book's tables: (table, column, type)."""
    return [
        (str(t), str(c), str(d))
        for t, c, d in con.execute(
            "SELECT c.table_name, c.column_name, c.data_type FROM information_schema.columns c"
            " JOIN information_schema.tables t USING (table_schema, table_name)"
            " WHERE t.table_type = 'BASE TABLE' AND (c.data_type IN ('DATE', 'TIMESTAMP')"
            " OR (c.data_type = 'VARCHAR' AND c.column_name LIKE '%month%')) ORDER BY 1, 2"
        ).fetchall()
    ]


def test_nothing_is_dated_after_the_window_end_except_scheduled_terms(book: tuple[str, Path]) -> None:
    key, db = book
    end = WINDOW_END.isoformat()
    with duckdb.connect(str(db), read_only=True) as con:
        columns = dated_columns(con)
        late = {}
        for table, column, kind in columns:
            if (table, column) in SCHEDULED:
                continue
            after = (
                f"{column} > '{end[:7]}' AND {column} <> ''"
                if kind == "VARCHAR"
                else f"CAST({column} AS DATE) > DATE '{end}'"
            )
            (n,) = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {after}").fetchone()  # type: ignore[misc]
            if n:
                late[f"{table}.{column}"] = n
    # the scan reads the columns the book has, so a new dated column is covered without a list to maintain
    assert {(t, c) for t, c, _ in columns} >= SCHEDULED | {
        ("fraud_cases", "closed_ts"),
        ("loans", "first_delinquent_month"),
    }
    assert late == {}, key


def test_an_open_fraud_case_has_no_closing_time(book: tuple[str, Path]) -> None:
    key, db = book
    with duckdb.connect(str(db), read_only=True) as con:
        wrong = con.execute(
            "SELECT COUNT(*) FILTER (WHERE status = 'open' AND closed_ts IS NOT NULL),"
            " COUNT(*) FILTER (WHERE status = 'closed' AND closed_ts IS NULL) FROM fraud_cases"
        ).fetchone()
    assert wrong == (0, 0), key


def test_the_loans_table_agrees_with_the_monthly_schedule(book: tuple[str, Path]) -> None:
    """A loan's status at the window end is its last monthly row's: charged off in a month of the window, still
    past due, matured, or current; a loan with no row matured before the window and reads paid off."""
    key, db = book
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(
            "WITH last AS (SELECT loan_id, arg_max(status, month_index) AS status FROM loan_balances_monthly"
            " GROUP BY 1)"
            " SELECT l.status_at_window_end, l.delinquency_outcome, CASE"
            "   WHEN last.status IS NULL THEN 'none'"
            "   WHEN last.status IN ('30dpd', '60dpd', '90dpd', '120dpd') THEN 'past due'"
            "   WHEN last.status = 'matured_balloon' THEN 'paid_off' ELSE last.status END AS last_row, COUNT(*)"
            " FROM loans l LEFT JOIN last USING (loan_id) GROUP BY ALL"
        ).fetchall()
        orphan = con.execute(
            "SELECT COUNT(*) FROM loans l WHERE l.status_at_window_end = 'charged_off' AND NOT EXISTS"
            " (SELECT 1 FROM loan_balances_monthly m WHERE m.loan_id = l.loan_id AND m.status = 'charged_off')"
        ).fetchone()
    allowed = {
        ("charged_off", "charged_off", "charged_off"),
        ("delinquent", "", "past due"),
        ("paid_off", "", "paid_off"),
        ("paid_off", "cured", "paid_off"),
        ("paid_off", "", "none"),
        ("current", "", "current"),
        ("current", "cured", "current"),
    }
    found = {(str(s), str(o), str(r)) for s, o, r, _ in rows}
    assert found <= allowed, f"{key}: {sorted(found - allowed)}"
    assert orphan == (0,), key
    with duckdb.connect(str(db), read_only=True) as con:
        unpaid = con.execute(
            "SELECT COUNT(*) FROM loans l WHERE l.status_at_window_end = 'delinquent' AND NOT EXISTS"
            " (SELECT 1 FROM loan_balances_monthly m WHERE m.loan_id = l.loan_id"
            f" AND m.month_index = {N_MONTHS - 1} AND m.days_past_due > 0 AND m.outstanding_balance > 0)"
        ).fetchone()
    assert unpaid == (0,), f"{key}: a delinquent loan without its past-due balance at the window end"
    assert ("delinquent", "", "past due") in found, f"{key}: no loan still past due at the window end"


def test_the_monthly_status_is_stored_in_full(book: tuple[str, Path]) -> None:
    key, db = book
    with duckdb.connect(str(db), read_only=True) as con:
        found = {str(s) for (s,) in con.execute("SELECT DISTINCT status FROM loan_balances_monthly").fetchall()}
        balloons = con.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE l.balloon_amount <= 0) FROM loan_balances_monthly m"
            " JOIN loans l USING (loan_id) WHERE m.status = 'matured_balloon'"
        ).fetchone()
    assert found <= set(MONTH_STATUSES), f"{key}: {sorted(found - set(MONTH_STATUSES))}"
    assert balloons is not None and balloons[1] == 0, key


def test_every_customer_starts_with_a_deposit_product(book: tuple[str, Path]) -> None:
    """The relationship starts with a product: the first opens the day the customer comes on the book."""
    key, db = book
    with duckdb.connect(str(db), read_only=True) as con:
        without, late = con.execute(
            "WITH f AS (SELECT customer_id, MIN(opened_date) first_open FROM accounts GROUP BY 1)"
            " SELECT COUNT(*) FILTER (WHERE f.customer_id IS NULL),"
            " COUNT(*) FILTER (WHERE f.first_open > c.on_book_since)"
            " FROM customers c LEFT JOIN f USING (customer_id)"
        ).fetchone()
    assert (without, late) == (0, 0), key


def test_a_closed_account_ends_at_zero_in_its_closing_month(book: tuple[str, Path]) -> None:
    """No closed account holds a balance, positive or negative, in its closing month or after it."""
    key, db = book
    with duckdb.connect(str(db), read_only=True) as con:
        closed, closing_rows, not_zero, positive_at_or_after = con.execute(
            "SELECT COUNT(DISTINCT a.account_id),"
            " COUNT(*) FILTER (WHERE b.month = substr(CAST(a.closed_date AS STRING), 1, 7)),"
            " COUNT(*) FILTER (WHERE b.month = substr(CAST(a.closed_date AS STRING), 1, 7) AND b.end_balance <> 0),"
            " COUNT(*) FILTER (WHERE b.month >= substr(CAST(a.closed_date AS STRING), 1, 7) AND b.end_balance > 0)"
            " FROM accounts a LEFT JOIN account_balances_monthly b USING (account_id) WHERE a.closed_date IS NOT NULL"
        ).fetchone()  # type: ignore[misc]
        after = con.execute(
            "SELECT COUNT(*) FROM account_balances_monthly b JOIN accounts a USING (account_id)"
            " WHERE b.month > substr(CAST(a.closed_date AS STRING), 1, 7)"
        ).fetchone()
    assert closed > 0, f"{key}: no closed account to test"
    assert closing_rows == closed, f"{key}: {closed - closing_rows} closed accounts without a closing-month row"
    assert (not_zero, positive_at_or_after, after) == (0, 0, (0,)), key


def test_the_closing_month_reconciles_the_balance_to_zero(book: tuple[str, Path]) -> None:
    """The closing month's opening balance and its transactions, all dated on or before the close, sum to zero; the
    last of them takes the balance out on the closing day, and the month's net_flow leaves that payout out."""
    key, db = book
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(
            "WITH c AS (SELECT account_id, closed_date, substr(CAST(closed_date AS STRING), 1, 7) AS m FROM accounts"
            " WHERE closed_date IS NOT NULL),"
            " t AS (SELECT t.account_id, SUM(t.amount) AS total,"
            " COUNT(*) FILTER (WHERE CAST(t.ts AS DATE) > c.closed_date) AS late,"
            " arg_max(t.amount, t.ts) AS last_amount, arg_max(t.category, t.ts) AS last_category, MAX(t.ts) AS last_ts,"
            " arg_max(t.counterparty, t.ts) AS last_counterparty"
            " FROM transactions t JOIN c ON c.account_id = t.account_id AND t.month = c.m GROUP BY 1)"
            " SELECT c.account_id, b.opening_balance, COALESCE(t.total, 0), COALESCE(t.late, 0), t.last_amount,"
            " t.last_category, CAST(t.last_ts AS DATE) = c.closed_date, b.net_flow, b.interest_paid, b.end_balance,"
            " t.last_counterparty"
            " FROM c JOIN account_balances_monthly b ON b.account_id = c.account_id AND b.month = c.m"
            " LEFT JOIN t ON t.account_id = c.account_id"
        ).fetchall()
    assert rows, f"{key}: no closed account to test"
    unreconciled = [r[0] for r in rows if abs(r[1] + r[2]) >= 0.01]
    late = [r[0] for r in rows if r[3]]
    paid = [r for r in rows if r[4] is not None]
    # the closing withdrawal (a closing deposit when overdrawn) is the month's last transaction, on the closing day,
    # to or from the customer's account at another bank
    not_a_payout = [
        r[0] for r in paid if r[5] not in ("withdrawal", "deposit") or not r[6] or r[10] != CLOSING_COUNTERPARTY
    ]
    # end_balance = opening_balance + net_flow + interest_paid + the payout: the flow leaves the closing balance out
    flow_with_payout = [r[0] for r in paid if abs(r[9] - r[1] - r[7] - r[8] - r[4]) >= 0.02]
    assert (unreconciled, late, not_a_payout, flow_with_payout) == ([], [], [], []), key
    assert len(paid) > 0.9 * len(rows), f"{key}: {len(paid)} of {len(rows)} closing months with a transaction"


def test_no_table_builder_caps_a_date_to_the_window_end() -> None:
    capped = [
        f"{p.name}:{i}: {line.strip()}"
        for p in sorted(TABLES.glob("*.py"))
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"minimum\([^)]*WINDOW_END", line)
    ]
    assert capped == []
