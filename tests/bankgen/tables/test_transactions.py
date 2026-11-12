from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from bankgen.context import Context

pytestmark = pytest.mark.unit


def test_ledger_reconciles_and_fraud_links(generated: tuple[str, Context, Path]) -> None:
    _key, _ctx, db = generated
    con = duckdb.connect(str(db), read_only=True)
    try:
        diff = con.execute(
            "SELECT ABS(SUM(b.end_balance - b.opening_balance - COALESCE(t.s, 0))) "
            "FROM account_balances_monthly b "
            "LEFT JOIN (SELECT account_id, month, SUM(amount) s FROM transactions GROUP BY 1, 2) t "
            "USING (account_id, month)"
        ).fetchone()[0]
        assert diff < 1.0
        orphans = con.execute(
            "SELECT COUNT(*) FROM transactions t LEFT JOIN accounts a USING (account_id) WHERE a.account_id IS NULL"
        ).fetchone()[0]
        assert orphans == 0
        fraud = con.execute("SELECT COUNT(*) FROM fraud_cases").fetchone()[0]
        linked = con.execute("SELECT COUNT(DISTINCT fraud_case_id) FROM transactions WHERE is_fraud").fetchone()[0]
        assert fraud == linked
    finally:
        con.close()
