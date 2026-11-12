from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bankgen.context import Context
from bankgen.tables.loans import _annuity_payment, _balance_after

pytestmark = pytest.mark.unit


def test_annuity_amortises_to_zero() -> None:
    principal = np.array([100_000.0])
    rate = np.array([4.0])
    n = np.array([120])
    pay = _annuity_payment(principal, rate, n)
    assert 1_000 < pay[0] < 1_100
    assert _balance_after(principal, rate, n, np.array([120]), np.array([False]))[0] == pytest.approx(0, abs=1)
    assert _balance_after(principal, rate, n, np.array([60]), np.array([True]))[0] == 100_000


def test_loans_and_schedule_agree(generated: tuple[str, Context, Path]) -> None:
    key, ctx, _ = generated
    loans = ctx.table("loans").to_pydict()
    sched = ctx.table("loan_balances_monthly").to_pydict()
    assert set(sched["loan_id"]) <= set(loans["loan_id"])
    assert all(m <= 120 for m in sched["days_past_due"])
    assert all(o < m for o, m in zip(loans["origination_date"], loans["maturity_date"], strict=True))
    if key == "cauchy":
        assert all(
            lbl in "ABCDEFG" for lbl, p in zip(loans["energy_label"], loans["product"], strict=True) if p == "mortgage"
        )
