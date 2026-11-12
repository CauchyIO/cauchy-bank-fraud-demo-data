from __future__ import annotations

import pytest

from bankgen.calendar import N_MONTHS
from bankgen.profiles import PROFILES, get_profile

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("key", list(PROFILES))
def test_profile_is_internally_consistent(key: str) -> None:
    p = get_profile(key)
    segs = set(p.segments)
    assert abs(sum(p.segments.values()) - 1.0) < 1e-6
    assert abs(sum(r.weight for r in p.regions) - 1.0) < 0.02
    for dp in p.deposit_products:
        assert set(dp.penetration) == segs
        assert dp.rate_path is None or dp.rate_path in p.rate_paths
    for lp in p.loan_products:
        assert set(lp.penetration) == segs
        assert lp.index_name is None or lp.index_name.lower() in p.rate_paths or lp.variable_share == 0
    for path in p.rate_paths.values():
        assert len(path) == N_MONTHS
    for ev in p.events:
        assert 0 <= ev.month_from <= ev.month_to < N_MONTHS
    assert abs(sum(p.fraud_types.values()) - 1.0) < 1e-6
    assert p.policy_rate_key in p.rate_paths


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError):
        get_profile("rabobank")


def test_payment_accounts_launch_in_may_2025() -> None:
    bank = get_profile("cauchy")
    payment = next(d for d in bank.deposit_products if d.product == "payment")
    assert payment.launch_month == 7  # 2025-05
    assert payment.is_transactional
