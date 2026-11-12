"""
The policy series carries the published 2026 moves. Each month holds the rate in force at its end. The ECB
deposit facility rose to 2.25% on 11 Jun 2026 and to 2.50% on 16 Sep 2026 (ECB key rates). Only the policy
series moves: the product rates keep their paths.
"""

from __future__ import annotations

import pytest

from bankgen.calendar import N_MONTHS, month_label
from bankgen.profiles import get_profile

pytestmark = pytest.mark.unit

PUBLISHED = {
    # bank, series, first month the test holds the series to, the rate in force at each later month's end
    ("cauchy", "ecb_dfr"): ("2026-06", {"2026-06": 2.25, "2026-07": 2.25, "2026-08": 2.25, "2026-09": 2.50}),
}
# the series that must not move
UNCHANGED = {
    "cauchy": {"savings": 1.20, "market_savings": 1.75, "annuity": 2.10, "mortgage_10y": 3.8},
}


@pytest.mark.parametrize(("bank", "series"), list(PUBLISHED))
def test_the_policy_series_equal_the_published_path_after_the_change(bank: str, series: str) -> None:
    p = get_profile(bank)
    assert p.policy_rate_key == series
    first, rates = PUBLISHED[(bank, series)]
    months = [month_label(i) for i in range(N_MONTHS)]
    after = [m for m in months if m >= first]
    assert after == sorted(rates), "every month from the change to the window's end is held"
    path = dict(zip(months, p.rate_paths[series], strict=True))
    assert {m: path[m] for m in after} == rates


@pytest.mark.parametrize("bank", list(UNCHANGED))
def test_the_other_rates_end_the_window_where_they_did(bank: str) -> None:
    p = get_profile(bank)
    assert {k: p.rate_paths[k][-1] for k in UNCHANGED[bank]} == UNCHANGED[bank]
