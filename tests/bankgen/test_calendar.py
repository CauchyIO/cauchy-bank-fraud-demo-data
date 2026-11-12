from __future__ import annotations

from datetime import date

import pytest

from bankgen.calendar import (
    N_MONTHS,
    WINDOW_END,
    day_weights,
    month_end,
    month_index_of,
    month_label,
    month_start,
    seasonal_multiplier,
)

pytestmark = pytest.mark.unit


def test_window_is_two_years() -> None:
    assert month_start(0) == date(2024, 10, 1)
    assert month_label(13) == "2025-11"
    assert month_end(N_MONTHS - 1) == WINDOW_END == date(2026, 9, 30)
    assert month_index_of(date(2026, 1, 15)) == 15


def test_day_weights_peak_on_paydays() -> None:
    dw = day_weights(0, peak_days=(23, 24, 25), peak=25.0)
    assert dw.weights.sum() == pytest.approx(1.0)
    assert dw.weights[22] > 10 * dw.weights[2]


def test_december_is_the_peak_month() -> None:
    assert seasonal_multiplier(2) == max(seasonal_multiplier(m) for m in range(N_MONTHS))
