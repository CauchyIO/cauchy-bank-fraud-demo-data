"""The 24-month window and the time multipliers every fact table shares."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from bankgen.rng import FloatArray, IntArray

WINDOW_START = date(2024, 10, 1)
N_MONTHS = 24


def month_start(month_index: int) -> date:
    y, m = divmod(WINDOW_START.month - 1 + month_index, 12)
    return date(WINDOW_START.year + y, m + 1, 1)


def month_end(month_index: int) -> date:
    return month_start(month_index + 1) - timedelta(days=1)


def month_label(month_index: int) -> str:
    return month_start(month_index).strftime("%Y-%m")


WINDOW_END = month_end(N_MONTHS - 1)


def month_index_of(d: date) -> int:
    """Months since the window start; negative before it, >= N_MONTHS after it."""
    return (d.year - WINDOW_START.year) * 12 + (d.month - WINDOW_START.month)


def days_in_month(month_index: int) -> int:
    return (month_end(month_index) - month_start(month_index)).days + 1


@dataclass(frozen=True)
class DayWeights:
    """Relative weight of each day in a month for timestamp placement."""

    weights: FloatArray  # length days_in_month, sums to 1


def day_weights(
    month_index: int,
    *,
    weekend_factor: float = 0.65,
    peak_days: tuple[int, ...] = (),
    peak: float = 1.0,
) -> DayWeights:
    """Fewer events at the weekend; optional peaks on given days of the month (1-based)."""
    n = days_in_month(month_index)
    start = month_start(month_index)
    w = np.ones(n, dtype=np.float64)
    for i in range(n):
        if (start + timedelta(days=i)).weekday() >= 5:
            w[i] *= weekend_factor
    for d in peak_days:
        if 1 <= d <= n:
            w[d - 1] *= 1.0 + peak
    return DayWeights(w / w.sum())


def random_timestamps(rng: np.random.Generator, month_index: int, size: int, dw: DayWeights) -> np.ndarray:
    """Timestamps inside the month: weighted day, business-hours-skewed time of day."""
    day = rng.choice(len(dw.weights), size=size, p=dw.weights)
    # hour-of-day mixture: mostly 08:00-21:00
    hour = np.clip(np.floor(rng.normal(14.0, 4.0, size=size)), 0, 23).astype(np.int64)
    seconds = (day * 86400 + hour * 3600 + rng.integers(0, 3600, size=size)).astype(np.int64)
    base = np.datetime64(datetime.combine(month_start(month_index), datetime.min.time()), "s")
    return base + seconds.astype("timedelta64[s]")


def seasonal_multiplier(month_index: int) -> float:
    """Retail spending seasonality: December peak, January and February trough, summer bump."""
    m = month_start(month_index).month
    table = {
        1: 0.88,
        2: 0.90,
        3: 0.97,
        4: 1.0,
        5: 1.02,
        6: 1.03,
        7: 1.05,
        8: 1.02,
        9: 0.98,
        10: 1.0,
        11: 1.06,
        12: 1.22,
    }
    return table[m]


def months_array() -> IntArray:
    return np.arange(N_MONTHS, dtype=np.int64)
