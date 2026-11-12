from __future__ import annotations

import numpy as np
import pytest

from bankgen.rng import lognormal_amount, make_rng, weighted_choice

pytestmark = pytest.mark.unit


def test_streams_are_independent_and_reproducible() -> None:
    a1 = make_rng(1, "x").random(5)
    a2 = make_rng(1, "x").random(5)
    b = make_rng(1, "y").random(5)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)


def test_weighted_choice_is_skewed() -> None:
    rng = make_rng(3, "t")
    out = weighted_choice(rng, ["a", "b"], [0.9, 0.1], 2000)
    assert 0.85 < (out == "a").mean() < 0.95


def test_lognormal_is_positive_and_long_tailed() -> None:
    rng = make_rng(3, "l")
    x = lognormal_amount(rng, 100.0, 0.9, 20000)
    assert x.min() > 0
    assert np.quantile(x, 0.99) > 5 * np.median(x)
