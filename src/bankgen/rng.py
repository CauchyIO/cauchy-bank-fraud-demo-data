"""Seeded random helpers. Every table draws from one Generator seeded per (profile, table)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def stream_token(name: str) -> int:
    """The number a stream name contributes to a generator's seed."""
    return sum(ord(c) * (i + 1) for i, c in enumerate(name))


def make_rng(seed: int, *stream: str | int) -> np.random.Generator:
    """One generator per named stream so that adding a table never reshuffles another.

    A stream is a name, or directly the number a name would give (``stream_token``), which lets a profile
    keep its draws when it is renamed."""
    tokens = [seed] + [s if isinstance(s, int) else stream_token(s) for s in stream]
    return np.random.default_rng(tokens)


def weighted_choice(
    rng: np.random.Generator, labels: Sequence[str], weights: Sequence[float], size: int
) -> NDArray[np.str_]:
    """Skewed categorical draw. Weights are normalised; never pass a flat vector on purpose."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    idx = rng.choice(len(labels), size=size, p=w)
    return np.asarray(labels, dtype=np.str_)[idx]


def lognormal_amount(
    rng: np.random.Generator, median: FloatArray | float, sigma: FloatArray | float, size: int
) -> FloatArray:
    """Always-positive, long-tailed amounts around a median."""
    mu = np.log(np.asarray(median, dtype=np.float64))
    out: FloatArray = rng.lognormal(mean=mu, sigma=np.asarray(sigma, dtype=np.float64), size=size)
    return out


def round_money(x: FloatArray, step: float = 0.01) -> FloatArray:
    out: FloatArray = np.round(x / step) * step
    return np.round(out, 2)


def bernoulli(rng: np.random.Generator, p: FloatArray | float, size: int) -> BoolArray:
    out: BoolArray = rng.random(size) < np.asarray(p, dtype=np.float64)
    return out
