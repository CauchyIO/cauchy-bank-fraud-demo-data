"""Column-oriented frames: dict of numpy arrays, converted to Arrow once at the end."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pyarrow as pa

Frame = dict[str, Any]


def to_table(frame: Frame) -> pa.Table:
    cols: dict[str, pa.Array[Any]] = {}
    for name, values in frame.items():
        arr = np.asarray(values)
        if arr.dtype.kind == "U":
            cols[name] = pa.array(arr.tolist(), type=pa.string())
        elif arr.dtype.kind == "O":
            cols[name] = pa.array(arr.tolist())
        elif arr.dtype.kind == "M" and np.datetime_data(arr.dtype)[0] == "D":
            cols[name] = pa.array(arr.astype("datetime64[D]"), type=pa.date32())
        elif arr.dtype.kind == "M":
            cols[name] = pa.array(arr.astype("datetime64[us]"), type=pa.timestamp("us"))
        else:
            cols[name] = pa.array(arr)
    return pa.table(cols)


def as_days(d: date) -> np.datetime64:
    return np.datetime64(d, "D")


def days_between(rng: np.random.Generator, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Uniform dates between per-row start and end (inclusive); end < start collapses to start."""
    span = np.maximum((end - start).astype("timedelta64[D]").astype(np.int64), 0)
    offset = (rng.random(len(span)) * (span + 1)).astype(np.int64)
    out: np.ndarray = start + offset.astype("timedelta64[D]")
    return out


def nullable_dates(values: np.ndarray, mask: np.ndarray) -> pa.Array[Any]:
    """date32 array with nulls where mask is False."""
    return pa.array(values.astype("datetime64[D]"), type=pa.date32(), mask=~mask)
