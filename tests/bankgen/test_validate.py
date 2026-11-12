from __future__ import annotations

from pathlib import Path

import pytest

from bankgen.context import Context
from bankgen.validate import validate

pytestmark = pytest.mark.unit


def test_structure_checks_pass_on_small_data(generated: tuple[str, Context, Path]) -> None:
    """Story checks need volume; structure checks must hold at any size."""
    _key, _ctx, db = generated
    import duckdb

    from bankgen.validate import COMMON_STRUCTURE, _fk_checks, run_checks

    con = duckdb.connect(str(db), read_only=True)
    try:
        results = run_checks(con, [*_fk_checks(), *COMMON_STRUCTURE])
    finally:
        con.close()
    failed = [
        (c.name, d)
        for c, ok, d in results
        if not ok and not any(w in c.name for w in ("peaks", "long-tailed", "skewed"))
    ]
    assert failed == []


def test_validate_returns_bool(generated: tuple[str, Context, Path], out_root: Path) -> None:
    key, _, _ = generated
    assert isinstance(validate(key, out_root, quiet=True), bool)
