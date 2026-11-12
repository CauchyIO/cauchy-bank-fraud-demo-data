"""Shared fixtures: a tiny generated dataset per profile, built once per session in tmp_path."""

from __future__ import annotations

from pathlib import Path

import pytest

from bankgen.context import Context
from bankgen.generate import generate
from bankgen.load import load

SMALL = 120


@pytest.fixture(scope="session")
def out_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("out")


@pytest.fixture(scope="session", params=["cauchy"])
def generated(request: pytest.FixtureRequest, out_root: Path) -> tuple[str, Context, Path]:
    key = str(request.param)
    ctx = generate(key, out_root, seed=7, n_customers=SMALL, log=False)
    db = load(key, out_root)
    return key, ctx, db
