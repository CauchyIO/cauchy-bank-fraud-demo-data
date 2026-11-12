from __future__ import annotations

from pathlib import Path

import pytest

from bankgen.cli import main

pytestmark = pytest.mark.unit


def test_cli_all_runs_on_tiny_data(tmp_path: Path) -> None:
    rc = main(["all", "--profile", "cauchy", "--customers", "60", "--out", str(tmp_path), "--seed", "3"])
    assert rc in (0, 1)  # story checks may legitimately fail at 60 customers
    assert (tmp_path / "cauchy" / "manifest.json").exists()
    assert (tmp_path / "cauchy.duckdb").exists()
