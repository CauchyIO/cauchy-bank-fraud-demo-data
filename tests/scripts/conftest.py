"""Puts ``scripts/`` on ``sys.path`` so the tests import ``publish_data`` and ``build_overlay``: the scripts are
files run with ``uv run``, not a package."""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
