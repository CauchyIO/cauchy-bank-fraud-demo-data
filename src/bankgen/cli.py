"""bankgen: generate | load | validate | all."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bankgen.generate import DEFAULT_CUSTOMERS, DEFAULT_SEED, generate
from bankgen.load import load
from bankgen.profiles import PROFILES

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "out"  # the repo root's data/out, ignored


def _profiles(arg: str) -> list[str]:
    return list(PROFILES) if arg == "all" else [arg]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bankgen", description=__doc__)
    ap.add_argument("command", choices=["generate", "load", "validate", "all"])
    ap.add_argument("--profile", default="all", choices=[*PROFILES, "all"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--customers", type=int, default=DEFAULT_CUSTOMERS)
    args = ap.parse_args(argv)
    failed = False
    for key in _profiles(args.profile):
        if args.command in ("generate", "all"):
            print(f"generate {key} (seed={args.seed}, customers={args.customers})")
            generate(key, args.out, seed=args.seed, n_customers=args.customers)
        if args.command in ("load", "all"):
            db = load(key, args.out)
            print(f"loaded {key} -> {db}")
        if args.command in ("validate", "all"):
            from bankgen.validate import validate

            ok = validate(key, args.out)
            failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
