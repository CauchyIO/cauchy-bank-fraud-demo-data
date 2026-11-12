"""The pipeline: dimensions -> customers -> accounts -> loans -> insurance -> ledger. Then Parquet."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from bankgen import __version__
from bankgen.calendar import N_MONTHS, WINDOW_END, WINDOW_START
from bankgen.context import Context
from bankgen.profiles import get_profile
from bankgen.tables import accounts, customers, dimensions, insurance, loans, transactions

DEFAULT_SEED = 20261013
DEFAULT_CUSTOMERS = 10_000

# transactions is written month by month by its builder; everything else is one file per table
TABLE_ORDER = [
    "regions",
    "partners",
    "merchants",
    "interest_rates",
    "story_events",
    "customers",
    "accounts",
    "loans",
    "loan_balances_monthly",
    "insurance_policies",
    "account_balances_monthly",
    "fraud_cases",
]


def generate(
    profile_key: str,
    out_root: Path,
    *,
    seed: int = DEFAULT_SEED,
    n_customers: int = DEFAULT_CUSTOMERS,
    log: bool = True,
) -> Context:
    profile = get_profile(profile_key)
    out_dir = out_root / profile_key
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(profile=profile, seed=seed, n_customers=n_customers, out_dir=out_dir)
    t0 = time.perf_counter()
    for name, builder in (
        ("dimensions", dimensions),
        ("customers", customers),
        ("accounts", accounts),
        ("loans", loans),
        ("insurance", insurance),
        ("transactions", transactions),
    ):
        t1 = time.perf_counter()
        builder.build(ctx)
        if log:
            print(f"  {profile_key}: {name:<13} {time.perf_counter() - t1:6.1f}s")
    for name in TABLE_ORDER:
        pq.write_table(ctx.tables[name], out_dir / f"{name}.parquet", compression="zstd")
    manifest = {
        "profile": profile_key,
        "bank": profile.bank_name,
        "seed": seed,
        "n_customers": n_customers,
        "generator_version": __version__,
        "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "months": N_MONTHS,
        "tables": {n: ctx.tables[n].num_rows for n in TABLE_ORDER},
        "transactions_files": sorted(p.name for p in (out_dir / "transactions").glob("*.parquet")),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if log:
        print(f"  {profile_key}: total {time.perf_counter() - t0:6.1f}s -> {out_dir}")
    return ctx
