from __future__ import annotations

from pathlib import Path

import pytest

from bankgen.context import Context

pytestmark = pytest.mark.unit


def test_every_customer_has_an_account_and_launch_is_respected(
    generated: tuple[str, Context, Path],
) -> None:
    key, ctx, _ = generated
    a = ctx.table("accounts").to_pydict()
    assert set(ctx.table("customers").to_pydict()["customer_id"]) == set(a["customer_id"])
    assert len(set(a["account_number"])) == len(a["account_number"])
    for dp in ctx.profile.deposit_products:
        if dp.launch_month is not None:
            opened = [d for d, p in zip(a["opened_date"], a["product"], strict=True) if p == dp.product]
            assert opened and min(opened).isoformat() >= "2025-05-01"
    if key == "cauchy":
        assert all(n.startswith("NL") and len(n) == 18 for n in a["account_number"])
