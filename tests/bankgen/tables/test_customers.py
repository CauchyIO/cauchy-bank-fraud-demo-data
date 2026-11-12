from __future__ import annotations

from pathlib import Path

import pytest

from bankgen.context import Context

pytestmark = pytest.mark.unit


def test_customers_have_valid_keys_and_skew(generated: tuple[str, Context, Path]) -> None:
    _key, ctx, _db = generated
    c = ctx.table("customers").to_pydict()
    assert len(set(c["customer_id"])) == ctx.n_customers
    assert set(c["region_id"]) <= set(ctx.profile.region_ids())
    partners = set(ctx.table("partners").to_pydict()["partner_id"])
    assert set(c["partner_id"]) - {""} <= partners
    counts = {s: c["segment"].count(s) for s in set(c["segment"])}
    assert max(counts.values()) > 2 * min(counts.values())
    assert min(c["credit_score"]) >= 300 and max(c["credit_score"]) <= 850
    if ctx.profile.acquisitions:
        assert any(o.startswith("acquired:") for o in c["origin"])
