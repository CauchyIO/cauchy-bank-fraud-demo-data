"""Insurance policies: the bank's bancassurance. The cross-sell table."""

from __future__ import annotations

import numpy as np

from bankgen.calendar import WINDOW_END, month_start
from bankgen.context import Context
from bankgen.frame import as_days, days_between, nullable_dates, to_table
from bankgen.rng import bernoulli, lognormal_amount, weighted_choice


def build(ctx: Context) -> None:
    p = ctx.profile
    rng = ctx.rng("insurance")
    cust = ctx.table("customers").to_pydict()
    cid = np.array(cust["customer_id"])
    seg = np.array(cust["segment"])
    since = np.array(cust["customer_since"], dtype="datetime64[D]")
    income = np.array(cust["annual_income"], dtype=np.float64)
    partner = np.array(cust["partner_id"])
    is_business = np.array(cust["is_business"])
    n = len(cid)
    has = bernoulli(rng, np.array([p.insurance_penetration.get(s, 0.0) for s in seg]), n)
    idx = np.flatnonzero(has)
    n_pol = rng.choice([1, 2, 3], size=len(idx), p=[0.62, 0.28, 0.10])
    rep = np.repeat(idx, n_pol)
    k = len(rep)
    prods = list(p.insurance_products)
    w = np.array(list(p.insurance_products.values()))
    product = weighted_choice(rng, prods, list(w), k)
    # business customers lean to commercial lines, consumers never hold them
    commercial = np.isin(product, ["commercial_property", "general_liability", "employee_benefits"])
    swap = commercial & ~is_business[rep]
    product[swap] = rng.choice(
        [x for x in prods if x not in ("commercial_property", "general_liability", "employee_benefits")],
        size=int(swap.sum()),
    )
    start = days_between(rng, since[rep], np.full(k, as_days(WINDOW_END)))
    # a quarter of policies are recent cross-sells inside the window
    recent = bernoulli(rng, 0.25, k)
    start[recent] = days_between(
        rng,
        np.maximum(since[rep][recent], as_days(month_start(0))),
        np.full(int(recent.sum()), as_days(WINDOW_END)),
    )
    median = np.array([p.insurance_premium_median[x] for x in product])
    inc_f = (income[rep] / np.array([p.income_median[s] for s in seg[rep]])) ** 0.4
    premium = np.round(lognormal_amount(rng, median * inc_f, 0.4, k), 0)
    lapsed = bernoulli(rng, 0.08, k)
    end = days_between(rng, start + np.timedelta64(90, "D"), np.full(k, as_days(WINDOW_END)))
    lapsed &= end <= as_days(WINDOW_END)
    commission_rate = np.round(np.clip(rng.normal(0.15, 0.03, size=k), 0.06, 0.25), 3)
    t = to_table(
        {
            "policy_id": np.array([f"I-{i + 1:06d}" for i in range(k)]),
            "customer_id": cid[rep],
            "product": product,
            "start_date": start.astype("datetime64[D]"),
            "annual_premium": premium,
            "commission_rate": commission_rate,
            "annual_commission": np.round(premium * commission_rate, 2),
            "channel": np.where(partner[rep] != "", p.partner_kind, "direct"),
            "partner_id": partner[rep],
            "status": np.where(lapsed, "lapsed", "active"),
            "currency": np.array([p.currency] * k),
        }
    )
    ctx.tables["insurance_policies"] = t.append_column("end_date", nullable_dates(end, lapsed))
