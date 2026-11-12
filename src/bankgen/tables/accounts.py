"""Deposit accounts: one row per product a customer holds; the ledger and balances come later."""

from __future__ import annotations

from typing import Any

import numpy as np

from bankgen.calendar import N_MONTHS, WINDOW_END, month_end, month_start
from bankgen.context import Context
from bankgen.frame import as_days, days_between, nullable_dates, to_table
from bankgen.rng import bernoulli, lognormal_amount

CD_TERMS = (6, 12, 24)


def _iban(rng: np.random.Generator, n: int, bank_code: str = "NNBA") -> np.ndarray:
    """Valid-checksum NL IBANs on a bank code; account numbers are random."""
    acct = rng.integers(10**9, 10**10 - 1, size=n)
    out = []
    for a in acct:
        bban = f"{bank_code}{a:010d}"
        digits = "".join(str(int(c, 36)) for c in f"{bban}NL00")
        check = 98 - int(digits) % 97
        out.append(f"NL{check:02d}{bban}")
    return np.array(out)


def _us_account(rng: np.random.Generator, n: int) -> np.ndarray:
    routing = "051400549"  # one routing number for the whole bank; synthetic
    acct = rng.integers(10**9, 10**10 - 1, size=n)
    return np.array([f"{routing}-{a:010d}" for a in acct])


def build(ctx: Context) -> None:
    p = ctx.profile
    rng = ctx.rng("accounts")
    cust = ctx.table("customers").to_pydict()
    cid = np.array(cust["customer_id"])
    seg = np.array(cust["segment"])
    since = np.array(cust["customer_since"], dtype="datetime64[D]")
    on_book = np.array(cust["on_book_since"], dtype="datetime64[D]")
    churned = np.array([d if d is not None else None for d in cust["churned_date"]])
    churn_mask = np.array([d is not None for d in churned])
    churn_date = np.array([d if d is not None else WINDOW_END for d in churned], dtype="datetime64[D]")
    origin = np.array(cust["origin"])
    income = np.array(cust["annual_income"], dtype=np.float64)
    positive_pay = np.array(cust["has_positive_pay"])
    n = len(cid)

    rows: dict[str, list[Any]] = {
        k: []
        for k in (
            "customer_id",
            "product",
            "opened",
            "closed",
            "closed_mask",
            "rate_key",
            "opening_balance",
            "origin",
            "term_months",
            "maturity",
            "maturity_mask",
            "positive_pay",
            "transactional",
        )
    }

    held_any = np.zeros(n, dtype=bool)
    # the relationship starts with a product: a customer's first deposit product (a transactional one
    # where the profile has one, else the first in the profile's order) opens the day the customer comes
    # on this bank's book; the launched products and the rest follow, in this order
    opening_order = sorted(
        p.deposit_products,
        key=lambda d: (d.launch_month is not None, not d.is_transactional, p.deposit_products.index(d)),
    )
    for dp in opening_order:
        prob = np.array([dp.penetration.get(s, 0.0) for s in seg])
        has = bernoulli(rng, prob, n)
        if dp.launch_month is not None:
            # adoption ramps after launch: front-loaded over the remaining months
            m = dp.launch_month
            remaining = N_MONTHS - m
            k = int(has.sum())
            # truncated exponential over the remaining months: front-loaded, no pile-up in the last month
            scale = remaining / 2.2
            u = rng.random(k)
            adopt_month = m + np.floor(-scale * np.log(1 - u * (1 - np.exp(-remaining / scale)))).astype(np.int64)
            adopt_month = np.minimum(adopt_month, m + remaining - 1)
            o_start = np.array([as_days(month_start(int(x))) for x in adopt_month])
            o_end = np.array([as_days(month_end(int(x))) for x in adopt_month])
            opened = days_between(rng, o_start, o_end)
            opened = np.maximum(opened, on_book[has])
            # a customer who had already left cannot open the product
            idx_has = np.flatnonzero(has)
            still_here = ~churn_mask[idx_has] | (opened < churn_date[idx_has])
            has[idx_has[~still_here]] = False
            opened = opened[still_here]
        else:
            # opened at some point since the relationship started, skewed to the start of it
            lag_days = (rng.exponential(400.0, size=int(has.sum()))).astype(np.int64)
            opened = since[has] + lag_days.astype("timedelta64[D]")
            idx_has = np.flatnonzero(has)
            first = ~held_any[idx_has]
            opened[first] = since[idx_has[first]]
            # acquired books appear on this bank's ledger from the conversion month
            opened = np.maximum(opened, on_book[has])
            # the book is observed at the window end: an opening dated after it has not happened yet, so the
            # account is not generated (no date is ever moved onto the window end); a customer who had
            # already left cannot open the product
            still_here = (opened <= as_days(WINDOW_END)) & (~churn_mask[idx_has] | (opened < churn_date[idx_has]))
            has[idx_has[~still_here]] = False
            opened = opened[still_here]
        idx = np.flatnonzero(has)
        k = len(idx)
        bal_median = np.array([dp.balance_median.get(s, 1000.0) for s in seg[idx]])
        income_factor = (income[idx] / np.array([p.income_median[s] for s in seg[idx]])) ** 0.5
        bal = lognormal_amount(rng, np.maximum(bal_median * income_factor, 50.0), dp.balance_sigma, k)
        closed = churn_date[idx]
        closed_mask = churn_mask[idx] & (closed > opened)
        term = np.zeros(k, dtype=np.int64)
        maturity = np.full(k, as_days(WINDOW_END))
        maturity_mask = np.zeros(k, dtype=bool)
        if dp.product == "cd":
            term = rng.choice(CD_TERMS, size=k, p=[0.3, 0.45, 0.25])
            # maturities spread across and beyond the window; a cluster matures mid-window (repricing story)
            maturity = opened + (term * 30.4).astype(np.int64).astype("timedelta64[D]")
            roll = maturity < as_days(month_start(0))
            while roll.any():
                maturity[roll] = maturity[roll] + (term[roll] * 30.4).astype(np.int64).astype("timedelta64[D]")
                roll = maturity < as_days(month_start(0))
            maturity_mask = np.ones(k, dtype=bool)
        rows["customer_id"] += list(cid[idx])
        rows["product"] += [dp.product] * k
        rows["opened"] += list(opened)
        rows["closed"] += list(closed)
        rows["closed_mask"] += list(closed_mask)
        rows["rate_key"] += [dp.rate_path or ""] * k
        rows["opening_balance"] += list(np.round(bal, 2))
        rows["origin"] += list(origin[idx])
        rows["term_months"] += list(term)
        rows["maturity"] += list(maturity)
        rows["maturity_mask"] += list(maturity_mask)
        rows["positive_pay"] += list(positive_pay[idx] & (dp.product == "business_checking"))
        rows["transactional"] += [dp.is_transactional] * k
        if dp.launch_month is None:
            held_any |= has

    # everyone holds at least one product opened with the relationship: the default for their profile,
    # also for a customer whose only product is one launched inside the window
    default = "checking" if p.partner_kind == "banker" else "savings"
    default_dp = next(d for d in p.deposit_products if d.product == default)
    idx = np.flatnonzero(~held_any)
    k = len(idx)
    if k:
        bal_median = np.array([default_dp.balance_median.get(s, 1000.0) or 1000.0 for s in seg[idx]])
        rows["customer_id"] += list(cid[idx])
        rows["product"] += [default] * k
        rows["opened"] += list(on_book[idx])
        rows["closed"] += list(churn_date[idx])
        rows["closed_mask"] += list(churn_mask[idx])
        rows["rate_key"] += [default_dp.rate_path or ""] * k
        rows["opening_balance"] += list(np.round(lognormal_amount(rng, bal_median, 0.8, k), 2))
        rows["origin"] += list(origin[idx])
        rows["term_months"] += [0] * k
        rows["maturity"] += [as_days(WINDOW_END)] * k
        rows["maturity_mask"] += [False] * k
        rows["positive_pay"] += [False] * k
        rows["transactional"] += [default_dp.is_transactional] * k

    m = len(rows["customer_id"])
    order = np.lexsort((np.array(rows["product"]), np.array(rows["customer_id"])))

    def take(key: str) -> np.ndarray:
        return np.array(rows[key])[order]

    number = _iban(rng, m) if p.country == "NL" else _us_account(rng, m)
    frame = {
        "account_id": np.array([f"A-{i + 1:06d}" for i in range(m)]),
        "customer_id": take("customer_id"),
        "product": take("product"),
        "account_number": number,
        "currency": np.array([p.currency] * m),
        "opened_date": take("opened").astype("datetime64[D]"),
        "is_transactional": take("transactional"),
        "is_interest_bearing": take("rate_key") != "",
        "rate_key": take("rate_key"),
        "opening_balance": take("opening_balance"),
        "term_months": take("term_months"),
        "origin": take("origin"),
        "has_positive_pay": take("positive_pay"),
    }
    t = to_table(frame)
    t = t.append_column("closed_date", nullable_dates(take("closed"), take("closed_mask")))
    t = t.append_column("maturity_date", nullable_dates(take("maturity"), take("maturity_mask")))
    ctx.tables["accounts"] = t
