"""Loans, their delinquency path, and the monthly balance schedule."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pyarrow as pa

from bankgen.calendar import N_MONTHS, WINDOW_END, month_end, month_label, month_start
from bankgen.context import Context
from bankgen.frame import as_days, nullable_dates, to_table
from bankgen.profiles.base import LoanProduct, Profile
from bankgen.rng import bernoulli, lognormal_amount, weighted_choice

STATUS_BY_DPD = {0: "current", 30: "30dpd", 60: "60dpd", 90: "90dpd", 120: "120dpd"}
# every value loan_balances_monthly.status takes; the array holding them is as wide as the longest
MONTH_STATUSES = (*STATUS_BY_DPD.values(), "charged_off", "paid_off", "matured_balloon")
MONTH_STATUS_DTYPE = f"U{max(len(s) for s in MONTH_STATUSES)}"
AMORTISING_OVER_25Y = {"cre_non_owner_occupied", "cre_owner_occupied", "multifamily"}  # balloon at term
NHG_LIMIT = {
    2005: 250_000,
    2009: 265_000,
    2013: 290_000,
    2015: 245_000,
    2018: 265_000,
    2019: 290_000,
    2020: 310_000,
    2021: 325_000,
    2022: 355_000,
    2023: 405_000,
    2024: 435_000,
    2025: 450_000,
    2026: 470_000,
}
HOUSE_PRICE_INDEX = {
    "NL": {
        1995: 42,
        2000: 75,
        2005: 100,
        2008: 118,
        2010: 112,
        2013: 95,
        2016: 108,
        2019: 140,
        2021: 170,
        2022: 190,
        2023: 185,
        2024: 200,
        2025: 215,
        2026: 224,
    },
    "US": {
        1995: 55,
        2000: 75,
        2005: 100,
        2007: 110,
        2010: 90,
        2012: 85,
        2016: 110,
        2019: 130,
        2021: 165,
        2022: 185,
        2023: 188,
        2024: 195,
        2025: 200,
        2026: 205,
    },
}
LABEL_VALUE_FACTOR = {"A": 1.06, "B": 1.02, "C": 1.0, "D": 0.97, "E": 0.94, "F": 0.92, "G": 0.90, "": 1.0}


def _step(table: Mapping[int, float], year: np.ndarray) -> np.ndarray:
    keys = np.array(sorted(table))
    vals = np.array([table[k] for k in keys], dtype=np.float64)
    pos = np.clip(np.searchsorted(keys, year, side="right") - 1, 0, len(keys) - 1)
    return vals[pos]


def historic_policy_rate(profile: Profile, year: np.ndarray) -> np.ndarray:
    """A coarse policy-rate history so pre-window originations carry a plausible coupon."""
    if profile.country == "NL":
        table = {
            2005: 2.0,
            2006: 2.8,
            2007: 3.8,
            2008: 3.5,
            2009: 1.0,
            2010: 1.0,
            2011: 1.2,
            2012: 0.8,
            2013: 0.5,
            2014: 0.2,
            2015: 0.05,
            2016: 0.0,
            2017: 0.0,
            2018: 0.0,
            2019: 0.0,
            2020: 0.0,
            2021: 0.0,
            2022: 1.0,
            2023: 3.5,
            2024: 3.5,
            2025: 2.2,
            2026: 2.0,
        }
    else:
        table = {
            2005: 3.2,
            2006: 5.0,
            2007: 5.0,
            2008: 2.0,
            2009: 0.2,
            2010: 0.2,
            2011: 0.2,
            2012: 0.2,
            2013: 0.2,
            2014: 0.2,
            2015: 0.3,
            2016: 0.5,
            2017: 1.1,
            2018: 1.9,
            2019: 2.2,
            2020: 0.4,
            2021: 0.1,
            2022: 2.5,
            2023: 5.1,
            2024: 5.0,
            2025: 4.1,
            2026: 3.6,
        }
    return np.array([table[min(max(int(y), 2005), 2026)] for y in year])


def _origination_dates(rng: np.random.Generator, lp: LoanProduct, k: int) -> np.ndarray:
    y0, y1 = lp.origination_years
    years = np.arange(y0, y1 + 1)
    w = np.linspace(1.0, 3.0, len(years))  # recent years weigh more
    year = rng.choice(years, size=k, p=w / w.sum())
    day = rng.integers(0, 365, size=k)
    # a date after the window end is kept as it is: the caller drops a loan that would only start after the window
    return np.asarray(np.array([as_days(date(int(y), 1, 1)) for y in year]) + day.astype("timedelta64[D]"))


def _maturity(orig: np.ndarray, term_months: np.ndarray) -> np.ndarray:
    return np.asarray(orig + (term_months * 30.4375).astype(np.int64).astype("timedelta64[D]"))


def _month_index(d: np.ndarray) -> np.ndarray:
    """Months since the window start of each date; negative before it, N_MONTHS or more after it."""
    return np.asarray((d.astype("datetime64[M]") - np.datetime64(month_start(0), "M")).astype(np.int64))


def _annuity_payment(principal: np.ndarray, annual_rate: np.ndarray, n_months: np.ndarray) -> np.ndarray:
    r = annual_rate / 100.0 / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = np.where(r > 0, principal * r / (1 - (1 + r) ** (-n_months)), principal / n_months)
    return np.asarray(pay)


def _balance_after(
    principal: np.ndarray,
    annual_rate: np.ndarray,
    n_months: np.ndarray,
    k: np.ndarray,
    interest_only: np.ndarray,
) -> np.ndarray:
    """Outstanding balance after k payments of an annuity (or the principal for interest-only)."""
    r = annual_rate / 100.0 / 12.0
    pay = _annuity_payment(principal, annual_rate, n_months)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        growth = (1 + r) ** k
        bal = np.where(r > 0, principal * growth - pay * (growth - 1) / r, principal - pay * k)
    bal = np.where(interest_only, principal, bal)
    return np.clip(np.where(k >= n_months, 0.0, bal), 0.0, None)


def build(ctx: Context) -> None:
    p = ctx.profile
    rng = ctx.rng("loans")
    cust = ctx.table("customers").to_pydict()
    cid = np.array(cust["customer_id"])
    seg = np.array(cust["segment"])
    since = np.array(cust["customer_since"], dtype="datetime64[D]")
    origin = np.array(cust["origin"])
    income = np.array(cust["annual_income"], dtype=np.float64)
    score = np.array(cust["credit_score"], dtype=np.float64)
    partner = np.array(cust["partner_id"])
    region = np.array(cust["region_id"])
    birth = np.array(cust["birth_date"], dtype="datetime64[D]")
    on_book = np.array(cust["on_book_since"], dtype="datetime64[D]")
    aow = np.array(cust["aow_date"], dtype="datetime64[D]")
    n = len(cid)
    partners = ctx.table("partners").to_pydict()
    pid = np.array(partners["partner_id"])
    p_origin = np.array(partners["origin"])
    p_region = np.array(partners["region_id"])
    rates = p.rate_paths

    cols: dict[str, list[Any]] = {
        k: []
        for k in (
            "customer_id",
            "product",
            "orig",
            "term",
            "amount",
            "rate",
            "rate_type",
            "index_name",
            "margin",
            "io",
            "ltv",
            "collateral",
            "energy_label",
            "nhg",
            "sold",
            "sale_margin",
            "partner",
            "origin",
            "fixed_end",
            "fixed_end_mask",
            "amort",
            "underwritten_by",
            "first_dq",
            "co_month",
            "cure_month",
        )
    }

    for lp in p.loan_products:
        prob = np.array([lp.penetration.get(s, 0.0) for s in seg])
        has = bernoulli(rng, prob, n)
        idx = np.flatnonzero(has)
        k = len(idx)
        if k == 0:
            continue
        orig = _origination_dates(rng, lp, k)
        io = bernoulli(rng, lp.interest_only_share, k)
        term_y = rng.integers(lp.term_years[0], lp.term_years[1] + 1, size=k)
        if lp.product == "mortgage":
            # the Dutch interest-only wave: written 1995-2010 on 30-year terms, so it matures 2025-2040
            io_year = rng.integers(1995, 2011, size=k)
            io_orig = np.array([as_days(date(int(y), 1, 1)) for y in io_year]) + rng.integers(0, 365, size=k).astype(
                "timedelta64[D]"
            )
            orig = np.where(io, io_orig, orig)
            term_y = np.where(io, 30, term_y)
        else:
            orig = np.maximum(orig, since[idx] + np.timedelta64(7, "D"))
        orig = np.maximum(orig, birth[idx] + np.timedelta64(int(23 * 365.25), "D"))
        # a loan that would only start after the window is not a loan yet
        keep = orig <= as_days(WINDOW_END) - np.timedelta64(30, "D")
        idx, orig, io, term_y = idx[keep], orig[keep], io[keep], term_y[keep]
        k = len(idx)
        if k == 0:
            continue
        term = term_y * 12
        # maturity cluster: a share of a collateral type matures inside a window range
        collateral = (
            weighted_choice(rng, list(lp.collateral_mix), list(lp.collateral_mix.values()), k)
            if lp.collateral_mix
            else np.full(
                k,
                "residential"
                if "mortgage" in lp.product or lp.product == "heloc"
                else "vehicle"
                if lp.product == "indirect_auto"
                else "unsecured",
            )
        )
        if lp.maturity_cluster:
            m_from, m_to, share, coll = lp.maturity_cluster
            target = (collateral == coll) & bernoulli(rng, share, k)
            mat_month = rng.integers(m_from, m_to + 1, size=k)
            mat = np.array([as_days(month_start(int(m))) for m in mat_month]) + rng.integers(0, 28, size=k).astype(
                "timedelta64[D]"
            )
            orig = np.where(target, mat - (term * 30.4).astype(np.int64).astype("timedelta64[D]"), orig)
        year = orig.astype("datetime64[Y]").astype(int) + 1970
        base = historic_policy_rate(p, year)
        income_factor = (income[idx] / np.array([p.income_median[s] for s in seg[idx]])) ** 0.6
        amount = np.round(lognormal_amount(rng, lp.amount_median * income_factor, lp.amount_sigma, k), -2)
        # a loan written years ago was sized against that year's prices, not today's
        hpi_then = _step(HOUSE_PRICE_INDEX[p.country], year) / HOUSE_PRICE_INDEX[p.country][2026]
        if lp.product in ("mortgage", "residential_mortgage", "heloc") or lp.collateral_mix:
            amount = np.round(amount * hpi_then, -2)
        variable = bernoulli(rng, lp.variable_share, k)
        margin = np.round(rng.normal(lp.rate_spread, 0.35, size=k), 2)
        rate = np.round(base + margin, 2)
        amort = np.where(
            np.isin(lp.product, list(AMORTISING_OVER_25Y)) | (lp.product in AMORTISING_OVER_25Y),
            np.maximum(term, 300),
            term,
        )
        if lp.product == "construction":
            io = np.ones(k, dtype=bool)  # interest-only until completion, principal at maturity
        underwritten_by = np.where(
            (orig < on_book[idx]) & np.char.startswith(origin[idx].astype(str), "acquired:"),
            np.char.replace(origin[idx].astype(str), "acquired:", ""),
            p.bank_name,
        )
        ltv = np.clip(rng.beta(5, 2.2, size=k), 0.2, 1.05)
        energy = (
            weighted_choice(rng, list(p.energy_labels), list(p.energy_labels.values()), k)
            if p.energy_labels and lp.product == "mortgage"
            else np.full(k, "")
        )
        nhg = (
            (amount <= _step(NHG_LIMIT, year)) & bernoulli(rng, 0.55, k)
            if lp.product == "mortgage"
            else np.zeros(k, dtype=bool)
        )
        sold = bernoulli(rng, lp.sold_to_secondary_share, k)
        # gain-on-sale margin compresses through the window
        m_idx = np.clip(((orig - as_days(month_start(0))).astype(np.int64) // 30), 0, N_MONTHS - 1)
        sale_margin = np.where(sold, np.round(rng.normal(319 - 1.0 * m_idx, 18, size=k), 0), 0.0)
        fixed_end = orig + (rng.choice([10, 20], size=k, p=[0.6, 0.4]) * 365).astype("timedelta64[D]")
        fixed_mask = (~variable) & (lp.product == "mortgage")
        # partner: the customer's; loans originated in the window in expansion markets go to hired bankers
        loan_partner = partner[idx].copy()
        if p.hired_partners:
            hired = pid[p_origin == "hired_2026"]
            hired_region = p_region[p_origin == "hired_2026"]
            in_window = orig >= as_days(month_start(p.hired_partners[0]))
            for rid in np.unique(hired_region):
                m = in_window & (region[idx] == rid) & bernoulli(rng, 0.65, k)
                if m.any():
                    loan_partner[m] = rng.choice(hired[hired_region == rid], size=int(m.sum()))

        # delinquency: monthly hazard x credit factor x cohort multiplier, first hit inside the window
        credit_factor = np.clip(1.0 + (700 - score[idx]) / 120.0, 0.3, 4.0)
        hazard = np.tile(lp.delinquency_hazard * credit_factor[:, None], (1, N_MONTHS))
        for ch in p.cohort_hazards:
            if lp.product in ch.products:
                rows = (origin[idx] == ch.origin) & (underwritten_by != p.bank_name)
                hazard[np.ix_(rows, np.arange(ch.month_from, N_MONTHS))] *= ch.multiplier
        active_from = np.clip(
            ((np.maximum(orig, on_book[idx]) - as_days(month_start(0))).astype(np.int64) // 30), 0, N_MONTHS
        )
        hits = rng.random((k, N_MONTHS)) < hazard
        hits[np.arange(N_MONTHS)[None, :] < active_from[:, None]] = False
        # a loan that has matured can no longer fall past due
        hits[np.arange(N_MONTHS)[None, :] > _month_index(_maturity(orig, term))[:, None]] = False
        any_hit = hits.any(axis=1)
        first_dq = np.where(any_hit, hits.argmax(axis=1), -1)
        charge = bernoulli(rng, lp.charge_off_share, k) & any_hit
        co_month = np.where(charge, first_dq + 4, -1)
        cure_month = np.where(any_hit & ~charge, first_dq + rng.integers(1, 4, size=k), -1)

        cols["customer_id"] += list(cid[idx])
        cols["product"] += [lp.product] * k
        cols["orig"] += list(orig)
        cols["term"] += list(term)
        cols["amount"] += list(amount)
        cols["rate"] += list(rate)
        cols["rate_type"] += list(np.where(variable, "variable", "fixed"))
        cols["index_name"] += list(np.where(variable, lp.index_name or "", ""))
        cols["margin"] += list(margin)
        cols["io"] += list(io)
        cols["ltv"] += list(np.round(ltv, 3))
        cols["collateral"] += list(collateral)
        cols["energy_label"] += list(energy)
        cols["nhg"] += list(nhg)
        cols["sold"] += list(sold)
        cols["sale_margin"] += list(sale_margin)
        cols["partner"] += list(loan_partner)
        cols["origin"] += list(origin[idx])
        cols["fixed_end"] += list(fixed_end)
        cols["fixed_end_mask"] += list(fixed_mask)
        cols["amort"] += list(amort)
        cols["underwritten_by"] += list(underwritten_by)
        cols["first_dq"] += list(first_dq)
        cols["co_month"] += list(co_month)
        cols["cure_month"] += list(cure_month)

    m = len(cols["customer_id"])
    arr = {k: np.array(v) for k, v in cols.items()}
    orig = arr["orig"].astype("datetime64[D]")
    term = arr["term"].astype(np.int64)
    maturity = _maturity(orig, term)
    loan_id = np.array([f"L-{i + 1:06d}" for i in range(m)])
    c_index = {c: i for i, c in enumerate(cid)}
    cust_row = np.array([c_index[c] for c in arr["customer_id"]])
    amort_all = arr["amort"].astype(np.int64)
    amount_all = arr["amount"].astype(np.float64)
    rate_all = arr["rate"].astype(np.float64)
    io_all = arr["io"].astype(bool)
    months_elapsed = np.clip(((as_days(WINDOW_END) - orig).astype(np.int64) // 30), 0, amort_all)
    balance_end = _balance_after(amount_all, rate_all, amort_all, months_elapsed, io_all)
    balloon = np.round(_balance_after(amount_all, rate_all, amort_all, term, io_all), 2)
    year_all = orig.astype("datetime64[Y]").astype(int) + 1970
    hpi = HOUSE_PRICE_INDEX[p.country]
    label_factor = np.array([LABEL_VALUE_FACTOR.get(str(e), 1.0) for e in arr["energy_label"]])
    value_now = np.round(
        amount_all / arr["ltv"].astype(np.float64) * (hpi[2026] / _step(hpi, year_all)) * label_factor, -2
    )
    current_ltv = np.round(balance_end / np.maximum(value_now, 1), 3)
    age_at_maturity = ((maturity - birth[cust_row]).astype(np.int64) / 365.25).astype(np.int64)
    beyond_aow = maturity > aow[cust_row]
    replacement = np.where(np.isin(seg[cust_row], ["business", "small_business", "commercial"]), 0.6, 0.7)
    income_at_maturity = np.where(
        beyond_aow,
        np.maximum(16_000.0 if p.country == "NL" else 22_000.0, replacement * income[cust_row]),
        income[cust_row],
    )
    # the as-of rule: the window end shows only what happened by then, in a month the loan was still on
    # the book, so a charge-off or cure drawn for a later month leaves the loan delinquent; loan_balances_monthly
    # applies the same months, so the two tables agree
    # a loan still past due when it matures stays on the book, unpaid, until it cures (and pays off) or is charged off
    resolved = np.where(arr["co_month"] >= 0, arr["co_month"], arr["cure_month"])
    overdue_at_maturity = (
        (arr["first_dq"] >= 0) & (arr["first_dq"] <= _month_index(maturity)) & (resolved > _month_index(maturity))
    )
    last_on_book = np.minimum(np.where(overdue_at_maturity, resolved, _month_index(maturity)), N_MONTHS - 1)
    co_seen = (arr["co_month"] >= 0) & (arr["co_month"] <= last_on_book)
    cure_seen = (arr["cure_month"] >= 0) & (arr["cure_month"] <= last_on_book)
    still_due = (arr["first_dq"] >= 0) & ~co_seen & ~cure_seen
    status_end = np.where(
        co_seen,
        "charged_off",
        np.where(still_due, "delinquent", np.where(maturity <= as_days(WINDOW_END), "paid_off", "current")),
    )
    outcome_end = np.where(co_seen, "charged_off", np.where(cure_seen, "cured", ""))
    loans = to_table(
        {
            "loan_id": loan_id,
            "customer_id": arr["customer_id"],
            "product": arr["product"],
            "origination_date": orig,
            "term_months": term,
            "maturity_date": maturity,
            "original_amount": arr["amount"].astype(np.float64),
            "interest_rate_pct": arr["rate"].astype(np.float64),
            "rate_type": arr["rate_type"],
            "index_name": arr["index_name"],
            "margin_pct": arr["margin"].astype(np.float64),
            "is_interest_only": arr["io"].astype(bool),
            "ltv": arr["ltv"].astype(np.float64),
            "current_property_value": value_now,
            "current_ltv": current_ltv,
            "amortization_months": amort_all,
            "balloon_amount": balloon,
            "underwritten_by": arr["underwritten_by"],
            "age_at_maturity": age_at_maturity,
            "matures_after_pension_age": beyond_aow,
            "income_at_maturity": np.round(income_at_maturity, -2),
            "collateral_type": arr["collateral"],
            "property_value": np.round(arr["amount"].astype(np.float64) / arr["ltv"].astype(np.float64), -2),
            "energy_label": arr["energy_label"],
            "has_nhg": arr["nhg"].astype(bool),
            "sold_to_secondary": arr["sold"].astype(bool),
            "sale_margin_bps": arr["sale_margin"].astype(np.float64),
            "partner_id": arr["partner"],
            "origin": arr["origin"],
            "currency": np.array([p.currency] * m),
            "status_at_window_end": status_end,
            "first_delinquent_month": np.where(
                arr["first_dq"] >= 0,
                [month_label(int(x)) if x >= 0 else "" for x in arr["first_dq"]],
                "",
            ),
            "delinquency_outcome": outcome_end,
        }
    )
    loans = loans.append_column(
        "fixed_rate_period_end",
        nullable_dates(arr["fixed_end"].astype("datetime64[D]"), arr["fixed_end_mask"].astype(bool)),
    )
    ctx.tables["loans"] = loans
    # the relationship started no later than the oldest loan
    earliest = np.full(n, as_days(WINDOW_END))
    np.minimum.at(earliest, cust_row, orig)
    new_since = np.minimum(since, earliest).astype("datetime64[D]")
    ct = ctx.tables["customers"]
    ci = ct.schema.get_field_index("customer_since")
    ctx.tables["customers"] = ct.set_column(ci, "customer_since", pa.array(new_since, type=pa.date32()))

    # ---- monthly schedule ----
    amount = arr["amount"].astype(np.float64)
    rate = arr["rate"].astype(np.float64)
    io = arr["io"].astype(bool)
    first_dq = arr["first_dq"].astype(np.int64)
    co_month = arr["co_month"].astype(np.int64)
    cure_month = arr["cure_month"].astype(np.int64)
    variable = arr["rate_type"] == "variable"
    index_name = arr["index_name"]
    margin = arr["margin"].astype(np.float64)
    fixed_end_all = arr["fixed_end"].astype("datetime64[D]")
    fixed_mask_all = arr["fixed_end_mask"].astype(bool)
    on_book_loan = on_book[cust_row]
    out: dict[str, list[Any]] = {
        k: []
        for k in (
            "loan_id",
            "month",
            "month_index",
            "balance",
            "rate",
            "scheduled",
            "paid",
            "interest",
            "dpd",
            "status",
        )
    }
    for mi in range(N_MONTHS):
        ms, me = as_days(month_start(mi)), as_days(month_end(mi))
        on_book_now = (maturity >= ms) | (overdue_at_maturity & (mi <= resolved))
        active = (orig <= me) & (on_book_loan <= me) & on_book_now & ((co_month < 0) | (co_month >= mi))
        if not active.any():
            continue
        a = np.flatnonzero(active)
        dq_active = (first_dq[a] >= 0) & (mi >= first_dq[a]) & ((cure_month[a] < 0) | (mi < cure_month[a]))
        k_paid = np.clip(((ms - orig[a]).astype(np.int64) // 30), 0, amort_all[a])
        # a loan past due at or after its maturity has not made its last payment: its balance is still owed
        k_paid = np.where(dq_active & (k_paid >= amort_all[a]), amort_all[a] - 1, k_paid)
        r = rate[a].copy()
        if "mortgage_10y" in rates:
            reset = fixed_mask_all[a] & (fixed_end_all[a] <= me)
            r[reset] = rates["mortgage_10y"][mi]
        for key in set(index_name[a]) - {""}:
            sel = (index_name[a] == key) & variable[a]
            idx_rate = rates[key.lower()][mi] if key.lower() in rates else rates[p.policy_rate_key][mi]
            r[sel] = np.round(idx_rate + margin[a][sel], 2)
        bal = _balance_after(amount[a], r, amort_all[a], k_paid, io[a])
        sched = np.where(io[a], amount[a] * r / 100 / 12, _annuity_payment(amount[a], r, amort_all[a]))
        interest = bal * r / 100 / 12
        dpd = np.where(dq_active, np.minimum((mi - first_dq[a] + 1) * 30, 120), 0)
        paid = np.where(dq_active, 0.0, sched)
        status = np.array([STATUS_BY_DPD[int(d)] for d in dpd], dtype=MONTH_STATUS_DTYPE)
        status[(co_month[a] == mi)] = "charged_off"
        status[(maturity[a] <= me) & ~dq_active] = np.where(
            balloon[a][(maturity[a] <= me) & ~dq_active] > 0, "matured_balloon", "paid_off"
        )
        out["loan_id"] += list(loan_id[a])
        out["month"] += [month_label(mi)] * len(a)
        out["month_index"] += [mi] * len(a)
        out["balance"] += list(np.round(bal, 2))
        out["rate"] += list(r)
        out["scheduled"] += list(np.round(sched, 2))
        out["paid"] += list(np.round(paid, 2))
        out["interest"] += list(np.round(interest, 2))
        out["dpd"] += list(dpd)
        out["status"] += list(status)
    ctx.tables["loan_balances_monthly"] = to_table(
        {
            "loan_id": np.array(out["loan_id"]),
            "month": np.array(out["month"]),
            "month_index": np.array(out["month_index"], dtype=np.int64),
            "outstanding_balance": np.array(out["balance"], dtype=np.float64),
            "interest_rate_pct": np.array(out["rate"], dtype=np.float64),
            "scheduled_payment": np.array(out["scheduled"], dtype=np.float64),
            "paid_amount": np.array(out["paid"], dtype=np.float64),
            "interest_income": np.array(out["interest"], dtype=np.float64),
            "days_past_due": np.array(out["dpd"], dtype=np.int64),
            "status": np.array(out["status"]),
        }
    )
