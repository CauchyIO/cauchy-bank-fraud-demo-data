"""Customers: segment, origin (legacy, acquired, new in window), region, partner, income, score."""

from __future__ import annotations

import numpy as np

from bankgen.calendar import N_MONTHS, WINDOW_END, month_start
from bankgen.context import Context
from bankgen.frame import as_days, days_between, nullable_dates, to_table
from bankgen.rng import bernoulli, lognormal_amount, weighted_choice

NEW_IN_WINDOW_SHARE = 0.10  # organic growth: share of non-acquired customers who joined inside the window
EXPANSION_BOOST_AFTER_HIRES = 2.5  # new customers land in expansion markets more often once bankers are hired
ACQUIRED_CHURN_MULTIPLIER = 2.2  # attrition in the 12 months after a conversion


def build(ctx: Context) -> None:
    p = ctx.profile
    n = ctx.n_customers
    rng = ctx.rng("customers")
    fake = ctx.faker("customers")
    partners = ctx.table("partners").to_pydict()

    segment = weighted_choice(rng, list(p.segments), list(p.segments.values()), n)

    # origin: acquired books first, then legacy / new in window
    origin = np.full(n, "legacy", dtype="U40")
    region = weighted_choice(rng, p.region_ids(), [r.weight for r in p.regions], n)
    converted_month = np.full(n, -1, dtype=np.int64)
    u = rng.random(n)
    lo = 0.0
    for acq in p.acquisitions:
        hi = lo + acq.customer_share
        mask = (u >= lo) & (u < hi)
        origin[mask] = f"acquired:{acq.bank}"
        region[mask] = rng.choice(acq.regions, size=int(mask.sum()))
        converted_month[mask] = acq.close_month
        lo = hi
    acquired = converted_month >= 0

    # customer_since: legacy skewed to the last decade, acquired = at their old bank, new = in window
    years_ago = np.minimum(rng.exponential(7.0, size=n), 30.0)
    since = as_days(month_start(0)) - (years_ago * 365.25).astype(np.int64).astype("timedelta64[D]")
    new_in_window = bernoulli(rng, NEW_IN_WINDOW_SHARE, n) & ~acquired
    hires_from = p.hired_partners[0] if p.hired_partners else N_MONTHS
    join_month = rng.integers(0, N_MONTHS, size=n)
    expansion_ids = [r.region_id for r in p.regions if r.expansion]
    if expansion_ids:
        late = new_in_window & (join_month >= hires_from)
        boost = bernoulli(rng, 1 - 1 / EXPANSION_BOOST_AFTER_HIRES, n) & late
        region[boost] = rng.choice(expansion_ids, size=int(boost.sum()))
    jm_start = np.array([as_days(month_start(int(m))) for m in join_month])
    jm_end = np.array([as_days(month_start(int(m) + 1)) - np.timedelta64(1, "D") for m in join_month])
    since[new_in_window] = days_between(rng, jm_start, jm_end)[new_in_window]
    origin[new_in_window] = "new"
    customer_since = since

    # partner: intermediaries for ~70%; bankers for everyone, hired bankers take the late joiners
    pid = np.array(partners["partner_id"])
    p_region = np.array(partners["region_id"])
    p_origin = np.array(partners["origin"])
    partner = np.full(n, "", dtype="U10")
    has_partner = bernoulli(rng, 0.70 if p.partner_kind == "intermediary" else 1.0, n)
    for rid in p.region_ids():
        in_region = (region == rid) & has_partner
        if not in_region.any():
            continue
        candidates = pid[p_region == rid]
        if len(candidates) == 0:
            candidates = pid
        partner[in_region] = rng.choice(candidates, size=int(in_region.sum()))
    if p.hired_partners:
        hired = pid[p_origin == "hired_2026"]
        hired_region = p_region[p_origin == "hired_2026"]
        late = new_in_window & (join_month >= hires_from)
        for rid in np.unique(hired_region):
            m = late & (region == rid) & bernoulli(rng, 0.6, n)
            if m.any():
                partner[m] = rng.choice(hired[hired_region == rid], size=int(m.sum()))
    for acq in p.acquisitions:
        m = origin == f"acquired:{acq.bank}"
        acq_partners = pid[p_origin == f"acquired:{acq.bank}"]
        if len(acq_partners) and m.any():
            partner[m] = rng.choice(acq_partners, size=int(m.sum()))

    # demographics
    age_lo = np.array([p.segment_age[s][0] for s in segment])
    age_hi = np.array([p.segment_age[s][1] for s in segment])
    age = age_lo + (rng.beta(2.0, 2.5, size=n) * (age_hi - age_lo)).astype(np.int64)
    birth = as_days(WINDOW_END) - (age * 365.25 + rng.integers(0, 365, size=n)).astype(np.int64).astype(
        "timedelta64[D]"
    )
    gender = weighted_choice(rng, ["F", "M", "X"], [0.49, 0.50, 0.01], n)
    first = np.array(
        [
            fake.first_name_female() if g == "F" else fake.first_name_male() if g == "M" else fake.first_name()
            for g in gender
        ]
    )
    last = np.array([fake.last_name() for _ in range(n)])
    city = np.array([fake.city() for _ in range(n)])
    postal = np.array([fake.postcode() for _ in range(n)])
    email = np.array(
        [
            f"{f.lower()}.{s.lower()}{rng.integers(1, 99)}@example.{'nl' if p.country == 'NL' else 'com'}".replace(
                " ", ""
            )
            for f, s in zip(first, last, strict=True)
        ]
    )

    income_median = np.array([p.income_median[s] for s in segment])
    income = np.round(lognormal_amount(rng, income_median, 0.45, n), -2)
    score_mean = np.array([p.credit_score_mean[s] for s in segment])
    score = np.clip(rng.normal(score_mean, p.credit_score_sd), *p.credit_score_range).astype(np.int64)

    is_business = np.isin(segment, ["business", "small_business", "commercial"])
    positive_pay = is_business & bernoulli(rng, p.positive_pay_share, n)

    # churn inside the window: annual rate by segment; acquired cohorts churn harder after conversion
    churn_p = np.array([p.churn_annual.get(s, 0.05) for s in segment]) * (N_MONTHS / 12)
    churn_p = np.where(acquired, churn_p * ACQUIRED_CHURN_MULTIPLIER, churn_p)
    churned = bernoulli(rng, np.minimum(churn_p, 0.6), n)
    earliest = np.maximum(customer_since, as_days(month_start(0)))
    earliest = np.where(
        acquired,
        np.array([as_days(month_start(max(int(m), 0))) for m in converted_month]),
        earliest,
    )
    churn_date = days_between(rng, earliest + np.timedelta64(30, "D"), np.full(n, as_days(WINDOW_END)))
    churned &= churn_date <= as_days(WINDOW_END)

    ctx.tables["customers"] = to_table(
        {
            "customer_id": np.array([f"C-{i + 1:06d}" for i in range(n)]),
            "first_name": first,
            "last_name": last,
            "gender": gender,
            "birth_date": birth.astype("datetime64[D]"),
            "age": age,
            "segment": segment,
            "region_id": region,
            "city": city,
            "postal_code": postal,
            "email": email,
            "partner_id": partner,
            "origin": origin,
            "customer_since": customer_since.astype("datetime64[D]"),
            "on_book_since": np.where(
                acquired,
                np.array([as_days(month_start(max(int(m), 0))) for m in converted_month]),
                customer_since,
            ).astype("datetime64[D]"),
            "aow_date": (birth + np.timedelta64(int(67 * 365.25), "D")).astype("datetime64[D]"),
            "annual_income": income,
            "credit_score": score,
            "is_business": is_business,
            "has_positive_pay": positive_pay,
            p.national_id_label: np.array([f"{rng.integers(0, 9999):04d}" for _ in range(n)]),
        }
    )
    t = ctx.tables["customers"]
    ctx.tables["customers"] = t.append_column("churned_date", nullable_dates(churn_date, churned))
