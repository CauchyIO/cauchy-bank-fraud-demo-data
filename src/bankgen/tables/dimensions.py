"""Regions, partners (intermediaries or bankers), merchants, interest rates, story events."""

from __future__ import annotations

import numpy as np
from faker import Faker

from bankgen.calendar import N_MONTHS, month_label, month_start
from bankgen.context import Context
from bankgen.frame import as_days, days_between, to_table
from bankgen.rng import weighted_choice

N_MERCHANTS = 3_000

# A name that fits what the money was for, per locale and category, so a customer's page reads "Groceries,
# Versmarkt Zwolle" and never a pharmaceutical group for the rent. Synthetic: built from the locale's
# surnames, first names and places; a fixed string is a role ("own savings"), not a company.
NAME_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "nl_NL": {
        "groceries": ("Supermarkt {last}", "Versmarkt {city}", "Buurtsuper {last}"),
        "transport": ("Tankstation {city}", "Taxi {last}", "Parkeren {city}", "Fietsenmaker {last}"),
        "dining": ("Restaurant {last}", "Eetcafé {city}", "Brasserie {last}", "Pizzeria {city}"),
        "online_shopping": ("Webshop {last}", "{last} Online", "{city} Webwinkel"),
        "ideal_payment": ("Webwinkel {last}", "{last} Online", "Bestellen {city}"),
        "health": ("Apotheek {city}", "Fysiotherapie {last}", "Tandarts {last}", "Huisarts {city}"),
        "salary": ("{last} B.V.", "{last} en {last2} B.V.", "Gemeente {city}", "Zorggroep {city}"),
        "pension": ("Pensioenfonds {city}", "Pensioenuitvoerder {last}"),
        "business_income": ("{last} B.V.", "{last} Holding", "{city} Handel B.V."),
        "rent_or_mortgage": ("Woningstichting {city}", "Vastgoed {last} B.V.", "Verhuur {last}"),
        "utilities": ("Energie {city}", "Waterbedrijf {city}", "{last} Energie"),
        "insurance_premium": ("{last} Verzekeringen", "Verzekeraar {city}"),
        "subscriptions": ("Sportschool {city}", "{last} Media", "{last} Streaming"),
        "cash_withdrawal": ("Geldautomaat {city}",),
        "transfer_to_savings": ("own savings",),
        "business_expense": ("Groothandel {last}", "{city} Zakelijke Diensten B.V."),
    },
    "en_US": {
        "groceries": ("{last}'s Market", "{city} Grocery", "Fresh Foods of {city}"),
        "gas": ("{city} Fuel", "{last} Gas & Go"),
        "dining": ("{last}'s Diner", "{city} Grill", "Cafe {last}"),
        "retail": ("{last} Outfitters", "{city} Hardware", "{last} Home Goods"),
        "healthcare": ("{city} Family Medicine", "{last} Pharmacy", "{city} Dental"),
        "payroll": ("{last} Industries", "{city} Health System", "{last} & {last2} LLC"),
        "social_security": ("Social Security Administration",),
        "customer_receipts": ("{last} Construction", "{city} Supply Co.", "{last} & {last2} LLC"),
        "card_settlements": ("{last} Merchant Services",),
        "zelle_received": ("{first} {last}",),
        "zelle_p2p": ("{first} {last}",),
        "rent_or_mortgage": ("{city} Property Management", "{last} Realty"),
        "utilities": ("{city} Electric", "{city} Water Authority", "{last} Energy"),
        "insurance_premium": ("{last} Insurance",),
        "subscriptions": ("{last} Streaming", "{city} Fitness", "{last} Media"),
        "check_payment": ("{last} Contracting", "{first} {last}"),
        "atm_withdrawal": ("ATM {city}",),
        "vendor_payment": ("{last} Supply Co.", "{city} Distributors"),
        "payroll_out": ("{last} Payroll Services",),
        "wire_out": ("{last} Holdings", "{city} Title & Escrow"),
        "transfer_to_savings": ("own savings",),
    },
}


def place(fake: Faker, places: tuple[str, ...]) -> str:
    """A town for a name or a merchant's city: one of the profile's `places`, Faker's city when it lists none."""
    return str(fake.random_element(places)) if places else str(fake.city())


def category_names(
    fake: Faker, locale: str, category: str, n: int, places: tuple[str, ...] = (), start: int = 0
) -> list[str]:
    """`n` synthetic names for a counterparty of `category`, from the locale's patterns; a company name otherwise.

    Draws only from `fake`, never from a numpy stream, so the ledger's amounts and choices stay as they were."""
    patterns = NAME_PATTERNS.get(locale, {}).get(category)
    if not patterns:
        return [fake.company() for _ in range(n)]
    out = []
    for i in range(start, start + n):
        pattern = patterns[i % len(patterns)]
        out.append(
            pattern.format(
                last=fake.last_name() if "{last}" in pattern else "",
                last2=fake.last_name() if "{last2}" in pattern else "",
                first=fake.first_name() if "{first}" in pattern else "",
                city=place(fake, places) if "{city}" in pattern else "",
            )
        )
    return out


def build_regions(ctx: Context) -> None:
    p = ctx.profile
    ctx.tables["regions"] = to_table(
        {
            "region_id": np.array([r.region_id for r in p.regions]),
            "region_name": np.array([r.name for r in p.regions]),
            "subdivision": np.array([r.subdivision for r in p.regions]),
            "country": np.array([p.country] * len(p.regions)),
            "customer_weight": np.array([r.weight for r in p.regions]),
            "acquired_bank": np.array([r.acquired_bank or "" for r in p.regions]),
            "is_expansion_market": np.array([r.expansion for r in p.regions]),
        }
    )


def build_partners(ctx: Context) -> None:
    """Intermediaries or relationship bankers, as the profile's partner_kind says."""
    p = ctx.profile
    rng = ctx.rng("partners")
    fake = ctx.faker("partners")
    n_hired = p.hired_partners[2] if p.hired_partners else 0
    n_acquired = round(p.n_partners * sum(a.customer_share for a in p.acquisitions))
    n_legacy = p.n_partners - n_hired - n_acquired
    region_ids = p.region_ids()
    weights = [r.weight for r in p.regions]

    rows: list[tuple[str, str, str, np.datetime64, str]] = []
    # legacy
    for rid in weighted_choice(rng, region_ids, weights, n_legacy):
        since = as_days(month_start(0)) - np.timedelta64(int(rng.integers(200, 9000)), "D")
        rows.append((rid, "legacy", "", since, ""))
    # came with an acquisition
    for acq in p.acquisitions:
        k = round(p.n_partners * acq.customer_share)
        for rid in rng.choice(acq.regions, size=k):
            rows.append(
                (
                    str(rid),
                    f"acquired:{acq.bank}",
                    acq.bank,
                    as_days(month_start(acq.close_month)),
                    "",
                )
            )
    # hired in the expansion markets
    if p.hired_partners:
        m_from, m_to, _ = p.hired_partners
        exp = [r.region_id for r in p.regions if r.expansion]
        for rid in rng.choice(exp, size=n_hired):
            m = int(rng.integers(m_from, m_to + 1))
            since = days_between(
                rng,
                np.array([as_days(month_start(m))]),
                np.array([as_days(month_start(m + 1)) - np.timedelta64(1, "D")]),
            )[0]
            rows.append((str(rid), "hired_2026", "", since, ""))

    n = len(rows)
    if p.partner_kind == "intermediary":
        names = [
            f"{fake.last_name()} {rng.choice(['Hypotheken', 'Financieel Advies', 'Advies', 'Assurantiën'])}"
            for _ in range(n)
        ]
    else:
        names = [fake.name() for _ in range(n)]
    # satisfaction is skewed high with a tail; hired bankers start a little lower
    sat = np.clip(rng.normal(7.9, 1.1, size=n), 2, 10)
    ctx.tables["partners"] = to_table(
        {
            "partner_id": np.array([f"P-{i + 1:04d}" for i in range(n)]),
            "partner_name": np.array(names),
            "partner_kind": np.array([p.partner_kind] * n),
            "region_id": np.array([r[0] for r in rows]),
            "origin": np.array([r[1] for r in rows]),
            "since_date": np.array([r[3] for r in rows], dtype="datetime64[D]"),
            "satisfaction_score": np.round(sat, 1),
        }
    )


def build_merchants(ctx: Context) -> None:
    p = ctx.profile
    rng = ctx.rng("merchants")
    fake = ctx.faker("merchants")
    cats = [c for c in p.txn_categories if c.merchant_based]
    cat_weights = [sum(c.monthly_rate.values()) for c in cats]
    category = weighted_choice(rng, [c.category for c in cats], cat_weights, N_MERCHANTS)
    region = weighted_choice(rng, p.region_ids(), [r.weight for r in p.regions], N_MERCHANTS)
    # one name per merchant, the category's patterns in turn, so the groceries are not all "Supermarkt ..."
    names = [category_names(fake, p.faker_locale, str(c), 1, p.places, start=i)[0] for i, c in enumerate(category)]
    cities = [place(fake, p.places) for _ in range(N_MERCHANTS)]
    # popularity: Zipf-like so a few merchants take most of the volume
    popularity = 1.0 / np.arange(1, N_MERCHANTS + 1) ** 0.9
    rng.shuffle(popularity)
    online = np.isin(category, ["online_shopping", "ideal_payment", "subscriptions"])
    ctx.tables["merchants"] = to_table(
        {
            "merchant_id": np.array([f"M-{i + 1:05d}" for i in range(N_MERCHANTS)]),
            "merchant_name": np.array(names),
            "merchant_category": category,
            "city": np.array(cities),
            "region_id": region,
            "is_online": online,
            "popularity_weight": popularity,
        }
    )


def build_interest_rates(ctx: Context) -> None:
    p = ctx.profile
    keys, months, labels, rates = [], [], [], []
    for key, pathv in p.rate_paths.items():
        for m in range(N_MONTHS):
            keys.append(key)
            months.append(m)
            labels.append(month_label(m))
            rates.append(pathv[m])
    ctx.tables["interest_rates"] = to_table(
        {
            "rate_key": np.array(keys),
            "month_index": np.array(months, dtype=np.int64),
            "month": np.array(labels),
            "rate_pct": np.array(rates, dtype=np.float64),
            "is_policy_rate": np.array([k == p.policy_rate_key for k in keys]),
        }
    )


def build_story_events(ctx: Context) -> None:
    p = ctx.profile
    ctx.tables["story_events"] = to_table(
        {
            "event_id": np.array([e.event_id for e in p.events]),
            "event_name": np.array([e.name for e in p.events]),
            "month_from": np.array([month_label(e.month_from) for e in p.events]),
            "month_to": np.array([month_label(e.month_to) for e in p.events]),
            "description": np.array([e.description for e in p.events]),
            "entity": np.array([e.entity for e in p.events]),
        }
    )


def build(ctx: Context) -> None:
    build_regions(ctx)
    build_partners(ctx)
    build_merchants(ctx)
    build_interest_rates(ctx)
    build_story_events(ctx)
