"""The profile model: everything that differs between the two banks, as data.

Shared generator code reads these fields; nothing bank-specific lives in the table builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Region:
    region_id: str
    name: str
    subdivision: str  # province or state
    weight: float  # share of customers
    acquired_bank: str | None = None  # customers here may originate from an acquired bank
    expansion: bool = False  # growth market: new bankers / intermediaries land here


@dataclass(frozen=True)
class Acquisition:
    bank: str
    close_month: int  # month index when the book was converted
    customer_share: float  # share of all customers that came with this bank
    regions: tuple[str, ...]  # region_ids its customers sit in


@dataclass(frozen=True)
class DepositProduct:
    product: str
    penetration: dict[str, float]  # segment -> probability a customer holds one
    rate_path: str | None  # key into Profile.rate_paths, None = non-interest-bearing
    balance_median: dict[str, float]  # segment -> opening balance median
    balance_sigma: float = 0.9
    launch_month: int | None = None  # product did not exist before this month (adoption ramps)
    is_transactional: bool = False  # carries the spending ledger
    business_only: bool = False


@dataclass(frozen=True)
class LoanProduct:
    product: str
    penetration: dict[str, float]  # segment -> probability
    amount_median: float
    amount_sigma: float
    term_years: tuple[int, int]  # min, max
    rate_spread: float  # over policy rate at origination, in points
    variable_share: float  # share floating on the index
    index_name: str | None
    collateral_mix: dict[str, float] | None = None  # collateral_type -> weight
    interest_only_share: float = 0.0
    sold_to_secondary_share: float = 0.0  # mortgage banking: sold with a gain-on-sale margin
    delinquency_hazard: float = 0.004  # monthly P(first delinquency), base
    charge_off_share: float = 0.35  # of delinquent loans that roll to charge-off
    origination_years: tuple[int, int] = (2008, 2026)
    maturity_cluster: tuple[int, int, float, str] | None = None  # (m_from, m_to, share, collateral)


@dataclass(frozen=True)
class TxnCategory:
    category: str
    rail: str
    direction: int  # +1 credit, -1 debit
    monthly_rate: dict[str, float]  # segment -> expected count per month
    amount_median: float
    amount_sigma: float
    merchant_based: bool = False
    payday: bool = False  # lands on salary days
    income_based: bool = False  # amount derived from the customer's annual income


@dataclass(frozen=True)
class FraudEvent:
    month_from: int
    month_to: int
    multiplier: float
    types: tuple[str, ...]
    target_product: str | None = None  # only accounts of this product
    target_opened_after: int | None = None  # only accounts opened at/after this month
    target_segments: tuple[str, ...] | None = None
    target_no_positive_pay: bool = False
    # draw this event's cases from a random stream of its own, so adding or tuning it leaves every other draw
    # of the ledger as it was; None draws from the ledger's stream, as the events written before it do
    stream: str | int | None = None


@dataclass(frozen=True)
class DepositOutflowEvent:
    """A rate cut or a competitor campaign that moves money out of a product."""

    month_from: int
    month_to: int
    product: str
    segments: tuple[str, ...]
    share_of_accounts: float  # share of matching accounts that move money
    fraction_moved: float  # share of balance moved out
    multi_product_only: bool = True


@dataclass(frozen=True)
class CohortHazard:
    """Extra delinquency for a cohort from a month onward (an acquired book turning)."""

    origin: str  # 'acquired:<bank>'
    products: tuple[str, ...]
    month_from: int
    multiplier: float


@dataclass(frozen=True)
class StoryEvent:
    event_id: str
    name: str
    month_from: int
    month_to: int
    description: str
    entity: str


@dataclass(frozen=True)
class Profile:
    key: str
    rng_key: str | int  # seeds every random stream of the profile (see rng.make_rng)
    bank_name: str
    country: str
    currency: str
    faker_locale: str
    partner_kind: str  # 'intermediary' | 'banker'
    n_partners: int
    hired_partners: tuple[int, int, int] | None  # (month_from, month_to, count) new hires in expansion
    regions: tuple[Region, ...]
    acquisitions: tuple[Acquisition, ...]
    segments: dict[str, float]
    segment_age: dict[str, tuple[int, int]]
    income_median: dict[str, float]
    credit_score_range: tuple[int, int]
    credit_score_mean: dict[str, float]
    credit_score_sd: float
    deposit_products: tuple[DepositProduct, ...]
    loan_products: tuple[LoanProduct, ...]
    txn_categories: tuple[TxnCategory, ...]
    salary_days: tuple[int, ...]
    fraud_types: dict[str, float]
    fraud_base_rate: float  # P(fraud case) per transactional account-month
    fraud_loss_median: float
    fraud_events: tuple[FraudEvent, ...]
    insurance_products: dict[str, float]
    insurance_penetration: dict[str, float]
    insurance_premium_median: dict[str, float]
    rate_paths: dict[str, tuple[float, ...]]  # 24 monthly values, percent
    policy_rate_key: str
    deposit_outflow_events: tuple[DepositOutflowEvent, ...] = ()
    cohort_hazards: tuple[CohortHazard, ...] = ()
    energy_labels: dict[str, float] | None = None  # NL mortgage collateral energy labels
    positive_pay_share: float = 0.0  # business customers with check/ACH fraud controls
    churn_annual: dict[str, float] = field(default_factory=dict)
    events: tuple[StoryEvent, ...] = ()
    national_id_label: str = "national_id_last4"
    # towns for counterparty and merchant names where the Faker locale's own are invented; empty uses Faker's cities
    places: tuple[str, ...] = ()

    def segment_names(self) -> list[str]:
        return list(self.segments)

    def region_ids(self) -> list[str]:
        return [r.region_id for r in self.regions]


def path(values: list[tuple[int, float]], n: int = 24) -> tuple[float, ...]:
    """Step path from (month_index, value) breakpoints; holds the last value."""
    out: list[float] = []
    cur = values[0][1]
    pts = dict(values)
    for m in range(n):
        if m in pts:
            cur = pts[m]
        out.append(cur)
    return tuple(out)
