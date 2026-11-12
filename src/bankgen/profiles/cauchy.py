"""Cauchy Bank: a fictitious Dutch retail bank of mortgages, savings, bank annuities and, since May 2025,
payment accounts. Every customer, account and figure it generates is synthetic."""

from __future__ import annotations

from bankgen.profiles.base import (
    DepositOutflowEvent,
    DepositProduct,
    FraudEvent,
    LoanProduct,
    Profile,
    Region,
    StoryEvent,
    TxnCategory,
    path,
)

SEG = ("young", "mass", "affluent", "senior", "business")


def _by(young: float, mass: float, affluent: float, senior: float, business: float) -> dict[str, float]:
    return dict(zip(SEG, (young, mass, affluent, senior, business), strict=True))


CAUCHY = Profile(
    key="cauchy",
    # the number the generator's random streams have always been seeded with, so the rows stay the same
    rng_key=330,
    bank_name="Cauchy Bank",
    country="NL",
    currency="EUR",
    faker_locale="nl_NL",
    partner_kind="intermediary",
    n_partners=120,
    hired_partners=None,
    regions=(
        Region("zuid_holland", "Zuid-Holland", "Zuid-Holland", 0.21),
        Region("noord_holland", "Noord-Holland", "Noord-Holland", 0.17),
        Region("noord_brabant", "Noord-Brabant", "Noord-Brabant", 0.15),
        Region("gelderland", "Gelderland", "Gelderland", 0.12),
        Region("utrecht", "Utrecht", "Utrecht", 0.08),
        Region("overijssel", "Overijssel", "Overijssel", 0.07),
        Region("limburg", "Limburg", "Limburg", 0.06),
        Region("friesland", "Friesland", "Friesland", 0.04),
        Region("groningen", "Groningen", "Groningen", 0.03),
        Region("drenthe", "Drenthe", "Drenthe", 0.03),
        Region("flevoland", "Flevoland", "Flevoland", 0.025),
        Region("zeeland", "Zeeland", "Zeeland", 0.02),
    ),
    acquisitions=(),
    segments=_by(0.22, 0.40, 0.13, 0.20, 0.05),
    segment_age={
        "young": (21, 34),
        "mass": (35, 54),
        "affluent": (38, 64),
        "senior": (55, 84),
        "business": (28, 62),
    },
    income_median=_by(34_000, 52_000, 110_000, 38_000, 68_000),
    credit_score_range=(300, 850),
    credit_score_mean=_by(640, 690, 760, 720, 680),
    credit_score_sd=55.0,
    deposit_products=(
        DepositProduct(
            "savings",
            _by(0.45, 0.60, 0.80, 0.70, 0.50),
            "savings",
            _by(3_000, 12_000, 60_000, 35_000, 20_000),
        ),
        DepositProduct(
            "payment",
            _by(0.30, 0.22, 0.18, 0.10, 0.20),
            None,
            _by(900, 1_800, 4_500, 2_200, 3_500),
            balance_sigma=0.7,
            launch_month=7,
            is_transactional=True,
        ),
        DepositProduct(
            "bank_annuity",
            _by(0.02, 0.15, 0.25, 0.35, 0.20),
            "annuity",
            _by(5_000, 25_000, 80_000, 60_000, 40_000),
        ),
        DepositProduct(
            "investment",
            _by(0.08, 0.10, 0.35, 0.12, 0.15),
            None,
            _by(15_000, 30_000, 120_000, 50_000, 40_000),
        ),
    ),
    loan_products=(
        LoanProduct(
            "mortgage",
            _by(0.25, 0.62, 0.70, 0.55, 0.50),
            280_000,
            0.45,
            (20, 30),
            1.6,
            0.02,
            None,
            interest_only_share=0.30,
            delinquency_hazard=0.0012,
            charge_off_share=0.12,
            origination_years=(2005, 2026),
        ),
        LoanProduct(
            "consumer_loan",
            _by(0.10, 0.09, 0.04, 0.05, 0.12),
            12_000,
            0.6,
            (3, 7),
            5.5,
            0.0,
            None,
            delinquency_hazard=0.006,
            charge_off_share=0.4,
            origination_years=(2021, 2026),
        ),
    ),
    txn_categories=(
        TxnCategory(
            "salary",
            "sepa_transfer",
            +1,
            _by(1, 1, 1, 0.25, 0),
            2_800,
            0.15,
            payday=True,
            income_based=True,
        ),
        TxnCategory("pension", "sepa_transfer", +1, _by(0, 0, 0, 1, 0), 1_900, 0.3, payday=True),
        TxnCategory("business_income", "sepa_transfer", +1, _by(0, 0, 0, 0, 6), 1_500, 1.0),
        TxnCategory("rent_or_mortgage", "direct_debit", -1, _by(1, 1, 1, 1, 1), 1_100, 0.3),
        TxnCategory("groceries", "card_pos", -1, _by(10, 14, 12, 10, 10), 38, 0.6, merchant_based=True),
        TxnCategory("utilities", "direct_debit", -1, _by(2, 2, 2, 2, 2), 140, 0.4),
        TxnCategory("insurance_premium", "direct_debit", -1, _by(1.2, 1.6, 2, 1.5, 1.8), 95, 0.5),
        TxnCategory("transport", "card_pos", -1, _by(6, 6, 5, 3, 8), 28, 0.8, merchant_based=True),
        TxnCategory("dining", "card_pos", -1, _by(7, 5, 7, 2, 5), 32, 0.7, merchant_based=True),
        TxnCategory("online_shopping", "ideal", -1, _by(6, 4, 5, 1.5, 3), 55, 0.9, merchant_based=True),
        TxnCategory("ideal_payment", "ideal", -1, _by(4, 3, 3, 1, 2), 42, 0.9, merchant_based=True),
        TxnCategory("subscriptions", "direct_debit", -1, _by(3, 3, 4, 1.5, 3), 14, 0.5),
        TxnCategory("cash_withdrawal", "atm", -1, _by(1.5, 1.5, 1, 2, 1), 80, 0.5),
        TxnCategory("transfer_to_savings", "transfer", -1, _by(0.8, 1, 1.5, 0.7, 0.5), 250, 0.9),
        TxnCategory("health", "card_pos", -1, _by(0.5, 0.8, 0.8, 1.5, 0.5), 45, 0.7, merchant_based=True),
        TxnCategory("business_expense", "sepa_transfer", -1, _by(0, 0, 0, 0, 8), 420, 1.0),
    ),
    salary_days=(23, 24, 25),
    fraud_types={
        "bank_helpdesk_scam": 0.33,
        "phishing": 0.25,
        "ideal_invoice_scam": 0.15,
        "account_takeover": 0.10,
        "card_skimming": 0.10,
        "investment_scam": 0.07,
    },
    fraud_base_rate=0.0025,
    fraud_loss_median=1_800,
    fraud_events=(
        FraudEvent(
            19,
            20,
            4.0,
            ("phishing", "bank_helpdesk_scam", "account_takeover"),
            target_product="payment",
            target_opened_after=7,
        ),
        # the same wave's phishing and bank-helpdesk campaign: two more cases per base case on the same accounts;
        # drawn from its own stream, so the rest of the book stays as it was
        FraudEvent(
            19,
            20,
            3.0,
            ("phishing", "bank_helpdesk_scam"),
            target_product="payment",
            target_opened_after=7,
            stream=905,  # the stream this campaign has always drawn from, so its cases stay the same
        ),
    ),
    insurance_products={
        "home": 0.28,
        "car": 0.27,
        "liability": 0.20,
        "term_life": 0.10,
        "disability": 0.05,
        "pension_accrual": 0.10,
    },
    insurance_penetration=_by(0.20, 0.40, 0.45, 0.35, 0.40),
    insurance_premium_median={
        "home": 420,
        "car": 780,
        "liability": 65,
        "term_life": 310,
        "disability": 1_400,
        "pension_accrual": 2_400,
    },
    rate_paths={
        # the ECB deposit facility at each month's end; the 2026 hikes are the published ones, 2,25% from June 2026 and
        # 2,50% from 16 Sep 2026 (ECB key rates); product rates do not follow them
        "ecb_dfr": path([(0, 3.25), (2, 3.00), (4, 2.75), (6, 2.25), (8, 2.00), (20, 2.25), (23, 2.50)]),
        "savings": path([(0, 1.75), (5, 1.60), (13, 1.20)]),
        "annuity": path([(0, 2.60), (6, 2.30), (13, 2.10)]),
        "market_savings": path([(0, 2.10), (5, 1.95), (9, 1.80), (13, 1.75)]),
        "mortgage_10y": path([(0, 3.9), (6, 3.6), (12, 3.7), (18, 3.8)]),
    },
    policy_rate_key="ecb_dfr",
    deposit_outflow_events=(DepositOutflowEvent(13, 15, "savings", ("affluent", "mass", "senior"), 0.50, 0.55),),
    cohort_hazards=(),
    energy_labels={"A": 0.26, "B": 0.15, "C": 0.25, "D": 0.14, "E": 0.10, "F": 0.06, "G": 0.04},
    positive_pay_share=0.0,
    churn_annual=_by(0.07, 0.04, 0.03, 0.03, 0.06),
    events=(
        StoryEvent(
            "E1",
            "Payment accounts launched",
            7,
            23,
            "Cauchy Bank adds payment accounts in May 2025; a minority of existing customers adopt them over the "
            "following months. Adopters hold slightly fewer other products (2,17 against 2,27) but churn about half "
            "as often (5,1% against 9,2%).",
            "accounts",
        ),
        StoryEvent(
            "E2",
            "Savings rate cut lags the market",
            13,
            15,
            "Savings rate cut to 1,20% in November 2025 while competitors are at 1,75%; "
            "multi-product affluent and senior customers move part of their savings out.",
            "accounts",
        ),
        StoryEvent(
            "E3",
            "Interest-only mortgage cohort",
            0,
            23,
            "Roughly 30% of mortgages are interest-only; the 276 that mature after the borrower's pension age all "
            "mature by 2040, and the 250 of them still live in September 2026 are the affordability-scan cohort.",
            "loans",
        ),
        StoryEvent(
            "E4",
            "Phishing wave on payment accounts",
            19,
            20,
            "Bank-helpdesk and phishing scams aimed at payment-account holders take fraud to six times the usual "
            "month in May and June 2026; 30 of the 40 scam cases hit accounts open six months or more, not new ones.",
            "fraud_cases",
        ),
        StoryEvent(
            "E5",
            "Energy-label upgrade campaign",
            16,
            23,
            "Only 26% of the mortgage book has label A or better; D to G collateral with "
            "headroom is the green-upsell target list.",
            "loans",
        ),
        StoryEvent(
            "E6",
            "ECB raises its deposit rate",
            20,
            23,
            "The ECB deposit facility rises to 2,25% in June 2026 and 2,50% in September; Cauchy Bank's savings rate holds "
            "at 1,20%, 1,30 points below the policy rate by September.",
            "rates",
        ),
    ),
    national_id_label="bsn_last4",
)
