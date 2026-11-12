"""The data dictionary: one description for the catalog, the schema, every table and view
and every column the generator writes, with the allowed values of the code columns, the grain of
each table, the as-of rule and the keys that join the tables.

The text names no bank. A bank's words come from its profile (`bankgen.profiles`) at render time:
`vocabulary` fills in `{bank_name}`, `{currency}`, `{country}`, `{partner_kind}` and `{partner_kinds}`,
`{policy_rate_key}`, `{business_segments}` and the window (`{window_from}`, `{window_to}`, `{as_of}`), and a
column's allowed values are either fixed codes of the generator or read from the profile (its products,
segments, rate series, acquired banks). One column's name is itself the profile's (`{national_id_label}`).

`describe(profile_key)` renders it for one bank. `bankgen.load` writes it into each DuckDB file with
`comment_statements`; `scripts/publish_data.py` writes the same statements, the catalog and schema
comments and the keys to Unity Catalog. `tests/bankgen/test_dictionary.py` fails when a table, view
or column has no description or a description names a column that does not exist.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from bankgen.calendar import N_MONTHS, WINDOW_END, month_label
from bankgen.profiles import get_profile
from bankgen.profiles.base import Profile
from bankgen.tables.loans import MONTH_STATUSES

Values = tuple[str, ...] | Callable[[Profile], tuple[str, ...]]


@dataclass(frozen=True)
class Column:
    text: str  # what the column means, its unit; may hold {placeholders}
    values: Values | None = None  # the allowed values of a code column


@dataclass(frozen=True)
class ForeignKey:
    column: str
    parent: str
    parent_column: str


@dataclass(frozen=True)
class Relation:
    text: str  # what one row is (the grain), what the table is for
    columns: dict[str, Column]
    primary_key: tuple[str, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    view: bool = False


@dataclass(frozen=True)
class Described:
    """The dictionary rendered for one bank: every comment, ready to write."""

    catalog: str
    schema: str
    tables: dict[str, str]  # relation -> its comment
    columns: dict[str, dict[str, str]]  # relation -> column -> its comment


# --- values the profile decides --------------------------------------------------------------------


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def origins(p: Profile) -> tuple[str, ...]:
    return ("legacy", "new", *(f"acquired:{a.bank}" for a in p.acquisitions))


def partner_origins(p: Profile) -> tuple[str, ...]:
    return ("legacy", *(f"acquired:{a.bank}" for a in p.acquisitions), *(("hired_2026",) if p.hired_partners else ()))


def deposit_products(p: Profile) -> tuple[str, ...]:
    return tuple(d.product for d in p.deposit_products)


def account_rate_keys(p: Profile) -> tuple[str, ...]:
    return ("", *_unique(d.rate_path for d in p.deposit_products if d.rate_path))


def loan_products(p: Profile) -> tuple[str, ...]:
    return tuple(lp.product for lp in p.loan_products)


def index_names(p: Profile) -> tuple[str, ...]:
    return ("", *_unique(lp.index_name for lp in p.loan_products if lp.index_name and lp.variable_share > 0))


def energy_labels(p: Profile) -> tuple[str, ...]:
    return ("", *(p.energy_labels or {}))


def underwriters(p: Profile) -> tuple[str, ...]:
    return (p.bank_name, *(a.bank for a in p.acquisitions))


def segments(p: Profile) -> tuple[str, ...]:
    return tuple(p.segments)


def business_segments(p: Profile) -> tuple[str, ...]:
    # the segments the customers builder marks is_business
    return tuple(s for s in p.segments if s in ("business", "small_business", "commercial"))


def rate_keys(p: Profile) -> tuple[str, ...]:
    return tuple(p.rate_paths)


def txn_categories(p: Profile) -> tuple[str, ...]:
    planted = ("transfer_from_savings", "deposit", "withdrawal", "interest", "outflow_to_competitor", "fraud")
    return _unique((*(c.category for c in p.txn_categories), *planted))


def merchant_categories(p: Profile) -> tuple[str, ...]:
    return tuple(c.category for c in p.txn_categories if c.merchant_based)


def fraud_types(p: Profile) -> tuple[str, ...]:
    return _unique((*p.fraud_types, *(t for e in p.fraud_events for t in e.types)))


def detectors(p: Profile) -> tuple[str, ...]:
    return ("customer", "rule", "model", *(("positive_pay",) if p.positive_pay_share > 0 else ()))


def insurance_products(p: Profile) -> tuple[str, ...]:
    return tuple(p.insurance_products)


def insurance_channels(p: Profile) -> tuple[str, ...]:
    return (p.partner_kind, "direct")


def partner_kinds(p: Profile) -> tuple[str, ...]:
    return (p.partner_kind,)


def acquired_banks(p: Profile) -> tuple[str, ...]:
    return ("", *_unique(r.acquired_bank for r in p.regions if r.acquired_bank))


def story_entities(p: Profile) -> tuple[str, ...]:
    return _unique(e.entity for e in p.events)


# --- shared column texts ---------------------------------------------------------------------------

MONTH = Column("Calendar month, a 'YYYY-MM' string.")
MONTH_INDEX = Column("Months since the window start: 0 is {window_from}, 23 is {window_to}.")
CURRENCY = Column("Currency of the amounts: always {currency}.")
ORIGIN = Column(
    "How the customer came to the bank: 'legacy' (a customer since before the window), 'new' (joined inside "
    "the window) or 'acquired:<bank>' (came with that acquired bank's book at its conversion).",
    origins,
)
CUSTOMER_ID = Column("The customer, customers.customer_id.")
ACCOUNT_ID = Column("The deposit account, accounts.account_id.")
LOAN_ID = Column("The loan, loans.loan_id.")
REGION_ID = Column("The region, regions.region_id.")


def fk(column: str, parent: str, parent_column: str | None = None) -> ForeignKey:
    return ForeignKey(column, parent, parent_column or column)


# --- the catalog, the schema and the relations ------------------------------------------------------

CATALOG = "{bank_name}'s demo data: synthetic only, generated by bankgen, no client data."

SCHEMA = (
    "{bank_name}'s banking book, synthetic: customers, their deposit accounts, loans and insurance policies, the "
    "transaction ledger, fraud cases, monthly balances, and the reference tables (regions, {partner_kinds}, "
    "merchants, interest rates). Amounts are in {currency}; rates and percentages are in percent unless a column "
    "says otherwise. As-of rule: the book is observed at {as_of}; the window runs from {window_from} to "
    "{window_to}, the monthly tables hold one row per month of it, and no event (an opening, an origination, a "
    "transaction, a departure) is dated after {as_of}; contract dates such as maturities can lie after it. Months "
    "are 'YYYY-MM' strings. An empty string, not NULL, marks an absent code or id (partner_id, merchant_id, "
    "fraud_case_id, energy_label, index_name). Join on the ids named in each table's comment. story_events is the "
    "generator's record of the stories it planted, not bank data."
)

RELATIONS: dict[str, Relation] = {
    "regions": Relation(
        "One row per region where the bank has customers: a reference table for customers, partners and merchants.",
        {
            "region_id": Column("Region code (snake_case); the key."),
            "region_name": Column("Display name of the region."),
            "subdivision": Column("The province or state the region lies in."),
            "country": Column("Country code: {country}."),
            "customer_weight": Column(
                "Relative weight with which the generator places customers in the region (about 0 to 0.4)."
            ),
            "acquired_bank": Column(
                "The acquired bank whose customers sit in this region; empty string when none.", acquired_banks
            ),
            "is_expansion_market": Column(
                "True for a growth market, where new {partner_kinds} were hired and new customers land more often."
            ),
        },
        primary_key=("region_id",),
    ),
    "partners": Relation(
        "One row per {partner_kind} who brings customers to the bank and serves them.",
        {
            "partner_id": Column("Partner id, 'P-' and four digits; the key."),
            "partner_name": Column("Synthetic name of the {partner_kind} (a firm or a person)."),
            "partner_kind": Column("What the partner is.", partner_kinds),
            "region_id": Column("The region the partner works in, regions.region_id."),
            "origin": Column(
                "Where the partner came from: 'legacy' (with the bank before the window), 'acquired:<bank>' (came "
                "with an acquired bank at its conversion) or 'hired_2026' (hired in an expansion market inside the "
                "window).",
                partner_origins,
            ),
            "since_date": Column("Date the partner started with the bank; for an acquired partner, the conversion."),
            "satisfaction_score": Column("Synthetic satisfaction score of the partner, 2 to 10, one decimal."),
        },
        primary_key=("partner_id",),
        foreign_keys=(fk("region_id", "regions"),),
    ),
    "merchants": Relation(
        "One row per merchant that card and online payments in the ledger go to; synthetic names that fit the "
        "category.",
        {
            "merchant_id": Column("Merchant id, 'M-' and five digits; the key."),
            "merchant_name": Column("Synthetic name that fits the merchant's category."),
            "merchant_category": Column(
                "Spending category, the same codes as transactions.category.", merchant_categories
            ),
            "city": Column("City of the merchant (synthetic)."),
            "region_id": Column("The merchant's region, regions.region_id."),
            "is_online": Column("True for a merchant that sells online."),
            "popularity_weight": Column(
                "Relative weight with which the generator picks the merchant for a payment, so that a few merchants "
                "take most of the volume; not a business measure."
            ),
        },
        primary_key=("merchant_id",),
        foreign_keys=(fk("region_id", "regions"),),
    ),
    "interest_rates": Relation(
        "One row per rate series per month of the window: the policy rate, the bank's own deposit rates, the "
        "market rates its products are compared with, and the indexes variable loans float on. accounts.rate_key "
        "names the series an account earns.",
        {
            "rate_key": Column("The rate series; {policy_rate_key} is the central bank's policy rate.", rate_keys),
            "month_index": MONTH_INDEX,
            "month": MONTH,
            "rate_pct": Column("Annual rate that month in percent: 3.25 means 3.25%."),
            "is_policy_rate": Column("True for the policy rate series ({policy_rate_key})."),
        },
        primary_key=("rate_key", "month"),
    ),
    "story_events": Relation(
        "The generator's record of the stories it planted in the synthetic data, one row per story: not bank data "
        "and nothing a bank would hold. The generator's checks read it; leave it out of any analysis.",
        {
            "event_id": Column("Story id; the key."),
            "event_name": Column("Short name of the story."),
            "month_from": Column("First month of the story, 'YYYY-MM'."),
            "month_to": Column("Last month of the story, 'YYYY-MM'."),
            "description": Column("What the generator planted, in words."),
            "entity": Column("The table the story shows in most.", story_entities),
        },
        primary_key=("event_id",),
    ),
    "customers": Relation(
        "One row per customer, still with the bank or gone (churned_date). Synthetic people and businesses: names, "
        "addresses, emails and ids are fake.",
        {
            "customer_id": Column("Customer id, 'C-' and six digits; the key."),
            "first_name": Column("Synthetic first name."),
            "last_name": Column("Synthetic last name."),
            "gender": Column("Gender code.", ("F", "M", "X")),
            "birth_date": Column("Date of birth."),
            "age": Column("Age in whole years at {as_of}."),
            "segment": Column("Customer segment.", segments),
            "region_id": Column("The region the customer lives in, regions.region_id."),
            "city": Column("City (synthetic)."),
            "postal_code": Column("Postal code (synthetic)."),
            "email": Column("Synthetic email address on an example domain."),
            "partner_id": Column(
                "The {partner_kind} who serves the customer, partners.partner_id; empty string when the customer "
                "came direct."
            ),
            "origin": ORIGIN,
            "customer_since": Column(
                "Date the relationship started; for an acquired customer, at the old bank. Where the relationship "
                "began with a loan, the loan's origination date. Never after {as_of}."
            ),
            "on_book_since": Column(
                "Date the customer came on this bank's book: customer_since, or for an acquired customer the first "
                "day of the conversion month. The customer's first deposit product opens that day."
            ),
            "aow_date": Column(
                "Date the customer reaches the state pension age (67 throughout this data); "
                "loans.matures_after_pension_age compares a loan's maturity with it."
            ),
            "annual_income": Column(
                "Gross annual income in {currency}, rounded to hundreds; for a business, its annual revenue."
            ),
            "credit_score": Column("Credit score on a 300 to 850 scale."),
            "is_business": Column("True for a business customer; the business segments are {business_segments}."),
            "has_positive_pay": Column(
                "True for a business customer who uses positive pay, the check and ACH fraud control; always false "
                "where the bank offers none."
            ),
            "{national_id_label}": Column("Last four digits of the customer's national identification number; fake."),
            "churned_date": Column("Date the customer left the bank; NULL while the customer stays."),
        },
        primary_key=("customer_id",),
        foreign_keys=(fk("region_id", "regions"), fk("partner_id", "partners")),
    ),
    "accounts": Relation(
        "One row per deposit product a customer holds or held (a payment or savings account, a term deposit and "
        "the like), opened no later than {as_of}. A customer's first deposit product opens on "
        "customers.on_book_since; further products follow later. The ledger (transactions) and "
        "account_balances_monthly hang off account_id.",
        {
            "account_id": Column("Account id, 'A-' and six digits; the key."),
            "customer_id": CUSTOMER_ID,
            "product": Column("Deposit product.", deposit_products),
            "account_number": Column(
                "Synthetic account number in the market's format (an IBAN, or a routing and account number)."
            ),
            "currency": CURRENCY,
            "opened_date": Column(
                "Date the account opened on this bank's ledger; for an acquired customer, the conversion at the "
                "earliest. Never after {as_of}."
            ),
            "is_transactional": Column("True for an account that carries day-to-day payments (the spending ledger)."),
            "is_interest_bearing": Column("True when the account earns interest (rate_key is set)."),
            "rate_key": Column(
                "The interest_rates series the account earns; empty string when it earns none.", account_rate_keys
            ),
            "opening_balance": Column(
                "Balance in {currency} the ledger starts from: at opening, or at the window start for an older account."
            ),
            "term_months": Column("Term of a fixed-term deposit in months; 0 for a product without a term."),
            "origin": ORIGIN,
            "has_positive_pay": Column("True for a business account under positive pay (the customer's control)."),
            "closed_date": Column(
                "Date the account closed when the customer left, its balance paid out that day; NULL while open."
            ),
            "maturity_date": Column(
                "Maturity of a fixed-term deposit, rolled forward to the window; NULL for other products. A contract "
                "date: it can lie after {as_of}."
            ),
        },
        primary_key=("account_id",),
        foreign_keys=(fk("customer_id", "customers"),),
    ),
    "loans": Relation(
        "One row per loan the bank holds or held, originated no later than {as_of}, older loans from before the "
        "window included; loan_balances_monthly has each loan's path month by month. Amounts in {currency}, rates "
        "in percent.",
        {
            "loan_id": Column("Loan id, 'L-' and six digits; the key."),
            "customer_id": Column("The borrower, customers.customer_id."),
            "product": Column("Loan product.", loan_products),
            "origination_date": Column(
                "Date the loan was originated; for an acquired loan, at the old bank. Never after {as_of}."
            ),
            "term_months": Column("Contractual term in months."),
            "maturity_date": Column("Contractual maturity: origination plus the term. It can lie after {as_of}."),
            "original_amount": Column("Amount lent at origination, {currency}."),
            "interest_rate_pct": Column(
                "Annual rate at origination in percent; the rate of a month, after resets and index moves, is "
                "loan_balances_monthly.interest_rate_pct."
            ),
            "rate_type": Column(
                "'fixed', or 'variable' (floats on index_name plus margin_pct).", ("fixed", "variable")
            ),
            "index_name": Column("The index a variable loan floats on; empty string otherwise.", index_names),
            "margin_pct": Column(
                "Spread over the policy rate at origination, in percentage points; a variable loan pays its index "
                "plus this margin."
            ),
            "is_interest_only": Column("True when the loan repays no principal before maturity."),
            "ltv": Column("Loan-to-value at origination: original_amount / property_value, 0.2 to about 1.05."),
            "current_property_value": Column(
                "Collateral value today: property_value indexed to the current price level (and the energy label "
                "where there is one), {currency}."
            ),
            "current_ltv": Column("Balance at the window end divided by current_property_value."),
            "amortization_months": Column(
                "Months the repayment schedule amortises over; longer than term_months for a loan with a balloon."
            ),
            "balloon_amount": Column(
                "Principal still due at maturity when the schedule amortises over more than the term, {currency}; "
                "0 otherwise."
            ),
            "underwritten_by": Column(
                "The bank that underwrote the loan: {bank_name}, or the acquired bank for a loan it wrote before its "
                "conversion.",
                underwriters,
            ),
            "age_at_maturity": Column("Borrower's age in whole years at maturity_date."),
            "matures_after_pension_age": Column(
                "True when maturity_date falls after the borrower's state pension date (customers.aow_date)."
            ),
            "income_at_maturity": Column(
                "Modelled borrower income at maturity, {currency}: today's income, or past the pension date 70% of "
                "it (60% for a business) with a state-pension floor."
            ),
            "collateral_type": Column("What secures the loan ('unsecured' when nothing does)."),
            "property_value": Column("Collateral value at origination: original_amount / ltv, {currency}."),
            "energy_label": Column(
                "Energy label of the mortgaged home, 'A' (best) to 'G'; empty string for other loans and where the "
                "market records none.",
                energy_labels,
            ),
            "has_nhg": Column(
                "True when the mortgage carries the national mortgage guarantee; always false where the market has "
                "none."
            ),
            "sold_to_secondary": Column("True when the loan was sold on the secondary market."),
            "sale_margin_bps": Column("Gain-on-sale margin of a sold loan in basis points; 0 when not sold."),
            "partner_id": Column(
                "The {partner_kind} who brought the loan, partners.partner_id; empty string when none."
            ),
            "origin": ORIGIN,
            "currency": CURRENCY,
            "status_at_window_end": Column(
                "The loan's state at {as_of}: 'charged_off' (charged off in a month of the window, with that "
                "month's 'charged_off' row in loan_balances_monthly), 'delinquent' (past due and neither cured nor "
                "charged off by then, on the book and unpaid if it matured), 'paid_off' (matured on or before "
                "{as_of}) or 'current'. "
                "It agrees with the loan's last row in loan_balances_monthly.",
                ("current", "delinquent", "paid_off", "charged_off"),
            ),
            "first_delinquent_month": Column(
                "Month the loan first fell 30 days past due inside the window, 'YYYY-MM'; empty string when never."
            ),
            "delinquency_outcome": Column(
                "What became of the loan's delinquency by {as_of}: 'cured' or 'charged_off'; empty string when it "
                "never fell past due or is still past due at {as_of}.",
                ("", "cured", "charged_off"),
            ),
            "fixed_rate_period_end": Column(
                "End of the fixed-rate period of a fixed-rate mortgage, after which the rate resets to the market "
                "mortgage rate; NULL for other loans. It can lie after {as_of}."
            ),
        },
        primary_key=("loan_id",),
        foreign_keys=(fk("customer_id", "customers"), fk("partner_id", "partners")),
    ),
    "loan_balances_monthly": Relation(
        "One row per loan per month while the loan is on the book (from origination, or the conversion of an "
        "acquired book, to maturity or charge-off), {window_from} to {window_to}. Balances are month-end, in "
        "{currency}; the book at a month is the sum of outstanding_balance over that month.",
        {
            "loan_id": LOAN_ID,
            "month": MONTH,
            "month_index": MONTH_INDEX,
            "outstanding_balance": Column("Principal outstanding at month end."),
            "interest_rate_pct": Column(
                "Annual rate that month in percent: the fixed rate, the reset rate once a fixed period has ended, "
                "or the index plus the margin."
            ),
            "scheduled_payment": Column("Payment due that month: the interest alone, or the annuity."),
            "paid_amount": Column("Amount paid that month: the scheduled payment, or 0 while past due."),
            "interest_income": Column("Interest the bank earned that month on the outstanding balance."),
            "days_past_due": Column(
                "Days past due at month end: 0, 30, 60, 90 or 120 (capped); delinquent means 30 or more."
            ),
            "status": Column(
                "The loan's state that month: 'current', '30dpd' to '120dpd' by days past due, 'charged_off' in the "
                "month of charge-off (the month after a loan that will not cure reaches 120 days past due; the "
                "row still carries the balance and 120 days), 'paid_off' or 'matured_balloon' (matured with a "
                "balloon still due) in the month of maturity. A loan past due when it matures stays on the book, "
                "unpaid, until it cures or is charged off.",
                MONTH_STATUSES,
            ),
        },
        primary_key=("loan_id", "month"),
        foreign_keys=(fk("loan_id", "loans"),),
    ),
    "insurance_policies": Relation(
        "One row per insurance policy a customer holds or held, sold by the bank (the cross-sell table), started "
        "no later than {as_of}. Premiums and commissions are per year, in {currency}.",
        {
            "policy_id": Column("Policy id, 'I-' and six digits; the key."),
            "customer_id": CUSTOMER_ID,
            "product": Column("Insurance product.", insurance_products),
            "start_date": Column("Date the policy started; never after {as_of}."),
            "annual_premium": Column("Premium per year, {currency}."),
            "commission_rate": Column("Share of the premium the bank earns as commission, 0.06 to 0.25."),
            "annual_commission": Column("Commission per year: annual_premium times commission_rate, {currency}."),
            "channel": Column("Sold by the customer's {partner_kind}, or 'direct'.", insurance_channels),
            "partner_id": Column(
                "The {partner_kind} who sold it, partners.partner_id; empty string for a direct sale."
            ),
            "status": Column("'active', or 'lapsed' (end_date set).", ("active", "lapsed")),
            "currency": CURRENCY,
            "end_date": Column("Date a lapsed policy ended; NULL while active."),
        },
        primary_key=("policy_id",),
        foreign_keys=(fk("customer_id", "customers"), fk("partner_id", "partners")),
    ),
    "account_balances_monthly": Relation(
        "One row per deposit account per month while the account is open, {window_from} to {window_to}, derived "
        "from the ledger so that it reconciles: end_balance = opening_balance + net_flow + interest_paid, except in "
        "the month an account closes, when the closing payout also leaves and end_balance is 0. Amounts in "
        "{currency}; the deposit book at a month is the sum of end_balance over that month.",
        {
            "account_id": ACCOUNT_ID,
            "month": MONTH,
            "month_index": MONTH_INDEX,
            "opening_balance": Column("Balance at the start of the month."),
            "net_flow": Column(
                "Money in minus money out in the month, before interest. It leaves out the payout of an account that "
                "closes in the month, as it leaves out the balance an account opens with."
            ),
            "interest_paid": Column("Interest credited to the account at month end."),
            "end_balance": Column("Balance at month end; 0 in the month the account closes."),
            "rate_pct": Column("Annual rate the account earned that month in percent; 0 when it earns none."),
        },
        primary_key=("account_id", "month"),
        foreign_keys=(fk("account_id", "accounts"),),
    ),
    "fraud_cases": Relation(
        "One row per fraud case on a transactional account, with what was taken, recovered and lost, in "
        "{currency}. The case's fraudulent transactions carry its case_id in transactions.fraud_case_id.",
        {
            "case_id": Column("Case id, 'F-' and six digits; the key."),
            "customer_id": CUSTOMER_ID,
            "account_id": ACCOUNT_ID,
            "first_txn_id": Column("The case's first fraudulent transaction, transactions.txn_id."),
            "opened_ts": Column(
                "When the case was opened (reported): up to five days after the first transaction, never after {as_of}."
            ),
            "month": Column("Month of the first fraudulent transaction, 'YYYY-MM'."),
            "fraud_type": Column("Kind of fraud.", fraud_types),
            "channel": Column("The payment rail the fraud used, as in transactions.rail."),
            "reported_month": Column("Month the case was opened, 'YYYY-MM' (the month of opened_ts)."),
            "amount": Column("Total taken in the case's transactions, a positive amount."),
            "recovered_amount": Column("Part of the amount recovered."),
            "loss_amount": Column("What the case lost: amount minus recovered_amount."),
            "detected_by": Column("Who or what detected the fraud.", detectors),
            "status": Column("'closed', or 'open' while the case was still open at {as_of}.", ("open", "closed")),
            "closed_ts": Column("When the case closed, never after {as_of}; NULL while the case is still open."),
            "currency": CURRENCY,
        },
        primary_key=("case_id",),
        foreign_keys=(
            fk("customer_id", "customers"),
            fk("account_id", "accounts"),
            fk("first_txn_id", "transactions", "txn_id"),
        ),
    ),
    "transactions": Relation(
        "The ledger: one row per transaction on a deposit account, {window_from} to {window_to}, never after "
        "{as_of}. amount is signed from the customer's view (negative leaves the account), in {currency}. "
        "Spending, income and payments land on transactional accounts; deposits, withdrawals, interest and "
        "outflows to competitors on the others; a closing account's payout, a withdrawal (a deposit when "
        "overdrawn) late on its closing day, on either.",
        {
            "txn_id": Column("Transaction id, unique across the ledger; the key."),
            "account_id": ACCOUNT_ID,
            "customer_id": CUSTOMER_ID,
            "ts": Column(
                "Timestamp of the transaction; interest is booked at midnight on the month's last day, or on the "
                "closing day of an account that closes in the month."
            ),
            "amount": Column("Signed amount, {currency}: negative leaves the account, positive comes in."),
            "rail": Column("Payment rail (card, direct debit, transfer, check, ATM and the like; 'interest')."),
            "category": Column(
                "What the money was for; 'outflow_to_competitor' is money moved to another bank, 'fraud' a "
                "fraudulent payment.",
                txn_categories,
            ),
            "merchant_id": Column(
                "The merchant of a card or online payment, merchants.merchant_id; empty string otherwise."
            ),
            "counterparty": Column(
                "The other party: the merchant's name for a merchant payment, a name that fits the category (an "
                "employer, a landlord, a utility, a person for a peer-to-peer payment), {bank_name} itself for "
                "interest, 'own account' or 'own savings' for a transfer between the customer's accounts, a "
                "competitor bank for an outflow, 'unknown' for fraud, 'external account (account closed)' for the "
                "payout of an account that closes."
            ),
            "is_fraud": Column("True for a fraudulent transaction (category 'fraud')."),
            "fraud_case_id": Column(
                "The case of a fraudulent transaction, fraud_cases.case_id; empty string otherwise."
            ),
            "currency": CURRENCY,
            "month": Column("Month of ts, 'YYYY-MM'."),
        },
        primary_key=("txn_id",),
        foreign_keys=(
            fk("account_id", "accounts"),
            fk("customer_id", "customers"),
            fk("merchant_id", "merchants"),
            fk("fraud_case_id", "fraud_cases", "case_id"),
        ),
    ),
    "v_customer_products": Relation(
        "View, one row per customer: how many products the customer holds or held. Deposit accounts and loans "
        "count whether open or closed; insurance counts active policies only.",
        {
            "customer_id": CUSTOMER_ID,
            "deposit_accounts": Column("Deposit accounts ever held, closed ones included."),
            "loans": Column("Loans ever held, repaid ones included."),
            "policies": Column("Active insurance policies."),
            "distinct_products": Column(
                "Distinct deposit products plus distinct loan products plus distinct active insurance products: "
                "the cross-sell measure."
            ),
        },
        foreign_keys=(fk("customer_id", "customers"),),
        view=True,
    ),
    "v_monthly_deposits": Relation(
        "View: deposit balances summed per month, product, region, segment and origin, from "
        "account_balances_monthly; prefer it for deposit series. Amounts in {currency}.",
        {
            "month": MONTH,
            "month_index": MONTH_INDEX,
            "product": Column("Deposit product, accounts.product.", deposit_products),
            "region_id": Column("The customer's region, regions.region_id."),
            "segment": Column("The customer's segment.", segments),
            "origin": ORIGIN,
            "end_balance": Column("Sum of month-end balances."),
            "interest_paid": Column("Sum of interest credited in the month."),
            "accounts": Column("Number of accounts open in the month."),
        },
        foreign_keys=(fk("region_id", "regions"),),
        view=True,
    ),
    "v_monthly_loans": Relation(
        "View: loan balances summed per month, product, collateral type, origin, rate type and region, from "
        "loan_balances_monthly; prefer it for loan series. Amounts in {currency}.",
        {
            "month": MONTH,
            "month_index": MONTH_INDEX,
            "product": Column("Loan product, loans.product.", loan_products),
            "collateral_type": Column("What secures the loans, loans.collateral_type."),
            "origin": ORIGIN,
            "rate_type": Column("'fixed' or 'variable'.", ("fixed", "variable")),
            "region_id": Column("The borrower's region, regions.region_id."),
            "outstanding_balance": Column("Sum of month-end outstanding balances."),
            "interest_income": Column("Sum of interest earned in the month."),
            "delinquent_balance": Column("Outstanding balance of the loans 30 or more days past due."),
            "charged_off_balance": Column("Outstanding balance of the loans charged off in the month."),
            "loans": Column("Number of loans on the book in the month."),
        },
        foreign_keys=(fk("region_id", "regions"),),
        view=True,
    ),
}


# --- rendering ---------------------------------------------------------------------------------------


def vocabulary(p: Profile) -> dict[str, str]:
    """The words a description may use as `{placeholder}`, from the bank's profile and the window."""
    return {
        "bank_name": p.bank_name,
        "country": p.country,
        "currency": p.currency,
        "partner_kind": p.partner_kind,
        "partner_kinds": f"{p.partner_kind[:-1]}ies" if p.partner_kind.endswith("y") else f"{p.partner_kind}s",
        "policy_rate_key": p.policy_rate_key,
        "national_id_label": p.national_id_label,
        "business_segments": ", ".join(literal(s) for s in business_segments(p)) or "none",
        "window_from": month_label(0),
        "window_to": month_label(N_MONTHS - 1),
        "as_of": f"{WINDOW_END.day} {WINDOW_END:%B %Y}",
    }


def literal(value: str) -> str:
    """A value as the descriptions quote it: 'savings', and '' for the empty string."""
    return f"'{value}'"


def values_of(column: Column, p: Profile) -> tuple[str, ...] | None:
    if column.values is None:
        return None
    return column.values(p) if callable(column.values) else column.values


def column_comment(column: Column, p: Profile) -> str:
    text = column.text.format_map(vocabulary(p))
    values = values_of(column, p)
    return f"{text} Values: {', '.join(literal(v) for v in values)}." if values else text


def joins_text(name: str, relation: Relation) -> str:
    """The joins of a relation, from its foreign keys and those of the tables that refer to it."""
    out = [f"{k.column} = {k.parent}.{k.parent_column}" for k in relation.foreign_keys]
    referenced = [
        f"{other}.{k.column}"
        for other, r in RELATIONS.items()
        if not r.view
        for k in r.foreign_keys
        if k.parent == name
    ]
    parts = []
    if relation.primary_key:
        parts.append(f"Key: {', '.join(relation.primary_key)}.")
    if out:
        parts.append(f"Joins: {'; '.join(out)}.")
    if referenced:
        parts.append(f"Referenced by {', '.join(referenced)}.")
    return " ".join(parts)


def table_comment(name: str, relation: Relation, p: Profile) -> str:
    text = relation.text.format_map(vocabulary(p))
    joins = joins_text(name, relation)
    return f"{text} {joins}" if joins else text


def column_name(name: str, p: Profile) -> str:
    """A column's name in the bank's schema: the dictionary key, or the profile's name for a `{placeholder}` key."""
    return name.format_map(vocabulary(p))


def describe(profile_key: str) -> Described:
    """The dictionary rendered for one bank."""
    p = get_profile(profile_key)
    words = vocabulary(p)
    return Described(
        catalog=CATALOG.format_map(words),
        schema=SCHEMA.format_map(words),
        tables={name: table_comment(name, r, p) for name, r in RELATIONS.items()},
        columns={
            name: {column_name(c, p): column_comment(col, p) for c, col in r.columns.items()}
            for name, r in RELATIONS.items()
        },
    )


def problems(described: Described, actual: Mapping[str, Iterable[str]]) -> list[str]:
    """What the dictionary and a schema disagree on: a relation or column with no description, a description of a
    relation or column that does not exist, an empty description. `actual` maps each relation to its columns."""
    out: list[str] = []
    for table, cols in actual.items():
        if table not in described.tables:
            out.append(f"{table}: no description")
            continue
        named = described.columns[table]
        cols = list(cols)
        out += [f"{table}.{c}: no description" for c in cols if c not in named]
        out += [f"{table}.{c}: described but not in the table" for c in named if c not in cols]
        out += [f"{table}.{c}: empty description" for c, text in named.items() if not text.strip()]
    out += [f"{t}: described but not in the schema" for t in described.tables if t not in actual]
    return out


def sql_string(text: str) -> str:
    """A SQL string literal, the same in DuckDB and Databricks SQL (the texts hold no backslash)."""
    return "'" + text.replace("'", "''") + "'"


def comment_statements(described: Described, relation: str, name: Callable[[str], str] = lambda t: t) -> list[str]:
    """`COMMENT ON TABLE` and one `COMMENT ON COLUMN` per column for one relation: statements DuckDB and Databricks
    SQL both accept, for a table or a view; `name` qualifies the relation (Unity Catalog's three-part name)."""
    full = name(relation)
    return [f"COMMENT ON TABLE {full} IS {sql_string(described.tables[relation])}"] + [
        f"COMMENT ON COLUMN {full}.{column} IS {sql_string(text)}"
        for column, text in described.columns[relation].items()
    ]
