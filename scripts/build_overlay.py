"""Build the fraud-ring overlay on Cauchy Bank's book: seeded mule accounts and the payments that reach them.

Run as ``uv run scripts/build_overlay.py --profile <PROFILE> --catalog <CATALOG> --schema <SCHEMA> [--warehouse
<id or name>] [--seed 20261105] [--apply]``, after ``scripts/publish_data.py`` has published the book to the same
schema. A dry run (the default) reads the book, builds the overlay, checks it and prints a summary and a
fingerprint of its rows; it writes nothing. ``--apply`` also replaces the three overlay tables in the same schema
and reads their row counts back.

The synthetic book has no receiving side for fraud (``bankgen`` writes an empty merchant and the counterparty
'unknown'), so this script supplies it:

- ``mule_accounts``: four accounts of ``RING-01``, and one lone mule for every case outside the ring.
- ``fraud_payment_beneficiaries``: the mule account each fraudulent payment went to.
- ``unreported_ring_payments``: 5 to 8 customers with no fraud case who also paid the ring in June 2026.

The ring takes 80% of the E4 wave (the phishing and bank-helpdesk cases of May and June 2026); 5 to 8 of its
victims pay two of the four accounts, which joins them into one cluster. The same seed and the same book give the
same rows. Re-run the script after any republish of the book: a change to the generator can change the case and
payment ids the overlay points at.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections.abc import Sequence
from dataclasses import astuple, dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

DEFAULT_SEED = 20261105

WAVE_MONTHS = ("2026-05", "2026-06")
WAVE_TYPES = ("phishing", "bank_helpdesk_scam")  # E4 in story_events
RING_ID = "RING-01"
RING_ACCOUNTS = 4
RING_SHARE = 0.8
TWO_ACCOUNT_VICTIMS = (5, 8)
UNREPORTED_PAYERS = (5, 8)
UNREPORTED_FROM = datetime(2026, 6, 8)
UNREPORTED_TO = datetime(2026, 6, 30, 23, 59, 59)
RING_OPENED_FROM = date(2026, 2, 1)
RING_OPENED_DAYS = 73  # up to 15 April 2026, before the wave's first payment
RAIL = "sepa_transfer"

# Fictitious receiving banks: no code is a real bank's, so no IBAN resembles a real account
BANKS = (("External Bank A", "EXTA"), ("External Bank B", "EXTB"), ("External Bank C", "EXTC"))
FIRST_NAMES = (
    "Daan", "Sem", "Lucas", "Milan", "Levi", "Noah", "Jesse", "Thijs", "Ruben", "Bram",
    "Emma", "Julia", "Mila", "Sophie", "Tess", "Anna", "Sara", "Lotte", "Fleur", "Iris",
)  # fmt: skip
LAST_NAMES = (
    "de Vries", "Jansen", "van den Berg", "Bakker", "Visser", "Smit", "Meijer", "de Boer", "Mulder",
    "de Groot", "Bos", "Vos", "Peters", "Hendriks", "van Leeuwen", "Dekker", "Brouwer", "de Wit",
)  # fmt: skip


@dataclass(frozen=True)
class FraudCase:
    """A fraud case of the book (``fraud_cases``)."""

    case_id: str
    customer_id: str
    account_id: str
    month: str
    fraud_type: str


@dataclass(frozen=True)
class FraudTxn:
    """A fraudulent payment of the book (``transactions`` where ``is_fraud``); the amount is a negative debit."""

    txn_id: int
    case_id: str
    ts: datetime
    amount: float


@dataclass(frozen=True)
class Candidate:
    """A customer with no fraud case and an open payment account, who may become an unreported payer."""

    customer_id: str
    account_id: str


@dataclass(frozen=True)
class MuleAccount:
    """A row of ``mule_accounts``; ``ring_id`` is ``''`` for a lone mule, as the book marks an absent id."""

    mule_account_id: str
    iban: str
    holder_name: str
    bank_name: str
    opened_date: date
    ring_id: str


@dataclass(frozen=True)
class Beneficiary:
    """A row of ``fraud_payment_beneficiaries``: where one fraudulent payment went."""

    txn_id: int
    case_id: str
    mule_account_id: str


@dataclass(frozen=True)
class UnreportedPayment:
    """A row of ``unreported_ring_payments``; the amount is the positive amount paid."""

    payment_id: str
    customer_id: str
    account_id: str
    ts: datetime
    amount: float
    rail: str
    mule_account_id: str


@dataclass(frozen=True)
class Overlay:
    """The three tables of the overlay."""

    mule_accounts: tuple[MuleAccount, ...]
    beneficiaries: tuple[Beneficiary, ...]
    unreported: tuple[UnreportedPayment, ...]


def nl_iban(bank_code: str, account: str) -> str:
    """Return a Dutch-format IBAN with valid check digits (ISO 13616, mod 97).

    Args:
        bank_code: The four-letter bank code.
        account: The ten-digit account number.

    Returns:
        The IBAN, e.g. ``NL91EXTA0123456789``.
    """
    rearranged = f"{bank_code}{account}NL00"
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return f"NL{98 - int(digits) % 97:02d}{bank_code}{account}"


def is_wave(case: FraudCase) -> bool:
    """Tell whether a case belongs to the E4 wave.

    Args:
        case: The fraud case.

    Returns:
        True for a phishing or bank-helpdesk case of May or June 2026.
    """
    return case.month in WAVE_MONTHS and case.fraud_type in WAVE_TYPES


def ring_is_connected(pairs: Sequence[tuple[int, int]]) -> bool:
    """Tell whether victims paying two accounts join all ring accounts into one cluster.

    Args:
        pairs: The two ring accounts (by index) each two-account victim paid.

    Returns:
        True when every ring account is reachable from the first.
    """
    reached, frontier = {0}, [0]
    while frontier:
        node = frontier.pop()
        for a, b in pairs:
            for here, there in ((a, b), (b, a)):
                if here == node and there not in reached:
                    reached.add(there)
                    frontier.append(there)
    return len(reached) == RING_ACCOUNTS


def _mule(rng: random.Random, mule_id: str, opened: date, ring_id: str) -> MuleAccount:
    bank_name, bank_code = rng.choice(BANKS)
    account = f"{rng.randrange(10**10):010d}"
    holder = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    return MuleAccount(mule_id, nl_iban(bank_code, account), holder, bank_name, opened, ring_id)


def build_overlay(
    cases: Sequence[FraudCase], fraud_txns: Sequence[FraudTxn], candidates: Sequence[Candidate], seed: int
) -> Overlay:
    """Build the overlay from the book's fraud and a seed; the same inputs and seed give the same rows.

    Args:
        cases: Every fraud case of the book.
        fraud_txns: Every fraudulent payment of the book.
        candidates: Customers with no fraud case and an open payment account.
        seed: The random seed.

    Returns:
        The three tables.

    Raises:
        ValueError: The book cannot hold the ring the design asks for.
    """
    rng = random.Random(seed)
    cases = sorted(cases, key=lambda c: c.case_id)
    txns_of: dict[str, list[FraudTxn]] = {c.case_id: [] for c in cases}
    for txn in sorted(fraud_txns, key=lambda t: t.txn_id):
        txns_of[txn.case_id].append(txn)

    wave = [c for c in cases if is_wave(c)]
    ring_cases = rng.sample(wave, round(RING_SHARE * len(wave)))  # in drawn order: round-robin over accounts
    primary = {c.case_id: i % RING_ACCOUNTS for i, c in enumerate(ring_cases)}
    eligible = sorted((c for c in ring_cases if len(txns_of[c.case_id]) >= 2), key=lambda c: c.case_id)
    if len(eligible) < TWO_ACCOUNT_VICTIMS[0]:
        raise ValueError(f"only {len(eligible)} ring cases have two payments; {TWO_ACCOUNT_VICTIMS[0]} are needed")
    for _ in range(1000):
        k = rng.randint(TWO_ACCOUNT_VICTIMS[0], min(TWO_ACCOUNT_VICTIMS[1], len(eligible)))
        two_account = rng.sample(eligible, k)
        pairs = [(primary[c.case_id], (primary[c.case_id] + 1) % RING_ACCOUNTS) for c in two_account]
        if ring_is_connected(pairs):
            break
    else:
        raise ValueError("no draw of two-account victims joins the four ring accounts")
    second = {c.case_id for c in two_account}

    ring_ids = {c.case_id for c in ring_cases}
    lone_cases = [c for c in cases if c.case_id not in ring_ids and txns_of[c.case_id]]  # a mule needs a payment
    mule_ids = [f"MA-{n:04d}" for n in range(1, RING_ACCOUNTS + len(lone_cases) + 1)]
    rng.shuffle(mule_ids)  # the ring's ids are not the first four

    ring = [
        _mule(rng, mule_ids[i], RING_OPENED_FROM + timedelta(days=rng.randint(0, RING_OPENED_DAYS)), RING_ID)
        for i in range(RING_ACCOUNTS)
    ]
    mules = list(ring)
    beneficiaries: list[Beneficiary] = []
    for case in sorted(ring_cases, key=lambda c: c.case_id):
        txns = txns_of[case.case_id]
        first = primary[case.case_id]
        for j, txn in enumerate(txns):
            last = j == len(txns) - 1 and case.case_id in second
            account = ring[(first + 1) % RING_ACCOUNTS] if last else ring[first]
            beneficiaries.append(Beneficiary(txn.txn_id, case.case_id, account.mule_account_id))
    for i, case in enumerate(lone_cases):
        txns = txns_of[case.case_id]
        first_day = min(t.ts for t in txns).date()
        mule = _mule(rng, mule_ids[RING_ACCOUNTS + i], first_day - timedelta(days=rng.randint(7, 60)), "")
        mules.append(mule)
        beneficiaries.extend(Beneficiary(t.txn_id, case.case_id, mule.mule_account_id) for t in txns)

    case_customers = {c.customer_id for c in cases}
    pool = sorted((c for c in candidates if c.customer_id not in case_customers), key=lambda c: c.customer_id)
    payers = rng.sample(pool, rng.randint(*UNREPORTED_PAYERS))
    ring_amounts = sorted(abs(t.amount) for c in ring_cases for t in txns_of[c.case_id])
    window = int((UNREPORTED_TO - UNREPORTED_FROM).total_seconds())
    unreported: list[UnreportedPayment] = []
    for payer in payers:
        for _ in range(rng.randint(1, 2)):
            unreported.append(
                UnreportedPayment(
                    f"UP-{len(unreported) + 1:03d}",
                    payer.customer_id,
                    payer.account_id,
                    UNREPORTED_FROM + timedelta(seconds=rng.randint(0, window)),
                    round(rng.choice(ring_amounts), 2),
                    RAIL,
                    rng.choice(ring).mule_account_id,
                )
            )

    return Overlay(
        tuple(sorted(mules, key=lambda m: m.mule_account_id)),
        tuple(sorted(beneficiaries, key=lambda b: b.txn_id)),
        tuple(unreported),
    )


def check_overlay(overlay: Overlay, cases: Sequence[FraudCase], fraud_txns: Sequence[FraudTxn]) -> list[str]:
    """Check the overlay against its rules: the ring's size and share, the two-account victims, the lone mules
    and the unreported payers.

    Args:
        overlay: The overlay to check.
        cases: Every fraud case of the book.
        fraud_txns: Every fraudulent payment of the book.

    Returns:
        One line per broken rule; empty when every rule holds.
    """
    failures: list[str] = []
    mules = {m.mule_account_id: m for m in overlay.mule_accounts}
    ring = {m.mule_account_id for m in overlay.mule_accounts if m.ring_id == RING_ID}
    if len(ring) != RING_ACCOUNTS:
        failures.append(f"{RING_ID} has {len(ring)} accounts, not {RING_ACCOUNTS}")
    if any(m.ring_id not in ("", RING_ID) for m in overlay.mule_accounts):
        failures.append("a mule account belongs to a ring other than RING-01")

    txn_case = {t.txn_id: t.case_id for t in fraud_txns}
    paid = [b.txn_id for b in overlay.beneficiaries]
    if sorted(paid) != sorted(txn_case):
        failures.append("the beneficiaries do not cover every fraudulent payment exactly once")
    if any(txn_case.get(b.txn_id) != b.case_id for b in overlay.beneficiaries):
        failures.append("a beneficiary names the wrong case for its payment")
    if any(b.mule_account_id not in mules for b in overlay.beneficiaries):
        failures.append("a beneficiary names an unknown mule account")

    accounts_of: dict[str, set[str]] = {}
    for b in overlay.beneficiaries:
        accounts_of.setdefault(b.case_id, set()).add(b.mule_account_id)
    wave = {c.case_id for c in cases if is_wave(c)}
    ring_cases = {case for case, accounts in accounts_of.items() if accounts & ring}
    if not ring_cases <= wave:
        failures.append("a case outside the wave pays the ring")
    share = len(ring_cases) / len(wave) if wave else 0.0
    if abs(share - RING_SHARE) > 0.05:
        failures.append(f"the ring takes {share:.0%} of the wave, not about {RING_SHARE:.0%}")
    two = [case for case in ring_cases if len(accounts_of[case]) == 2]
    if not TWO_ACCOUNT_VICTIMS[0] <= len(two) <= TWO_ACCOUNT_VICTIMS[1]:
        failures.append(f"{len(two)} ring victims pay two accounts, not 5 to 8")
    index = {mule_id: i for i, mule_id in enumerate(sorted(ring))}
    pairs = []
    for case in two:
        left, right = sorted(accounts_of[case])
        pairs.append((index[left], index[right]))
    if len(ring) == RING_ACCOUNTS and not ring_is_connected(pairs):
        failures.append("the two-account victims do not join the four ring accounts")

    payers_of: dict[str, set[str]] = {}
    for case, accounts in accounts_of.items():
        for account in accounts:
            payers_of.setdefault(account, set()).add(case)
    for case, accounts in accounts_of.items():
        if case in ring_cases:
            continue
        lone = len(accounts) == 1 and all(mules[a].ring_id == "" and payers_of[a] == {case} for a in accounts)
        if not lone:
            failures.append(f"case {case} does not pay a lone mule of its own")
    if any(m.mule_account_id not in payers_of and m.ring_id == "" for m in overlay.mule_accounts):
        failures.append("a lone mule receives no payment")

    case_customers = {c.customer_id for c in cases}
    per_payer: dict[str, int] = {}
    for p in overlay.unreported:
        per_payer[p.customer_id] = per_payer.get(p.customer_id, 0) + 1
        if p.mule_account_id not in ring:
            failures.append(f"unreported payment {p.payment_id} does not go to {RING_ID}")
        if (p.ts.year, p.ts.month) != (2026, 6) or p.amount <= 0:
            failures.append(f"unreported payment {p.payment_id} is not a positive payment in June 2026")
    if not UNREPORTED_PAYERS[0] <= len(per_payer) <= UNREPORTED_PAYERS[1]:
        failures.append(f"{len(per_payer)} unreported payers, not 5 to 8")
    if any(not 1 <= n <= 2 for n in per_payer.values()):
        failures.append("an unreported payer makes other than one or two payments")
    if set(per_payer) & case_customers:
        failures.append("an unreported payer has a fraud case")
    return failures


def summary(overlay: Overlay, cases: Sequence[FraudCase]) -> dict[str, int]:
    """Count what the overlay holds, for the run's report.

    Args:
        overlay: The overlay.
        cases: Every fraud case of the book.

    Returns:
        The counts by name.
    """
    ring = {m.mule_account_id for m in overlay.mule_accounts if m.ring_id == RING_ID}
    accounts_of: dict[str, set[str]] = {}
    for b in overlay.beneficiaries:
        accounts_of.setdefault(b.case_id, set()).add(b.mule_account_id)
    ring_cases = [case for case, accounts in accounts_of.items() if accounts & ring]
    return {
        "wave cases": sum(1 for c in cases if is_wave(c)),
        "ring cases": len(ring_cases),
        "two-account victims": sum(1 for case in ring_cases if len(accounts_of[case]) == 2),
        "lone mules": len(overlay.mule_accounts) - len(ring),
        "mule accounts": len(overlay.mule_accounts),
        "beneficiaries": len(overlay.beneficiaries),
        "unreported payers": len({p.customer_id for p in overlay.unreported}),
        "unreported payments": len(overlay.unreported),
    }


def fingerprint(overlay: Overlay) -> str:
    """Hash every row of the overlay, so two runs can be compared at a glance.

    Args:
        overlay: The overlay.

    Returns:
        The first 16 hex digits of a SHA-256 over the rows.
    """
    digest = hashlib.sha256()
    for rows in (overlay.mule_accounts, overlay.beneficiaries, overlay.unreported):
        for row in rows:
            digest.update(repr(astuple(row)).encode())
    return digest.hexdigest()[:16]


# --- Databricks I/O ---------------------------------------------------------------------------------


def sql_literal(value: object) -> str:
    """Render a Python value as a Databricks SQL literal.

    Args:
        value: A str, int, float, date or datetime.

    Returns:
        The literal.

    Raises:
        TypeError: The value's type has no literal here.
    """
    if isinstance(value, datetime):
        return f"TIMESTAMP_NTZ'{value.isoformat(sep=' ')}'"
    if isinstance(value, date):
        return f"DATE'{value.isoformat()}'"
    if isinstance(value, bool):
        raise TypeError("booleans are not written by the overlay")
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    raise TypeError(f"no SQL literal for {type(value).__name__}")


TABLES: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "mule_accounts": (
        "One row per receiving account of the bank's fraud, synthetic: the four accounts of RING-01 and one lone "
        "mule per case outside the ring (scripts/build_overlay.py).",
        [
            ("mule_account_id", "STRING", "'MA-' and four digits; the key."),
            ("iban", "STRING", "Synthetic Dutch-format IBAN at a fictitious bank."),
            ("holder_name", "STRING", "Synthetic name of the account holder."),
            ("bank_name", "STRING", "The fictitious receiving bank."),
            ("opened_date", "DATE", "When the mule account was opened."),
            ("ring_id", "STRING", "The ring it belongs to (RING-01); an empty string for a lone mule."),
        ],
    ),
    "fraud_payment_beneficiaries": (
        "One row per fraudulent payment of the book: the mule account it went to, synthetic.",
        [
            ("txn_id", "BIGINT", "transactions.txn_id, a row with is_fraud true."),
            ("case_id", "STRING", "fraud_cases.case_id of the payment."),
            ("mule_account_id", "STRING", "mule_accounts.mule_account_id: where the money went."),
        ],
    ),
    "unreported_ring_payments": (
        "One row per payment into RING-01 that no one has reported, synthetic and not in transactions.",
        [
            ("payment_id", "STRING", "'UP-' and three digits; the key."),
            ("customer_id", "STRING", "customers.customer_id, a customer with no fraud case."),
            ("account_id", "STRING", "accounts.account_id, that customer's payment account."),
            ("ts", "TIMESTAMP_NTZ", "When the payment was made, in June 2026."),
            ("amount", "DOUBLE", "The amount paid in EUR, positive."),
            ("rail", "STRING", "The payment rail: sepa_transfer."),
            ("mule_account_id", "STRING", "mule_accounts.mule_account_id, one of the four RING-01 accounts."),
        ],
    ),
}


def run_sql(w: WorkspaceClient, wh: str, statement: str) -> list[list[str]]:
    """Run one statement on the warehouse and return every row, as strings.

    Args:
        w: The workspace client.
        wh: The warehouse id.
        statement: The SQL statement.

    Returns:
        The rows of the result, empty for a statement without one.

    Raises:
        SystemExit: The statement failed.
    """
    from databricks.sdk.service.sql import StatementState

    response = w.statement_execution.execute_statement(statement=statement, warehouse_id=wh, wait_timeout="50s")
    while response.status and response.status.state in (StatementState.PENDING, StatementState.RUNNING):
        response = w.statement_execution.get_statement(response.statement_id or "")
    state = response.status.state if response.status else None
    if state != StatementState.SUCCEEDED:
        error = response.status.error.message if response.status and response.status.error else state
        sys.exit(f"SQL failed: {error}\n{statement[:300]}")
    rows: list[list[str]] = list(response.result.data_array or []) if response.result else []
    chunk = response.result.next_chunk_index if response.result else None
    while chunk is not None:
        part = w.statement_execution.get_statement_result_chunk_n(response.statement_id or "", chunk)
        rows.extend(part.data_array or [])
        chunk = part.next_chunk_index
    return rows


def read_book(w: WorkspaceClient, wh: str, source: str) -> tuple[list[FraudCase], list[FraudTxn], list[Candidate]]:
    """Read the fraud cases, the fraudulent payments and the candidate payers from the book.

    Args:
        w: The workspace client.
        wh: The warehouse id.
        source: The book's ``catalog.schema``.

    Returns:
        The cases, the payments and the candidates.
    """
    cases = [
        FraudCase(str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]))
        for r in run_sql(w, wh, f"SELECT case_id, customer_id, account_id, month, fraud_type FROM {source}.fraud_cases")
    ]
    txns = [
        FraudTxn(int(str(r[0])), str(r[1]), datetime.fromisoformat(str(r[2])), float(str(r[3])))
        for r in run_sql(
            w, wh, f"SELECT txn_id, fraud_case_id, CAST(ts AS STRING), amount FROM {source}.transactions WHERE is_fraud"
        )
    ]
    candidates = [
        Candidate(str(r[0]), str(r[1]))
        for r in run_sql(
            w,
            wh,
            f"SELECT c.customer_id, MIN(a.account_id) FROM {source}.customers c "
            f"JOIN {source}.accounts a ON a.customer_id = c.customer_id "
            "WHERE a.product = 'payment' AND a.closed_date IS NULL AND c.churned_date IS NULL AND NOT c.is_business "
            f"AND c.customer_id NOT IN (SELECT customer_id FROM {source}.fraud_cases) GROUP BY c.customer_id",
        )
    ]
    return cases, txns, candidates


def write_overlay(w: WorkspaceClient, wh: str, overlay: Overlay, target: str) -> dict[str, int]:
    """Replace the three overlay tables and read their row counts back.

    Args:
        w: The workspace client.
        wh: The warehouse id.
        overlay: The overlay to write.
        target: The ``catalog.schema`` to write to.

    Returns:
        The row count of each table as stored.
    """
    rows_of: dict[str, Sequence[MuleAccount | Beneficiary | UnreportedPayment]] = {
        "mule_accounts": overlay.mule_accounts,
        "fraud_payment_beneficiaries": overlay.beneficiaries,
        "unreported_ring_payments": overlay.unreported,
    }
    counts: dict[str, int] = {}
    for table, (comment, columns) in TABLES.items():
        name = f"{target}.{table}"
        cols = ", ".join(f"{col} {kind} COMMENT {sql_literal(text)}" for col, kind, text in columns)
        run_sql(w, wh, f"CREATE OR REPLACE TABLE {name} ({cols}) COMMENT {sql_literal(comment)}")
        values = ", ".join("(" + ", ".join(sql_literal(v) for v in astuple(row)) + ")" for row in rows_of[table])
        run_sql(w, wh, f"INSERT INTO {name} VALUES {values}")
        counts[table] = int(str(run_sql(w, wh, f"SELECT COUNT(*) FROM {name}")[0][0]))
    return counts


def main(argv: list[str] | None = None) -> int:
    """Read the book, build and check the overlay, and with ``--apply`` write it.

    Args:
        argv: The arguments, or None for ``sys.argv``.

    Returns:
        The exit code.
    """
    import os

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound

    from publish_data import WAREHOUSE_VARIABLE, resolve_warehouse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--profile", required=True, help="Databricks CLI profile of the workspace")
    ap.add_argument("--catalog", required=True, help="the catalog the book was published to")
    ap.add_argument("--schema", required=True, help="the schema the book was published to; the overlay joins it")
    ap.add_argument(
        "--warehouse", help=f"SQL warehouse id or name (default: {WAREHOUSE_VARIABLE}, else the only serverless one)"
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    ap.add_argument("--apply", action="store_true", help="write the tables; without it, a dry run")
    args = ap.parse_args(argv)

    w = WorkspaceClient(profile=args.profile)
    wh, _ = resolve_warehouse(w, args.warehouse, os.environ)
    schema = f"{args.catalog}.{args.schema}"
    try:
        w.schemas.get(schema)
    except NotFound:
        sys.exit(f"{schema} does not exist; publish the book to it first (scripts/publish_data.py)")
    cases, txns, candidates = read_book(w, wh, schema)
    print(f"read {schema}: {len(cases)} cases, {len(txns)} fraudulent payments, {len(candidates)} candidate payers")
    overlay = build_overlay(cases, txns, candidates, args.seed)
    for name, count in summary(overlay, cases).items():
        print(f"  {name}: {count}")
    print(f"  fingerprint (seed {args.seed}): {fingerprint(overlay)}")
    failures = check_overlay(overlay, cases, txns)
    for failure in failures:
        print(f"FAILED: {failure}")
    if failures:
        return 1
    print("checks: every rule holds")

    if not args.apply:
        print(f"dry run: nothing written to {schema}")
        return 0
    counts = write_overlay(w, wh, overlay, schema)
    expected = {
        "mule_accounts": len(overlay.mule_accounts),
        "fraud_payment_beneficiaries": len(overlay.beneficiaries),
        "unreported_ring_payments": len(overlay.unreported),
    }
    print(f"written to {schema}: {counts}")
    if counts != expected:
        sys.exit(f"the stored counts {counts} differ from the overlay's {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
