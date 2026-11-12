"""The ledger: spending on transactional accounts, interest and transfers on the others, fraud
cases, and the monthly balance table derived from the same ledger (so it always reconciles)."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bankgen.calendar import (
    N_MONTHS,
    day_weights,
    days_in_month,
    month_end,
    month_index_of,
    month_label,
    month_start,
    random_timestamps,
    seasonal_multiplier,
)
from bankgen.context import Context
from bankgen.frame import as_days, to_table
from bankgen.profiles.base import FraudEvent, Profile
from bankgen.rng import bernoulli, lognormal_amount, weighted_choice
from bankgen.tables.dimensions import category_names

RAIL_BY_FRAUD = {
    "card_skimming": ("card_pos", "debit_card"),
    "phishing": ("sepa_transfer", "ach_debit"),
    "bank_helpdesk_scam": ("sepa_transfer", "ach_debit"),
    "account_takeover": ("sepa_transfer", "zelle"),
    "ideal_invoice_scam": ("ideal", "ach_debit"),
    "investment_scam": ("sepa_transfer", "wire"),
    "counterfeit_check": ("check", "check"),
    "check_washing": ("check", "check"),
    "ach_fraud": ("ach_debit", "ach_debit"),
    "zelle_scam": ("zelle", "zelle"),
}
COUNTERPARTY_POOL = 40
# the other side of a closing account's payout: the customer's account at another bank, not one with this bank
CLOSING_COUNTERPARTY = "external account (account closed)"
COMPETITORS = {
    "NL": ["ING", "Rabobank", "ABN AMRO", "bunq", "Openbank"],
    "US": ["Truist", "Atlantic Union Bank", "Chase", "Capital One", "Navy Federal"],
}
CREDIT_DRIFT = 0.995  # expected credits exceed expected debits by this factor, so balances grow slowly
NET_INCOME_FACTOR = 0.72  # payroll lands net of tax


def credit_scale(p: Profile) -> dict[str, float]:
    """Per segment: the factor on credit amounts that makes E[credits] = CREDIT_DRIFT x E[debits].

    E[lognormal(median, sigma)] = median * exp(sigma^2 / 2); income-based categories pay
    NET_INCOME_FACTOR x median income / 12 per month regardless of how many instalments.
    """
    out: dict[str, float] = {}
    for seg_name in p.segments:
        credits = debits = 0.0
        for cat in p.txn_categories:
            rate = cat.monthly_rate.get(seg_name, 0.0)
            if rate <= 0:
                continue
            if cat.income_based:
                expected = NET_INCOME_FACTOR * p.income_median[seg_name] / 12.0
            else:
                expected = rate * cat.amount_median * math.exp(cat.amount_sigma**2 / 2)
            if cat.direction > 0:
                credits += expected
            else:
                debits += expected
        out[seg_name] = CREDIT_DRIFT * debits / credits if credits > 0 else 1.0
    return out


REPORT_DELAY_S = 5 * 86400  # a fraud case is reported up to five days after its first transaction


def _fraud_multiplier(
    ev: FraudEvent,
    mi: int,
    product: np.ndarray,
    opened_mi: np.ndarray,
    segment: np.ndarray,
    positive_pay: np.ndarray,
) -> np.ndarray:
    if not (ev.month_from <= mi <= ev.month_to):
        return np.ones(len(product))
    m = np.ones(len(product), dtype=bool)
    if ev.target_product:
        m &= product == ev.target_product
    if ev.target_opened_after is not None:
        m &= opened_mi >= ev.target_opened_after
    if ev.target_segments:
        m &= np.isin(segment, ev.target_segments)
    if ev.target_no_positive_pay:
        m &= ~positive_pay
    return np.where(m, ev.multiplier, 1.0)


def build(ctx: Context) -> None:
    p: Profile = ctx.profile
    rng = ctx.rng("transactions")
    fake = ctx.faker("transactions")
    acc = ctx.table("accounts").to_pydict()
    cust = ctx.table("customers").to_pydict()
    merch = ctx.table("merchants").to_pydict()

    c_index = {c: i for i, c in enumerate(cust["customer_id"])}
    a_cust = np.array([c_index[c] for c in acc["customer_id"]])
    seg = np.array(cust["segment"])[a_cust]
    income = np.array(cust["annual_income"], dtype=np.float64)[a_cust]
    account_id = np.array(acc["account_id"])
    customer_id = np.array(acc["customer_id"])
    product = np.array(acc["product"])
    opened = np.array(acc["opened_date"], dtype="datetime64[D]")
    closed_raw = np.array(acc["closed_date"])
    closed = np.array([d if d is not None else month_end(N_MONTHS - 1) for d in closed_raw], dtype="datetime64[D]")
    # an account that closes runs its balance to zero in its closing month: its transactions that month
    # are dated on or before the closing day, and a closing withdrawal (a closing deposit when it is overdrawn) at the
    # end of that day takes the balance to zero, so its last row reads zero and there is no row after it. The month's
    # net_flow leaves that closing balance out, as it leaves out the balance an account opens with: it is the flow on
    # the accounts that stay, so a customer leaving does not read as a rate-driven outflow (the deposit stories hold)
    close_mi = np.array([month_index_of(d) if d is not None else N_MONTHS for d in closed_raw], dtype=np.int64)
    close_day = (closed - closed.astype("datetime64[M]").astype("datetime64[D]")).astype(np.int64)  # 0 = the 1st
    close_midnight = closed.astype("datetime64[s]")
    close_end_of_day = close_midnight + np.timedelta64(86400 - 1, "s")
    transactional = np.array(acc["is_transactional"])
    rate_key = np.array(acc["rate_key"])
    positive_pay = np.array(acc["has_positive_pay"])
    balance = np.array(acc["opening_balance"], dtype=np.float64).copy()
    maturity_raw = np.array(acc["maturity_date"])
    maturity = np.array(
        [d if d is not None else month_end(N_MONTHS - 1) for d in maturity_raw],
        dtype="datetime64[D]",
    )
    n_acc = len(account_id)
    opened_mi = np.clip(((opened - as_days(month_start(0))).astype(np.int64) // 30), -1, N_MONTHS)
    # product counts per customer for the "multi-product" outflow story
    products_per_customer = np.bincount(a_cust, minlength=len(cust["customer_id"]))[a_cust]
    activity = np.exp(rng.normal(0.0, 0.4, size=n_acc))  # per-account activity factor
    scale_by_segment = credit_scale(p)
    credit_factor = np.array([scale_by_segment[s_] for s_ in seg])
    opened_s = opened.astype("datetime64[s]") + np.timedelta64(8 * 3600, "s")

    def stamp(rows: np.ndarray, mi: int, dw, r: np.random.Generator = rng) -> np.ndarray:  # type: ignore[no-untyped-def]
        """Timestamps inside the month, never before the account opened and never after the day it closed.

        In an account's closing month the drawn day is scaled onto the days up to the closing day, keeping the time of
        day, so nothing lands after the close and nothing piles up on it; no extra draw, so the stream is unchanged."""
        ts = random_timestamps(r, mi, len(rows), dw)
        closing = close_mi[rows] == mi
        if closing.any():
            base = np.datetime64(datetime.combine(month_start(mi), datetime.min.time()), "s")
            day, tod = np.divmod((ts[closing] - base).astype(np.int64), 86400)
            day = day * (close_day[rows[closing]] + 1) // days_in_month(mi)
            # the payout closes the closing day at 23:59:59: no other transaction of the month takes that second
            tod = np.minimum(tod, 86400 - 2)
            ts[closing] = base + (day * 86400 + tod).astype("timedelta64[s]")
        return np.asarray(np.maximum(ts, opened_s[rows]))

    # merchant lookup by category, weighted by popularity
    m_cat = np.array(merch["merchant_category"])
    m_id = np.array(merch["merchant_id"])
    m_pop = np.array(merch["popularity_weight"], dtype=np.float64)
    merchants_by_cat = {c: (m_id[m_cat == c], m_pop[m_cat == c] / m_pop[m_cat == c].sum()) for c in np.unique(m_cat)}
    # the counterparty of a merchant payment is the merchant's name, looked up by id (ids are in order, M-00001 up)
    m_name = np.array(merch["merchant_name"])
    counterparties = {
        c.category: np.array(category_names(fake, p.faker_locale, c.category, COUNTERPARTY_POOL, p.places))
        for c in p.txn_categories
        if not c.merchant_based
    }
    cp_weights = 1.0 / np.arange(1, COUNTERPARTY_POOL + 1) ** 0.8
    cp_weights /= cp_weights.sum()

    out_dir = ctx.out_dir / "transactions"
    out_dir.mkdir(parents=True, exist_ok=True)
    txn_counter = 0
    case_counter = 0
    fraud_rows: dict[str, list[Any]] = {
        k: []
        for k in (
            "case_id",
            "customer_id",
            "account_id",
            "first_txn_id",
            "opened_ts",
            "fraud_type",
            "amount",
            "recovered",
            "loss",
            "detected_by",
            "status",
            "closed_ts",
            "month",
            "channel",
        )
    }
    bal_rows: dict[str, list[Any]] = {
        k: []
        for k in (
            "account_id",
            "month",
            "month_index",
            "opening",
            "net_flow",
            "interest",
            "end",
            "rate",
            "active",
        )
    }
    window_end_s = np.datetime64(datetime.combine(month_end(N_MONTHS - 1), datetime.max.time()), "s")
    fraud_types = list(p.fraud_types)
    fraud_w = list(p.fraud_types.values())

    def fraud_cases(
        r: np.random.Generator, hit: np.ndarray, ftype: np.ndarray, mi: int, frames: list[pa.Table], debit: np.ndarray
    ) -> None:
        """The month's fraud cases on the hit accounts, drawn from r: their transactions, the debits (added to the
        balance array `debit`) and the case rows."""
        nonlocal txn_counter, case_counter
        k = len(hit)
        if not k:
            return
        n_txn = r.choice([1, 2, 3], size=k, p=[0.55, 0.3, 0.15])
        rows = np.repeat(hit, n_txn)
        total = len(rows)
        case_ids = np.array([f"F-{case_counter + i + 1:06d}" for i in range(k)])
        case_of_row = np.repeat(case_ids, n_txn)
        ftype_row = np.repeat(ftype, n_txn)
        amt = -np.round(lognormal_amount(r, p.fraud_loss_median / n_txn.repeat(n_txn), 0.8, total), 2)
        fraud_rail = np.array([RAIL_BY_FRAUD[ft][0 if p.country == "NL" else 1] for ft in ftype_row])
        ts = stamp(rows, mi, dw_plain, r)
        frames.append(
            to_table(
                {
                    "txn_id": np.arange(txn_counter, txn_counter + total, dtype=np.int64),
                    "account_id": account_id[rows],
                    "customer_id": customer_id[rows],
                    "ts": ts,
                    "amount": amt,
                    "rail": fraud_rail,
                    "category": np.full(total, "fraud"),
                    "merchant_id": np.full(total, ""),
                    "counterparty": np.full(total, "unknown"),
                    "is_fraud": np.ones(total, dtype=bool),
                    "fraud_case_id": case_of_row,
                }
            )
        )
        first_txn = np.arange(txn_counter, txn_counter + total)[np.cumsum(np.r_[0, n_txn[:-1]])]
        txn_counter += total
        np.add.at(debit, rows, amt)
        case_amount = -np.bincount(np.repeat(np.arange(k), n_txn), weights=amt, minlength=k)
        pp = positive_pay[hit] & np.isin(ftype, ["counterfeit_check", "check_washing", "ach_fraud"])
        detected = np.where(
            pp,
            "positive_pay",
            weighted_choice(r, ["customer", "rule", "model"], [0.5, 0.3, 0.2], k),
        )
        recovered_share = np.where(pp, 1.0, np.clip(r.beta(1.2, 3.0, size=k), 0, 1))
        recovered = np.round(case_amount * recovered_share, 2)
        # the as-of rule: a case is reported by the window end, so a report delay that would land
        # after it is shortened in proportion to the time left (no pile-up on the last day), and a case
        # still open at the window end has no closing time yet
        first_ts = ts[np.cumsum(np.r_[0, n_txn[:-1]])]
        delay = r.integers(0, REPORT_DELAY_S, size=k)
        room = (window_end_s - first_ts).astype(np.int64)
        delay = np.where(
            first_ts + delay.astype("timedelta64[s]") > window_end_s, delay * room // REPORT_DELAY_S, delay
        )
        opened_ts = first_ts + delay.astype("timedelta64[s]")
        closed_ts = opened_ts + r.integers(2 * 86400, 60 * 86400, size=k).astype("timedelta64[s]")
        status = np.where(closed_ts > window_end_s, "open", "closed")
        closed_ts = np.where(status == "open", np.datetime64("NaT", "s"), closed_ts)
        fraud_rows["case_id"] += list(case_ids)
        fraud_rows["customer_id"] += list(customer_id[hit])
        fraud_rows["account_id"] += list(account_id[hit])
        fraud_rows["first_txn_id"] += list(first_txn)
        fraud_rows["opened_ts"] += list(opened_ts)
        fraud_rows["fraud_type"] += list(ftype)
        fraud_rows["amount"] += list(np.round(case_amount, 2))
        fraud_rows["recovered"] += list(recovered)
        fraud_rows["loss"] += list(np.round(case_amount - recovered, 2))
        fraud_rows["detected_by"] += list(detected)
        fraud_rows["status"] += list(status)
        fraud_rows["closed_ts"] += list(closed_ts)
        fraud_rows["month"] += [month_label(mi)] * k
        fraud_rows["channel"] += list(fraud_rail[np.cumsum(np.r_[0, n_txn[:-1]])])
        case_counter += k

    # an event with its own stream draws from it alone; the others share the ledger's stream. Its debits
    # are kept apart from `balance` until the balances are written, so the overdraft relief the ledger's stream
    # draws from `balance` is the same with or without the event: the account still shows the loss
    ledger_events = [fev for fev in p.fraud_events if fev.stream is None]
    own_stream_events = [
        (fev, ctx.rng("transactions", "fraud_event", fev.stream)) for fev in p.fraud_events if fev.stream
    ]
    own_stream_taken = np.zeros(n_acc)

    for mi in range(N_MONTHS):
        ms, me = as_days(month_start(mi)), as_days(month_end(mi))
        active = (opened <= me) & (closed >= ms)
        season = seasonal_multiplier(mi)
        dw_plain = day_weights(mi)
        dw_pay = day_weights(mi, peak_days=p.salary_days, peak=25.0)
        frames: list[pa.Table] = []
        month_opening = balance.copy()
        month_taken = own_stream_taken.copy()
        month_interest = np.zeros(n_acc)
        rate_now = np.array([p.rate_paths[k][mi] if k else 0.0 for k in rate_key])

        # ---- spending ledger on transactional accounts ----
        ta = np.flatnonzero(active & transactional)
        for cat in p.txn_categories:
            lam = np.array([cat.monthly_rate.get(s, 0.0) for s in seg[ta]]) * activity[ta]
            if cat.merchant_based:
                lam = lam * season
            counts = rng.poisson(lam)
            total = int(counts.sum())
            if total == 0:
                continue
            rows = np.repeat(ta, counts)
            if cat.income_based:
                monthly_rate = np.array([cat.monthly_rate.get(s, 1.0) for s in seg[rows]])
                amt = (
                    NET_INCOME_FACTOR
                    * income[rows]
                    / 12.0
                    / np.maximum(monthly_rate, 0.25)
                    * np.exp(rng.normal(0, 0.08, size=total))
                )
            else:
                inc_f = (income[rows] / np.array([p.income_median[s] for s in seg[rows]])) ** 0.3
                amt = lognormal_amount(rng, cat.amount_median * inc_f, cat.amount_sigma, total)
            if cat.direction > 0:
                amt = amt * credit_factor[rows]
            amt = np.round(amt, 2) * cat.direction
            ts = stamp(rows, mi, dw_pay if cat.payday else dw_plain)
            if cat.merchant_based and cat.category in merchants_by_cat:
                ids, w = merchants_by_cat[cat.category]
                merchant = rng.choice(ids, size=total, p=w)
                counterparty = m_name[np.searchsorted(m_id, merchant)]
            else:
                merchant = np.full(total, "")
                counterparty = rng.choice(counterparties[cat.category], size=total, p=cp_weights)
            frames.append(
                to_table(
                    {
                        "txn_id": np.arange(txn_counter, txn_counter + total, dtype=np.int64),
                        "account_id": account_id[rows],
                        "customer_id": customer_id[rows],
                        "ts": ts,
                        "amount": amt,
                        "rail": np.full(total, cat.rail),
                        "category": np.full(total, cat.category),
                        "merchant_id": merchant,
                        "counterparty": counterparty,
                        "is_fraud": np.zeros(total, dtype=bool),
                        "fraud_case_id": np.full(total, ""),
                    }
                )
            )
            txn_counter += total
            np.add.at(balance, rows, amt)

        # overdraft relief: consumers pull from savings when the account dips below zero
        neg = ta[(balance[ta] < -200) & ~np.isin(seg[ta], ["business", "small_business", "commercial"])]
        neg = neg[bernoulli(rng, 0.7, len(neg))]
        if len(neg):
            amt = np.round(-balance[neg] + lognormal_amount(rng, 150.0, 0.6, len(neg)), 2)
            frames.append(
                to_table(
                    {
                        "txn_id": np.arange(txn_counter, txn_counter + len(neg), dtype=np.int64),
                        "account_id": account_id[neg],
                        "customer_id": customer_id[neg],
                        "ts": stamp(neg, mi, dw_plain),
                        "amount": amt,
                        "rail": np.full(len(neg), "transfer" if p.country == "NL" else "internal_transfer"),
                        "category": np.full(len(neg), "transfer_from_savings"),
                        "merchant_id": np.full(len(neg), ""),
                        "counterparty": np.full(len(neg), "own savings"),
                        "is_fraud": np.zeros(len(neg), dtype=bool),
                        "fraud_case_id": np.full(len(neg), ""),
                    }
                )
            )
            txn_counter += len(neg)
            np.add.at(balance, neg, amt)

        # ---- savings-type accounts: deposits, withdrawals, interest, story outflows ----
        sa = np.flatnonzero(active & ~transactional)
        if len(sa):
            dep = sa[bernoulli(rng, 0.24, len(sa))]
            wd = sa[bernoulli(rng, 0.18, len(sa))]
            dep_amt = np.round(lognormal_amount(rng, np.maximum(month_opening[dep] * 0.03, 50), 0.9, len(dep)), 2)
            wd_amt = -np.round(
                np.minimum(
                    lognormal_amount(rng, np.maximum(month_opening[wd] * 0.05, 50), 0.9, len(wd)),
                    np.maximum(balance[wd], 0) * 0.9,
                ),
                2,
            )
            # story: outflow events (rate cut, CD repricing)
            ev_idx: list[np.ndarray] = []
            ev_amt: list[np.ndarray] = []
            for ev in p.deposit_outflow_events:
                if not (ev.month_from <= mi <= ev.month_to):
                    continue
                m = (product[sa] == ev.product) & np.isin(seg[sa], ev.segments)
                if ev.multi_product_only:
                    m &= products_per_customer[sa] >= 2
                if ev.product == "cd":
                    m &= (maturity[sa] >= ms) & (maturity[sa] <= me)  # only CDs maturing this month
                cand = sa[m]
                cand = cand[
                    bernoulli(
                        rng,
                        ev.share_of_accounts / (1 if ev.product == "cd" else (ev.month_to - ev.month_from + 1)),
                        len(cand),
                    )
                ]
                if len(cand):
                    ev_idx.append(cand)
                    ev_amt.append(
                        -np.round(
                            np.maximum(balance[cand], 0) * ev.fraction_moved * np.exp(rng.normal(0, 0.1, len(cand))),
                            2,
                        )
                    )
            interest = np.round(np.maximum(balance[sa], 0) * rate_now[sa] / 100 / 12, 2)
            month_interest[sa] = interest
            groups = [
                (dep, dep_amt, "deposit", "transfer"),
                (wd, wd_amt, "withdrawal", "transfer"),
                (sa, interest, "interest", "interest"),
            ]
            for i_arr, a_arr in zip(ev_idx, ev_amt, strict=True):
                groups.append((i_arr, a_arr, "outflow_to_competitor", "transfer"))
            for i_arr, a_arr, catname, rail in groups:
                k = len(i_arr)
                if k == 0:
                    continue
                rail_name = rail if p.country == "NL" else ("internal_transfer" if rail == "transfer" else rail)
                frames.append(
                    to_table(
                        {
                            "txn_id": np.arange(txn_counter, txn_counter + k, dtype=np.int64),
                            "account_id": account_id[i_arr],
                            "customer_id": customer_id[i_arr],
                            "ts": stamp(i_arr, mi, dw_plain)
                            if catname != "interest"
                            # interest at midnight on the month's last day, or on the closing day of an account
                            # that closes this month
                            else np.where(
                                close_mi[i_arr] == mi,
                                close_midnight[i_arr],
                                np.datetime64(datetime.combine(month_end(mi), datetime.min.time()), "s"),
                            ),
                            "amount": a_arr,
                            "rail": np.full(k, rail_name),
                            "category": np.full(k, catname),
                            "merchant_id": np.full(k, ""),
                            "counterparty": np.full(
                                k,
                                p.bank_name
                                if catname == "interest"
                                else "own account"
                                if catname != "outflow_to_competitor"
                                else rng.choice(COMPETITORS[p.country], size=k),
                            ),
                            "is_fraud": np.zeros(k, dtype=bool),
                            "fraud_case_id": np.full(k, ""),
                        }
                    )
                )
                txn_counter += k
                np.add.at(balance, i_arr, a_arr)

        # ---- fraud cases on transactional accounts ----
        if len(ta):
            hit_parts = [ta[bernoulli(rng, p.fraud_base_rate, len(ta))]]
            type_parts = [weighted_choice(rng, fraud_types, fraud_w, len(hit_parts[0]))]
            for fev in ledger_events:
                mult = _fraud_multiplier(fev, mi, product[ta], opened_mi[ta], seg[ta], positive_pay[ta])
                extra = ta[bernoulli(rng, p.fraud_base_rate * (mult - 1.0), len(ta))]
                hit_parts.append(extra)
                type_parts.append(rng.choice(np.array(fev.types), size=len(extra)))
            fraud_cases(rng, np.concatenate(hit_parts), np.concatenate(type_parts).astype("U32"), mi, frames, balance)
            for fev, ev_rng in own_stream_events:
                if not (fev.month_from <= mi <= fev.month_to):
                    continue
                mult = _fraud_multiplier(fev, mi, product[ta], opened_mi[ta], seg[ta], positive_pay[ta])
                extra = ta[bernoulli(ev_rng, p.fraud_base_rate * (mult - 1.0), len(ta))]
                ftype = ev_rng.choice(np.array(fev.types), size=len(extra)).astype("U32")
                fraud_cases(ev_rng, extra, ftype, mi, frames, own_stream_taken)

        # ---- the accounts that close this month: the balance leaves at the end of the closing day ----
        # (no draw from any stream, so every other row of the ledger is the same as without it; the own-stream fraud
        # debits are part of the balance the customer takes; the account has no row after this month)
        closing = np.flatnonzero(active & (close_mi == mi))
        closing_out = np.zeros(n_acc)
        closing_out[closing] = -np.round(balance[closing] + own_stream_taken[closing], 2)
        shut = closing[closing_out[closing] != 0]
        if len(shut):
            frames.append(
                to_table(
                    {
                        "txn_id": np.arange(txn_counter, txn_counter + len(shut), dtype=np.int64),
                        "account_id": account_id[shut],
                        "customer_id": customer_id[shut],
                        "ts": close_end_of_day[shut],
                        "amount": closing_out[shut],
                        "rail": np.full(len(shut), "transfer" if p.country == "NL" else "internal_transfer"),
                        "category": np.where(closing_out[shut] < 0, "withdrawal", "deposit"),
                        "merchant_id": np.full(len(shut), ""),
                        "counterparty": np.full(len(shut), CLOSING_COUNTERPARTY),
                        "is_fraud": np.zeros(len(shut), dtype=bool),
                        "fraud_case_id": np.full(len(shut), ""),
                    }
                )
            )
            txn_counter += len(shut)
        # the balance shown is exactly zero (the balance plus the own-stream debits kept apart), not a rounding residue
        balance[closing] = -own_stream_taken[closing]

        # ---- write the month ----
        if frames:
            month_table = pa.concat_tables(frames)
            month_table = month_table.append_column("currency", pa.array([p.currency] * month_table.num_rows))
            month_table = month_table.append_column("month", pa.array([month_label(mi)] * month_table.num_rows))
            pq.write_table(month_table, out_dir / f"part-{month_label(mi)}.parquet", compression="zstd")
        # ---- monthly balances from the same ledger ----
        a = np.flatnonzero(active)
        bal_rows["account_id"] += list(account_id[a])
        bal_rows["month"] += [month_label(mi)] * len(a)
        bal_rows["month_index"] += [mi] * len(a)
        opening_shown = month_opening[a] + month_taken[a]
        end_shown = balance[a] + own_stream_taken[a]
        bal_rows["opening"] += list(np.round(opening_shown, 2))
        bal_rows["net_flow"] += list(np.round(end_shown - opening_shown - month_interest[a] - closing_out[a], 2))
        bal_rows["interest"] += list(np.round(month_interest[a], 2))
        bal_rows["end"] += list(np.round(end_shown, 2))
        bal_rows["rate"] += list(rate_now[a])
        bal_rows["active"] += [True] * len(a)

    ctx.tables["account_balances_monthly"] = to_table(
        {
            "account_id": np.array(bal_rows["account_id"]),
            "month": np.array(bal_rows["month"]),
            "month_index": np.array(bal_rows["month_index"], dtype=np.int64),
            "opening_balance": np.array(bal_rows["opening"], dtype=np.float64),
            "net_flow": np.array(bal_rows["net_flow"], dtype=np.float64),
            "interest_paid": np.array(bal_rows["interest"], dtype=np.float64),
            "end_balance": np.array(bal_rows["end"], dtype=np.float64),
            "rate_pct": np.array(bal_rows["rate"], dtype=np.float64),
        }
    )
    ctx.tables["fraud_cases"] = to_table(
        {
            "case_id": np.array(fraud_rows["case_id"]),
            "customer_id": np.array(fraud_rows["customer_id"]),
            "account_id": np.array(fraud_rows["account_id"]),
            "first_txn_id": np.array(fraud_rows["first_txn_id"], dtype=np.int64),
            "opened_ts": np.array(fraud_rows["opened_ts"], dtype="datetime64[s]"),
            "month": np.array(fraud_rows["month"]),
            "fraud_type": np.array(fraud_rows["fraud_type"]),
            "channel": np.array(fraud_rows["channel"]),
            "reported_month": np.array([str(t)[:7] for t in np.array(fraud_rows["opened_ts"], dtype="datetime64[s]")]),
            "amount": np.array(fraud_rows["amount"], dtype=np.float64),
            "recovered_amount": np.array(fraud_rows["recovered"], dtype=np.float64),
            "loss_amount": np.array(fraud_rows["loss"], dtype=np.float64),
            "detected_by": np.array(fraud_rows["detected_by"]),
            "status": np.array(fraud_rows["status"]),
            "closed_ts": np.array(fraud_rows["closed_ts"], dtype="datetime64[s]"),
            "currency": np.array([p.currency] * len(fraud_rows["case_id"])),
        }
    )
