"""The fraud-ring overlay's rules on a small book shaped like Cauchy Bank's: 39 wave cases, 10 other cases in
the wave's months, 57 outside them, one to three payments each; no network, no Databricks."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import pytest

import build_overlay as bo

pytestmark = pytest.mark.unit


def book() -> tuple[list[Any], list[Any], list[Any]]:
    """A made-up book: the cases, their fraudulent payments, and 40 candidate payers."""
    shapes = (
        [("2026-05" if i % 2 else "2026-06", "phishing" if i % 3 else "bank_helpdesk_scam") for i in range(39)]
        + [("2026-05", "account_takeover")] * 10
        + [(f"2025-{1 + i % 12:02d}", "card_skimming") for i in range(57)]
    )
    cases, txns = [], []
    for i, (month, fraud_type) in enumerate(shapes):
        case_id = f"F-{i + 1:06d}"
        cases.append(bo.FraudCase(case_id, f"C-{i + 1:06d}", f"A-{i + 1:06d}", month, fraud_type))
        start = datetime(int(month[:4]), int(month[5:]), 3)
        for j in range(1 + i % 3):
            txns.append(bo.FraudTxn(1000 * (i + 1) + j, case_id, start + timedelta(days=j), -250.0 * (j + 1)))
    candidates = [bo.Candidate(f"C-{500 + k:06d}", f"A-{500 + k:06d}") for k in range(40)]
    return cases, txns, candidates


def test_the_same_seed_gives_the_same_rows_and_another_seed_others() -> None:
    cases, txns, candidates = book()
    first = bo.build_overlay(cases, txns, candidates, 20261105)
    assert bo.build_overlay(cases, txns, candidates, 20261105) == first
    assert bo.build_overlay(list(reversed(cases)), list(reversed(txns)), candidates, 20261105) == first
    assert bo.build_overlay(cases, txns, candidates, 7) != first


@pytest.mark.parametrize("seed", range(40))
def test_every_rule_holds_whatever_the_seed(seed: int) -> None:
    cases, txns, candidates = book()
    overlay = bo.build_overlay(cases, txns, candidates, seed)
    assert bo.check_overlay(overlay, cases, txns) == []


def test_the_ring_takes_31_of_the_39_wave_cases_and_every_other_case_has_a_lone_mule() -> None:
    cases, txns, candidates = book()
    counts = bo.summary(bo.build_overlay(cases, txns, candidates, 20261105), cases)
    assert counts["wave cases"] == 39
    assert counts["ring cases"] == 31
    assert 5 <= counts["two-account victims"] <= 8
    assert counts["lone mules"] == len(cases) - 31
    assert counts["beneficiaries"] == len(txns)
    assert 5 <= counts["unreported payers"] <= 8


def test_the_checks_catch_a_broken_overlay() -> None:
    cases, txns, candidates = book()
    overlay = bo.build_overlay(cases, txns, candidates, 20261105)
    ring = next(m.mule_account_id for m in overlay.mule_accounts if m.ring_id == "RING-01")
    outside = next(c.case_id for c in cases if c.month.startswith("2025"))
    moved = tuple(replace(b, mule_account_id=ring) if b.case_id == outside else b for b in overlay.beneficiaries)
    failures = bo.check_overlay(replace(overlay, beneficiaries=moved), cases, txns)
    assert "a case outside the wave pays the ring" in failures
    dropped = bo.check_overlay(replace(overlay, beneficiaries=overlay.beneficiaries[1:]), cases, txns)
    assert "the beneficiaries do not cover every fraudulent payment exactly once" in dropped


def test_an_iban_has_valid_check_digits_at_a_fictitious_bank() -> None:
    iban = bo.nl_iban("EXTA", "0123456789")
    assert iban.startswith("NL") and iban[4:8] == "EXTA" and len(iban) == 18
    rearranged = iban[4:] + iban[:4]
    assert int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1
