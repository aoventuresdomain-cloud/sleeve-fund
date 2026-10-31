"""TRIAL-CORRECT (Advisor, B0's ruling 31 Oct; the DA's conditions): one correcting row for a basket try whose register
Sharpe was the close-priced comparison's. Dry run by default; refused unless the row and the sheet are one try's; the
variant, N, the round's tries and no_selection never move, and sharpes() reads the book's Sharpe after."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from sleeve_fund.research import rounds, trial_correction as tc, trials
from sleeve_fund.research.study import idea_of
from sleeve_fund.store import Store

LABEL, DATASET = "qrb0_range_reversion_adx", "binance-basket-btc-eth-sol-usdt-store-60m"
BOOK = -1.6689877


def _register(tmp_path):
    reg = trials.TrialsRegister(Store(f"sqlite:///{tmp_path}/t.db"))
    from sleeve_fund.research import rescore

    params = rescore.request("qr01_trend_ema_long", 240, "binance", "BTC/USDT").params
    start, end = datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2026, 5, 1, tzinfo=timezone.utc)
    rid = reg.record(definition_hash="k1", idea_hash=trials.legacy_idea_hash("rules"), name=f"basket:{LABEL}",
                     family="rules", settings={"basket": ["BTC", "ETH", "SOL"], "params": params, "parent": "p"},
                     dataset=DATASET, stage="in_sample", source="study", sharpe=1.80, data_start=start, data_end=end)
    return reg, rid, idea_of(params, None)


def _sheet(tmp_path, ledger=True, after=timedelta(minutes=30), hold="| Hold bar (long-only baskets) | pooled Sharpe "
           "1.21 (net +409.7%, CAGR +72.1%) against ...|"):
    stem = f"{LABEL}_{DATASET}_{datetime.now(timezone.utc) + after:%Y%m%d-%H%M%S}"
    (tmp_path / f"{stem}.md").write_text(f"# Basket study\n\n{hold}\n", encoding="utf-8")
    if ledger:
        (tmp_path / f"{stem}.results.json").write_text(json.dumps({"check": {"basket": {"sharpe": BOOK}}}))
    return stem


def test_a_dry_run_writes_nothing_and_apply_adds_one_row_of_the_same_variant(tmp_path, capfd):
    reg, rid, family_idea = _register(tmp_path)
    stem = _sheet(tmp_path)
    idea = trials.legacy_idea_hash("rules")
    before = (reg.counts(idea), reg.no_selection(idea), rounds.used(reg, {family_idea}))
    assert tc.main(["--row", rid, "--sheet", stem], store=reg.store, sheets=tmp_path) == 0
    assert "dry run: nothing written" in capfd.readouterr().out and len(reg._counted()) == 1
    assert tc.main(["--row", rid, "--sheet", stem, "--apply"], store=reg.store, sheets=tmp_path) == 0
    out = capfd.readouterr().out
    assert "CHECK FAILED" not in out and f"Sharpe 1.8 -> {BOOK} (the sheet's daily ledger (results.json))" in out
    after = (reg.counts(idea), reg.no_selection(idea), rounds.used(reg, {family_idea}))
    assert after[0]["variants"] == before[0]["variants"] == 1 and after[0]["evaluations"] == 2
    assert after[1] == before[1] and after[2] == before[2] == 1  # no_selection and the round's tries don't move
    assert reg.sharpes(idea) == [pytest.approx(BOOK)]  # the variant's latest row: the book's Sharpe
    original = next(t for t in reg._counted() if t["id"] == rid)
    assert original["sharpe"] == pytest.approx(1.80)  # write-once: the wrong row stays, read past
    # a second correction of the same try is refused: two rows have the key now
    assert tc.main(["--row", rid, "--sheet", stem, "--apply"], store=reg.store, sheets=tmp_path) == 1
    assert "2 counted rows have this basket key" in capfd.readouterr().out


def test_without_a_ledger_the_sheets_printed_book_sharpe_is_used_and_said(tmp_path):
    reg, rid, _ = _register(tmp_path)
    original, fixed, source = tc.plan(reg, rid, tmp_path / f"{_sheet(tmp_path, ledger=False)}.md")
    assert fixed["sharpe"] == 1.21 and source == tc.SHEET_ONLY
    assert trials._variant(fixed) == trials._variant(original) and fixed["id"] != original["id"]
    assert fixed["code_version"] == original["code_version"]  # the original's, not today's code


@pytest.mark.parametrize("kw, words", [
    (dict(after=timedelta(hours=6)), "another run's sheet"),  # a later re-run of the same basket
    (dict(after=-timedelta(minutes=5)), "another run's sheet"),
    (dict(ledger=False, hold="no hold line"), "0 hold-bar Sharpe lines"),
])
def test_a_sheet_that_isnt_this_trys_is_refused(tmp_path, kw, words):
    reg, rid, _ = _register(tmp_path)
    with pytest.raises(ValueError, match=words):
        tc.plan(reg, rid, tmp_path / f"{_sheet(tmp_path, **kw)}.md")


def test_another_basket_or_a_non_basket_row_is_refused(tmp_path, capfd):
    reg, rid, _ = _register(tmp_path)
    other = f"other_{DATASET}_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    (tmp_path / f"{other}.md").write_text("x")
    with pytest.raises(ValueError, match="is not qrb0_range_reversion_adx on"):
        tc.plan(reg, rid, tmp_path / f"{other}.md")
    single = reg.record(definition_hash="s1", idea_hash="i", name="qr01", family="trend", settings={}, dataset="d",
                        stage="in_sample", source="study", sharpe=0.5)
    with pytest.raises(ValueError, match="not a basket try"):
        tc.plan(reg, single, tmp_path / f"{other}.md")
    assert tc.main(["--row", "not-an-id", "--sheet", other], store=reg.store, sheets=tmp_path) == 2


def test_an_already_correct_row_is_refused(tmp_path):
    reg, rid, _ = _register(tmp_path)
    stem = _sheet(tmp_path)
    (tmp_path / f"{stem}.results.json").write_text(json.dumps({"check": {"basket": {"sharpe": 1.80}}}))
    with pytest.raises(ValueError, match="already 1.8"):
        tc.plan(reg, rid, tmp_path / f"{stem}.md")
