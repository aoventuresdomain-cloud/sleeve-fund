"""H0 check 6a (prereg-family-h-trend-pullback-31oct.md, Advisor 31 Oct 05:57 UTC): the V5 side of the correlation is
one frozen-V5 basket on the round's pinned dates, unguarded at full size, recorded as no try, with the holdout locked."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from sleeve_fund.research import rescore, rounds
from sleeve_fund.research.pinned import parse
from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import definition_hash, load_definition_file

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "h0-v5-correlation-series-list.md"
ROUND = ROOT / "configs" / "studies" / "next-round-pinned-dates-list.md"


def test_the_list_reads_the_rounds_pinned_dates():
    text = LIST.read_text()
    p, r = parse(text), parse(ROUND.read_text())
    assert (p.data_start, p.research_end, p.holdout_start, p.holdout_end) == (
        date(2024, 1, 1), date(2026, 4, 30), date(2026, 5, 1), date(2026, 10, 30))
    assert p.folds == r.folds and rescore.fold_days(p) == (366, 485, 183)


def test_it_is_frozen_v5_full_size_unguarded_and_never_a_try():
    text = LIST.read_text()
    assert rescore.rows_of(text) == [(1, "v5_lo_adx", 240)]
    assert definition_hash(load_definition_file(definition_path(PREREGISTERED["V5_LO_ADX"]["definition"]))) == (
        "5e2fd5a2055facf4")
    confirming = rescore.confirming_of(text)
    assert confirming == {"fund_guards": False, "size_scale": 1.0}  # a size scale: basket_run records no try
    assert rounds.round_of(text, [1]) is None  # no round, so no budget is spent and no try line is written
    req = rescore.request("v5_lo_adx", 240, "binance", "BTC/USDT,ETH/USDT,SOL/USDT", parse(text),
                          confirming=confirming)
    assert tuple(req.pairs) == ("BTC/USDT", "ETH/USDT", "SOL/USDT") and req.params["market"] == "perp"
    assert "allow_short" not in req.params  # V5 is long only
    assert (req.use_holdout, req.fund_guards, req.size_scale) == (False, False, 1.0)
    req.validate()


def test_the_list_dry_runs_as_not_new_tries(capfd):
    assert rescore.main([str(LIST), "--pair", "BTC/USDT,ETH/USDT,SOL/USDT", "--dry-run"], run=lambda *a, **k: None,
                        journal=lambda: None) == 0
    out = capfd.readouterr().out
    assert "not new tries: fund guards OFF (research only), size scale 1" in out and "dry run: 1 checked, 0 run" in out
