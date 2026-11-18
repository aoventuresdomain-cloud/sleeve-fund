"""The study-speed gate (Advisor 17 Nov, #365): H0 re-run once on the 4-rung study, on the round's pinned dates,
unguarded at full size, recorded as no try, with the holdout sealed, and asked for exactly as H0's counted row was
(next-round-pinned-dates-list.md row 3) but for the size scale that marks it a confirming run."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from sleeve_fund.research import rescore, rounds
from sleeve_fund.research.pinned import parse
from sleeve_fund.strategies.definitions import definition_hash, load_definition_file

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "h0-fast-gate-list.md"
ROUND = ROOT / "configs" / "studies" / "next-round-pinned-dates-list.md"
H0 = ROOT / "configs" / "definitions" / "variants" / "qrh0_trend_pullback.toml"
PAIRS = "BTC/USDT,ETH/USDT,SOL/USDT"


def test_the_list_reads_the_rounds_pinned_dates():
    p, r = parse(LIST.read_text()), parse(ROUND.read_text())
    assert (p.data_start, p.research_end, p.holdout_start, p.holdout_end) == (
        date(2024, 1, 1), date(2026, 4, 30), date(2026, 5, 1), date(2026, 10, 30))
    assert p.folds == r.folds and rescore.fold_days(p) == (366, 485, 183)


def test_it_is_h0_full_size_unguarded_never_a_try_and_otherwise_its_counted_rows_request():
    text = LIST.read_text()
    assert rescore.rows_of(text) == [(1, "qrh0_trend_pullback", 240)]
    assert definition_hash(load_definition_file(H0)) == "2161c99b4cf29d08"
    confirming = rescore.confirming_of(text)
    assert confirming == {"fund_guards": False, "size_scale": 1.0}  # a size scale: basket_run records no try
    assert rounds.round_of(text, [1]) is None and rescore.risk_of(text) == {}
    req = rescore.request("qrh0_trend_pullback", 240, "binance", PAIRS, parse(text), confirming=confirming)
    counted = rescore.request("qrh0_trend_pullback", 240, "binance", PAIRS, parse(ROUND.read_text()),
                              confirming=rescore.confirming_of(ROUND.read_text()))
    assert (req.use_holdout, req.fund_guards, req.size_scale, counted.size_scale) == (False, False, 1.0, None)
    assert req.params == counted.params and req.pairs == counted.pairs and req.pinned == counted.pinned
    req.validate()


def test_the_list_dry_runs_as_not_new_tries(capfd):
    assert rescore.main([str(LIST), "--pair", PAIRS, "--dry-run"], run=lambda *a, **k: None,
                        journal=lambda: None) == 0
    out = capfd.readouterr().out
    assert "not new tries: fund guards OFF (research only), size scale 1" in out and "dry run: 1 checked, 0 run" in out
