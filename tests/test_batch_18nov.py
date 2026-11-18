"""Idea batch S2 and S3 (Advisor ruled 18 Nov after the 2024 signal count): the counted list runs the two cleared rows
as family B tries 28 and 29, each sized to 40% a year of 30-day volatility, and every basket sheet that sizes to a
volatility target prints the target and each asset's median size and capped share on the test-window entries.
Figures in fixtures are synthetic."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from sleeve_fund.research import basket as bk
from sleeve_fund.research import rescore, rounds
from sleeve_fund.research.pinned import parse
from sleeve_fund.research.variants import BATCH_VOL
from sleeve_fund.strategies.rules import VOL_SIZE_LIMIT

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "batch-18nov-list.md"
ROWS = ["qrs2_london_asia_range_fade", "qrs3_shock_fade_4h"]
PAIRS = "BTC/USDT,ETH/USDT,SOL/USDT"


def test_the_list_is_the_two_cleared_rows_as_family_b_tries_28_and_29():
    text = LIST.read_text(encoding="utf-8")
    rows = rescore.rows_of(text)
    assert rows == [(1, ROWS[0], 240), (2, ROWS[1], 240)]
    r = rounds.round_of(text, [n for n, _, _ in rows])
    assert r.name == "batch-18nov" and r.families == {"B": (1, 2)} and r.prior == {"B": 27}
    assert "family B try 28 " in rounds.try_line(r, 1, 0) and "family B try 29 " in rounds.try_line(r, 2, 1)
    assert "registered try 1 of 2" in rounds.try_line(r, 1, 0)


def test_it_is_a_counted_run_on_the_rounds_pinned_dates():
    text = LIST.read_text(encoding="utf-8")
    pinned = parse(text)
    assert pinned == parse((ROOT / "configs" / "studies" / "next-round-pinned-dates-list.md").read_text())
    counted = (ROOT / "configs" / "studies" / "idea-batch-17nov-list.md").read_text(encoding="utf-8")
    assert rescore.confirming_of(text) == rescore.confirming_of(counted)  # guards off, no size scale: a try
    assert "Size scale" not in text
    for _, name, minutes in rescore.rows_of(text):
        req = rescore.request(name, minutes, "binance", PAIRS, pinned)
        assert req.params["vol_target_size"] == BATCH_VOL == {"annual": 0.40, "lookback_days": 30}
        assert req.params["market"] == "perp" and tuple(req.pairs) == tuple(PAIRS.split(","))
        assert (req.use_holdout, req.train_days, req.test_days, req.holdout_days) == (False, 366, 485, 183)
        assert req.risk_profile is None
        req.validate()


def test_the_list_dry_runs(capfd):
    assert rescore.main([str(LIST), "--pair", PAIRS, "--dry-run"], run=lambda *a, **k: None,
                        journal=lambda: None) == 0
    assert "dry run: 2 checked, 0 run" in capfd.readouterr().out


WINDOWS = ((date(2025, 1, 1), date(2025, 6, 30)),)


def _entry(close: str, factor: float | None, cut: bool) -> dict:
    sig = {"bar": {"close_ts": close}, "sized_by": VOL_SIZE_LIMIT if cut else "risk profile cap"}
    if factor is not None:
        sig["vol_factor"] = factor
    return {"intent": "entry", "signal": sig}


def _leg(entries: list[dict], skipped: int = 0) -> SimpleNamespace:
    decisions = {f"O-20250101-00000{i}-X": d for i, d in enumerate(entries)}
    decisions["O-exit"] = {"intent": "exit", "signal": {"rule": "long.exit"}}
    return SimpleNamespace(decisions=decisions, rule_counts={"vol_skipped": skipped} if skipped else {})


def _run() -> SimpleNamespace:
    return SimpleNamespace(legs={
        "BTC-USDT": _leg([_entry("2024-12-31T20:00:00+00:00", 0.2, True),  # the train year: left out
                          _entry("2025-01-01T00:00:00+00:00", 0.9, True),  # closes 00:00 1 Jan: 31 Dec's, left out
                          _entry("2025-02-01T04:00:00+00:00", 0.5, True),
                          _entry("2025-03-01T04:00:00+00:00", 0.7, True),
                          _entry("2025-04-01T04:00:00+00:00", None, False)], skipped=3),
        "ETH-USDT": _leg([_entry("2025-05-01T04:00:00+00:00", 1.0, False)]),
        "SOL-USDT": _leg([_entry("2025-07-01T04:00:00+00:00", 0.4, True)])})  # after the window: left out


def test_sizes_are_read_from_the_test_window_entries_only():
    by = bk.vol_sizes(_run(), WINDOWS, {"BTC-USDT": "BTC", "ETH-USDT": "ETH", "SOL-USDT": "SOL"})
    assert by["BTC"] == bk.VolSizes((0.5, 0.7, 1.0), 1, 3)
    assert by["ETH"] == bk.VolSizes((1.0,), 1, 0)
    assert by["SOL"] == bk.VolSizes((), 0, 0)


def _study(params: dict) -> SimpleNamespace:
    legs = {"BTC-USDT": ("BTC", 240), "ETH-USDT": ("ETH", 240), "SOL-USDT": ("SOL", 240)}
    return SimpleNamespace(basket=SimpleNamespace(params=params), base=SimpleNamespace(run=_run()), windows=WINDOWS,
                           legs=legs)


def test_the_sheet_prints_the_target_and_each_assets_median_size_and_capped_share():
    [line, blank] = bk._vol_lines(_study({"vol_target_size": BATCH_VOL}))
    assert blank == ""
    assert line.startswith("**Volatility target:** entries sized to 40% a year of volatility over 30 days")
    assert "BTC 3 entries, median size 0.70x, capped 33%" in line
    assert "ETH 1 entries, median size 1.00x, capped 100%" in line
    assert "SOL no entries" in line
    assert "Over the whole run, warm-up and train days included, 3 entries were skipped" in line


def test_a_basket_without_a_volatility_target_prints_no_line():
    assert bk._vol_lines(_study({})) == []


@pytest.mark.parametrize("screened", [False, True], ids=["full sheet", "screened sheet"])
def test_a_real_basket_sized_to_the_target_prints_the_line_on_its_sheet(tmp_path, monkeypatch, screened):
    """End to end on a synthetic basket: the engine's own entries carry their sizes, and the sheet reads them, on the
    full sheet and on a fast study's screened one (CR F1 on #371: a row screened out before fees shows its sizing only
    there). Screened here for too few trades, under a trade floor raised past the fixture's."""
    from test_b38_basket_study import CAPITAL, LOOSE, RESEARCH_END, _asset, _register
    from test_b38_basket_study import WINDOWS as SYN

    from sleeve_fund.strategies.definitions import to_params

    trend = {"version": 1, "reason": "a fixture", "blocks": {"avg": {"kind": "sma", "period": 3}},
             "long": {"entry": {"left": "close", "op": "crosses_above", "right": "avg"},
                      "exit": {"left": "close", "op": "<", "right": "avg"}}}
    b = bk.Basket("v", "rules", {**to_params(trend), "vol_target_size": BATCH_VOL},
                  (_asset("BTC", 2, 0.03), _asset("ETH", 5, 0.03), _asset("SOL", 9, 0.03)), "2026-10-20")
    if screened:
        monkeypatch.setattr(bk, "MIN_TRADES", 10_000)
    r = bk.basket_study(b, SYN, RESEARCH_END, register=_register(tmp_path), dataset="syn", capital=CAPITAL,
                        profile=LOOSE, warm_days=40, fast=screened)
    assert bool(r.screened) is screened
    by = bk.vol_sizes(r.base.run, r.windows, {leg: v[0] for leg, v in r.legs.items()})
    assert set(by) == {"BTC", "ETH", "SOL"} and all(v.sizes for v in by.values())
    assert any(s < 1 for v in by.values() for s in v.sizes)  # volatile fixture: the target cuts entries
    sheet = bk.render(r)
    assert "**Volatility target:** entries sized to 40% a year of volatility over 30 days" in sheet
    for a in ("BTC", "ETH", "SOL"):
        assert f"{a} {len(by[a].sizes)} entries, median size " in sheet
