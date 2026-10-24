"""Improve-loop round 0 (research/quant-researcher/improve-loop-plan-24oct.md; PM 24 Oct: the top 3, long and short in
one Strategy): qr23 and qr21 as long and short, each long leg its parent's exactly and the short leg its mirror, and
the BTC study list on round 1's dates with the holdout locked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pytest

from sleeve_fund.research import rescore
from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import check_definition, definition_hash, load_definition_file, to_params

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "qr-improve-r0-list.md"


def _mirror_stop(stop: dict) -> dict:
    (op, args), = stop["level"].items()
    return {"level": {{"sub": "add", "add": "sub"}[op]: args}}


@pytest.mark.parametrize("key", ["QR23_LS", "QR21_LS"])
def test_each_long_leg_is_its_parent_and_the_short_leg_its_mirror(key):
    v = PREREGISTERED[key]
    mine = load_definition_file(definition_path(v["definition"]))
    parent = load_definition_file(definition_path(v["parent"]))
    assert mine["blocks"] == parent["blocks"] and mine["long"] == {**parent["long"], "reverse": True}
    # the mirror: short where the long leaves, cover where it enters (qr23: below the lower band, above the upper one)
    assert mine["short"] == {"entry": parent["long"]["exit"], "exit": parent["long"]["entry"], "reverse": True}
    if key == "QR23_LS":
        assert "exits" not in mine and "exits" not in parent  # no stop, as the parent
    else:
        assert mine["exits"] == {"long": parent["exits"], "short": {"stop": _mirror_stop(parent["exits"]["stop"])}}
    check_definition(mine, bar_minutes=1440)
    assert definition_hash(mine) != definition_hash(parent)


def test_the_list_is_two_btc_rows_on_the_round1_dates_on_the_perpetual_with_shorts():
    from sleeve_fund.research.pinned import parse

    text = LIST.read_text()
    round1 = (ROOT / "configs" / "studies" / "qr-round1-list.md").read_text()
    assert rescore.rows_of(text) == [(1, "qr23_ls", 1440), (2, "qr21_ls", 1440)]
    assert parse(text) == parse(round1) and parse(text).holdout_days == rescore.HOLDOUT_DAYS
    for name, key in (("qr23_ls", "QR23_LS"), ("qr21_ls", "QR21_LS")):
        req = rescore.request(name, 1440, "binance", "BTC/USDT", parse(text))
        defn = load_definition_file(definition_path(PREREGISTERED[key]["definition"]))
        assert req.params["definition"] == defn and req.params["market"] == "perp" and req.params["allow_short"] is True
        assert (req.use_holdout, req.holdout_days, req.train_days, req.test_days) == (False, 365, 730, 365)
        req.validate()


def test_the_list_dry_runs(capfd):
    assert rescore.main([str(LIST), "--dry-run"], run=lambda *a, **k: None, journal=lambda: None) == 0
    assert "dry run: 2 checked, 0 run" in capfd.readouterr().out


@pytest.mark.parametrize("name", ["qr23_vma_band_sma50_ls", "qr21_trend_sma200_ls", "qr01b_trend_multispeed_ls_v3_adx"])
def test_every_long_and_short_declares_the_reversal(name):
    """Advisor 24 Oct (#161's declared reversal): the round 0 long and short Strategies flip on the signal candle."""
    defn = load_definition_file(definition_path(name))
    assert defn["long"]["reverse"] is True and defn["short"]["reverse"] is True


def _run(defn: dict, close: np.ndarray, instrument, half_spread: float = 0.0005):
    from sleeve_fund.research.runner import run_backtest

    idx = pd.date_range("2024-01-01", periods=len(close), freq="1D", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1e3}, index=idx)
    res = run_backtest("rules", df, instrument, params={**to_params(defn), "market": "perp", "allow_short": True},
                       bar_minutes=1440, half_spread=half_spread)
    f = res.fills.sort_values("ts_last")
    return [(f.loc[o, "ts_last"], str(f.loc[o, "side"]), res.decisions[o]["intent"], f.loc[o], df.loc[f.loc[o, "ts_last"],
             "close"]) for o in f.index]


def test_a_flip_on_candle_k_is_two_fills_at_its_close_each_with_its_fee_and_half_spread(instrument):
    """Advisor 24 Oct pin: a flip on candle k gives exactly two fills at k's close (the old side's exit, the other
    side's entry), two fees and two half spreads, and no same-direction reopen."""
    defn = load_definition_file(definition_path("qr23_vma_band_sma50_ls"))
    defn["blocks"]["sma"]["period"] = 5  # a short average, so a made-up swing flips it often
    fills = _run(defn, 100 + 20 * np.sin(np.arange(200) / 6.0), instrument)
    by_ts: dict = {}
    for ts, side, intent, row, close in fills:
        by_ts.setdefault(ts, []).append((side, intent, row, close))
    flips = [v for v in by_ts.values() if len(v) == 2]
    assert len(flips) >= 5 and all(len(v) <= 2 for v in by_ts.values())
    for pair in flips:
        assert sorted(i for _, i, _, _ in pair) == ["entry", "exit"]
        assert pair[0][0] == pair[1][0]  # the exit of one side and the other side's entry trade the same way
        for side, _, row, close in pair:
            assert len(row["commissions"]) == 1 and float(str(row["commissions"][0]).split()[0]) > 0  # its own fee
            px = float(row["avg_px"])
            assert px <= close * (1 - 0.0005) if side == "SELL" else px >= close * (1 + 0.0005)  # its half spread
    sides = [side for _, side, intent, _, _ in fills if intent == "entry"]
    assert all(a != b for a, b in zip(sides, sides[1:]))  # every new entry is the other side: no same-side reopen


ADX_GATED = {
    "version": 1, "name": "flip-gate", "title": "Flip gate", "reason": "test",
    "blocks": {"sma": {"kind": "sma", "period": 5}, "adx": {"kind": "adx", "period": 2, "timeframe": "1d"}},
    "long": {"entry": {"left": "close", "op": ">", "right": "sma"}, "exit": {"left": "close", "op": "<", "right": "sma"},
             "reverse": True},
    "short": {"entry": {"all": [{"left": "close", "op": "<", "right": "sma"},
                                {"left": "adx", "op": "<=", "right": 25, "no_value": "pass"}]},
              "exit": {"left": "close", "op": ">", "right": "sma"}, "reverse": True},
}


@pytest.mark.parametrize("limit, short_opens", [(25, False), (100, True)])
def test_a_reversal_entry_still_passes_its_gate_or_the_long_exits_flat(instrument, limit, short_opens):
    """Advisor 24 Oct (V3): the reversal entry is the side's whole entry rule, ADX gate included: with ADX above the
    limit at the flip the long exits flat and no short opens."""
    import copy

    defn = copy.deepcopy(ADX_GATED)
    defn["short"]["entry"]["all"][1]["right"] = limit
    close = np.concatenate([np.full(40, 100.0), np.linspace(100, 160, 30), np.linspace(160, 60, 30)])  # strong trends
    fills = _run(defn, close, instrument, half_spread=0)
    long_exit = next(ts for ts, side, intent, _, _ in fills if intent == "exit" and side == "SELL")
    shorts = [ts for ts, side, intent, _, _ in fills if intent == "entry" and side == "SELL"]
    assert (long_exit in shorts) is short_opens
