"""Improve-loop round 2 (Researcher 24 Oct, Advisor cleared): qr23 round 0 exactly plus qr21 round 1's slope rule on
its 50-day average, the one-row BTC list on round 0's dates, and the Advisor's pins: every entry holds its band and
slope conditions, and a reversal's other leg passes its own slope check or the Strategy goes flat."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from sleeve_fund.research import rescore
from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import check_definition, definition_hash, load_definition_file, to_params

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "qr-improve-r2-list.md"


def _load() -> tuple[dict, dict]:
    v = PREREGISTERED["QR23_R2"]
    return load_definition_file(definition_path(v["definition"])), load_definition_file(definition_path(v["parent"]))


def test_qr23_r2_is_round_0_plus_a_slope_check_on_each_entry():
    mine, r0 = _load()
    assert PREREGISTERED["QR23_R2"]["parent"] == "qr23_vma_band_sma50_ls"  # round 0, not the reverted round 1
    assert mine["blocks"] == {**r0["blocks"], "sma50_ago": {"kind": "lag", "period": 20, "input": "sma"}}
    assert "exits" not in mine
    for side, op in (("long", ">"), ("short", "<")):
        assert mine[side] == {**r0[side], "entry": {"all": [r0[side]["entry"],
                                                            {"left": "sma", "op": op, "right": "sma50_ago"}]}}
        assert mine[side]["reverse"] is True
    assert check_definition(mine, bar_minutes=1440).warmup[None] >= 50 + 21
    assert definition_hash(mine) != definition_hash(r0)


def test_the_list_is_one_btc_row_on_round_0s_dates_on_the_perpetual_with_shorts():
    from sleeve_fund.research.pinned import parse

    text = LIST.read_text()
    r0 = (ROOT / "configs" / "studies" / "qr-improve-r0-list.md").read_text()
    assert rescore.rows_of(text) == [(1, "qr23_r2", 1440)]
    assert parse(text) == parse(r0) and parse(text).holdout_days == rescore.HOLDOUT_DAYS
    req = rescore.request("qr23_r2", 1440, "binance", "BTC/USDT", parse(text))
    assert req.params["definition"] == _load()[0]
    assert req.params["market"] == "perp" and req.params["allow_short"] is True
    assert (req.use_holdout, req.holdout_days, req.train_days, req.test_days) == (False, 365, 730, 365)
    req.validate()


def test_the_list_dry_runs(capfd):
    assert rescore.main([str(LIST), "--dry-run"], run=lambda *a, **k: None, journal=lambda: None) == 0
    assert "dry run: 1 checked, 0 run" in capfd.readouterr().out


# --- the Advisor's pins, on a short-period copy so a made-up path turns it -----------------------------------------

SMA, AGO = 10, 5
# A seeded random walk: its band crosses come both with and against the average's slope, so both kinds of reversal happen.
WALK = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, 300)))


def _run(close: np.ndarray, instrument) -> tuple[list, list, pd.DataFrame]:
    from sleeve_fund.research.runner import run_backtest

    d = copy.deepcopy(_load()[0])
    d["blocks"]["sma"]["period"], d["blocks"]["sma50_ago"]["period"] = SMA, AGO
    idx = pd.date_range("2024-01-01", periods=len(close), freq="1D", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1e3}, index=idx)
    res = run_backtest("rules", df, instrument, params={**to_params(d), "market": "perp", "allow_short": True},
                       bar_minutes=1440, half_spread=0)
    f = res.fills
    rows = sorted((f.loc[o, "ts_last"], str(f.loc[o, "side"]), res.decisions[o]["intent"]) for o in f.index)
    sma = df["close"].rolling(SMA).mean()
    df["sma"], df["ago"] = sma, sma.shift(AGO)
    return ([(ts, s) for ts, s, i in rows if i == "entry"], [(ts, s) for ts, s, i in rows if i == "exit"], df)


def test_every_entry_holds_its_band_and_slope_conditions(instrument):
    entries, _, df = _run(WALK, instrument)
    assert len(entries) >= 10
    for ts, side in entries:
        c, m, ago = df.loc[ts, "close"], df.loc[ts, "sma"], df.loc[ts, "ago"]
        assert (c > 1.01 * m and m > ago) if side == "BUY" else (c < 0.99 * m and m < ago)


def test_a_reversal_whose_other_leg_fails_its_slope_goes_flat(instrument):
    """Advisor 24 Oct: on a reversal the other leg must pass its own slope check, or the Strategy goes flat."""
    entries, exits, df = _run(WALK, instrument)
    opened = set(entries)
    for exit_side, falling in (("SELL", True), ("BUY", False)):  # a long's exit sells, a short's buys
        legs = [ts for ts, side in exits if side == exit_side]
        passes = [ts for ts in legs if bool(df.loc[ts, "sma"] < df.loc[ts, "ago"]) is falling]
        fails = [ts for ts in legs if ts not in passes]
        assert passes and fails
        assert all((ts, exit_side) in opened for ts in passes)  # passed: the other side opens on the same candle
        assert not any((ts, exit_side) in opened for ts in fails)  # failed: out, and flat
