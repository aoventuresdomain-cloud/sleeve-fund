"""qr01b round 2, V4 "shorts only in a down market" (Researcher 24 Oct, Advisor cleared): V3 exactly plus two daily
blocks and one short-entry condition, the BTC list on V3's dates with the holdout locked, and the pin that a short
entry is refused on a day whose last completed daily close is above the average."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from sleeve_fund.research import rescore
from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import (
    check_definition,
    definition_hash,
    load_definition_file,
    to_params,
)

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "qr01b-r2-list.md"
GATE = {"left": "close1d", "op": "<", "right": "sma200d"}


def _v4() -> dict:
    return load_definition_file(definition_path(PREREGISTERED["V4_SHORT_SMA200"]["definition"]))


def test_v4_is_v3_plus_two_daily_blocks_and_one_short_condition():
    v4, v3 = _v4(), load_definition_file(definition_path("qr01b_trend_multispeed_ls_v3_adx"))
    assert v4["blocks"] == {**v3["blocks"], "close1d": {"kind": "sma", "period": 1, "timeframe": "1d"},
                            "sma200d": {"kind": "sma", "period": 200, "timeframe": "1d"}}
    assert v4["long"] == v3["long"] and v4["exits"] == v3["exits"]
    assert v4["short"]["exit"] == v3["short"]["exit"] and v4["short"]["reverse"] is True
    assert v4["short"]["entry"] == {"all": v3["short"]["entry"]["all"] + [GATE]}  # no no_value: fails until it has one
    assert v4["name"] == "qr01b-trend-multispeed-ls-v4-short-sma200"
    check_definition(v4, bar_minutes=240)
    assert definition_hash(v4) != definition_hash(v3)


def test_the_list_is_one_btc_row_on_v3s_dates_on_the_perpetual_with_shorts():
    from sleeve_fund.research.pinned import parse

    text = LIST.read_text()
    v3_list = (ROOT / "configs" / "studies" / "qr01b-v3-adx-list.md").read_text()
    assert rescore.rows_of(text) == [(1, "v4_short_sma200", 240)]
    assert parse(text) == parse(v3_list) and parse(text).holdout_days == rescore.HOLDOUT_DAYS
    req = rescore.request("v4_short_sma200", 240, "binance", "BTC/USDT", parse(text))
    assert req.params["definition"] == _v4() and req.params["market"] == "perp" and req.params["allow_short"] is True
    assert (req.use_holdout, req.holdout_days, req.train_days, req.test_days) == (False, 365, 730, 365)
    req.validate()


def test_the_list_dry_runs(capfd):
    assert rescore.main([str(LIST), "--dry-run"], run=lambda *a, **k: None, journal=lambda: None) == 0
    assert "dry run: 1 checked, 0 run" in capfd.readouterr().out


def _short_entries(defn: dict, close: np.ndarray, instrument) -> list[pd.Timestamp]:
    from sleeve_fund.research.runner import run_backtest

    idx = pd.date_range("2024-01-01", periods=len(close), freq="4h", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1e3}, index=idx)
    res = run_backtest("rules", df, instrument, params={**to_params(defn), "market": "perp", "allow_short": True},
                       bar_minutes=240, half_spread=0)
    f = res.fills
    return sorted(f.loc[o, "ts_last"] for o in f.index
                  if res.decisions[o]["intent"] == "entry" and str(f.loc[o, "side"]) == "SELL")


def _small_v4(sma_days: int) -> dict:
    """V4 with short periods so a made-up path can turn it; the ADX gate opened, so only the new rule decides."""
    d = copy.deepcopy(_v4())
    for k, p in {"f1": 2, "s1": 4, "f2": 3, "s2": 6, "f3": 4, "s3": 8}.items():
        d["blocks"][k]["period"] = p
    d["blocks"]["sma200d"]["period"] = sma_days
    d["blocks"]["adx"]["period"] = d["blocks"]["atr"]["period"] = 2  # short warm-ups, so the path's early days trade
    for side in ("long", "short"):
        d[side]["entry"]["all"][3]["right"] = 100
    return d


def _without_gate(d: dict) -> dict:
    d = copy.deepcopy(d)
    d["short"]["entry"]["all"].remove(GATE)
    return d


def test_a_short_is_refused_on_a_day_whose_last_daily_close_is_above_the_average(instrument):
    """Pin (Researcher 24 Oct): 20 rising days, then day 20 falls inside the day. The 4h trend turns down on day 20,
    but day 19's close (the last completed one) is above the average, so no short opens on day 20; without the rule one
    does. Day 20 closes below the average, so the next day may short."""
    up = np.repeat(np.linspace(100, 140, 20), 6)          # days 0-19, six 4h candles a day
    crash = np.linspace(138, 60, 6)                       # day 20: falls through the average inside the day
    down = np.linspace(59, 40, 18)                        # days 21-23
    close = np.concatenate([up, crash, down])
    day20 = pd.Timestamp("2024-01-21", tz="UTC")
    day21 = pd.Timestamp("2024-01-22", tz="UTC")
    gated, ungated = _short_entries(_small_v4(5), close, instrument), _short_entries(_without_gate(_small_v4(5)), close,
                                                                                     instrument)
    assert any(day20 <= ts < day21 for ts in ungated)    # the 4h rules alone short on day 20
    assert not any(ts < day21 for ts in gated)           # the rule refuses it: day 19 closed above the average
    assert any(ts >= day21 for ts in gated)              # day 20 closed below it, so the next day shorts


def test_no_decision_before_the_200_day_average_exists():
    """No short until the average exists: the warm-up waits for 200 completed days (and the part day it starts in) on
    4h candles, so the first decision always reads a valued average."""
    from sleeve_fund.strategies.rules import Rules

    assert Rules.warmup_needed({"definition": _v4()}, 240) >= 201 * 6
    assert check_definition(_v4(), bar_minutes=240).warmup[1440] == 200
