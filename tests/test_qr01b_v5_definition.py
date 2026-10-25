"""qr01b V5, "long-only with the ADX gate" (Researcher 25 Oct, Advisor cleared as try 28; the PM chose one long-only
test 25 Oct): V4 exactly with its short leg deleted, which makes it the long-only parent plus V3's daily ADX gate; the
BTC list on V4's dates with the holdout locked; and the pin that it never opens a short."""

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
LIST = ROOT / "configs" / "studies" / "qr01b-v5-lo-list.md"
ADX_GATE = {"left": "adx", "op": "<=", "right": 25, "no_value": "pass"}


def _v5() -> dict:
    return load_definition_file(definition_path(PREREGISTERED["V5_LO_ADX"]["definition"]))


def test_v5_is_v4_without_its_short_leg():
    v5, v4 = _v5(), load_definition_file(definition_path("qr01b_trend_multispeed_ls_v4_short_sma200"))
    assert "short" not in v5 and set(v5["exits"]) == {"long"} and v5["exits"]["long"] == v4["exits"]["long"]
    assert v5["blocks"] == {k: b for k, b in v4["blocks"].items() if k not in ("close1d", "sma200d")}
    assert v5["long"] == {k: v for k, v in v4["long"].items() if k != "reverse"}  # nothing to flip into
    assert v5["name"] == "qr01b-trend-multispeed-lo-v5-adx"
    check_definition(v5, bar_minutes=240)
    assert definition_hash(v5) != definition_hash(v4)


def test_v5_is_the_long_only_parent_plus_the_daily_adx_gate():
    v5, parent = _v5(), load_definition_file(definition_path(PREREGISTERED["V5_LO_ADX"]["parent"]))
    assert v5["blocks"] == {**parent["blocks"], "adx": {"kind": "adx", "period": 14, "timeframe": "1d"}}
    assert v5["long"]["entry"] == {"all": parent["long"]["entry"]["all"] + [ADX_GATE]}
    assert v5["long"]["exit"] == parent["long"]["exit"]
    assert v5["exits"]["long"] == parent["exits"]


def test_the_list_is_one_btc_row_on_v4s_dates_on_the_perpetual_long_only():
    from sleeve_fund.research.pinned import parse

    text = LIST.read_text()
    v4_list = (ROOT / "configs" / "studies" / "qr01b-r2-list.md").read_text()
    assert rescore.rows_of(text) == [(1, "v5_lo_adx", 240)]
    assert parse(text) == parse(v4_list) and parse(text).holdout_days == rescore.HOLDOUT_DAYS
    req = rescore.request("v5_lo_adx", 240, "binance", "BTC/USDT", parse(text))
    assert req.params["definition"] == _v5() and req.params["market"] == "perp" and not req.params.get("allow_short")
    assert (req.use_holdout, req.holdout_days, req.train_days, req.test_days) == (False, 365, 730, 365)
    req.validate()


def test_the_list_dry_runs(capfd):
    assert rescore.main([str(LIST), "--dry-run"], run=lambda *a, **k: None, journal=lambda: None) == 0
    assert "dry run: 1 checked, 0 run" in capfd.readouterr().out


def test_it_trades_long_and_never_opens_a_short(instrument):
    """Pin: on a seeded walk that trends both ways, a short-period copy (ADX gate opened) opens longs and every
    selling fill closes one; even with shorts allowed by the venue, the definition has no short to open."""
    from sleeve_fund.research.runner import run_backtest

    d = copy.deepcopy(_v5())
    for k, p in {"f1": 2, "s1": 4, "f2": 3, "s2": 6, "f3": 4, "s3": 8}.items():
        d["blocks"][k]["period"] = p
    d["blocks"]["adx"]["period"] = d["blocks"]["atr"]["period"] = 2
    d["long"]["entry"]["all"][3]["right"] = 100
    close = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, 300)))
    idx = pd.date_range("2024-01-01", periods=len(close), freq="4h", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995, "close": close, "volume": 1e3},
                      index=idx)
    res = run_backtest("rules", df, instrument, params={**to_params(d), "market": "perp", "allow_short": True},
                       bar_minutes=240, half_spread=0)
    f = res.fills
    intents = [(res.decisions[o]["intent"], str(f.loc[o, "side"])) for o in f.index]
    assert ("entry", "BUY") in intents
    assert ("entry", "SELL") not in intents
