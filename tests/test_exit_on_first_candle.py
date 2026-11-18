"""An exit decided on the first candle after an entry fills on that candle (Advisor 18 Nov 17:38, before the H
combination runs). An entry that fills in parts at one moment has its resting stop resized at the next bar, just
before the decision; that resize, while the venue held it, counted as the last decision still being carried out, so
the exit waited a candle. Figures are synthetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.base import LongFlatStrategy
from sleeve_fund.strategies.definitions import to_params

AT_MIDNIGHT = {"left": {"time": "hour"}, "op": "<=", "right": 0}
AT_FOUR = [{"left": {"time": "hour"}, "op": ">=", "right": 4}, {"left": {"time": "hour"}, "op": "<=", "right": 4}]


def _bars(volume: float) -> pd.DataFrame:
    n = 6 * 30
    idx = pd.date_range("2024-01-01 04:00", periods=n, freq="4h", tz="UTC")
    c = 100 + np.sin(np.arange(n) / 5) * 2
    return pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": volume}, index=idx)


def _definition(side: str, how: str) -> dict:
    sign = "sub" if side == "long" else "add"
    exits = {"stop": {"level": {sign: ["close", 5]}}}
    rules = {"entry": {"all": [AT_MIDNIGHT]}}
    if how == "time stop":
        exits["time_stop"] = {"bars": 1}
    else:
        rules["exit"] = {"all": AT_FOUR}
    return {"version": 1, "reason": "A test case, written before it is run.", side: rules, "exits": {side: exits}}


def _trips(res) -> list:
    rows = [(res.fills.loc[o, "ts_last"], res.decisions[o]["intent"]) for o in res.fills.index]
    rows.sort()
    return [(a[0], b[0]) for a, b in zip(rows[::2], rows[1::2], strict=False) if a[1] == "entry"]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("how", ["exit rule", "time stop"])
@pytest.mark.parametrize("volume", [1e3, 1e9])  # thin: the entry fills in parts; deep: at once
def test_an_exit_on_the_first_candle_after_the_entry_fills_on_it(instrument, monkeypatch, side, how, volume):
    resized = []
    modify = LongFlatStrategy.modify_order

    def watched(self, coid, *a, **k):
        if self.decisions.get(str(coid), {}).get("intent") == "stop_loss":
            resized.append(self.clock.utc_now())
        return modify(self, coid, *a, **k)

    monkeypatch.setattr(LongFlatStrategy, "modify_order", watched)
    params = {**to_params(_definition(side, how)), "market": "perp", "allow_short": side == "short"}
    res = run_backtest("rules", _bars(volume), instrument, params=params, bar_minutes=240, half_spread=0)
    trips = _trips(res)
    assert len(trips) >= 20
    for opened, closed in trips:
        assert opened.hour == 0 and closed - opened == pd.Timedelta(hours=4), (opened, closed)
    if volume < 1e6:  # the case that waited: the stop resized at the bar the exit is decided on
        assert any(t.hour == 4 for t in resized)
