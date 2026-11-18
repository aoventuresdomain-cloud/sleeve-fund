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
    assert not res.handler_errors
    trips = _trips(res)
    assert len(trips) >= 20
    for opened, closed in trips:
        assert opened.hour == 0 and closed - opened == pd.Timedelta(hours=4), (opened, closed)
    # No orphan stop (Advisor): the stop and its resize are cancelled with the close, so nothing but entries and their
    # exits ever fills, every exit closes the whole position, and the book is flat until the next entry
    assert {res.decisions[o]["intent"] for o in res.fills.index} == {"entry", "exit"}
    signed = res.fills["filled_qty"].astype(float).where(res.fills["side"] == ("BUY" if side == "long" else "SELL"),
                                                         -res.fills["filled_qty"].astype(float))
    held = signed.groupby(res.fills["ts_last"]).sum().sort_index().cumsum()
    assert all(abs(held[closed]) < 1e-9 for _, closed in trips)  # (a last entry may still be open at the data's end)
    flat = res.exposure[[t for t in res.exposure.index if t.hour not in (0, 4)]]
    assert (flat.abs() < 1e-12).all()
    if volume < 1e6:  # the case that waited: the stop resized at the bar the exit is decided on
        assert any(t.hour == 4 for t in resized)
    else:  # full fills: no resize, so the change can't touch them (each trip as before, on the same candles)
        assert resized == []


@pytest.mark.parametrize("backtest, resting_stops, status, busy", [
    (True, False, "PENDING_UPDATE", False),  # the backtest's stop being resized: not busy
    (False, True, "PENDING_UPDATE", False),  # paper with resting stops on (A1): the same
    (False, False, "PENDING_UPDATE", True),  # paper's default has no resting stop: as before
    (True, False, "PENDING_CANCEL", True),  # a cancel in flight: a close is under way, still busy
    (True, False, "SUBMITTED", True),  # a stop not yet at the venue: still busy
])
def test_only_a_resting_stop_loss_being_resized_is_not_waited_on(backtest, resting_stops, status, busy):
    from types import SimpleNamespace

    from nautilus_trader.model import OrderStatus, OrderType

    order = SimpleNamespace(client_order_id="S1", status=getattr(OrderStatus, status), is_closed=False,
                            order_type=OrderType.STOP_MARKET)
    strat = SimpleNamespace(
        _risk_stop_id=None, strategy_id="S", _unsent=list, _backtest=backtest,
        _resting_stops=lambda: resting_stops, _target_order=lambda o: False,
        cache=SimpleNamespace(orders_inflight=lambda **k: [order]), decisions={"S1": {"intent": "stop_loss"}})
    assert LongFlatStrategy._busy(strat) is busy
    strat.decisions["S1"]["intent"] = "exit"  # any other order being changed still holds the decision
    assert LongFlatStrategy._busy(strat) is True
