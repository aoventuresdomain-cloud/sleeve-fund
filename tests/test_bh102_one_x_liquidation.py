"""BH-102 (HoE, Advisor 21 Oct 2026): a 1x isolated perp position liquidates at maintenance like any other, funding
drains its margin until it does, equity never goes below zero, and the venue liquidation runs in every perp
backtest, a benchmark with no risk profile and no runtime included. DA's repro: a funded 1x hold went to -113%."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sleeve_fund import margin

MM = 0.005  # a flat maintenance rate


def test_a_1x_long_has_a_liquidation_price_at_maintenance():
    """Fully paid for, its margin is its notional less the entry fee: it liquidates where its value less the fee
    falls to the maintenance on its entry value (formula B, the nearer here), not never."""
    qty, entry, fee = 1.0, 30_000.0, 16.5
    liq = margin.isolated_liquidation(-fee, qty, entry, 1.0, MM, fees=fee)
    assert liq == pytest.approx(entry * MM + fee / qty)
    # With cash to spare beside it (a Strategy that sized under its balance), the same: only its margin is at risk.
    assert margin.isolated_liquidation(2_000.0, qty, entry, 1.0, MM, fees=fee) == pytest.approx(liq)
    # A payment out of that margin brings it closer, as on any leverage.
    assert margin.shifted_liquidation(liq, qty, -3_000.0, MM, entry=entry) == pytest.approx(liq + 3_000.0)
    # A short at 1x still liquidates above, as it always did.
    assert margin.isolated_liquidation(2 * entry - fee, -qty, entry, 1.0, MM, fees=fee) > entry


def _funded_hold(tmp_path, monkeypatch, bull_rate: float):
    """DA's BH-102 repro: a 1x BTC perp hold through the engine on a stylised 2021-2025 path, funding paid every 8h
    at bull_rate until Dec 2021, then 0.002% to 2023 and 0.01% after; no venue called."""
    from test_binance import INFO

    from sleeve_fund import funding
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import binance_contract, venue

    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    inst = b.instrument("BTC", "USDT")
    knots = [("2021-01-01", 29000), ("2021-04-14", 64000), ("2021-07-20", 30000), ("2021-11-10", 69000),
             ("2022-06-18", 19000), ("2022-11-21", 15800), ("2023-12-31", 42000), ("2024-03-14", 73000),
             ("2024-12-17", 106000), ("2025-12-31", 94000)]
    idx = pd.date_range("2021-01-01", "2025-12-31", freq="1D", tz="UTC") + pd.Timedelta(days=1)
    k = pd.Series([v for _, v in knots],
                  index=pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1) for d, _ in knots]))
    c = np.exp(np.log(k).reindex(k.index.union(idx)).interpolate("time")).reindex(idx)
    bars = pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": 1e9}, index=idx)
    bars["high"] = bars[["open", "high"]].max(axis=1)
    bars["low"] = bars[["open", "low"]].min(axis=1)
    times = pd.date_range("2021-01-01", "2026-01-01", freq="8h", tz="UTC")
    rate = np.where(times < pd.Timestamp("2021-12-01", tz="UTC"), bull_rate,
                    np.where(times < pd.Timestamp("2023-01-01", tz="UTC"), 0.00002, 0.0001))
    p = funding._path("BINANCE", "BTC/USDT")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), float(r)] for t, r in zip(times, rate)]}))
    return run_backtest("buy_and_hold", bars, inst, {"market": "perp"}, starting_capital=10_000.0), bars


def _intents(res) -> list[str]:
    return [res.decisions.get(str(i), {}).get("intent") for i in res.fills.index]


def test_a_funded_1x_hold_with_no_risk_profile_is_liquidated_and_never_goes_below_zero(tmp_path, monkeypatch):
    res, bars = _funded_hold(tmp_path, monkeypatch, 0.0005)
    assert float(res.equity.min()) >= 0.0, "isolated margin: never below zero (it went to -113%)"
    intents = _intents(res)
    assert intents.count("liquidation") == 1, intents
    # Flat after it: the hold doesn't re-buy with what is left (DA (3), pending the Advisor).
    assert intents[-1] == "liquidation" and len(res.fills) == 2



def test_a_1x_hold_on_funding_it_can_carry_is_never_liquidated(tmp_path, monkeypatch):
    """At an ordinary bull rate the margin carries the funding through the 2022 low: the same hold as before."""
    res, _ = _funded_hold(tmp_path, monkeypatch, 0.0001)
    assert "liquidation" not in _intents(res) and len(res.fills) == 1
    assert float(res.equity.min()) > 0.0
