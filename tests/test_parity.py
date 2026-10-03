"""Parity: the engine's trend filter trades exactly as the research rules say it should.

The reference below is the lab's trend-filter rules (sleeve_fund/lab/trend_filter.py:
exponential averages, volatility targeting, the rebalance band), written out in plain pandas and
held as a quantity between trades, as any real account does. The engine must match it to within
25 basis points a year with the same number of trades, which is what lets research run on the
engine alone.

One difference from the lab is deliberate: the lab's return maths re-weighted every bar for free
(its position was a constant share of equity between trades). Real holdings drift with the price
until the next trade, so the reference does too.
"""

import math

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.venues import venue


def _bars(minutes, years=3, seed=7, vol=0.6):
    rng = np.random.default_rng(seed)
    per_day = 1440 // minutes
    n = int(years * 365 * per_day)
    regime = max(n // 12, 1)
    drift = np.repeat(rng.choice([-1, 1], size=n // regime + 1) * 0.3 / (365 * per_day), regime)[:n]
    r = drift + rng.normal(0, vol / math.sqrt(365 * per_day), n)
    c = 20_000 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2022-01-01", periods=n, freq=f"{minutes}min", tz="UTC") + pd.Timedelta(minutes=minutes)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998,
                         "close": c, "volume": 10.0}, index=idx)  # stamped at the close


def _reference_weights(close, minutes, fast, slow, vol_target, lookback_days=30, band=0.25):
    ef = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    es = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    on = (ef > es).astype(float).where(es.notna(), 0.0)
    if not vol_target:
        return on
    per_day = 1440 / minutes
    vol = close.pct_change().rolling(int(lookback_days * per_day), min_periods=int(10 * per_day)).std()
    target = (on * (vol_target / (vol * math.sqrt(365 * per_day))).clip(upper=1.0)).fillna(0.0).to_numpy()
    held, w = np.zeros(len(target)), 0.0
    for i, t in enumerate(target):
        if t == 0.0 or w == 0.0 or abs(t - w) > band * w:
            w = t
        held[i] = w
    return pd.Series(held, index=close.index)


def _reference_equity(close, weights, fee, start, buffer=0.0):
    cash, qty, prev = start, 0.0, 0.0
    trades = 0
    for c, w in zip(close.to_numpy(), weights.to_numpy()):
        if w != prev:
            eq = cash + qty * c
            target = eq * w / c
            if target > qty:  # buying: what free cash allows after the fee
                target = min(target, qty + cash * (1 - buffer - fee) / c)
            dq = target - qty
            cash -= dq * c + abs(dq) * c * fee
            qty, prev, trades = target, w, trades + 1
    return cash + qty * close.iloc[-1], trades


@pytest.mark.parametrize("minutes, fast, slow, vol_target", [
    (240, 21, 55, 0.40),  # the lab's lead variant: 4-hour, volatility-targeted to 40%
    (1440, 20, 50, 0.0),  # daily, all or nothing
])
def test_engine_trend_filter_matches_the_research_rules(minutes, fast, slow, vol_target):
    k = venue()
    prices = _bars(minutes)
    params = {"fast": fast, "slow": slow, "ema": 1, "vol_target": vol_target}
    res = run_backtest("trend_filter", prices, k.instrument("BTC", "USD"), params, starting_capital=1_000_000,
                       bar_minutes=minutes)
    w = _reference_weights(prices["close"], minutes, fast, slow, vol_target)
    # All-or-nothing entries keep the engine's 1% cash buffer, as paper does.
    ref_end, ref_trades = _reference_equity(prices["close"], w, float(k.fees.taker), 1_000_000,
                                            buffer=0.0 if vol_target else 0.01)
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr = lambda end: (end / 1_000_000) ** (1 / years) - 1  # noqa: E731
    assert ref_trades > 10  # the test means something only if the strategy trades
    assert len(res.fills) == ref_trades
    assert abs(cagr(res.equity.iloc[-1]) - cagr(ref_end)) < 0.0025  # 25 bp a year
