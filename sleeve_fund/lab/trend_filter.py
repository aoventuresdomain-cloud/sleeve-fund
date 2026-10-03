"""Long-only trend filter: the benchmark that beat every strategy in the first lab round.

Hold an instrument while its fast EMA is above its slow EMA on closed bars, otherwise hold cash.
The signal is decided at a bar's close and held over the next bar, so nothing is known early.

Sizing variants:
- full: 100% of the sleeve's share when in the market (the benchmark as first tested),
- vol target: exposure = target volatility / recent realised volatility, capped at 100% (spot, no
  borrowing). To keep turnover (and fees) down, the weight is only changed when it drifts more
  than a band away from the target.

Returns are per instrument, daily, net of fees and slippage on every change in exposure.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sleeve_fund.lab.data import resample
from sleeve_fund.lab.sim import COSTS


@dataclass(frozen=True)
class Params:
    bar_min: int = 1440
    ema_fast: int = 20
    ema_slow: int = 50
    sizing: str = "full"  # "full" or "vol_target"
    vol_target: float = 0.40  # annualised, per instrument
    vol_lookback_days: int = 30
    rebalance_band: float = 0.25  # relative drift before the weight is reset to target

    def as_dict(self) -> dict:
        return asdict(self)


def target_weights(bars: pd.DataFrame, p: Params) -> pd.Series:
    """Desired exposure per bar (0 to 1), decided at the bar's close."""
    c = bars["close"]
    ef = c.ewm(span=p.ema_fast, adjust=False, min_periods=p.ema_fast).mean()
    es = c.ewm(span=p.ema_slow, adjust=False, min_periods=p.ema_slow).mean()
    on = (ef > es).astype(float).where(es.notna(), 0.0)
    if p.sizing == "full":
        return on
    per_day = 1440 / p.bar_min
    r = c.pct_change()
    vol = r.rolling(int(p.vol_lookback_days * per_day), min_periods=int(10 * per_day)).std() * math.sqrt(365 * per_day)
    return (on * (p.vol_target / vol).clip(upper=1.0)).fillna(0.0)


def held_weights(target: pd.Series, band: float) -> pd.Series:
    """Weights actually held: jump to 0 or from 0 at once, otherwise move only when the target
    drifts more than `band` (relative) from what is held."""
    out = np.zeros(len(target))
    w = 0.0
    for i, t in enumerate(target.to_numpy()):
        if t == 0.0 or w == 0.0 or abs(t - w) > band * w:
            w = t
        out[i] = w
    return pd.Series(out, index=target.index)


def daily_returns(minute: pd.DataFrame, p: Params, *, cost: str = "kraken_taker", start=None) -> pd.Series:
    bars = resample(minute, p.bar_min) if p.bar_min > 1 else minute
    w = target_weights(bars, p)
    if p.sizing != "full":
        w = held_weights(w, p.rebalance_band)
    pos = w.shift(1).fillna(0.0)  # decided at the close, held over the next bar
    c = COSTS[cost]
    ret = bars["close"].pct_change().fillna(0.0) * pos - pos.diff().abs().fillna(pos.iloc[0]) * (c["fee"] + c["slip"]) / 1e4
    if start is not None:
        ret = ret[ret.index >= pd.Timestamp(start)]
    return (1 + ret).groupby(ret.index.floor("D")).prod() - 1


def exposure(minute: pd.DataFrame, p: Params, start=None) -> pd.Series:
    """Average share of capital in the market, daily (for the report)."""
    bars = resample(minute, p.bar_min) if p.bar_min > 1 else minute
    w = target_weights(bars, p)
    if p.sizing != "full":
        w = held_weights(w, p.rebalance_band)
    pos = w.shift(1).fillna(0.0)
    if start is not None:
        pos = pos[pos.index >= pd.Timestamp(start)]
    return pos.groupby(pos.index.floor("D")).mean()
