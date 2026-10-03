"""Shared building blocks from the v4 strategy doc, computed without look-ahead.

Every column on a bar uses only that bar and earlier bars, so it is known at the bar's close.
Lookbacks that the doc gives in clock time are converted to bars from the bar size, so a
strategy means the same thing on 1-minute and 15-minute candles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_ANCHORS = ("utc", "us_open")  # 00:00 UTC, or the US cash open (09:30 New York, DST-aware)


def session_ids(index: pd.DatetimeIndex, anchor: str) -> pd.Series:
    """The session each bar belongs to, as the session's start date (24-hour sessions)."""
    if anchor == "utc":
        start = index.floor("D")
    elif anchor == "us_open":
        ny = index.tz_convert("America/New_York")
        shifted = ny - pd.Timedelta(hours=9, minutes=30)
        start = (shifted.floor("D") + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
    else:
        raise ValueError(f"unknown session anchor {anchor!r}")
    return pd.Series(start, index=index)


def bar_minutes(df: pd.DataFrame) -> int:
    return int(round(pd.Series(df.index).diff().dropna().median() / pd.Timedelta(minutes=1)))


def bars_for(minutes: float, bar: int) -> int:
    return max(1, int(round(minutes / bar)))


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder's RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(dn > 0, 100.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman: net move over n bars / sum of absolute bar moves. 1 = clean trend, 0 = chop."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return net / path.replace(0, np.nan)


def anchored_vwap(df: pd.DataFrame, anchor_pos: int) -> pd.Series:
    """VWAP from bar `anchor_pos` (inclusive) onwards; NaN before it."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    v = df["volume"].to_numpy()
    out = np.full(len(df), np.nan)
    pv = np.cumsum((tp.to_numpy() * v)[anchor_pos:])
    cv = np.cumsum(v[anchor_pos:])
    with np.errstate(invalid="ignore", divide="ignore"):
        out[anchor_pos:] = pv / cv
    return pd.Series(out, index=df.index)


def sessions(df: pd.DataFrame, anchor: str) -> pd.DataFrame:
    """One row per session: open, high, low, close, volume, and ATR(14) of completed sessions."""
    sid = session_ids(df.index, anchor)
    g = df.groupby(sid.values)
    s = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum(), "bars": g["close"].size()})
    s["atr14"] = atr(s["high"], s["low"], s["close"], 14)
    return s


CROSS_SAMPLE_MIN = 15


def with_session_blocks(df: pd.DataFrame, anchor: str = "utc", opening_range_min: int = 30,
                        relvol_sessions: int = 20) -> pd.DataFrame:
    """Adds session VWAP and sigma bands, the prior session's levels and ATR, the opening range,
    elapsed time, VWAP crossings, share of closes above VWAP, and relative volume."""
    bar = bar_minutes(df)
    out = df.copy()
    sid = session_ids(df.index, anchor)
    out["session"] = sid.values
    tp = (df["high"] + df["low"] + df["close"]) / 3
    g = out.groupby("session", sort=False)
    cv = g["volume"].cumsum()
    pv = (tp * df["volume"]).groupby(sid.values).cumsum()
    pv2 = (tp * tp * df["volume"]).groupby(sid.values).cumsum()
    vwap = pv / cv.replace(0, np.nan)
    out["vwap"] = vwap.ffill()
    var = (pv2 / cv.replace(0, np.nan) - vwap**2).clip(lower=0)
    out["vwap_sd"] = np.sqrt(var).ffill()
    out["elapsed"] = np.round((df.index - pd.DatetimeIndex(sid)) / pd.Timedelta(minutes=1)).astype(int) + bar
    # Prior completed session: levels and daily ATR (known from the session's first bar).
    sess = sessions(df, anchor)
    prior = sess[["high", "low", "close", "atr14"]].shift(1)
    out["prior_high"] = out["session"].map(prior["high"])
    out["prior_low"] = out["session"].map(prior["low"])
    out["prior_close"] = out["session"].map(prior["close"])
    out["atr_d"] = out["session"].map(prior["atr14"])
    # Session so far.
    out["sess_high"] = g["high"].cummax()
    out["sess_low"] = g["low"].cummin()
    # Opening range: known only once its window has closed.
    in_or = out["elapsed"] <= opening_range_min
    orh = out["high"].where(in_or).groupby(sid.values).transform("max")
    orl = out["low"].where(in_or).groupby(sid.values).transform("min")
    out["or_high"] = orh.where(~in_or)
    out["or_low"] = orl.where(~in_or)
    # VWAP side and crossings.
    side = np.sign(out["close"] - out["vwap"])
    out["above_share"] = (side > 0).astype(float).groupby(sid.values).cumsum() / g.cumcount().add(1)
    # Crossings are counted on the VWAP side sampled every 15 minutes, so the count means the same
    # thing whatever the bar size (on 1-minute bars noise around VWAP would otherwise count as crosses).
    on_grid = (out["elapsed"] % CROSS_SAMPLE_MIN == 0) & (side != 0)
    sampled = side.where(on_grid)
    prev = sampled.groupby(sid.values).ffill().groupby(sid.values).shift()
    flips = on_grid & prev.notna() & (sampled != prev)
    out["vwap_crosses"] = flips.astype(int).groupby(sid.values).cumsum()
    # Relative volume: session volume so far vs the same elapsed time over the previous N sessions.
    k = out["elapsed"]
    piv = cv.groupby([sid.values, k.values]).last().unstack()
    base = piv.shift(1).rolling(relvol_sessions, min_periods=relvol_sessions // 2).mean()
    stacked = base.stack()
    ref = pd.Series(stacked.values, index=pd.MultiIndex.from_tuples(stacked.index))
    keys = pd.MultiIndex.from_arrays([sid.values, k.values])
    out["relvol"] = cv.to_numpy() / ref.reindex(keys).to_numpy()
    return out


def vwap_slope_atr_per_hour(out: pd.DataFrame, window_min: int = 60) -> pd.Series:
    """Change in session VWAP over the last window, per hour, in units of daily ATR. Early in
    the session the window is the session so far (from the first bar's VWAP)."""
    bar = bar_minutes(out)
    n = bars_for(window_min, bar)
    g = out["vwap"].groupby(out["session"])
    prev = g.shift(n)
    first = g.transform("first")
    span = np.minimum(window_min, out["elapsed"] - bar).astype(float)
    prev = prev.where(prev.notna(), first)
    slope = (out["vwap"] - prev) / (span.replace(0, np.nan) / 60) / out["atr_d"]
    return slope


TREND_UP, TREND_DOWN, BALANCE, UNCLEAR = "trend_up", "trend_down", "balance", "unclear"


def classify(out: pd.DataFrame, *, side_share: float = 0.8, trend_slope: float = 0.15, balance_slope: float = 0.05,
             min_crosses: int = 3, relvol_min: float = 1.2) -> pd.Series:
    """The doc's day-type rules, evaluated on every bar from what is known at its close.
    Strategies read it at the decision time and hourly after."""
    slope = vwap_slope_atr_per_hour(out)
    up_share, dn_share = out["above_share"], 1 - out["above_share"]
    or_known = out["or_high"].notna()
    run_up = _run_since((out["close"] > out["or_high"]) & or_known, or_known, out["session"])
    run_dn = _run_since((out["close"] < out["or_low"]) & or_known, or_known, out["session"])
    trend_up = (up_share >= side_share) & (slope > trend_slope) & run_up & (out["relvol"] > relvol_min)
    trend_dn = (dn_share >= side_share) & (slope < -trend_slope) & run_dn & (out["relvol"] > relvol_min)
    balance = ((out["vwap_crosses"] >= min_crosses) & (slope.abs() < balance_slope)
               & (out["sess_high"] <= out["prior_high"]) & (out["sess_low"] >= out["prior_low"]))
    label = pd.Series(UNCLEAR, index=out.index)
    label[balance] = BALANCE
    label[trend_up] = TREND_UP
    label[trend_dn] = TREND_DOWN
    label[slope.isna() | out["atr_d"].isna() | ~or_known] = UNCLEAR
    return label


def _run_since(outside: pd.Series, known: pd.Series, session: pd.Series) -> pd.Series:
    """True where price broke out of the opening range and has not closed back inside since the
    first break: the doc's "broken in one direction and not closed back inside"."""
    broke = outside.groupby(session).cummax()  # has broken at some point
    # Once broken, any later close not outside means it closed back inside (or crossed over).
    failed = (broke & ~outside & known).groupby(session).cummax()
    return broke & ~failed
