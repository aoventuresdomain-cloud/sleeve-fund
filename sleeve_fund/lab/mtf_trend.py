"""Strategy 4 of the v4 doc: multi-timeframe crossover with anchored-VWAP entry.

The higher timeframe (4-hour by default) sets direction with an EMA crossover and filters chop
with Kaufman's efficiency ratio. Entries come on the lower timeframe (15-minute) when price
pulls back to the anchored VWAP of the current trend (or the higher-timeframe EMA) and RSI
resets and turns. Higher-timeframe values reach the lower timeframe only once their bar has
closed, so nothing is known early.

Not yet in this version: the doc's alternative anchor at the latest higher swing low, and the
one add-on entry. Both are noted in the report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sleeve_fund.lab import blocks
from sleeve_fund.lab.data import resample
from sleeve_fund.lab.sim import COSTS, Book


@dataclass(frozen=True)
class Params:
    htf_min: int = 240
    ema_fast: int = 21
    ema_slow: int = 55
    er_len: int = 20
    er_min: float = 0.3
    er_exit: float = 0.15
    anchor_lookback: int = 20
    near_atr: float = 0.25
    rsi_len: int = 14
    rsi_reset: float = 40.0
    stop_atr: float = 0.25
    partial_r: float = 2.0
    partial_frac: float = 1 / 3
    chandelier_atr: float = 3.0
    risk: float = 0.0075
    max_leverage: float = 3.0

    def as_dict(self) -> dict:
        return asdict(self)


def higher_timeframe(ltf: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Higher-timeframe columns aligned to lower-timeframe bars, available only after the
    higher bar closes."""
    h = resample(ltf, p.htf_min) if p.htf_min > blocks.bar_minutes(ltf) else ltf.copy()
    h = h.copy()
    h["ema_f"] = h["close"].ewm(span=p.ema_fast, adjust=False, min_periods=p.ema_fast).mean()
    h["ema_s"] = h["close"].ewm(span=p.ema_slow, adjust=False, min_periods=p.ema_slow).mean()
    h["er"] = blocks.efficiency_ratio(h["close"], p.er_len)
    h["atr"] = blocks.atr(h["high"], h["low"], h["close"], 14)
    h["dir"] = np.sign(h["ema_f"] - h["ema_s"]).fillna(0)
    # The bar a crossover happened on, and the extreme of the bars before it: the trend's anchor.
    cross = h["dir"].ne(h["dir"].shift()) & h["dir"].ne(0) & h["dir"].shift().notna()
    # Anchor time: the bar of that lowest low (long) / highest high (short) before the cross.
    anchor_t = pd.Series(pd.NaT, index=h.index, dtype="datetime64[ns, UTC]")
    lows, highs, idx = h["low"].to_numpy(), h["high"].to_numpy(), h.index
    for j in np.flatnonzero(cross.to_numpy()):
        lo = max(0, j - p.anchor_lookback)
        if j - lo < 2:
            continue
        seg = slice(lo, j)
        k = lo + (int(np.argmin(lows[seg])) if h["dir"].iloc[j] > 0 else int(np.argmax(highs[seg])))
        anchor_t.iloc[j] = idx[k]
    h["anchor_t"] = anchor_t.ffill()
    # Shift to availability: a higher bar's values are usable from its close (= next bar's open).
    avail = h.shift(1)
    cols = ["ema_f", "ema_s", "er", "atr", "dir", "anchor_t", "high", "low", "close"]
    out = avail[cols].reindex(ltf.index, method="ffill")
    out.columns = [f"h_{c}" for c in cols]
    out["h_new"] = ltf.index.isin(h.index)  # first lower bar after a higher bar closed
    return out


def prepare(ltf: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = ltf.join(higher_timeframe(ltf, p))
    out["atr_l"] = blocks.atr(out["high"], out["low"], out["close"], 14)
    out["rsi"] = blocks.rsi(out["close"], p.rsi_len)
    # Anchored VWAP of the current trend, restarted at each new anchor.
    tp = (out["high"] + out["low"] + out["close"]) / 3
    seg = out["h_anchor_t"].ne(out["h_anchor_t"].shift()).cumsum()
    started = out.index >= out["h_anchor_t"].fillna(pd.Timestamp.max.tz_localize("UTC"))
    pv = (tp * out["volume"]).where(started, 0).groupby(seg).cumsum()
    v = out["volume"].where(started, 0).groupby(seg).cumsum()
    out["avwap"] = (pv / v.replace(0, np.nan))
    return out


def run(ltf: pd.DataFrame, p: Params, *, cost: str = "kraken_pro_taker", long_only: bool = False, start=None,
        prepared: pd.DataFrame | None = None) -> Book:
    out = prepared if prepared is not None else prepare(ltf, p)
    book = Book(risk=p.risk, max_leverage=p.max_leverage, long_only=long_only, cost=cost)
    idx = out.index
    o, h, l, c = (out[k].to_numpy() for k in ("open", "high", "low", "close"))
    d, er, emaf = out["h_dir"].to_numpy(), out["h_er"].to_numpy(), out["h_ema_f"].to_numpy()
    hatr, hhigh, hlow, hnew = (out[k].to_numpy() for k in ("h_atr", "h_high", "h_low", "h_new"))
    av, atr_l, rsi = out["avwap"].to_numpy(), out["atr_l"].to_numpy(), out["rsi"].to_numpy()
    i0 = max(1, int(np.searchsorted(idx, pd.Timestamp(start))) if start is not None else 1)
    in_pb, pb_ext, dipped, reset, pb_side = False, np.nan, False, False, 0
    partial, trail, best = False, np.nan, np.nan
    for i in range(i0, len(idx)):
        t = idx[i]
        tr = book.trade
        if tr is not None and tr.entry_time < t:
            if book.check_stop(t, o[i], h[i], l[i]):
                continue
            side = tr.side
            tgt = tr.entry_px + side * p.partial_r * tr.risk_per_unit
            if not partial and ((side > 0 and h[i] >= tgt) or (side < 0 and l[i] <= tgt)):
                book.exit(t, tgt, "target_2R", p.partial_frac)
                partial = True
            if hnew[i] and np.isfinite(hatr[i]):  # a higher bar just closed: update the chandelier stop
                best = max(best, hhigh[i]) if side > 0 else min(best, hlow[i])
                trail = best - side * p.chandelier_atr * hatr[i]
                tr.stop_px = max(tr.stop_px, trail) if side > 0 else min(tr.stop_px, trail)
                if d[i] == -side or (np.isfinite(er[i]) and er[i] < p.er_exit):
                    book.exit(t, c[i], "trend_over")
            if book.trade is None:
                partial, trail, best = False, np.nan, np.nan
            continue
        if book.trade is not None:
            continue
        side = int(d[i]) if np.isfinite(d[i]) else 0
        if side == 0 or not (np.isfinite(er[i]) and er[i] > p.er_min) or not np.isfinite(atr_l[i]):
            in_pb = False
            continue
        if side != pb_side:
            in_pb, pb_side = False, side
        levels = [x for x in (av[i], emaf[i]) if np.isfinite(x)]
        if not levels:
            continue
        near = any((l[i] <= x + p.near_atr * atr_l[i]) if side > 0 else (h[i] >= x - p.near_atr * atr_l[i]) for x in levels)
        if not in_pb and near:
            in_pb, pb_ext, dipped, reset = True, (l[i] if side > 0 else h[i]), False, False
        if not in_pb:
            continue
        pb_ext = min(pb_ext, l[i]) if side > 0 else max(pb_ext, h[i])
        if side > 0:
            if rsi[i] < p.rsi_reset:
                dipped = True
            elif dipped:
                reset = True
            go = reset and c[i] > h[i - 1]
        else:
            if rsi[i] > 100 - p.rsi_reset:
                dipped = True
            elif dipped:
                reset = True
            go = reset and c[i] < l[i - 1]
        if go:
            stop = pb_ext - side * p.stop_atr * atr_l[i]
            if book.enter(t, side, c[i], stop, "avwap_pullback"):
                partial, best = False, (h[i] if side > 0 else l[i])
                trail = np.nan
            in_pb = False
    if book.trade is not None:
        book.exit(idx[-1], c[-1], "data_end")
    return book


def crossover_benchmark(ltf: pd.DataFrame, p: Params, *, cost: str = "kraken_pro_taker", long_only: bool = False,
                        start=None) -> pd.Series:
    """The doc's benchmark: the plain higher-timeframe EMA crossover, always in the market (flat
    instead of short when long-only), fully invested. Returns daily returns."""
    h = resample(ltf, p.htf_min)
    ef = h["close"].ewm(span=p.ema_fast, adjust=False, min_periods=p.ema_fast).mean()
    es = h["close"].ewm(span=p.ema_slow, adjust=False, min_periods=p.ema_slow).mean()
    pos = np.sign(ef - es).shift(1).fillna(0)  # decided at the bar close, held over the next bar
    if long_only:
        pos = pos.clip(lower=0)
    ret = h["close"].pct_change().fillna(0) * pos
    c = COSTS[cost]
    ret -= pos.diff().abs().fillna(0) * (c["fee"] + c["slip"]) / 1e4
    if start is not None:
        ret = ret[ret.index >= pd.Timestamp(start)]
    return (1 + ret).groupby(ret.index.floor("D")).prod() - 1


def buy_and_hold(ltf: pd.DataFrame, start=None) -> pd.Series:
    d = ltf["close"].groupby(ltf.index.floor("D")).last()
    r = d.pct_change().fillna(0)
    return r[r.index >= pd.Timestamp(start).floor("D")] if start is not None else r
