"""Strategies 1 and 2 of the v4 doc, joined by the day-type classifier.

1. Trend-day VWAP pullback: on a confirmed trend day, buy the first orderly pullback to VWAP
   once short-term RSI resets and turns (mirror for down days).
2. Balance-day VWAP band reversion: on a range day, fade a stretch to the outer band when at
   least two exhaustion signs appear, targeting the 1-sigma band and then VWAP.

The day type is read at the decision time and re-checked hourly, from bars already closed.
Lookbacks given in the doc as bars of 5 minutes are held constant in clock time, so the same
parameters mean the same thing on other candle sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from sleeve_fund.lab import blocks
from sleeve_fund.lab.sim import Book

SESSION_MIN = 24 * 60


@dataclass(frozen=True)
class Params:
    anchor: str = "utc"
    decide_at: int = 60
    # classifier
    side_share: float = 0.8
    trend_slope: float = 0.15
    balance_slope: float = 0.05
    relvol_min: float = 1.2
    use_classifier: bool = True
    # strategy 1
    s1: bool = True
    pullback_band: float = 0.5  # touch VWAP + this many sigma counts as a pullback
    rsi_fast_min: int = 35  # RSI(7) on 5-minute bars
    rsi_reset: float = 40.0
    no_entry_last_min: int = 90
    s1_partial_r: float = 1.5
    max_entries: int = 2
    # strategy 2
    s2: bool = True
    entry_band: float = 2.0
    rsi_slow_min: int = 70  # RSI(14) on 5-minute bars
    vol_mult: float = 2.0
    need_exhaustion: int = 2
    time_stop_min: int = 120
    # sizing
    risk: float = 0.005
    max_leverage: float = 4.0

    def as_dict(self) -> dict:
        return asdict(self)


def prepare(bars: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = blocks.with_session_blocks(bars, p.anchor)
    label = blocks.classify(out, side_share=p.side_share, trend_slope=p.trend_slope, balance_slope=p.balance_slope,
                            relvol_min=p.relvol_min)
    if not p.use_classifier:  # benchmark: every day is tradable; strategy 1 follows the side of VWAP
        side = np.where(out["above_share"] >= 0.5, blocks.TREND_UP, blocks.TREND_DOWN)
        label = pd.Series(side, index=out.index).where(out["atr_d"].notna(), blocks.UNCLEAR)
    check = (out["elapsed"] >= p.decide_at) & ((out["elapsed"] - p.decide_at) % 60 == 0)
    day = label.where(check).groupby(out["session"]).ffill().fillna(blocks.UNCLEAR)
    out["day_type"] = day
    out["balance_ok"] = (day == blocks.BALANCE) if p.use_classifier else out["atr_d"].notna()
    bar = blocks.bar_minutes(bars)
    out["rsi_fast"] = blocks.rsi(out["close"], blocks.bars_for(p.rsi_fast_min, bar))
    out["rsi_slow"] = blocks.rsi(out["close"], blocks.bars_for(p.rsi_slow_min, bar))
    out["vol_avg"] = out["volume"].rolling(blocks.bars_for(20 * 5, bar)).mean().shift(1)
    out["last_bar"] = out["session"].ne(out["session"].shift(-1))
    return out


def run(bars: pd.DataFrame, p: Params, *, cost: str = "kraken_pro_taker", long_only: bool = False,
        start=None, prepared: pd.DataFrame | None = None) -> Book:
    """Trade one instrument. Bars before `start` only warm up the indicators."""
    out = prepared if prepared is not None else prepare(bars, p)
    book = Book(risk=p.risk, max_leverage=p.max_leverage, long_only=long_only, cost=cost)
    bar = blocks.bar_minutes(bars)
    idx = out.index
    o, h, l, c = (out[k].to_numpy() for k in ("open", "high", "low", "close"))
    vw, sd = out["vwap"].to_numpy(), out["vwap_sd"].to_numpy()
    rf, rs = out["rsi_fast"].to_numpy(), out["rsi_slow"].to_numpy()
    vol, vavg = out["volume"].to_numpy(), out["vol_avg"].to_numpy()
    el, day = out["elapsed"].to_numpy(), out["day_type"].to_numpy()
    bal = out["balance_ok"].to_numpy()
    last, sess = out["last_bar"].to_numpy(), out["session"].to_numpy()
    first_i = int(np.searchsorted(idx, pd.Timestamp(start))) if start is not None else 1
    first_i = max(first_i, 1)

    entries = 0
    s1 = {1: _S1(), -1: _S1()}
    s2 = {1: _S2(), -1: _S2()}
    sess_hi_hist: list = []  # (high, rsi) per bar this session, for divergence
    kind = None  # which strategy holds the open trade
    partial = False
    entered_el = 0
    for i in range(first_i, len(idx)):
        t = idx[i]
        if sess[i] != sess[i - 1]:
            entries = 0
            s1 = {1: _S1(), -1: _S1()}
            s2 = {1: _S2(), -1: _S2()}
            sess_hi_hist = []
        sess_hi_hist.append((h[i], l[i], rs[i]))
        tr = book.trade
        # --- manage the open trade (stop first, then targets and trailing exits) ---
        if tr is not None and tr.entry_time < t:
            if book.check_stop(t, o[i], h[i], l[i]):
                tr = None
            elif kind == 1:
                r = tr.risk_per_unit
                tgt = tr.entry_px + tr.side * p.s1_partial_r * r
                if not partial and ((tr.side > 0 and h[i] >= tgt) or (tr.side < 0 and l[i] <= tgt)):
                    book.exit(t, tgt, "target_1.5R", 0.5)
                    partial = True
                if book.trade is not None and ((tr.side > 0 and c[i] < vw[i]) or (tr.side < 0 and c[i] > vw[i])):
                    book.exit(t, c[i], "vwap_trail")
            elif kind == 2:
                t1 = vw[i] - tr.side * sd[i]  # long: VWAP - 1 sigma; short: VWAP + 1 sigma
                if not partial and ((tr.side > 0 and h[i] >= t1) or (tr.side < 0 and l[i] <= t1)):
                    book.exit(t, t1, "target_1sigma", 0.5)
                    partial = True
                if book.trade is not None and ((tr.side > 0 and h[i] >= vw[i]) or (tr.side < 0 and l[i] <= vw[i])):
                    book.exit(t, vw[i], "target_vwap")
                if book.trade is not None and (
                        (tr.side < 0 and c[i] > vw[i] + 3 * sd[i]) or (tr.side > 0 and c[i] < vw[i] - 3 * sd[i])):
                    book.exit(t, c[i], "beyond_3sigma")
                if book.trade is not None and el[i] - entered_el >= p.time_stop_min:
                    book.exit(t, c[i], "time_stop")
                if book.trade is not None and p.use_classifier and day[i] in (blocks.TREND_UP, blocks.TREND_DOWN):
                    book.exit(t, c[i], "turned_trend")
            if book.trade is not None and last[i]:
                book.exit(t, c[i], "session_end")
            if book.trade is None:
                kind, partial = None, False
            continue
        if book.trade is not None:
            continue
        if not (np.isfinite(vw[i]) and np.isfinite(sd[i]) and sd[i] > 0):
            continue
        can_enter = entries < p.max_entries and el[i] <= SESSION_MIN - p.no_entry_last_min and not last[i]
        # --- strategy 1: trend-day pullback ---
        if p.s1 and day[i] in (blocks.TREND_UP, blocks.TREND_DOWN):
            side = 1 if day[i] == blocks.TREND_UP else -1
            st = s1[side]
            sig = st.step(side, h[i], l[i], c[i], h[i - 1], l[i - 1], vw[i], sd[i], rf[i], p)
            if sig is not None and can_enter:
                stop = min(sig, vw[i] - 0.5 * sd[i]) if side > 0 else max(sig, vw[i] + 0.5 * sd[i])
                if book.enter(t, side, c[i], stop, "trend_pullback"):
                    kind, partial, entered_el = 1, False, el[i]
                    entries += 1
                    continue
        # --- strategy 2: balance-day band fade ---
        if p.s2 and bal[i]:
            for side in (-1, 1):  # -1 fades the upper band, +1 the lower
                st = s2[side]
                sig = st.step(side, i, h, l, c, vw, sd, rs, vol, vavg, sess_hi_hist, p)
                if sig is not None and can_enter and book.trade is None:
                    if book.enter(t, side, c[i], sig, "band_fade"):
                        kind, partial, entered_el = 2, False, el[i]
                        entries += 1
    if book.trade is not None:
        book.exit(idx[-1], c[-1], "data_end")
    return book


class _S1:
    """Pullback state for one direction of strategy 1."""

    def __init__(self) -> None:
        self.armed = False
        self.in_pb = False
        self.extreme = np.nan
        self.dipped = False
        self.reset = False

    def step(self, side, hi, lo, cl, prev_hi, prev_lo, vw, sd, rsi_f, p):
        band = vw + side * p.pullback_band * sd
        if not self.in_pb:
            if (side > 0 and cl > band) or (side < 0 and cl < band):
                self.armed = True
            if self.armed and ((side > 0 and lo <= band) or (side < 0 and hi >= band)):
                self.in_pb, self.extreme, self.dipped, self.reset = True, (lo if side > 0 else hi), False, False
            else:
                return None
        self.extreme = min(self.extreme, lo) if side > 0 else max(self.extreme, hi)
        if (side > 0 and cl < vw - 0.25 * sd) or (side < 0 and cl > vw + 0.25 * sd):
            self.in_pb = self.armed = False  # too deep: not an orderly pullback
            return None
        if side > 0:
            if rsi_f < p.rsi_reset:
                self.dipped = True
            elif self.dipped:
                self.reset = True
            trigger = self.reset and cl > prev_hi
        else:
            if rsi_f > 100 - p.rsi_reset:
                self.dipped = True
            elif self.dipped:
                self.reset = True
            trigger = self.reset and cl < prev_lo
        if trigger:
            ext = self.extreme
            self.in_pb = self.armed = False
            return ext
        return None


class _S2:
    """Stretch-and-exhaustion state for one side of strategy 2 (side -1 = short at the upper band)."""

    def __init__(self) -> None:
        self.active = False
        self.extreme = np.nan
        self.flags: set = set()

    def step(self, side, i, h, l, c, vw, sd, rs, vol, vavg, hist, p):
        band = vw[i] - side * p.entry_band * sd[i]  # short: VWAP + 2 sigma; long: VWAP - 2 sigma
        stretched = (h[i] >= band) if side < 0 else (l[i] <= band)
        if stretched:
            if not self.active:
                self.active, self.extreme, self.flags = True, (h[i] if side < 0 else l[i]), set()
            self.extreme = max(self.extreme, h[i]) if side < 0 else min(self.extreme, l[i])
            rng = h[i] - l[i]
            pos = (c[i] - l[i]) / rng if rng > 0 else 0.5
            if np.isfinite(vavg[i]) and vol[i] > p.vol_mult * vavg[i] and ((side < 0 and pos < 0.5) or (side > 0 and pos > 0.5)):
                self.flags.add("volume")
            if _divergence(side, hist, h[i], l[i], rs[i]):
                self.flags.add("divergence")
        if self.active and i > 0 and ((side < 0 and rs[i - 1] >= 70 > rs[i]) or (side > 0 and rs[i - 1] <= 30 < rs[i])):
            self.flags.add("rsi_cross")
        if self.active and not stretched:
            closed_back = (c[i] < band) if side < 0 else (c[i] > band)
            ok = closed_back and len(self.flags) >= p.need_exhaustion
            ext = self.extreme
            self.active = False
            if ok:
                return ext + 0.25 * sd[i] if side < 0 else ext - 0.25 * sd[i]
        return None


def _divergence(side, hist, hi, lo, rsi_now, gap: int = 3) -> bool:
    """Price beyond the previous swing extreme of the session while RSI is not."""
    prior = hist[:-gap] if len(hist) > gap else []
    if not prior or not np.isfinite(rsi_now):
        return False
    if side < 0:
        j = max(range(len(prior)), key=lambda k: prior[k][0])
        return hi > prior[j][0] and np.isfinite(prior[j][2]) and rsi_now < prior[j][2]
    j = min(range(len(prior)), key=lambda k: prior[k][1])
    return lo < prior[j][1] and np.isfinite(prior[j][2]) and rsi_now > prior[j][2]


def intraday_momentum(bars: pd.DataFrame, *, anchor: str = "utc", decide_at: int = 60, cost: str = "kraken_pro_taker",
                      long_only: bool = False, start=None) -> pd.DataFrame:
    """The doc's benchmark: from the decision time to the session end, hold in the direction of
    the session so far. Fully invested (1x), one round trip a day."""
    from sleeve_fund.lab.sim import COSTS

    out = blocks.with_session_blocks(bars, anchor)
    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    c = COSTS[cost]
    per_side = (c["fee"] + c["slip"]) / 1e4
    rows = []
    for sid, s in out.groupby("session", sort=True):
        d = s[s["elapsed"] == decide_at]
        if d.empty or s["elapsed"].iloc[-1] < SESSION_MIN - 60:
            continue
        side = 1 if d["close"].iloc[0] >= s["open"].iloc[0] else -1
        if long_only and side < 0:
            continue
        ret = side * (s["close"].iloc[-1] / d["close"].iloc[0] - 1) - 2 * per_side
        rows.append({"entry_time": d.index[0], "exit_time": s.index[-1], "side": side, "ret": ret, "pnl": ret,
                     "r": np.nan, "fees": 2 * per_side, "reason": "intraday_momentum", "exit_why": "session_end"})
    return pd.DataFrame(rows)
