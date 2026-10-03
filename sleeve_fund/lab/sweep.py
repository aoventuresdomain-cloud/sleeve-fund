"""Strategy 3 of the v4 doc: liquidity-sweep reversal, core levels only.

Pre-registered levels (to limit data snooping, as the doc asks): the prior session's high and
low, and the Asia session's high and low (00:00 to 08:00 UTC, usable once that window closes).
A sweep pokes beyond a level by a fraction of daily ATR, fails by closing back inside within a
few bars, and needs one confirmation (volume, RSI divergence, or a close back inside the
1-sigma VWAP band). The trade targets session VWAP. Skipped when the day is a trend day in the
sweep's direction, because then it is a breakout, not a sweep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sleeve_fund.lab import blocks
from sleeve_fund.lab.sim import Book


@dataclass(frozen=True)
class Params:
    anchor: str = "utc"
    sweep_atr: float = 0.1
    fail_bars: int = 2
    vol_mult: float = 1.5
    rsi_len_min: int = 210  # RSI(14) on 15-minute bars
    stop_atr: float = 0.1
    time_stop_min: int = 12 * 60
    trend_filter: bool = True
    opposite: bool = False  # benchmark: trade the breakout instead of the reversal
    risk: float = 0.005
    max_leverage: float = 4.0

    def as_dict(self) -> dict:
        return asdict(self)


def prepare(bars: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = blocks.with_session_blocks(bars, p.anchor)
    out["label"] = blocks.classify(out)
    bar = blocks.bar_minutes(bars)
    out["rsi"] = blocks.rsi(out["close"], blocks.bars_for(p.rsi_len_min, bar))
    out["vol_avg"] = out["volume"].rolling(20).mean().shift(1)
    # Asia session range, usable from 08:00 UTC that day.
    day = out.index.floor("D")
    asia = (out.index - day) < pd.Timedelta(hours=8)
    ah = out["high"].where(asia).groupby(day).transform("max")
    al = out["low"].where(asia).groupby(day).transform("min")
    out["asia_high"] = ah.where(~asia)
    out["asia_low"] = al.where(~asia)
    return out


def run(bars: pd.DataFrame, p: Params, *, cost: str = "kraken_pro_taker", long_only: bool = False, start=None,
        prepared: pd.DataFrame | None = None) -> Book:
    out = prepared if prepared is not None else prepare(bars, p)
    book = Book(risk=p.risk, max_leverage=p.max_leverage, long_only=long_only, cost=cost)
    bar = blocks.bar_minutes(bars)
    idx = out.index
    o, h, l, c = (out[k].to_numpy() for k in ("open", "high", "low", "close"))
    vw, sd, atr = out["vwap"].to_numpy(), out["vwap_sd"].to_numpy(), out["atr_d"].to_numpy()
    rsi, vol, vavg = out["rsi"].to_numpy(), out["volume"].to_numpy(), out["vol_avg"].to_numpy()
    lab, sess = out["label"].to_numpy(), out["session"].to_numpy()
    levels = {k: out[k].to_numpy() for k in ("prior_high", "prior_low", "asia_high", "asia_low")}
    i0 = max(1, int(np.searchsorted(idx, pd.Timestamp(start))) if start is not None else 1)
    pending: dict = {}  # level name -> (bars since sweep, extreme, confirmed flags)
    used: set = set()
    entered_i, partial = 0, False
    sess_hist: list = []
    for i in range(i0, len(idx)):
        t = idx[i]
        if sess[i] != sess[i - 1]:
            pending, used, sess_hist = {}, set(), []
        sess_hist.append((h[i], l[i], rsi[i]))
        tr = book.trade
        if tr is not None and tr.entry_time < t:
            if not book.check_stop(t, o[i], h[i], l[i]):
                side = tr.side
                if not p.opposite:
                    if not partial and ((side < 0 and l[i] <= vw[i]) or (side > 0 and h[i] >= vw[i])):
                        book.exit(t, vw[i], "target_vwap", 0.5)
                        partial = True
                    t2 = vw[i] + side * sd[i]
                    if book.trade is not None and partial and ((side < 0 and l[i] <= t2) or (side > 0 and h[i] >= t2)):
                        book.exit(t, t2, "target_band")
                if book.trade is not None and (i - entered_i) * bar >= p.time_stop_min:
                    book.exit(t, c[i], "time_stop")
            if book.trade is None:
                partial = False
            continue
        if not (np.isfinite(atr[i]) and np.isfinite(vw[i])):
            continue
        for name, arr in levels.items():
            lvl = arr[i]
            if not np.isfinite(lvl) or name in used:
                continue
            up = name.endswith("high")
            poke = (h[i] > lvl + p.sweep_atr * atr[i]) if up else (l[i] < lvl - p.sweep_atr * atr[i])
            if poke and name not in pending:
                pending[name] = [0, h[i] if up else l[i], set()]
            if name not in pending:
                continue
            st = pending[name]
            st[1] = max(st[1], h[i]) if up else min(st[1], l[i])
            if np.isfinite(vavg[i]) and vol[i] > p.vol_mult * vavg[i]:
                st[2].add("volume")
            if _diverges(up, sess_hist, rsi[i]):
                st[2].add("divergence")
            back = (c[i] < lvl) if up else (c[i] > lvl)
            if back and ((up and c[i] < vw[i] + sd[i]) or (not up and c[i] > vw[i] - sd[i])):
                st[2].add("inside_band")
            if back:
                del pending[name]
                used.add(name)
                trend_against = p.trend_filter and lab[i] == (blocks.TREND_UP if up else blocks.TREND_DOWN)
                if st[2] and not trend_against and book.trade is None:
                    side = -1 if up else 1
                    stop = st[1] + (p.stop_atr * atr[i] if up else -p.stop_atr * atr[i])
                    if p.opposite:  # breakout benchmark: same risk distance, the other way
                        side, stop = -side, c[i] - (-side) * abs(stop - c[i])
                    if book.enter(t, side, c[i], stop, "sweep_reversal" if not p.opposite else "breakout"):
                        entered_i, partial = i, False
                continue
            st[0] += 1
            if st[0] > p.fail_bars:
                del pending[name]
                used.add(name)  # held beyond the level: a real break, not a sweep
    if book.trade is not None:
        book.exit(idx[-1], c[-1], "data_end")
    return book


def _diverges(up: bool, hist: list, rsi_now: float, gap: int = 3) -> bool:
    prior = hist[:-gap]
    if not prior or not np.isfinite(rsi_now):
        return False
    if up:
        j = max(range(len(prior)), key=lambda k: prior[k][0])
        return hist[-1][0] > prior[j][0] and np.isfinite(prior[j][2]) and rsi_now < prior[j][2]
    j = min(range(len(prior)), key=lambda k: prior[k][1])
    return hist[-1][1] < prior[j][1] and np.isfinite(prior[j][2]) and rsi_now > prior[j][2]
