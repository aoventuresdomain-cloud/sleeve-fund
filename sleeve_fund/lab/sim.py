"""Trade bookkeeping shared by the lab strategies.

Strategies decide; this module turns their decisions into trades and an equity curve with
honest fills and costs:

- entries fill at the signal bar's close plus slippage (the doc's "enter on the trigger close"),
- stops fill at the stop price, or at the bar's open if the bar gapped through it, plus slippage,
- if a bar touches both the stop and a target, the stop is assumed to come first,
- every fill pays the fee for its side; costs are set per scenario so results can be read at
  retail venue fees and at the low-cost levels the doc assumes.

Sizing follows the doc: risk a fixed share of equity per trade (distance to the stop), capped by
a leverage limit. Long-only runs (spot, no borrowing) cap leverage at 1 and drop short signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Per-side costs in basis points. "plan" is the project plan's Kraken assumption (taker every fill).
COSTS = {
    "plan_kraken_taker": {"fee": 80.0, "slip": 2.0},
    "kraken_pro_taker": {"fee": 40.0, "slip": 2.0},
    "high_volume_taker": {"fee": 10.0, "slip": 2.0},
    "institutional": {"fee": 2.0, "slip": 1.0},
    "zero": {"fee": 0.0, "slip": 0.0},
}


@dataclass
class Trade:
    side: int  # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_px: float
    stop_px: float
    qty: float
    equity_at_entry: float
    reason: str = ""
    exits: list = field(default_factory=list)  # (time, px, qty, why)
    fees: float = 0.0
    initial_stop: float = float("nan")  # the stop at entry; R is measured against it even if the stop trails

    def __post_init__(self) -> None:
        if self.initial_stop != self.initial_stop:
            self.initial_stop = self.stop_px

    @property
    def open_qty(self) -> float:
        return self.qty - sum(q for _, _, q, _ in self.exits)

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_px - self.initial_stop)

    @property
    def pnl(self) -> float:
        gross = sum(self.side * (px - self.entry_px) * q for _, px, q, _ in self.exits)
        return gross - self.fees

    @property
    def r_multiple(self) -> float:
        risk = self.risk_per_unit * self.qty
        return self.pnl / risk if risk else 0.0

    @property
    def exit_time(self) -> pd.Timestamp | None:
        return self.exits[-1][0] if self.exits else None


class Book:
    """One instrument's account: at most one open trade at a time, compounding equity."""

    def __init__(self, *, equity: float = 100_000.0, risk: float = 0.005, max_leverage: float = 4.0,
                 long_only: bool = False, cost: str = "kraken_pro_taker") -> None:
        self.equity = equity
        self.start = equity
        self.risk = risk
        self.max_leverage = 1.0 if long_only else max_leverage
        self.long_only = long_only
        c = COSTS[cost]
        self.fee = c["fee"] / 1e4
        self.slip = c["slip"] / 1e4
        self.trade: Trade | None = None
        self.trades: list[Trade] = []

    def _fill(self, px: float, side: int) -> float:
        return px * (1 + self.slip * side)

    def enter(self, t, side: int, px: float, stop: float, reason: str = "") -> bool:
        if self.trade is not None or (self.long_only and side < 0):
            return False
        if not (stop > 0) or (side > 0 and stop >= px) or (side < 0 and stop <= px):
            return False
        fill = self._fill(px, side)
        per_unit = abs(fill - stop)
        qty = min(self.equity * self.risk / per_unit, self.equity * self.max_leverage / fill)
        if qty <= 0 or not math.isfinite(qty):
            return False
        self.trade = Trade(side, t, fill, stop, qty, self.equity, reason)
        self.trade.fees += fill * qty * self.fee
        return True

    def exit(self, t, px: float, why: str, fraction: float = 1.0) -> None:
        tr = self.trade
        if tr is None:
            return
        q = tr.open_qty if fraction >= 1 else tr.qty * fraction
        q = min(q, tr.open_qty)
        fill = self._fill(px, -tr.side)
        tr.exits.append((t, fill, q, why))
        tr.fees += fill * q * self.fee
        if tr.open_qty <= tr.qty * 1e-9:
            self.equity += tr.pnl
            self.trades.append(tr)
            self.trade = None

    def check_stop(self, t, bar_open: float, high: float, low: float) -> bool:
        """Stop first: if this bar reached the stop, exit there (or at the open on a gap)."""
        tr = self.trade
        if tr is None:
            return False
        if tr.side > 0 and low <= tr.stop_px:
            self.exit(t, min(tr.stop_px, bar_open), "stop")
            return True
        if tr.side < 0 and high >= tr.stop_px:
            self.exit(t, max(tr.stop_px, bar_open), "stop")
            return True
        return False


def trades_frame(trades: list[Trade], instrument: str = "") -> pd.DataFrame:
    return pd.DataFrame([{
        "instrument": instrument, "side": t.side, "entry_time": t.entry_time, "exit_time": t.exit_time,
        "entry_px": t.entry_px, "stop_px": t.stop_px, "pnl": t.pnl, "ret": t.pnl / t.equity_at_entry,
        "r": t.r_multiple, "fees": t.fees, "reason": t.reason, "exit_why": t.exits[-1][3] if t.exits else "",
    } for t in trades])


def daily_returns(tf: pd.DataFrame, start, end) -> pd.Series:
    """Each trade's return booked on its exit day; days without exits are flat (0)."""
    days = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).floor("D"), freq="D")
    if tf.empty:
        return pd.Series(0.0, index=days)
    r = (1 + tf.set_index(pd.to_datetime(tf["exit_time"]).dt.floor("D"))["ret"]).groupby(level=0).prod() - 1
    return r.reindex(days, fill_value=0.0)


def stats(tf: pd.DataFrame, daily: pd.Series) -> dict:
    """Headline numbers. Sharpe and volatility use 365 days (the instruments trade daily)."""
    n = len(tf)
    eq = (1 + daily).cumprod()
    years = max(len(daily) / 365.0, 1 / 365)
    vol = float(daily.std(ddof=1) * math.sqrt(365)) if len(daily) > 1 else float("nan")
    out = {
        "trades": n,
        "trades_per_month": n / (len(daily) / 30.44) if len(daily) else 0.0,
        "total_return": float(eq.iloc[-1] - 1) if len(eq) else 0.0,
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if len(eq) and eq.iloc[-1] > 0 else -1.0,
        "sharpe": float(daily.mean() / daily.std(ddof=1) * math.sqrt(365)) if len(daily) > 1 and daily.std() > 0 else 0.0,
        "vol": vol,
        "max_dd": float((1 - eq / eq.cummax()).max()) if len(eq) else 0.0,
        "win_rate": float((tf["pnl"] > 0).mean()) if n else float("nan"),
        "avg_r": float(tf["r"].mean()) if n else float("nan"),
        "expectancy": float(tf["ret"].mean()) if n else float("nan"),
        "profit_factor": (float(tf.loc[tf.pnl > 0, "pnl"].sum() / -tf.loc[tf.pnl < 0, "pnl"].sum())
                          if n and (tf.pnl < 0).any() else float("nan")),
        "fees_share_of_gross": (float(tf["fees"].sum() / (tf["pnl"] + tf["fees"]).abs().sum())
                                if n and (tf["pnl"] + tf["fees"]).abs().sum() else float("nan")),
    }
    return out


def basket_check(per_market: dict[str, pd.DataFrame]) -> dict:
    """The doc's cross-asset pass mark: positive net expectancy on at least 60% of markets, and no
    market contributes more than 30% of total profit."""
    exp = {k: (v["ret"].mean() if len(v) else float("nan")) for k, v in per_market.items()}
    pnl = {k: (np.log1p(v["ret"]).sum() if len(v) else 0.0) for k, v in per_market.items()}
    traded = [k for k in exp if exp[k] == exp[k]]
    positive = [k for k in traded if exp[k] > 0]
    total_profit = sum(p for p in pnl.values() if p > 0)
    top = max((p / total_profit for p in pnl.values() if p > 0), default=float("nan")) if total_profit > 0 else float("nan")
    share_ok = len(positive) >= 0.6 * len(per_market)
    conc_ok = total_profit > 0 and top <= 0.30
    return {"positive_markets": len(positive), "markets": len(per_market), "largest_profit_share": top,
            "passes": bool(share_ok and conc_ok)}
