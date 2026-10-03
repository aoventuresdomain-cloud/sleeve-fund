"""Performance metrics. Crypto trades every day, so annualisation uses 365."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365
EULER_GAMMA = 0.5772156649


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def summary(returns: pd.Series) -> dict[str, float]:
    """Headline metrics from a daily return series."""
    r = returns.dropna()
    n = len(r)
    if n < 2:
        raise ValueError("need at least 2 daily returns")
    equity = (1 + r).cumprod()
    years = n / PERIODS_PER_YEAR
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))
    mean = float(r.mean() * PERIODS_PER_YEAR)
    downside = r[r < 0]
    dvol = float(math.sqrt((downside**2).sum() / n) * math.sqrt(PERIODS_PER_YEAR))
    mdd = max_drawdown(pd.concat([pd.Series([1.0]), equity], ignore_index=True))
    return {
        "days": n,
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": mean / vol if vol > 0 else 0.0,
        "sortino": mean / dvol if dvol > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else 0.0,
    }


def _money(entry) -> float:
    """Sum a fills-report commissions cell (Money string or list of them)."""
    if entry is None or (isinstance(entry, float) and math.isnan(entry)):
        return 0.0
    return sum(float(str(m).split()[0]) for m in (entry if isinstance(entry, (list, tuple)) else [entry]))


def fills_to_rows(fills: pd.DataFrame) -> list[dict]:
    """Backtest fills report -> the same shape the live journal stores (side, qty, price, fee)."""
    if fills is None or fills.empty:
        return []
    f = fills.sort_values("ts_last")
    fees = f["commissions"] if "commissions" in f else [0.0] * len(f)
    return [
        {"side": "BUY" if str(side).endswith("BUY") else "SELL", "qty": float(q), "price": float(px), "fee": _money(c)}
        for side, q, px, c in zip(f["side"], f["filled_qty"], f["avg_px"], fees)
    ]


def trades(rows: list[dict]) -> list[dict]:
    """Closed round trips (flat -> long -> flat) with P&L after fees, oldest first.

    rows: fills in time order with side, qty, price, fee (quote currency). Partial
    fills are fine: a trip closes when the coin position returns to (about) zero.
    """
    out, qty, cost, proceeds, fees = [], 0.0, 0.0, 0.0, 0.0
    bought = sold = 0.0
    opened = entry_order = None
    for r in rows:
        if r["side"] == "BUY":
            if qty <= 1e-12:
                opened, entry_order = r.get("ts"), r.get("order_id")
            qty += r["qty"]
            bought += r["qty"]
            cost += r["qty"] * r["price"]
        else:
            if qty <= 0:
                continue  # a sell with nothing open (e.g. journal started mid-trip)
            qty -= r["qty"]
            sold += r["qty"]
            proceeds += r["qty"] * r["price"]
        fees += r["fee"]
        if cost and qty <= 1e-12:
            pnl = proceeds - cost - fees
            out.append({"pnl": pnl, "ret": pnl / cost, "cost": cost, "fees": fees, "qty": bought,
                        "entry_px": cost / bought, "exit_px": proceeds / sold if sold else float("nan"),
                        "opened": opened, "closed": r.get("ts"),
                        # Journal order ids, so the dashboard can show why the trade was opened and closed.
                        "entry_order": entry_order, "exit_order": r.get("order_id")})
            qty, cost, proceeds, fees, bought, sold = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return out


def trade_stats(trips: list[dict]) -> dict:
    """Win rate, average win/loss, expectancy and profit factor, all after fees."""
    n = len(trips)
    wins = [t for t in trips if t["pnl"] > 0]
    losses = [t for t in trips if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in losses)
    avg = lambda xs, k: sum(x[k] for x in xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n if n else float("nan"),
        "pnl": sum(t["pnl"] for t in trips),
        "avg_win": avg(wins, "ret"),
        "avg_loss": avg(losses, "ret"),
        "expectancy": avg(trips, "ret"),  # average return per trade after fees
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else float("nan"),
        "best": max((t["ret"] for t in trips), default=float("nan")),
        "worst": min((t["ret"] for t in trips), default=float("nan")),
    }


def round_trips(fills: pd.DataFrame) -> list[float]:
    """Return per round trip after fees, from a backtest fills report."""
    return [t["ret"] for t in trades(fills_to_rows(fills))]


def turnover_per_year(fills: pd.DataFrame, equity: pd.Series) -> float:
    """Traded notional per year as a multiple of average equity."""
    if fills is None or fills.empty:
        return 0.0
    notional = (fills["filled_qty"].astype(float) * fills["avg_px"].astype(float)).sum()
    years = max(len(equity) / PERIODS_PER_YEAR, 1e-9)
    return float(notional / equity.mean() / years)


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Sharpe you'd expect from the best of n_trials skill-less strategies.

    Bailey and Lopez de Prado (2014). This is the hurdle the idea counter raises:
    the more variants tried, the higher the best one's Sharpe must be to mean anything.
    """
    if n_trials <= 1:
        return 0.0
    z = NormalDist()
    return sharpe_std * (
        (1 - EULER_GAMMA) * z.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * z.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe_probability(returns: pd.Series, n_trials: int, trial_sharpes: list[float]) -> float:
    """Probability the true Sharpe beats the best-of-n luck hurdle (daily units internally)."""
    r = returns.dropna()
    n = len(r)
    if n < 30 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    daily_trials = np.asarray(trial_sharpes, dtype=float) / math.sqrt(PERIODS_PER_YEAR)
    sr_std = float(daily_trials.std(ddof=1)) if len(daily_trials) > 1 else 0.0
    hurdle = expected_max_sharpe(n_trials, sr_std)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(NormalDist().cdf((sr - hurdle) * math.sqrt(n - 1) / denom))
