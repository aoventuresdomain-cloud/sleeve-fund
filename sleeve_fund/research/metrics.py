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


def round_trips(fills: pd.DataFrame) -> list[float]:
    """Return per round trip (buy then sell), gross of fees, from a fills report."""
    if fills is None or fills.empty:
        return []
    f = fills.sort_values("ts_last")
    trips, entry = [], None
    for side, px in zip(f["side"].astype(str), f["avg_px"].astype(float)):
        if side.endswith("BUY") and entry is None:
            entry = px
        elif side.endswith("SELL") and entry is not None:
            trips.append(px / entry - 1)
            entry = None
    return trips


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
