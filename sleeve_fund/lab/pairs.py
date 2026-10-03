"""Strategy 5 of the v4 doc: regime-switching pairs (ETH/BTC and any pair).

On hourly bars: a Kalman filter tracks the hedge ratio between the two log prices; the spread
is log(A) - beta * log(B). A spread that is mean-reverting (Hurst below 0.45, half-life 6 hours
to 10 days, and cointegrated with correlation above 0.7 at the monthly check) is faded at
two standard deviations when its RSI turns; a trending spread (Hurst above 0.55 and EMAs
apart) is followed on an EMA crossover; anything else is left alone.

Simplifications in this version, stated in the report: stops and exits act on hourly closes;
the trend-regime entry is the crossover itself (not the pullback to the spread's anchored VWAP);
the short leg pays a flat borrow or funding cost per year. Pairs need shorting, so there is no
long-only version.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sleeve_fund.lab import blocks
from sleeve_fund.lab.sim import COSTS

ADF_CRIT_10 = -3.04  # Engle-Granger, two variables, 10% (MacKinnon), large samples


@dataclass(frozen=True)
class Params:
    z_window_h: int = 30 * 24
    z_entry: float = 2.0
    z_stop: float = 3.5
    hurst_window: int = 100
    revert_hurst: float = 0.45
    trend_hurst: float = 0.55
    hl_min_h: float = 6.0
    hl_max_h: float = 240.0
    corr_min: float = 0.7
    coint_window_d: int = 180
    corr_window_d: int = 90
    ema_fast: int = 20
    ema_slow: int = 100
    trend_gap_sd: float = 0.5
    kalman_delta: float = 1e-7
    kalman_ve: float = 1e-3
    regime_switch: bool = True  # False = static-hedge benchmark: always the revert rules
    allow_trend: bool = True
    risk: float = 0.0075
    max_gross: float = 3.0
    borrow_pa: float = 0.05  # cost of carrying the short leg, per year

    def as_dict(self) -> dict:
        return asdict(self)


def kalman_beta(y: np.ndarray, x: np.ndarray, delta: float, ve: float) -> tuple[np.ndarray, np.ndarray]:
    """Time-varying beta and intercept of y on x, each estimate using only data before that bar."""
    n = len(y)
    beta = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    theta = np.zeros(2)
    P = np.eye(2)
    Q = delta / (1 - delta) * np.eye(2)
    for t in range(n):
        F = np.array([x[t], 1.0])
        P = P + Q
        yhat = F @ theta
        S = F @ P @ F + ve
        K = P @ F / S
        beta[t], alpha[t] = theta  # the prior estimate: known before this bar's value is used
        theta = theta + K * (y[t] - yhat)
        P = P - np.outer(K, F) @ P
    return beta, alpha


def hurst(series: pd.Series, window: int, lags=range(2, 21)) -> pd.Series:
    """Rolling Hurst exponent from how the spread's dispersion grows with the lag."""
    logs, lag_logs = [], []
    for L in lags:
        d = series.diff(L)
        logs.append(np.log(d.rolling(window).std()))
        lag_logs.append(math.log(L))
    Y = np.column_stack([s.to_numpy() for s in logs])
    X = np.array(lag_logs)
    Xc = X - X.mean()
    slope = ((Y - Y.mean(axis=1, keepdims=True)) * Xc).sum(axis=1) / (Xc**2).sum()
    return pd.Series(slope, index=series.index)


def half_life(series: pd.Series, window: int) -> pd.Series:
    """Rolling Ornstein-Uhlenbeck half-life in bars: regress the change on the previous level."""
    lag = series.shift(1)
    d = series - lag
    cov = d.rolling(window).cov(lag)
    var = lag.rolling(window).var()
    lam = cov / var
    return (-math.log(2) / lam).where(lam < 0)


def adf_stat(e: np.ndarray) -> float:
    """Dickey-Fuller t-statistic (no lags, no constant) on residuals."""
    e = e[np.isfinite(e)]
    if len(e) < 50:
        return float("nan")
    de, lag = np.diff(e), e[:-1]
    lag = lag - lag.mean()
    b = (lag @ de) / (lag @ lag)
    resid = de - b * lag
    se = math.sqrt(resid @ resid / (len(de) - 1) / (lag @ lag))
    return b / se


def monthly_gate(la: pd.Series, lb: pd.Series, p: Params) -> pd.Series:
    """At each month start: is the pair correlated and cointegrated over the past windows?
    The answer holds for the month (known at the month's first bar)."""
    months = la.index.tz_convert(None).to_period("M").unique()
    gate = pd.Series(False, index=la.index)
    ra, rb = la.diff(), lb.diff()
    for m in months:
        t0 = m.start_time.tz_localize("UTC")
        past = (la.index < t0)
        a_c = la[past & (la.index >= t0 - pd.Timedelta(days=p.coint_window_d))]
        b_c = lb.reindex(a_c.index)
        r_win = past & (la.index >= t0 - pd.Timedelta(days=p.corr_window_d))
        if len(a_c) < 24 * 60 or r_win.sum() < 24 * 30:
            continue
        corr = ra[r_win].corr(rb[r_win])
        X = np.column_stack([b_c.to_numpy(), np.ones(len(b_c))])
        coef, *_ = np.linalg.lstsq(X, a_c.to_numpy(), rcond=None)
        stat = adf_stat(a_c.to_numpy() - X @ coef)
        ok = bool(corr > p.corr_min and stat < ADF_CRIT_10)
        gate[(la.index >= t0) & (la.index < (m + 1).start_time.tz_localize("UTC"))] = ok
    return gate


def prepare(a: pd.DataFrame, b: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Hourly frame for the pair with spread, z-score, regime and indicators."""
    df = pd.DataFrame({"pa": a["close"], "pb": b["close"]}).dropna()
    la, lb = np.log(df["pa"]), np.log(df["pb"])
    if p.regime_switch:
        # Centre on the first prices (known at the start) so beta and the intercept are separable.
        beta, alpha = kalman_beta((la - la.iloc[0]).to_numpy(), (lb - lb.iloc[0]).to_numpy(), p.kalman_delta,
                                  p.kalman_ve)
        alpha = alpha + la.iloc[0] - beta * lb.iloc[0]
    else:  # static hedge: OLS over the past 180 days, refreshed monthly
        beta, alpha = np.full(len(df), np.nan), np.full(len(df), np.nan)
        for m in df.index.tz_convert(None).to_period("M").unique():
            t0 = m.start_time.tz_localize("UTC")
            past = (df.index < t0) & (df.index >= t0 - pd.Timedelta(days=p.coint_window_d))
            if past.sum() < 24 * 60:
                continue
            X = np.column_stack([lb[past], np.ones(past.sum())])
            coef, *_ = np.linalg.lstsq(X, la[past], rcond=None)
            month = (df.index >= t0) & (df.index < (m + 1).start_time.tz_localize("UTC"))
            beta[month], alpha[month] = coef[0], coef[1]
    df["beta"] = beta
    # log(A) - beta * log(B), less the fitted intercept so a slowly drifting level doesn't swamp it.
    df["spread"] = la - df["beta"] * lb - alpha
    w = p.z_window_h
    df["mu"] = df["spread"].rolling(w, min_periods=w // 2).mean()
    df["sd"] = df["spread"].rolling(w, min_periods=w // 2).std()
    df["z"] = (df["spread"] - df["mu"]) / df["sd"]
    df["rsi"] = blocks.rsi(df["spread"], 14)
    df["hurst"] = hurst(df["spread"], p.hurst_window)
    df["hl"] = half_life(df["spread"], p.hurst_window)
    df["ema_f"] = df["spread"].ewm(span=p.ema_fast, adjust=False).mean()
    df["ema_s"] = df["spread"].ewm(span=p.ema_slow, adjust=False).mean()
    df["gate"] = monthly_gate(la, lb, p)
    revert = (df["hurst"] < p.revert_hurst) & df["hl"].between(p.hl_min_h, p.hl_max_h) & df["gate"]
    trend = (df["hurst"] > p.trend_hurst) & ((df["ema_f"] - df["ema_s"]).abs() > p.trend_gap_sd * df["sd"])
    df["regime"] = np.where(revert, "revert", np.where(trend, "trend", "unclear"))
    if not p.regime_switch:
        df["regime"] = np.where(df["beta"].notna(), "revert", "unclear")
    return df


def run(df: pd.DataFrame, p: Params, *, cost: str = "kraken_pro_taker", start=None, name: str = "") -> pd.DataFrame:
    """Simulate on hourly closes. Returns a trades table like sim.trades_frame."""
    c = COSTS[cost]
    per_side = (c["fee"] + c["slip"]) / 1e4
    idx = df.index
    pa, pb, z, rsi = df["pa"].to_numpy(), df["pb"].to_numpy(), df["z"].to_numpy(), df["rsi"].to_numpy()
    sd, beta, reg = df["sd"].to_numpy(), df["beta"].to_numpy(), df["regime"].to_numpy()
    ef, es, hl = df["ema_f"].to_numpy(), df["ema_s"].to_numpy(), df["hl"].to_numpy()
    i0 = max(1, int(np.searchsorted(idx, pd.Timestamp(start))) if start is not None else 1)
    equity, pos, rows = 1.0, None, []
    for i in range(i0, len(idx)):
        if pos is not None:
            side, qa, qb, kind = pos["side"], pos["qa"], pos["qb"], pos["kind"]
            exit_why = None
            if kind == "revert":
                if (side < 0 and z[i] <= 0) or (side > 0 and z[i] >= 0):
                    exit_why = "z_zero"
                elif abs(z[i]) > p.z_stop:
                    exit_why = "z_stop"
                elif p.regime_switch and reg[i] == "trend":
                    exit_why = "turned_trend"
                elif pos["max_bars"] and i - pos["i"] >= pos["max_bars"]:
                    exit_why = "time_stop"
            else:
                if (side > 0 and ef[i] < es[i]) or (side < 0 and ef[i] > es[i]):
                    exit_why = "opposite_cross"
            if exit_why or i == len(idx) - 1:
                pnl = qa * (pa[i] - pos["pa"]) + qb * (pb[i] - pos["pb"])
                fees = (abs(qa) * pa[i] + abs(qb) * pb[i]) * per_side + pos["fees"]
                short_notional = abs(qa) * pos["pa"] if qa < 0 else abs(qb) * pos["pb"]
                carry = short_notional * p.borrow_pa * (idx[i] - pos["t"]).total_seconds() / (365 * 86400)
                net = pnl - fees - carry
                rows.append({"instrument": name, "side": side, "entry_time": pos["t"], "exit_time": idx[i],
                             "entry_px": pos["spread"], "stop_px": np.nan, "pnl": net, "ret": net / pos["equity"],
                             "r": net / (pos["risk"] or np.nan), "fees": fees + carry, "reason": kind,
                             "exit_why": exit_why or "data_end"})
                equity += net
                pos = None
            continue
        if not (np.isfinite(z[i]) and np.isfinite(sd[i]) and sd[i] > 0 and np.isfinite(beta[i])):
            continue
        side, kind = 0, None
        if reg[i] == "revert" and np.isfinite(rsi[i - 1]):
            if z[i] > p.z_entry and rsi[i - 1] >= 70 > rsi[i]:
                side, kind = -1, "revert"
            elif z[i] < -p.z_entry and rsi[i - 1] <= 30 < rsi[i]:
                side, kind = 1, "revert"
        elif p.regime_switch and p.allow_trend and reg[i] == "trend":
            if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
                side, kind = 1, "trend"
            elif ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
                side, kind = -1, "trend"
        if not side:
            continue
        # Size so a move to the stop (|z| = 3.5, or 2 spread sd for trend trades) loses `risk` of equity.
        dist = (p.z_stop - abs(z[i])) * sd[i] if kind == "revert" else 2 * sd[i]
        dist = max(dist, 0.25 * sd[i])
        b = abs(beta[i])
        notional = min(equity * p.risk / dist, equity * p.max_gross / (1 + b))  # notional of leg A
        qa = side * notional / pa[i]
        qb = -side * np.sign(beta[i]) * b * notional / pb[i]
        fees = (abs(qa) * pa[i] + abs(qb) * pb[i]) * per_side
        max_bars = int(3 * hl[i]) if kind == "revert" and np.isfinite(hl[i]) else 0
        pos = {"side": side, "qa": qa, "qb": qb, "kind": kind, "pa": pa[i], "pb": pb[i], "t": idx[i], "i": i,
               "fees": fees, "equity": equity, "risk": equity * p.risk, "spread": df["spread"].iloc[i],
               "max_bars": max_bars}
    return pd.DataFrame(rows, columns=["instrument", "side", "entry_time", "exit_time", "entry_px", "stop_px", "pnl",
                                       "ret", "r", "fees", "reason", "exit_why"])


def pair_names(instruments: list[str]) -> list[tuple[str, str]]:
    """Every pair, the larger market second (as in ETH/BTC)."""
    order = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "SUI/USD"]
    ranked = sorted(instruments, key=lambda s: order.index(s) if s in order else 99)
    return [(ranked[j], ranked[i]) for i in range(len(ranked)) for j in range(i + 1, len(ranked))]
