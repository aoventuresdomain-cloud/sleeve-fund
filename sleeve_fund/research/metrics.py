"""Performance metrics. The venue trades every day, so annualisation uses 365. Everything is judged
on daily returns, whatever the bar length: an hourly or minute strategy's curve is cut to its daily
closes first (daily_returns), so its Sharpe is annualised as daily and its bootstrap blocks are days."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365
EULER_GAMMA = 0.5772156649


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def _intraday(series: pd.Series) -> bool:
    return (len(series) > 2 and isinstance(series.index, pd.DatetimeIndex)
            and pd.Series(series.index).diff().median() < pd.Timedelta("1D"))


def daily_closes(equity: pd.Series) -> pd.Series:
    """An equity curve at its daily closes (stamped at the close, midnight UTC); daily curves as they are.

    Only whole days count: a curve that ends mid-day has no close for that day yet, so the part day
    is left out rather than passed off as a full one. A curve that starts mid-day opens its first day
    with its first mark (nothing is held before the first bar), so that day's move still counts."""
    if not _intraday(equity):
        return equity
    closes = equity.resample("1D", closed="right", label="right").last().dropna()
    if len(closes) and closes.index[-1] > equity.index[-1]:
        closes = closes.iloc[:-1]
    start = equity.index[0].floor("1D")
    if equity.index[0] > start:
        closes = pd.concat([pd.Series([equity.iloc[0]], index=[start]), closes])
    return closes


def whole_days(returns: pd.Series, first_close: pd.Timestamp, bar: pd.Timedelta) -> pd.Series:
    """The daily returns whose whole day lies at or after a window's first bar (closing at
    first_close, `bar` long), so a day that straddles the window's start never counts for it."""
    return returns[returns.index - pd.Timedelta("1D") >= first_close - bar]


def daily_returns(equity: pd.Series) -> pd.Series:
    """Day-on-day returns of an equity curve at any bar length."""
    return returns_from_equity(daily_closes(equity))


def years_covered(series: pd.Series) -> float:
    """The time a curve covers, in years: its span plus one bar."""
    if len(series) < 2 or not isinstance(series.index, pd.DatetimeIndex):
        return max(len(series) / PERIODS_PER_YEAR, 1e-9)
    bar = pd.Series(series.index).diff().median()
    return max((series.index[-1] - series.index[0] + bar) / pd.Timedelta(days=PERIODS_PER_YEAR), 1e-9)


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
        {"side": "BUY" if str(side).endswith("BUY") else "SELL", "qty": float(q), "price": float(px), "fee": _money(c),
         "ts": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts, "order_id": str(oid)}
        for side, q, px, c, ts, oid in zip(f["side"], f["filled_qty"], f["avg_px"], fees, f["ts_last"], f.index)
    ]


def trades(rows: list[dict]) -> list[dict]:
    """Closed round trips (flat -> long -> flat) with P&L after fees, oldest first.

    rows: fills in time order with side, qty, price, fee (quote currency). Partial
    fills are fine: a trip closes when the position returns to (about) zero.
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
    return float(notional / equity.mean() / years_covered(equity))


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


def deflated_sharpe_probability(returns: pd.Series, n_trials: int, trial_sharpes: list[float] | None = None) -> float:
    """Probability the true Sharpe beats the best-of-n luck hurdle (daily units internally).

    With trial_sharpes None, the spread of trial Sharpes is the no-skill sampling error 1/sqrt(T).
    Use that when the tried variants include fee-destroyed ones (Sharpe -10 and worse), whose
    spread would make any hurdle absurd; the lab found this in its first round."""
    r = returns.dropna()
    n = len(r)
    if n < 30 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    if trial_sharpes is None:
        sr_std = 1 / math.sqrt(n - 1)
    else:
        daily_trials = np.asarray(trial_sharpes, dtype=float) / math.sqrt(PERIODS_PER_YEAR)
        sr_std = float(daily_trials.std(ddof=1)) if len(daily_trials) > 1 else 0.0
    hurdle = expected_max_sharpe(n_trials, sr_std)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(NormalDist().cdf((sr - hurdle) * math.sqrt(n - 1) / denom))


def _long_run(x: np.ndarray) -> tuple[float, float, float, int]:
    """(variance, long-run variance, the weighted sum Politis and White's block length uses, n) of a
    series, from its autocovariances under a flat-top kernel whose width the data picks: the first lag
    after which k_n autocorrelations in a row are insignificant, doubled."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    k_n = max(5, math.ceil(math.log10(n)))
    m_max = math.ceil(math.sqrt(n)) + k_n
    var = float(x @ x) / n
    if var <= 0:
        return 0.0, 0.0, 0.0, n
    acv = np.array([float(x[: n - k] @ x[k:]) / n for k in range(m_max + 1)])
    rho = np.abs(acv[1:] / var)
    threshold = 2 * math.sqrt(math.log10(n) / n)
    m_hat = next((m for m in range(len(rho) - k_n + 1) if (rho[m:m + k_n] < threshold).all()), m_max)
    m = min(2 * max(m_hat, 1), m_max)
    lags = np.arange(-m, m + 1)
    t = np.abs(lags) / m
    flat_top = np.where(t <= 0.5, 1.0, np.where(t <= 1, 2 * (1 - t), 0.0))
    acv_l = acv[np.abs(lags)]
    return var, float((flat_top * acv_l).sum()), float((flat_top * np.abs(lags) * acv_l).sum()), n


def block_length(x: np.ndarray) -> int:
    """The circular-bootstrap block length a series needs to keep its own persistence: Politis and
    White (2004), with Patton, Politis and White's (2009) correction. Slow trend P&L, which stays up or
    down for weeks, needs blocks of weeks; a short fixed block resamples it as if each day were new
    and makes luck look like skill."""
    var, spectrum0, g, n = _long_run(x)
    if var <= 0 or spectrum0 <= 0:
        return 1
    b_max = math.ceil(min(3 * math.sqrt(n), n / 3))
    b = (2 * g * g / ((4 / 3) * spectrum0 ** 2)) ** (1 / 3) * n ** (1 / 3)
    return int(min(max(math.ceil(b), 1), b_max))


MIN_INDEPENDENT_DAYS = 250


def independent_days(x: np.ndarray) -> float:
    """Roughly how many independent observations a daily series holds: n times its variance over its
    long-run variance (every autocorrelation that matters, summed, not just yesterday's). Returns that
    drift for weeks or months at a time carry far less evidence than their day count, and no bootstrap
    recovers it: 60-day regimes with a lag-1 autocorrelation of only 0.3 hold about one observation in
    twenty. Never more than n."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or x.std() == 0:
        return float(len(x))
    var, spectrum0, _, n = _long_run(x)
    if spectrum0 <= var:
        return float(n)  # no persistence (or mean reversion): every day counts
    return n * var / spectrum0


def sharpe_beats_probability(strategy: pd.Series, benchmark: pd.Series, n_trials: int, *, block: int | None = None,
                             n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """How sure we can be that the strategy's Sharpe truly beats the benchmark's, out of sample, once
    the variants tried are allowed for. Returns (probability, hurdle), Sharpes annualised.

    The two daily return series are resampled together in blocks (keeping volatility clusters, trend
    persistence and the pairing between them), 2,000 times. The block is at least 10 days and long
    enough for the more persistent of the two series and their difference (block_length). Each resample gives a Sharpe difference; the
    hurdle is the difference the best of `n_trials` skill-less variants would show by luck alone
    (expected_max_sharpe, with the resamples' spread as the no-skill error). The probability is the
    share of resamples above it. Fixed seed, so a tear sheet reads the same every time. Under 60 days,
    or under MIN_INDEPENDENT_DAYS independent observations, it returns NaN: not enough to judge."""
    pair = pd.concat([strategy, benchmark], axis=1, join="inner").dropna()
    n = len(pair)
    if n < 60:
        return float("nan"), float("nan")
    x = pair.to_numpy(dtype=float)
    if min(independent_days(x[:, 0]), independent_days(x[:, 0] - x[:, 1])) < MIN_INDEPENDENT_DAYS:
        return float("nan"), float("nan")  # too little independent evidence to judge
    if block is None:
        block = max(10, *(block_length(c) for c in (x[:, 0], x[:, 1], x[:, 0] - x[:, 1])))
    if n < 3 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, -(-n // block)))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]  # circular blocks
    sample = x[idx]  # (n_boot, n, 2)
    mean, std = sample.mean(axis=1), sample.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0, mean / std, 0.0) * math.sqrt(PERIODS_PER_YEAR)
    diff = sharpe[:, 0] - sharpe[:, 1]
    hurdle = expected_max_sharpe(n_trials, float(diff.std(ddof=1)))
    return float((diff > hurdle).mean()), hurdle
