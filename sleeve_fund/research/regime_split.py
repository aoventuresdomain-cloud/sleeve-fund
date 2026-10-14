"""R3: a Strategy's results split by market regime, a report only (Researcher tool gap R3; Advisor 6 Oct 14:05 UK).

Fixed, backward-looking labels read at each trade's entry, deliberately NOT the P2-5 regime model:
- bull / bear: the daily close above its 200-day simple average, or not;
- high / low volatility: 30-day realised volatility of daily log returns above the median of its own trailing year, or
  not;
- trend / range / neutral: Wilder's ADX(14) on daily bars above 25, below 20, or between.

A trade takes the labels of the last daily close at or before its entry. A day without enough history for a label is
"unlabelled" (200 closes for the average; 30 returns plus a year of them for the volatility median; 28 bars for ADX).
Per bucket: trades, net P&L after fees and funding, its share of the total (every trade's, the unlabelled included,
so the rows add up; shown only when the total is above 0), the share of days in that regime, and over those days the
Strategy's own return compounded from its daily equity beside buy and hold's price return (no fees), like for like
(Advisor 14 Oct 09:25 UK). A day's return is the regime's when the close before it carried that label, as a trade
entered that day would. A bucket under MIN_TRADES trades reads "too few". When the net P&L is positive and more
than DEPENDENT_SHARE of it comes from one bucket, the dimension is flagged "regime-dependent". Nothing here gates G1,
and a regime filter added after looking at this is a new counted variant in the trials register."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sleeve_fund.research import metrics

SMA_DAYS = 200
VOL_DAYS = 30
VOL_YEAR = 365
ADX_DAYS = 14
ADX_TREND, ADX_RANGE = 25.0, 20.0
MIN_TRADES = 20
DEPENDENT_SHARE = 0.8
UNLABELLED = "unlabelled"
DIMENSIONS = {"direction": ("bull", "bear"), "volatility": ("high", "low"), "trend": ("trend", "neutral", "range")}
TITLES = {"direction": "Bull or bear (close vs 200-day average)",
          "volatility": "Volatility (30-day vs its trailing-year median)",
          "trend": "Trend or range (ADX 14: above 25 trend, below 20 range)"}


def daily(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily open, high, low, close from bars of any length that divides a day, stamped at the day's close (midnight
    UTC) as the study's daily bars are; daily bars as they are. A part day at the end is left out."""
    if len(bars) < 2 or (bars.index[1] - bars.index[0]) >= pd.Timedelta(days=1):
        return bars[["open", "high", "low", "close"]].astype(float)
    out = bars.resample("1D", closed="right", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if bars.index[-1] < out.index[-1]:
        out = out.iloc[:-1]
    return out.astype(float)


def adx(d: pd.DataFrame, n: int = ADX_DAYS) -> pd.Series:
    """Wilder's ADX(n): true range and directional moves smoothed with Wilder's 1/n average; NaN for the first 2n
    bars."""
    high, low, close = d["high"], d["low"], d["close"]
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    up, down = high.diff(), -low.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    wilder = lambda s: s.ewm(alpha=1 / n, adjust=False).mean()  # noqa: E731
    atr = wilder(tr)
    pdi, mdi = 100 * wilder(plus) / atr, 100 * wilder(minus) / atr
    dx = (100 * (pdi - mdi).abs() / (pdi + mdi)).fillna(0.0)
    out = wilder(dx)
    out.iloc[: 2 * n] = np.nan
    return out


def labels(d: pd.DataFrame) -> pd.DataFrame:
    """Each day's three labels from closes up to and including that day's: backward-looking only."""
    close = d["close"]
    sma = close.rolling(SMA_DAYS, min_periods=SMA_DAYS).mean()
    vol = np.log(close).diff().rolling(VOL_DAYS, min_periods=VOL_DAYS).std()
    median = vol.rolling(VOL_YEAR, min_periods=VOL_YEAR).median()
    a = adx(d)
    out = pd.DataFrame(index=d.index)
    out["direction"] = np.where(sma.isna(), UNLABELLED, np.where(close > sma, "bull", "bear"))
    out["volatility"] = np.where(median.isna(), UNLABELLED, np.where(vol > median, "high", "low"))
    out["trend"] = np.where(a.isna(), UNLABELLED,
                            np.where(a > ADX_TREND, "trend", np.where(a < ADX_RANGE, "range", "neutral")))
    return out


@dataclass(frozen=True)
class Bucket:
    name: str
    trades: int
    net_pnl: float
    share: float | None  # of the dimension's total net P&L; None when the total isn't positive
    days: float  # share of the period's days in this regime
    hold_return: float | None  # buy and hold's price return compounded over those days; None with no benchmark
    too_few: bool
    strategy_return: float | None = None  # the Strategy's own, from its daily equity over the same days


@dataclass(frozen=True)
class Split:
    dimension: str
    buckets: tuple[Bucket, ...]
    dependent: str | None  # the bucket over DEPENDENT_SHARE of a positive net P&L, else None


def _over(returns: pd.Series | None, prior: pd.Series, name: str) -> float | None:
    """`returns` (daily) compounded over the days whose previous close carried label `name`."""
    if returns is None:
        return None
    on = prior.index[prior == name]
    return float((1 + returns.reindex(on).dropna()).prod() - 1)


def split(trips: list[dict], bars: pd.DataFrame, hold_equity: pd.Series | None = None,
          equity: pd.Series | None = None) -> list[Split]:
    """trips: closed round trips (metrics.trades: opened, pnl); bars: the period's price bars; hold_equity: buy and
    hold's equity over the same period; equity: the Strategy's own over it. One Split per dimension."""
    lab = labels(daily(bars))
    hold = metrics.daily_returns(hold_equity) if hold_equity is not None and len(hold_equity) > 1 else None
    own = metrics.daily_returns(equity) if equity is not None and len(equity) > 1 else None
    closed = [t for t in trips if t.get("opened") is not None and np.isfinite(t.get("pnl", np.nan))]
    at = _utc(lab.index).searchsorted(_utc(pd.DatetimeIndex([pd.Timestamp(t["opened"]) for t in closed])),
                                       side="right") - 1
    out = []
    for dim, names in DIMENSIONS.items():
        tag = [lab[dim].iloc[i] if i >= 0 else UNLABELLED for i in at]
        total = sum(t["pnl"] for t in closed)
        days = lab[dim].value_counts(normalize=True)
        prior = lab[dim].shift(1, fill_value=UNLABELLED)
        buckets = []
        for name in (*names, UNLABELLED):
            mine = [t["pnl"] for t, g in zip(closed, tag) if g == name]
            if name == UNLABELLED and not mine and name not in days:
                continue
            buckets.append(Bucket(name, len(mine), float(sum(mine)), float(sum(mine)) / total if total > 0 else None,
                                  float(days.get(name, 0.0)), _over(hold, prior, name), len(mine) < MIN_TRADES,
                                  _over(own, prior, name)))
        top = max((b for b in buckets if b.share is not None), key=lambda b: b.share, default=None)
        out.append(Split(dim, tuple(buckets), top.name if top is not None and top.share > DEPENDENT_SHARE else None))
    return out


def _utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Fill times and bar stamps in one frame: a naive stamp is UTC, as the engine's are."""
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def render(splits: list[Split]) -> list[str]:
    """The tear sheet's section: one small table per dimension, report only."""
    out = ["## Results by regime (report only)", "",
           "Labels are read at each trade's entry from closes before it. This never gates G1; a regime filter added "
           "after reading it is a new counted variant.", ""]
    for s in splits:
        flag = f" **Regime-dependent:** over {DEPENDENT_SHARE:.0%} of the net P&L came in {s.dependent}." \
            if s.dependent else ""
        out += [f"{TITLES[s.dimension]}.{flag}", "", "| Regime | Trades (by entry) | Net P&L | Share of P&L | "
                "Time in regime | Strategy over those days | Buy and hold over those days (price only, no fees) |",
                "|---|---|---|---|---|---|---|"]
        for b in s.buckets:
            trades = f"{b.trades} (too few)" if b.too_few else str(b.trades)
            share = "n/a (total <= 0)" if b.share is None else f"{b.share:.0%}"
            own = "n/a" if b.strategy_return is None else f"{b.strategy_return:+.1%}"
            hold = "n/a" if b.hold_return is None else f"{b.hold_return:+.1%}"
            out.append(f"| {b.name} | {trades} | {b.net_pnl:,.2f} | {share} | {b.days:.0%} | {own} | {hold} |")
        out.append("")
    return out
