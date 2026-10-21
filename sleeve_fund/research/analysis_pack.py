"""Fund-grade analysis pack (R-17, R-18, R-19; spec research/quant-researcher/fund-grade-analysis-spec.md sections 3-6):
the risk and return figures, the trades and streaks with the trade-order luck test, and the cost breakdown, each a pure
function of what a study already holds (its daily equity, buy-and-hold's, its closed trades and its costs). Nothing is
run. section() writes them as one tear-sheet section; money is in the account's currency, shown as $ as the sheet does.

Definitions (HoE 21 Oct, the Advisor checks the maths):
- daily returns at the daily closes (metrics.daily_returns); a year is 365 days, as metrics.summary
- rolling 12 months: 365-day windows, only once a full window exists; return = prod(1 + r) - 1, Sharpe = mean / sd x
  sqrt(365)
- best and worst day, week and month: daily returns compounded over complete UTC calendar weeks (Monday to Sunday) and
  months; a part period at either end is left out
- drawdowns: distinct episodes on daily equity, from the peak to the first close back at or above it; days to recover
  are peak to recovery; one still open at the end is "not recovered" and runs to the end for time under water
- VaR and expected shortfall at 95%, historical: with k = ceil(5% of n), VaR is minus the k-th worst daily return and ES
  minus the mean of the k worst
- correlation and beta to buy-and-hold: Pearson on the days both have; beta = cov / var(buy-and-hold)
- trades: a trade with P&L <= 0 is a loss (as metrics.trade_stats); payoff ratio = average win / |average loss|
- trade-order luck test: each trade's return on the equity before it (P&L / equity before) is reordered and compounded
  from the same start, ignoring the time between trades; 10,000 orders drawn with a fixed seed, or every order exactly
  when there are at most 7 trades (7! = 5,040); percentiles by numpy's linear interpolation
- costs: fees and funding as recorded; the spread and slippage are modelled (half the spread plus the study's slippage on
  orders that take liquidity, in the fill prices), never measured, so they are one line and never split
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd

from sleeve_fund.research.metrics import PERIODS_PER_YEAR, daily_closes, fills_to_rows, trades

SEED = 20261021  # the luck test's draws, fixed so a sheet always reads the same
DRAWS = 10_000
EXACT_UP_TO = 7  # 7! = 5,040 orders: every one, under DRAWS
VAR_LEVEL = 0.95
WORST_DRAWDOWNS = 5
BEST_TRADES = 5


# R-17: returns and risk
def rolling_12m(returns: pd.Series) -> pd.DataFrame:
    """Each day's trailing 365-day return and Sharpe, from the first day a full window exists."""
    r = returns.dropna()
    if len(r) < PERIODS_PER_YEAR:
        return pd.DataFrame(columns=["return", "sharpe"], dtype=float)
    ret = (1 + r).rolling(PERIODS_PER_YEAR).apply(np.prod, raw=True) - 1
    sd = r.rolling(PERIODS_PER_YEAR).std(ddof=1)
    sharpe = (r.rolling(PERIODS_PER_YEAR).mean() / sd * math.sqrt(PERIODS_PER_YEAR)).where(sd > 0, 0.0)
    return pd.DataFrame({"return": ret, "sharpe": sharpe}).dropna()


def _by_day(returns: pd.Series) -> pd.Series:
    """Daily returns keyed by the UTC day they are the move of (a return stamped at a close is the day before's)."""
    r = returns.dropna()
    days = (r.index - pd.Timedelta("1D")).tz_localize(None) if r.index.tz is not None else r.index - pd.Timedelta("1D")
    return pd.Series(r.to_numpy(), index=days.normalize())


def _calendar(returns: pd.Series, rule: str) -> pd.Series:
    """Daily returns compounded over each complete period of `rule` (every day of it held); a part period at either
    end left out."""
    r = _by_day(returns)
    if r.empty:
        return pd.Series(dtype=float)
    periods = r.index.to_period(rule)
    grouped = (1 + r).groupby(periods)
    out = grouped.prod() - 1
    held = grouped.size()
    whole = [held[p] == (p.end_time.normalize() - p.start_time.normalize()).days + 1 for p in out.index]
    return out[whole]


def best_worst(returns: pd.Series) -> dict:
    """{"day" | "week" | "month": (best, its start, worst, its start)}, or None where no complete period exists."""
    out = {}
    periods = {"day": _calendar(returns, "D"), "week": _calendar(returns, "W-SUN"), "month": _calendar(returns, "M")}
    for name, s in periods.items():
        out[name] = (None if s.empty else
                     (float(s.max()), s.idxmax().start_time, float(s.min()), s.idxmin().start_time))
    return out


@dataclass
class Drawdown:
    depth: float  # negative: the fall from the peak, as a fraction of it
    start: pd.Timestamp  # the peak
    trough: pd.Timestamp
    recovered: pd.Timestamp | None  # the first close back at or above the peak; None while open

    def days(self, end: pd.Timestamp) -> int:
        """Peak to recovery, or to `end` while not recovered."""
        return int(((self.recovered or end) - self.start) / pd.Timedelta("1D"))


def drawdowns(equity: pd.Series) -> list[Drawdown]:
    """Every distinct drawdown episode of a daily equity curve, oldest first."""
    out, peak_ts, peak, trough_ts, trough = [], None, -math.inf, None, None
    for ts, v in equity.items():
        if v >= peak:
            if trough is not None and trough < peak:
                out.append(Drawdown(trough / peak - 1, peak_ts, trough_ts, ts))
            peak_ts, peak, trough_ts, trough = ts, v, None, None
        elif trough is None or v < trough:
            trough_ts, trough = ts, v
    if trough is not None and trough < peak:
        out.append(Drawdown(trough / peak - 1, peak_ts, trough_ts, None))
    return out


def worst_drawdowns(equity: pd.Series, n: int = WORST_DRAWDOWNS) -> list[Drawdown]:
    return sorted(drawdowns(equity), key=lambda d: (d.depth, d.start))[:n]


def longest_under_water(equity: pd.Series) -> Drawdown | None:
    if equity.empty:
        return None
    end = equity.index[-1]
    return max(drawdowns(equity), key=lambda d: (d.days(end), -d.start.value), default=None)


def var_es(returns: pd.Series, level: float = VAR_LEVEL) -> tuple[float, float]:
    """Historical daily VaR and expected shortfall at `level`, as positive losses."""
    r = np.sort(returns.dropna().to_numpy())
    if not len(r):
        return float("nan"), float("nan")
    k = math.ceil((1 - level) * len(r) - 1e-9)
    k = max(k, 1)
    return float(-r[k - 1]), float(-r[:k].mean())


def correlation_beta(returns: pd.Series, bench: pd.Series) -> tuple[float, float]:
    both = pd.concat([returns, bench], axis=1, join="inner").dropna()
    if len(both) < 2 or both.iloc[:, 1].var(ddof=1) == 0:
        return float("nan"), float("nan")
    a, b = both.iloc[:, 0], both.iloc[:, 1]
    return float(a.corr(b)), float(a.cov(b) / b.var(ddof=1))


# R-18: trades, streaks and the luck test
def _streaks(wins: list[bool]) -> list[tuple[bool, int, int]]:
    """(won, first index, length) for each run of wins or losses."""
    out, i = [], 0
    while i < len(wins):
        j = i
        while j < len(wins) and wins[j] == wins[i]:
            j += 1
        out.append((wins[i], i, j - i))
        i = j
    return out


def _path(start: float, rets) -> np.ndarray:
    return start * np.concatenate([[1.0], np.cumprod(1 + np.asarray(rets, dtype=float))])


def _max_dd(path: np.ndarray) -> float:
    return float((path / np.maximum.accumulate(path) - 1).min())


def _longest_loss_run(rets) -> int:
    best = run = 0
    for x in rets:
        run = run + 1 if x <= 0 else 0
        best = max(best, run)
    return best


def trade_figures(trips: list[dict], start: float) -> dict:
    """Payoff ratio, largest win and loss, streaks (each losing streak with the fall it caused in trade-by-trade
    equity) and the best 5 trades' share of the total profit. `trips`: closed trades oldest first, as metrics.trades;
    `start`: the equity before the first."""
    pnl = [t["pnl"] for t in trips]
    wins = [p > 0 for p in pnl]
    won = [t for t in trips if t["pnl"] > 0]
    lost = [t for t in trips if t["pnl"] <= 0]
    avg_w = sum(t["pnl"] for t in won) / len(won) if won else float("nan")
    avg_l = sum(t["pnl"] for t in lost) / len(lost) if lost else float("nan")
    equity = start + np.concatenate([[0.0], np.cumsum(pnl)])
    runs = _streaks(wins)
    losing = [(i, n) for w, i, n in runs if not w]
    falls = []
    for i, n in losing:  # the fall from the equity before the streak (or the peak before it) to its lowest point
        peak = equity[: i + 1].max()
        falls.append((n, float(equity[i + 1: i + n + 1].min() / peak - 1)))
    total = sum(pnl)
    top = sorted(pnl, reverse=True)[:BEST_TRADES]
    largest_win = max(trips, key=lambda t: t["pnl"]) if won else None
    largest_loss = min(trips, key=lambda t: t["pnl"]) if lost else None
    return {
        "trades": len(trips),
        "payoff_ratio": avg_w / abs(avg_l) if won and lost and avg_l != 0 else float("nan"),
        "avg_win": avg_w, "avg_loss": avg_l,
        "largest_win": (largest_win["pnl"], largest_win["ret"]) if largest_win else None,
        "largest_loss": (largest_loss["pnl"], largest_loss["ret"]) if largest_loss else None,
        "longest_win_streak": max((n for w, _, n in runs if w), default=0),
        "longest_loss_streak": max((n for w, _, n in runs if not w), default=0),
        # the losing streaks, longest first (then the deeper fall): (length, fall from the peak before it)
        "loss_streaks": sorted(falls, key=lambda x: (-x[0], x[1])),
        "best5_share": sum(top) / total if total > 0 else None,
    }


def trade_returns(trips: list[dict], start: float) -> list[float]:
    """Each trade's P&L as a share of the equity before it (compounding as sized)."""
    out, eq = [], start
    for t in trips:
        out.append(t["pnl"] / eq if eq else 0.0)
        eq += t["pnl"]
    return out


def reshuffle(rets: list[float], start: float, draws: int = DRAWS, seed: int = SEED) -> dict | None:
    """The trade-order luck test: P5, P50 and P95 of end value, max drawdown and longest losing streak over reorderings
    of the same trade returns. Every order when there are at most EXACT_UP_TO trades, else `draws` drawn with `seed`."""
    if not rets:
        return None
    x = np.asarray(rets, dtype=float)
    if len(x) <= EXACT_UP_TO:
        orders, exact = [list(p) for p in permutations(range(len(x)))], True
    else:
        rng = np.random.default_rng(seed)
        orders, exact = [rng.permutation(len(x)) for _ in range(draws)], False
    end, dd, streak = [], [], []
    for o in orders:
        p = _path(start, x[o])
        end.append(p[-1])
        dd.append(_max_dd(p))
        streak.append(_longest_loss_run(x[o]))
    pct = lambda v: tuple(float(np.percentile(v, q)) for q in (5, 50, 95))  # noqa: E731
    return {"orders": len(orders), "exact": exact, "end_value": pct(end), "max_drawdown": pct(dd),
            "longest_loss_streak": pct(streak)}


# R-19: costs
def costs(run) -> dict:
    """Fees, funding, the insurance fund's share of a liquidation and the modelled spread and slippage of one run, and
    the gross P&L they reconcile: gross (at the prices before the spread) less fees and the spread and slippage, plus
    funding (+ received, - paid) and insurance, is the net P&L, the equity's change over the run, open position
    marked."""
    fees, spread = float(run.fees_paid), float(run.spread_paid)
    funding = sum(float(f["amount"]) for f in run.funding)
    insurance = sum(float(f["amount"]) for f in run.insurance)
    net = float(run.equity.iloc[-1]) - float(run.starting_capital)
    return {"fees": fees, "spread_slippage": spread, "funding": funding, "insurance": insurance,
            "gross": net + fees + spread - funding - insurance, "net": net}


def per_year(amount: float, equity: pd.Series) -> float:
    """`amount` as a share of the average equity a year."""
    span = (equity.index[-1] - equity.index[0]) / pd.Timedelta(days=PERIODS_PER_YEAR) if len(equity) > 1 else 0
    mean = float(equity.mean())
    return amount / mean / span if span and mean else float("nan")


# The sheet's section
def _p(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2%}"


def _m(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def _d(ts) -> str:
    return "" if ts is None else f"{ts:%Y-%m-%d}"


def caveats(r) -> list[str]:
    run = r.full_period
    out = ["- Survivorship: the asset pool is today's assets; ones delisted since are not in it.",
           "- Fills are pessimistic where only coarse bars say what traded (stops and targets inside one bar), until "
           "execution bars come (R-01)."]
    if run.funding_held:
        out.append(f"- Funding: {run.funding_at_baseline} of {run.funding_held} held settlements were charged a "
                   "baseline rate for a missing venue rate.")
    return out


def section(r) -> list[str]:
    """The R-17/R-18/R-19 section of a tear sheet. Window: the full period (labelled) until R-11's scored window."""
    from sleeve_fund.research.metrics import daily_returns

    run, bench = r.full_period, r.full_period_benchmark
    eq, beq = daily_closes(run.equity), daily_closes(bench.equity)
    ret, bret = daily_returns(run.equity), daily_returns(bench.equity)
    trips = [t for t in trades(fills_to_rows(run.fills), run.shorts, run.funding, run.insurance)
             if t.get("closed") is not None]
    end = eq.index[-1] if len(eq) else None
    out = ["## Risk, trades and costs", "",
           "Over the full research period (the scored window replaces it once R-11 lands). UTC calendar periods.", ""]

    roll, broll = rolling_12m(ret), rolling_12m(bret)
    out.append("| Figure | Strategy | Buy and hold |")
    out.append("|---|---|---|")
    for label, a, b in (("Rolling 12-month return, lowest / highest",
                         f"{_p(roll['return'].min() if len(roll) else None)} / {_p(roll['return'].max() if len(roll) else None)}",
                         f"{_p(broll['return'].min() if len(broll) else None)} / {_p(broll['return'].max() if len(broll) else None)}"),
                        ("Rolling 12-month Sharpe, lowest / highest",
                         f"{roll['sharpe'].min():.2f} / {roll['sharpe'].max():.2f}" if len(roll) else "n/a",
                         f"{broll['sharpe'].min():.2f} / {broll['sharpe'].max():.2f}" if len(broll) else "n/a")):
        out.append(f"| {label} | {a} | {b} |")
    bw, bbw = best_worst(ret), best_worst(bret)
    for name in ("day", "week", "month"):
        cell = lambda v: ("n/a" if v is None else f"{_p(v[0])} ({_d(v[1])}) / {_p(v[2])} ({_d(v[3])})")  # noqa: E731
        out.append(f"| Best / worst {name} | {cell(bw[name])} | {cell(bbw[name])} |")
    var, es = var_es(ret)
    bvar, bes = var_es(bret)
    out.append(f"| Daily VaR 95% (historical) | {var:.2%} | {bvar:.2%} |")
    out.append(f"| Daily expected shortfall 95% | {es:.2%} | {bes:.2%} |")
    corr, beta = correlation_beta(ret, bret)
    out.append(f"| Correlation / beta to buy and hold | {corr:.2f} / {beta:.2f} | 1.00 / 1.00 (itself) |")
    under = longest_under_water(eq)
    out.append(f"| Longest time under water | {under.days(end) if under else 0} days"
               + (f" ({_d(under.start)} to {_d(under.recovered) or 'not recovered'})" if under else "") + " | |")
    out.append("")
    out.append("Five worst drawdowns:")
    out.append("")
    out.append("| Depth | Peak | Trough | Recovered | Days to recover |")
    out.append("|---|---|---|---|---|")
    for d in worst_drawdowns(eq):
        out.append(f"| {d.depth:.2%} | {_d(d.start)} | {_d(d.trough)} | {_d(d.recovered) or 'not recovered'} | "
                   f"{d.days(end) if d.recovered else 'not recovered'} |")
    out.append("")

    start = float(run.starting_capital)
    tf = trade_figures(trips, start)
    out.append("| Trades | |")
    out.append("|---|---|")
    out.append(f"| Payoff ratio (average win / average loss) | "
               + ("n/a" if math.isnan(tf["payoff_ratio"]) else f"{tf['payoff_ratio']:.2f}") + " |")
    for k, label in (("largest_win", "Largest win"), ("largest_loss", "Largest loss")):
        v = tf[k]
        out.append(f"| {label} | " + ("n/a" if v is None else f"{_m(v[0])} ({_p(v[1])})") + " |")
    out.append(f"| Longest winning / losing streak | {tf['longest_win_streak']} / {tf['longest_loss_streak']} trades |")
    worst = tf["loss_streaks"][:3]
    out.append("| Losing streaks, longest first, with the fall each caused | "
               + ("none" if not worst else "; ".join(f"{n} in a row: {f:.2%}" for n, f in worst)) + " |")
    out.append(f"| Best {BEST_TRADES} trades' share of total profit | "
               + ("n/a (no total profit)" if tf["best5_share"] is None else f"{tf['best5_share']:.0%}") + " |")
    out.append("")
    luck = reshuffle(trade_returns(trips, start), start)
    if luck:
        how = (f"every one of the {luck['orders']:,} orders" if luck["exact"] else
               f"{luck['orders']:,} orders drawn with seed {SEED}")
        out.append(f"Trade-order luck test: the same {len(trips)} trades reordered ({how}), each compounding on the "
                   "equity before it, ignoring the time between trades.")
        out.append("")
        out.append("| | 5th percentile | Median | 95th percentile |")
        out.append("|---|---|---|---|")
        e, d, s = luck["end_value"], luck["max_drawdown"], luck["longest_loss_streak"]
        out.append(f"| End value from {_m(start)} | {_m(e[0])} | {_m(e[1])} | {_m(e[2])} |")
        out.append(f"| Max drawdown | {d[0]:.2%} | {d[1]:.2%} | {d[2]:.2%} |")
        out.append(f"| Longest losing streak | {s[0]:.0f} | {s[1]:.0f} | {s[2]:.0f} |")
        out.append("")

    c = costs(run)
    pa = lambda x: _p(per_year(x, run.equity)).lstrip("+")  # noqa: E731
    out.append("| Costs | Money | % of average equity a year |")
    out.append("|---|---|---|")
    out.append(f"| Gross P&L, before costs | {_m(c['gross'])} | {pa(c['gross'])} |")
    out.append(f"| Fees | {_m(-c['fees'])} | {pa(-c['fees'])} |")
    out.append(f"| Spread and slippage (modelled at half the spread plus {getattr(r, 'ladder_slippage', 0.0):.2%} on "
               f"orders that take liquidity, not measured) | {_m(-c['spread_slippage'])} | {pa(-c['spread_slippage'])} |")
    out.append(f"| Funding (+ received, - paid) | {_m(c['funding'])} | {pa(c['funding'])} |")
    if c["insurance"]:
        out.append(f"| Liquidation shortfall the venue's insurance fund took (+ taken off the loss) | "
                   f"{_m(c['insurance'])} | |")
    out.append(f"| Net P&L (the equity's change, an open position marked) | {_m(c['net'])} | {pa(c['net'])} |")
    out.append("")
    out.append("Net = gross - fees - spread and slippage + funding + the insurance fund's share; funding is + when "
               "received and - when paid, and the insurance fund's share is + (the part of a liquidation's loss the "
               "fund took, never charged to the account).")
    out.append("")
    out += caveats(r)
    out.append("")
    return out
