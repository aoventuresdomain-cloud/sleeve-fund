"""B-38 (Advisor spec, 20 Oct 2026): a multi-asset basket's study report.

The basket is judged as one: ONE fund run of the whole basket (research.multi_asset's legs, each sized off the fund,
the compliance tests trimming as on paper), its daily net P&L summed across the assets over the fund's equity, on the
out-of-sample days only (the walk-forward's test windows; settings are fixed, so one run cut to them, Advisor fork 1).
PASS needs all four:

1. it beats random entries: its pooled Sharpe strictly above the draws' 95th percentile (each asset keeps its trade
   count, sides and holding times, only the entry times move, inside the test windows; pooled the same way; 1,000
   draws, a fixed seed, recorded). Both sides are ungated, each trade at its intended notional (what it asked the
   compliance tests for), over the fund's equity, which is like with like only while the tests trim or refuse under
   5% of the intended entry notional on the test days (Advisor fork 3, B38-PLAN a/b): at 5% or more the check is
   not judged, and the basket can't pass;
2. net above 0 at twice the venue's fees, a real re-run (B-22);
3. net above 0 with its best asset left out, a real re-run that never counts as a try;
4. the locked holdout, sealed through the study and opened once as its own final step (open_holdout), net above 0
   at the venue's fees; opening it marks it used for every asset in the basket.

Beside the verdict: the pooled break-even fee (one fee per side on every fill, real runs on the cost ladder), the try
count (one per fixed basket and settings, each also one of its model's 20 configurations: B38-PLAN c) and "next:
4-6 weeks paper". Per-asset lines are diagnostic only."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from sleeve_fund import markets
from sleeve_fund.instruments import FeeSchedule
from sleeve_fund.research import trials
from sleeve_fund.research.holdout import HoldoutLocks
from sleeve_fund.research.metrics import daily_closes, fills_to_rows, trades
from sleeve_fund.research.multi_asset import FIXED_LIST, AssetData, check_assets, leg_names, underlying
from sleeve_fund.research.portfolio_run import LegSpec, PortfolioResult, run_portfolio
from sleeve_fund.research.random_entry import BAR, DRAWS, HELD_BACK_MAX, Trade, _random_entries
from sleeve_fund.research.candidates import BUDGET_PER_IDEA, configurations
from sleeve_fund.research.study import COST_LADDER, _between, doubled
from sleeve_fund.risk import PORTFOLIO, PortfolioProfile

# Advisor 27 Oct (fast lane), research only: the run a basket's edge verdict and joint draws are judged on, every
# fund limit switched off as B-32 switches one off (None), so nothing trims, pauses or halts it. Paper and live never
# take it: their limits are the runtime's profile (tests/test_basket_guards.py pins that).
UNGUARDED = PortfolioProfile(gross=None, net_instrument=None, margin=None, open_risk=None, drawdown=None,
                             daily_loss=None)
GATE_TOUCH_CAP = 0.05  # Advisor fork 3: the ungated random benchmark is like-with-like only below this
CONCENTRATED = 0.5  # the top asset's share of the pooled out-of-sample net P&L above which the basket is flagged
GROSS_CAP = 1.01  # guards off or scaled: a leg's notional over its own account's equity as it opens, lot rounding apart
NEXT = "next: 4-6 weeks paper"
DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class BasketAsset:
    """One asset's whole history: decision bars of bar_minutes (daily, or 4-hour on a pinned list) and the bars of
    exec_minutes the engine fills against (hourly; 1-minute under a 4-hour list, as its single sheets fill), each
    stamped at its close. The study cuts each run's window (and its warm-up before it) from these."""

    instrument: Any
    prices: pd.DataFrame
    exec_prices: pd.DataFrame
    bar_minutes: int = 1440
    exec_minutes: int = 60


@dataclass(frozen=True)
class Basket:
    name: str
    strategy: str
    params: dict
    assets: tuple[BasketAsset, ...]
    universe_date: str  # when the fixed list was chosen (the fixed-list warning)

    @property
    def names(self) -> list[str]:
        return [underlying(a.instrument) for a in self.assets]


@dataclass(frozen=True)
class TradeAt:
    """A round trip on an asset's daily closes: in at bar `entry`'s close, out at bar `exit`'s, `notional` at entry.
    exit_cost: False when `exit` is a window's last day the trade was held past (clip): no exit is paid there."""

    entry: int
    exit: int
    side: int
    notional: float
    exit_cost: bool = True


@dataclass
class Measured:
    """A fund run of the basket, measured on the out-of-sample days."""

    run: PortfolioResult
    assets: list[str]
    pnl: pd.DataFrame  # each asset's daily net P&L, one column per asset (diagnostic only)
    returns: pd.Series  # the pooled daily net return
    net: float  # compounded over the days
    passed: bool = False


@dataclass
class LeftOut:
    asset: str
    share: float  # its part of the pooled out-of-sample net P&L
    concentrated: bool
    run: Measured
    random: RandomBeat | None = None  # on the reduced universe: information only

    @property
    def net(self) -> float:
        return self.run.net

    @property
    def passed(self) -> bool:
        return self.run.net > 0


@dataclass
class RandomBeat:
    judged: bool
    percentile: float  # the share of draws whose pooled Sharpe is below the basket's, in percent
    beats: bool  # strictly above the draws' 95th percentile
    draws: int
    gate_touched: float
    sharpe: float = math.nan
    median: float = math.nan
    bar: float = math.nan  # the draws' 95th percentile
    seed: int = 0


@dataclass
class BasketStudy:
    basket: Basket
    windows: tuple[tuple[date, date], ...]
    research_end: date
    base: Measured
    double_fees: Measured
    left_out: LeftOut
    random: RandomBeat
    gate: tuple[float, float]  # (notional the compliance tests trimmed or refused, intended entry notional), test days
    ladder: list[tuple[float, float]] = field(default_factory=list)  # (fee per side, pooled net)
    breakeven: tuple[float | None, str] = (None, "not tested")
    asset_breakeven: dict[str, tuple[float | None, str]] = field(default_factory=dict)
    tries: int = 0  # baskets of this model tried
    configuration: int = 0  # this basket's place among its model's configurations (of BUDGET_PER_IDEA)
    key: str = ""
    # STUDY-ASSET-LIST (Advisor ruling 21 Oct), with basket_study(filled=True): random entry filled as the engine fills
    # on the pooled book (it decides in place of `random`, which stays close-priced, without a verdict), the hold bar
    # (long-only baskets only) and, for a basket with SOL, the same study without it (diagnostic only).
    pooled: Any = None
    hold: Any = None
    without_sol: dict | None = None
    # Advisor 27 Oct: with the fund guards off (the edge verdict's run, counted once), or every leg at size_scale of
    # its unscaled size (the guarded run, fitness only, never a try)
    unguarded: bool = False
    size_scale: float | None = None
    counted: bool = True
    held: dict = field(default_factory=dict)  # own-sized runs: held_peaks by asset, information only (Advisor 28 Oct)

    @property
    def gate_touched(self) -> float:
        cut, asked = self.gate
        return cut / asked if asked else 0.0

    @property
    def verdict(self) -> str:
        return verdict(self, None)


@dataclass
class Holdout:
    period: tuple[date, date]
    run: Measured

    @property
    def net(self) -> float:
        return self.run.net

    @property
    def passed(self) -> bool:
        return self.run.net > 0


def base_fees(instrument, params: dict) -> FeeSchedule:
    """The venue's schedule a leg pays when it is given none (runner.plan_run's own default)."""
    return markets.fees_for(params, FeeSchedule(instrument.maker_fee, instrument.taker_fee), str(instrument.id.venue))


def _half_spread(instrument, params: dict) -> float:
    from sleeve_fund.venues import venue as venue_profile

    return float(markets.half_spread_for(params, venue_profile(str(instrument.id.venue)).assumed_half_spread,
                                         str(instrument.id.venue)))


def _cut(a: BasketAsset, first: date, last: date, warm_days: int) -> AssetData:
    """The bars of the days `first` to `last` (a bar closing at 00:00 is the day before's), and the `warm_days`
    decision bars before them (bars, not days, on a 4-hour list); nothing after `last`."""
    start, end = pd.Timestamp(first, tz="UTC"), pd.Timestamp(last + timedelta(days=1), tz="UTC")
    p, h = a.prices, a.exec_prices
    warm = p[p.index <= start].iloc[-warm_days:]
    if len(warm) < warm_days:
        unit = "days" if a.bar_minutes == 1440 else f"{a.bar_minutes}-minute bars"
        raise ValueError(f"{underlying(a.instrument)} has {len(warm)} {unit} of history before {first}, not the "
                         f"{warm_days} {unit} of warm-up the run needs")
    window = p[(p.index > start) & (p.index <= end)]
    if not len(window) or window.index[-1] != end:
        raise ValueError(f"{underlying(a.instrument)}'s history doesn't reach {last}")
    # Positional slices: views, never copies of a run's 1-minute bars (27 Oct: three assets' went past memory)
    lo, hi = h.index.searchsorted(start, "right"), h.index.searchsorted(end, "right")
    return AssetData(a.instrument, window, h.iloc[lo:hi], warm)


def _run(basket: Basket, assets: list[BasketAsset], first: date, last: date, fees, capital: float,
         profile: PortfolioProfile, warm_days: int,
         scale: float | None = None) -> tuple[PortfolioResult, list[str], list[LegSpec]]:
    """One fund run of `assets` over the days `first` to `last`. fees: a schedule for every leg, or None for each
    leg's venue fees. The fund's capital is split evenly across the legs' accounts; under the fund guards every leg
    sizes off the fund (B-38, as before). Guards off or at a size scale, every leg sizes off its own account instead:
    off the fund, each leg asked the whole fund's size on its share of the cash, so a 3-asset basket ran at 3x gross
    and a leg could lose its whole account (27 Oct, SOL). scale: every leg's account holds scale of its even share
    and the fund keeps the rest as cash no Strategy holds, so each entry is scale x the leg's current equity (it
    compounds as the unscaled leg does) and the book is (1 - scale) x capital + scale x the unscaled one (Advisor 27
    Oct, check 2a); None: as before."""
    share = 1.0 if scale is None else scale
    data = [_cut(a, first, last, warm_days) for a in assets]
    if len(data) > 1:
        check_assets(data)
    names = leg_names(basket.name, [d.instrument for d in data])
    specs = [LegSpec(n, basket.strategy, d.instrument, d.prices, d.exec_prices, d.warmup_prices,
                     exec_minutes=a.exec_minutes, bar_minutes=a.bar_minutes, params=dict(basket.params),
                     capital=capital * share / len(data), fees=fees)
             for n, d, a in zip(names, data, assets)]
    return run_portfolio(specs, profile, unallocated=capital * (1 - share),
                         fund_sized=not own_sized(profile, scale)), names, specs


def own_sized(profile: PortfolioProfile, scale: float | None) -> bool:
    """Guards off or at a size scale, every leg sizes off its own account (_run), never the fund."""
    return scale is not None or profile == UNGUARDED


def _gross(run: PortfolioResult) -> pd.Series | None:
    """The legs' summed notional over the fund's equity at every leg mark, None when nothing was held."""
    held = {n: (r.exposure * r.equity).abs() for n, r in run.legs.items() if r.exposure is not None and len(r.exposure)}
    if not held:
        return None
    gross = pd.concat(held, axis=1).sort_index().ffill().fillna(0.0).sum(axis=1)
    return gross / run.book.reindex(gross.index, method="ffill")


def gross_peak(run: PortfolioResult) -> float:
    """The most the legs held at once, every position's notional summed over the fund's equity, at the legs' marks."""
    ratio = _gross(run)
    return 0.0 if ratio is None else float(ratio.max())


def fill_peaks(run: PortfolioResult) -> dict[str, tuple[float, pd.Timestamp]]:
    """Each leg's largest exposure (notional over its own account's equity before the fill) left by a fill that grew
    its position, an open or an add, and when (Advisor 28 Oct). Sized off its own account a leg's fills leave it at
    1x of it (its account holds k of its share at a size scale, so 1x of that); off the whole fund, about N times."""
    out = {}
    for n, r in run.legs.items():
        pos, worst = 0.0, None
        for f in fills_to_rows(r.fills):
            after = pos + (f["qty"] if f["side"] == "BUY" else -f["qty"])
            if abs(after) > abs(pos) + 1e-12:
                ts = pd.Timestamp(f["ts"])
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
                before = r.equity[r.equity.index < ts]
                equity = float(before.iloc[-1]) if len(before) else float(r.starting_capital)
                x = abs(after) * f["price"] / equity if equity > 0 else math.inf
                if worst is None or x > worst[0]:
                    worst = (x, ts)
            pos = after
        if worst is not None:
            out[n] = worst
    return out


def held_peaks(run: PortfolioResult) -> dict[str, tuple[float, pd.Timestamp, float, float]]:
    """Each leg's highest exposure (x its own account) at a mark while held, when, what that position opened at, and
    the fees and funding it paid since it opened as a share of its notional then: a long drifts to about its opening
    exposure / (1 - that) as they come off its cash (Advisor 28 Oct); information only, never a stop."""
    out = {}
    for n, r in run.legs.items():
        if r.exposure is None or not len(r.exposure) or not (r.exposure.abs() > 0).any():
            continue
        e = r.exposure.abs()
        at = e.idxmax()
        rows = [f for f in fills_to_rows(r.fills) if pd.Timestamp(f["ts"]).tz_localize(None) <= at.tz_localize(None)]
        pos, opened, at_open = 0.0, None, math.nan
        for f in rows:
            after = pos + (f["qty"] if f["side"] == "BUY" else -f["qty"])
            if pos == 0 and after != 0:
                opened = pd.Timestamp(f["ts"]).tz_localize(None)
                before = r.equity[r.equity.index.tz_localize(None) < opened]
                equity = float(before.iloc[-1]) if len(before) else float(r.starting_capital)
                at_open = abs(after) * f["price"] / equity if equity > 0 else math.nan
            pos = after
        if opened is None:
            continue
        fees = sum(f["fee"] for f in rows if pd.Timestamp(f["ts"]).tz_localize(None) >= opened)
        paid = -sum(x["amount"] for x in (r.funding or [])
                    if opened <= pd.Timestamp(x["ts"]).tz_localize(None) <= at.tz_localize(None))
        notional = float(e[at]) * float(r.equity[at])
        out[n] = (float(e[at]), at, at_open, (float(fees) + paid) / notional if notional else math.nan)
    return out


def _own_sized_checks(run: PortfolioResult) -> None:
    """Advisor 27-28 Oct (b): a run whose legs size off their own accounts grows no leg's position above 1x of its own
    account, open or add (so the legs together never open above the fund), or it stops. Drift while held is not
    sizing and is never stopped (held_peaks prints it): 28 Oct, all three legs long at once for weeks paid enough
    funding to read 1.02x, and a 5% move against a 1x short reads 1.105x. The message says which leg, when and by how
    much; it reaches the refusal note, never the log."""
    over = {n: v for n, v in fill_peaks(run).items() if v[0] > GROSS_CAP}
    if over:
        legs = "; ".join(f"{n} at {x:.4f} at {t}" for n, (x, t) in over.items())
        worst = max(x for x, _ in over.values())
        raise ValueError(f"refused: a fill left a leg at {worst:.2f}x its own account's equity (at most "
                         f"{GROSS_CAP:.2f}x with each leg sized off its own account): no sheet. {legs}")


def _held_lines(study: "BasketStudy") -> list[str]:
    if not study.held:
        return []
    legs = "; ".join(f"{a} {x:.4f}x on {t:%d %b %Y %H:%M} (opened at {x0:.4f}x; fees and funding since then "
                     f"{c:.2%} of its notional)" for a, (x, t, x0, c) in study.held.items())
    return [f"**Highest held exposure per leg (x own account):** {legs}. Information only: held drift is costs or a "
            "move against a short, never sizing.", ""]


def equity_before(book: pd.Series, index: pd.DatetimeIndex, capital: float) -> pd.Series:
    """The fund's equity at the close of the day before each of `index`'s days, from the fund book's daily closes
    (the capital before its first): every P&L booked before counts, a day between test windows too (CR B-38)."""
    return book.shift(1).reindex(index).fillna(capital)


def pooled_returns(pnl: pd.DataFrame, book: pd.Series, capital: float) -> pd.Series:
    """The pooled daily net return: the day's net P&L summed across the assets, over the fund's equity at the day
    before's close (equity_before), the same denominator the random check divides by."""
    return pnl.sum(axis=1) / equity_before(book, pnl.index, capital)


def _daily_pnl(run: PortfolioResult, names: list[str], specs: list[LegSpec], assets: list[str]) -> pd.DataFrame:
    cols = {}
    for n, s, a in zip(names, specs, assets):
        eq = run.legs[n].equity
        closes = eq.resample("1D", closed="right", label="right").last().dropna()
        closes = closes[closes.index <= s.prices.index[-1]]
        cols[a] = closes.diff().fillna(closes.iloc[0] - s.capital)
    return pd.DataFrame(cols).fillna(0.0)


def _bar(spec: LegSpec) -> pd.Timedelta:
    return pd.Timedelta(minutes=spec.bar_minutes)


def _oos(index: pd.DatetimeIndex, windows, bar: pd.Timedelta = DAY) -> np.ndarray:
    """Which of these bars (each of length `bar`, stamped at its close) close on a test day."""
    days = [(t - bar).date() for t in index]
    return np.array([any(a <= d <= b for a, b in windows) for d in days], dtype=bool)


def _measure(run, names, specs, assets, windows, capital) -> Measured:
    pnl = _daily_pnl(run, names, specs, assets)
    pnl = pnl[_oos(pnl.index, windows)]
    returns = pooled_returns(pnl, daily_closes(run.book), capital)
    return Measured(run, assets, pnl, returns, float((1 + returns).prod() - 1) if len(returns) else 0.0)


def _decided_on(d: dict) -> date:
    return (pd.Timestamp(d["ts"]) - pd.Timedelta(1, "ns")).date()


def gate_touched(run: PortfolioResult, windows) -> tuple[float, float]:
    """On the test days: the entry notional the compliance tests trimmed or refused, and the intended entry notional
    asked of them (each decision's requested quantity at its expected fill); B38-PLAN a."""
    asked = cut = 0.0
    for d in run.decisions:
        if not any(a <= _decided_on(d) <= b for a, b in windows):
            continue
        px, req, ok = float(d["price"]), float(d["requested"]), float(d["approved"])
        asked += req * px
        cut += (req - ok) * px
    return cut, asked


def top_contributor(totals: pd.Series) -> tuple[str, float, bool]:
    """The asset with the largest out-of-sample net P&L, its share of the pooled net, and whether that is over half."""
    top = str(totals.idxmax())
    whole = float(totals.sum())
    share = float(totals[top]) / whole if whole else math.nan
    return top, share, bool(share > CONCENTRATED)


def ladder_breakeven(rungs: list[tuple[float, float]]) -> tuple[float | None, str]:
    """The fee per side at which the pooled out-of-sample net is 0, interpolating log(1 + return) between the two
    rungs either side of it (study._between); "<0" when it loses even at the bottom rung, "> top rung" when it still
    makes money at the top. Never extrapolated."""
    rungs = sorted(rungs)
    if not rungs or rungs[0][1] <= 0:
        return None, "<0"
    if rungs[-1][1] > 0:
        return None, "> top rung"
    for (f0, r0), (f1, r1) in zip(rungs, rungs[1:]):
        if r0 > 0 >= r1:
            value = _between(f0, r0, f1, r1)
            return value, f"{value:.3%} per side"
    return None, "> top rung"  # unreachable: the bottom rung is above 0 and the top isn't


def trade_pnl(closes: np.ndarray, trips: list[TradeAt], cost_per_side: float) -> np.ndarray:
    """Each bar's net P&L from these trades on these closes: each holds notional / its entry close in units from its
    entry's close to its exit's, paying cost_per_side on its entry notional and, when it really exits there
    (exit_cost), on its exit notional."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    pos, cost = np.zeros(n + 1), np.zeros(n)
    for t in trips:
        q = t.notional / c[t.entry]
        pos[t.entry + 1] += q * t.side
        pos[t.exit + 1] -= q * t.side
        cost[t.entry] += cost_per_side * t.notional
        if t.exit_cost:
            cost[t.exit] += cost_per_side * q * c[t.exit]
    move = np.r_[0.0, np.diff(c)]
    return np.cumsum(pos)[:n] * move - cost


def pooled_trade_returns(legs: list[tuple[np.ndarray, list[TradeAt], float]], equity: np.ndarray) -> np.ndarray:
    """Each day's P&L from every asset's trades (closes, trades, cost per side), summed, over the fund's equity at the
    day before's close: the pooled daily return the random benchmark compares, its draws and the basket alike."""
    return sum(trade_pnl(c, t, cps) for c, t, cps in legs) / np.asarray(equity, dtype=float)


def draw_trades(rng: np.random.Generator, trips: list[TradeAt], ranges: list[tuple[int, int]]) -> list[TradeAt]:
    """A random draw of these trades: in each range, the same trades (count, sides, holding times, notionals) at
    random entry times, without overlap (random_entry._random_entries)."""
    out = []
    for start, end in ranges:
        inside = [t for t in trips if start <= t.entry and t.exit <= end]
        if not inside:
            continue
        holds = np.array([t.exit - t.entry for t in inside])
        order, entries = _random_entries(rng, start, end, holds)
        out.extend(TradeAt(int(e), int(e + holds[k]), inside[k].side, inside[k].notional, inside[k].exit_cost)
                   for e, k in zip(entries, order))
    return out


def _trips(run: PortfolioResult, name: str, spec: LegSpec, params: dict) -> list[TradeAt]:
    """A leg's closed round trips on its daily closes, at their intended notional: in at the last close at or before
    the entry fill, out at the last close at or before the exit fill, at least one bar. The intended notional is what
    the opening entry asked the compliance tests for, its requested quantity at its expected fill: the leg's last
    sent opening entry (kind "entry", approved above 0) decided on a bar closing at or before the trip's first fill.
    Adds and trips still open never shift another trip's (CR B-38 a)."""
    idx = spec.prices.index
    opens = sorted((pd.Timestamp(d["bar_ts"]), float(d["requested"]) * float(d["price"])) for d in run.decisions
                   if d["strategy"] == name and d.get("kind", "entry") == "entry" and float(d["approved"]) > 0)
    bars = pd.DatetimeIndex([b for b, _ in opens])
    out = []
    for t in trades(fills_to_rows(run.legs[name].fills), shorts=markets.is_perp(params)):
        if t["closed"] is None:
            continue
        opened = pd.Timestamp(t["opened"])
        e = max(int(idx.searchsorted(opened, side="right")) - 1, 0)
        x = max(int(idx.searchsorted(pd.Timestamp(t["closed"]), side="right")) - 1, e + 1)
        k = int(bars.searchsorted(opened, side="right")) - 1
        if x < len(idx):
            out.append(TradeAt(e, x, int(t["side"]), opens[k][1] if k >= 0 else float(t["cost"])))
    return out


def clip(trips: list[TradeAt], ranges: list[tuple[int, int]]) -> list[TradeAt]:
    """The trades entered inside a test window, each cut at that window's last day when held past it (the fold-edge
    rule: only its marks inside the window count, Advisor fork 1), the same for the basket's trades and the draws'.
    Its entry is paid; its exit only when it exits inside the window: a cut is no exit, as the fund book's daily marks
    pay none (Advisor, B-38 review 20 Oct). A trade entered outside every window is dropped, as the draws' entries
    are inside the windows only."""
    out = []
    for t in trips:
        for start, end in ranges:
            if start <= t.entry <= end:
                out.append(TradeAt(t.entry, min(t.exit, end), t.side, t.notional, t.exit_cost and t.exit <= end))
                break
    return out


def _ranges(index: pd.DatetimeIndex, windows, bar: pd.Timedelta = DAY) -> list[tuple[int, int]]:
    """The out-of-sample days as runs of bar positions, (first, last) inclusive: windows that abut are one run, so a
    trade across a fold's edge is drawn with the rest."""
    inside = _oos(index, windows, bar)
    out, start = [], None
    for i, on in enumerate(inside):
        if on and start is None:
            start = i
        if not on and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(inside) - 1))
    return out


def _sharpe(r: np.ndarray) -> float:
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return float(np.mean(r) / sd * math.sqrt(365)) if sd > 0 else math.nan


def random_beat(m: Measured, names, specs, params: dict, windows, capital: float, touched: tuple[float, float],
                draws: int = DRAWS, seed: int = 0) -> RandomBeat:
    """The basket against random entries, pooled the same way (B38-PLAN b): both sides ungated, each trade at its
    intended notional, each day's P&L summed across the assets over the fund's equity at the day before's close (the
    basket's own book), on the test days only; the draws' entries inside the test windows. Not judged when the
    compliance tests trimmed or refused GATE_TOUCH_CAP of the intended entry notional or more."""
    if draws < DRAWS:
        raise ValueError(f"refused: the random check draws at least {DRAWS:,} times, not {draws:,}")
    cut, asked = touched
    share = cut / asked if asked else 0.0
    if share >= GATE_TOUCH_CAP:
        return RandomBeat(False, math.nan, False, draws, share, seed=seed)
    clock = specs[0].prices.index
    ranges = _ranges(clock, windows, _bar(specs[0]))
    # Each bar's P&L is summed into its day (the day a fund book's daily close labels it with: the bar's close rounded
    # up to midnight), so a 4-hour basket is judged on daily returns as a daily one is.
    days = clock.ceil("D")
    day_index = pd.DatetimeIndex(days.unique())
    of_day = day_index.get_indexer(days)
    keep = _oos(day_index, windows)
    legs = []
    for n, s in zip(names, specs):
        closes = s.prices["close"].to_numpy(dtype=float)
        cps = float(base_fees(s.instrument, params).taker) if s.fees is None else float(s.fees.taker)
        legs.append((closes, clip(_trips(m.run, n, s, params), ranges), cps + _half_spread(s.instrument, params)))
    if not any(t for _, t, _ in legs):
        return RandomBeat(False, math.nan, False, draws, share, seed=seed)
    book = daily_closes(m.run.book)
    equity = equity_before(book, day_index, capital).to_numpy(dtype=float)

    def pooled(trips_by_leg) -> float:
        pnl = sum(trade_pnl(c, t, cps) for (c, _, cps), t in zip(legs, trips_by_leg))
        return _sharpe((np.bincount(of_day, weights=pnl, minlength=len(day_index)) / equity)[keep])

    actual = pooled([t for _, t, _ in legs])
    rng = np.random.default_rng(seed)
    drawn = np.array([pooled([draw_trades(rng, t, ranges) for _, t, _ in legs]) for _ in range(draws)])
    bar = float(np.nanpercentile(drawn, BAR))
    pct = float((drawn < actual).mean() * 100)
    return RandomBeat(True, pct, beats_bar(actual, bar), draws, share, actual, float(np.nanmedian(drawn)), bar, seed)


def beats_bar(actual: float, bar: float) -> bool:
    """Strictly above the draws' 95th percentile: level with it is no beat."""
    return bool(actual > bar)


def random_words(r: RandomBeat) -> str:
    if not r.judged:
        if r.gate_touched >= GATE_TOUCH_CAP:
            return (f"not judged: the compliance tests trimmed {r.gate_touched:.1%} of entry notional "
                    f"({GATE_TOUCH_CAP:.0%} or more), so the random benchmark needs gated re-runs")
        return "not judged: no closed trades to draw from"
    return (f"{'PASS' if r.beats else 'FAIL'}: the pooled Sharpe {r.sharpe:.2f} is at the {r.percentile:.1f}th "
            f"percentile of {r.draws:,} random draws (their {BAR:.0f}th {r.bar:.2f}, median {r.median:.2f}; "
            f"needs strictly above the {BAR:.0f}th; seed {r.seed})")


def basket_key(basket: Basket) -> str:
    """One try: the fixed basket (its assets, in order: leg order can change reservations and fills) and its
    settings; a rule-builder definition's title renames it on screen only (B-16), so a retitle is the same try."""
    key = {"strategy": basket.strategy, "params": trials._untitled(basket.params), "assets": basket.names}
    minutes = basket.assets[0].bar_minutes if basket.assets else 1440
    if minutes != 1440:  # a daily basket's key is as it always was
        key["minutes"] = minutes
    return trials.content_hash(key)


def idea_hash(basket: Basket) -> str:
    """The basket's idea: its model's (the parent idea), whose 20-configuration budget each basket try counts in, and
    whose holdout locks the basket opens, one per asset."""
    return trials.legacy_idea_hash(basket.strategy)


def record_try(register: trials.TrialsRegister, basket: Basket, dataset: str, data_start, data_end,
               sharpe: float | None = None) -> tuple[int, int]:
    """Count the basket as tried, once per fixed basket and settings (an edit is a new try), as a counted
    configuration of its model's idea: a trials row keyed by the basket's structural hash, linked to its parent
    setting (B38-PLAN c, F236-4). Returns (baskets of this model tried, this basket's configuration number)."""
    idea, key = idea_hash(basket), basket_key(basket)
    rows = [t for t in register._counted(idea)]
    if key not in {t["definition_hash"] for t in rows}:
        register.record(definition_hash=key, idea_hash=idea, name=f"basket:{basket.name}",
                        family=trials._family(basket.strategy),
                        settings={"basket": basket.names, "params": basket.params,
                                  "parent": trials.legacy_definition_hash(basket.strategy, basket.params)},
                        dataset=dataset, stage="in_sample", source="study", sharpe=sharpe,
                        data_start=data_start, data_end=data_end)
        rows = register._counted(idea)
    baskets = {t["definition_hash"] for t in rows if str(t["definition_name"]).startswith("basket:")}
    ordered = sorted(configurations(register, idea),
                     key=lambda h: min(_created(t) for t in rows if t["definition_hash"] == h))
    return len(baskets), ordered.index(key) + 1


def _created(t: dict) -> pd.Timestamp:
    ts = pd.Timestamp(t["created_at"])
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def basket_study(basket: Basket, windows, research_end: date, *, register: trials.TrialsRegister, dataset: str,
                 capital: float = 10_000.0, profile: PortfolioProfile = PORTFOLIO, warm_days: int = 200,
                 ladder: bool = True, draws: int = DRAWS, seed: int = 0, filled: bool = False,
                 scale: float | None = None, counted: bool = True) -> BasketStudy:
    """The basket's study over the walk-forward's test windows ((first day, last day), in order), never reading a bar
    after research_end. Its holdout stays sealed: open_holdout is the separate final step. filled: a study request's
    asset list (STUDY-ASSET-LIST): random entry judged filled on the pooled book, the hold bar, and without SOL.
    scale: every leg at that share of its unscaled size (_run). counted False: a confirming run (Advisor 27 Oct, the
    guarded run at a size scale), whose settings are counted by their unguarded run, so it records no try."""
    windows = tuple(tuple(w) for w in windows)
    if not windows or windows[-1][1] > research_end:
        raise ValueError(f"the test windows must end by the research end {research_end}")
    first, last = windows[0][0], windows[-1][1]
    assets = list(basket.assets)
    names_of = basket.names

    own = own_sized(profile, scale)

    def measured(subset, fees):
        run, names, specs = _run(basket, subset, first, last, fees, capital, profile, warm_days, scale)
        if own:
            _own_sized_checks(run)
        return _measure(run, names, specs, [underlying(a.instrument) for a in subset], windows, capital), names, specs

    base, names, specs = measured(assets, None)
    base.passed = base.net > 0
    touched = gate_touched(base.run, windows)
    twice_fees = {str(a.instrument.id): doubled(base_fees(a.instrument, basket.params)) for a in assets}
    if len({(f.maker, f.taker) for f in twice_fees.values()}) != 1:
        raise ValueError("the basket's assets pay different fees: a leg's own doubled fees aren't supported yet")
    twice, _, _ = measured(assets, next(iter(twice_fees.values())))
    twice.passed = twice.net > 0
    top, share, concentrated = top_contributor(base.pnl.sum())
    rest = [a for a, n in zip(assets, names_of) if n != top]
    out, out_names, out_specs = measured(rest, None)
    left = LeftOut(top, share, concentrated, out)
    left.random = random_beat(out, out_names, out_specs, basket.params, windows, capital,
                              gate_touched(out.run, windows), draws, seed)
    rungs, per_asset = [], {a: [] for a in names_of}
    if ladder:
        for f in COST_LADDER:
            m, _, _ = measured(assets, FeeSchedule(maker=_dec(f), taker=_dec(f)))
            rungs.append((f, m.net))
            for a in names_of:
                per_asset[a].append((f, float(m.pnl[a].sum()) / (capital / len(assets))))
    study = BasketStudy(
        basket, windows, research_end, base, twice, left,
        random_beat(base, names, specs, basket.params, windows, capital, touched, draws, seed), touched,
        ladder=rungs, breakeven=ladder_breakeven(rungs) if rungs else (None, "not tested"),
        asset_breakeven={a: ladder_breakeven(r) for a, r in per_asset.items() if r})
    study.key = basket_key(basket)
    study.unguarded, study.size_scale, study.counted = profile == UNGUARDED, scale, counted
    if own:
        by_leg = held_peaks(base.run)
        study.held = {a: by_leg[n] for a, n in zip(names_of, names) if n in by_leg}
    if filled:
        study.pooled = pooled_random(base, names, specs, basket.params, windows, capital, touched, draws, seed)
        study.hold = hold_bar(base, specs, basket.params, windows, capital)
        if "SOL" in names_of:  # Advisor Q3: shown beside, its own re-run; never a condition or a try
            rest = [a for a, n in zip(assets, names_of) if n != "SOL"]
            m, rn, rs = measured(rest, None)
            study.without_sol = {"assets": [n for n in names_of if n != "SOL"], "net": m.net,
                                 "sharpe": _sharpe(m.returns.to_numpy()),
                                 "random": pooled_random(m, rn, rs, basket.params, windows, capital,
                                                         gate_touched(m.run, windows), draws, seed)}
    if own and filled:  # Advisor 27 Oct (a): trips that don't add up to the book stop the whole sheet
        for name, r in (("the basket", study.pooled), ("without SOL", (study.without_sol or {}).get("random"))):
            if r is not None and r.unreconciled:
                trips, book = r.reconciled
                raise ValueError(f"refused: {name}'s pooled trips add to {trips:+.1%} against the book's {book:+.1%} "
                                 "over them: the trades and the book disagree, so no sheet and no try")
    if counted:
        study.tries, study.configuration = record_try(
            register, basket, dataset, _since(first, warm_days, basket),
            pd.Timestamp(last + timedelta(days=1), tz="UTC"),
            None if math.isnan(study.random.sharpe) else study.random.sharpe)
    return study


def _since(first: date, warm_days: int, basket: Basket) -> pd.Timestamp:
    """The first bar any run of the study read, by its close: the warm-up's (as run_study records data_start)."""
    start = pd.Timestamp(first, tz="UTC")
    return min(a.prices.index[a.prices.index <= start][-warm_days:][0] for a in basket.assets)


def _dec(x: float):
    from decimal import Decimal

    return Decimal(repr(x))


def open_holdout(study: BasketStudy, period: tuple[date, date], locks: HoldoutLocks, *, basket: Basket | None = None,
                 capital: float = 10_000.0, profile: PortfolioProfile = PORTFOLIO, warm_days: int = 200) -> Holdout:
    """The locked holdout, once, for the basket the study judged (refused when the basket or its settings differ:
    B38-PLAN d): claimed for every asset before anyone looks (refused, naming the asset, when any is already spent or
    read), one fund run at the venue's fees, then each asset's lock marked opened. A look that fails after the claim
    spends every asset's holdout."""
    basket = study.basket if basket is None else basket
    if basket_key(basket) != study.key:
        raise ValueError("refused: this basket or its settings differ from the ones the study judged, so its holdout "
                         "can't be opened; study the edited basket first (a new try)")
    if period[0] <= study.research_end:
        raise ValueError(f"refused: the holdout starts {period[0]}, inside the research period (to "
                         f"{study.research_end})")
    idea = idea_hash(basket)
    first, last = period
    # By its bars' closes, as run_study keys its holdout and records what it read: the first closes after `first`.
    start, end = pd.Timestamp(first + timedelta(days=1), tz="UTC"), pd.Timestamp(last + timedelta(days=1), tz="UTC")
    for a in basket.names:
        why = locks.refusal(idea, a, start, end)
        if why:
            raise ValueError(why)
    claimed = []
    for a in basket.names:
        if not locks.open(idea, a, start, end):
            for b in claimed:
                locks.crashed(idea, b)
            raise ValueError(f"the holdout on {a} is held by another study")
        claimed.append(a)
    try:
        run, names, specs = _run(basket, list(basket.assets), first, last, None, capital, profile, warm_days)
        m = _measure(run, names, specs, basket.names, [period], capital)
    except Exception:
        for a in claimed:
            locks.crashed(idea, a)
        raise
    m.passed = m.net > 0
    for a in claimed:
        locks.settle(idea, a, None)
    return Holdout(period, m)


def verdict(study: BasketStudy, holdout: Holdout | None) -> str:
    """PASS only with all four. FAIL when one is judged and fails; NOT JUDGED when the random check couldn't be
    (it can't pass then); NOT YET while the holdout is sealed."""
    pooled, hold = getattr(study, "pooled", None), getattr(study, "hold", None)
    if pooled is not None:  # STUDY-ASSET-LIST: the filled pooled comparison decides; the hold bar once it is proven
        random_failed, judged = pooled.verdict == "FAIL", pooled.verdict == "PASS"
        held = hold is None or (BH102_LIVE and hold.beats)
        random_failed = random_failed or (hold is not None and BH102_LIVE and not hold.beats)
    else:
        random_failed, judged, held = study.random.judged and not study.random.beats, study.random.judged, True
    failed = random_failed or not study.double_fees.passed \
        or not study.left_out.passed or (holdout is not None and not holdout.passed)
    if failed:
        return "FAIL"
    if not judged or not held:
        return "NOT JUDGED"
    return "NOT YET" if holdout is None else "PASS"


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.1%}"


def render(study: BasketStudy, holdout: Holdout | None = None) -> str:
    """The basket's report. The verdict lines are the pooled run's; per-asset lines are diagnostic only."""
    b, base = study.basket, study.base
    days = len(base.returns)
    v = verdict(study, holdout)
    pooled = getattr(study, "pooled", None)
    not_judged = "NOT JUDGED: the random benchmark couldn't be judged, so the basket can't pass"
    if pooled is not None and pooled.verdict == "PASS" and study.hold is not None and not BH102_LIVE:
        not_judged = f"NOT JUDGED: the hold bar is {UNPROVEN}, so the basket can't pass yet"
    head = {"NOT YET": "NOT YET: the locked holdout is sealed until its own final step",
            "NOT JUDGED": not_judged, "PASS": "PASS", "FAIL": "FAIL"}[v]
    if study.size_scale is not None and not study.unguarded:
        head = (f"FITNESS ONLY: a guarded run with every leg at {study.size_scale:g} of its size; the edge "
                f"verdict is the unguarded run's (this run's own would read {v})")
    lo = study.left_out
    if pooled is None:
        random_rows = [f"| Beats random entries (95th percentile) | {random_words(study.random)} |"]
        left_random = random_words(lo.random) if lo.random is not None else None
        concentrated = "; concentrated: over half the net from one asset"
        viewed = ""
    else:
        random_rows = [f"| Beats random entries, filled on the pooled book (95th percentile) | {pooled_words(pooled, study.random.seed)} |",
                       f"| Hold bar (long-only baskets) | {hold_words(study.hold)} |",
                       f"| Random entries, close-priced | {close_priced_words(study.random)} |"]
        left_random = close_priced_words(lo.random) if lo.random is not None else None
        concentrated = "; CONCENTRATED: over half the pooled profit from one asset"
        viewed = "; holdout already viewed; paper is the forward test"
    lines = [
        f"# Basket study: {b.name} ({b.strategy} on {', '.join(b.names)})", "",
        f"**Basket verdict:** {head}", "",
        f"_{FIXED_LIST.format(date=b.universe_date)}_", "",
        *_run_lines(study),
        (f"**Tries:** {study.tries} (one per fixed basket and settings); this basket is try {study.configuration} of "
         f"{BUDGET_PER_IDEA} for {b.strategy}'s family" if study.counted else
         "**Tries:** not a new try: a confirming run of settings already counted (Advisor 27 Oct)"), "",
        *([LEGS_NOT_REBALANCED, ""] if pooled is not None else []),
        *_held_lines(study),
        f"Out-of-sample: {study.windows[0][0]} to {study.windows[-1][1]}, {days} days, one fund run of the whole "
        f"basket at the venue's fees: net {_pct(base.net)}.", "",
        "| Condition | Result |", "| --- | --- |",
        *random_rows,
        f"| Net above 0 at 2x the venue's fees | {'PASS' if study.double_fees.passed else 'FAIL'}, net "
        f"{_pct(study.double_fees.net)} |",
        f"| Best asset left out ({lo.asset}, {lo.share:.0%} of the pooled net) | {'PASS' if lo.passed else 'FAIL'}, "
        f"net {_pct(lo.net)}{concentrated if lo.concentrated else ''}"
        f"{'; info, without it: ' + left_random if left_random is not None else ''} |",
        "| Locked holdout at the venue's fees | " + (
            "sealed: opened once, as its own final step" if holdout is None else
            f"{'PASS' if holdout.passed else 'FAIL'}, net {_pct(holdout.net)} ({holdout.period[0]} to "
            f"{holdout.period[1]})") + viewed + " |", "",
        f"The compliance tests trimmed or refused {study.gate[0]:,.2f} of {study.gate[1]:,.2f} intended entry notional "
        f"on the test days ({study.gate_touched:.1%} of entry notional).", "",
        f"**Pooled break-even fee:** {study.breakeven[1]} (one fee per side on every fill, real runs on the cost "
        "ladder)", "",
        *_without_sol_lines(study),
        "## Per asset (diagnostic only, never a verdict)", "",
        "| Asset | Out-of-sample net P&L | Share | Break-even fee |", "| --- | --- | --- | --- |",
    ]
    totals = base.pnl.sum()
    whole = float(totals.sum())
    for a in base.assets:
        be = study.asset_breakeven.get(a, (None, "not tested"))[1]
        lines.append(f"| {a} | {totals[a]:,.2f} | {totals[a] / whole if whole else math.nan:.0%} | {be} |")
    lines += ["", f"On a PASS, {NEXT}." if v != "PASS" else f"PASS, so {NEXT}.", ""]
    return "\n".join(lines)


# Advisor 27 Oct (check 2b): the basket's convention, beside a pool worked by hand
LEGS_NOT_REBALANCED = ("_Each asset's leg compounds on its own even share of the fund, never rebalanced; a pool "
                       "rebalanced daily differs, and part of any gap from one is that._")


def worst_drawdown(returns: pd.Series) -> float:
    """The pooled book's deepest fall from its high, on its out-of-sample daily returns compounded (0 when flat)."""
    if not len(returns):
        return 0.0
    equity = np.concatenate([[1.0], (1 + returns.to_numpy()).cumprod()])
    return float((1 - equity / np.maximum.accumulate(equity)).max())


def largest_share(run: PortfolioResult, windows) -> float:
    """The Strategy's largest share of the fund's book at any mark on the test days: the book less the cash no
    Strategy holds, over the book (a scaled run's share drifts above k as it makes money: Advisor 27 Oct)."""
    book = run.book[_oos(run.book.index, windows, pd.Timedelta(hours=1))]
    return float(((book - run.unallocated) / book).max()) if len(book) else math.nan


def _run_lines(study: BasketStudy) -> list[str]:
    """What kind of run the sheet is, for a confirming run (Advisor 27 Oct), with the drawdown its k is set from (the
    unguarded run) or the drawdown in fund terms and the Strategy's largest share of the book (the scaled run);
    nothing for an ordinary one."""
    dd = worst_drawdown(study.base.returns)
    if study.unguarded:
        words = ("**Run:** fund guards OFF, research only: no fund limit trimmed, paused or halted anything, so the "
                 "edge verdict and joint draws are judged here; its drawdown is not the fund's")
    elif study.size_scale is not None:
        words = (f"**Run:** fund guards on, as paper, every leg at {study.size_scale:g} of its size (its account "
                 f"holds {study.size_scale:g} of its even share, the rest the fund's idle cash): fitness only")
    else:
        return []
    words += (f"; every leg at {study.size_scale:g} of its size" if study.unguarded
              and study.size_scale is not None else "") + "."
    if study.size_scale is None:
        figures = (f"**Pooled worst drawdown:** {dd:.2%} (daily closes, out of sample); a size scale k = 10% / this "
                   "sets the guarded run")
    else:
        figures = (f"**Worst drawdown in fund terms:** {dd:.2%} (daily closes, out of sample; never rebalanced to k, so "
                   f"it can pass k x the unscaled one). **Largest Strategy share of the book:** "
                   f"{largest_share(study.base.run, study.windows):.1%}")
    return [words, "", figures, ""]


# ---- STUDY-ASSET-LIST (Advisor ruling 21 Oct): random entry on the pooled book, filled as the engine fills ---------

@dataclass
class PooledLeg:
    """One asset's side of the pooled random-entry comparison: its realised trips on its decision bars, each one's
    return as the engine booked it (fees, fills, funding) and its intended entry notional over the fund's capital,
    and the pricer that fills its draws as the engine fills the definition (fills.Pricer)."""

    asset: str
    trades: list
    realised: np.ndarray
    weights: np.ndarray
    windows: list[tuple[int, int]]
    pricer: Any
    gaps: np.ndarray
    stops: np.ndarray
    in_market: float


def random_entry_pooled(legs: list[PooledLeg], reconciled: tuple, draws: int = DRAWS, seed: int = 0,
                        why_not: str = ""):
    """PR E's comparison on the pooled book (Advisor Q1): the statistic is the sum of every asset's realised trip P&L
    at its intended notional, over the fund's capital; each draw moves the entry times of EVERY asset's trips at once,
    each asset keeping its own trip count, sides and holding periods (as caps) inside its windows, filled by its own
    pricer, its P&L the sum across the assets. reconciled: (the trips' P&L, the book's moves over them), over the
    fund's capital: further apart than RECONCILE_PP is NOT JUDGED, as are draws held back above HELD_BACK_MAX."""
    from sleeve_fund.research.random_entry import (
        DISAGREE, WEAK_EXPOSURE, RandomEntryResult, _filled_windows, _random_entries, _sharpe)

    rng = np.random.default_rng(seed)
    per_leg = [(leg, _filled_windows(leg.trades, leg.windows)[0]) for leg in legs]
    n = sum(len(w[2]) for _, pw in per_leg for w in pw)
    if n == 0:
        return RandomEntryResult(math.nan, math.nan, 0.0, 0.0, 0, 0.0, draws, False, False, reconciled=reconciled,
                                 why_not=why_not)
    actual, gapped, strategy_stops = [], [], 0
    for leg, pw in per_leg:
        for w in pw:
            k = np.asarray(w[4])
            actual.append(leg.realised[k] * leg.weights[k])
            gapped.append((leg.realised[k] + leg.gaps[k]) * leg.weights[k])
            strategy_stops += int(leg.stops[k].sum())
    actual = np.concatenate(actual)
    strategy_total, gap_total = float(actual.sum()), float(np.concatenate(gapped).sum())
    rets, sharpes = np.empty(draws), np.empty(draws)
    held_back = stopped = priced = 0
    for d in range(draws):
        parts = []
        for leg, pw in per_leg:  # one draw: every asset's entries move together
            for start, end, holds, sides, ks in pw:
                order, entries = _random_entries(rng, start, end, holds)
                fills = leg.pricer.draws(entries, holds[order], sides[order])
                held_back, stopped, priced = held_back + fills.held_back, stopped + fills.stopped, priced + len(entries)
                parts.append(fills.returns * leg.weights[np.asarray(ks)[order]])
        r = np.concatenate(parts)
        rets[d], sharpes[d] = float(r.sum()), _sharpe(r)
    ret_pct = float((rets < strategy_total).mean() * 100)
    sharpe_pct = float((sharpes < _sharpe(actual)).mean() * 100)
    exposure = float(np.mean([leg.in_market for leg in legs])) if legs else 0.0
    return RandomEntryResult(
        return_percentile=ret_pct, sharpe_percentile=sharpe_pct, strategy_return=strategy_total,
        median_random_return=float(np.median(rets)), trades=n, exposure=exposure, draws=draws,
        weak=exposure > WEAK_EXPOSURE, disagree=abs(ret_pct - sharpe_pct) > DISAGREE, reconciled=reconciled,
        held_back=held_back, held_back_share=held_back / priced if priced else 0.0,
        stopped_share=stopped / priced if priced else 0.0, why_not=why_not, strategy_stops=strategy_stops,
        draw_stops=stopped / draws, gap_return=gap_total - strategy_total,
        gap_percentile=float((rets < gap_total).mean() * 100))


def _intended(run: PortfolioResult, name: str) -> tuple[pd.DatetimeIndex, list[float]]:
    """A leg's sent opening entries (kind "entry", approved above 0), by the bar they were decided on, each with its
    intended notional: its requested quantity at its expected fill (as _trips reads them)."""
    opens = sorted((pd.Timestamp(d["bar_ts"]), float(d["requested"]) * float(d["price"])) for d in run.decisions
                   if d["strategy"] == name and d.get("kind", "entry") == "entry" and float(d["approved"]) > 0)
    return pd.DatetimeIndex([b for b, _ in opens]), [x for _, x in opens]


def pooled_legs(m: Measured, names, specs, params: dict, windows, capital: float) -> tuple[list[PooledLeg], tuple,
                                                                                            str]:
    """Each asset's side of the pooled comparison from the fund run, the pooled reconcile (trips' P&L, the book's moves,
    over the fund's capital), and why the draws can't be filled ('' when they can). The draws fill on each asset's own
    execution bars, as its leg did, at its venue fee and half spread, with its funding settlements."""
    from sleeve_fund.research.fills import unmodelled
    from sleeve_fund.research.study import DrawFill, _settlement_rates, _trip_bars, _trip_reconcile
    from sleeve_fund.instruments import taker_slippage

    definition = params.get("definition")
    minutes = specs[0].bar_minutes
    why = "a hand-coded model has no rule definition to fill its draws from" if not definition else \
        unmodelled(definition, minutes) or ""
    clock = specs[0].prices.index
    legs, pnl, moved = [], 0.0, 0.0
    for n, s in zip(names, specs):
        leg = m.run.legs[n]
        # The draws' levels read the warm-up too, as the leg's own did: positions count from the warm-up's first bar.
        bars, off = pd.concat([s.warmup_prices, s.prices]), len(s.warmup_prices)
        idx = bars.index
        # a trip may open on the close before
        ranges = [(off + s0 - 1, off + e) for s0, e in _ranges(clock, windows, _bar(s))]
        taker = float(base_fees(s.instrument, params).taker) if s.fees is None else float(s.fees.taker)
        half = _half_spread(s.instrument, params)
        pricer = None
        if not why:
            fill = DrawFill(definition, minutes, s.exec_prices, _settlement_rates(params, s.instrument, bars), taker)
            pricer = fill.pricer(bars, taker + half, half)
        rows = trades(fills_to_rows(leg.fills), leg.shorts, leg.funding, open_trip=True) \
            if leg.fills is not None and not leg.fills.empty else []
        decided, notionals = _intended(m.run, n)
        placed, realised, weights, gaps, stops, held = [], [], [], [], [], 0
        for start, end in ranges:
            # From the close before the window, or the leg's first mark when that is the warm-up's (nothing held then).
            out = _trip_reconcile(rows, leg.equity, max(idx[start], leg.equity.index[0]), idx[end], [], leg.decisions)
            p, mv, _ = out["trip_reconcile"]
            pnl, moved = pnl + p, moved + mv
            for t in out["trip_returns"]:
                e, x = _trip_bars(idx, t["opened"], t["closed"], end)
                k = int(decided.searchsorted(t["opened"], side="right")) - 1
                placed.append(Trade(e, x, int(t["side"])))
                realised.append(t["ret"])
                weights.append((notionals[k] if k >= 0 else 0.0) / capital)
                gaps.append(t.get("stop_gap", 0.0) - t.get("stop_level", 0.0) * float(taker_slippage(half)))
                stops.append(bool(t.get("stopped")))
                held += x - e
        bars_in = sum(e - s0 + 1 for s0, e in ranges)
        legs.append(PooledLeg(underlying(s.instrument), placed, np.asarray(realised, dtype=float),
                              np.asarray(weights, dtype=float), ranges, pricer, np.asarray(gaps, dtype=float),
                              np.asarray(stops, dtype=bool), held / bars_in if bars_in else 0.0))
    return legs, (pnl / capital, moved / capital), why


def pooled_random(m: Measured, names, specs, params: dict, windows, capital: float, touched: tuple[float, float],
                  draws: int = DRAWS, seed: int = 0):
    """The pooled book against random entries filled as the engine fills (Advisor Q1), NOT JUDGED when the
    compliance tests trimmed or refused GATE_TOUCH_CAP of the intended entry notional or more."""
    from sleeve_fund.research.random_entry import RandomEntryResult

    if draws < DRAWS:
        raise ValueError(f"refused: the random check draws at least {DRAWS:,} times, not {draws:,}")
    cut, asked = touched
    share = cut / asked if asked else 0.0
    legs, reconciled, why = pooled_legs(m, names, specs, params, windows, capital)
    if share >= GATE_TOUCH_CAP:
        why = (f"the compliance tests trimmed {share:.1%} of entry notional ({GATE_TOUCH_CAP:.0%} or more), so "
               "the ungated draws aren't like with like")
    if why:
        trades_n = sum(len(leg.trades) for leg in legs)
        return RandomEntryResult(math.nan, math.nan, 0.0, 0.0, trades_n, 0.0, draws, False, False,
                                 reconciled=reconciled, why_not=why)
    return random_entry_pooled(legs, reconciled, draws, seed)


# ---- STUDY-ASSET-LIST: the hold bar on the pooled book (Advisor Q2, BH-102 rule 3 pooled) --------------------------

PRICE_ONLY = "price-only hold (no spot market at this venue)"
FUNDED = "funded perp hold"
# BH-102 rule 3: until PR E and BH-102 are live, the hold bar's figures are unproven and a basket can't PASS on it.
BH102_LIVE = False
UNPROVEN = "unproven until BH-102 is live"


@dataclass
class HoldBar:
    """The better (by Sharpe over the joined out-of-sample days) of two equal-weight holds of the basket's assets,
    never rebalanced, each the fund's capital split evenly at the first test day: the funded 1x perp hold and the
    price-only hold, chosen once and named. The basket passes it with its pooled Sharpe strictly above the bar's."""

    perp_sharpe: float  # nan: no funded perp hold (a spot market, or a missing funding rate)
    price_sharpe: float
    chosen: str
    bar_sharpe: float
    bar_net: float
    bar_cagr: float
    basket_sharpe: float
    basket_net: float
    basket_cagr: float
    liquidated: list[str] = field(default_factory=list)  # assets whose funded hold was liquidated (flat after)

    @property
    def beats(self) -> bool:
        return bool(self.basket_sharpe > self.bar_sharpe)


def long_only(params: dict) -> bool:
    """No short side: a rule-builder definition without one, and no allow_short (Advisor Q2: only these get a bar)."""
    definition = params.get("definition") or {}
    return not params.get("allow_short") and "short" not in definition


def _cagr(net: float, days: int) -> float:
    return (1 + net) ** (365 / days) - 1 if days and net > -1 else math.nan


def _net(r: pd.Series) -> float:
    return float((1 + r).prod() - 1) if len(r) else 0.0


def hold_bar(base: Measured, specs, params: dict, windows, capital: float, hold=None) -> HoldBar | None:
    """The hold bar for a long-only basket (None for one with a short side): the funded 1x perp hold through the
    engine (each leg's own buy-and-hold run; a liquidated leg stays flat and is named), and the price-only hold of the
    same closes paying the spot taker fee and half spread once in and once out, with no funding and no liquidation.
    hold: (bars, instrument, params, capital) -> a run or its equity, for the funded legs (default: the engine's)."""
    if not long_only(params):
        return None
    from sleeve_fund.research.study import _liquidations

    days = base.returns.index
    keep = days[_oos(days, windows)]
    if not len(keep):
        return None
    first = keep[0] - DAY  # bought at the close before the first test day
    each = capital / len(specs)
    price_parts, perp_parts, liquidated = [], [], []
    perp = markets.is_perp(params)
    for s in specs:
        bars = pd.concat([s.warmup_prices.iloc[-1:], s.prices])  # from the warm-up's last close
        closes = bars["close"]
        if s.bar_minutes != 1440:  # held from a day's close to a day's close, as the daily hold
            closes = closes.resample("1D", closed="right", label="right").last().dropna()
        closes = closes[(closes.index >= first) & (closes.index <= keep[-1])]
        c = float(base_fees(s.instrument, {}).taker) + _half_spread(s.instrument, {})
        value = each * (1 - c) * closes / float(closes.iloc[0])
        value.iloc[-1] *= 1 - c  # out at the last test day's close
        price_parts.append(value)
        if perp:
            run = (hold or partial(_funded_hold, bar_minutes=s.bar_minutes))(bars, s.instrument, params, each)
            eq = run.equity if hasattr(run, "equity") else run
            daily = eq.resample("1D", closed="right", label="right").last().dropna()
            daily = daily[(daily.index >= first) & (daily.index <= keep[-1])]
            perp_parts.append(daily * (each / float(daily.iloc[0])))
            if hasattr(run, "fills") and _liquidations(run, first, keep[-1], "hold"):
                liquidated.append(underlying(s.instrument))

    def returns(parts) -> pd.Series:
        v = pd.concat(parts, axis=1).ffill().sum(axis=1)
        return v.pct_change().reindex(keep).dropna()

    price = returns(price_parts)
    perp_r = returns(perp_parts) if perp_parts else None
    price_s = _sharpe(price.to_numpy())
    perp_s = _sharpe(perp_r.to_numpy()) if perp_r is not None else math.nan
    if perp_r is not None and not (perp_s <= price_s) and not math.isnan(perp_s):
        chosen, bar = FUNDED, perp_r
    else:
        chosen, bar = PRICE_ONLY if perp else "price-only hold", price
    mine = base.returns.reindex(keep).dropna()
    return HoldBar(perp_s, price_s, chosen, _sharpe(bar.to_numpy()), _net(bar), _cagr(_net(bar), len(bar)),
                   _sharpe(mine.to_numpy()), _net(mine), _cagr(_net(mine), len(mine)), liquidated)


def _funded_hold(bars: pd.DataFrame, instrument, params: dict, capital: float, bar_minutes: int = 1440):
    """One asset's funded 1x perp buy-and-hold through the engine, on its decision bars from the warm-up's last
    close."""
    from sleeve_fund.research.runner import run_backtest

    return run_backtest("buy_and_hold", bars, instrument, {"market": markets.market_of(params)},
                        starting_capital=capital, bar_minutes=bar_minutes)


def hold_words(h: HoldBar | None) -> str:
    if h is None:
        return "no hold bar: the basket has a short side (Advisor: long-only baskets only)"
    perp = "n/a" if math.isnan(h.perp_sharpe) else f"{h.perp_sharpe:.2f}"
    out = (f"pooled Sharpe {h.basket_sharpe:.2f} (net {_pct(h.basket_net)}, CAGR {_pct(h.basket_cagr)}) against the "
           f"{h.chosen}'s {h.bar_sharpe:.2f} (net {_pct(h.bar_net)}, CAGR {_pct(h.bar_cagr)}); bar chosen once: "
           f"{FUNDED} {perp}, {PRICE_ONLY} {h.price_sharpe:.2f}")
    if h.liquidated:
        out += f"; the funded hold of {', '.join(h.liquidated)} was liquidated and stayed flat"
    if not BH102_LIVE:
        return f"{out}; {UNPROVEN}"
    return f"{'PASS' if h.beats else 'FAIL'}: {out}"


def pooled_words(r, seed: int = 0) -> str:
    """The filled pooled comparison in words, its verdict first (PR E's, on the pooled book)."""
    from sleeve_fund.research.random_entry import MARGINAL, NOT_JUDGED, STOP_FILL_LINE

    if r.trades == 0:
        return "N/A: no out-of-sample trades to compare"
    if r.why_not:
        return f"{NOT_JUDGED}: {r.why_not}"
    trips, book = r.reconciled
    out = (f"the pooled trips' P&L, {r.strategy_return:+.1%} of the fund's capital, beats {r.return_percentile:.0f}% of "
           f"{r.draws:,} random-entry draws that move every asset's entries at once, each asset keeping its own "
           f"trade count, sides and holding periods (median {r.median_random_return:+.1%}; bar: "
           f"{BAR:.0f}%; seed {seed}); each draw filled as the engine fills the definition's exits on the asset's own "
           f"bars ({r.stopped_share:.0%} of the draws' trips stopped out); the trips add to {trips:+.1%} against the "
           f"book's {book:+.1%} over them{' (more than 0.5 points apart: not judged)' if r.unreconciled else ''}")
    if r.held_back:
        out += f"; {r.held_back} draw trips ({r.held_back_share:.1%}) priced flat, as the engine would have held them back"
    if r.too_many_held_back:
        out += f"; not judged: too many draws held back (above {HELD_BACK_MAX:.0%})"
    if r.weak:
        out += f"; in the market {r.exposure:.0%} of the time, so random entries can hardly differ: a weak test"
    out += (f"; {STOP_FILL_LINE}: {r.strategy_stops} stop exits in the Strategy's trips; at the level its P&L would be "
            f"{r.gap_return * 100:+.2f} points of capital higher")
    verdict = "NOT JUDGED" if r.verdict in ("WEAK", NOT_JUDGED) else r.verdict
    return f"{MARGINAL}: {out}" if r.marginal else f"{verdict}: {out}"


def close_priced_words(r: RandomBeat) -> str:
    """B-38's close-priced comparison, shown without a verdict of its own (HoE ruling 21 Oct: the filled one decides)."""
    from sleeve_fund.research.random_entry import CLOSE_PRICED

    if not r.judged:
        return f"{CLOSE_PRICED}; not judged"
    return (f"the pooled Sharpe {r.sharpe:.2f} is at the {r.percentile:.1f}th percentile of {r.draws:,} random draws "
            f"(their {BAR:.0f}th {r.bar:.2f}); {CLOSE_PRICED}, no verdict")


def _without_sol_lines(study: BasketStudy) -> list[str]:
    w = getattr(study, "without_sol", None)
    if not w:
        return []
    r = w["random"]
    pct = "not judged" if r.why_not or r.trades == 0 else f"{r.return_percentile:.0f}th percentile of the filled draws"
    return [f"**Without SOL** ({', '.join(w['assets'])}; its own re-run, diagnostic only, never a condition or a try): "
            f"net {_pct(w['net'])}, Sharpe {w['sharpe']:.2f}, random entry {pct}.", ""]
