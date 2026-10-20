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
from sleeve_fund.research.random_entry import BAR, DRAWS, _random_entries
from sleeve_fund.research.candidates import BUDGET_PER_IDEA, configurations
from sleeve_fund.research.study import COST_LADDER, _between, doubled
from sleeve_fund.risk import PORTFOLIO, PortfolioProfile

GATE_TOUCH_CAP = 0.05  # Advisor fork 3: the ungated random benchmark is like-with-like only below this
CONCENTRATED = 0.5  # the top asset's share of the pooled out-of-sample net P&L above which the basket is flagged
NEXT = "next: 4-6 weeks paper"
DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class BasketAsset:
    """One asset's whole history: daily decision bars and the hourly bars the engine fills against, each stamped at
    its close. The study cuts each run's window (and its warm-up before it) from these."""

    instrument: Any
    prices: pd.DataFrame
    exec_prices: pd.DataFrame


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
    """A round trip on an asset's daily closes: in at bar `entry`'s close, out at bar `exit`'s, `notional` at entry."""

    entry: int
    exit: int
    side: int
    notional: float


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
    """The bars of the days `first` to `last` (a daily bar closing at 00:00 is the day before's), and the `warm_days`
    daily bars before them; nothing after `last`."""
    start, end = pd.Timestamp(first, tz="UTC"), pd.Timestamp(last + timedelta(days=1), tz="UTC")
    p, h = a.prices, a.exec_prices
    warm = p[p.index <= start].iloc[-warm_days:]
    if len(warm) < warm_days:
        raise ValueError(f"{underlying(a.instrument)} has {len(warm)} days of history before {first}, not the "
                         f"{warm_days} days of warm-up the run needs")
    window = p[(p.index > start) & (p.index <= end)]
    if not len(window) or window.index[-1] != end:
        raise ValueError(f"{underlying(a.instrument)}'s history doesn't reach {last}")
    return AssetData(a.instrument, window, h[(h.index > start) & (h.index <= end)], warm)


def _run(basket: Basket, assets: list[BasketAsset], first: date, last: date, fees, capital: float,
         profile: PortfolioProfile, warm_days: int) -> tuple[PortfolioResult, list[str], list[LegSpec]]:
    """One fund run of `assets` over the days `first` to `last`. fees: a schedule for every leg, or None for each
    leg's venue fees. The fund's capital is split evenly across the legs' accounts; every leg sizes off the fund."""
    data = [_cut(a, first, last, warm_days) for a in assets]
    if len(data) > 1:
        check_assets(data)
    names = leg_names(basket.name, [d.instrument for d in data])
    specs = [LegSpec(n, basket.strategy, d.instrument, d.prices, d.exec_prices, d.warmup_prices,
                     params=dict(basket.params), capital=capital / len(data), fees=fees)
             for n, d in zip(names, data)]
    return run_portfolio(specs, profile, fund_sized=True), names, specs


def pooled_returns(pnl: pd.DataFrame, capital: float) -> pd.Series:
    """The pooled daily net return: the day's net P&L summed across the assets, over the fund's equity at the day
    before's close (the capital plus every P&L booked before)."""
    day = pnl.sum(axis=1)
    before = capital + day.cumsum().shift(1, fill_value=0.0)
    return day / before


def _daily_pnl(run: PortfolioResult, names: list[str], specs: list[LegSpec], assets: list[str]) -> pd.DataFrame:
    cols = {}
    for n, s, a in zip(names, specs, assets):
        eq = run.legs[n].equity
        closes = eq.resample("1D", closed="right", label="right").last().dropna()
        closes = closes[closes.index <= s.prices.index[-1]]
        cols[a] = closes.diff().fillna(closes.iloc[0] - s.capital)
    return pd.DataFrame(cols).fillna(0.0)


def _oos(index: pd.DatetimeIndex, windows) -> np.ndarray:
    days = [(t - DAY).date() for t in index]
    return np.array([any(a <= d <= b for a, b in windows) for d in days], dtype=bool)


def _measure(run, names, specs, assets, windows, capital) -> Measured:
    pnl = _daily_pnl(run, names, specs, assets)
    pnl = pnl[_oos(pnl.index, windows)]
    returns = pooled_returns(pnl, capital)
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
    entry's close to its exit's, paying cost_per_side on its entry notional and on its exit notional."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    pos, cost = np.zeros(n + 1), np.zeros(n)
    for t in trips:
        q = t.notional / c[t.entry]
        pos[t.entry + 1] += q * t.side
        pos[t.exit + 1] -= q * t.side
        cost[t.entry] += cost_per_side * t.notional
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
        out.extend(TradeAt(int(e), int(e + holds[k]), inside[k].side, inside[k].notional)
                   for e, k in zip(entries, order))
    return out


def _trips(run: PortfolioResult, name: str, spec: LegSpec, params: dict) -> list[TradeAt]:
    """A leg's closed round trips on its daily closes, at their intended notional: in at the last close at or before
    the entry fill, out at the last close at or before the exit fill, at least one bar. The intended notional is what
    the entry asked the compliance tests for, its requested quantity at its expected fill: the leg's entries that
    were sent (approved above 0), in order, are its trips, in order."""
    idx = spec.prices.index
    sent = [float(d["requested"]) * float(d["price"]) for d in run.decisions
            if d["strategy"] == name and float(d["approved"]) > 0]
    out = []
    for k, t in enumerate(trades(fills_to_rows(run.legs[name].fills), shorts=markets.is_perp(params))):
        if t["closed"] is None:
            continue
        e = max(int(idx.searchsorted(pd.Timestamp(t["opened"]), side="right")) - 1, 0)
        x = max(int(idx.searchsorted(pd.Timestamp(t["closed"]), side="right")) - 1, e + 1)
        if x < len(idx):
            out.append(TradeAt(e, x, int(t["side"]), sent[k] if k < len(sent) else float(t["cost"])))
    return out


def clip(trips: list[TradeAt], ranges: list[tuple[int, int]]) -> list[TradeAt]:
    """Each trade cut to the test window it overlaps (the fold-edge rule: only its marks inside the windows count),
    the same for the basket's trades and the draws'. A trade outside every window is dropped."""
    out = []
    for t in trips:
        for start, end in ranges:
            e, x = max(t.entry, start), min(t.exit, end)
            if x > e:
                out.append(TradeAt(e, x, t.side, t.notional))
                break
    return out


def _ranges(index: pd.DatetimeIndex, windows) -> list[tuple[int, int]]:
    """The out-of-sample days as runs of bar positions, (first, last) inclusive: windows that abut are one run, so a
    trade across a fold's edge is drawn with the rest."""
    inside = _oos(index, windows)
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
    cut, asked = touched
    share = cut / asked if asked else 0.0
    if share >= GATE_TOUCH_CAP:
        return RandomBeat(False, math.nan, False, draws, share, seed=seed)
    clock = specs[0].prices.index
    keep = _oos(clock, windows)
    ranges = _ranges(clock, windows)
    legs = []
    for n, s in zip(names, specs):
        closes = s.prices["close"].to_numpy(dtype=float)
        cps = float(base_fees(s.instrument, params).taker) if s.fees is None else float(s.fees.taker)
        legs.append((closes, clip(_trips(m.run, n, s, params), ranges), cps + _half_spread(s.instrument, params)))
    if not any(t for _, t, _ in legs):
        return RandomBeat(False, math.nan, False, draws, share, seed=seed)
    book = daily_closes(m.run.book)
    equity = book.shift(1).reindex(clock).fillna(capital).to_numpy(dtype=float)

    def pooled(trips_by_leg) -> float:
        return _sharpe(pooled_trade_returns([(c, t, cps) for (c, _, cps), t in zip(legs, trips_by_leg)], equity)[keep])

    actual = pooled([t for _, t, _ in legs])
    rng = np.random.default_rng(seed)
    drawn = np.array([pooled([draw_trades(rng, t, ranges) for _, t, _ in legs]) for _ in range(draws)])
    bar = float(np.nanpercentile(drawn, BAR))
    pct = float((drawn < actual).mean() * 100)
    return RandomBeat(True, pct, bool(actual > bar), draws, share, actual, float(np.nanmedian(drawn)), bar, seed)


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
    settings."""
    return trials.content_hash({"strategy": basket.strategy, "params": basket.params, "assets": basket.names})


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
                 ladder: bool = True, draws: int = DRAWS, seed: int = 0) -> BasketStudy:
    """The basket's study over the walk-forward's test windows ((first day, last day), in order), never reading a bar
    after research_end. Its holdout stays sealed: open_holdout is the separate final step."""
    windows = tuple(tuple(w) for w in windows)
    if not windows or windows[-1][1] > research_end:
        raise ValueError(f"the test windows must end by the research end {research_end}")
    first, last = windows[0][0], windows[-1][1]
    assets = list(basket.assets)
    names_of = basket.names

    def measured(subset, fees):
        run, names, specs = _run(basket, subset, first, last, fees, capital, profile, warm_days)
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
    study.tries, study.configuration = record_try(
        register, basket, dataset, _since(first, warm_days, basket), pd.Timestamp(last + timedelta(days=1), tz="UTC"),
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
    failed = (study.random.judged and not study.random.beats) or not study.double_fees.passed \
        or not study.left_out.passed or (holdout is not None and not holdout.passed)
    if failed:
        return "FAIL"
    if not study.random.judged:
        return "NOT JUDGED"
    return "NOT YET" if holdout is None else "PASS"


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.1%}"


def render(study: BasketStudy, holdout: Holdout | None = None) -> str:
    """The basket's report. The verdict lines are the pooled run's; per-asset lines are diagnostic only."""
    b, base = study.basket, study.base
    days = len(base.returns)
    v = verdict(study, holdout)
    head = {"NOT YET": "NOT YET: the locked holdout is sealed until its own final step",
            "NOT JUDGED": "NOT JUDGED: the random benchmark couldn't be judged, so the basket can't pass",
            "PASS": "PASS", "FAIL": "FAIL"}[v]
    lo = study.left_out
    lines = [
        f"# Basket study: {b.name} ({b.strategy} on {', '.join(b.names)})", "",
        f"**Basket verdict:** {head}", "",
        f"_{FIXED_LIST.format(date=b.universe_date)}_", "",
        f"**Tries:** {study.tries} (one per fixed basket and settings); this basket is try {study.configuration} of "
        f"{BUDGET_PER_IDEA} for {b.strategy}'s family", "",
        f"Out-of-sample: {study.windows[0][0]} to {study.windows[-1][1]}, {days} days, one fund run of the whole "
        f"basket at the venue's fees: net {_pct(base.net)}.", "",
        "| Condition | Result |", "| --- | --- |",
        f"| Beats random entries (95th percentile) | {random_words(study.random)} |",
        f"| Net above 0 at 2x the venue's fees | {'PASS' if study.double_fees.passed else 'FAIL'}, net "
        f"{_pct(study.double_fees.net)} |",
        f"| Best asset left out ({lo.asset}, {lo.share:.0%} of the pooled net) | {'PASS' if lo.passed else 'FAIL'}, "
        f"net {_pct(lo.net)}{'; concentrated: over half the net from one asset' if lo.concentrated else ''}"
        f"{'; info, without it: ' + random_words(lo.random) if lo.random is not None else ''} |",
        "| Locked holdout at the venue's fees | " + (
            "sealed: opened once, as its own final step" if holdout is None else
            f"{'PASS' if holdout.passed else 'FAIL'}, net {_pct(holdout.net)} ({holdout.period[0]} to "
            f"{holdout.period[1]})") + " |", "",
        f"The compliance tests trimmed or refused {study.gate[0]:,.2f} of {study.gate[1]:,.2f} intended entry notional "
        f"on the test days ({study.gate_touched:.1%} of entry notional).", "",
        f"**Pooled break-even fee:** {study.breakeven[1]} (one fee per side on every fill, real runs on the cost "
        "ladder)", "",
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
