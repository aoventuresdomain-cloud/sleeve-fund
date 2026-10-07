"""A full G1 study for one idea on one dataset.

1. Reserve the most recent `holdout_days` as a holdout. It is never touched
   unless the PM asks, and the ledger records when it was.
2. Parameter sensitivity: every grid point over the whole research period.
3. Walk-forward: on each rolling fold pick the best params on the train
   window, then trade them on the following test window. The stitched test
   windows are the out-of-sample result the tear sheet leads with.
"""

from __future__ import annotations

import itertools
import logging
import math
from decimal import Decimal
from dataclasses import dataclass, field

import pandas as pd
from nautilus_trader.model import CurrencyPair

from sleeve_fund import funding
from sleeve_fund.instruments import FeeSchedule, pair_of
from sleeve_fund import markets
from sleeve_fund.markets import PERP
from sleeve_fund.research.ledger import IdeaLedger, opened_words
from sleeve_fund.research.random_entry import RandomEntryResult, RandomSideResult, Trade, random_entry, random_side
from sleeve_fund.research.metrics import (
    daily_returns,
    fills_to_rows,
    round_trips,
    summary,
    trade_stats,
    trades,
    turnover_per_year,
    whole_days,
)
from sleeve_fund.research.runner import BacktestResult, run_backtest
from sleeve_fund.strategies import check_perp_sizing
from sleeve_fund.strategies.base import IdeaSpec


log_ = logging.getLogger(__name__)


def _window_funding(marks: list, after: pd.Timestamp, end: pd.Timestamp) -> dict:
    """A window's funding, its settlements in (after, end], `after` being the close of the bar before its first:
    of its held settlements, those charged the baseline for a missing rate, and the longest stretch of held time
    without the venue's rate (funding.baseline_summary)."""
    inside = [m for m in marks if after < pd.Timestamp(m[0]) <= end]
    n, m, longest = funding.baseline_summary(inside)
    return {"funding_held": m, "funding_baseline": n, "funding_gap": longest, "funding_window_marks": inside}


def _oos_funding(folds) -> tuple[int, int, pd.Timedelta]:
    """(N, M, the longest stretch) over the out-of-sample windows joined: the test windows are contiguous, so a
    stretch without the venue's rate runs on across a window's edge, broken only by a stored real rate (Advisor,
    18:18; QA P1-O17a-2). Folds without their marks (older results) fall back to the per-window figures."""
    n = sum(getattr(f, "funding_baseline", 0) for f in folds)
    m = sum(getattr(f, "funding_held", 0) for f in folds)
    joined = sorted((mk for f in folds for mk in getattr(f, "funding_window_marks", ())), key=lambda mk: mk[0])
    if joined:
        return n, m, funding.baseline_summary(joined)[2]
    return n, m, max((getattr(f, "funding_gap", pd.Timedelta(0)) for f in folds), default=pd.Timedelta(0))


def _funding_check(r) -> tuple[str, str]:
    """PASS, WARN or NOT JUDGED on a study's out-of-sample funding (funding.baseline_check; Advisor, 6 Oct 2026). A
    simulated perp has no venue's rates at all: every settlement is the baseline, and it is not judged until a
    modelled rate series is in the trials register before the run (none can be registered yet). Nor is a run whose
    venue's settlements don't fit the schedule charged (funding.schedule_mismatch)."""
    n, m, longest = _oos_funding(r.folds)  # read with defaults: a fold that never held a perp carries no funding
    verdict, words = funding.baseline_check(n, m, longest, "out-of-sample")
    full = getattr(r, "full_period", None)
    if getattr(full, "funding_schedule", ""):
        return "NOT JUDGED", f"{full.funding_schedule}; {words}"
    if getattr(full, "funding_simulated", False):
        return "NOT JUDGED", (f"a simulated perpetual, with no venue's funding rates: {words}; no modelled rate "
                              "series was in the trials register before the run")
    return verdict, words


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    chosen: dict
    train_sharpe: float
    test: dict
    benchmark_test: dict | None  # None: the perp hold isn't priced over this window (its funding isn't all known)
    # Round trips opened and closed inside the test window: what out-of-sample is judged on. A trip carried in
    # from training, or still open when the window ends, is left out and counted (Advisor, 5 Oct 2026).
    test_trades: int = 0
    carried_in: int = 0
    carried_out: int = 0
    # The counted trips as (opened, closed, side), for the random-entry and random-side benchmarks.
    trips: list = field(default_factory=list)
    # The test window's bars, and how many of them held any position, carried-in and still-open ones too: the
    # random-entry benchmark's "in the market" (Independent Quant Advisor, 6 Oct 2026, QA P1-R2).
    window_bars: int = 0
    in_market_bars: int = 0
    # When the risk guard halted the run before the test window ended: the day, why, and whether
    # it was in the training stretch the run traded through first. A halted fold is flat from then on.
    halted: str = ""
    halted_before_test: bool = False  # halted in the training stretch: the whole test window sat flat
    # Every grid point's Sharpe and round trips on this fold's training stretch: the surface the choice
    # was made on, so the nearby-settings check can centre on what this fold chose (P1-G2).
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)
    # A perpetual's funding settlements a position was held through in the test window, how many of them were
    # charged the baseline for a missing rate, and the longest stretch of held time without the rate (QA P1-O17).
    funding_held: int = 0
    funding_baseline: int = 0
    funding_gap: pd.Timedelta = pd.Timedelta(0)
    funding_window_marks: list = field(default_factory=list, repr=False)  # (ts, missing, held) in the window
    unscored: bool = False  # no setting scored on training, so nothing was chosen and the test window sat flat
    # Every liquidation in the test window (_liquidations): a G1 finding whatever the P&L (GAP-LIQ-CAP).
    liquidations: list = field(default_factory=list)

    @property
    def closed_in_window(self) -> int:
        """Round trips closed inside the test window, carried in or not: whether the window traded at all."""
        return self.test_trades + self.carried_in


@dataclass
class StudyResult:
    spec: IdeaSpec
    dataset: str
    synthetic: bool
    research_start: pd.Timestamp
    research_end: pd.Timestamp
    holdout_days: int
    default_params: dict
    full_period: BacktestResult
    full_period_benchmark: BacktestResult
    sensitivity: pd.DataFrame
    folds: list[Fold]
    oos_returns: pd.Series
    oos_benchmark_returns: pd.Series
    holdout: dict | None = None
    holdout_benchmark: dict | None = None
    fee_note: str = ""
    fee_basis: FeeSchedule | None = None  # the schedule the strategy's runs paid: its market's, or the instrument's
    notes: list[str] = field(default_factory=list)
    instrument: str = ""  # BASE/QUOTE, and the bar length tested: a G1 pass counts for exactly these
    bar_minutes: int = 1440
    venue: str = ""
    risk_profile: str | None = None
    settings: str = ""  # the risk profile, exits and windows the study ran with, so two sheets can be told apart
    # Exceptions the strategy raised in its runs, as (run, handler, repr) for each run that raised, and
    # how many in all: results built on a broken handler can't be judged (review round 8, M8-3).
    errors: list = field(default_factory=list)
    error_count: int = 0
    holdout_withheld: str = ""  # why the holdout asked for was left closed
    # The holdout run's funding (_window_funding), counted and judged as out-of-sample is (Advisor, 6 Oct 2026).
    holdout_funding: dict = field(default_factory=dict)
    # The chosen settings over the research period at each fee of COST_LADDER: what costs the idea survives.
    cost_ladder: list[LadderRung] = field(default_factory=list)
    ladder_slippage: float = 0.0  # charged on every rung on top of the half spread, on orders that take liquidity
    # The settings tuning on the whole research period picks (best in-sample Sharpe on the grid): what the
    # cost ladder and the final nearby-settings check centre on, never the defaults (P1-G2).
    chosen_params: dict = field(default_factory=dict)
    breakeven: str = ""  # the chosen settings' break-even fee in words, re-run to verify it (P1-G1)
    # Out-of-sample timing against random entry times, and for strategies that can go short, side against random
    # sides (v2 P1-7, C3 and C3b).
    random_entry: RandomEntryResult | None = None
    random_side: RandomSideResult | None = None
    # Runs of this idea whose trials-register count failed (QA P1-T8): they count in N, but their Sharpes are
    # missing from the spread the bar is set by, so G1 can't judge until they are re-counted (Advisor, 6 Oct 2026).
    failed_counts: int = 0
    # The buy and hold G1 compares with (Advisor, 7 Oct 2026, RE-COST): on a perpetual, a 1x long perp hold paying the
    # perp's fees and funding, through the same engine, priced only over windows whose funding is all known.
    # oos_compare_returns: the strategy's out-of-sample days the hold covers, what the benchmark checks compare.
    hold_market: str = "spot"
    hold_note: str = ""  # what the hold pays for funding, and how much of out-of-sample it covers
    hold_insufficient: bool = False  # covers under HOLD_COVERAGE_FLOOR of the out-of-sample days: no verdict
    hold_full_period: bool = True  # the hold covers the whole research period (the parameter-robustness check)
    oos_compare_returns: pd.Series | None = None
    spot_hold_returns: pd.Series | None = None  # "holding spot instead", out-of-sample, shown beside a perp hold
    funding_stress: dict | None = None  # a simulated perp's research period at FUNDING_STRESS_RATE

    @property
    def compare_returns(self) -> pd.Series:
        return self.oos_returns if self.oos_compare_returns is None else self.oos_compare_returns

    # Every liquidation in out-of-sample (the folds' test windows) or the opened holdout, as _liquidations gives
    # them: each a G1 finding whatever the P&L (Independent Quant Advisor 6 Oct 23:42, 7 Oct 00:19 (5)).
    holdout_liquidations: list = field(default_factory=list)

    @property
    def liquidations(self) -> list[dict]:
        return [liq for f in self.folds for liq in getattr(f, "liquidations", [])] + list(
            getattr(self, "holdout_liquidations", []))

    @property
    def not_judged(self) -> str:
        """Why G1 can't judge this study, in words, or '' when it can (review round 8, M8-3 and M8-4):
        - the strategy raised errors, so its orders after them may be wrong;
        - the risk guard left half or more of the test windows blind, or out-of-sample without a trade. A window
          is blind when a halt before it began kept it flat, or a halt left it without a trade: no
          information, and failing on it would spend the idea (review round 9, N7). A halt inside a
          test window that still traded is a result, and so is a window the signal never traded in."""
        if self.error_count:
            from sleeve_fund.strategies.base import handler_error_words

            run, handler, what = self.errors[0]
            return (f"the strategy raised {self.error_count} error{'s' if self.error_count != 1 else ''} in "
                    f"{len(self.errors)} of its runs, the first {handler_error_words(handler, what)}, so its "
                    "orders after that may be wrong")
        k = getattr(self, "failed_counts", 0)  # a stand-in result (tests) may lack it
        if k:
            return (f"N uncertain: {k} earlier run{'s' if k != 1 else ''} of this idea ran but couldn't be counted in "
                    "full, so the bar is missing their Sharpes until they are re-counted")
        blind = [f for f in self.folds if f.halted_before_test or (f.halted and f.closed_in_window == 0)]
        # Half blind is not judged either: the other half then holds the halted windows' stubs too, so the
        # verdict would rest on a window or two (review round 10, M10-1).
        if blind and (2 * len(blind) >= len(self.folds) or self.oos_trades == 0):
            halted = sum(1 for f in self.folds if f.halted)
            return (f"the risk guard halted the strategy in {halted} of {len(self.folds)} folds, leaving {len(blind)} "
                    f"of {len(self.folds)} test windows flat or without a trade, and out-of-sample closed "
                    f"{self.oos_trades} trade{'s' if self.oos_trades != 1 else ''}, so at least half of it sat flat "
                    "rather than testing the idea")
        verdict, words = _funding_check(self)
        if verdict == "NOT JUDGED":
            return f"its funding isn't the venue's: {words}"
        return ""

    @property
    def funding_baseline(self) -> tuple[int, int, pd.Timedelta]:
        """Out-of-sample funding settlements charged the baseline for a missing rate, of those a position was held
        through, and the longest stretch of held time without the venue's rate: what the G1 funding check reads."""
        return _oos_funding(self.folds)

    @property
    def funding_check(self) -> tuple[str, str]:
        return _funding_check(self)

    @property
    def holdout_funding_baseline(self) -> tuple[int, int, pd.Timedelta]:
        h = self.holdout_funding
        return h.get("funding_baseline", 0), h.get("funding_held", 0), h.get("funding_gap", pd.Timedelta(0))

    @property
    def holdout_not_judged(self) -> str:
        """Why the opened holdout can't be read, or '': the same funding rule as out-of-sample, since the holdout is
        the last check before G1 means anything (Advisor, 6 Oct 2026)."""
        verdict, words = funding.baseline_check(*self.holdout_funding_baseline, "holdout")
        return words if self.holdout and verdict == "NOT JUDGED" else ""

    @property
    def round_trips(self) -> list[float]:
        return round_trips(self.full_period.fills, self.full_period.shorts)

    @property
    def oos_trades(self) -> int:
        """Round trips opened and closed inside the walk-forward test windows: the trades out-of-sample stands on."""
        return sum(f.test_trades for f in self.folds)

    @property
    def excluded_trades(self) -> int:
        """Round trips at the test windows' edges, left out of the count: carried in from training, or still
        open when the window ended."""
        return sum(f.carried_in + f.carried_out for f in self.folds)

    @property
    def trade_stats(self) -> dict:
        return trade_stats(trades(fills_to_rows(self.full_period.fills)))

    @property
    def turnover(self) -> float:
        return turnover_per_year(self.full_period.fills, self.full_period.equity)


# Fee per side the cost ladder tests every idea at (PM, 5 Oct 2026): free, the low-fee perp venues' maker
# and taker rates, a mid venue, and a high-fee spot venue's taker rate (the stress case). The Independent
# Quant Advisor (6 Oct 2026, P1-G1) added 0.15-0.40%, where most spot taker fees sit: return bends with the
# fee, so a wide gap there misplaced the break-even.
COST_LADDER = (0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.008)
# Slippage beyond the spread each rung also pays on orders that take liquidity (strategy sprint, PM approved
# 5 Oct 2026): 2 basis points on the deepest books (BTC, ETH), 5 on the rest.
DEEP_BOOKS = ("BTC", "ETH")
# A re-run at the computed break-even confirms it when its net return is within this of zero (P1-G1).
BREAKEVEN_TOLERANCE = 0.0001  # 0.01% of starting capital, QA's bar (m-G6)
BREAKEVEN_RUNS = 6  # re-runs allowed to home in on it before the figure stays "interpolated"


def ladder_slippage(pair: str) -> float:
    return 0.0002 if pair.split("/")[0].upper() in DEEP_BOOKS else 0.0005


@dataclass
class LadderRung:
    fee: float  # charged per side, maker and taker alike
    total_return: float
    sharpe: float
    round_trips: int
    fees_paid: float
    # The random-entry benchmark at this rung's own cost (RE-COST): the strategy's out-of-sample trips priced on the
    # bars' closes, and the median of the random draws, both paying this fee plus the rung's spread and slippage.
    oos_timing_return: float | None = None
    random_return: float | None = None


def _log_growth(total_return: float) -> float:
    return math.log1p(max(total_return, -0.999999))


def _between(lo_fee: float, lo_ret: float, hi_fee: float, hi_ret: float) -> float:
    """Where log(1 + return) crosses zero between two fees. Fees compound with every trade, so return is
    convex in the fee and a straight line through return overstates the break-even; log growth falls
    about linearly with the fee (QA P1-G1: 0.291% reported against a true 0.163%)."""
    a, b = _log_growth(lo_ret), _log_growth(hi_ret)
    return lo_fee + (hi_fee - lo_fee) * a / (a - b)


def breakeven_fee(rungs: list[LadderRung], run_at=None) -> tuple[float | None, str]:
    """The fee per side at which the idea stops making money over the period, and that in words. Found
    between the ladder's two rungs either side of zero return, interpolating log(1 + return). With
    `run_at(fee) -> total return`, the figure is re-run to confirm the net is about zero, homing in between
    the rungs when it isn't, and called "verified"; without it, "interpolated". None when it made no trades,
    loses money even at no fee, or still makes money at the top rung."""
    rungs = sorted(rungs, key=lambda r: r.fee)
    if not rungs:
        return None, "not tested"
    if rungs[0].round_trips == 0 and rungs[-1].fees_paid == 0:  # never filled: no trades, not a loss (QA F6)
        return None, "made no trades, so there is no fee to break even on"
    if rungs[0].total_return <= 0:
        return None, f"loses money even at {rungs[0].fee:.2%} fees"
    for lo, hi in zip(rungs, rungs[1:]):
        if hi.total_return <= 0:
            span = f"between {lo.fee:.2%} and {hi.fee:.2%}"
            lo_fee, lo_ret, hi_fee, hi_ret = lo.fee, lo.total_return, hi.fee, hi.total_return
            fee = _between(lo_fee, lo_ret, hi_fee, hi_ret)
            if run_at is None:
                return fee, f"stops making money at about {fee:.3%} per side (interpolated {span})"
            for _ in range(BREAKEVEN_RUNS):
                ret = run_at(fee)
                if abs(ret) <= BREAKEVEN_TOLERANCE:
                    return fee, (f"stops making money at about {fee:.3%} per side (verified: re-run at that fee, "
                                 f"net return {ret:+.3%}, within {BREAKEVEN_TOLERANCE:.2%})")
                if ret > 0:
                    lo_fee, lo_ret = fee, ret
                else:
                    hi_fee, hi_ret = fee, ret
                fee = _between(lo_fee, lo_ret, hi_fee, hi_ret)
            return fee, (f"stops making money at about {fee:.3%} per side (interpolated {span}; "
                         f"{BREAKEVEN_RUNS} re-runs did not settle within {BREAKEVEN_TOLERANCE:.2%})")
    return None, f"still makes money at {rungs[-1].fee:.2%} per side, the top of the ladder"


def grid(param_grid: dict[str, list]) -> list[dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(param_grid[k] for k in keys))]
    # Strategies may reject nonsensical combos (e.g. fast >= slow); filter those here.
    return [c for c in combos if not ("fast" in c and "slow" in c and c["fast"] >= c["slow"])]


def bar_minutes_of(prices: pd.DataFrame) -> int:
    """The bar length of a price table, from the usual gap between its rows."""
    if len(prices) < 2:
        return 1440
    return max(1, round(pd.Series(prices.index).diff().median() / pd.Timedelta(minutes=1)))


def _bar_words(minutes: int) -> str:
    return "day" if minutes == 1440 else f"{minutes // 60}-hour bar" if minutes % 60 == 0 else f"{minutes}-minute bar"


def run_study(
    spec: IdeaSpec,
    prices: pd.DataFrame,
    instrument: CurrencyPair,
    dataset: str,
    ledger: IdeaLedger,
    default_params: dict | None = None,
    synthetic: bool = False,
    holdout_days: int = 365,
    train_days: int = 3 * 365,
    test_days: int = 365,
    use_holdout: bool = False,
    starting_capital: float = 10_000.0,
    exits: dict | None = None,
    position_cap: float | None = None,
    risk_profile: str | None = None,
    exec_prices: pd.DataFrame | None = None,
    half_spread: float | None = None,
    progress=None,
    register=None,
    locks=None,
) -> StudyResult:
    """prices: bars of any length that divides a day (daily, hourly, 15-minute, 1-minute...). The holdout,
    train and test windows are in days whatever the bar, and every statistic is on daily returns.
    exits: optional stop_loss / take_profit / risk_per_trade applied to every strategy run.
    position_cap: the largest share of capital in the position, as the paper risk profile allows. It
    applies to the benchmark too, so G1 compares the strategy with buy and hold at the same exposure.
    risk_profile: run every strategy backtest under the paper runtime with this profile, as the backtest
    page and paper do: its position cap (which then sets position_cap), drawdown halt and daily-loss
    pause. The benchmark is held at the same cap but never halted: it is what holding would have done.
    exec_prices: shorter bars over the same period (from the history store), which the engine matches
    resting orders against and the risk guard values the book on, between decision bars.
    half_spread: the spread charged on orders that take liquidity (sleeve_fund.spreads.resolve gives
    the measured one); None uses the venue's assumption.
    progress: called with the share of the study's backtests done, 0 to 1."""
    from sleeve_fund.venues import VENUES

    perpetual = getattr(VENUES.get(str(instrument.id.venue)), "perpetual", False)
    for combo in grid(spec.param_grid) or [{}]:  # refused up front, before any data is read or run
        check_perp_sizing(spec.name, {**(default_params or spec.default_params or {}), **combo,
                                      **({"market": PERP} if perpetual else {})})
    minutes = bar_minutes_of(prices)
    if 1440 % minutes:
        raise ValueError(f"{minutes}-minute bars don't divide a day; use 1, 5, 15, 30, 60, 240 or 1440")
    per_day = 1440 // minutes
    bar = pd.Timedelta(minutes=minutes)
    holdout_bars, train_bars, test_bars = holdout_days * per_day, train_days * per_day, test_days * per_day
    if len(prices) < holdout_bars + train_bars + test_bars:
        raise ValueError(
            f"{len(prices) / per_day:,.0f} days of bars is too short for holdout {holdout_days} + train "
            f"{train_days} + test {test_days} days"
        )
    research = prices.iloc[:-holdout_bars] if holdout_bars else prices
    combos = grid(spec.param_grid)
    default_params = default_params or spec.default_params or (combos[0] if combos else {})

    exits = {k: v for k, v in (exits or {}).items() if v is not None}
    from sleeve_fund.venues import VENUES

    profile = VENUES.get(str(instrument.id.venue))
    # A venue that lists perpetuals only has no spot: every run, the benchmark too, holds its perpetual.
    market = {"market": PERP} if profile is not None and profile.perpetual else {}
    strategy_cap = None
    if risk_profile is not None:
        from sleeve_fund.risk import position_cap as cap_of
        from sleeve_fund.risk import profile as risk_profile_of

        # On a perpetual, the margin cap times the leverage cap, as paper and the backtest page size it; the
        # benchmark holds that exposure up to all of the capital, as the backtest page's does (review round 13, E13-1).
        strategy_cap = cap_of(risk_profile_of(risk_profile), market)
        position_cap = min(strategy_cap, 1.0)
    if position_cap is not None and not 0 < position_cap <= 1:
        raise ValueError(f"position_cap {position_cap} outside (0, 1]")
    if half_spread is None:
        from sleeve_fund.venues import venue as venue_profile

        half_spread = venue_profile(str(instrument.id.venue)).assumed_half_spread
    spread_used = half_spread

    def paid(params: dict) -> FeeSchedule:
        """The schedule a run with these settings pays (runner.run_backtest's own default: its market's, or the
        instrument's). The benchmarks, the fee note and the variant's key take it too, so a perp study's strategy and
        its random entries pay the same fee on the same trades (RE-COST)."""
        return markets.fees_for({**params, **market}, FeeSchedule(instrument.maker_fee, instrument.taker_fee),
                                str(instrument.id.venue))

    exec_minutes = bar_minutes_of(exec_prices) if exec_prices is not None and len(exec_prices) else None
    if exec_minutes is not None and (exec_minutes >= minutes or minutes % exec_minutes):
        raise ValueError(f"{exec_minutes}-minute execution bars don't divide the {minutes}-minute decision bars")

    errors: list = []
    error_count = [0]
    folds_n = max(0, (len(research) - train_bars - test_bars) // test_bars + 1)
    total = (1 + len(combos) * (1 + len(COST_LADDER)) + 2 + folds_n * (len(combos) + 1) + len(COST_LADDER)
             + (2 if use_holdout and holdout_days else 0))
    done = [0]

    def bt(name: str, df: pd.DataFrame, params: dict, benchmark: bool = False,
           fees: FeeSchedule | None = None, slippage: float = 0.0) -> BacktestResult:
        done[0] += 1
        if progress is not None:
            progress(min(done[0] / total, 0.99))
        guarded = risk_profile is not None and not benchmark
        params = {**params, **market}
        if not benchmark:
            params = {**params, **exits}
        if position_cap is not None and not guarded:
            params = {**params, "position_cap_pct": position_cap}
        fine = None
        if exec_minutes is not None and not benchmark:  # nothing rests, nothing to guard
            fine = exec_prices[(exec_prices.index > df.index[0] - bar) & (exec_prices.index <= df.index[-1])]
        res = run_backtest(name, df, instrument, params, starting_capital=starting_capital, bar_minutes=minutes,
                           risk_profile=risk_profile if guarded else None, exec_prices=fine,
                           exec_minutes=exec_minutes or 1, half_spread=half_spread + slippage, fees=fees)
        if res.handler_errors:
            errors.append((f"{name} {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y}", *res.handler_errors[0]))
            error_count[0] += res.handler_error_count or len(res.handler_errors)
        return res

    def log(params: dict, stage: str, sharpe: float, data: pd.DataFrame) -> str | None:
        # Exit settings make it a different variant, so they count towards the idea counter.
        full = {**params, **exits}
        line = ledger.record(idea=spec.name, family=spec.family, params=full, dataset=dataset, stage=stage,
                             sharpe=sharpe)
        if register is None:
            return None
        # The same evaluation in the trials register, with the bars it read (v2 P1-7, C4). Its id is the
        # counter line's, so folding the counter in later doesn't count it twice.
        from sleeve_fund.research.trials import legacy_definition_hash, legacy_idea_hash, line_id, run_setup

        setup = run_setup(risk_profile=risk_profile, fee=float(paid(params).taker) + spread_used,
                          windows=(train_days, test_days, holdout_days))
        return register.record(
            definition_hash=legacy_definition_hash(spec.name, full, setup), idea_hash=legacy_idea_hash(spec.name),
            name=spec.name, family=spec.family, settings=full, dataset=dataset,
            stage="holdout" if stage == "holdout" else "in_sample", source="study", sharpe=sharpe,
            row_id=line_id(line), data_start=data.index[0].to_pydatetime(), data_end=data.index[-1].to_pydatetime())

    # Benchmark over the research period; sliced for every comparison below.
    bench = bt("buy_and_hold", research, {}, benchmark=True)
    bench_ret = daily_returns(bench.equity)
    # A strategy on a perpetual is compared with holding that perpetual at 1x, its fees and funding paid through the
    # same engine (Advisor, 7 Oct 2026), and only over windows whose funding is all known; holding spot is shown
    # beside it where the venue lists spot. A perp venue's own buy and hold above is already a perp.
    hold_market = markets.market_of({**default_params, **market})
    perp_hold = hold_market != markets.SPOT

    def hold_returns(first, last) -> pd.Series | None:
        """The perp hold's daily returns over the bars closing first..last, or None when a funding settlement in it
        has no known rate."""
        if missing_funding({"market": hold_market}, instrument, first - bar, last):
            return None
        window = prices[(prices.index >= first - bar) & (prices.index <= last)]
        run = bt("buy_and_hold", window, {"market": hold_market}, benchmark=True)
        return whole_days(daily_returns(run.equity), first, bar)

    hold_full_period = not perp_hold or not missing_funding({"market": hold_market}, instrument, research.index[0],
                                                           research.index[-1])
    if perp_hold and hold_full_period and not market:  # a spot venue: the research period's hold is the perp's too
        bench = bt("buy_and_hold", research, {"market": hold_market}, benchmark=True)
        bench_ret = daily_returns(bench.equity)
    spot_ret = daily_returns(bt("buy_and_hold", research, {}, benchmark=True).equity) \
        if perp_hold and not (profile is not None and profile.perpetual) else None
    compare_parts, spot_parts, oos_days = [], [], [0]

    def fold_benchmark(test_ret: pd.Series, first, last) -> pd.Series:
        oos_days[0] += len(test_ret)
        if spot_ret is not None:
            spot_parts.append(spot_ret.reindex(test_ret.index).dropna())
        if not perp_hold:
            compare_parts.append(test_ret)
            return bench_ret.reindex(test_ret.index).dropna()
        held = hold_returns(first, last)
        if held is None:
            return pd.Series(dtype=float)
        held = held.reindex(test_ret.index).dropna()
        compare_parts.append(test_ret.reindex(held.index))
        return held

    # 1. Sensitivity over the full research period.
    # The cost ladder: a variant over the research period at each fee. Not logged as variants, since the
    # strategy is the same; only what it pays changes. Every variant gets one (v2 P1-6), so a break-even fee
    # sits next to each grid point, not only the default's.
    slip = ladder_slippage(pair_of(instrument))

    def cost_ladder(params: dict) -> list[LadderRung]:
        rungs = []
        for fee in COST_LADDER:
            res = bt(spec.name, research, params, fees=FeeSchedule(maker=Decimal(str(fee)), taker=Decimal(str(fee))),
                     slippage=slip)
            eq = res.equity
            rungs.append(LadderRung(fee=fee, total_return=float(eq.iloc[-1] / starting_capital - 1) if len(eq) else 0.0,
                                    sharpe=summary(daily_returns(eq))["sharpe"],
                                    round_trips=len(round_trips(res.fills, res.shorts)), fees_paid=res.fees_paid))
        return rungs

    rows, ladders = [], []
    full_default = None
    for params in combos:
        res = bt(spec.name, research, params)
        m = summary(daily_returns(res.equity))
        log(params, "sensitivity", m["sharpe"], research)
        rungs = cost_ladder(params)
        ladders.append(rungs)
        fee, words = breakeven_fee(rungs)
        rows.append({**params, **m, "round_trips": len(round_trips(res.fills, res.shorts)), "fees": res.fees_paid,
                     "breakeven_fee": float("nan") if fee is None else fee, "breakeven": words})
        if params == default_params:
            full_default = res
    if full_default is None:
        full_default = bt(spec.name, research, default_params)
    sensitivity = pd.DataFrame(rows)
    # The settings tuning on the whole period would pick; the ladder and its verified break-even are theirs.
    sharpes = [r["sharpe"] if math.isfinite(r["sharpe"]) else -math.inf for r in rows]
    if rows and max(sharpes) > -math.inf:
        at = sharpes.index(max(sharpes))
        chosen, ladder = combos[at], ladders[at]
    else:
        at, chosen, ladder = None, default_params, cost_ladder(default_params)

    def run_at(fee: float) -> float:
        eq = bt(spec.name, research, chosen, fees=FeeSchedule(maker=Decimal(str(fee)), taker=Decimal(str(fee))),
                slippage=slip).equity
        return float(eq.iloc[-1] / starting_capital - 1) if len(eq) else 0.0

    fee, breakeven = breakeven_fee(ladder, run_at)
    if at is not None:
        sensitivity.loc[at, ["breakeven_fee", "breakeven"]] = [float("nan") if fee is None else fee, breakeven]

    # 2. Walk-forward.
    folds: list[Fold] = []
    oos_parts, bench_parts = [], []
    start = 0
    while start + train_bars + test_bars <= len(research):
        train = research.iloc[start : start + train_bars]
        through_test = research.iloc[start : start + train_bars + test_bars]
        test_idx = through_test.index[train_bars:]
        best, best_sharpe = None, float("-inf")
        surface = []
        for params in combos:
            fit = bt(spec.name, train, params)
            m = summary(daily_returns(fit.equity))
            log(params, "wf_train", m["sharpe"], train)
            surface.append({**params, "sharpe": m["sharpe"], "round_trips": len(round_trips(fit.fills, fit.shorts))})
            if m["sharpe"] > best_sharpe:
                best, best_sharpe = params, m["sharpe"]
        if best is None:
            # No setting scored a Sharpe on training (none traded, so every one was NaN): nothing was chosen. The
            # fold sits flat through its test window and fails the nearby-settings check, saying why (CR, #156).
            days = whole_days(bench_ret, test_idx[0], bar)
            test_ret = pd.Series(0.0, index=days.index[days.index <= test_idx[-1]])
            folds.append(Fold(train_start=train.index[0], train_end=train.index[-1], test_end=test_idx[-1], chosen={},
                              train_sharpe=float("nan"), test=summary(test_ret),
                              benchmark_test=_summary_or_none(b_ret := fold_benchmark(test_ret, test_idx[0],
                                                                                      test_idx[-1])),
                              window_bars=len(test_idx), unscored=True, grid=pd.DataFrame(surface)))
            oos_parts.append(test_ret)
            bench_parts.append(b_ret)
            start += test_bars
            continue
        # Trade the chosen params continuously through the test window so the
        # position carried in from training is realistic, then score only the test days.
        run = bt(spec.name, through_test, best)
        test_ret = whole_days(daily_returns(run.equity), test_idx[0], bar)  # the test window's days, not the train's
        b_ret = fold_benchmark(test_ret, test_idx[0], test_idx[-1])
        folds.append(
            Fold(
                train_start=train.index[0],
                train_end=train.index[-1],
                test_end=test_idx[-1],
                chosen=best,
                train_sharpe=best_sharpe,
                test=summary(test_ret),
                benchmark_test=_summary_or_none(b_ret),
                **_window_trips(trades(fills_to_rows(run.fills), run.shorts, open_trip=True), test_idx[0]),
                window_bars=len(test_idx),
                in_market_bars=_in_market_bars(run.exposure, test_idx[0], test_idx[-1]),
                halted=_halt_words(run.risk_events, test_idx[0], test_idx[-1]),
                halted_before_test=_halted_before(run.risk_events, test_idx[0]),
                grid=pd.DataFrame(surface),
                **_window_funding(run.funding_marks, train.index[-1], test_idx[-1]),
                liquidations=_liquidations(run, test_idx[0], test_idx[-1],
                                           f"out-of-sample test window {len(folds) + 1}"),
            )
        )
        oos_parts.append(test_ret)
        bench_parts.append(b_ret)
        start += test_bars

    run_fees = paid(chosen)
    covered = len(pd.concat(compare_parts)) if compare_parts else 0
    hold_insufficient = perp_hold and covered < HOLD_COVERAGE_FLOOR * oos_days[0]
    if perp_hold:
        terms = markets.terms({"market": hold_market}, str(instrument.id.venue))
        paying = (f"assumed funding of {terms.funding_rate:.2%} every settlement, as the strategy's runs pay"
                  if terms.funding_venue is None else "the venue's settled funding")
        hold_note = (f"buy and hold is a 1x long perpetual paying its fees and {paying}; it covers {covered} of "
                     f"{oos_days[0]} out-of-sample days, the windows whose funding is all known")
        if hold_insufficient:
            hold_note += (f": insufficient funding history (under {HOLD_COVERAGE_FLOOR:.0%}), so it gives no verdict; "
                          "random entry still judges")
    else:
        hold_note = ""
    result = StudyResult(
        spec=spec,
        dataset=dataset,
        synthetic=synthetic,
        research_start=research.index[0],
        research_end=research.index[-1],
        holdout_days=holdout_days,
        default_params=default_params,
        full_period=full_default,
        full_period_benchmark=bench,
        sensitivity=sensitivity,
        folds=folds,
        oos_returns=pd.concat(oos_parts),
        oos_benchmark_returns=pd.concat(bench_parts),
        oos_compare_returns=pd.concat(compare_parts) if compare_parts else pd.Series(dtype=float),
        spot_hold_returns=pd.concat(spot_parts) if spot_parts else None,
        hold_market=hold_market,
        hold_full_period=hold_full_period,
        hold_note=hold_note,
        hold_insufficient=hold_insufficient,
        instrument=pair_of(instrument),
        bar_minutes=minutes,
        venue=str(instrument.id.venue),
        risk_profile=risk_profile,
        settings=(f"{risk_profile + ' risk profile' if risk_profile else 'no risk profile'} · "
                  f"exits: {_exit_words(exits) if exits else 'the signal only'} · "
                  f"walk-forward {train_days} days training, {test_days} days testing"),
        errors=errors,
        error_count=error_count[0],
        cost_ladder=ladder,
        ladder_slippage=slip,
        chosen_params=chosen,
        breakeven=breakeven,
        fee_note=(f"{float(run_fees.maker):.2%} maker on post-only orders, {float(run_fees.taker):.2%} taker "
                  f"on every other order, plus {spread_used:.3%} of the price as half the bid-ask spread on orders "
                  f"that take liquidity; break-even: {breakeven}"),
        fee_basis=run_fees,
    )
    # Every trip pays the taker fee and half the spread each way, as the study's own runs do on market orders.
    # A perpetual's trades and draws are liquidated as the engine books one, at the risk profile's leverage cap.
    leverage = risk_profile_of(risk_profile).max_leverage if risk_profile is not None else None
    result.random_entry, result.random_side = _benchmarks(
        research, folds, test_bars, float(run_fees.taker) + spread_used, full_default.shorts,
        leverage, market.get("market"))
    for rung in ladder:  # the benchmark at each rung's own cost, as the rung's runs pay it (RE-COST)
        at_rung, _ = _benchmarks(research, folds, test_bars, rung.fee + spread_used + slip, False, leverage,
                                 market.get("market"))
        if at_rung.trades:
            rung.oos_timing_return, rung.random_return = at_rung.strategy_return, at_rung.median_random_return
    if hold_market == markets.PERP and not market:
        # A simulated perp's funding is an assumed 0.01% a settlement, light in strong uptrends: the research period
        # again with funding at FUNDING_STRESS_RATE, strategy and hold alike (Advisor, 7 Oct 2026).
        def ret(name: str, params: dict, benchmark: bool = False) -> float:
            eq = bt(name, research, params, benchmark=benchmark).equity
            return float(eq.iloc[-1] / starting_capital - 1) if len(eq) else 0.0

        stress = {"market": markets.PERP_FUNDING_STRESS}
        result.funding_stress = {
            "rate": markets.FUNDING_STRESS_RATE, "assumed": markets.LOW_FEE_PERP.funding_rate,
            "strategy": (ret(spec.name, chosen), ret(spec.name, {**chosen, **stress})),
            "hold": (ret("buy_and_hold", {"market": hold_market}, True), ret("buy_and_hold", stress, True)),
        }
    if risk_profile is not None:
        result.notes.append(
            f"Every run trades under the {risk_profile} risk profile, as paper does: positions capped at "
            f"{strategy_cap:.0%} of capital{' in notional' if market else ''}, the drawdown halt (flat for the rest of the run, as paper stays halted "
            "until you resume it) and the daily-loss pause. The buy-and-hold benchmark is held at "
            f"{'the same exposure' if position_cap == strategy_cap else f'{position_cap:.0%} of capital'} and never halted.")
    else:
        result.notes.append(
            f"Positions are capped at {position_cap:.0%} of capital, as the paper risk profile allows, and the "
            "buy-and-hold benchmark is held at the same exposure. No drawdown halt or daily-loss pause." if position_cap
            is not None else
            "Positions are uncapped (all of the capital), and so is the benchmark; paper trades at its risk profile's cap.")
    if exec_minutes is not None:
        result.notes.append(
            f"Resting orders (stops, targets, post-only orders) are matched on {exec_minutes}-minute bars between "
            f"decisions, and the risk guard values the book on each, as the backtest page does.")
    elif risk_profile is not None:
        result.notes.append(f"The risk guard values the book once a {_bar_words(minutes)}; paper does every 30 seconds.")
    if minutes < 1440:
        result.notes.append(
            f"Traded on {minutes}-minute bars; every figure here is on daily returns (closes at 00:00 UTC), "
            "so Sharpe is annualised as daily and the bootstrap resamples days, as for a daily strategy.")
    if market:
        result.notes.append("The venue lists perpetuals only, so every run traded the perpetual, long only, "
                            "paying the funding the venue settled.")
    if exits:
        result.notes.append(
            "Exits on top of the signal: " + _exit_words(exits)
            + ". The stop rests at the venue and fills at its level (at market at once if the price is already "
            "through it); the target fills at its level when the price trades through it. A bar that reaches "
            "both takes the stop: the adverse side goes first. "
            "Paper watches both on every trade."
        )

    if register is not None:
        from sleeve_fund.research.trials import legacy_idea_hash

        result.failed_counts = register.failed(legacy_idea_hash(spec.name))

    # 3. Holdout, only on request, using the most recent fold's choice. A study G1 can't judge leaves it
    # closed: opening it would spend it on no information (review round 8, M8-4).
    # Nor is it opened twice: the first look was its one use, at any bar length (review round 10, M10-1).
    opened = ledger.holdout_opened(spec.name, dataset) if use_holdout and holdout_days else None
    # The database lock (v2 P1-7, C4): one look per idea and underlying, at any timeframe or venue, and none
    # while an earlier evaluation of the idea read the held-back days.
    locked = ""
    if use_holdout and holdout_days and locks is not None:
        from sleeve_fund.research.holdout import MIN_UNDATED_HOLDOUT_TRADES, underlying_of
        from sleeve_fund.research.trials import legacy_idea_hash

        idea_hash, underlying = legacy_idea_hash(spec.name), underlying_of(pair_of(instrument))
        locked = locks.refusal(idea_hash, underlying, prices.index[-holdout_bars], prices.index[-1])
    if use_holdout and holdout_days and result.not_judged:
        result.holdout_withheld = f"left closed, though asked for: G1 can't judge this study ({result.not_judged})"
    elif opened is not None:
        result.holdout_withheld = (f"left closed, though asked for: this model's holdout on {result.instrument} was "
                                   f"opened {opened_words(opened)}, and a second look can't be fresh")
    elif use_holdout and holdout_days and folds[-1].unscored:
        result.holdout_withheld = ("left closed, though asked for: no setting scored on the last fold's training "
                                   "stretch, so there is no choice to test on it")
    elif locked:
        result.holdout_withheld = f"left closed, though asked for: {locked}"
    elif (use_holdout and holdout_days and locks is not None and locks.undated(idea_hash)
          and (few := _holdout_trades(bt(spec.name, prices, folds[-1].chosen), prices.index[-holdout_bars]))
          < MIN_UNDATED_HOLDOUT_TRADES):
        # Counted before the claim, and only the count is shown: spending the one clean look on too few trades
        # to judge would waste it (Advisor, C4 option b).
        result.holdout_withheld = (f"left closed, though asked for: no holdout yet: the held-back days hold {few} "
                                   f"trades, and {MIN_UNDATED_HOLDOUT_TRADES} are needed: "
                                   f"{MIN_UNDATED_HOLDOUT_TRADES - few} more")
    elif use_holdout and holdout_days and locks is not None and not locks.open(
            idea_hash, underlying, prices.index[-holdout_bars], prices.index[-1]):
        # Claimed before the look: another study took the lock since the check above, so this one doesn't look.
        result.holdout_withheld = ("left closed, though asked for: another study opened this idea's holdout on "
                                   f"{underlying} while this one ran")
    elif use_holdout and holdout_days:
        chosen = folds[-1].chosen
        try:
            run = bt(spec.name, prices, chosen)
            h_start = prices.index[-holdout_bars]
            h_ret = whole_days(daily_returns(run.equity), h_start, bar)
            result.holdout = summary(h_ret)
            if perp_hold:
                hb_ret = hold_returns(h_start, prices.index[-1])
                result.holdout_benchmark = None if hb_ret is None else _summary_or_none(hb_ret)
                if hb_ret is None:
                    result.notes.append("The holdout has no buy-and-hold line: a funding settlement in it has no "
                                        "known rate, and a missing rate is never filled in.")
            else:
                b_all = bt("buy_and_hold", prices, {}, benchmark=True)
                result.holdout_benchmark = summary(whole_days(daily_returns(b_all.equity), h_start, bar))
            result.holdout_funding = _window_funding(run.funding_marks, prices.index[-holdout_bars - 1],
                                                     prices.index[-1])
            result.holdout_liquidations = _liquidations(run, h_start, prices.index[-1], "holdout")
        except Exception:
            if locks is not None:
                # The lock was claimed before the look, so the holdout is spent: recorded as a crash, not a result.
                locks.crashed(idea_hash, underlying)
                log_.error("holdout of %s on %s spent by a crash, not a failed result: the lock stays", spec.name,
                           underlying)
            raise
        trial = log(chosen, "holdout", result.holdout["sharpe"], prices.iloc[-holdout_bars:])
        if locks is not None:
            locks.settle(idea_hash, underlying, trial)
    return result


STOP_MISSING = "missing"
STOP_BEYOND_HALF = "beyond half the distance to liquidation"
STOP_GAPPED = "gapped past the stop"


def _liquidations(run: BacktestResult, start, end, window: str) -> list[dict]:
    """Every liquidation whose fill landed in [start, end] of this run, from its journal, as {window, ts, x,
    stop_px, stop_ok, stop_why, needs_ack}. x: what the liquidation lost on the quantity it closed, at the average
    entry price with that quantity's share of the entry fees, plus its own fee (exactly the margin plus the entry
    and liquidation fees, GAP-LIQ-CAP); a part of the position closed earlier, at a profit or a loss, is not in it.
    The stop: the position's protective stop, the last closing-side order with a trigger journaled while it was
    held. A gap past a stop within half the distance to liquidation needs the PM's acknowledgement for G1; a missing
    stop, or one beyond half way, is a G1 FAIL with no override (Independent Quant Advisor 7 Oct 00:19 (5))."""
    j = getattr(run, "journal", None)
    if j is None or j.sleeve_row is None:
        return []
    name = j.sleeve_row.name
    orders = {o["order_id"]: o for o in j.orders(name, limit=1_000_000)}
    fills = sorted(j.fills(name, limit=1_000_000), key=lambda f: (_utc(f["ts"]), f.get("id") or 0))
    lo, hi = _utc(start), _utc(end)
    found: dict[str, dict] = {}
    held, avg, fee_unit, opened, entry, entry_px = 0.0, 0.0, 0.0, None, None, None
    for f in fills:
        side, qty, px, fee = (1 if f["side"] == "BUY" else -1), float(f["qty"]), float(f["price"]), float(f["fee"])
        o = orders.get(f["order_id"], {})
        if abs(held) < 1e-12:
            held, avg, fee_unit, opened, entry, entry_px = 0.0, 0.0, 0.0, _utc(f["ts"]), o, px
        if held == 0 or side * held > 0:  # opening or adding: the average entry and the entry fees per unit
            avg = (abs(held) * avg + qty * px) / (abs(held) + qty)
            fee_unit = (abs(held) * fee_unit + fee) / (abs(held) + qty)
            held += side * qty
            continue
        closed = min(qty, abs(held))  # reducing, at most to flat (a fill past flat opens the rest the other way)
        if o.get("intent") == "liquidation" and lo <= _utc(f["ts"]) <= hi:
            liq = found.setdefault(f["order_id"], {"window": window, "ts": _utc(f["ts"]), "order_id": f["order_id"],
                                                   "opened": opened, "entry": entry, "entry_px": entry_px,
                                                   "closing_side": f["side"], "x": 0.0})
            held_sign = 1 if held > 0 else -1
            liq["x"] += held_sign * closed * (avg - px) + closed * fee_unit + fee * closed / qty
        held += side * qty
        if abs(held) < 1e-12 or side * held > 0:  # flat, or reversed: the rest is a new position from this fill
            rest = abs(held) if side * held > 0 else 0.0
            held, avg, fee_unit = side * rest, px, (fee / qty if rest else 0.0)
            opened, entry, entry_px = (_utc(f["ts"]), o, px) if rest else (None, None, None)
    for liq in found.values():
        liq["x"] = round(liq["x"], 2)
    return [_judged(liq, orders) for liq in found.values()]


def _judged(liq: dict, orders: dict) -> dict:
    entry = liq.pop("entry") or {}
    sig = entry.get("signal") or {}
    liq_px, entry_px = sig.get("liquidation_px"), liq.pop("entry_px")
    opened, at, side = liq.pop("opened"), liq["ts"], liq.pop("closing_side")
    stops = [o for o in orders.values() if o.get("side") == side and (o.get("signal") or {}).get("trigger") is not None
             and opened is not None and opened <= _utc(o["ts"]) <= at and o.get("intent") in ("stop_loss", "liquidation")]
    stop_px = float(stops[-1]["signal"]["trigger"]) if stops else None
    if stop_px is None:
        why = STOP_MISSING
    elif liq_px and entry_px and abs(stop_px - entry_px) > 0.5 * abs(liq_px - entry_px) + 1e-9:
        why = STOP_BEYOND_HALF
    else:
        why = STOP_GAPPED
    liq.pop("order_id")
    return {**liq, "x": liq.get("x"), "stop_px": stop_px, "stop_ok": why == STOP_GAPPED, "stop_why": why,
            "needs_ack": why == STOP_GAPPED}


def _holdout_trades(run: BacktestResult, start) -> int:
    """Round trips opened and closed inside the holdout."""
    return sum(1 for t in trades(fills_to_rows(run.fills), run.shorts)
               if t["closed"] is not None and _utc(t["opened"]) >= _utc(start))


def _window_trips(trips: list[dict], test_start) -> dict:
    """A test-window run's round trips sorted into the counted ones and those at the window's edges."""
    start = _utc(test_start)
    counted = [t for t in trips if t["opened"] is not None and t["closed"] is not None and _utc(t["opened"]) >= start]
    carried_in = sum(1 for t in trips if t["closed"] is not None and _utc(t["closed"]) >= start
                     and (t["opened"] is None or _utc(t["opened"]) < start))
    carried_out = sum(1 for t in trips if t["closed"] is None)  # still open at the window's end, wherever it opened
    return {"test_trades": len(counted), "carried_in": carried_in, "carried_out": carried_out,
            "trips": [(_utc(t["opened"]), _utc(t["closed"]), int(t["side"])) for t in counted]}


def _in_market_bars(exposure: pd.Series, first, last) -> int:
    """The bars from first to last, inclusive, that closed holding a position of any size or side."""
    if exposure is None or exposure.empty:
        return 0
    idx = exposure.index
    inside = exposure[(idx >= first) & (idx <= last)]
    return int((inside.abs() > 1e-9).sum())


def _benchmarks(prices: pd.DataFrame, folds: list[Fold], test_bars: int, cost_per_side: float, shorts: bool,
                leverage: float | None = None, market: str | None = None):
    """The random-entry benchmark, and the random-side test when the strategy can go short, on the folds'
    counted trips. Each trip is placed on the bars it was opened and closed in, and both sides of the comparison
    are priced on those bars' closes, so the benchmark compares timing, not fills. leverage: the risk profile's cap,
    for a fold that traded the perpetual (its chosen market, else `market`): its trips and their draws are
    liquidated as the engine books one, losing the margin and fees, never the move past it (Independent Quant
    Advisor 7 Oct 00:19 (4))."""
    index = prices.index.tz_localize("UTC") if prices.index.tz is None else prices.index

    def bar(ts) -> int:
        return max(int(index.searchsorted(ts, side="right")) - 1, 0)

    windows, placed = [], []
    for f in folds:
        end = bar(f.test_end)
        start = end - test_bars + 1
        windows.append((start, end))
        last = start
        for opened, closed, side in sorted(f.trips):
            entry = max(bar(opened), last)  # one position at a time on the bar grid too
            out = min(max(bar(closed), entry + 1), end)
            if out <= entry:
                continue
            perp = f.chosen.get("market", market) == PERP
            placed.append(Trade(entry, out, side, leverage=leverage if perp else None))
            last = out
    closes = prices["close"].to_numpy(dtype=float)
    bars = sum(f.window_bars for f in folds)
    in_market = sum(f.in_market_bars for f in folds) / bars if bars else None
    entry = random_entry(closes, placed, windows, cost_per_side, in_market=in_market)
    side = random_side(closes, placed, windows, cost_per_side) if shorts else None
    return entry, side


HOLD_COVERAGE_FLOOR = 0.5  # the share of out-of-sample days a perp hold must cover to judge (Advisor, 7 Oct 2026)


def _summary_or_none(returns: pd.Series) -> dict | None:
    return summary(returns) if len(returns.dropna()) >= 2 else None


def missing_funding(params: dict, instrument, after: pd.Timestamp, until: pd.Timestamp, root=None) -> int | None:
    """How many funding settlements in (after, until] a perpetual hold on the strategy's market has no settled rate
    for: None when the market is spot, 0 for a simulated perp, which charges its terms' fixed rate as its runs do.
    The perp buy and hold is priced only over a window with none missing; a missing rate is never filled in
    (Advisor, 7 Oct 2026)."""
    from datetime import timedelta

    from sleeve_fund import funding

    terms = markets.terms(params, str(instrument.id.venue))
    if terms is None:
        return None
    if terms.funding_venue is None:
        return 0
    series = funding.rates(terms.funding_venue, pair_of(instrument), root)
    due = markets.funding_times(_utc(after).to_pydatetime(), _utc(until).to_pydatetime(),
                                terms.funding_hours)
    return sum(funding.rate_at(series, pd.Timestamp(t)) is None for t in due)


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t


# What stops a run trading until a person steps in: the risk guard's halt, and a reconcile mismatch
# (the engine's book disagreeing with the journal halts it too; review round 9, B9-1 and M9-1).
HALT_KINDS = ("risk_halt", "reconcile_mismatch")


def _halt_reason(e: dict) -> str:
    if e["kind"] == "reconcile_mismatch":
        return "reconcile mismatch: the engine's book disagreed with the journal"
    return e["message"].split(";")[0]


def _halted_before(events: list[dict], test_start: pd.Timestamp) -> bool:
    return any(e["kind"] in HALT_KINDS and _utc(e["ts"]) < _utc(test_start) for e in events)


def _halt_words(events: list[dict], test_start: pd.Timestamp, test_end: pd.Timestamp) -> str:
    """'12 Mar 2026 (drawdown 20.3% hit the 20% limit), in the training stretch' for the first halt
    at or before the test window's end, else ''."""
    for e in events:
        if e["kind"] not in HALT_KINDS:
            continue
        ts = _utc(e["ts"])
        if ts > _utc(test_end):
            return ""
        where = "in the training stretch the run traded through first" if ts < _utc(test_start) else "in the test window"
        return f"{ts:%d %b %Y} ({_halt_reason(e)}), {where}"
    return ""


def _exit_words(exits: dict) -> str:
    words = []
    if "stop_loss" in exits:
        words.append(f"stop-loss {exits['stop_loss']:.1%} below entry")
    if "stop_atr" in exits:
        words.append(f"stop-loss {exits['stop_atr']:g} average true ranges (over {exits.get('atr_bars', 14)} bars) "
                     "below entry, set at each entry")
    if "stop_swing_bars" in exits:
        words.append(f"stop-loss at the lowest low of the last {exits['stop_swing_bars']} bars, set at each entry")
    if "take_profit" in exits:
        words.append(f"take-profit {exits['take_profit']:.1%} above entry")
    if "take_profit_r" in exits:
        words.append(f"take-profit making {exits['take_profit_r']:g}R after costs")
    if "risk_per_trade" in exits:
        words.append(f"each trade sized to lose {exits['risk_per_trade']:.1%} of equity at the stop")
    return ", ".join(words)
