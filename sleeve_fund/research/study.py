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
import json
import logging
import math
from decimal import Decimal
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from nautilus_trader.model import CurrencyPair

from sleeve_fund import funding
from sleeve_fund.instruments import FeeSchedule, pair_of, pct, taker_slippage
from sleeve_fund import markets, risk
from sleeve_fund.markets import PERP
from sleeve_fund.research import equal_exposure, periods, placebo, regime_split
from sleeve_fund.research.ledger import GAP_STAGE, IdeaLedger, opened_words
from sleeve_fund.research.random_entry import (RandomEntryResult, RandomSideResult, Trade, random_entry,
                                               random_entry_filled, random_side, random_side_filled)
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
from sleeve_fund.spreads import SpreadSeries
from sleeve_fund.strategies import check_perp_sizing
from sleeve_fund.strategies.base import STOP_INTENTS, IdeaSpec
from sleeve_fund.strategies.rules import FIRST_TOUCH_RERUN

# P1-D13 (Advisor 6 Oct 18:16 to 19:16): G1 stands on fills that don't depend on what traded first inside a bar. With a
# resting exit (a stop, a target, the risk guard's stops, a perpetual's liquidation price) the out-of-sample windows
# and the holdout must run on 1-minute bars; 5-minute ones are a one-way test (a pass counts once a 1-minute spot check
# holds; a fail must be re-run at 1 minute); anything coarser can't be judged.
NO_1M = "no 1-minute execution data"
ONE_WAY_MINUTES = 5
RESTING_INTENTS = (*STOP_INTENTS, "take_profit")


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
    modelled rate series is in the trials register before the run (none can be registered yet). A run across a
    spell of shorter settlements is judged: each is charged at its own time (SOL-FUND, Advisor 21 Oct 2026)."""
    n, m, longest = _oos_funding(r.folds)  # read with defaults: a fold that never held a perp carries no funding
    verdict, words = funding.baseline_check(n, m, longest, "out-of-sample")
    full = getattr(r, "full_period", None)
    if getattr(full, "funding_simulated", False):
        return "NOT JUDGED", (f"a simulated perpetual, with no venue's funding rates: {words}; no modelled rate "
                              "series was in the trials register before the run")
    return verdict, words


@dataclass
class HoldCurve:
    """A hold priced outside the engine (the price-only hold, BH-102 rule 3): its equity is all a sheet reads of the
    full-period benchmark."""
    equity: pd.Series


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
    # ABSORB-GAP: the gaps past bankruptcy the venue absorbed in the test window, as {"ts", "amount"}; the window's
    # `test` is charged them (_charge_absorbed), and these are its days uncharged, for information only.
    absorbed: list = field(default_factory=list)
    uncharged_returns: pd.Series | None = field(default=None, repr=False)
    # Each counted trip's share of those gaps, as (opened, gap over its entry notional): what random entry charges
    # the strategy's own trip (Advisor ~02:55 UK, option A).
    trip_charges: list = field(default_factory=list)
    # G5 research side: a strength-sized variant's counted trips as (opened, size factor f), from their entry orders
    # (_trip_sizes); random entry sizes them and their draws by these. Empty for a flat version.
    trip_sizes: list = field(default_factory=list)
    # The sum over the test window's bars of |position notional / equity| (flat bars 0): its time-weighted gross
    # exposure is this over window_bars (strength_report, Advisor G5-DW5 Q2).
    gross_exposure: float = 0.0
    # M1 item 9 (PM 30 Oct): the window's own trade list (periods.trade_rows of its run, every trip it booked, one still
    # open at its end marked there) and when it starts trading, for the test-window trade list and its recompute.
    trade_rows: list = field(default_factory=list, repr=False)
    trade_from: pd.Timestamp | None = None
    # RANDOM-ENTRY: each counted trip as the engine booked it, {"opened", "closed", "side", "ret" (pnl after fees,
    # fills and funding over its entry notional), "exit_ratio" (exit over entry notional)}, for the strategy's side
    # of random entry; and (the counted trips' pnl, the window's equity change less its carried trips', the
    # window's opening equity), reconciling the two (_trip_reconcile).
    trip_returns: list = field(default_factory=list)
    trip_reconcile: tuple | None = None

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
    # R3: results by regime on the full research period's default-settings run (report only; regime_split).
    by_regime: list = field(default_factory=list)
    random_entry: RandomEntryResult | None = None
    random_side: RandomSideResult | None = None
    # P1-D13: what the out-of-sample windows and the holdout were matched on (None: their decision bars alone), whether
    # any resting level took part (NO_1M), the fill labels the strategy's runs carried, and on 5-minute bars the one-way
    # rule: "fail" (not final), "void" (the 1-minute spot check broke one-sidedness) or "unchecked" (no 1-minute bars
    # for it), with the spot check's window, seed and gap.
    oos_exec_minutes: int | None = None
    resting_exits: bool = False
    fill_labels: list = field(default_factory=list)
    # R-28 (retired by PAPER-TIERS-3): a sheet saved before it still says BTC's tiers were assumed; new runs leave it ""
    margin_note: str = ""
    one_way: str = ""
    spot_check: dict | None = None
    # The falsifier (Advisor 18:16): the 1-minute run's end equity less the coarse run's, over starting capital, for the
    # research period at the default settings; None when the study's own runs were on 1-minute bars or none were stored.
    bars_only_gap: float | None = None
    # R-11: how the study was scored (walk_forward or continuous), and a continuous window's warm-up in bars
    score_window: str = "walk_forward"
    warmup_bars: int | None = None
    # R-14: a continuous window's random-entry benchmark per test year (calendar year of the scored window)
    random_entry_by_year: dict = field(default_factory=dict)
    # R-07: a volatility-sized variant against its parent at the same average gross exposure (equal_exposure)
    equal_exposure: equal_exposure.EqualExposure | None = None
    # R-24: a filtered variant's filter against a placebo of its parent (placebo); None for any other study
    placebo: placebo.PlaceboResult | None = None
    # Runs of this idea whose trials-register count failed (QA P1-T8): they count in N, but their Sharpes are
    # missing from the spread the bar is set by, so G1 can't judge until they are re-counted (Advisor, 6 Oct 2026).
    failed_counts: int = 0
    # The idea family the study counted under: an optimiser's proposal's own structural idea (candidates.idea_hash,
    # CR F236-1), or the hand-coded model's, by its name, when None.
    idea_hash: str | None = None
    # The buy and hold G1 compares with (Advisor, 7 Oct 2026, RE-COST): on a perpetual, a 1x long perp hold paying the
    # perp's fees and funding, through the same engine, priced only over windows whose funding is all known.
    # oos_compare_returns: the strategy's out-of-sample days the hold covers, what the benchmark checks compare.
    hold_market: str = "spot"
    hold_note: str = ""  # what the hold pays for funding, and how much of out-of-sample it covers
    hold_insufficient: bool = False  # covers under HOLD_COVERAGE_FLOOR of the out-of-sample days: no verdict
    hold_full_period: bool = True  # the hold covers the whole research period (the parameter-robustness check)
    # RESEARCH-SPOT-FEES (Advisor, 21 Oct): a spot-lowfee sheet's chosen settings over the research period at our
    # current entry-tier spot fee (0.40% maker, 0.80% taker), and beside it the same settings on the perpetual with its
    # funding, on the same days: {"spot": {...}, "perp": {...}} of end value, cagr, sharpe, max_drawdown, funding_paid.
    spot_entry_tier: LadderRung | None = None
    spot_vs_perp: dict | None = None
    oos_compare_returns: pd.Series | None = None
    spot_hold_returns: pd.Series | None = None  # "holding spot instead", out-of-sample, shown beside a perp hold
    # BH-102 rule 3 (Advisor, 21 Oct): a perp study's bar is the higher-Sharpe of the perp hold and the spot (or
    # price-only) hold over the compared out-of-sample days, chosen once; every benchmark reader takes that one.
    bar_hold: str = ""  # "perp hold", "spot hold" or PRICE_HOLD; "" on a spot study (its one hold)
    bar_note: str = ""  # which set the bar, with both holds' Sharpe, as the sheet prints it
    perp_hold_returns: pd.Series | None = None  # the perp hold, out-of-sample, the bar or shown beside it
    hold_liquidated: list = field(default_factory=list)  # when the perp hold was liquidated, as "<date> (<where>)"
    funding_stress: dict | None = None  # a simulated perp's research period at FUNDING_STRESS_RATE
    # D13-INTRAMINUTE-JUMP (3), GAP-DOWN: out-of-sample again with STOP_STRESS more on each stop exit. charge: the
    # daily returns that costs, by day; stops: the stop exits; median_stop: the entries' median stop distance.
    stop_stress: dict = field(default_factory=dict)
    # The probability of backtest overfitting (research.pbo, R4) over the sensitivity grid and every other counted
    # setting of the idea family, with its variant count.
    pbo: object = None
    # The ablation report (research.ablation), when one was run for this definition: shown, never judged.
    ablation: object = None
    # B-22 (Advisor B22-DOUBLE-FEES): the out-of-sample windows re-run at twice the venue's fees (_double_fees), and
    # their daily net returns, on the out-of-sample days. None for a study whose windows weren't re-run.
    double_fees: dict | None = None
    double_fee_returns: pd.Series | None = None

    @property
    def compare_returns(self) -> pd.Series:
        return self.oos_returns if self.oos_compare_returns is None else self.oos_compare_returns

    # Every liquidation in out-of-sample (the folds' test windows) or the opened holdout, as _liquidations gives
    # them: each a G1 finding whatever the P&L (Independent Quant Advisor 6 Oct 23:42, 7 Oct 00:19 (5)).
    holdout_liquidations: list = field(default_factory=list)
    # ABSORB-GAP: out-of-sample's days and the holdout's summary with the absorbed gaps left uncharged, for information
    # only (every test reads the charged ones), and the holdout's absorbed gaps; None when nothing was absorbed.
    oos_uncharged_returns: pd.Series | None = None
    holdout_absorbed: list = field(default_factory=list)
    holdout_uncharged: dict | None = None
    random_entry_uncharged: RandomEntryResult | None = None
    random_side_uncharged: RandomSideResult | None = None

    @property
    def liquidations(self) -> list[dict]:
        return [liq for f in self.folds for liq in getattr(f, "liquidations", [])] + list(
            getattr(self, "holdout_liquidations", []))

    # R2-G1: for a model with a first_touch rule, the out-of-sample share of ambiguous candles and, when it is over
    # FIRST_TOUCH_FLIP_SHARE, both resolutions' out-of-sample summaries and which one G1 judged ("as ruled" or
    # "opposite"). Empty for a model without one.
    first_touch: dict = field(default_factory=dict)
    # The research period's closes, for the analysis pack's price-only lines (pack (a) and (c), 21 Oct).
    research_closes: pd.Series | None = None

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
        # Funding's reason first, so it is never hidden behind D13's no-1-minute one.
        verdict, words = _funding_check(self)
        if verdict == "NOT JUDGED":
            return f"its funding isn't the venue's: {words}"
        if getattr(self, "resting_exits", False):
            minutes = getattr(self, "oos_exec_minutes", None)
            if minutes is None or minutes > ONE_WAY_MINUTES:
                return (f"{NO_1M}: its resting exits fill inside a bar, and "
                        + ("on bars alone" if minutes is None else f"on {minutes}-minute bars")
                        + " what traded first is unknown, so the out-of-sample windows and the holdout need 1-minute bars")
            one_way, check = getattr(self, "one_way", ""), getattr(self, "spot_check", None) or {}
            if one_way == "fail":
                return ("the out-of-sample windows ran on 5-minute bars, where a fail is not final: re-run them on "
                        "1-minute bars before dropping the idea")
            if one_way == "void":
                return (f"the 5-minute pass did not hold on 1-minute bars in the window {check.get('window')} (drawn "
                        f"with seed {check.get('seed')}), so it is void: the whole out-of-sample must run on 1-minute "
                        "bars before G1")
            if one_way == "unchecked":
                return ("the 5-minute pass couldn't be spot-checked: no 1-minute bars for the window drawn "
                        f"({check.get('window')}, seed {check.get('seed')}); it counts once one holds on them")
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
        return trade_stats(trades(fills_to_rows(self.full_period.fills), self.full_period.shorts,
                                  self.full_period.funding, self.full_period.insurance))  # R-04 shorts; R-16 flows

    @property
    def turnover(self) -> float:
        return turnover_per_year(self.full_period.fills, self.full_period.equity)


# Fee per side the cost ladder tests every idea at (PM, 5 Oct 2026): free, the low-fee perp venues' maker
# and taker rates, a mid venue, and a high-fee spot venue's taker rate (the stress case). The Independent
# Quant Advisor (6 Oct 2026, P1-G1) added 0.15-0.40%, where most spot taker fees sit: return bends with the
# fee, so a wide gap there misplaced the break-even.
# R2-G1 (Advisor 6 Oct ~22:07 and ~22:11): a first_touch rule's ambiguous candles (both levels first reached in one
# minute, or a minute missing or inconsistent before the first reach) resolve by a fixed rule, not by what traded.
# Over this share of the candles that reached either level, G1 re-runs with the opposite resolution and judges on the
# worse of the two. The rule's own threshold, so the report's rerun_opposite_resolution and the study agree.
FIRST_TOUCH_FLIP_SHARE = FIRST_TOUCH_RERUN


# R2-G1-F1 (Advisor 7 Oct): too few out-of-sample level-reaching candles to trust the share also forces the re-run.
FIRST_TOUCH_MIN_REACHED = 20


def first_touch_counts(runs) -> tuple[int, int] | None:
    """(ambiguous, reached) candles across every first_touch rule of these runs: ambiguous is same_minute + unknown,
    reached is the candles that reached either level. None when none of them has a first_touch rule."""
    ambiguous = reached = 0
    seen = False
    for run in runs:
        for st in (run.first_touch or {}).values():
            seen = True
            ambiguous += st["same_minute"] + st["unknown"]
            reached += st["reached"]
    return (ambiguous, reached) if seen else None


def first_touch_share(runs) -> float | None:
    """Ambiguous candles over the candles that reached either level, across every first_touch rule of these runs;
    None when none of them has a first_touch rule."""
    counts = first_touch_counts(runs)
    if counts is None:
        return None
    ambiguous, reached = counts
    return ambiguous / reached if reached else 0.0


def _resolved(runs) -> str:
    """How the runs' first_touch rules resolved an ambiguous candle, by rule path: "long.entry false when ambiguous"."""
    out = {}
    for run in runs:
        for path, st in (run.first_touch or {}).items():
            out.setdefault(path, st.get("resolved", ""))
    return "; ".join(f"{p} {r}" for p, r in sorted(out.items()))


def needs_flip(share: float | None, fold_shares=(), reached: int | None = None) -> bool:
    """Whether G1 must also judge the opposite resolution (Advisor 7 Oct): the pooled out-of-sample share or any
    fold's strictly over FIRST_TOUCH_FLIP_SHARE, or fewer than FIRST_TOUCH_MIN_REACHED level-reaching candles."""
    if share is None:
        return False
    return (share > FIRST_TOUCH_FLIP_SHARE or any(f > FIRST_TOUCH_FLIP_SHARE for f in fold_shares)
            or (reached is not None and reached < FIRST_TOUCH_MIN_REACHED))


def _worse(a: pd.Series, b: pd.Series) -> bool:
    """Whether daily returns a are worse than b on G1's measure, the Sharpe (no Sharpe, as with no trades, is worst),
    then the total return; a tie is not worse."""
    def key(r: pd.Series) -> tuple[float, float]:
        sharpe = summary(r)["sharpe"]
        return (-math.inf if math.isnan(sharpe) else sharpe, float((1 + r).prod() - 1) if len(r) else 0.0)
    return key(a) < key(b)


def doubled(fees: FeeSchedule) -> FeeSchedule:
    """B-22: exactly twice a schedule, maker and taker both."""
    return FeeSchedule(maker=fees.maker * 2, taker=fees.taker * 2)


def _double_fees(base: pd.Series, twice: pd.Series, runs: list, fees: FeeSchedule) -> dict | None:
    """B-22's stored result (what the page shows, Advisor B22-DOUBLE-FEES): the out-of-sample period's net return at
    twice the fees (`twice`, daily) beside the base (`base`); passed when it is above 0. runs: (base run, double-fee
    run) of each window re-run, for their round trips and whether the trades themselves moved (a real re-run). Plain
    values only. None when no window chose a setting to re-run."""
    if not runs or not len(twice):
        return None
    net = float((1 + twice).prod() - 1)

    def trips(run) -> int:
        return len(round_trips(run.fills, run.shorts))

    def same(a, b) -> bool:
        if not len(a) or not len(b):  # a window that never traded has no fill columns at all
            return len(a) == len(b)
        return len(a) == len(b) and bool((a["filled_qty"].to_numpy() == b["filled_qty"].to_numpy()).all()
                                          and (a["avg_px"].to_numpy() == b["avg_px"].to_numpy()).all())

    return {"passed": net > 0, "net_return": net, "base_net_return": float((1 + base).prod() - 1),
            "maker": fees.maker, "taker": fees.taker,
            "oos_start": twice.index[0], "oos_end": twice.index[-1], "oos_days": len(twice),
            "round_trips": sum(trips(t) for _, t in runs), "base_round_trips": sum(trips(b) for b, _ in runs),
            "trades_differ": not all(same(b.fills, t.fills) for b, t in runs)}


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


# R-11 (PM chose "Five years" 21 Oct 10:13 UK; Advisor SCORE-WINDOW addendum, warm-up rule as amended 21 Oct): a
# pinned-setting study fits nothing on its training folds, so it may instead warm up, then score continuously to the
# research end (the holdout stays locked). The warm-up is the longest any block needs: an exact rolling block (SMA,
# Donchian, rolling volume...) its N bars, as the library counts them; a recursive one (EMA, Wilder's ATR and RSI)
# WARMUP_MULTIPLE x N. A study that tunes settings keeps the walk-forward.
WALK_FORWARD, CONTINUOUS = "walk_forward", "continuous"
SCORE_WINDOWS = (WALK_FORWARD, CONTINUOUS)
WARMUP_MULTIPLE = 10
MIN_SCORED_DAYS = 365
# Recursive blocks, by kind: the settings whose length decays recursively (an RSI divergence adds its swing room).
RECURSIVE_LENGTHS = {"ema": ("period",), "atr": ("period",), "rsi": ("period",), "keltner": ("period", "atr_period"),
                     "rsi_divergence": ("rsi_period",), "adx": ("period",)}


def time_warmup_minutes(kind: str, settings: dict) -> int | None:
    """The minutes a block warms up over when its warm-up is in time rather than candles (Advisor 21 Oct): a calendar
    period or session (warmup_minutes), a day-anchored VWAP's one day, the funding decile's trailing window_days of
    settlements. None for a block that warms up over candles."""
    from sleeve_fund.strategies.definitions import BUILDER_BLOCKS, FUNDING_BLOCKS
    from sleeve_fund.strategies.timeframes import DAY_MINUTES

    cls = BUILDER_BLOCKS[kind]
    if hasattr(cls, "warmup_minutes"):
        return int(cls.warmup_minutes(settings))
    if kind == "vwap" and settings["anchor"] == "day":
        return DAY_MINUTES
    if kind in FUNDING_BLOCKS:
        return int(settings["window_days"]) * DAY_MINUTES
    return None


def block_warmup(kind: str, settings: dict, candle_minutes: int | None = None) -> int:
    """R-11: the candles of `candle_minutes` (the block's own size) one block needs before a continuous window scores:
    WARMUP_MULTIPLE x N for a recursive block; a block that warms up over time, its minutes in candles as the builder
    counts a calendar period, ceil(minutes / candle) + 1; otherwise its own warm-up as warmup_for reads it (N, or N + 1
    where its first value needs the bar before)."""
    from sleeve_fund.strategies.definitions import BUILDER_BLOCKS

    cls = BUILDER_BLOCKS[kind]
    s = cls.check_settings({k: v for k, v in settings.items() if k not in ("kind", "timeframe", "input")})
    if kind in RECURSIVE_LENGTHS:
        bars = WARMUP_MULTIPLE * max(int(s[k]) for k in RECURSIVE_LENGTHS[kind])
        return bars + (s["left"] + s["right"] + s["max_gap"] if kind == "rsi_divergence" else 0)
    minutes = time_warmup_minutes(kind, s)
    if minutes is not None:
        if not candle_minutes:
            raise ValueError(f"a {kind} block warms up over time: its warm-up in candles needs the candle's minutes")
        return -(-minutes // candle_minutes) + 1
    return int(cls.warmup_bars(**s))  # what warmup_for reads for the block


def continuous_warmup(strategy: str, params: dict, bar_minutes: int) -> int:
    """R-11: the warm-up bars of a continuous window, from the rule itself: the most any of its blocks needs
    (block_warmup, in candles of the block's size, after the warm-up of a block it is fed from, as the builder adds
    them), a slower-timeframe block's counted in decision candles, with one more of its candles for the part candle a
    run starts in (as Rules.warmup_needed counts it)."""
    from sleeve_fund.strategies.definitions import FIELDS, check_definition, timeframe_minutes

    if strategy != "rules" or not params.get("definition"):
        raise ValueError("a continuous scoring window reads its warm-up from a rule-builder definition's indicators")
    definition = params["definition"]
    check_definition(definition, bar_minutes=bar_minutes)
    blocks = definition.get("blocks") or {}
    own: dict = {}

    def settle(bid: str) -> int:  # candles of the block's size until it and every block it is fed from have settled
        if bid not in own:
            block = blocks[bid]
            candle = timeframe_minutes(block["timeframe"]) if block.get("timeframe") else bar_minutes
            src = block.get("input")
            fed = settle(src.partition(".")[0]) if src and src not in FIELDS else 0  # chained blocks share a timeframe
            own[bid] = block_warmup(block["kind"], block, candle) + fed
        return own[bid]

    need = 0
    for bid, block in blocks.items():
        bars = settle(bid)
        if block.get("timeframe"):
            tf = timeframe_minutes(block["timeframe"])
            bars = (bars + 1) * tf // bar_minutes if tf > bar_minutes else bars
        need = max(need, bars)
    if need <= 0:
        raise ValueError("the definition names no indicator to set a continuous window's warm-up from")
    return need


def _day_words(ts: pd.Timestamp, bar: pd.Timedelta) -> str:
    """'19 Oct 2025': the UTC day a bar belongs to (the day its open falls on)."""
    from sleeve_fund.research.pinned import day_of

    return f"{day_of(ts, bar):%d %b %Y}"


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
    half_spread: float | SpreadSeries | None = None,
    progress=None,
    oos_exec_prices: pd.DataFrame | None = None,
    minute_loader=None,
    register=None,
    locks=None,
    idea_hash: str | None = None,
    pinned=None,
    score_window: str = "walk_forward",
    placebo_parent: dict | None = None,
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
    oos_exec_prices: the bars the out-of-sample windows and the holdout run on instead, 1-minute ones for G1 (P1-D13);
    None runs them on exec_prices too. Each window and the holdout starts flat.
    minute_loader: (start, end) -> 1-minute bars over that span, for the spot check of a 5-minute pass.
    half_spread: the spread charged on orders that take liquidity: sleeve_fund.spreads.series gives the measured
    ones, each run charging the one in force at each time (SPREAD-PIT); None uses the venue's assumption.
    progress: called with the share of the study's backtests done, 0 to 1.
    idea_hash: the idea family its trials count under, holdout lock included: an optimiser's proposal's own structural
    idea (candidates.idea_hash), so its runs count towards that idea's budget (CR F236-1). None: a rule-builder study's
    own definition's (CR F236-3), else the hand-coded model's, by its name.
    pinned: a study list's pinned dates (research.pinned, B-37): only the days they name are read, whatever the day
    it runs, the holdout is theirs, and the walk-forward must be their folds; None: the history as given."""
    from sleeve_fund.venues import VENUES

    perpetual = getattr(VENUES.get(str(instrument.id.venue)), "perpetual", False)
    # RESEARCH-SPOT-FEES: asked for on the research spot market, a perp venue's study trades spot, its hold too
    if markets.market_of(default_params or spec.default_params) == markets.SPOT_LOWFEE:
        perpetual = False
    for combo in grid(spec.param_grid) or [{}]:  # refused up front, before any data is read or run
        check_perp_sizing(spec.name, {**(default_params or spec.default_params or {}), **combo,
                                      **({"market": PERP} if perpetual else {})})
    if pinned is not None:
        from sleeve_fund.research import pinned as pinning

        if holdout_days != pinned.holdout_days:
            raise ValueError(f"pinned dates: the holdout is {pinned.holdout_days} days, not {holdout_days}")
        prices = pinning.cut(prices, pinned)
        exec_prices, oos_exec_prices = pinning.upto(exec_prices, pinned.end), pinning.upto(oos_exec_prices, pinned.end)
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
    if pinned is not None:
        pinning.check_folds(pinned, research, train_bars, test_bars, bar)
    combos = grid(spec.param_grid)
    if default_params and combos == [{}]:
        # A model with no grid run on the caller's settings (a rule-builder definition): those settings are the one
        # point, so every run trades them, not the model's empty defaults.
        combos = [dict(default_params)]
    default_params = default_params or spec.default_params or (combos[0] if combos else {})
    idea_hash = idea_of(default_params, idea_hash)
    if score_window not in SCORE_WINDOWS:
        raise ValueError(f"score_window must be one of {', '.join(SCORE_WINDOWS)}, not {score_window!r}")
    warmup_bars = None
    if score_window == CONTINUOUS:  # R-11: one window, the warm-up then scored continuously to the research end
        if len(combos) != 1:
            raise ValueError(f"a continuous scoring window is for pinned settings only: this study tunes "
                             f"{len(combos)} settings, so it keeps the walk-forward")
        warmup_bars = continuous_warmup(spec.name, combos[0], minutes)
        if warmup_bars + MIN_SCORED_DAYS * per_day > len(research):
            raise ValueError(f"the warm-up of {warmup_bars:,} bars leaves under {MIN_SCORED_DAYS} days to score "
                             f"in the {len(research) / per_day:,.0f}-day research period")
        train_bars, test_bars = warmup_bars, len(research) - warmup_bars

    exits = {k: v for k, v in (exits or {}).items() if v is not None}
    from sleeve_fund.venues import VENUES

    profile = VENUES.get(str(instrument.id.venue))
    # A venue that lists perpetuals only has no spot: every run, the benchmark too, holds its perpetual.
    market = ({"market": markets.SPOT_LOWFEE} if markets.market_of(default_params) == markets.SPOT_LOWFEE else
              {"market": PERP} if profile is not None and profile.perpetual else {})
    strategy_cap = None
    if risk_profile is not None:
        from sleeve_fund.risk import position_cap as cap_of
        from sleeve_fund.risk import profile as risk_profile_of

        # On a perpetual, the margin cap times the leverage cap, as paper and the backtest page size it; the
        # benchmark holds that exposure up to all of the capital, as the backtest page's does (review round 13, E13-1).
        strategy_cap = cap_of(risk_profile_of(risk_profile), {**default_params, **market})  # its own cap too (B-35 a)
        position_cap = min(strategy_cap, 1.0)
    if position_cap is not None and not 0 < position_cap <= 1:
        raise ValueError(f"position_cap {position_cap} outside (0, 1]")
    if half_spread is None:
        from sleeve_fund.venues import venue as venue_profile

        half_spread = venue_profile(str(instrument.id.venue)).assumed_half_spread
    series = half_spread if isinstance(half_spread, SpreadSeries) else None
    # One figure where a single value is needed (the setup's cost, the random-entry benchmarks): the spread in force
    # at the end of the data, as the latest measurement was before SPREAD-PIT. The runs charge each in its time.
    spread_used = series.at(int(prices.index[-1].value)) if series is not None else half_spread
    spread_words = (series.text(int((prices.index[0] - bar).value), int(prices.index[-1].value))
                    if series is not None else f"{spread_used:.3%} of the price as half the bid-ask spread")

    def paid(params: dict) -> FeeSchedule:
        """The schedule a run with these settings pays (runner.run_backtest's own default: its market's, or the
        instrument's). The benchmarks, the fee note and the variant's key take it too, so a perp study's strategy and
        its random entries pay the same fee on the same trades (RE-COST)."""
        return markets.fees_for({**params, **market}, FeeSchedule(instrument.maker_fee, instrument.taker_fee),
                                str(instrument.id.venue))

    exec_minutes = bar_minutes_of(exec_prices) if exec_prices is not None and len(exec_prices) else None
    if exec_minutes is not None and (exec_minutes >= minutes or minutes % exec_minutes):
        raise ValueError(f"{exec_minutes}-minute execution bars don't divide the {minutes}-minute decision bars")
    if oos_exec_prices is None or not len(oos_exec_prices):
        oos_exec_prices, oos_minutes = exec_prices, exec_minutes
    else:
        oos_minutes = bar_minutes_of(oos_exec_prices)
        if oos_minutes >= minutes or minutes % oos_minutes:
            raise ValueError(f"{oos_minutes}-minute execution bars don't divide the {minutes}-minute decision bars")
    labels: list[str] = []
    resting = [perpetual]  # a perpetual's liquidation price always rests (Advisor 19:00)

    errors: list = []
    error_count = [0]
    folds_n = max(0, (len(research) - train_bars - test_bars) // test_bars + 1)
    total = (1 + len(combos) * (1 + len(COST_LADDER)) + 2 + folds_n * (len(combos) + 2) + len(COST_LADDER)
             + (2 if use_holdout and holdout_days else 0))
    done = [0]

    def bt(name: str, df: pd.DataFrame, params: dict, benchmark: bool = False,
           fees: FeeSchedule | None = None, slippage: float = 0.0, oos: bool = False, trade_from=None,
           feed: pd.DataFrame | None = None, flip: bool = False,
           count_from: pd.Timestamp | None = None, family: bool = False, market_over: dict | None = None) -> BacktestResult:
        """One run. oos: an out-of-sample window or the holdout, on oos_exec_prices, flat until trade_from (a bar's
        close time). feed: these execution bars instead. flip: a first_touch rule's ambiguous candles resolve the
        other way (R2-G1); count_from: its report counts only the candles closing from then on (a window's test
        candles). family: a setting counted for this idea before, re-run for PBO (R4): params already hold the exits
        it was tried with, and what it raises or rests is its own, not this study's."""
        done[0] += 1
        if progress is not None:
            progress(min(done[0] / total, 0.99))
        guarded = risk_profile is not None and not benchmark
        params = {**params, **(market if market_over is None else market_over)}  # market_over: the spot sheet's perp line
        if not benchmark and not family:
            params = {**params, **exits}
        if position_cap is not None and not guarded:
            params = {**params, "position_cap_pct": position_cap}
        if trade_from is not None:
            params = {**params, "trade_from": pd.Timestamp(trade_from).value}
        source = feed if feed is not None else oos_exec_prices if oos else exec_prices
        fine = None
        if source is not None and len(source) and not benchmark:  # nothing rests, nothing to guard
            fine = source[(source.index > df.index[0] - bar) & (source.index <= df.index[-1])]
        res = _charge_absorbed(run_backtest(
            name, df, instrument, params, starting_capital=starting_capital, bar_minutes=minutes,
            risk_profile=risk_profile if guarded else None, exec_prices=fine,
            exec_minutes=bar_minutes_of(source) if fine is not None else 1, fees=fees,
            half_spread=series.plus(slippage) if series is not None else half_spread + slippage,
            first_touch_flip=flip, first_touch_count_from=count_from))
        if family:
            return res
        if not benchmark:
            labels.extend(x for x in res.labels if x not in labels)
            resting[0] = resting[0] or any(d.get("intent") in RESTING_INTENTS for d in res.decisions.values())
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
            definition_hash=legacy_definition_hash(spec.name, full, setup),
            idea_hash=idea_hash or legacy_idea_hash(spec.name),
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
    perp_hold = markets.is_perp({"market": hold_market})
    # RESEARCH-SPOT-FEES (Advisor, 21 Oct): a spot-lowfee sheet's bar is the price held, paying the spot fee and half the
    # spread once in and once out, with no funding; never mixed with the perp bar.
    spot_lowfee = hold_market == markets.SPOT_LOWFEE
    hold_liquidated: list[str] = []
    price_cost = (float(markets.SPOT_LOWFEE_FEES.taker) if spot_lowfee else PRICE_HOLD_TAKER) + spread_used

    def hold_returns(first, last) -> pd.Series | None:
        """The perp hold's daily returns over the bars closing first..last, or None when a funding settlement in it
        has no known rate."""
        if missing_funding({"market": hold_market}, instrument, first - bar, last):
            return None
        window = prices[(prices.index >= first - bar) & (prices.index <= last)]
        run = bt("buy_and_hold", window, {"market": hold_market}, benchmark=True)
        hold_liquidated.extend(f"{t:%d %b %Y} (the window to {last:%b %Y})" for t in liquidated_at(run))
        return whole_days(daily_returns(run.equity), first, bar)

    def price_hold_returns(first, last) -> pd.Series:
        """The price-only hold over the bars closing first..last, paying its entry and exit (as hold_returns)."""
        window = prices[(prices.index >= first - bar) & (prices.index <= last)]["close"]
        return whole_days(daily_returns(price_hold_equity(window, price_cost, starting_capital)), first, bar)

    hold_full_period = not perp_hold or not missing_funding({"market": hold_market}, instrument, research.index[0],
                                                           research.index[-1])
    if perp_hold and hold_full_period and not market:  # a spot venue: the research period's hold is the perp's too
        bench = bt("buy_and_hold", research, {"market": hold_market}, benchmark=True)
        bench_ret = daily_returns(bench.equity)
    if spot_lowfee:
        bench = HoldCurve(price_hold_equity(research["close"], price_cost, starting_capital))
        bench_ret = daily_returns(bench.equity)
    hold_liquidated.extend(f"{t:%d %b %Y} (the research period)" for t in (liquidated_at(bench) if perp_hold else []))
    spot_bench = bt("buy_and_hold", research, {}, benchmark=True) \
        if perp_hold and not (profile is not None and profile.perpetual) else None
    spot_ret = daily_returns(spot_bench.equity) if spot_bench is not None else None
    price_only = perp_hold and spot_ret is None  # a perp-only venue: the price held stands for spot (BH-102 rule 3)
    alt_label = PRICE_HOLD if price_only else "spot hold" if spot_ret is not None else ""
    def fold_benchmark(test_ret: pd.Series, first, last) -> tuple[pd.Series, tuple]:
        """A test window's benchmark, and what the window adds to the comparison (its days, the strategy's days the
        hold covers, the spot hold's), kept per window so a first_touch re-run can replace them (R2-G1)."""
        spot = (price_hold_returns(first, last).reindex(test_ret.index).dropna() if price_only else
                spot_ret.reindex(test_ret.index).dropna() if spot_ret is not None else None)
        if spot_lowfee:  # each window's own hold, in and out at the spot fee, as a perp hold's window is
            return price_hold_returns(first, last).reindex(test_ret.index).dropna(), (len(test_ret), test_ret, spot)
        if not perp_hold:
            return bench_ret.reindex(test_ret.index).dropna(), (len(test_ret), test_ret, spot)
        held = hold_returns(first, last)
        if held is None:
            return pd.Series(dtype=float), (len(test_ret), None, spot)
        held = held.reindex(test_ret.index).dropna()
        return held, (len(test_ret), test_ret.reindex(held.index), spot)

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
    grid_returns = {}  # every configuration's daily net returns, discarded ones too: what PBO is fed (R4)
    for i, params in enumerate(combos):
        res = bt(spec.name, research, params)
        grid_returns[i] = daily_returns(res.equity)
        m = summary(grid_returns[i])
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
    # The research period shown (trade list, costs, regimes) on the out-of-sample windows' 1-minute bars, as paper
    # fills; the settings sweep stays on the coarse bars, which the falsifier below measures against.
    coarse_default = full_default
    one_minute = minutes > 1 and oos_minutes == 1
    if one_minute and (exec_minutes or minutes) > 1:
        full_default = bt(spec.name, research, default_params, feed=oos_exec_prices)
    elif one_minute:  # every run already on 1-minute bars: the falsifier's coarse side is the bars alone
        kept = list(labels)  # its fills are the falsifier's, not this study's
        try:
            coarse_default = bt(spec.name, research, default_params, feed=oos_exec_prices.iloc[:0])
        except ValueError:  # a first_touch rule can't run on bars alone, so there is nothing to falsify against
            coarse_default = None
        labels[:] = kept
    sensitivity = pd.DataFrame(rows)
    overfit = _overfit(grid_returns, [{**p, **exits} for p in combos], register, spec, research, bt, idea_hash)
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
    def scored(k, train, through_test, test_idx, best, best_sharpe, surface, run):
        """Test window k's fold from its run: its days, not the train's, scored alone."""
        # The test window's days, not the train's, its first measured from the flat capital it started with (P1-D13).
        test_ret = whole_days(daily_returns(_flat_until(run, test_idx[0] - bar)), test_idx[0], bar)
        b_ret, parts = fold_benchmark(test_ret, test_idx[0], test_idx[-1])
        absorbed = _absorbed_in(run, test_idx[0] - bar, test_idx[-1])
        window_trips = trades(fills_to_rows(run.fills), run.shorts, open_trip=True)
        booked = trades(fills_to_rows(run.fills), run.shorts, run.funding, open_trip=True)  # R-16: with funding
        return (
            Fold(
                train_start=train.index[0],
                train_end=train.index[-1],
                test_end=test_idx[-1],
                chosen=best,
                train_sharpe=best_sharpe,
                test=summary(test_ret),
                benchmark_test=_summary_or_none(b_ret),
                **_window_trips(window_trips, test_idx[0] - bar),
                window_bars=len(test_idx),
                in_market_bars=_in_market_bars(run.exposure, test_idx[0], test_idx[-1]),
                halted=_halt_words(run.risk_events, test_idx[0], test_idx[-1]),
                halted_before_test=_halted_before(run.risk_events, test_idx[0]),
                grid=pd.DataFrame(surface),
                **_window_funding(run.funding_marks, train.index[-1], test_idx[-1]),
                liquidations=_liquidations(run, test_idx[0], test_idx[-1], f"out-of-sample test window {k + 1}"),
                absorbed=absorbed,
                uncharged_returns=_uncharged(run, test_idx[0], bar) if absorbed else None,
                trip_charges=_trip_charges(window_trips, test_idx[0] - bar, absorbed),
                trip_sizes=_trip_sizes(window_trips, test_idx[0] - bar, run.fills, run.decisions),
                gross_exposure=_gross_exposure(run.exposure, test_idx[0], test_idx[-1]),
                trade_rows=periods.trade_rows(run),
                trade_from=test_idx[0] - bar,
                **_trip_reconcile(booked, _flat_until(run, test_idx[0] - bar), test_idx[0] - bar, test_idx[-1],
                                  _trip_charges(window_trips, test_idx[0] - bar, absorbed), run.decisions),
            ),
            test_ret, b_ret, parts, run,
        )

    folds: list[Fold] = []
    fold_runs: list = []
    windows = []  # (index in folds, inputs, run) of each window that chose a setting
    oos_parts, bench_parts, compare_by = [], [], []  # compare_by: fold_benchmark's parts, per window
    stress_parts = []  # _stop_stress, per window that chose a setting
    start = 0
    while start + train_bars + test_bars <= len(research):
        train = research.iloc[start : start + train_bars]
        through_test = research.iloc[start : start + train_bars + test_bars]
        test_idx = through_test.index[train_bars:]
        best, best_sharpe = None, float("-inf")
        surface = []
        if warmup_bars is not None:  # continuous: the one pinned setting, nothing fitted on the warm-up
            best, best_sharpe, combos_fit = combos[0], float("nan"), []
        else:
            combos_fit = combos
        for params in combos_fit:
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
            b_ret, parts = fold_benchmark(test_ret, test_idx[0], test_idx[-1])
            folds.append(Fold(train_start=train.index[0], train_end=train.index[-1], test_end=test_idx[-1], chosen={},
                              train_sharpe=float("nan"), test=summary(test_ret), benchmark_test=_summary_or_none(b_ret),
                              window_bars=len(test_idx), unscored=True, grid=pd.DataFrame(surface)))
            oos_parts.append(test_ret)
            bench_parts.append(b_ret)
            compare_by.append(parts)
            start += test_bars
            continue
        # The chosen params over training and the test window, trading only from the window's first bar: it starts
        # flat (P1-D13 19:00), and the training bars before it warm the model up. Scored on the test days alone.
        window = (train, through_test, test_idx, best, best_sharpe, surface)
        # The run's first_touch report counts the test candles only: the share G1 reads is out of sample (CR #171).
        run = bt(spec.name, through_test, best, oos=True, trade_from=test_idx[0] - bar, count_from=test_idx[0])
        fold_runs.append((through_test, test_idx, best, run))
        fold, test_ret, b_ret, parts, run = scored(len(folds), *window, run)
        stress_parts.append(_stop_stress(run, test_idx, bar, test_ret))
        windows.append((len(folds), window, run))
        folds.append(fold)
        compare_by.append(parts)
        oos_parts.append(test_ret)
        bench_parts.append(b_ret)
        start += test_bars

    # R2-G1-F1: each window run counts its test candles only, so these are out-of-sample counts, pooled and per fold.
    first_touch = {}
    counts = first_touch_counts(run for _, _, run in windows)
    share = flip = None
    if counts is not None:
        ambiguous, reached = counts
        share = ambiguous / reached if reached else 0.0
        fold_shares = [first_touch_share([run]) or 0.0 for _, _, run in windows]
        flip = needs_flip(share, fold_shares, reached)
        first_touch = {"share": share, "ambiguous": ambiguous, "reached": reached, "fold_shares": fold_shares,
                       "threshold": FIRST_TOUCH_FLIP_SHARE, "min_reached": FIRST_TOUCH_MIN_REACHED, "flipped": flip,
                       "judged_on": "as ruled", "resolved": _resolved(run for _, _, run in windows)}
    if flip:
        # Every window again with its ambiguous candles resolved the other way; G1 then judges the worse.
        alt_folds, alt_oos, alt_bench, alt_compare = list(folds), list(oos_parts), list(bench_parts), list(compare_by)
        for k, window, _ in windows:
            alt_folds[k], alt_oos[k], alt_bench[k], alt_compare[k], flipped = scored(
                k, *window, bt(spec.name, window[1], window[3], oos=True,
                                trade_from=window[2][0] - bar, flip=True, count_from=window[2][0]))
            first_touch["opposite_resolved"] = _resolved([flipped])
        ruled, opposite = pd.concat(oos_parts), pd.concat(alt_oos)
        first_touch["as_ruled"], first_touch["opposite"] = summary(ruled), summary(opposite)
        if _worse(opposite, ruled):
            folds, oos_parts, bench_parts, compare_by = alt_folds, alt_oos, alt_bench, alt_compare
            first_touch["judged_on"] = "opposite"
    # B-22 (Advisor B22-DOUBLE-FEES): each window again with the settings it chose, at twice its fees, its spread,
    # slippage and funding as they were, judged on what G1 judged (the opposite resolution when that was worse). A real
    # re-run, never the fees subtracted: paying more leaves less to trade with, so the trades themselves move. A window
    # that chose nothing sat flat, at any fee. Never the holdout, and never counted again in G1's first-touch share.
    twice_parts, twice_runs = list(oos_parts), []
    for k, window, run in windows:
        rerun = bt(spec.name, window[1], window[3], oos=True, trade_from=window[2][0] - bar,
                   fees=doubled(paid(window[3])), flip=first_touch.get("judged_on") == "opposite")
        twice_parts[k] = scored(k, *window, rerun)[1]
        twice_runs.append((run, rerun))
    double_fee_returns = pd.concat(twice_parts) if twice_parts else pd.Series(dtype=float)
    # BH-102 rule 3 (Advisor, 21 Oct): a long-only perp study must beat the HIGHER of the perp hold and the spot (or
    # price-only) hold, by Sharpe over the out-of-sample days both cover, chosen once for the whole sheet: every reader
    # of the benchmark (G1's Sharpe test, stop stress, the parameter checks, the full-period and holdout rows) then
    # takes that hold, never one here and the other there. A perp hold without enough known funding still gives no
    # verdict, as before, and so does one missing any window (TEC-BH102-HOLD 4, pending the Advisor).
    bar_hold, bar_note, switched = ("perp hold" if perp_hold else ""), "", False
    spots = [spot for _, _, spot in compare_by]
    if perp_hold and alt_label and any(s is not None and len(s) for s in spots):
        perp_cmp = pd.concat([b for b in bench_parts if len(b)]) if any(len(b) for b in bench_parts) else \
            pd.Series(dtype=float)
        alt_cmp = pd.concat([s for s in spots if s is not None])
        both = perp_cmp.index.intersection(alt_cmp.index)
        ps, ss = _sharpe_or_floor(perp_cmp.reindex(both)), _sharpe_or_floor(alt_cmp.reindex(both))
        # TEC-BH102-HOLD item 4, ruled (A) by the Advisor (21 Oct), deferred to after Deploy 2: a fold missing a
        # settlement will get the perp hold at the baseline rate, labelled, the bar still chosen once per sheet. Until
        # then a perp hold that misses any window keeps today's rule: its windows unjudged, the other hold never in
        misses = any(part is None for _, part, _ in compare_by)
        switched = ss > ps and not misses
        bar_hold = alt_label if switched else "perp hold"
        words = lambda s: "n/a" if s == float("-inf") else f"{s:.2f}"  # noqa: E731
        bar_note = (f"bar: {'the ' + alt_label if switched else 'the perp hold'}, the higher Sharpe over the "
                    f"{len(both)} out-of-sample days both cover (perp hold {words(ps)}, {alt_label} {words(ss)})")
    perp_shown = (pd.concat([b for b in bench_parts if len(b)]) if perp_hold and any(len(b) for b in bench_parts)
                  else None)
    if switched:
        import dataclasses

        for k, spot in enumerate(spots):
            spot = spot if spot is not None else pd.Series(dtype=float)
            bench_parts[k] = spot
            compare_by[k] = (compare_by[k][0], oos_parts[k].reindex(spot.index).dropna(), spot)
            folds[k] = dataclasses.replace(folds[k], benchmark_test=_summary_or_none(spot))
        bench = (spot_bench if spot_bench is not None else
                 HoldCurve(price_hold_equity(research["close"], price_cost, starting_capital)))
    oos_days = [sum(days for days, _, _ in compare_by)]
    compare_parts = [part for _, part, _ in compare_by if part is not None]
    spot_parts = [spot for _, _, spot in compare_by if spot is not None]
    run_fees = paid(chosen)
    spot_entry_tier = spot_vs_perp = None
    if spot_lowfee:
        tier = bt(spec.name, research, chosen, fees=SPOT_ENTRY_TIER_FEES, slippage=slip)
        spot_entry_tier = LadderRung(fee=float(SPOT_ENTRY_TIER_FEES.taker),
                                     total_return=float(tier.equity.iloc[-1] / starting_capital - 1)
                                     if len(tier.equity) else 0.0,
                                     sharpe=summary(daily_returns(tier.equity))["sharpe"],
                                     round_trips=len(round_trips(tier.fills, tier.shorts)), fees_paid=tier.fees_paid)

        def line(res) -> dict:
            s = summary(daily_returns(res.equity))
            return {"end": float(res.equity.iloc[-1]), "cagr": s["cagr"], "sharpe": s["sharpe"],
                    "max_drawdown": s["max_drawdown"], "funding_paid": -sum(f["amount"] for f in res.funding)}

        spot_vs_perp = {"spot": line(bt(spec.name, research, chosen)),
                        "perp": line(bt(spec.name, research, chosen, market_over={"market": PERP}))}
    covered = len(pd.concat(compare_parts)) if compare_parts else 0
    hold_insufficient = perp_hold and not switched and covered < HOLD_COVERAGE_FLOOR * oos_days[0]
    if perp_hold:
        terms = markets.terms({"market": hold_market}, str(instrument.id.venue))
        paying = (f"assumed funding of {terms.funding_rate:.2%} every settlement, as the strategy's runs pay"
                  if terms.funding_venue is None else "the venue's settled funding")
        hold_note = (f"buy and hold is a 1x long perpetual paying its fees and {paying}; it covers {covered} of "
                     f"{oos_days[0]} out-of-sample days, the windows whose funding is all known")
        if switched:
            hold_note = (f"the {alt_label} set the bar, so it is the benchmark on every line of this sheet; the 1x long "
                         f"perpetual paying its fees and {paying} is shown beside it")
        if hold_insufficient:
            hold_note += (f": insufficient funding history (under {HOLD_COVERAGE_FLOOR:.0%}), so it gives no verdict; "
                          "random entry still judges")
    elif spot_lowfee:
        hold_note = (f"buy and hold is the price held, paying the {markets.SPOT_LOWFEE_FEES.taker:.2%} spot fee and half "
                     "the spread once in and once out, with no funding (research only)")
    else:
        hold_note = ""
    dists = [x for _, _, d in stress_parts for x in d]
    stop_stress = {"charge": pd.concat([c for c, _, _ in stress_parts]) if stress_parts else pd.Series(dtype=float),
                   "stops": sum(n for _, n, _ in stress_parts), "bp": STOP_STRESS * 1e4,
                   "median_stop": float(pd.Series(dists).median()) if dists else None,
                   "required_below": STOP_STRESS_REQUIRED_BELOW}
    result = StudyResult(
        stop_stress=stop_stress,
        idea_hash=idea_hash,
        spec=spec,
        dataset=dataset,
        synthetic=synthetic,
        research_start=research.index[0],
        research_end=research.index[-1],
        holdout_days=holdout_days,
        default_params=default_params,
        full_period=full_default,
        full_period_benchmark=bench,
        research_closes=research["close"].astype(float),
        sensitivity=sensitivity,
        pbo=overfit,
        folds=folds,
        oos_returns=pd.concat(oos_parts),
        oos_benchmark_returns=pd.concat(bench_parts),
        oos_compare_returns=pd.concat(compare_parts) if compare_parts else pd.Series(dtype=float),
        spot_hold_returns=pd.concat(spot_parts) if spot_parts and not price_only else None,
        perp_hold_returns=perp_shown,
        bar_hold=bar_hold,
        bar_note=bar_note,
        hold_liquidated=hold_liquidated,
        hold_market=hold_market,
        hold_full_period=hold_full_period,
        hold_note=hold_note,
        hold_insufficient=hold_insufficient,
        instrument=pair_of(instrument),
        bar_minutes=minutes,
        venue=str(instrument.id.venue),
        risk_profile=risk_profile,
        settings=(f"{risk_profile + ' risk profile' if risk_profile else 'no risk profile'} · "
                  f"exits: {_exit_words(exits) if exits else 'the signal only'}{_definition_target_words(default_params)} · "
                  + (f"walk-forward {train_days} days training, {test_days} days testing" if warmup_bars is None
                     else f"continuous: a warm-up of {warmup_bars:,} bars ({warmup_bars / per_day:,.0f} days), then "
                          f"scored {_day_words(research.index[warmup_bars], bar)} to "
                          f"{_day_words(research.index[-1], bar)}")),
        errors=errors,
        error_count=error_count[0],
        cost_ladder=ladder,
        first_touch=first_touch,
        ladder_slippage=slip,
        chosen_params=chosen,
        breakeven=breakeven,
        oos_exec_minutes=oos_minutes,
        fill_labels=labels,
        fee_note=(f"{pct(run_fees.maker)} maker on post-only orders, {pct(run_fees.taker)} taker "
                  f"on every other order, plus {spread_words} on orders that take liquidity; break-even: {breakeven}"
                  + (f" ({markets.SPOT_LOWFEE_WORDS})" if market.get("market") == markets.SPOT_LOWFEE else "")),
        fee_basis=run_fees,
        spot_entry_tier=spot_entry_tier,
        spot_vs_perp=spot_vs_perp,
        double_fee_returns=double_fee_returns,
        double_fees=_double_fees(pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float), double_fee_returns,
                                 twice_runs, doubled(run_fees)),
        oos_uncharged_returns=pd.concat(
            [part if f.uncharged_returns is None else f.uncharged_returns for f, part in zip(folds, oos_parts)])
        if any(f.absorbed for f in folds) else None,
    )
    # Every trip pays the taker fee and half the spread each way, as the study's own runs do on market orders.
    # A perpetual's trades and draws are liquidated as the engine books one, at the risk profile's leverage cap.
    # B-35 (a): lowered by the Strategy's own leverage cap (max_leverage), as the engine's margin and liquidation are
    leverage = (risk.capped_leverage(risk_profile_of(risk_profile), default_params)
                if risk_profile is not None else None)
    # The regime table's hold column is "price only, no fees": the closes themselves, never the funded hold's equity,
    # which a funding drain can take below zero (REGIME-BH, 21 Oct).
    result.by_regime = regime_split.split(
        trades(fills_to_rows(full_default.fills), full_default.shorts, full_default.funding),
        research, research["close"].astype(float), full_default.equity)
    fill = _draw_fill(spec, folds, minutes, oos_exec_prices, instrument, research, float(run_fees.taker))
    result.random_entry, result.random_side = _benchmarks(
        research, folds, test_bars, float(run_fees.taker), series if series is not None else spread_used,
        full_default.shorts, leverage, market.get("market"), True, fill)
    if any(f.trip_charges for f in folds):  # the strategy's trips uncharged, for information only (ABSORB-GAP)
        result.random_entry_uncharged, result.random_side_uncharged = _benchmarks(
            research, folds, test_bars, float(run_fees.taker), series if series is not None else spread_used,
            full_default.shorts, leverage, market.get("market"), False, fill)
    result.score_window, result.warmup_bars = score_window, warmup_bars
    if warmup_bars is not None and windows:  # R-14: the percentile per test year too
        _k, (_, _, test_idx, *_), run = windows[0]
        result.random_entry_by_year = {
            year: _benchmarks(research, [fold], window_bars, float(run_fees.taker),
                              series if series is not None else spread_used, full_default.shorts, leverage,
                              market.get("market"))[0]
            for year, fold, window_bars in _years(folds[0], run, test_idx)}
    if (default_params or {}).get("vol_target_size") is not None:  # R-07: against the parent at equal exposure
        result.equal_exposure = _equal_exposure(
            result, research, folds, windows, oos_parts, bt, scored, bar,
            lambda fs: _benchmarks(research, fs, test_bars, float(run_fees.taker),
                                   series if series is not None else spread_used, full_default.shorts, leverage,
                                   market.get("market"))[0])
    if placebo_parent is not None:  # R-24: a filtered variant, its filter against a placebo of its parent
        result.placebo = _placebo(result, placebo_parent, windows, bt, bar, float(run_fees.taker),
                                  series if series is not None else spread_used, oos_exec_prices, minutes,
                                  f"{pair_of(instrument)} {(default_params or {}).get('market') or market.get('market') or 'spot'}")
    for rung in ladder:  # the benchmark at each rung's own cost, as the rung's runs pay it (RE-COST)
        at_rung, _ = _benchmarks(research, folds, test_bars, rung.fee + slip,
                                 series if series is not None else spread_used, full_default.shorts, leverage,
                                 market.get("market"), True, fill,
                                 random_side_test=False)  # the rung: random entry only (CR F1)
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
            f"{strategy_cap:.0%} of capital{' in notional' if markets.is_perp(market) else ''}, the drawdown halt (flat for the rest of the run, as paper stays halted "
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
    if oos_minutes is not None and oos_minutes != exec_minutes:
        result.notes.append(f"The out-of-sample windows and the holdout are matched on {oos_minutes}-minute bars"
                            + (", and so are the research period's trades, costs and regime table. The settings "
                               "sweep, the overfitting check (PBO) and the cost ladder, including the break-even fee, "
                               f"stay on the coarser {exec_minutes or minutes}-minute bars." if one_minute else "."))
    elif risk_profile is not None:
        result.notes.append(f"The risk guard values the book once a {_bar_words(minutes)}; paper does every 30 seconds.")
    if minutes < 1440:
        result.notes.append(
            f"Traded on {minutes}-minute bars; every figure here is on daily returns (closes at 00:00 UTC), "
            "so Sharpe is annualised as daily and the bootstrap resamples days, as for a daily strategy.")
    if market.get("market") == markets.SPOT_LOWFEE:
        result.notes.insert(0, f"Assumed spot fee {markets.SPOT_LOWFEE_FEES.taker:.2%} (research only); paper needs the "
                               "account's own rate. Spot priced on perpetual prices, so the gap between the two prices is "
                               "ignored (small for daily rules; it does not cover funding).")
        result.notes.append(f"Every run, the buy and hold too, traded spot on the venue's prices: long only, in a cash "
                            f"account with no margin and no funding, at the fees above ({markets.SPOT_LOWFEE_WORDS}).")
    elif market:
        result.notes.append(f"The venue lists perpetuals only, so every run traded the perpetual, "
                            f"{_sides_words(full_default, default_params)}, paying the funding the venue settled.")
    if exits:
        result.notes.append(
            "Exits on top of the signal: " + _exit_words(exits)
            + ". The stop rests at the venue and fills at its level (at market at once if the price is already "
            "through it), with its slippage; the target fills when the price reaches it, at its level less the "
            "slippage. A bar that reaches both takes the stop, unless it opened through the target. On bars coarser "
            "than a minute, where what traded first inside a bar is unknown, a stop the bar traded through fills at "
            "the bar's worst price. Paper watches both on every trade."
        )
    counts = full_default.rule_counts or {}
    if "funding_no_rate" in counts:  # R-06
        result.notes.append(
            f"Funding rate decile: {counts['funding_no_rate']:,} of {len(research):,} decision candles had no usable "
            "stored funding rate (the latest due settlement missing, or too few in its window), so no entry gated on "
            "it could open on them: a missing rate is never assumed. Apart from those, "
            f"{counts.get('funding_top_decile', 0):,} read the top decile (90% or more of the trailing window's rates "
            "strictly below the latest). Only rates settled strictly before each candle's close are read.")
    if "vol_skipped" in counts:  # R-07
        vol = default_params.get("vol_target_size") or {}
        result.notes.append(
            f"Entries are sized to {vol.get('annual', 0):.0%} a year of volatility, read over "
            f"{vol.get('lookback_days', 0)} days of daily closes at each entry: never above the size or caps it would "
            f"otherwise have, set at entry and not rebalanced. {counts['vol_skipped']:,} entries were skipped for "
            "want of the daily closes to read it.")
    if spec.name == "rules" and len(full_default.exposure):  # R-07: what V3/V4 are judged against the parent at
        result.notes.append(f"Average gross exposure over the research period: "
                            f"{float(full_default.exposure.abs().mean()):.1%} of equity.")
    result.resting_exits = resting[0]
    if result.resting_exits and oos_minutes is not None and 1 < oos_minutes <= ONE_WAY_MINUTES and not result.not_judged:
        _one_way(result, ledger, fold_runs, bt, minute_loader, bar)
    if one_minute:
        # The falsifier: the research period at the default settings on 1-minute bars against the coarse run.
        if coarse_default is not None and len(full_default.equity) and len(coarse_default.equity):
            result.bars_only_gap = float(full_default.equity.iloc[-1] - coarse_default.equity.iloc[-1]) / starting_capital
            net = float(full_default.equity.iloc[-1]) / starting_capital - 1
            if abs(result.bars_only_gap) > 0.2 * abs(net):  # Advisor 22 Oct: only when it would change the reading
                by = f" by {abs(result.bars_only_gap) / abs(net):.0%}" if net else ""
                result.notes.append(f"1-minute and bar-only results differ{by}; read with care.")
            ledger.record(idea=spec.name, family=spec.family, params={**default_params, **exits}, dataset=dataset,
                          stage=GAP_STAGE, sharpe=summary(daily_returns(coarse_default.equity))["sharpe"],
                          extra={"bars_only_gap": round(result.bars_only_gap, 8)})

    if register is not None:
        from sleeve_fund.research.trials import legacy_idea_hash

        result.failed_counts = register.failed(idea_hash or legacy_idea_hash(spec.name))

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

        idea, underlying = idea_hash or legacy_idea_hash(spec.name), underlying_of(pair_of(instrument))
        locked = locks.refusal(idea, underlying, prices.index[-holdout_bars], prices.index[-1])
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
    elif (use_holdout and holdout_days and locks is not None and locks.undated(idea)
          and (few := _holdout_trades(bt(spec.name, prices, folds[-1].chosen), prices.index[-holdout_bars]))
          < MIN_UNDATED_HOLDOUT_TRADES):
        # Counted before the claim, and only the count is shown: spending the one clean look on too few trades
        # to judge would waste it (Advisor, C4 option b).
        result.holdout_withheld = (f"left closed, though asked for: no holdout yet: the held-back days hold {few} "
                                   f"trades, and {MIN_UNDATED_HOLDOUT_TRADES} are needed: "
                                   f"{MIN_UNDATED_HOLDOUT_TRADES - few} more")
    elif use_holdout and holdout_days and locks is not None and not locks.open(
            idea, underlying, prices.index[-holdout_bars], prices.index[-1]):
        # Claimed before the look: another study took the lock since the check above, so this one doesn't look.
        result.holdout_withheld = ("left closed, though asked for: another study opened this idea's holdout on "
                                   f"{underlying} while this one ran")
    elif use_holdout and holdout_days:
        chosen = folds[-1].chosen
        h_start = prices.index[-holdout_bars]
        try:
            run = bt(spec.name, prices, chosen, oos=True, trade_from=h_start - bar)  # starts flat, as each window does
            h_ret = whole_days(daily_returns(_flat_until(run, h_start - bar)), h_start, bar)
            if first_touch.get("flipped"):
                # The one look covers both resolutions, and the holdout too is judged on the worse (R2-G1).
                flipped = bt(spec.name, prices, chosen, oos=True, trade_from=h_start - bar, flip=True)
                other = whole_days(daily_returns(_flat_until(flipped, h_start - bar)), h_start, bar)
                worse = _worse(other, h_ret)
                result.first_touch["holdout_judged_on"] = "opposite" if worse else "as ruled"
                if worse:
                    h_ret, run = other, flipped  # its liquidations too are the judged run's
            result.holdout = summary(h_ret)
            if switched and price_only:  # the bar's hold, as on every other line (BH-102 rule 3)
                result.holdout_benchmark = _summary_or_none(price_hold_returns(h_start, prices.index[-1]))
            elif switched:  # the spot hold bought at the close before the holdout, as the price-only one is (Advisor)
                b_h = bt("buy_and_hold", prices[prices.index >= h_start - bar], {}, benchmark=True)
                result.holdout_benchmark = summary(whole_days(daily_returns(b_h.equity), h_start, bar))
            elif perp_hold:
                hb_ret = hold_returns(h_start, prices.index[-1])
                result.holdout_benchmark = None if hb_ret is None else _summary_or_none(hb_ret)
                if hb_ret is None:
                    result.notes.append("The holdout has no buy-and-hold line: a funding settlement in it has no "
                                        "known rate, and a missing rate is never filled in.")
            elif spot_lowfee:
                result.holdout_benchmark = _summary_or_none(price_hold_returns(h_start, prices.index[-1]))
            else:
                b_all = bt("buy_and_hold", prices, {}, benchmark=True)
                result.holdout_benchmark = summary(whole_days(daily_returns(b_all.equity), h_start, bar))
            result.holdout_funding = _window_funding(run.funding_marks, prices.index[-holdout_bars - 1],
                                                     prices.index[-1])
            result.holdout_liquidations = _liquidations(run, h_start, prices.index[-1], "holdout")
            result.holdout_absorbed = _absorbed_in(run, h_start - bar, prices.index[-1])
            if result.holdout_absorbed:
                result.holdout_uncharged = summary(_uncharged(run, h_start, bar))
        except Exception:
            if locks is not None:
                # The lock was claimed before the look, so the holdout is spent: recorded as a crash, not a result.
                locks.crashed(idea, underlying)
                log_.error("holdout of %s on %s spent by a crash, not a failed result: the lock stays", spec.name,
                           underlying)
            raise
        trial = log(chosen, "holdout", result.holdout["sharpe"], prices.iloc[-holdout_bars:])
        if locks is not None:
            locks.settle(idea, underlying, trial)
    return result


STOP_STRESS = 0.002  # D13-INTRAMINUTE-JUMP (3): 20 bp more on each stop exit, a jump through the stop paper may see
STOP_STRESS_REQUIRED_BELOW = 0.005  # GAP-DOWN (Advisor 15 Oct): the line is required under this median stop distance


def _stop_stress(run: BacktestResult, test_idx, bar: pd.Timedelta, test_ret: pd.Series) -> tuple[pd.Series, int, list]:
    """A test window's stop stress: the daily returns lost if each stop exit filled in it cost STOP_STRESS more of its
    notional, charged on the equity curve from its fill on, with no re-sizing after it; how many stop exits; and
    the stop distance of each entry filled in it."""
    fills = run.fills
    lo, hi = test_idx[0] - bar, test_idx[-1]
    cost = pd.Series(0.0, index=run.equity.index)
    stops, dist = 0, []
    for o, f in (fills.iterrows() if fills is not None and len(fills) else ()):
        ts = pd.to_datetime(f["ts_last"], utc=True)
        if not lo < ts <= hi:
            continue
        d = run.decisions.get(o) or {}
        if d.get("intent") == "stop_loss":  # a resting stop's exit; a liquidation books at bankruptcy
            stops += 1
            cost[cost.index >= ts] += STOP_STRESS * float(f["filled_qty"]) * float(f["avg_px"])
        elif d.get("intent") == "entry" and (d.get("signal") or {}).get("stop_frac"):
            dist.append(float(d["signal"]["stop_frac"]))
    if not stops:
        return pd.Series(0.0, index=test_ret.index), 0, dist
    from types import SimpleNamespace

    stressed = SimpleNamespace(equity=run.equity - cost, starting_capital=run.starting_capital)
    s_ret = whole_days(daily_returns(_flat_until(stressed, lo)), test_idx[0], bar)
    return (test_ret - s_ret.reindex(test_ret.index)).fillna(0.0), stops, dist


def _charge_absorbed(run: BacktestResult) -> BacktestResult:
    """ABSORB-GAP (INSURE-RULE point 5; Advisor 15 Oct ~02:05 UK): a liquidation is booked at its bankruptcy price
    (GAP-LIQ-CAP), so a run that gapped past it never pays the gap the venue absorbed, and looks better than the trade
    was. Research charges it: the run's equity less the absorbed gaps to date, stepped down on the first bar closing
    at or after each liquidation, nothing re-sized ("gap charged, no re-sizing"). Every test that reads returns then
    reads the charged curve; the engine's own is kept as uncharged_equity, for information only. A run that absorbed
    nothing is left as it was."""
    if not run.absorbed or not len(run.equity):
        return run
    index = run.equity.index
    steps = pd.Series(0.0, index=index)
    for a in run.absorbed:
        at = int(index.searchsorted(_utc(a["ts"]), side="left"))
        if at < len(index):
            steps.iloc[at] += float(a["amount"])
    run.uncharged_equity = run.equity
    run.equity = (run.equity - steps.cumsum()).rename(run.equity.name)
    return run


def _absorbed_in(run: BacktestResult, after, end) -> list[dict]:
    """The run's absorbed gaps (ABSORB-GAP) in the bars closing after `after` up to `end`: a window's own."""
    lo, hi = _utc(after), _utc(end)
    return [{"ts": _utc(a["ts"]), "amount": float(a["amount"])} for a in run.absorbed if lo < _utc(a["ts"]) <= hi]


def _uncharged(run: BacktestResult, first, bar: pd.Timedelta) -> pd.Series:
    """A window's days from its first bar `first` on the engine's own curve, the absorbed gaps uncharged."""
    flat = run.uncharged_equity.copy()
    flat[flat.index <= first - bar] = run.starting_capital
    return whole_days(daily_returns(flat), first, bar)


def _flat_until(run: BacktestResult, start) -> pd.Series:
    """A window's equity, flat at the run's starting capital up to and at `start`, the close of the bar before the
    window: a state rule enters at that close, so the entry's fee and spread fall on the window's first day, which
    the window pays for itself (P1-D13 [flat-rule])."""
    equity = run.equity.copy()
    equity[equity.index <= start] = run.starting_capital
    return equity


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
    # ABSORB-GAP: the gap past bankruptcy the venue absorbed on each liquidation, recorded when the engine booked it:
    # the liquidation's own, the last one filled at or before it.
    liquidated = sorted((_utc(f["ts"]), f["order_id"]) for f in fills
                        if orders.get(f["order_id"], {}).get("intent") == "liquidation")
    gaps: dict[str, float] = {}
    for a in getattr(run, "absorbed", []):
        owner = [oid for ts, oid in liquidated if ts <= _utc(a["ts"])]
        if owner:
            gaps[owner[-1]] = gaps.get(owner[-1], 0.0) + float(a["amount"])
    for liq in found.values():
        liq["x"] = round(liq["x"], 2)
        liq["absorbed"] = round(gaps.get(liq["order_id"], 0.0), 2)
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


def _one_way(result: StudyResult, ledger: IdeaLedger, fold_runs: list, bt, minute_loader, bar: pd.Timedelta) -> None:
    """The 5-minute one-way rule (Advisor 18:36 and 19:00): a fail is not final until re-run on 1-minute bars; a pass
    counts once one out-of-sample window, drawn at random with its seed recorded, ends no better on 5-minute bars than
    on 1-minute ones (fees only). If it ends better, the pass is void."""
    import secrets

    import numpy as np

    from sleeve_fund.research.tearsheet import g1_checks, g1_verdict

    verdict, _ = g1_verdict(g1_checks(result, ledger))
    if verdict == "FAIL":
        result.one_way = "fail"
        return
    if verdict != "PASS" or not fold_runs:
        return
    seed = secrets.randbelow(2**31)
    through, test_idx, params, coarse = fold_runs[int(np.random.default_rng(seed).integers(len(fold_runs)))]
    window = f"{test_idx[0]:%d %b %Y} to {test_idx[-1]:%d %b %Y}"
    fine = minute_loader(through.index[0] - bar, through.index[-1]) if minute_loader is not None else None
    if fine is None or not len(fine):
        result.one_way, result.spot_check = "unchecked", {"window": window, "seed": seed}
        return
    exact = bt(result.spec.name, through, params, trade_from=test_idx[0] - bar, feed=fine)
    gap = float(exact.equity.iloc[-1] - coarse.equity.iloc[-1])
    tolerance = abs(exact.fees_paid - coarse.fees_paid) + 0.01
    result.spot_check = {"window": window, "seed": seed, "gap": round(gap, 2)}
    if -gap > tolerance:
        result.one_way = "void"


def idea_of(params: dict | None, idea_hash: str | None) -> str | None:
    """The idea family a study counts under. A rule-builder definition counts under its own structural idea
    (candidates.idea_hash), so unrelated definitions never share the "rules" family's trial count, holdout lock and
    PBO family, and a parent and its retunes share one (CR F236-3). A caller's idea_hash must be that idea: any other
    would file the runs where they don't count, so it is refused (HoQA F236-4). The parent's idea is covered by the
    same check, since a retune shares it and a restructure is its own idea (candidates.validate_candidate). None for
    a hand-coded model, which counts under its name, or the caller's idea_hash as given."""
    if not isinstance((params or {}).get("definition"), dict):
        return idea_hash
    from sleeve_fund.research.candidates import idea_hash as structural

    own = structural(params["definition"])
    if idea_hash is not None and idea_hash != own:
        raise ValueError(f"idea_hash {idea_hash} isn't this definition's own idea {own}; its runs would count under "
                         "another idea's budget and holdout lock")
    return own


def _overfit(grid_returns: dict, grid_settings: list[dict], register, spec: IdeaSpec, research: pd.DataFrame, bt,
             idea_hash: str | None = None):
    """R4's PBO (Advisor 13 Oct 16:50): the sensitivity grid plus every other counted setting of the idea family in the
    trials register, each re-run on this study's research period (never the holdout) and fee model, as daily net
    returns. Ablation and portfolio rows don't count there (trials._counted), so they stay out. A counted setting that
    can't be re-run is listed and labels the result partial. A family of one pre-registered setting has no selection
    (TrialsRegister.no_selection, Advisor 13 Oct 18:05)."""
    import json

    from sleeve_fund.research.pbo import pbo

    columns = {f"grid {i}": r for i, r in grid_returns.items()}
    partial = []
    if register is not None:
        from sleeve_fund.research.pbo import no_selection
        from sleeve_fund.research.trials import legacy_idea_hash

        idea = idea_hash or legacy_idea_hash(spec.name)  # a candidate's own family (CR F236-1)
        if why := register.no_selection(idea):
            return no_selection(why)
        seen = {json.dumps(s, sort_keys=True) for s in grid_settings}
        for t in register._counted(idea):
            try:
                settings = json.loads(t["settings"]) if isinstance(t["settings"], str) else t["settings"]
                if t["status"] == "failed" and set(settings) == {"params"}:  # trials.failed_row wraps them
                    settings = settings["params"]
                key = json.dumps(settings, sort_keys=True)
            except (TypeError, ValueError, AttributeError) as e:
                partial.append(f"trial {t['id']}: its settings can't be read ({e})")
                continue
            if key in seen:
                continue
            seen.add(key)
            try:
                res = bt(spec.name, research, settings, family=True)
            except Exception as e:  # noqa: BLE001 - any failure is listed, never fatal to the study
                partial.append(f"trial {t['id']}: {type(e).__name__}: {e}")
                continue
            if res.handler_errors:
                partial.append(f"trial {t['id']}: the strategy raised errors on re-run")
                continue
            columns[f"trial {t['id']}"] = daily_returns(res.equity)
    return pbo(pd.DataFrame(columns), partial=tuple(partial))


def _window_trips(trips: list[dict], test_start) -> dict:
    """A test-window run's round trips sorted into the counted ones and those at the window's edges."""
    start = _utc(test_start)
    counted = [t for t in trips if t["opened"] is not None and t["closed"] is not None and _utc(t["opened"]) >= start]
    carried_in = sum(1 for t in trips if t["closed"] is not None and _utc(t["closed"]) >= start
                     and (t["opened"] is None or _utc(t["opened"]) < start))
    carried_out = sum(1 for t in trips if t["closed"] is None)  # still open at the window's end, wherever it opened
    return {"test_trades": len(counted), "carried_in": carried_in, "carried_out": carried_out,
            "trips": [(_utc(t["opened"]), _utc(t["closed"]), int(t["side"])) for t in counted]}


def _trip_charges(trips: list[dict], test_start, absorbed: list[dict]) -> list[tuple]:
    """ABSORB-GAP: each absorbed gap on the counted trip it closed, the last opened before it, over that trip's entry
    notional (the base random entry's trip returns are on), as (opened, charge); a gap on a trip carried in or still
    open is in no counted trip, as that trip is in no comparison."""
    start = _utc(test_start)
    counted = [t for t in trips if t["opened"] is not None and t["closed"] is not None and _utc(t["opened"]) >= start]
    out: dict = {}
    for a in absorbed:
        owner = [t for t in counted if _utc(t["opened"]) < a["ts"]]
        if owner and owner[-1]["cost"]:
            key = _utc(owner[-1]["opened"])
            out[key] = out.get(key, 0.0) + a["amount"] / owner[-1]["cost"]
    return sorted(out.items())


def _trip_sizes(trips: list[dict], test_start, fills: pd.DataFrame, decisions: dict) -> list[tuple]:
    """G5: each counted trip's size factor f, from the signal of the order whose fill opened it, as (opened, f); a
    trip whose entry carried none (a flat version) is left out, so it sizes at 1. R-07: a volatility-sized entry's
    share of the size it would otherwise have had is its f the same way (with signal strength too, the product)."""
    if fills is None or fills.empty:
        return []
    start = _utc(test_start)
    at = {}
    for ts, oid in zip(fills["ts_last"], fills.index):
        at.setdefault(_utc(ts), str(oid))
    out = []
    for t in trips:
        if t["opened"] is None or t["closed"] is None or _utc(t["opened"]) < start:
            continue
        sig = (decisions.get(at.get(_utc(t["opened"])), {}) or {}).get("signal") or {}
        f, v = sig.get("size_factor"), sig.get("vol_factor")
        if f is not None or v is not None:
            out.append((_utc(t["opened"]), float(1.0 if f is None else f) * float(1.0 if v is None else v)))
    return out


def _years(fold: Fold, run: BacktestResult, test_idx) -> list[tuple[int, Fold, int]]:
    """R-14: a continuous window cut into its test years, each as (year, a fold of it, its bars). A year holds the
    trips opened in it, whole: its window runs from its first bar to the later of the year's last bar and the close
    of its last trip, and the next year starts after that, so no trip is split and no bar is in two years."""
    out, at, taken = [], 0, 0
    trips = sorted(fold.trips)
    for year in sorted(set(test_idx.year)):
        if at >= len(test_idx):
            break
        last = max(int(test_idx.searchsorted(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), side="right")) - 1, at)
        # the trips opened on this year's bars (a bar is the year its close falls in, 00:00 1 Jan the old year's)
        mine = [t for t in trips[taken:] if _utc(t[0]) <= test_idx[last]]
        taken += len(mine)
        if mine:
            last = max(last, int(test_idx.searchsorted(max(_utc(t[1]) for t in mine), side="left")))
        last = min(last, len(test_idx) - 1)
        first_ts, last_ts = test_idx[at], test_idx[last]
        out.append((year, replace(fold, test_end=last_ts, trips=mine, window_bars=last - at + 1,
                                  in_market_bars=_in_market_bars(run.exposure, first_ts, last_ts)), last - at + 1))
        at = last + 1
    return out


def _test_exposure(run: BacktestResult | None, test_idx) -> pd.Series:
    """|exposure| at each of a test window's bars, 0 where flat or never run (a window that chose nothing)."""
    if run is None or run.exposure is None or run.exposure.empty:
        return pd.Series(0.0, index=test_idx)
    return run.exposure.abs().reindex(test_idx).fillna(0.0)


def _equal_exposure(result: StudyResult, research: pd.DataFrame, folds: list[Fold], windows: list, oos_parts: list,
                    bt, scored, bar: pd.Timedelta, benchmark) -> equal_exposure.EqualExposure:
    """R-07: the parent (the variant's settings without vol_target_size) re-run on every window that chose a setting,
    as the variant ran it, never logged as a try; a window that chose nothing sat flat for both. Then k and the
    comparison (equal_exposure), with the parent's random entry at every trip sized k (so costed at k x notional).
    benchmark: the study's random entry for a list of folds."""
    idx = research.index
    variant_runs = {k: run for k, _, run in windows}
    parent_folds, parent_parts, parent_runs = list(folds), list(oos_parts), {}
    for k, window, _ in windows:
        params = {p: v for p, v in window[3].items() if p != "vol_target_size"}
        test_idx = window[2]
        run = bt(result.spec.name, window[1], params, oos=True, trade_from=test_idx[0] - bar, count_from=test_idx[0])
        parent_folds[k], parent_parts[k] = scored(k, *window[:3], params, *window[4:], run)[:2]
        parent_runs[k] = run
    bars = [idx[(idx > f.train_end) & (idx <= f.test_end)] for f in folds]
    v_exp = pd.concat([_test_exposure(variant_runs.get(k), b) for k, b in enumerate(bars)])
    p_exp = pd.concat([_test_exposure(parent_runs.get(k), b) for k, b in enumerate(bars)])
    k, _ = equal_exposure.exposure_ratio(v_exp, p_exp)
    scale = k if math.isfinite(k) else 0.0
    scaled_folds = []
    for f in parent_folds:
        sizes = dict(f.trip_sizes)
        scaled_folds.append(replace(f, trip_sizes=[(o, scale * sizes.get(o, 1.0)) for o, _, _ in f.trips]))
    scaled_pct = benchmark(scaled_folds).return_percentile if scale > 0 else math.nan
    from sleeve_fund.research.random_entry import BAR

    own = (result.random_entry.trades > 0 and result.random_entry.return_percentile >= BAR
           and bool((result.double_fees or {}).get("passed")))  # (iii): its own edge, before any comparison
    return equal_exposure.compare(v_exp, p_exp, result.oos_returns, pd.concat(parent_parts),
                                  result.random_entry.return_percentile, scaled_pct, own_edge=own)


def _placebo(result: StudyResult, parent: dict, windows: list, bt, bar: pd.Timedelta, taker: float, spread,
             exec_prices: pd.DataFrame | None, minutes: int, where: str) -> placebo.PlaceboResult:
    """R-24: the variant's one continuous window again, the parent and the placebo's draws on the fast path, and the
    calibration runs (parent, variant, the first draws) on the full engine as the window ran, never logged as a try.
    Costs as random entry charges them (the taker fee and the half spread in force each side), a stop's slippage and
    the share of equity a trip holds as the variant's own run had them."""
    import numpy as np

    from sleeve_fund.portfolio.sizing import DEFAULT_STOP_SLIPPAGE

    if result.score_window != CONTINUOUS or len(windows) != 1:
        return placebo.PlaceboResult(why_not="the placebo runs on a continuous scoring window only")
    _, window, run = windows[0]
    df, test_idx, params = window[1], window[2], window[3]
    variant = params["definition"]
    first = test_idx[0] - bar  # trade_from: the last warm-up close may open the window's first trip
    ns = df.index.as_unit("ns").asi8 if df.index.tz is not None else df.index.tz_localize("UTC").as_unit("ns").asi8
    half = spread.at_many(ns) if isinstance(spread, SpreadSeries) else np.full(len(df), float(spread or 0.0))
    fine = None
    if exec_prices is not None and len(exec_prices):
        fine = exec_prices[(exec_prices.index > df.index[0] - bar) & (exec_prices.index <= df.index[-1])]

    def trips_of(r: BacktestResult) -> list:
        return _window_trips(trades(fills_to_rows(r.fills), r.shorts, open_trip=True), first)["trips"]

    def net_of(r: BacktestResult) -> float:
        days = whole_days(daily_returns(_flat_until(r, first)), test_idx[0], bar)
        return float((1 + days).prod() - 1)

    own = [t for t in trades(fills_to_rows(run.fills), run.shorts, open_trip=True) if t["opened"] is not None]
    held = [t["cost"] / float(run.equity[run.equity.index <= t["opened"]].iloc[-1]) for t in own
            if len(run.equity[run.equity.index <= t["opened"]])]
    size = float(np.median(held)) if held else 1.0

    def full(defn: dict, blocked) -> tuple[list, float]:
        if defn is variant and not blocked:
            return trips_of(run), net_of(run)
        p = {**{k: v for k, v in params.items() if k != "definition_hash"}, "definition": defn}  # hashed on the run
        if blocked:
            p["blocked_entries"] = [int(ns[b]) for b in sorted(blocked)]
        r = bt(result.spec.name, df, p, oos=True, trade_from=first, count_from=test_idx[0])
        return trips_of(r), net_of(r)

    step = (int((fine.index[1] - fine.index[0]) / pd.Timedelta(minutes=1)) if fine is not None and len(fine) > 1
            else None)
    bars_words = f"{step}-minute execution bar" if step else "candle (bars only, no execution bars)"
    out = placebo.placebo(parent, variant, df, minutes, int(np.searchsorted(ns, first.value, side="left")),
                          taker + half, funding=run.funding_source, full=full, exec_prices=fine,
                          stop_slippage=np.maximum(half, DEFAULT_STOP_SLIPPAGE) - half, size=size)
    out.size, out.where = size, where
    out.stop_words = (f"A stop fills as the engine fills a resting stop: on the first {bars_words} to trade through "
                      f"its level, at that bar's open where it opened through, else at that bar's low (the engine's "
                      f"pessimistic fill inside the bar), worse by the larger of half the spread and "
                      f"{DEFAULT_STOP_SLIPPAGE:.2%}; the variant and every draw alike.")
    return out


def _gross_exposure(exposure: pd.Series, first, last) -> float:
    """The sum of |exposure| over the bars from first to last, inclusive: flat bars add 0."""
    if exposure is None or exposure.empty:
        return 0.0
    idx = exposure.index
    return float(exposure[(idx >= first) & (idx <= last)].abs().sum())


def _sides_words(run, params: dict) -> str:
    """R-04: the sides the research period's default run traded, in words; with no trades, the sides it may take."""
    sides = {t["side"] for t in trades(fills_to_rows(run.fills), run.shorts)}
    if not sides:
        sides = {1, -1} if (params or {}).get("allow_short") else {1}
    return {frozenset({1}): "long only", frozenset({-1}): "short only"}.get(frozenset(sides), "long and short")


def _trip_reconcile(trips: list[dict], equity: pd.Series, test_start, last, charges: list[tuple],
                    decisions: dict | None = None) -> dict:
    """RANDOM-ENTRY: a window's counted trips as the engine booked them (pnl after fees, fills and funding over
    entry notional, less any ABSORB-GAP charge), and their pnl against what the window's equity did less its carried
    trips': a trip still open at the end from the close before it opened, one carried in to the close it ended at.
    equity: the window's (flat at the starting capital up to and at test_start). decisions: the run's, by order id;
    a trip its stop closed carries stop_gap, what filling that exit at the stop's level, as the draws' are, adds to
    its return (STOP-FILL gap; Advisor ruling (A), HoE 21 Oct)."""
    start = _utc(test_start)
    charge = dict(charges)
    idx = equity.index if equity.index.tz is not None else equity.index.tz_localize("UTC")

    def at(ts, side: str) -> float:  # equity at the last close before ts ("before"), or the first at or after it
        i = int(idx.searchsorted(_utc(ts), side="left"))
        return float(equity.iloc[max(i - 1, 0)] if side == "before" else equity.iloc[min(i, len(equity) - 1)])

    base, end = float(equity.iloc[idx.searchsorted(start, side="right") - 1]), at(last, "after")
    counted, pnl, carried = [], 0.0, 0.0
    for t in trips:
        if t["closed"] is None:
            carried += end - at(t["opened"], "before") if _utc(t["opened"]) >= start else end - base
        elif t["opened"] is None or _utc(t["opened"]) < start:
            if _utc(t["closed"]) >= start:
                carried += at(t["closed"], "after") - base
        else:
            c = charge.get(_utc(t["opened"]), 0.0)
            counted.append({"opened": _utc(t["opened"]), "closed": _utc(t["closed"]), "side": int(t["side"]),
                            "ret": (t["pnl"] / t["cost"] if t["cost"] else 0.0) - c,
                            "exit_ratio": t["exit_px"] / t["entry_px"] if t["entry_px"] else 1.0,
                            **_stop_gap(t, (decisions or {}).get(str(t.get("exit_order"))))})
            pnl += t["pnl"] - c * t["cost"]
    return {"trip_returns": counted, "trip_reconcile": (pnl, end - base - carried, base)}


def _stop_gap(trip: dict, decision: dict | None) -> dict:
    """A trip's stop exit, if its stop closed it: {"stopped": True, "stop_gap": side x (the stop's level - the
    price it filled at) over the entry price, "stop_level": the level over the entry price}; else {}. The draws book
    a stop at its level less the taker's slippage (_filled_benchmarks takes that off the gap, Advisor PR E round 1)."""
    if not decision or decision.get("intent") != "stop_loss":
        return {}
    sig, side, entry = decision.get("signal") or {}, int(trip["side"]), trip["entry_px"]
    level = sig.get("trigger", sig.get("stop_px"))
    if level is None and sig.get("stop_loss") is not None:
        level = entry * (1 - side * float(sig["stop_loss"]))
    gap = side * (float(level) - trip["exit_px"]) / entry if level is not None and entry else 0.0
    return {"stopped": True, "stop_gap": gap, "stop_level": float(level) / entry if level is not None and entry else 0.0}


def _in_market_bars(exposure: pd.Series, first, last) -> int:
    """The bars from first to last, inclusive, that closed holding a position of any size or side."""
    if exposure is None or exposure.empty:
        return 0
    idx = exposure.index
    inside = exposure[(idx >= first) & (idx <= last)]
    return int((inside.abs() > 1e-9).sum())


@dataclass
class DrawFill:
    """RANDOM-ENTRY: what filled draws need beyond the prices: the definition the strategy traded (a rule-builder
    one; None keeps the close-priced comparison), the execution bars stops are matched on, each settlement's rate as
    the engine charges it (_settlement_rates; None: spot) and the taker fee the strategy's own runs paid."""

    definition: dict | None
    bar_minutes: int
    exec_prices: pd.DataFrame | None
    funding: tuple | None
    run_taker: float
    refused: str = ""  # why a rule definition's exits can't be filled as the engine fills them: NOT JUDGED
    pricers: dict = field(default_factory=dict)

    def pricer(self, prices: pd.DataFrame, cost_per_side, half_spread=0.0):
        from sleeve_fund.research.fills import Pricer

        key = id(prices)
        if key not in self.pricers:
            self.pricers[key] = Pricer(self.definition, prices, self.bar_minutes, cost_per_side, self.exec_prices,
                                       self.funding, half_spread)
        return self.pricers[key].at_cost(cost_per_side, half_spread)


def _benchmarks(prices: pd.DataFrame, folds: list[Fold], test_bars: int, taker: float, spread, shorts: bool,
                leverage: float | None = None, market: str | None = None, charged: bool = True,
                fill: DrawFill | None = None, random_side_test: bool = True):
    """The random-entry benchmark, and the random-side test when the strategy can go short, on the folds'
    counted trips. Each trip is placed on the bars it was opened and closed in, and both sides of the comparison
    are priced on those bars' closes, so the benchmark compares timing, not fills. spread: a half spread, or a
    SpreadSeries whose value in force at each bar's close is charged there, for the strategy's trips and the draws
    alike (SPREAD-PIT). leverage: the risk profile's cap, for a fold that traded the perpetual (its chosen market,
    else `market`): its trips and their draws are liquidated as the engine books one, losing the margin and fees,
    never the move past it (Independent Quant Advisor 7 Oct 00:19 (4)); and, charged, a liquidated trip of the
    strategy's pays the gap the venue absorbed past it, its draws never (ABSORB-GAP, Advisor ~02:55 UK, option A).
    random_side_test: False skips the random-side test for a caller that reads random entry only (the cost ladder's
    rungs). Not `side`: the trip loop below binds that name (CR M1)."""
    index = prices.index.tz_localize("UTC") if prices.index.tz is None else prices.index
    half = spread.at_many(index.as_unit("ns").asi8) if isinstance(spread, SpreadSeries) else spread
    cost_per_side = taker + half

    def bar(ts) -> int:
        return max(int(index.searchsorted(ts, side="right")) - 1, 0)

    if fill is not None and fill.refused:
        entry, side = _benchmarks(prices, folds, test_bars, taker, spread, shorts, leverage, market, charged,
                                  random_side_test=random_side_test)
        entry.why_not = fill.refused
        return entry, side
    if fill is not None and fill.definition is not None:
        return _filled_benchmarks(prices, folds, test_bars, taker, half, cost_per_side, shorts and random_side_test,
                                  leverage, market, charged, fill, bar)
    windows, placed = [], []
    for f in folds:
        end = bar(f.test_end)
        start = end - test_bars + 1
        windows.append((start, end))
        last = start
        charges = dict(getattr(f, "trip_charges", []) or []) if charged else {}
        sizes = dict(getattr(f, "trip_sizes", []) or [])
        for opened, closed, side in sorted(f.trips):
            entry = max(bar(opened), last)  # one position at a time on the bar grid too
            out = min(max(bar(closed), entry + 1), end)
            if out <= entry:
                continue
            perp = f.chosen.get("market", market) == PERP
            placed.append(Trade(entry, out, side, leverage=leverage if perp else None, charge=charges.get(opened, 0.0),
                                size=sizes.get(opened, 1.0)))
            last = out
    closes = prices["close"].to_numpy(dtype=float)
    bars = sum(f.window_bars for f in folds)
    in_market = sum(f.in_market_bars for f in folds) / bars if bars else None
    entry = random_entry(closes, placed, windows, cost_per_side, in_market=in_market)
    return entry, random_side(closes, placed, windows, cost_per_side) if shorts and random_side_test else None


def _filled_benchmarks(prices, folds, test_bars, taker, half, cost_per_side, shorts, leverage, market, charged,
                       fill: DrawFill, bar):
    """RANDOM-ENTRY (HoE 14:08 and 14:43 UK 21 Oct; Advisor): the strategy's side is its realised trips as the engine
    booked them (fees, fills, funding; less an ABSORB-GAP charge, charged), a same-candle trip kept, at `taker` in
    place of the fee its runs paid (a cost-ladder rung); each draw keeps a trip's side and holding period, as a cap,
    and is filled as the engine fills the definition's own exits (fills.Pricer). The trips must reconcile to the
    windows' equity within RECONCILE_PP, else the comparison is NOT JUDGED."""
    index = prices.index.tz_localize("UTC") if prices.index.tz is None else prices.index
    pricer = fill.pricer(prices, cost_per_side, half)
    windows, placed, realised, gaps, stops = [], [], [], [], []
    trips_total = equity_total = 1.0
    for f in folds:
        end = bar(f.test_end)
        start = end - test_bars  # the close before the window: a counted trip may open on it (_window_trips)
        windows.append((start, end))
        sizes = dict(getattr(f, "trip_sizes", []) or [])
        perp = f.chosen.get("market", market) == PERP
        for t in f.trip_returns:
            entry, out = _trip_bars(index, t["opened"], t["closed"], end)
            placed.append(Trade(entry, out, t["side"], leverage=leverage if perp else None,
                                size=sizes.get(t["opened"], 1.0)))
            r = t["ret"] + (fill.run_taker - taker) * (1 + t["exit_ratio"])  # a rung's fee in place of the run's
            uncharged = dict(getattr(f, "trip_charges", []) or []).get(t["opened"], 0.0) if not charged else 0.0
            realised.append(r + uncharged)
            # The STOP-FILL gap at the draws' fill: the level less the taker's slippage, not the bare level.
            slip = float(taker_slippage(half[out] if np.ndim(half) else half))
            gaps.append(t.get("stop_gap", 0.0) - t.get("stop_level", 0.0) * slip)
            stops.append(bool(t.get("stopped")))
        if f.trip_reconcile is not None:
            pnl, moved, base = f.trip_reconcile
            if base:
                trips_total *= 1 + pnl / base
                equity_total *= 1 + moved / base
    bars = sum(f.window_bars for f in folds)
    in_market = sum(f.in_market_bars for f in folds) / bars if bars else None
    realised = np.asarray(realised, dtype=float)
    entry = random_entry_filled(placed, realised, windows, pricer, (trips_total - 1, equity_total - 1),
                                in_market=in_market, gaps=np.asarray(gaps, dtype=float),
                                stops=np.asarray(stops, dtype=bool))
    side = random_side_filled(placed, realised, windows, pricer) if shorts else None
    return entry, side


def _draw_fill(spec, folds: list[Fold], bar_minutes: int, exec_prices, instrument, prices: pd.DataFrame,
               run_taker: float) -> DrawFill | None:
    """RANDOM-ENTRY: how this study's draws are filled: from the one rule definition its windows traded, or None (a
    hand-coded model: the close-priced comparison, said on the sheet)."""
    from sleeve_fund.research.fills import unmodelled

    if spec.name != "rules":
        return None
    chosen = [f.chosen for f in folds if f.trip_returns]
    defs = {json.dumps(c.get("definition"), sort_keys=True) for c in chosen}
    if len(defs) != 1 or not chosen or not chosen[0].get("definition"):
        return None
    definition = chosen[0]["definition"]
    why = unmodelled(definition, bar_minutes)
    return DrawFill(None if why else definition, bar_minutes, exec_prices,
                    _settlement_rates(chosen[0], instrument, prices), run_taker,
                    refused=f"its draws can't be filled as the engine fills its exits: {why}" if why else "")


def _trip_bars(index: pd.DatetimeIndex, opened, closed, end: int) -> tuple[int, int]:
    """A realised trip on the candle grid: entered at the last close at or before it opened, out at the first close
    at or after it closed (a stop inside a candle is out at that candle), no later than `end`; a trip opened and
    closed on one candle keeps a hold of 0, never pushed out a candle."""
    entry = max(int(index.searchsorted(opened, side="right")) - 1, 0)
    return entry, min(max(int(index.searchsorted(closed, side="left")), entry), end)


def _settlement_rates(params: dict, instrument, prices: pd.DataFrame, root=None) -> tuple | None:
    """RANDOM-ENTRY: each funding settlement over `prices` as the engine charges it, (times ns, rates, baseline):
    the venue's stored rate, else the baseline both sides pay (a missing one, or every one of a simulated perp's);
    None on spot."""
    terms = markets.terms(params, str(instrument.id.venue))
    if terms is None or not len(prices):
        return None
    idx = prices.index if prices.index.tz is not None else prices.index.tz_localize("UTC")
    times = markets.funding_times(idx[0].to_pydatetime(), idx[-1].to_pydatetime(), terms.funding_hours)
    baseline = markets.baseline_rate(terms)
    if terms.funding_venue is None:
        rates, base = [baseline] * len(times), [True] * len(times)
    else:
        pair = pair_of(instrument)
        series, cap = funding.rates(terms.funding_venue, pair, root), funding.cap_of(terms.funding_venue, pair)
        found = [funding.rate_at(series, pd.Timestamp(t), cap) for t in times]
        rates, base = [baseline if r is None else r for r in found], [r is None for r in found]
    return (np.array([pd.Timestamp(t).value for t in times], dtype=np.int64), np.array(rates, dtype=float),
            np.array(base, dtype=bool))


HOLD_COVERAGE_FLOOR = 0.5  # the share of out-of-sample days a perp hold must cover to judge (Advisor, 7 Oct 2026)


PRICE_HOLD = "price-only hold (no spot market at this venue)"
# shortcut: the base-tier spot taker fee both venues publish (0.10%), not read from a fee schedule: no spot account.
PRICE_HOLD_TAKER = 0.001
# RESEARCH-SPOT-FEES (Advisor, 21 Oct): our current entry-tier spot fee, the extra line on every spot-lowfee sheet
SPOT_ENTRY_TIER_FEES = FeeSchedule(maker=Decimal("0.004"), taker=Decimal("0.008"))


def price_hold_equity(closes: pd.Series, cost: float, capital: float) -> pd.Series:
    """BH-102 rule 3 (Advisor, 21 Oct): on a venue with no spot market, the asset's own price held, bought at the
    first close and sold at the last, each paying `cost` (the base-tier spot taker fee plus half the spread): no
    funding, nothing to liquidate. A free hold would be the harder bar. The first close shows the capital
    before the entry, as the engine's hold does, so the entry fee falls in the first day's return and a window's
    returns pay both fees."""
    eq = capital * (1 - cost) * closes.astype(float) / float(closes.iloc[0])
    eq.iloc[0] = capital
    eq.iloc[-1] = eq.iloc[-1] * (1 - cost)
    return eq


def liquidated_at(run) -> list[pd.Timestamp]:
    """When a run's position was liquidated: each fill of an order the engine decided as a liquidation."""
    return [pd.Timestamp(f["ts"]) for f in fills_to_rows(run.fills)
            if (run.decisions.get(f["order_id"]) or {}).get("intent") == "liquidation"]


def _sharpe_or_floor(returns: pd.Series) -> float:
    s = summary(returns)["sharpe"] if len(returns.dropna()) >= 2 else float("nan")
    return float("-inf") if s is None or math.isnan(s) else float(s)


def _summary_or_none(returns: pd.Series) -> dict | None:
    return summary(returns) if len(returns.dropna()) >= 2 else None


def missing_funding(params: dict, instrument, after: pd.Timestamp, until: pd.Timestamp, root=None) -> int | None:
    """How many funding settlements in (after, until] a perpetual hold on the strategy's market has no settled rate
    for: None when the market is spot, 0 for a simulated perp, which charges its terms' fixed rate as its runs do.
    The perp buy and hold is priced only over a window with none missing; a missing rate is never filled in
    (Advisor, 7 Oct 2026)."""

    from sleeve_fund import funding

    terms = markets.terms(params, str(instrument.id.venue))
    if terms is None:
        return None
    if terms.funding_venue is None:
        return 0
    series = funding.rates(terms.funding_venue, pair_of(instrument), root)
    # The settlements the engine charges (markets.settlement_times): the venue's own record times, so a 4-hour stretch
    # is checked at every 4 hours, not only at the profile's fixed hours (Advisor, 21 Oct)
    due = markets.settlement_times(_utc(after).to_pydatetime(), _utc(until).to_pydatetime(), terms.funding_hours,
                                   series)
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


def _definition_target_words(params: dict | None) -> str:
    """A rule-builder definition's own take-profit (TP-EXIT), beside the study's exits: {r} is R in price, never the
    study-wide take_profit_r's R after costs (HoE 21 Oct), so the sheet says which."""
    exits = ((params or {}).get("definition") or {}).get("exits") or {}
    found = [(side, e["take_profit"]) for side, e in (("", exits), *((f" ({s})", exits.get(s) or {})
                                                             for s in ("long", "short"))) if "take_profit" in e]
    return "".join(f", take-profit at {tp['r']:g}R (price){side}" if "r" in tp
                   else f", take-profit at the level {tp['level']}{side}" for side, tp in found)


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
