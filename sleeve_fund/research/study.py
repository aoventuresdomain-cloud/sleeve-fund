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
from decimal import Decimal
from dataclasses import dataclass, field

import pandas as pd
from nautilus_trader.model import CurrencyPair

from sleeve_fund.instruments import FeeSchedule, pair_of
from sleeve_fund.markets import PERP
from sleeve_fund.research.ledger import IdeaLedger, opened_words
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
from sleeve_fund.strategies.base import IdeaSpec


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    chosen: dict
    train_sharpe: float
    test: dict
    benchmark_test: dict
    test_trades: int = 0  # round trips closed inside the test window: what out-of-sample was judged on
    # When the risk guard halted the run before the test window ended: the day, why, and whether
    # it was in the training stretch the run traded through first. A halted fold is flat from then on.
    halted: str = ""
    halted_before_test: bool = False  # halted in the training stretch: the whole test window sat flat


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
    # The default params over the research period at each fee of COST_LADDER: what costs the idea survives.
    cost_ladder: list[LadderRung] = field(default_factory=list)
    ladder_slippage: float = 0.0  # charged on every rung on top of the half spread, on orders that take liquidity

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
        blind = [f for f in self.folds if f.halted_before_test or (f.halted and f.test_trades == 0)]
        # Half blind is not judged either: the other half then holds the halted windows' stubs too, so the
        # verdict would rest on a window or two (review round 10, M10-1).
        if blind and (2 * len(blind) >= len(self.folds) or self.oos_trades == 0):
            halted = sum(1 for f in self.folds if f.halted)
            return (f"the risk guard halted the strategy in {halted} of {len(self.folds)} folds, leaving {len(blind)} "
                    f"of {len(self.folds)} test windows flat or without a trade, and out-of-sample closed "
                    f"{self.oos_trades} trade{'s' if self.oos_trades != 1 else ''}, so at least half of it sat flat "
                    "rather than testing the idea")
        return ""

    @property
    def round_trips(self) -> list[float]:
        return round_trips(self.full_period.fills, self.full_period.shorts)

    @property
    def oos_trades(self) -> int:
        """Round trips closed inside the walk-forward test windows: the trades out-of-sample stands on."""
        return sum(f.test_trades for f in self.folds)

    @property
    def trade_stats(self) -> dict:
        return trade_stats(trades(fills_to_rows(self.full_period.fills)))

    @property
    def turnover(self) -> float:
        return turnover_per_year(self.full_period.fills, self.full_period.equity)


# Fee per side the cost ladder tests every idea at (PM, 5 Oct 2026): free, the low-fee perp venues' maker
# and taker rates, a mid venue, and a high-fee spot venue's taker rate (the stress case).
COST_LADDER = (0.0, 0.0002, 0.0005, 0.001, 0.008)
# Slippage beyond the spread each rung also pays on orders that take liquidity (strategy sprint, PM approved
# 5 Oct 2026): 2 basis points on the deepest books (BTC, ETH), 5 on the rest.
DEEP_BOOKS = ("BTC", "ETH")


def ladder_slippage(pair: str) -> float:
    return 0.0002 if pair.split("/")[0].upper() in DEEP_BOOKS else 0.0005


@dataclass
class LadderRung:
    fee: float  # charged per side, maker and taker alike
    total_return: float
    sharpe: float
    round_trips: int
    fees_paid: float


def breakeven_fee(rungs: list[LadderRung]) -> tuple[float | None, str]:
    """The fee per side at which the idea stops making money over the period, and that in words. Found
    between the ladder's two rungs either side of zero return, by straight-line interpolation (fees scale
    with turnover, so return falls about linearly with the fee between rungs). None when it loses money even
    at no fee, or still makes money at the top rung."""
    rungs = sorted(rungs, key=lambda r: r.fee)
    if not rungs:
        return None, "not tested"
    if rungs[0].total_return <= 0:
        return None, f"loses money even at {rungs[0].fee:.2%} fees"
    for lo, hi in zip(rungs, rungs[1:]):
        if hi.total_return <= 0:
            fee = lo.fee + (hi.fee - lo.fee) * lo.total_return / (lo.total_return - hi.total_return)
            return fee, f"stops making money at about {fee:.3%} per side (between {lo.fee:.2%} and {hi.fee:.2%})"
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
    if risk_profile is not None:
        from sleeve_fund.risk import profile as risk_profile_of

        position_cap = risk_profile_of(risk_profile).max_position_pct
    if position_cap is not None and not 0 < position_cap <= 1:
        raise ValueError(f"position_cap {position_cap} outside (0, 1]")
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
    if half_spread is None:
        from sleeve_fund.venues import venue as venue_profile

        half_spread = venue_profile(str(instrument.id.venue)).assumed_half_spread
    spread_used = half_spread

    exec_minutes = bar_minutes_of(exec_prices) if exec_prices is not None and len(exec_prices) else None
    if exec_minutes is not None and (exec_minutes >= minutes or minutes % exec_minutes):
        raise ValueError(f"{exec_minutes}-minute execution bars don't divide the {minutes}-minute decision bars")

    errors: list = []
    error_count = [0]
    folds_n = max(0, (len(research) - train_bars - test_bars) // test_bars + 1)
    total = (1 + len(combos) + 1 + folds_n * (len(combos) + 1) + len(COST_LADDER)
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

    def log(params: dict, stage: str, sharpe: float) -> None:
        # Exit settings make it a different variant, so they count towards the idea counter.
        ledger.record(idea=spec.name, family=spec.family, params={**params, **exits}, dataset=dataset, stage=stage,
                      sharpe=sharpe)

    # Benchmark over the research period; sliced for every comparison below.
    bench = bt("buy_and_hold", research, {}, benchmark=True)
    bench_ret = daily_returns(bench.equity)

    # 1. Sensitivity over the full research period.
    rows = []
    full_default = None
    for params in combos:
        res = bt(spec.name, research, params)
        m = summary(daily_returns(res.equity))
        log(params, "sensitivity", m["sharpe"])
        rows.append({**params, **m, "round_trips": len(round_trips(res.fills, res.shorts)), "fees": res.fees_paid})
        if params == default_params:
            full_default = res
    if full_default is None:
        full_default = bt(spec.name, research, default_params)
    sensitivity = pd.DataFrame(rows)

    # The cost ladder: the default params over the research period at each fee. Not logged as variants, since
    # the strategy is the same; only what it pays changes.
    ladder, slip = [], ladder_slippage(pair_of(instrument))
    for fee in COST_LADDER:
        res = bt(spec.name, research, default_params, fees=FeeSchedule(maker=Decimal(str(fee)), taker=Decimal(str(fee))),
                 slippage=slip)
        eq = res.equity
        ladder.append(LadderRung(fee=fee, total_return=float(eq.iloc[-1] / starting_capital - 1) if len(eq) else 0.0,
                                 sharpe=summary(daily_returns(eq))["sharpe"],
                                 round_trips=len(round_trips(res.fills, res.shorts)), fees_paid=res.fees_paid))

    # 2. Walk-forward.
    folds: list[Fold] = []
    oos_parts, bench_parts = [], []
    start = 0
    while start + train_bars + test_bars <= len(research):
        train = research.iloc[start : start + train_bars]
        through_test = research.iloc[start : start + train_bars + test_bars]
        test_idx = through_test.index[train_bars:]
        best, best_sharpe = None, float("-inf")
        for params in combos:
            m = summary(daily_returns(bt(spec.name, train, params).equity))
            log(params, "wf_train", m["sharpe"])
            if m["sharpe"] > best_sharpe:
                best, best_sharpe = params, m["sharpe"]
        # Trade the chosen params continuously through the test window so the
        # position carried in from training is realistic, then score only the test days.
        run = bt(spec.name, through_test, best)
        test_ret = whole_days(daily_returns(run.equity), test_idx[0], bar)  # the test window's days, not the train's
        b_ret = bench_ret.reindex(test_ret.index).dropna()
        folds.append(
            Fold(
                train_start=train.index[0],
                train_end=train.index[-1],
                test_end=test_idx[-1],
                chosen=best,
                train_sharpe=best_sharpe,
                test=summary(test_ret),
                benchmark_test=summary(b_ret),
                test_trades=sum(1 for t in trades(fills_to_rows(run.fills), run.shorts)
                                if t["closed"] is not None and _utc(t["closed"]) >= _utc(test_idx[0])),
                halted=_halt_words(run.risk_events, test_idx[0], test_idx[-1]),
                halted_before_test=_halted_before(run.risk_events, test_idx[0]),
            )
        )
        oos_parts.append(test_ret)
        bench_parts.append(b_ret)
        start += test_bars

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
        fee_note=(f"{float(instrument.maker_fee):.2%} maker on post-only orders, {float(instrument.taker_fee):.2%} taker "
                  f"on every other order, plus {spread_used:.3%} of the price as half the bid-ask spread on orders "
                  "that take liquidity"),
    )
    if risk_profile is not None:
        result.notes.append(
            f"Every run trades under the {risk_profile} risk profile, as paper does: positions capped at "
            f"{position_cap:.0%} of capital, the drawdown halt (flat for the rest of the run, as paper stays halted "
            "until you resume it) and the daily-loss pause. The buy-and-hold benchmark is held at the same exposure "
            "and never halted.")
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
        result.notes.append(f"{profile.label} lists perpetuals only, so every run traded the perpetual, long only, "
                            "paying the funding the venue settled.")
    if exits:
        result.notes.append(
            "Exits on top of the signal: " + _exit_words(exits)
            + ". Both rest at the venue: the stop fills at its level (at market at once if the price is already "
            "through it), the target at its level, and within a bar the extreme nearer the open trades first. "
            "Paper watches both on every trade."
        )

    # 3. Holdout, only on request, using the most recent fold's choice. A study G1 can't judge leaves it
    # closed: opening it would spend it on no information (review round 8, M8-4).
    # Nor is it opened twice: the first look was its one use, at any bar length (review round 10, M10-1).
    opened = ledger.holdout_opened(spec.name, dataset) if use_holdout and holdout_days else None
    if use_holdout and holdout_days and result.not_judged:
        result.holdout_withheld = f"left closed, though asked for: G1 can't judge this study ({result.not_judged})"
    elif opened is not None:
        result.holdout_withheld = (f"left closed, though asked for: this model's holdout on {result.instrument} was "
                                   f"opened {opened_words(opened)}, and a second look can't be fresh")
    elif use_holdout and holdout_days:
        chosen = folds[-1].chosen
        run = bt(spec.name, prices, chosen)
        b_all = bt("buy_and_hold", prices, {}, benchmark=True)
        h_start = prices.index[-holdout_bars]
        h_ret = whole_days(daily_returns(run.equity), h_start, bar)
        hb_ret = whole_days(daily_returns(b_all.equity), h_start, bar)
        result.holdout = summary(h_ret)
        result.holdout_benchmark = summary(hb_ret)
        log(chosen, "holdout", result.holdout["sharpe"])
    return result


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
