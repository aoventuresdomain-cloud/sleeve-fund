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
from dataclasses import dataclass, field

import pandas as pd
from nautilus_trader.model import CurrencyPair

from sleeve_fund.research.ledger import IdeaLedger
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
    holdout_reused: bool = False
    fee_note: str = ""
    notes: list[str] = field(default_factory=list)
    instrument: str = ""  # BASE/QUOTE, and the bar length tested: a G1 pass counts for exactly these
    bar_minutes: int = 1440
    venue: str = ""
    settings: str = ""  # the risk profile, exits and windows the study ran with, so two sheets can be told apart

    @property
    def round_trips(self) -> list[float]:
        return round_trips(self.full_period.fills)

    @property
    def trade_stats(self) -> dict:
        return trade_stats(trades(fills_to_rows(self.full_period.fills)))

    @property
    def turnover(self) -> float:
        return turnover_per_year(self.full_period.fills, self.full_period.equity)


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
    if half_spread is None:
        from sleeve_fund.venues import venue as venue_profile

        half_spread = venue_profile(str(instrument.id.venue)).assumed_half_spread
    spread_used = half_spread

    exec_minutes = bar_minutes_of(exec_prices) if exec_prices is not None and len(exec_prices) else None
    if exec_minutes is not None and (exec_minutes >= minutes or minutes % exec_minutes):
        raise ValueError(f"{exec_minutes}-minute execution bars don't divide the {minutes}-minute decision bars")

    folds_n = max(0, (len(research) - train_bars - test_bars) // test_bars + 1)
    total = 1 + len(combos) + 1 + folds_n * (len(combos) + 1) + (2 if use_holdout and holdout_days else 0)
    done = [0]

    def bt(name: str, df: pd.DataFrame, params: dict, benchmark: bool = False) -> BacktestResult:
        done[0] += 1
        if progress is not None:
            progress(min(done[0] / total, 0.99))
        guarded = risk_profile is not None and not benchmark
        if not benchmark:
            params = {**params, **exits}
        if position_cap is not None and not guarded:
            params = {**params, "position_cap_pct": position_cap}
        fine = None
        if exec_minutes is not None and not benchmark:  # nothing rests, nothing to guard
            fine = exec_prices[(exec_prices.index > df.index[0] - bar) & (exec_prices.index <= df.index[-1])]
        return run_backtest(name, df, instrument, params, starting_capital=starting_capital, bar_minutes=minutes,
                            risk_profile=risk_profile if guarded else None, exec_prices=fine,
                            exec_minutes=exec_minutes or 1, half_spread=half_spread)

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
        rows.append({**params, **m, "round_trips": len(round_trips(res.fills)), "fees": res.fees_paid})
        if params == default_params:
            full_default = res
    if full_default is None:
        full_default = bt(spec.name, research, default_params)
    sensitivity = pd.DataFrame(rows)

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
        instrument=str(instrument.id.symbol),
        bar_minutes=minutes,
        venue=str(instrument.id.venue),
        settings=(f"{risk_profile + ' risk profile' if risk_profile else 'no risk profile'} · "
                  f"exits: {_exit_words(exits) if exits else 'the signal only'} · "
                  f"walk-forward {train_days} days training, {test_days} days testing"),
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
    if exits:
        result.notes.append(
            "Exits on top of the signal: " + _exit_words(exits)
            + ". Both rest at the venue: the stop fills at its level (at market at once if the price is already "
            "through it), the target at its level, and within a bar the extreme nearer the open trades first. "
            "Paper watches both on every trade."
        )

    # 3. Holdout, only on request, using the most recent fold's choice.
    if use_holdout and holdout_days:
        result.holdout_reused = ledger.holdout_used(spec.name, dataset)
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


def _exit_words(exits: dict) -> str:
    words = []
    if "stop_loss" in exits:
        words.append(f"stop-loss {exits['stop_loss']:.1%} below entry")
    if "stop_atr" in exits:
        words.append(f"stop-loss {exits['stop_atr']:g} average true ranges (over {exits.get('atr_bars', 14)} bars) "
                     "below entry, set at each entry")
    if "stop_swing_bars" in exits:
        words.append(f"stop-loss under the lowest low of the last {exits['stop_swing_bars']} bars, set at each entry")
    if "take_profit" in exits:
        words.append(f"take-profit {exits['take_profit']:.1%} above entry")
    if "take_profit_r" in exits:
        words.append(f"take-profit making {exits['take_profit_r']:g}R after costs")
    if "risk_per_trade" in exits:
        words.append(f"each trade sized to lose {exits['risk_per_trade']:.1%} of equity at the stop")
    return ", ".join(words)
