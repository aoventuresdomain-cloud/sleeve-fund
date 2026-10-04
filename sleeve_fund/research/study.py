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
    fills_to_rows,
    returns_from_equity,
    round_trips,
    summary,
    trade_stats,
    trades,
    turnover_per_year,
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
) -> StudyResult:
    """exits: optional stop_loss / take_profit / risk_per_trade applied to every strategy run.
    position_cap: the largest share of capital in the position, as the paper risk profile allows. It
    applies to the benchmark too, so G1 compares the strategy with buy and hold at the same exposure."""
    if position_cap is not None and not 0 < position_cap <= 1:
        raise ValueError(f"position_cap {position_cap} outside (0, 1]")
    if len(prices) < holdout_days + train_days + test_days:
        raise ValueError(
            f"{len(prices)} bars is too short for holdout {holdout_days} + train {train_days} + test {test_days}"
        )
    research = prices.iloc[:-holdout_days] if holdout_days else prices
    combos = grid(spec.param_grid)
    default_params = default_params or spec.default_params or (combos[0] if combos else {})

    exits = {k: v for k, v in (exits or {}).items() if v is not None}

    def bt(name: str, df: pd.DataFrame, params: dict) -> BacktestResult:
        if name != "buy_and_hold":
            params = {**params, **exits}
        if position_cap is not None:
            params = {**params, "position_cap_pct": position_cap}
        return run_backtest(name, df, instrument, params, starting_capital=starting_capital)

    def log(params: dict, stage: str, sharpe: float) -> None:
        # Exit settings make it a different variant, so they count towards the idea counter.
        ledger.record(idea=spec.name, family=spec.family, params={**params, **exits}, dataset=dataset, stage=stage,
                      sharpe=sharpe)

    # Benchmark over the research period; sliced for every comparison below.
    bench = bt("buy_and_hold", research, {})
    bench_ret = returns_from_equity(bench.equity)

    # 1. Sensitivity over the full research period.
    rows = []
    full_default = None
    for params in combos:
        res = bt(spec.name, research, params)
        m = summary(returns_from_equity(res.equity))
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
    while start + train_days + test_days <= len(research):
        train = research.iloc[start : start + train_days]
        through_test = research.iloc[start : start + train_days + test_days]
        test_idx = through_test.index[train_days:]
        best, best_sharpe = None, float("-inf")
        for params in combos:
            m = summary(returns_from_equity(bt(spec.name, train, params).equity))
            log(params, "wf_train", m["sharpe"])
            if m["sharpe"] > best_sharpe:
                best, best_sharpe = params, m["sharpe"]
        # Trade the chosen params continuously through the test window so the
        # position carried in from training is realistic, then score only the test days.
        run = bt(spec.name, through_test, best)
        test_ret = returns_from_equity(run.equity).loc[test_idx[1:]]
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
        start += test_days

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
        bar_minutes=bar_minutes_of(prices),
        venue=str(instrument.id.venue),
        fee_note=(f"{instrument.id.venue}: {float(instrument.maker_fee):.2%} maker / {float(instrument.taker_fee):.2%} "
                  "taker, taker charged on every order"),
    )
    result.notes.append(
        f"Positions are capped at {position_cap:.0%} of capital, as the paper risk profile allows, and the "
        "buy-and-hold benchmark is held at the same exposure." if position_cap is not None else
        "Positions are uncapped (all of the capital), and so is the benchmark; paper trades at its risk profile's cap.")
    if exits:
        result.notes.append(
            "Exits on top of the signal: " + ", ".join(f"{k.replace('_', ' ')} {v:.1%}" for k, v in exits.items())
            + ". The stop rests at the venue and fills at its level (or the open on a gap); the target is checked on "
            "the bar's high and sold at the close, and the stop wins when one bar touches both. Paper checks both on every trade."
        )

    # 3. Holdout, only on request, using the most recent fold's choice.
    if use_holdout and holdout_days:
        result.holdout_reused = ledger.holdout_used(spec.name, dataset)
        chosen = folds[-1].chosen
        run = bt(spec.name, prices, chosen)
        b_all = bt("buy_and_hold", prices, {})
        h_idx = prices.index[-holdout_days:]
        h_ret = returns_from_equity(run.equity).reindex(h_idx).dropna()
        hb_ret = returns_from_equity(b_all.equity).reindex(h_idx).dropna()
        result.holdout = summary(h_ret)
        result.holdout_benchmark = summary(hb_ret)
        log(chosen, "holdout", result.holdout["sharpe"])
    return result
