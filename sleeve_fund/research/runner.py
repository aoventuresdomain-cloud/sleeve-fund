"""Run one strategy over one price history in a NautilusTrader backtest.

Returns a daily equity curve marked at each bar's close, plus fills and fees,
so the benchmark and the strategy are measured identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.model import AccountType, Currency, CurrencyPair, Money, OmsType, TraderId

from sleeve_fund.data import bar_type_for, decision_bar_type, to_bars
from sleeve_fund.instruments import FeeSchedule, ScheduleFeeModel, fill_model
from sleeve_fund.strategies import REGISTRY


@dataclass
class BacktestResult:
    strategy: str
    params: dict
    equity: pd.Series  # quote-currency equity at each bar close
    exposure: pd.Series  # fraction of equity in the position at each bar close
    fills: pd.DataFrame
    fees_paid: float
    starting_capital: float
    # Why each order was sent, keyed by client order id (the fills report's index).
    decisions: dict = field(default_factory=dict)


def run_backtest(
    strategy_name: str,
    prices: pd.DataFrame,
    instrument: CurrencyPair,
    params: dict | None = None,
    starting_capital: float = 10_000.0,
    log_level: str = "ERROR",
    runtime=None,
    bar_minutes: int = 1440,
    exec_prices: pd.DataFrame | None = None,
    exec_minutes: int = 1,
) -> BacktestResult:
    """prices: bars of `bar_minutes` length indexed by close time, as the history store returns them.

    exec_prices: optional shorter bars (`exec_minutes` long) over the same period. The engine is then
    fed these and builds the strategy's bars from them, so an order resting between decisions (a
    maker order, a stop) is matched against every minute, not just the next decision bar. Without
    them nothing trades between decision bars: a maker order that waits less than a bar never fills
    and goes at market, so the backtest is charged the taker fee."""
    if strategy_name not in REGISTRY:
        raise KeyError(f"unknown strategy {strategy_name!r}; known: {sorted(REGISTRY)}")
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive")
    params = dict(params or {})
    strategy_cls, config_cls = REGISTRY[strategy_name]

    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId.from_str("RESEARCH-001"),
            logging=LoggerConfig(stdout_level=getattr(LogLevel, log_level)),
        )
    )
    try:
        quote: Currency = instrument.quote_currency
        base: Currency = instrument.base_currency
        engine.add_venue(
            venue=instrument.id.venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            base_currency=None,
            starting_balances=_opening_balances(starting_capital, quote, base, runtime),
            fee_model=ScheduleFeeModel(FeeSchedule(instrument.maker_fee, instrument.taker_fee)),
            fill_model=fill_model(),
        )
        engine.add_instrument(instrument)
        if exec_prices is not None and not exec_prices.empty:
            bar_type = decision_bar_type(instrument, bar_minutes, exec_minutes)
            engine.add_data(to_bars(exec_prices, instrument, bar_type_for(instrument, exec_minutes)))
        else:
            bar_type = bar_type_for(instrument, bar_minutes)
            engine.add_data(to_bars(prices, instrument, bar_type))
        config = config_cls(
            instrument_id=instrument.id,
            bar_type=bar_type,
            assumed_taker_fee=float(instrument.taker_fee),
            **params,
        )
        strategy = strategy_cls(config).attach_runtime(runtime)
        engine.add_strategy(strategy)
        engine.run()

        fills = engine.generate_order_fills_report()
        account = engine.generate_account_report(instrument.id.venue)
        equity, exposure = _mark_to_market(account, prices, quote.code, base.code, starting_capital)
        fees = _fees_paid(fills)
        return BacktestResult(
            strategy=strategy_name,
            params=params,
            equity=equity,
            exposure=exposure,
            fills=fills,
            fees_paid=fees,
            starting_capital=starting_capital,
            decisions=dict(strategy.decisions),
        )
    finally:
        engine.dispose()


def _opening_balances(starting_capital: float, quote: Currency, base: Currency, runtime) -> list[Money]:
    """A sleeve runtime opens from its journal (as a paper restart does); research opens in cash."""
    if runtime is None:
        return [Money(starting_capital, quote)]
    book = runtime.book
    return [Money(book["cash"], quote)] + ([Money(book["qty"], base)] if book["qty"] > 0 else [])


def _mark_to_market(
    account: pd.DataFrame,
    prices: pd.DataFrame,
    quote: str,
    base: str,
    starting_capital: float,
) -> tuple[pd.Series, pd.Series]:
    """Forward-fill account balances onto bar closes and value the position at the close."""
    idx = prices.index
    if account.empty:
        cash = pd.Series(starting_capital, index=idx)
        held = pd.Series(0.0, index=idx)
    else:
        acct = account.copy()
        acct.index = pd.to_datetime(acct.index, utc=True)
        acct["total"] = acct["total"].astype(float)
        by_ccy = acct.pivot_table(index=acct.index, columns="currency", values="total", aggfunc="last")
        by_ccy = by_ccy.reindex(by_ccy.index.union(idx)).sort_index().ffill()
        cash = by_ccy.get(quote, pd.Series(0.0, index=by_ccy.index)).reindex(idx).fillna(starting_capital)
        held = by_ccy.get(base, pd.Series(0.0, index=by_ccy.index)).reindex(idx).fillna(0.0)
    position_value = held * prices["close"]
    equity = cash + position_value
    exposure = (position_value / equity).clip(lower=0.0)
    return equity.rename("equity"), exposure.rename("exposure")


def _fees_paid(fills: pd.DataFrame) -> float:
    """Sum commissions from the fills report (one quote-currency Money string per order)."""
    if fills is None or fills.empty or "commissions" not in fills:
        return 0.0
    total = 0.0
    for entry in fills["commissions"]:
        for money in entry if isinstance(entry, (list, tuple)) else [entry]:
            total += float(str(money).split()[0])
    return total
