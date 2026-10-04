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

from sleeve_fund import markets
from sleeve_fund.data import bar_type_for, decision_bar_type, to_bars
from sleeve_fund.instruments import BOOK_SHARE, FeeSchedule, ScheduleFeeModel, fill_model
from sleeve_fund.store import utcnow as _utcnow
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
    # With a risk profile: the halts and pauses the runtime made, oldest first, as paper would.
    risk_events: list = field(default_factory=list)
    # Half the bid-ask spread paid on orders that took liquidity (already in the fills' prices).
    spread_paid: float = 0.0
    half_spread: float = 0.0
    # With a risk profile: the runtime's journal (orders, fills, marks, events), for Store.save_backtest.
    journal: object = None
    # Exceptions the strategy's handlers raised, as (handler, repr): the engine would hide them.
    handler_errors: list = field(default_factory=list)
    handler_error_count: int = 0  # every one, where handler_errors keeps the first hundred
    # A perpetual's funding payments as {"ts", "amount"} (+ received, - paid), oldest first.
    funding: list = field(default_factory=list)

    @property
    def shorts(self) -> bool:
        """Traded a perpetual: a sell from flat opens a short (metrics.trades)."""
        return markets.is_perp(self.params)


CHUNK_BARS = 100_000  # bars handed to the engine at a time
# The share of each bar's traded volume the simulated venue offers this strategy's orders. The engine
# turns a bar into four prints (open, high, low, close) of a quarter of its volume each, and a resting
# order fills only against the prints that trade through its price, at most a print's size each. With
# the whole volume on offer, a post-only order joining the queue could take half a bar's volume or
# more, ahead of everyone already there, which flatters maker fills most where the maker fee saving is
# the edge (minute to hourly bars). At a fifth, a resting order takes at most 5% of a bar's volume per
# print it trades through, so about a tenth of the bar, and the rest waits for the next bar. A market
# order takes the first print's share at the price and the rest one tick worse.
# BOOK_SHARE (sleeve_fund.instruments) is the share; paper measures its maker fills against the same.


def _book_volume(feed: pd.DataFrame, instrument) -> pd.DataFrame:
    """The feed with each bar's volume cut to BOOK_SHARE, never to zero where something traded (a
    zero-volume bar makes no market at all)."""
    step = float(instrument.size_increment)
    v = feed["volume"].astype(float)
    shown = (v * BOOK_SHARE).where(v <= 0, (v * BOOK_SHARE).clip(lower=step))
    return feed.assign(volume=shown)


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
    risk_profile: str | None = None,
    half_spread: float | None = None,
    progress=None,
) -> BacktestResult:
    """prices: bars of `bar_minutes` length indexed by close time, as the history store returns them.

    exec_prices: optional shorter bars (`exec_minutes` long) over the same period. The engine is then
    fed these and builds the strategy's bars from them, so an order resting between decisions (a
    maker order, a stop) is matched against every minute, not just the next decision bar. Without
    them nothing trades between decision bars: a maker order that waits less than a bar never fills
    and goes at market, so the backtest is charged the taker fee.

    risk_profile: run with the paper runtime on a throwaway journal (SleeveRuntime.for_backtest), so
    the profile's position cap, drawdown halt and daily-loss pause act exactly as they would in paper.
    Without one (research), the strategy trades unguarded and uncapped unless position_cap_pct is set.

    half_spread: half the bid-ask spread, as a fraction of the price, paid by every order that takes
    liquidity (bars carry trade prices; a real market order buys at the ask and sells at the bid).
    None uses the venue profile's cautious assumption; sleeve_fund.spreads.resolve gives a measured
    one. The fills report shows the prices after the spread, as paper fills on the bid or ask would.

    progress: with a risk profile, called with the simulated time at every bar, to report how far a
    long run has got."""
    if strategy_name not in REGISTRY:
        raise KeyError(f"unknown strategy {strategy_name!r}; known: {sorted(REGISTRY)}")
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive")
    params = dict(params or {})
    strategy_cls, config_cls = REGISTRY[strategy_name]
    perp = markets.is_perp(params)
    fees = markets.fees_for(params, FeeSchedule(instrument.maker_fee, instrument.taker_fee))
    if half_spread is None:
        from sleeve_fund.venues import venue as venue_profile

        half_spread = markets.half_spread_for(params, venue_profile(str(instrument.id.venue)).assumed_half_spread)
    if not 0 <= half_spread < 0.05:
        raise ValueError(f"half spread {half_spread} outside [0, 5%)")
    if risk_profile is not None:
        if runtime is not None:
            raise ValueError("pass a runtime or a risk profile, not both")
        from sleeve_fund.paper.runtime import SleeveRuntime

        params.pop("position_cap_pct", None)  # the runtime's profile sets the cap
        bt = bar_type_for(instrument, bar_minutes)
        runtime = SleeveRuntime.for_backtest(
            strategy=strategy_name, instrument=str(instrument.id.symbol), bar_spec=str(bt).split(f"{instrument.id}-", 1)[1],
            starting_balance=starting_capital, risk_profile=risk_profile, params=params, bar_seconds=bar_minutes * 60)
        runtime.progress = progress

    if runtime is not None:
        # Whatever runtime is passed, this is a replay of bars: there is no trade feed between them, so
        # stops must rest at the simulated venue (as in every backtest), not wait for a trade that never
        # comes and go at the next close. Same bars, same stop, same fill, with or without a runtime.
        runtime.backtest = True

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
            # A perp trades on margin (it can go short); the venue's leverage is set above every profile's
            # cap so our own leverage and liquidation guards, not the simulated venue, decide.
            account_type=AccountType.MARGIN if perp else AccountType.CASH,
            default_leverage=markets.VENUE_LEVERAGE if perp else None,
            base_currency=None,
            starting_balances=_opening_balances(starting_capital, quote, base, runtime, perp),
            fee_model=(fee_model := ScheduleFeeModel(fees, half_spread=half_spread)),
            fill_model=fill_model(),
            # Within a bar, the extreme nearer the open trades first: a bar that opens near its low hits a
            # stop before a target, rather than always high-then-low.
            bar_adaptive_high_low_ordering=True,
        )
        engine.add_instrument(instrument)
        if exec_prices is not None and not exec_prices.empty:
            bar_type = decision_bar_type(instrument, bar_minutes, exec_minutes)
            feed, feed_type = exec_prices, bar_type_for(instrument, exec_minutes)
        else:
            bar_type = bar_type_for(instrument, bar_minutes)
            feed, feed_type = prices, bar_type
        feed = _book_volume(feed, instrument)
        config = config_cls(
            instrument_id=instrument.id,
            bar_type=bar_type,
            assumed_taker_fee=float(fees.taker),
            assumed_half_spread=half_spread,
            volume_scale=BOOK_SHARE,
            **params,
        )
        strategy = strategy_cls(config).attach_runtime(runtime)
        engine.add_strategy(strategy)
        # Fed in slices so memory stays at one slice of engine bars however long the run: five years
        # of minutes at once is about 2.6 million bar objects. Streaming gives the same result.
        for i in range(0, len(feed), CHUNK_BARS):
            engine.add_data(to_bars(feed.iloc[i:i + CHUNK_BARS], instrument, feed_type))
            engine.run(streaming=True)
            engine.clear_data()
        engine.end()

        fills = _spread_into_prices(engine.generate_order_fills_report(), fee_model.spread_paid)
        account = engine.generate_account_report(instrument.id.venue)
        if perp:
            equity, exposure = _perp_mark_to_market(fills, strategy.funding_log, prices, _opening_cash(starting_capital, runtime))
        else:
            equity, exposure = _mark_to_market(account, prices, quote.code, base.code, starting_capital)
        fees_paid = _fees_paid(fills)
        return BacktestResult(
            strategy=strategy_name,
            params=params,
            equity=equity,
            exposure=exposure,
            fills=fills,
            fees_paid=fees_paid,
            starting_capital=starting_capital,
            decisions=dict(strategy.decisions),
            risk_events=runtime.risk_events() if runtime is not None and runtime.backtest else [],
            spread_paid=sum(fee_model.spread_paid.values()),
            half_spread=half_spread,
            journal=runtime.store if risk_profile is not None else None,
            funding=[{"ts": ts, "amount": a} for ts, a in strategy.funding_log],
            handler_errors=list(strategy.handler_errors),
            handler_error_count=strategy.handler_error_count,
        )
    finally:
        if runtime is not None:
            # The runtime's clock is a closure over the strategy, a reference cycle the garbage
            # collector would otherwise free on whatever thread it runs on, which the engine forbids.
            runtime.now = _utcnow
        engine.dispose()


def _opening_balances(starting_capital: float, quote: Currency, base: Currency, runtime, perp: bool = False) -> list[Money]:
    """A sleeve runtime opens from its journal (as a paper restart does); research opens in cash. A
    margin account holds only the quote currency (its positions are not balances)."""
    if runtime is None:
        return [Money(starting_capital, quote)]
    book = runtime.book
    if perp:
        return [Money(_opening_cash(starting_capital, runtime), quote)]
    return [Money(book["cash"], quote)] + ([Money(book["qty"], base)] if book["qty"] > 0 else [])


def _opening_cash(starting_capital: float, runtime) -> float:
    return starting_capital if runtime is None else float(runtime.book["cash"])


def _perp_mark_to_market(
    fills: pd.DataFrame, funding: list[tuple], prices: pd.DataFrame, opening_cash: float,
) -> tuple[pd.Series, pd.Series]:
    """A margin account keeps realised profit only, so value a perp book spot-style from the fills
    themselves: cash moves by every fill's notional and fee (a short sale adds cash), funding adds or
    takes its payments, and equity is cash plus the signed position at the close. Exposure is the
    position's gross value over equity, signed: negative while short."""
    idx = prices.index
    flows = []  # (ts, cash change, qty change)
    if fills is not None and not fills.empty:
        for _, f in fills.iterrows():
            qty = float(f["filled_qty"])
            if qty == 0:
                continue
            side = 1.0 if str(f["side"]).endswith("BUY") else -1.0
            fee = sum(float(str(m).split()[0]) for m in
                      (f["commissions"] if isinstance(f["commissions"], (list, tuple)) else [f["commissions"]]))
            flows.append((pd.Timestamp(f["ts_last"]), -side * qty * float(f["avg_px"]) - fee, side * qty))
    for ts, amount in funding:
        flows.append((pd.Timestamp(ts), float(amount), 0.0))
    if not flows:
        return (pd.Series(opening_cash, index=idx).rename("equity"),
                pd.Series(0.0, index=idx).rename("exposure"))
    df = pd.DataFrame(flows, columns=["ts", "cash", "qty"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.groupby("ts").sum().sort_index().cumsum()
    df = df.reindex(df.index.union(idx)).sort_index().ffill().fillna(0.0).reindex(idx)
    cash = opening_cash + df["cash"]
    position_value = df["qty"] * prices["close"]
    equity = cash + position_value
    exposure = position_value / equity
    return equity.rename("equity"), exposure.rename("exposure")


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


def _spread_into_prices(fills: pd.DataFrame, spread_paid: dict[str, float]) -> pd.DataFrame:
    """The engine charges the spread with the commission (bars have no bid or ask to fill on). Move
    it into each order's average price, so the report reads as a fill on the ask or bid would, and
    leave the commissions as the venue's fee alone. Cash and equity are the same either way."""
    if fills is None or fills.empty or not spread_paid:
        return fills
    fills = fills.copy()
    for coid, spread in spread_paid.items():
        if coid not in fills.index or spread <= 0:
            continue
        qty = float(fills.at[coid, "filled_qty"])
        px = float(fills.at[coid, "avg_px"])
        buy = str(fills.at[coid, "side"]).endswith("BUY")
        fills.at[coid, "avg_px"] = str(px + spread / qty if buy else px - spread / qty)
        entry = fills.at[coid, "commissions"]
        moneys = [str(m) for m in (entry if isinstance(entry, (list, tuple)) else [entry])]
        first, ccy = moneys[0].split()
        fills.at[coid, "commissions"] = [f"{float(first) - spread:.2f} {ccy}", *moneys[1:]]
    return fills


def _fees_paid(fills: pd.DataFrame) -> float:
    """Sum commissions from the fills report (one quote-currency Money string per order)."""
    if fills is None or fills.empty or "commissions" not in fills:
        return 0.0
    total = 0.0
    for entry in fills["commissions"]:
        for money in entry if isinstance(entry, (list, tuple)) else [entry]:
            total += float(str(money).split()[0])
    return total
