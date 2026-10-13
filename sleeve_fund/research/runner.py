"""Run one strategy over one price history in a NautilusTrader backtest.

Returns a daily equity curve marked at each bar's close, plus fills and fees,
so the benchmark and the strategy are measured identically.
"""

from __future__ import annotations

import importlib
import math

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.model import AccountType, Currency, CurrencyPair, Money, OmsType, TraderId

from sleeve_fund import bars as bar_rule
from sleeve_fund import funding, markets, open_risk
from sleeve_fund.data import bar_type_for, decision_bar_type, to_bars
from sleeve_fund.instruments import BOOK_SHARE, BarOpens, ExecBars, FeeSchedule, ScheduleFeeModel, fill_model, pair_of
from sleeve_fund.spreads import SpreadSeries
from sleeve_fund.store import utcnow as _utcnow
from sleeve_fund.strategies import REGISTRY, check_perp_sizing
from sleeve_fund.strategies.definitions import uses_first_touch


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
    half_spread: float = 0.0  # the one in force when the run began
    # Which spreads the run charged, measured or assumed, and from when (sleeve_fund.spreads.SpreadSeries.report).
    spreads_used: dict = field(default_factory=dict)
    spread_text: str = ""
    # With a risk profile: the runtime's journal (orders, fills, marks, events), for Store.save_backtest.
    journal: object = None
    # Exceptions the strategy's handlers raised, as (handler, repr): the engine would hide them.
    handler_errors: list = field(default_factory=list)
    handler_error_count: int = 0  # every one, where handler_errors keeps the first hundred
    # Entries filled on a decision candle in which an exit, stop or target of the position also filled (Advisor
    # 22:30): a stop or target inside the candle may re-enter at its close, and this says how often.
    reentries_on_exit_candle: int = 0
    # A perpetual's funding payments as {"ts", "amount"} (+ received, - paid), oldest first.
    funding: list = field(default_factory=list)
    # A perpetual's settlements as (ts, the venue's rate missing, a position held), and from them N of M held
    # settlements charged the baseline and the longest stretch of held time without the rate (QA P1-O17).
    funding_marks: list = field(default_factory=list)
    funding_at_baseline: int = 0
    funding_held: int = 0
    funding_baseline_longest: pd.Timedelta = pd.Timedelta(0)
    # A simulated perp: no venue's rates at all, every settlement charged the baseline (Advisor, 6 Oct 2026)
    funding_simulated: bool = False
    # Why the venue's stored settlements don't fit the schedule charged, or "" (funding.schedule_mismatch)
    funding_schedule: str = ""
    # Shortfalls past the bankruptcy price the venue's insurance fund took, as {"ts", "amount"}.
    insurance: list = field(default_factory=list)
    # How far the fills can be trusted, when they relied on what traded first inside a bar (P1-D13): e.g.
    # "bars-only: stop fills pessimistic". Empty when no resting level ever lay inside a bar.
    labels: list = field(default_factory=list)
    # Filled orders decided before the model's indicators had settled (its warmup_needed), each flagged
    # "unsettled" in its decision; kept as the model trades them (Independent Quant Advisor, 6 Oct, 5.1).
    unsettled_fills: int = 0
    # Rule-builder first_touch rules (R2), by rule path ("long.entry", "long.entry[1]", "long.exit"): judged, true,
    # reached, same_minute, unknown, ambiguous_share and the rest (Rules.first_touch_stats).
    first_touch: dict = field(default_factory=dict)
    # A perpetual's entries the interim open-risk limit (sleeve_fund.open_risk) would have refused in paper, against
    # this strategy's own equity: a single-strategy backtest counts them and doesn't gate.
    open_risk_binds: int = 0
    # The largest open risk one of those entries would have carried, as a share of this strategy's equity.
    open_risk_max: float = 0.0
    # Why paper would refuse to start these settings, when it would (a stopless model above 1x on a perp): the run
    # still goes ahead so the risk can be measured, labelled (QA P1-S8).
    paper_refusal: str | None = None
    # In a portfolio run (P2-7): what its gate did to the opening orders, "<entry|rebalance>_<outcome>" -> count, with
    # outcome sent, trimmed, below_minimum (trimmed under the venue's minimum, so not sent) or refused (QA F213-2).
    portfolio_gate: dict = field(default_factory=dict)

    @property
    def shorts(self) -> bool:
        """Traded a perpetual: a sell from flat opens a short (metrics.trades)."""
        return markets.is_perp(self.params)


CHUNK_BARS = 100_000  # bars handed to the engine at a time
BARS_ONLY_LABEL = "bars-only: stop fills pessimistic"
LIQUIDATION_CHECK = " (liquidation check)"


def fills_label(exec_minutes: int | None, relied: set) -> list[str]:
    """The label a result carries when its fills relied on what traded first inside a bar (Advisor, P1-D13 18:36 and
    19:00): on bars only, or on execution bars longer than a minute, and only when some resting level lay inside a
    bar's range. When only the liquidation price did, it says so."""
    if not relied:
        return []
    label = BARS_ONLY_LABEL if not exec_minutes else f"execution bars {exec_minutes} min: pessimistic fills"
    return [label + ("" if "fill" in relied else LIQUIDATION_CHECK)]
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


PAPER_REFUSED = "would be refused on paper (stopless above 1x)"


def paper_refusal(strategy: str, params: dict | None, risk_profile: str | None) -> str | None:
    """PAPER_REFUSED when paper wouldn't start these settings (strategies.check_perp_stop), else None."""
    from sleeve_fund.strategies import check_perp_stop

    if risk_profile is None:
        return None
    try:
        check_perp_stop(strategy, params, risk_profile)
    except ValueError:
        return PAPER_REFUSED
    return None


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
    half_spread: float | SpreadSeries | None = None,
    progress=None,
    fees: FeeSchedule | None = None,
    warmup_prices: pd.DataFrame | None = None,
    first_touch_flip: bool = False,
    first_touch_count_from: pd.Timestamp | None = None,
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
    one. The fills report shows the prices after the spread, as paper fills on the bid or ask would. A
    SpreadSeries (sleeve_fund.spreads.series, SPREAD-PIT) charges each fill the measurement in force when its bar
    opened, never a later one, and the venue's assumption before the first; the result says which it charged.

    progress: with a risk profile, called with the simulated time at every bar, to report how far a
    long run has got.

    fees: charge this schedule instead of the market's or the instrument's (the cost ladder).

    warmup_prices: bars of `bar_minutes` before `prices`, fed to the strategy first without trading, as a paper
    strategy's warm-up from the history store is, so the window opens on settled indicators.

    first_touch_flip: resolve a rule-builder first_touch the other way when a candle is ambiguous (true in an entry,
    false in an exit), for the G1 check on the worse of the two (Advisor, 6 Oct ~22:07). first_touch_count_from: its
    report counts only the candles closing from then on, so a study reads a window's test candles alone."""
    plan = plan_run(strategy_name, prices, instrument, params, starting_capital, runtime, bar_minutes, exec_prices,
                    exec_minutes, risk_profile, half_spread, progress, fees, warmup_prices, first_touch_flip,
                    first_touch_count_from)
    engine = new_engine(log_level)
    legs = []
    try:
        legs.append(add_leg(engine, plan))
        feed_legs(engine, legs)
        return leg_result(engine, legs[0])
    finally:
        for leg in legs:
            release_leg(leg)
        engine.dispose()


@dataclass
class RunPlan:
    """One strategy's run as run_backtest works it out before any engine exists: checked, with its fees, spread
    and runtime settled. A portfolio run (research.portfolio_run) plans each strategy so and adds them all to one
    engine, each on its own venue clone, so a strategy trades there exactly as it would alone."""

    strategy_name: str
    strategy_cls: type
    config_cls: type
    prices: pd.DataFrame
    instrument: CurrencyPair
    params: dict
    starting_capital: float
    runtime: object
    risk_profile: str | None
    bar_minutes: int
    exec_prices: pd.DataFrame | None
    exec_minutes: int
    fees: FeeSchedule
    series: SpreadSeries
    half_spread: float
    perp: bool
    touches: bool
    minutes_in: bool
    fine: bool
    start_ns: int
    end_ns: int
    warmup_prices: pd.DataFrame | None
    first_touch_flip: bool
    first_touch_count_from: pd.Timestamp | None


@dataclass
class Leg:
    """A planned strategy added to an engine: the instrument as the engine lists it (the plan's own, or its venue
    clone in a portfolio run), the strategy and its venue's fee model, and the bars it is fed."""

    plan: RunPlan
    instrument: CurrencyPair
    strategy: object
    fee_model: ScheduleFeeModel
    feed: pd.DataFrame
    feed_type: object
    coarse: bool
    exec_minutes: int | None


def plan_run(strategy_name, prices, instrument, params=None, starting_capital=10_000.0, runtime=None,
             bar_minutes=1440, exec_prices=None, exec_minutes=1, risk_profile=None, half_spread=None, progress=None,
             fees=None, warmup_prices=None, first_touch_flip=False, first_touch_count_from=None) -> RunPlan:
    """run_backtest's checks and settings, before the engine (its arguments, as it documents them)."""
    if strategy_name not in REGISTRY:
        raise KeyError(f"unknown strategy {strategy_name!r}; known: {sorted(REGISTRY)}")
    check_perp_sizing(strategy_name, params)
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive")
    params = dict(params or {})
    strategy_cls, config_cls = REGISTRY[strategy_name]
    touches = bool(params.get("definition")) and uses_first_touch(params["definition"])
    minutes_in = touches and bar_minutes > 1 and exec_prices is not None and not exec_prices.empty \
        and exec_minutes == 1
    first_minute = prices.index[0] - pd.Timedelta(minutes=bar_minutes - 1)  # the first decision candle's first
    if minutes_in and (exec_prices.index[0] > first_minute or exec_prices.index[-1] < prices.index[-1]):
        # the engine decides on candles built from these minutes: one they don't reach is never judged at all
        raise ValueError(f"the first_touch rule needs the 1-minute bars of every decision candle: they run "
                         f"{exec_prices.index[0]} to {exec_prices.index[-1]}, the decision candles {prices.index[0]} "
                         f"to {prices.index[-1]}")
    perp = markets.is_perp(params)
    if fees is None:
        fees = markets.fees_for(params, FeeSchedule(instrument.maker_fee, instrument.taker_fee), str(instrument.id.venue))
    if half_spread is None:
        from sleeve_fund.venues import venue as venue_profile

        half_spread = markets.half_spread_for(params, venue_profile(str(instrument.id.venue)).assumed_half_spread,
                                              str(instrument.id.venue))
    series = half_spread if isinstance(half_spread, SpreadSeries) else SpreadSeries.constant(half_spread, "given")
    for q in [series.assumed] + [q for _, q in series.points]:
        if not 0 <= q.half_spread < 0.05:
            raise ValueError(f"half spread {q.half_spread} outside [0, 5%)")
    fine = exec_prices is not None and not exec_prices.empty
    first = exec_prices if fine else prices
    start_ns = int(first.index[0].value) - (exec_minutes if fine else bar_minutes) * 60_000_000_000 if len(first) else 0
    end_ns = int(first.index[-1].value) if len(first) else 0
    half_spread = series.at(start_ns)
    if risk_profile is not None:
        if runtime is not None:
            raise ValueError("pass a runtime or a risk profile, not both")
        from sleeve_fund.paper.runtime import SleeveRuntime

        params.pop("position_cap_pct", None)  # the runtime's profile sets the cap
        bt = bar_type_for(instrument, bar_minutes)
        runtime = SleeveRuntime.for_backtest(
            strategy=strategy_name, instrument=pair_of(instrument), bar_spec=str(bt).split(f"{instrument.id}-", 1)[1],
            starting_balance=starting_capital, risk_profile=risk_profile, params=params, bar_seconds=bar_minutes * 60)
        runtime.progress = progress

    if runtime is not None:
        # Whatever runtime is passed, this is a replay of bars: there is no trade feed between them, so
        # stops must rest at the simulated venue (as in every backtest), not wait for a trade that never
        # comes and go at the next close. Same bars, same stop, same fill, with or without a runtime.
        runtime.backtest = True
    return RunPlan(strategy_name, strategy_cls, config_cls, prices, instrument, params, starting_capital, runtime,
                   risk_profile, bar_minutes, exec_prices, exec_minutes, fees, series, half_spread, perp, touches,
                   minutes_in, fine, start_ns, end_ns, warmup_prices, first_touch_flip, first_touch_count_from)


def new_engine(log_level: str = "ERROR") -> BacktestEngine:
    return BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId.from_str("RESEARCH-001"),
            logging=LoggerConfig(stdout_level=getattr(LogLevel, log_level)),
        )
    )


def add_leg(engine: BacktestEngine, plan: RunPlan, instrument: CurrencyPair | None = None, before_add=None,
            **config) -> Leg:
    """Add the planned strategy to `engine` on `instrument` (default: the plan's own; a portfolio run passes the
    plan's venue clone), with its own venue, account, fee model and bars. before_add(strategy): called before the
    engine takes the strategy (a portfolio run joins its gate there). config: more strategy settings (an
    order_id_tag, so several strategies in one engine keep their order ids apart)."""
    instrument = instrument or plan.instrument
    params, prices, runtime = plan.params, plan.prices, plan.runtime
    exec_prices, exec_minutes, bar_minutes = plan.exec_prices, plan.exec_minutes, plan.bar_minutes
    quote: Currency = instrument.quote_currency
    base: Currency = instrument.base_currency
    fee_model = ScheduleFeeModel(plan.fees, half_spread=plan.half_spread)
    engine.add_venue(
        venue=instrument.id.venue,
        oms_type=OmsType.NETTING,
        # A perp trades on margin (it can go short); the venue's leverage is set above every profile's
        # cap so our own leverage and liquidation guards, not the simulated venue, decide.
        account_type=AccountType.MARGIN if plan.perp else AccountType.CASH,
        default_leverage=markets.VENUE_LEVERAGE if plan.perp else None,
        base_currency=None,
        starting_balances=_opening_balances(plan.starting_capital, quote, base, runtime, plan.perp),
        fee_model=fee_model,
        modules=[BarOpens(fee_model)],  # a stop filled in a bar that opened through the target: the target
        fill_model=fill_model(),
        # Within a bar, the extreme nearer the open trades first for orders resting here (a post-only
        # entry, the stops). The stop and target don't race on it: the target is judged after the bar,
        # so the stop goes first (Advisor NA-2, LongFlatStrategy._bar_target).
        bar_adaptive_high_low_ordering=True,
    )
    engine.add_instrument(instrument)
    if plan.fine:
        bar_type = decision_bar_type(instrument, bar_minutes, exec_minutes)
        feed, feed_type = exec_prices, bar_type_for(instrument, exec_minutes)
        coarse = exec_minutes > 1
    else:
        bar_type = bar_type_for(instrument, bar_minutes)
        feed, feed_type = prices, bar_type
        # Bars alone: a minute bar is the finest there is, so only longer ones are booked pessimistically.
        coarse, exec_minutes = bar_minutes > 1, None
    feed = _book_volume(feed, instrument)
    cfg = plan.config_cls(
        instrument_id=instrument.id,
        bar_type=bar_type,
        assumed_taker_fee=float(plan.fees.taker),
        assumed_half_spread=plan.half_spread,  # the value at the start: a series then gives the one in force
        volume_scale=BOOK_SHARE,
        **config,
        **params,
    )
    strategy = plan.strategy_cls(cfg).attach_runtime(runtime)
    if plan.minutes_in:
        strategy.minute_source, strategy.range_source = minutes_from(exec_prices), ranges_from(prices)
    for node in getattr(getattr(strategy, "rules", None), "touches", ()):
        node.flip = plan.first_touch_flip
        node.count_from = (None if plan.first_touch_count_from is None
                           else int(pd.Timestamp(plan.first_touch_count_from).value))
    strategy.fee_model = fee_model  # a target booked at its level (ScheduleFeeModel.booked)
    if plan.series.points:  # each bar's fills charge the measurement in force when it opened (BarOpens)
        fee_model.spread_series = strategy.spread_series = plan.series
    # A model defined outside the library (a test's probe) has no SPEC: its params are all it has.
    spec = getattr(importlib.import_module(plan.strategy_cls.__module__), "SPEC", None)
    strategy.settle_bars_needed = plan.strategy_cls.warmup_needed(
        {**(spec.default_params if spec is not None else {}), **params}, bar_minutes)
    warmup_prices = plan.warmup_prices
    if warmup_prices is not None and not warmup_prices.empty:
        if warmup_prices.index[-1] >= prices.index[0]:
            raise ValueError("warmup_prices must end before the backtest's first bar")
        strategy.preload = list(to_bars(_book_volume(warmup_prices, instrument), instrument,
                                        bar_type_for(instrument, bar_minutes)))
    if before_add is not None:
        before_add(strategy)
    engine.add_strategy(strategy)
    # Every resting stop is booked by the fee model with its slippage, and on bars too coarse to say what traded
    # first inside one, pessimistically against the bar (P1-D13).
    strategy.pessimistic = coarse
    fee_model.exit_info = strategy._exit_booking
    fee_model.now = strategy.clock.timestamp_ns
    if coarse:
        fee_model.bars = ExecBars(feed)
    if exec_prices is not None and not exec_prices.empty:  # the engine builds the decision bars from them: store rule
        built = decision_bars(exec_prices, bar_minutes, plan.exec_minutes)
        strategy.expect_bars(built.index.as_unit("ns").asi8.tolist())
        thin = built[built["degraded"]]
        strategy.mark_degraded(dict(zip(thin.index.as_unit("ns").asi8.tolist(), thin["missing"].astype(int))))
        part = built[built["missing"] > 0]
        strategy.mark_missing(dict(zip(part.index.as_unit("ns").asi8.tolist(), part["missing"].astype(int))))
    if "missing" in prices.columns:  # every bar's absent minutes, for the slower candles built from them (P1-4)
        part = prices[prices["missing"].fillna(0).astype(int) > 0]
        strategy.mark_missing(dict(zip(part.index.as_unit("ns").asi8.tolist(), part["missing"].astype(int))))
    if "degraded" in prices.columns:  # bars built with too many minutes missing: no entries on them (board 5a)
        thin = prices[prices["degraded"].astype(bool)]
        strategy.mark_degraded(dict(zip(thin.index.as_unit("ns").asi8.tolist(), thin["missing"].astype(int))))
    if strategy._portfolio is not None:
        # The portfolio gate measures a stopless spot entry too, and from the window's first day: the ATR is read
        # over the warm-up as well (Advisor 8 Oct 03:05, R5).
        strategy.set_daily_atr(open_risk.daily_atr_lookup(
            prices if warmup_prices is None or warmup_prices.empty else pd.concat([warmup_prices, prices])))
    elif plan.perp:
        strategy.set_daily_atr(open_risk.daily_atr_lookup(prices))
    return Leg(plan, instrument, strategy, fee_model, feed, feed_type, coarse, exec_minutes)


def feed_legs(engine: BacktestEngine, legs: list[Leg]) -> None:
    """Run the engine over every leg's bars. Fed in slices so memory stays at one slice of engine bars however long
    the run: five years of minutes at once is about 2.6 million bar objects. Streaming gives the same result. Each
    slice covers the same span of time for every leg, so the engine never goes back in time between slices."""
    if len(legs) == 1:
        leg = legs[0]
        for i in range(0, len(leg.feed), CHUNK_BARS):
            engine.add_data(to_bars(leg.feed.iloc[i:i + CHUNK_BARS], leg.instrument, leg.feed_type))
            engine.run(streaming=True)
            engine.clear_data()
    else:
        stamps = sorted(set().union(*(leg.feed.index for leg in legs)))
        for i in range(0, len(stamps), CHUNK_BARS):
            lo, hi = stamps[i], stamps[min(i + CHUNK_BARS, len(stamps)) - 1]
            for leg in legs:
                part = leg.feed.loc[lo:hi]
                if len(part):
                    engine.add_data(to_bars(part, leg.instrument, leg.feed_type))
            engine.run(streaming=True)
            engine.clear_data()
    engine.end()


def leg_result(engine: BacktestEngine, leg: Leg) -> BacktestResult:
    """The leg's BacktestResult, once the engine has run: what run_backtest returns for it."""
    plan, strategy, fee_model, instrument = leg.plan, leg.strategy, leg.fee_model, leg.instrument
    prices, runtime, starting_capital = plan.prices, plan.runtime, plan.starting_capital
    fills = _spread_into_prices(engine.generate_order_fills_report(), fee_model.spread_paid, fee_model.fee_paid)
    if fills is not None and not fills.empty and "instrument_id" in fills.columns:
        fills = fills[fills["instrument_id"].astype(str) == str(instrument.id)]  # this leg's own, in one engine
    fills = _liquidations_booked(fills, strategy.liquidation_books)
    account = engine.generate_account_report(instrument.id.venue)
    if plan.perp:
        equity, exposure = _perp_mark_to_market(fills, strategy.funding_log, prices,
                                                _opening_cash(starting_capital, runtime),
                                                list(strategy.insurance_log))
    else:
        equity, exposure = _mark_to_market(account, prices, instrument.quote_currency.code,
                                           instrument.base_currency.code, starting_capital)
    fees_paid = _fees_paid(fills)
    touched = strategy.first_touch_stats() if plan.touches else {}
    if (plan.touches and plan.bar_minutes > 1 and not plan.minutes_in
            and any(st["missing"] for st in touched.values())):
        raise ValueError(
            f"the first_touch rule needed the 1-minute bars of {sum(st['missing'] for st in touched.values())}"
            " candles that reached both its levels, and the run had none: run it with those minutes (exec_prices, "
            "exec_minutes = 1)")
    return BacktestResult(
        strategy=plan.strategy_name,
        params=plan.params,
        equity=equity,
        exposure=exposure,
        fills=fills,
        fees_paid=fees_paid,
        starting_capital=starting_capital,
        decisions=dict(strategy.decisions),
        risk_events=runtime.risk_events() if runtime is not None and runtime.backtest else [],
        spread_paid=sum(fee_model.spread_paid.values()),
        half_spread=plan.half_spread,
        spreads_used=plan.series.report(plan.start_ns, plan.end_ns),
        spread_text=plan.series.text(plan.start_ns, plan.end_ns),
        journal=runtime.store if plan.risk_profile is not None else None,
        funding=[_funding_row(strategy, ts, a, k) for ts, a, k in strategy.funding_log],
        funding_marks=list(strategy.funding_marks),
        funding_simulated=strategy._cfg.perp is not None and strategy._cfg.perp.funding_venue is None,
        funding_schedule=_funding_schedule(strategy, instrument, prices),
        **dict(zip(("funding_at_baseline", "funding_held", "funding_baseline_longest"),
                   funding.baseline_summary(strategy.funding_marks, _funding_interval(strategy)))),
        insurance=[{"ts": ts, "amount": a} for ts, a in strategy.insurance_log],
        handler_errors=list(strategy.handler_errors),
        unsettled_fills=sum(1 for o in (fills.index if fills is not None else ())
                            if strategy.decisions.get(o, {}).get("unsettled")),
        handler_error_count=strategy.handler_error_count,
        labels=fills_label(leg.exec_minutes, fee_model.intrabar) if leg.coarse else [],
        first_touch=touched,
        reentries_on_exit_candle=reentries_on_exit_candle(fills, strategy.decisions, plan.bar_minutes),
        open_risk_binds=strategy.open_risk_binds,
        open_risk_max=strategy.open_risk_max,
        paper_refusal=paper_refusal(plan.strategy_name, plan.params, plan.risk_profile),
        portfolio_gate=dict(strategy.portfolio_gate),
    )


def release_leg(leg: Leg) -> None:
    if leg.plan.runtime is not None:
        # The runtime's clock is a closure over the strategy, a reference cycle the garbage
        # collector would otherwise free on whatever thread it runs on, which the engine forbids.
        leg.plan.runtime.now = _utcnow
    leg.fee_model.exit_info = leg.fee_model.now = None  # closures over the strategy: the same cycle


def ranges_from(df: pd.DataFrame):
    """A first_touch range source over the decision candles indexed by close time: close ns -> (high, low), or None
    for a candle not among them. Their range is the venue's, so a minute missing from exec_prices is still in it."""
    ts = df.index.as_unit("ns").asi8
    hl = df[["high", "low"]].to_numpy(dtype=float)

    def source(until: int):
        i = np.searchsorted(ts, until)
        return (float(hl[i, 0]), float(hl[i, 1])) if i < len(ts) and ts[i] == until else None

    return source


def minutes_from(df: pd.DataFrame):
    """A first_touch minute source over 1-minute bars indexed by close time: (after ns, until ns) -> the bars closing in
    (after, until] as (close ns, open, high, low, close)."""
    ts = df.index.as_unit("ns").asi8  # UTC nanoseconds, as data.to_bars stamps the bars
    ohlc = df[["open", "high", "low", "close"]].to_numpy(dtype=float)

    def source(after: int, until: int) -> list:
        i, j = np.searchsorted(ts, after, side="right"), np.searchsorted(ts, until, side="right")
        return [(int(ts[k]), *map(float, ohlc[k])) for k in range(i, j)]

    return source


def reentries_on_exit_candle(fills: pd.DataFrame | None, decisions: dict, bar_minutes: int) -> int:
    """Entries filled on a decision candle, (close - bar, close], in which a non-entry order also filled."""
    if fills is None or fills.empty:
        return 0
    entry = np.array([decisions.get(o, {}).get("intent") == "entry" for o in fills.index], dtype=bool)
    ts = pd.DatetimeIndex(fills["ts_last"]).as_unit("ns").asi8
    exits = np.sort(ts[~entry])
    at = ts[entry]
    # an exit in (entry - bar, entry]: the count of exits at or before the entry beats those at or before its open
    return int(np.sum(np.searchsorted(exits, at, side="right") > np.searchsorted(exits, at - bar_minutes * 60_000_000_000,
                                                                                 side="right")))


def decision_bars(exec_prices: pd.DataFrame, bar_minutes: int, exec_minutes: int) -> pd.DataFrame:
    """The decision bars the engine builds from execution bars stamped at their close, by close time: how many of
    each one's minutes are missing and whether that makes it degraded (sleeve_fund.bars). A decision bar none of
    whose execution bars exist isn't listed: the engine would make it up flat from nothing (QA P1-D1)."""
    idx = exec_prices.index
    period = pd.Timedelta(minutes=bar_minutes)
    close = (idx - pd.Timedelta(1, "ns")).floor(period) + period
    # Each execution bar counts its own minutes: all of them, less any the store says it was built without (QA P1-D10).
    have = exec_minutes - (exec_prices["missing"].fillna(0).astype("int64").to_numpy()
                           if "missing" in exec_prices.columns else 0)
    present = pd.Series(have, index=close).groupby(level=0).sum().clip(lower=0, upper=bar_minutes)
    out = pd.DataFrame({"missing": (bar_minutes - present).astype("int64")})
    out["degraded"] = [bar_rule.degraded(int(m), bar_minutes) for m in out["missing"]]
    return out


def _opening_balances(starting_capital: float, quote: Currency, base: Currency, runtime, perp: bool = False) -> list[Money]:
    """A sleeve runtime opens from its journal (as a paper restart does); research opens in cash. A
    margin account holds only the quote currency (its positions are not balances)."""
    if runtime is None:
        return [Money(starting_capital, quote)]
    book = runtime.book
    if perp:
        return [Money(_opening_cash(starting_capital, runtime), quote)]
    return [Money(book["cash"], quote)] + ([Money(book["qty"], base)] if book["qty"] > 0 else [])


def _funding_row(strategy, ts, amount: float, kind: str) -> dict:
    """A BacktestResult.funding row; a settled charge paid by a snapped record keeps its audit note (QA P1-O17a-14)."""
    row = {"ts": ts, "amount": amount, "kind": kind}
    if kind == "settled" and ts in strategy.funding_notes:
        row["note"] = strategy.funding_notes[ts]
    return row


def _funding_schedule(strategy, instrument, prices: pd.DataFrame) -> str:
    """funding.schedule_mismatch over the run, for a perp charged the venue's own rates."""
    terms = strategy._cfg.perp
    if terms is None or terms.funding_venue is None or not len(prices):
        return ""
    return funding.schedule_mismatch(terms.funding_venue, pair_of(instrument), terms.funding_hours,
                                     start=prices.index[0], end=prices.index[-1]) or ""


def _funding_interval(strategy) -> pd.Timedelta:
    """The time between the perpetual's settlements, for the held time a stretch without the venue's rate spans."""
    terms = strategy._cfg.perp
    return pd.Timedelta(markets.funding_interval(terms.funding_hours)) if terms is not None else pd.Timedelta(0)


def _opening_cash(starting_capital: float, runtime) -> float:
    return starting_capital if runtime is None else float(runtime.book["cash"])


def _perp_mark_to_market(
    fills: pd.DataFrame, funding: list[tuple], prices: pd.DataFrame, opening_cash: float,
    insurance: list | tuple = (),
) -> tuple[pd.Series, pd.Series]:
    """A margin account keeps realised profit only, so value a perp book spot-style from the fills
    themselves: cash moves by every fill's notional and fee (a short sale adds cash), funding adds or
    takes its payments, and equity is cash plus the signed position at the close. Exposure is the
    position's gross value over equity, signed: negative while short.

    insurance: (time, amount) for each close past the bankruptcy price (LongFlatStrategy._cover_shortfall): the
    insurance fund takes a loss past the position's isolated margin, so the amount comes back to cash; and should
    this book's own arithmetic still leave flat cash below zero (its fill prices carry the spread, and its fees
    round apart from the venue's by fractions of a cent), it is brought back to zero."""
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
    for ts, amount, *_ in funding:
        flows.append((pd.Timestamp(ts), float(amount), 0.0))
    for ts, amount in sorted((pd.Timestamp(t), float(a)) for t, a in insurance):
        before = [(c, q) for t, c, q in flows if t <= ts]
        cash_now, qty_now = opening_cash + sum(c for c, _ in before), sum(q for _, q in before)
        if abs(qty_now) < 1e-9:  # flat: equity is cash
            # The margin-cap part as the strategy booked it; any part that only brought equity to zero, by this
            # book's own arithmetic.
            amount = max(amount if cash_now + amount >= 0 else 0.0, 0.0) or math.ceil(max(-cash_now, 0.0) * 100) / 100
            if amount:
                flows.append((ts, amount, 0.0))
    if not flows:
        return (pd.Series(opening_cash, index=idx).rename("equity"),
                pd.Series(0.0, index=idx).rename("exposure"))
    df = pd.DataFrame(flows, columns=["ts", "cash", "qty"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.groupby("ts").sum().sort_index().cumsum()
    df = df.reindex(df.index.union(idx)).sort_index().ffill().fillna(0.0).reindex(idx)
    cash = opening_cash + df["cash"]
    qty = df["qty"].where(df["qty"].abs() >= 1e-9, 0.0)  # float residue of the fills' sum is flat (a lot is 1e-8)
    position_value = qty * prices["close"]
    equity = cash + position_value
    exposure = (position_value / equity).where(position_value != 0, 0.0)
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


def _spread_into_prices(fills: pd.DataFrame, spread_paid: dict[str, float],
                        fee_paid: dict[str, float] | None = None) -> pd.DataFrame:
    """The engine charges the spread with the commission (bars have no bid or ask to fill on). Move
    it into each order's average price, so the report reads as a fill on the ask or bid would, and
    leave the commissions as the venue's fee alone. Cash and equity are the same either way.
    With fee_paid (each order's unrounded venue fee), the commission shows that fee to the cent and the
    cent the charge's rounding left goes into the price with the spread: subtracting the exact spread
    from a rounded charge showed a 0% fee as -0.01 (QA m-G7). A target booked at its level carries the
    difference from its fill in the charge too (ScheduleFeeModel.booked), so it moves into the price the same way."""
    if fills is None or fills.empty or not (spread_paid or fee_paid):
        return fills
    fills = fills.copy()
    for coid in {**(fee_paid or {}), **spread_paid}:
        spread = spread_paid.get(coid, 0.0)
        if coid not in fills.index or (spread <= 0 and coid not in (fee_paid or {})):
            continue
        qty = float(fills.at[coid, "filled_qty"])
        px = float(fills.at[coid, "avg_px"])
        buy = str(fills.at[coid, "side"]).endswith("BUY")
        entry = fills.at[coid, "commissions"]
        moneys = [str(m) for m in (entry if isinstance(entry, (list, tuple)) else [entry])]
        first, ccy = moneys[0].split()
        fee = round(fee_paid[coid], 2) if fee_paid and coid in fee_paid else round(float(first) - spread, 2)
        moved = float(first) - fee  # the spread, and whatever rounding left in the charge
        fills.at[coid, "avg_px"] = str(px + moved / qty if buy else px - moved / qty)
        fills.at[coid, "commissions"] = [f"{fee:.2f} {ccy}", *moneys[1:]]
    return fills


def _liquidations_booked(fills: pd.DataFrame, books: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """GAP-LIQ-CAP: each liquidation order as the strategy booked it (LongFlatStrategy._book_liquidation), at the
    bankruptcy price with the fee on the liquidation (trigger) price, in place of the venue's fill at the market's
    price: the same loss of exactly X as its journal, and none of the market's gap in any price, fee or trip."""
    if fills is None or fills.empty or not books:
        return fills
    fills = fills.copy()
    for coid, (px, fee) in books.items():
        if coid not in fills.index:
            continue
        entry = fills.at[coid, "commissions"]
        moneys = [str(m) for m in (entry if isinstance(entry, (list, tuple)) else [entry])]
        fills.at[coid, "avg_px"] = str(px)
        fills.at[coid, "commissions"] = [f"{fee:.2f} {moneys[0].split()[1]}", *moneys[1:]]
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
