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
from sleeve_fund.instruments import BOOK_SHARE, BarOpens, FeeSchedule, ScheduleFeeModel, fill_model, pair_of
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
    half_spread: float = 0.0
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
    half_spread: float | None = None,
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
    one. The fills report shows the prices after the spread, as paper fills on the bid or ask would.

    progress: with a risk profile, called with the simulated time at every bar, to report how far a
    long run has got.

    fees: charge this schedule instead of the market's or the instrument's (the cost ladder).

    warmup_prices: bars of `bar_minutes` before `prices`, fed to the strategy first without trading, as a paper
    strategy's warm-up from the history store is, so the window opens on settled indicators.

    first_touch_flip: resolve a rule-builder first_touch the other way when a candle is ambiguous (true in an entry,
    false in an exit), for the G1 check on the worse of the two (Advisor, 6 Oct ~22:07). first_touch_count_from: its
    report counts only the candles closing from then on, so a study reads a window's test candles alone."""
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
    if not 0 <= half_spread < 0.05:
        raise ValueError(f"half spread {half_spread} outside [0, 5%)")
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
            modules=[BarOpens(fee_model)],  # a stop filled in a bar that opened through the target: the target
            fill_model=fill_model(),
            # Within a bar, the extreme nearer the open trades first for orders resting here (a post-only
            # entry, the stops). The stop and target don't race on it: the target is judged after the bar,
            # so the stop goes first (Advisor NA-2, LongFlatStrategy._bar_target).
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
        if minutes_in:
            strategy.minute_source, strategy.range_source = minutes_from(exec_prices), ranges_from(prices)
        for node in getattr(getattr(strategy, "rules", None), "touches", ()):
            node.flip = first_touch_flip
            node.count_from = None if first_touch_count_from is None else int(pd.Timestamp(first_touch_count_from).value)
        strategy.fee_model = fee_model  # a target booked at its level (ScheduleFeeModel.booked)
        # A model defined outside the library (a test's probe) has no SPEC: its params are all it has.
        spec = getattr(importlib.import_module(strategy_cls.__module__), "SPEC", None)
        strategy.settle_bars_needed = strategy_cls.warmup_needed(
            {**(spec.default_params if spec is not None else {}), **params}, bar_minutes)
        if warmup_prices is not None and not warmup_prices.empty:
            if warmup_prices.index[-1] >= prices.index[0]:
                raise ValueError("warmup_prices must end before the backtest's first bar")
            strategy.preload = list(to_bars(_book_volume(warmup_prices, instrument), instrument,
                                            bar_type_for(instrument, bar_minutes)))
        engine.add_strategy(strategy)
        if exec_prices is not None and not exec_prices.empty:  # the engine builds the decision bars from them: store rule
            built = decision_bars(exec_prices, bar_minutes, exec_minutes)
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
        if perp:
            strategy.set_daily_atr(open_risk.daily_atr_lookup(prices))
        # Fed in slices so memory stays at one slice of engine bars however long the run: five years
        # of minutes at once is about 2.6 million bar objects. Streaming gives the same result.
        for i in range(0, len(feed), CHUNK_BARS):
            engine.add_data(to_bars(feed.iloc[i:i + CHUNK_BARS], instrument, feed_type))
            engine.run(streaming=True)
            engine.clear_data()
        engine.end()

        fills = _spread_into_prices(engine.generate_order_fills_report(), fee_model.spread_paid, fee_model.fee_paid)
        fills = _liquidations_booked(fills, strategy.liquidation_books)
        account = engine.generate_account_report(instrument.id.venue)
        if perp:
            equity, exposure = _perp_mark_to_market(fills, strategy.funding_log, prices,
                                                    _opening_cash(starting_capital, runtime),
                                                    list(strategy.insurance_log))
        else:
            equity, exposure = _mark_to_market(account, prices, quote.code, base.code, starting_capital)
        fees_paid = _fees_paid(fills)
        touched = strategy.first_touch_stats() if touches else {}
        if touches and bar_minutes > 1 and not minutes_in and any(st["missing"] for st in touched.values()):
            raise ValueError(
                f"the first_touch rule needed the 1-minute bars of {sum(st['missing'] for st in touched.values())}"
                " candles that reached both its levels, and the run had none: run it with those minutes (exec_prices, "
                "exec_minutes = 1)")
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
            first_touch=touched,
            reentries_on_exit_candle=reentries_on_exit_candle(fills, strategy.decisions, bar_minutes),
            open_risk_binds=strategy.open_risk_binds,
            open_risk_max=strategy.open_risk_max,
            paper_refusal=paper_refusal(strategy_name, params, risk_profile),
        )
    finally:
        if runtime is not None:
            # The runtime's clock is a closure over the strategy, a reference cycle the garbage
            # collector would otherwise free on whatever thread it runs on, which the engine forbids.
            runtime.now = _utcnow
        engine.dispose()


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
