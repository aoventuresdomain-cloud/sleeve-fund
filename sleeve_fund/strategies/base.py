"""Strategy template.

Every strategy is a plain NautilusTrader Strategy (no AI in the order path) plus
an IdeaSpec that records where it came from and what it needs. The same class
runs in backtest and in paper trading. Phase 1 is spot, so positions are long or
flat: a strategy says what share of the sleeve to hold (target_weight, 0 to 1),
or simply in or out (want_long).
"""

from __future__ import annotations

import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Any

from nautilus_trader.config import StrategyConfig
from datetime import timedelta

from nautilus_trader.core import UUID4
from nautilus_trader.model import (Bar, BarType, ClientOrderId, ContingencyType, InstrumentId, LimitOrder, OrderSide,
                                   OrderStatus, OrderType, Price, PriceType, Quantity, StopMarketOrder, TimeInForce,
                                   TriggerType)
from nautilus_trader.trading import Strategy

from sleeve_fund.data import bar_minutes
from sleeve_fund.instruments import BOOK_SHARE

# Orders the signal asks for may wait for a maker fill; protective exits (stop-loss, take-profit,
# risk halts, PM flatten) always go at market, because getting out matters more than the fee.
MAKER_INTENTS = ("entry", "exit", "rebalance")


@dataclass(frozen=True)
class IdeaSpec:
    """The plain-English idea and its contract with the research loop."""

    name: str
    family: str  # e.g. trend, momentum, vol-target; the idea counter groups by this
    idea: str  # the PM's words
    rules: str  # what the code actually does
    asset_class: str = "spot"
    data_needs: str = "daily OHLCV"
    benchmark: str = "buy_and_hold"
    default_risk_profile: str = "balanced"
    param_grid: dict[str, list] = field(default_factory=dict)
    default_params: dict = field(default_factory=dict)
    known_weaknesses: str = ""
    # One sentence with {param} placeholders, filled from a sleeve's own settings for display.
    summary: str = ""


# StrategyConfig is a native type: Python passes the same keyword arguments to its
# __new__, which reads the base fields (strategy_id etc.) from them. So __init__ must
# not forward them, and we reject anything unrecognised so a typo in a sleeve file fails.
# The most of a bar's traded volume one buy may take (see LongFlatConfig.max_participation).
MAX_PARTICIPATION = 0.25
# Paper's price watchdog: minutes without a trade or a quote before it warns, and before it treats the
# feed as dead and stops reporting, so the supervisor restarts the process and it reconnects.
STALE_PRICE_WARN_MINUTES = 5
STALE_PRICE_RESTART_MINUTES = 15

_BASE_FIELDS = {
    "strategy_id",
    "order_id_tag",
    "oms_type",
    "external_order_instrument_ids",
    "manage_contingent_orders",
    "manage_gtd_expiry",
    "manage_stop",
    "use_uuid_client_order_ids",
    "log_events",
    "log_commands",
}


class LongFlatConfig(StrategyConfig):
    def __init__(
        self,
        *,
        instrument_id: InstrumentId,
        bar_type: BarType,
        cash_buffer: float = 0.01,
        max_notional: float | None = None,
        assumed_taker_fee: float | None = None,
        assumed_half_spread: float = 0.0,
        warmup_bars: int = 0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        risk_per_trade: float | None = None,
        position_cap_pct: float | None = None,
        rebalance_band: float | None = None,
        maker_wait_minutes: int | None = None,
        max_participation: float | None = MAX_PARTICIPATION,
        volume_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        unknown = set(kwargs) - _BASE_FIELDS
        if unknown:
            raise TypeError(f"unknown strategy parameters: {sorted(unknown)}")
        super().__init__()
        if assumed_taker_fee is None:
            raise ValueError("assumed_taker_fee is required: pass the venue profile's taker fee")
        if not 0 <= assumed_taker_fee < 0.05:
            raise ValueError(f"assumed_taker_fee {assumed_taker_fee} outside [0, 0.05)")
        if not 0 <= assumed_half_spread < 0.05:
            raise ValueError(f"assumed_half_spread {assumed_half_spread} outside [0, 0.05)")
        if warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")
        if not 0 <= cash_buffer < 0.5:
            raise ValueError(f"cash_buffer {cash_buffer} outside [0, 0.5)")
        if max_notional is not None and max_notional <= 0:
            raise ValueError("max_notional must be positive")
        for label, v, hi in (("stop_loss", stop_loss, 0.5), ("take_profit", take_profit, 10.0),
                             ("risk_per_trade", risk_per_trade, 0.2)):
            if v is not None and not 0 < v <= hi:
                raise ValueError(f"{label} {v} outside (0, {hi}]; use a fraction, e.g. 0.05 for 5%")
        if position_cap_pct is not None and not 0 < position_cap_pct <= 1:
            raise ValueError(f"position_cap_pct {position_cap_pct} outside (0, 1]")
        if rebalance_band is not None and not 0 <= rebalance_band < 1:
            raise ValueError(f"rebalance_band {rebalance_band} outside [0, 1)")
        if rebalance_band is not None and (stop_loss or take_profit or risk_per_trade):
            raise ValueError("stop-loss, take-profit and risk per trade work on all-or-nothing positions; "
                             "they can't be combined with rebalancing to a target weight yet")
        if maker_wait_minutes is not None:
            if int(maker_wait_minutes) != maker_wait_minutes or maker_wait_minutes < 1:
                raise ValueError("the wait before going to market must be a whole number of minutes, at least 1")
            if maker_wait_minutes >= bar_minutes(bar_type):
                raise ValueError(f"the wait before going to market ({maker_wait_minutes:g} minutes) must be shorter than "
                                 f"one bar ({bar_minutes(bar_type)} minutes), so each order settles before the next decision")
            maker_wait_minutes = int(maker_wait_minutes)
        if max_participation is not None and not 0 < max_participation <= 1:
            raise ValueError(f"max_participation {max_participation} outside (0, 1]")
        if not 0 < volume_scale <= 1:
            raise ValueError(f"volume_scale {volume_scale} outside (0, 1]")
        if risk_per_trade is not None and stop_loss is None:
            raise ValueError("risk_per_trade needs a stop_loss (size = equity x risk / loss at the stop)")
        if take_profit is not None:
            leg = assumed_taker_fee + assumed_half_spread
            cost = leg + (1 + take_profit) * leg  # the fee and half spread in, then out on the larger value
            if take_profit <= cost:
                raise ValueError(f"Take-profit {take_profit:.2%} doesn't cover the round trip's fees and spread "
                                 f"({cost:.2%}), so every target hit would lose money")
        self.instrument_id = instrument_id
        self.bar_type = bar_type
        # Leave room for the taker fee and rounding so a full-size buy never rejects.
        self.cash_buffer = cash_buffer
        # Hard cap on the quote-currency value of any single buy (sleeve budget).
        self.max_notional = max_notional
        # Venue instruments may not carry fee rates (some venues leave them out), so sizing uses this.
        self.assumed_taker_fee = assumed_taker_fee
        # Half the bid-ask spread a market order pays, for sizing to a loss at the stop. Paper uses the
        # live quotes when it has them.
        self.assumed_half_spread = assumed_half_spread
        # Live/paper only: bars to request from the venue at start to warm indicators.
        self.warmup_bars = warmup_bars
        # Optional exits on top of the strategy's own signal, as fractions of the entry price.
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        # Optional sizing: lose at most this fraction of equity if the stop is hit.
        self.risk_per_trade = risk_per_trade
        # Backtest only: the risk profile's position cap (a share of equity), so a backtest sizes
        # exactly as paper does. Paper and live take the cap from the sleeve's runtime instead.
        self.position_cap_pct = position_cap_pct
        # None: once in, hold until the signal says out (all-or-nothing). A number: when the target
        # weight moves more than this share away from the weight last traded to, trade back to it.
        self.rebalance_band = rebalance_band
        # A buy is at most this share of what traded in the bar it decided on, in every mode, so a
        # backtest can't fill far more than the market traded and paper sizes the same way. None: no cap.
        self.max_participation = max_participation
        # Backtest only: the share of each bar's volume the simulated venue was shown (see
        # research.runner.BOOK_SHARE), so the strategy reads the bar's real volume back. 1 in paper.
        self.volume_scale = volume_scale
        # None: every order is a market order and pays the taker fee. A number: entries, signal exits
        # and rebalances first rest as a post-only limit one tick inside the last price (maker fee
        # if filled); whatever is unfilled after this many minutes is cancelled and sent at market.
        self.maker_wait_minutes = maker_wait_minutes


# Handlers whose exceptions the engine would swallow without a trace (on_bar and timers it logs).
REPORTED_HANDLERS = ("on_trade", "on_quote", "on_order_accepted", "on_order_rejected", "on_order_denied",
                     "on_order_canceled", "on_order_expired", "on_order_filled")


class LongFlatStrategy(Strategy):
    """Holds a share of the sleeve between 0% and 100%, never short. Subclasses implement
    want_long() for all-or-nothing, or target_weight() for anything in between."""

    def __init__(self, config: LongFlatConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self.instrument = None
        self._last_bar_ts = 0
        self._last_close = None
        self._bid = self._ask = None  # the venue's best quotes (paper and live, and replayed quotes)
        self.recorder = None  # sleeve_fund.paper.recorder.Recorder, when a paper run is being recorded
        self._last_tick_ns = 0
        self._mark_warned = False
        self._entry_px = None  # average entry price of the open position
        self._entry_qty = 0.0
        self._exit_lock = False  # after a stop/target exit, wait for the signal to reset before re-entering
        self._pending_exit = None  # a sell waiting for every working order to close first
        self._sent: list = []  # client order ids of orders sent, until the venue has them (see _unsent)
        self._cancel_on_accept: set[str] = set()  # orders to cancel as soon as the venue has them
        self._exec_type = None  # backtest: the shorter bars the decision bars are built from
        # Volumes of the last day's decision bars, for the participation cap on buys.
        self._volumes: deque[float] = deque(maxlen=max(1, 1440 // bar_minutes(config.bar_type)))
        self._noted: set[str] = set()  # warnings already logged; each is said once until it clears
        self._last_market_ns: int | None = None  # the latest trade or quote, for the price watchdog
        self._held_w = None  # the target weight last traded to (None: not known yet, e.g. after a restart)
        self._maker: dict[str, dict] = {}  # working post-only orders: intent, reason and signal by client order id
        self._fallback: set[str] = set()  # post-only orders this strategy cancelled for running out of time
        # Paper: volume traded through each working post-only order's price, to measure its fills by.
        self._through: dict[str, float] = {}
        self._tape_last = None  # the latest trade counted into _through (time, trade id)
        self._overfilled: set[str] = set()  # post-only orders already reported as filled past the tape
        # SleeveRuntime in paper/live (journal, PM controls, risk guard); None in backtest.
        # Attach with attach_runtime() before the strategy is added to a node or engine.
        self.runtime = None
        # Paper/live on bars built from live trades: loads warm-up bars from the venue history store
        # (the venue can't serve those bars). Attach with attach_history(); None requests them from
        # the venue's candles instead.
        self.history_loader = None
        # Every order's intent, reason and signal by client order id, in backtests too, so a
        # backtest can show why each trade happened exactly as paper and live do.
        self.decisions: dict[str, dict] = {}
        # Exceptions raised in these handlers, oldest first, as (handler, repr). See _reporting.
        self.handler_errors: list[tuple[str, str]] = []
        for name in REPORTED_HANDLERS:
            setattr(self, name, self._reporting(name, getattr(self, name)))

    def _reporting(self, name: str, handler):
        """The engine swallows an exception raised in an order or market data handler without a log
        line, so a broken fill handler would quietly stop exits, journaling or the risk guard. Catch
        it here instead: log it as an error, keep it in handler_errors, and put it in the strategy's
        events (paper). Wrapped per instance, so a subclass's own handler is covered too."""
        def run(*args, **kwargs):
            try:
                return handler(*args, **kwargs)
            except Exception as exc:
                self.handler_errors.append((name, repr(exc)))
                self.log.error(f"strategy handler {name} failed: {exc!r}\n{traceback.format_exc()}")
                if self.runtime is not None:
                    try:
                        self.runtime.store.event(self.runtime.name, "error", "handler_failed",
                                                 f"{name} failed: {exc!r}", ts=self.runtime.now())
                    except Exception:  # the journal itself failing must not hide the first error
                        pass
        run.__name__ = name
        return run

    def attach_runtime(self, runtime) -> "LongFlatStrategy":
        self.runtime = runtime
        return self

    def attach_history(self, loader) -> "LongFlatStrategy":
        """loader(instrument, bar_type, limit) -> the latest `limit` complete bars, oldest first."""
        self.history_loader = loader
        return self

    @property
    def _backtest(self) -> bool:
        """Replaying history (with or without a runtime): no live trade feed, so stops and targets rest
        at the simulated venue."""
        return self.runtime is None or self.runtime.backtest

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self._cfg.instrument_id)
        if self.instrument is None:
            self.log.error(f"instrument {self._cfg.instrument_id} not found")
            if self.runtime is not None:  # surface on the dashboard, e.g. a pair the venue doesn't list
                self.runtime.store.event(self.runtime.name, "error", "instrument_not_found",
                                         f"{self._cfg.instrument_id} is not listed on the venue")
            self.stop()
            return
        if self.recorder is not None:
            self.recorder.start(self.instrument)
        self._last_market_ns = self.clock.timestamp_ns()  # the watchdog counts from the start
        if self._cfg.warmup_bars:
            if self.history_loader is not None and str(self._cfg.bar_type).endswith("INTERNAL"):
                self._warm_from_history()
            else:
                self.request_bars(self._cfg.bar_type, limit=self._cfg.warmup_bars)
        self.subscribe_bars(self._cfg.bar_type)
        if self.runtime is not None:
            self.runtime.on_start(self._cfg.assumed_taker_fee, now=lambda: self.clock.utc_now().replace(microsecond=0))
            book = self.runtime.book
            if book["qty"] > 0 and book["entry_px"]:  # carried over from before a restart
                self._entry_px, self._entry_qty = book["entry_px"], book["qty"]
            if self.runtime.backtest:
                # A backtest marks and guards from its bars: every execution bar when it is fed shorter
                # bars than it decides on (paper does every 30 s), otherwise every decision bar.
                if self._cfg.bar_type.is_composite():
                    self._exec_type = self._cfg.bar_type.composite()
                    self.subscribe_bars(self._exec_type)
                return
            # Trades give a fresh price for marking and the risk guard between (daily) bars.
            self.subscribe_trades(self._cfg.instrument_id)
            # Quotes put the venue's bid and ask in the simulated book (each trade updates it too), so
            # a market order fills at the latest ask, bid or trade rather than only the last trade, and
            # a maker order can join the best bid or ask.
            self.subscribe_quotes(self._cfg.instrument_id)
            # Ticks are driven by market data (trades and bars, throttled) because a clock timer
            # alone did not fire in the live node; the timer stays as a backup for quiet markets.
            self.clock.set_timer("sleeve-tick", timedelta(seconds=self.runtime.tick_seconds), callback=self._on_tick)

    def attach_recorder(self, recorder) -> "LongFlatStrategy":
        self.recorder = recorder
        return self

    def on_trade(self, tick) -> None:
        if self.recorder is not None:
            self.recorder.trade(tick)
        self._last_close = tick.price.as_double()  # freshest price for marking between bars
        self._market_seen()
        if self._maker and not self._backtest:
            self._tape(tick)
        self._check_exits(self._last_close)
        self._maybe_tick()

    def on_quote(self, quote) -> None:
        if self.recorder is not None:
            self.recorder.quote(quote)
        bid, ask = quote.bid_price.as_double(), quote.ask_price.as_double()
        if not 0 < bid <= ask:
            return
        if self._bid is None:
            self.log.info(f"first quote: bid {bid} ask {ask}")
        self._bid, self._ask = bid, ask
        self._market_seen()
        if self.runtime is not None:
            self.runtime.on_quote(bid, ask, venue=str(self._cfg.instrument_id.venue))

    def _on_exec_bar(self, bar: Bar) -> None:
        """Backtest: value the book and run the risk guard on every execution bar, so a halt or a
        daily-loss pause fires within the decision bar, as paper's 30-second ticks would."""
        self._last_close = bar.close.as_double()
        if self.clock.timestamp_ns() != self._last_tick_ns:  # the decision bar at this time may have ticked
            self._on_tick()

    def _maybe_tick(self) -> None:
        if self.runtime is None:
            return
        now = self.clock.timestamp_ns()
        if now - self._last_tick_ns >= self.runtime.tick_seconds * 1_000_000_000:
            self._on_tick()

    def update_indicators(self, bar: Bar) -> None:
        """Override to feed indicators. Called once per bar, historical or live, in time order."""

    def _accept(self, bar: Bar) -> bool:
        # Indicators are fed by hand rather than registered, so warm-up bars and live
        # bars can't double-count. Anything at or before the last bar seen is ignored.
        if bar.ts_event <= self._last_bar_ts:
            return False
        self._last_bar_ts = bar.ts_event
        self._volumes.append(bar.volume.as_double() / self._cfg.volume_scale)
        self.update_indicators(bar)
        return True

    def on_historical_bars(self, bars) -> None:
        for bar in sorted(bars, key=lambda b: b.ts_event):
            self._accept(bar)
        self.log.info(f"warmed up on {len(bars)} historical bars")

    def _warm_from_history(self) -> None:
        """Feed the indicators the latest stored bars, so a model on bars built from live trades
        is ready on its first live bar instead of waiting out its longest look-back."""
        want = self._cfg.warmup_bars
        try:
            bars = self.history_loader(self.instrument, self._cfg.bar_type, want)
        except Exception as exc:  # noqa: BLE001 - a missing or stale store must not stop the sleeve
            bars, why = [], str(exc)
        else:
            why = "the history store has none"
        level = "info" if bars else "warning"
        if bars:
            self.on_historical_bars(bars)
            msg = f"Loaded {len(bars)} of {want} warm-up bars from the history store"
            # Bars between the last one loaded and the first live one are a hole the indicators skip.
            step = bar_minutes(self._cfg.bar_type) * 60_000_000_000
            missing = int((time.time_ns() - bars[-1].ts_event) // step)
            if missing > 0:
                level = "warning"
                msg += (f"; the latest closed {missing * bar_minutes(self._cfg.bar_type)} minutes before now, so "
                        f"{missing} bar{'s' if missing != 1 else ''} before the first live bar "
                        f"{'are' if missing != 1 else 'is'} missing")
        else:
            msg = f"No warm-up bars loaded ({why}); the model waits for {want} live bars"
        self.log.info(msg)
        if self.runtime is not None:
            self.runtime.store.event(self.runtime.name, level, "warmup", msg)

    def want_long(self, bar: Bar) -> bool | None:
        """True = be long, False = be flat, None = not enough data yet (do nothing)."""
        raise NotImplementedError

    @classmethod
    def warmup_needed(cls, params: dict, bar_minutes: int) -> int:
        """Bars of history to load so every indicator is ready on the first live bar. The default
        is twice the longest whole-number setting; strategies that know better override it."""
        longest = max((v for v in params.values() if isinstance(v, int) and not isinstance(v, bool)), default=0)
        return 2 * longest

    def target_weight(self, bar: Bar) -> float | None:
        """Share of the sleeve to hold from this bar's close, 0 to 1; None = not enough data yet.
        The default maps want_long() to all (1) or nothing (0)."""
        target = self.want_long(bar)
        return None if target is None else (1.0 if target else 0.0)

    def explain(self, bar: Bar, target: bool) -> tuple[str, dict]:
        """Why want_long() just said `target`: one plain-English sentence and the indicator values
        behind it. Called straight after want_long() on the same bar, so it sees the same state.
        It is journaled with the order, so the reason is the one the strategy acted on."""
        return ("Signal to be long" if target else "Signal to be flat"), {}

    def on_bar(self, bar: Bar) -> None:
        if self._exec_type is not None and bar.bar_type == self._exec_type:
            self._on_exec_bar(bar)
            return
        if not self._accept(bar):
            return
        self.log.info(f"bar {bar}")
        self._last_close = bar.close.as_double()
        self._maybe_tick()
        if self._pending_exit is not None:
            return
        if self._check_exits(self._last_close):
            return
        raw = self.target_weight(bar)
        if raw is None:
            return
        # The risk profile's position cap is a ceiling on the weight, the same in every mode.
        w = min(max(float(raw), 0.0), 1.0, self._cap_pct())
        if w == 0:
            self._exit_lock = False
        if self._busy() or self._maker_working():
            return  # the last decision is still being carried out
        is_long = self._is_long()
        close = bar.close.as_double()
        if w > 0 and not is_long:
            if self._exit_lock:
                return
            if self.runtime is not None and not self.runtime.can_open():
                return
            reason, values = self.explain(bar, True)
            extra = {"target_weight": round(float(raw), 6)} if raw < 1 else {}
            # The cap is its own limit in _buy_all, so the journal says which one set the size.
            self._buy_all(bar, reason, {**values, **extra, "close": close}, weight=min(max(float(raw), 0.0), 1.0))
            self._held_w = w
        elif w == 0 and is_long:
            reason, values = self.explain(bar, False)
            self._sell_all("exit", reason, {**values, "close": close})
            self._held_w = 0.0
        elif w > 0 and is_long and self._cfg.rebalance_band is not None:
            if self._held_w is None:  # e.g. after a restart: start from what is actually held
                equity, _, qty, _ = self._mark()
                self._held_w = qty * close / equity if equity > 0 else w
            if abs(w - self._held_w) > self._cfg.rebalance_band * self._held_w:
                reason, values = self.explain(bar, True)
                if self._rebalance(bar, w, reason, {**values, "target_weight": round(float(raw), 6), "close": close}):
                    self._held_w = w

    def _cap_pct(self) -> float:
        if self.runtime is not None:
            return float(self.runtime.profile.max_position_pct)
        return float(self._cfg.position_cap_pct) if self._cfg.position_cap_pct is not None else 1.0

    def _volume_cap(self, bar: Bar) -> Decimal | None:
        """The most a buy may be worth on this bar: max_participation of what traded in an average bar
        over the last day (the bar itself for daily bars), so one quiet minute doesn't shrink an order
        a deep book would fill. Exits are not capped, since getting out matters more; entries capped
        this way keep them in proportion."""
        if self._cfg.max_participation is None or not self._volumes:
            return None
        avg = sum(self._volumes) / len(self._volumes)
        if avg <= 0:
            # Nothing traded all day: a gap filled with flat bars, or a feed without volume. There is
            # nothing to take a share of, so the cap stands aside rather than block every entry unseen.
            self._note("no_volume", "No traded volume over the last day, so the volume cap is off until there is")
            return None
        self._noted.discard("no_volume")
        return Decimal(str(avg * self._cfg.max_participation)) * bar.close.as_decimal()

    def _note(self, kind: str, msg: str, level: str = "warning") -> None:
        """Log a warning and put it in the strategy's events, once until it clears (_noted.discard)."""
        if kind in self._noted:
            return
        self._noted.add(kind)
        self.log.warning(msg)
        if self.runtime is not None:
            self.runtime.store.event(self.runtime.name, level, kind, msg, ts=self.runtime.now())

    def _rebalance(self, bar: Bar, w: float, reason: str, values: dict) -> bool:
        """Trade part of the position so it is worth `w` of the sleeve's equity at this close."""
        equity, _, qty, _ = self._mark()
        price = bar.close.as_decimal()
        diff = Decimal(str(equity * w)) - Decimal(str(qty)) * price
        step = self.instrument.size_increment.as_decimal()
        signal = {**values, "from_weight": round(self._held_w or 0.0, 6), "to_weight": round(w, 6)}
        if diff > 0:
            account = self._account()
            quote = self._codes("quote")
            bal = next((b for c, b in account.balances().items() if str(c.code) in quote), None) if account else None
            if bal is None:
                return False
            fee = Decimal(str(self._cfg.assumed_taker_fee))
            budget = min(diff, bal.free.as_decimal() * (Decimal(1) - fee - Decimal(str(self._cfg.cash_buffer))))
            if self._cfg.max_notional is not None:
                budget = min(budget, Decimal(str(self._cfg.max_notional)))
            capped = (cap := self._volume_cap(bar)) is not None and cap < budget
            if capped:
                budget = cap
            side, size = OrderSide.BUY, (budget / price).quantize(step, rounding=ROUND_DOWN)
            if capped and 0 < size and size >= self._min_qty():
                signal["sized_by"] = "share of the bar's volume"
        else:
            side = OrderSide.SELL
            size = min((-diff / price).quantize(step, rounding=ROUND_DOWN), self._position_qty(free=True))
        if size <= 0 or size < self._min_qty():
            if side == OrderSide.BUY and capped:
                self._note("buy_skipped", f"Buy skipped: {self._cfg.max_participation:.0%} of the bar's average "
                           f"volume is worth {cap:,.2f}, below the smallest order the venue takes")
            return False
        self._noted.discard("buy_skipped")
        self._submit(side, size, "rebalance", reason, signal)
        return True

    def _check_exits(self, price: float) -> bool:
        """Stop-loss / take-profit against the average entry. True if an exit was sent.

        Paper and live call this on every trade, so both levels are watched tick by tick. A
        backtest only sees whole bars, so both exits rest at the venue instead (_rest_exits)."""
        cfg = self._cfg
        if self._entry_px is None or not (cfg.stop_loss or cfg.take_profit) or price <= 0:
            return False
        if self._busy():
            return False
        if self._backtest:
            return False  # both exits rest at the venue (_rest_exits) and fill at their levels
        move = price / self._entry_px - 1
        hit = ("stop_loss" if cfg.stop_loss and move <= -cfg.stop_loss
               else "take_profit" if cfg.take_profit and move >= cfg.take_profit else None)
        if hit is None:
            return False
        self.log.info(f"{hit} at {price} ({move:+.2%} from entry {self._entry_px})")
        if self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "info", hit, f"exit at {price:,.4f}, {move:+.2%} from entry",
                                     ts=self.runtime.now())
        level = cfg.stop_loss if hit == "stop_loss" else cfg.take_profit
        reason = (f"{'Stop-loss' if hit == 'stop_loss' else 'Take-profit'}: price {price:,.6g} is {move:+.2%} from "
                  f"the {self._entry_px:,.6g} entry, past the {level:.1%} {'stop' if hit == 'stop_loss' else 'target'}")
        values = {"entry_px": self._entry_px, "move": move, hit: level}
        self._exit_lock = True
        self._entry_px = None  # don't fire again while the sell is in flight
        self._sell_all(hit, reason, values)
        return True

    def _buy_all(self, bar: Bar, reason: str = "Signal to be long", values: dict | None = None,
                 weight: float = 1.0) -> None:
        account = self._account()
        if account is None:
            self.log.warning("no account yet; skipping buy")
            return
        quote = self._codes("quote")
        bal = next((b for c, b in account.balances().items() if str(c.code) in quote), None)
        if bal is None:
            self.log.warning(f"no {quote} balance; skipping buy")
            return
        free = bal.free
        # Every limit on the size, so the journal can say which one set it.
        # Free cash leaves room for the taker fee and rounding so a full-size buy never rejects;
        # the other limits are sizes in their own right, and their fee comes out of the spare cash.
        room = Decimal(1) - Decimal(str(self._cfg.cash_buffer)) - Decimal(str(self._cfg.assumed_taker_fee))
        limits = {"free cash": free.as_decimal() * room}
        if weight < 1:
            limits["target weight"] = Decimal(str((self._mark()[0] or float(free.as_decimal())) * weight))
        if self._cfg.max_notional is not None:
            limits["largest order cap"] = Decimal(str(self._cfg.max_notional))
        if self.runtime is not None:
            limits[f"{self.runtime.profile.name} risk profile cap"] = Decimal(
                str(self.runtime.position_budget(self._mark()[0])))
        elif self._cfg.position_cap_pct is not None:
            equity = self._mark()[0] or float(free.as_decimal())
            limits["risk profile cap"] = Decimal(str(equity * self._cfg.position_cap_pct))
        if self._cfg.risk_per_trade:
            equity = self._mark()[0] or float(free.as_decimal())
            limits["risk per trade"] = Decimal(str(equity * self._cfg.risk_per_trade / self._loss_at_stop()))
        if (cap := self._volume_cap(bar)) is not None:
            limits["share of the bar's volume"] = cap
        size_by = min(limits, key=limits.get)
        budget = limits[size_by]
        step = self.instrument.size_increment.as_decimal()
        qty = (budget / bar.close.as_decimal()).quantize(step, rounding=ROUND_DOWN)
        min_qty = self.instrument.min_quantity.as_decimal() if self.instrument.min_quantity else step
        if qty <= 0 or qty < min_qty:
            self._note("buy_skipped", f"Buy skipped: the {size_by} limit ({float(budget):,.2f}) buys {qty}, below "
                       f"the smallest order the venue takes ({min_qty})")
            return
        self._noted.discard("buy_skipped")
        signal = {**(values or {}), "close": bar.close.as_double(), "sized_by": size_by,
                  "budget": round(float(budget), 2)}
        if self._cfg.stop_loss:
            # What this position loses if the stop is hit, costs included: one R, for the trade's R multiple.
            loss = self._loss_at_stop()
            signal["risk_amount"] = round(float(qty * bar.close.as_decimal()) * loss, 2)
            if self._cfg.take_profit:  # what the target makes, after the same costs, in R
                tp, cost = self._cfg.take_profit, self._round_trip_cost()
                signal["planned_r"] = round((tp - cost - (1 + tp) * cost) / loss, 2)
        self._submit(OrderSide.BUY, qty, "entry", reason, signal)

    def _loss_at_stop(self) -> float:
        """The share of a position's cost lost if its stop is hit: the stop distance, plus the taker fee
        and half the spread to buy, plus the same on what is left to sell. A 6% stop at 0.8% taker and a
        0.05% half spread loses 7.65%, not 6%."""
        cost, stop = self._round_trip_cost(), self._cfg.stop_loss
        return stop + cost + (1 - stop) * cost

    def _round_trip_cost(self) -> float:
        """Each leg's cost as a share of its notional: the taker fee and half the spread."""
        return self._cfg.assumed_taker_fee + self._half_spread()

    def _half_spread(self) -> float:
        if self._bid is not None and self._ask is not None and self._bid > 0 and self._ask >= self._bid:
            return (self._ask - self._bid) / (self._ask + self._bid)
        return self._cfg.assumed_half_spread

    def _submit(self, side, qty: Decimal, intent: str, reason: str, signal: dict, market: bool = False) -> None:
        """Send an order, journaling it with its reason first so the record exists before the venue sees
        it. With maker_wait_minutes set, signal-driven orders rest as post-only limits first."""
        quantity = Quantity.from_decimal_dp(qty, self.instrument.size_precision)
        wait = self._cfg.maker_wait_minutes
        last, tick = self._price(), self.instrument.price_increment.as_double()
        signal = {k: (round(v, 8) if isinstance(v, float) else v) for k, v in signal.items()}
        signal.setdefault("price", last)
        maker = bool(wait and not market and intent in MAKER_INTENTS and last > 2 * tick)
        if maker:
            # Join the best bid (to buy) or ask (to sell), so the order adds liquidity rather than taking
            # it. Without quotes (a backtest on bars), one tick inside the last trade.
            if self._bid is not None and self._ask is not None:
                raw = self._bid if side == OrderSide.BUY else self._ask
            else:
                raw = last - tick if side == OrderSide.BUY else last + tick
            px = Price(raw, self.instrument.price_precision)
            order = self.order_factory.limit(instrument_id=self._cfg.instrument_id, order_side=side, quantity=quantity,
                                             price=px, time_in_force=TimeInForce.GTC, post_only=True)
            signal.update(order_type="maker", limit_px=px.as_double(), maker_wait_minutes=wait)
        else:
            order = self.order_factory.market(instrument_id=self._cfg.instrument_id, order_side=side,
                                              quantity=quantity, time_in_force=TimeInForce.GTC)
            if wait:
                signal["order_type"] = "market"
        coid = str(order.client_order_id)
        self._sent.append(order.client_order_id)
        self.decisions[coid] = {"intent": intent, "reason": reason, "signal": signal}
        if self.runtime is not None:
            self.runtime.on_order(order_id=coid, side="BUY" if side == OrderSide.BUY else "SELL",
                                  qty=float(qty), intent=intent, reason=reason, signal=signal,
                                  order_type="POST-ONLY LIMIT" if maker else "MARKET")
        self.submit_order(order)
        if maker:
            self._maker[coid] = {"intent": intent, "reason": reason, "signal": signal}
            self.clock.set_time_alert(f"maker-{coid}", self.clock.utc_now() + timedelta(minutes=wait),
                                      callback=self._maker_timeout)

    def _maker_timeout(self, event) -> None:
        """The post-only order has waited long enough: cancel what is left; the cancel's confirmation
        sends the rest at market (on_order_canceled), so nothing is sold or bought twice."""
        coid = event.name.removeprefix("maker-")
        order = self.cache.order(ClientOrderId(coid))
        if coid in self._maker and order is not None and not order.is_closed and self._pending_exit is None:
            self._fallback.add(coid)
            self.cancel_order(order.client_order_id)

    def _maker_working(self) -> bool:
        """True while a post-only order is still resting. Forgets any the venue has closed without
        this strategy hearing (say a cancel that crossed with a fill), so one lost event can't stop
        the strategy deciding for good."""
        for coid in list(self._maker):
            order = self.cache.order(ClientOrderId(coid))
            if order is None or order.is_closed:
                self._maker.pop(coid)
                self._cancel_alert(coid)
        return bool(self._maker)

    def _cancel_alert(self, coid: str) -> None:
        name = f"maker-{coid}"
        if name in self.clock.timer_names():
            self.clock.cancel_timer(name)

    def _finish_at_market(self, coid: str, info: dict, why: str) -> None:
        """Send the unfilled rest of a post-only order at market, sized to what the account can do now."""
        order = self.cache.order(ClientOrderId(coid))
        if order is None:
            return
        step = self.instrument.size_increment.as_decimal()
        left = order.leaves_qty.as_decimal()
        if order.side == OrderSide.BUY:
            account, price = self._account(), Decimal(str(self._price()))
            bal = next((b for c, b in account.balances().items() if str(c.code) in self._codes("quote")),
                       None) if account else None
            room = Decimal(1) - Decimal(str(self._cfg.cash_buffer)) - Decimal(str(self._cfg.assumed_taker_fee))
            # The price may have moved since the order was sized; never spend more than the cash allows.
            affordable = bal.free.as_decimal() * room / price if bal is not None and price > 0 else Decimal(0)
            qty = min(left, affordable).quantize(step, rounding=ROUND_DOWN)
        else:
            qty = min(left, self._position_qty(free=True)).quantize(step, rounding=ROUND_DOWN)
        if qty <= 0 or qty < self._min_qty():
            self.log.info(f"post-only order {coid} {why}; the rest ({qty}) is below the minimum order size")
            if (self._backtest and order.side == OrderSide.BUY and order.filled_qty.as_double() > 0
                    and self._pending_exit is None):
                self._rest_exits()
            return
        reason = f"{info['reason']}. The post-only order {why}, so the rest went at market"
        signal = {**{k: v for k, v in info["signal"].items() if k != "price"}, "maker_order": coid}
        self._submit(order.side, qty, info["intent"], reason, signal, market=True)

    def _tape(self, tick) -> None:
        """Paper: count a trade towards each working post-only order whose price it traded through."""
        key = (tick.ts_event, str(tick.trade_id))
        if key == self._tape_last:
            return
        self._tape_last = key
        px, size = tick.price.as_double(), tick.size.as_double()
        for coid in self._maker:
            order = self.cache.order(ClientOrderId(coid))
            if order is None:
                continue
            limit = order.price.as_double()
            if (px < limit) if order.is_buy else (px > limit):
                self._through[coid] = self._through.get(coid, 0.0) + size

    def _check_maker_fill(self, coid: str, order) -> None:
        """Paper's simulated venue fills a post-only order in full once the price trades through it,
        however little traded; a backtest gives it at most BOOK_SHARE of the volume that trades
        through. Say so, once per order, when a paper fill goes past what the backtest would allow, so
        a maker strategy's paper results on a thin market aren't read as the backtest's."""
        last = self.cache.trade(self._cfg.instrument_id)
        if last is not None:
            self._tape(last)  # the venue may have filled on a trade this strategy hasn't heard yet
        through = self._through.get(coid, 0.0)
        filled, allowed = order.filled_qty.as_double(), BOOK_SHARE * through
        step = self.instrument.size_increment.as_double()
        if filled <= allowed + step or coid in self._overfilled:
            return
        self._overfilled.add(coid)
        msg = (f"Post-only order {coid} has filled {filled:.8g} while {through:.8g} traded through its price; a "
               f"backtest would fill at most {allowed:.8g} ({BOOK_SHARE:.0%}). Paper's maker fills run ahead of "
               "the backtest's where little trades")
        self.log.warning(msg)
        if self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "warning", "maker_fill_above_tape", msg, ts=self.runtime.now())

    def _forget_tape(self, coid: str) -> None:
        self._through.pop(coid, None)
        self._overfilled.discard(coid)

    def _min_qty(self) -> Decimal:
        step = self.instrument.size_increment.as_decimal()
        return self.instrument.min_quantity.as_decimal() if self.instrument.min_quantity else step

    def _position_qty(self, free: bool = False) -> Decimal:
        """Quantity held, read from the account rather than positions, so a book restored from the
        journal after a restart (a balance with no position object) is still recognised."""
        account = self._account()
        if account is None:
            return Decimal(0)
        bals = account.balances() if free else account.balances_total()
        codes = self._codes("base")
        total = Decimal(0)
        for cur, b in bals.items():
            if str(cur.code) in codes:
                total += (b.free if free else b).as_decimal()
        return total

    def _unsent(self) -> list:
        """Orders still on their way to the venue (INITIALIZED): a backtest's order sent from inside a bar,
        tick or fill handler reaches the venue only after that handler returns, so until then neither
        the account nor orders_open nor orders_inflight shows it. A risk halt's flatten and the bar's own
        exit once both sold the whole position that way; so did a halt on the bar an entry filled, while
        the entry's stop and target were still unsent, and the target then sold it a second time."""
        unsent = []
        for coid in self._sent:
            order = self.cache.order(coid)
            if order is not None and order.status == OrderStatus.INITIALIZED:
                unsent.append(order)
        self._sent = [o.client_order_id for o in unsent]
        return unsent

    def _working(self) -> list:
        """Every order of this strategy not yet closed: resting at the venue, in flight, or not yet sent."""
        iid = self._cfg.instrument_id
        seen, working = set(), []
        for order in (*self.cache.orders_open(instrument_id=iid, strategy_id=self.strategy_id),
                      *self.cache.orders_inflight(instrument_id=iid, strategy_id=self.strategy_id), *self._unsent()):
            if order.client_order_id not in seen and not order.is_closed:
                seen.add(order.client_order_id)
                working.append(order)
        return working

    def _busy(self) -> bool:
        return bool(self.cache.orders_inflight(strategy_id=self.strategy_id)) or bool(self._unsent())

    def _is_long(self) -> bool:
        return self._position_qty() >= self._min_qty()

    def _sell_all(self, intent: str = "exit", reason: str = "Signal to be flat", values: dict | None = None) -> None:
        working = self._working()
        if working:
            # A working order (the backtest's stop and target, a post-only order, or any order not yet at
            # the venue) holds part of the position or cash, or could still trade it. Cancel each one, and
            # sell what is left once every one has closed (_resume_exit). A market order just fills.
            self._pending_exit = (intent, reason, values)
            for order in working:
                if order.order_type == OrderType.MARKET:
                    continue
                if order.status in (OrderStatus.INITIALIZED, OrderStatus.SUBMITTED):
                    self._cancel_on_accept.add(str(order.client_order_id))  # the venue can't cancel it yet
                elif order.status != OrderStatus.PENDING_CANCEL:
                    self.cancel_order(order.client_order_id)
            return
        step = self.instrument.size_increment.as_decimal()
        qty = self._position_qty(free=True).quantize(step, rounding=ROUND_DOWN)
        if qty <= 0 or qty < self._min_qty():
            return
        self._submit(OrderSide.SELL, qty, intent, reason, dict(values or {}))

    # --- sleeve runtime hooks (paper/live only) --------------------------------

    def _price(self) -> float:
        px = self.cache.price(self._cfg.instrument_id, PriceType.LAST)
        if px is not None:
            return px.as_double()
        if self._last_close:
            return self._last_close
        # A quiet instrument may not trade for minutes after start; its quotes still value the book.
        if self._bid and self._ask:
            return (self._bid + self._ask) / 2
        return 0.0

    def _codes(self, side: str) -> set[str]:
        """Currency codes for one side of the pair: the venue's (e.g. ZUSD) and the plain one (USD)."""
        base, quote = str(self._cfg.instrument_id.symbol).split("/")
        cur = self.instrument.base_currency if side == "base" else self.instrument.quote_currency
        return {str(cur.code), base if side == "base" else quote}

    def _account(self):
        account = self.portfolio.account(self._cfg.instrument_id.venue)
        if account is None:  # live sandbox: ask the cache directly
            account = self.cache.account_for_venue(self._cfg.instrument_id.venue)
        return account

    def _mark(self) -> tuple[float, float, float, float]:
        """(equity, cash, position qty, price) in the quote currency."""
        price = self._price()
        account = self._account()
        if account is None or self.instrument is None:
            return 0.0, 0.0, 0.0, price
        # Match balances by currency code: the live venue's instrument currencies are not always
        # the same objects as the sandbox account's, so balance_total(currency) can miss.
        totals = {str(cur.code): m.as_double() for cur, m in account.balances_total().items()}
        cash = sum(totals.get(c, 0.0) for c in self._codes("quote"))
        qty = sum(totals.get(c, 0.0) for c in self._codes("base"))
        return cash + qty * price, cash, qty, price

    def _market_seen(self) -> None:
        self._last_market_ns = self.clock.timestamp_ns()
        if self._noted & {"stale_price", "feed_dead"}:
            self._noted -= {"stale_price", "feed_dead"}
            if self.runtime is not None:
                self.runtime.store.event(self.runtime.name, "info", "price_feed_back", "Market data is arriving again",
                                         ts=self.runtime.now())

    def _feed_dead(self) -> bool:
        """Paper's price watchdog. The tick timer keeps the heartbeat going on its own, so a feed that
        has gone silent would look healthy while every mark and guard check used a frozen price. Past
        STALE_PRICE_WARN_MINUTES it says so; past STALE_PRICE_RESTART_MINUTES it stops reporting, and
        the supervisor restarts the process, which reconnects to the venue."""
        if self._backtest or self._last_market_ns is None:
            return False
        minutes = (self.clock.timestamp_ns() - self._last_market_ns) / 60e9
        if minutes >= STALE_PRICE_WARN_MINUTES:
            self._note("stale_price", f"No trade or quote from the venue for {minutes:.0f} minutes; marks and the "
                       f"risk guard are using the last price, {self._price():,.6g}")
        if minutes >= STALE_PRICE_RESTART_MINUTES:
            self._note("feed_dead", f"No market data for {minutes:.0f} minutes: the price feed looks dead, so this "
                       "process stops reporting and the supervisor restarts it to reconnect", level="error")
            return True
        return False

    def _on_tick(self, _event=None) -> None:
        self._last_tick_ns = self.clock.timestamp_ns()
        if self._feed_dead():
            return  # no heartbeat, so the supervisor restarts the process
        try:
            equity, cash, qty, price = self._mark()
            if price <= 0 or equity <= 0:
                # Still alive, just can't value the book yet: heartbeat, and say why once.
                self.runtime.store.heartbeat(self.runtime.name)
                if not self._mark_warned:
                    self._mark_warned = True
                    acct = self._account()
                    why = (f"account={'missing' if acct is None else 'ok'} price={price} cash={cash} qty={qty} "
                           f"balances={({str(c.code): m.as_double() for c, m in acct.balances_total().items()} if acct else {})} "
                           f"quote={self.instrument.quote_currency.code if self.instrument else '?'}")
                    self.log.warning(f"cannot mark sleeve yet: {why}")
                    self.runtime.store.event(self.runtime.name, "warning", "mark_unavailable", why)
                return
            if self.runtime.reconcile_due() and not self.cache.orders_inflight(strategy_id=self.strategy_id):
                tol = self.instrument.size_increment.as_double() / 2
                if not self.runtime.reconcile(cash=cash, qty=qty, qty_tolerance=tol):
                    self.cancel_all_orders(self._cfg.instrument_id)  # halted: no trading, no flattening
            if self.runtime.tick(equity=equity, cash=cash, qty=qty, price=price) == "flatten":
                self.cancel_all_orders(self._cfg.instrument_id)
                intent, reason = self.runtime.flatten_why or ("pm_flatten", "Flattened")
                self._sell_all(intent, reason, {"equity": equity, "peak": self.runtime.peak})
        except Exception as exc:  # never let bookkeeping kill the sleeve silently
            self.log.error(f"sleeve tick failed: {exc!r}")
            self.runtime.store.event(self.runtime.name, "error", "tick_failed", repr(exc))

    # Order lifecycle into the journal; fills are journaled in on_order_filled.
    def _order_status(self, event, status: str) -> None:
        if self.runtime is not None:
            self.runtime.on_order_status(str(event.client_order_id), status, str(getattr(event, "reason", "") or ""))

    def on_order_accepted(self, event) -> None:
        self._order_status(event, "accepted")
        coid = str(event.client_order_id)
        if coid in self._cancel_on_accept:  # a flatten was waiting for the venue to have this order
            self._cancel_on_accept.discard(coid)
            order = self.cache.order(event.client_order_id)
            if order is not None and order.is_open and order.status != OrderStatus.PENDING_CANCEL:
                self.cancel_order(order.client_order_id)

    def _resume_exit(self, coid: str) -> None:
        """An order closed: if a sell was waiting for every working order to close, send it now."""
        self._cancel_on_accept.discard(coid)
        if self._pending_exit is not None and not self._working():
            intent, reason, values = self._pending_exit
            self._pending_exit = None
            self._sell_all(intent, reason, values)

    def on_order_rejected(self, event) -> None:
        self._order_status(event, "rejected")
        coid = str(event.client_order_id)
        order = self.cache.order(event.client_order_id)
        if (order is not None and order.order_type == OrderType.STOP_MARKET and self._backtest
                and self._pending_exit is None):
            self._stop_rejected(coid, str(getattr(event, "reason", "") or ""))
        info = self._maker.pop(coid, None)
        if info is not None:  # e.g. the price moved and a post-only order would have taken liquidity
            self._cancel_alert(coid)
            if self._pending_exit is None:
                self._finish_at_market(coid, info, f"was rejected ({getattr(event, 'reason', '') or 'no reason given'})")
        self._resume_exit(coid)

    def _stop_rejected(self, coid: str, why: str) -> None:
        """Backtests: the venue refused the resting stop, most often because the price was already
        through it when the entry filled (a gap). Its linked target goes with it, so nothing would
        protect the position until the next decision. Sell at market instead, as paper does on the
        first trade past the stop."""
        if not self._is_long():
            return
        signal = self.decisions[coid]["signal"]
        price = self._price()
        trigger = f"{signal.get('trigger', 0):,.6g}"
        reason = (f"Stop-loss: the price was already through the {trigger} stop when the entry filled, so the venue "
                  "refused the resting stop and the position was sold at market" if "in the market" in why else
                  f"Stop-loss: the venue refused the resting stop at {trigger} ({why or 'no reason given'}), so the "
                  "position was sold at market")
        self.log.info(reason)
        if self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "warning", "stop_rejected", reason, ts=self.runtime.now())
        self._exit_lock = True
        self._sell_all("stop_loss", reason, {**signal, "price": price})

    def on_order_denied(self, event) -> None:
        self._order_status(event, "denied")
        self._resume_exit(str(event.client_order_id))

    def on_order_canceled(self, event) -> None:
        self._order_status(event, "canceled")
        coid = str(event.client_order_id)
        info = self._maker.pop(coid, None)
        self._cancel_alert(coid)
        self._forget_tape(coid)
        timed_out = coid in self._fallback
        self._fallback.discard(coid)
        # Only an order this strategy cancelled for time goes on at market: a cancel from a halt, a
        # stop-loss or a shutdown means the decision no longer stands.
        if info is not None and timed_out and self._pending_exit is None:
            wait = self._cfg.maker_wait_minutes
            self._finish_at_market(coid, info, f"was not filled within {wait} minute{'s' if wait != 1 else ''}")
        self._resume_exit(coid)

    def on_order_expired(self, event) -> None:
        self._order_status(event, "expired")
        self._resume_exit(str(event.client_order_id))

    def on_order_filled(self, event) -> None:
        coid = str(event.client_order_id)
        order = self.cache.order(event.client_order_id)
        done = order is None or order.is_closed
        if coid in self._maker and order is not None and not self._backtest:
            self._check_maker_fill(coid, order)
        if coid in self._maker and done:
            self._maker.pop(coid)
            self._cancel_alert(coid)
            self._forget_tape(coid)
        qty, px = event.last_qty.as_double(), event.last_px.as_double()
        if event.is_buy:
            cost = (self._entry_px or 0.0) * self._entry_qty + qty * px
            self._entry_qty += qty
            self._entry_px = cost / self._entry_qty
        else:
            self._entry_qty = max(self._entry_qty - qty, 0.0)
            if self._entry_qty <= 1e-12:
                self._entry_px, self._entry_qty = None, 0.0
        if self._backtest:
            if event.is_buy and done and self._pending_exit is None:
                self._rest_exits()
            elif self.decisions.get(str(event.client_order_id), {}).get("intent") in ("stop_loss", "take_profit"):
                self._exit_lock = True  # as in paper: no re-entry until the signal has gone flat
        if self.runtime is not None:
            fee = event.commission.as_double() if event.commission is not None else 0.0
            self.runtime.on_fill(
                side="BUY" if event.is_buy else "SELL",
                qty=event.last_qty.as_double(),
                price=event.last_px.as_double(),
                fee=fee,
                order_id=str(event.client_order_id),
                trade_id=str(event.trade_id),
            )
        if done:
            self._resume_exit(coid)

    def _rest_exits(self) -> None:
        """Backtests: after an entry fills, rest the exits at the venue as real orders would sit there.
        (Paper and live watch every trade instead.) The stop is a sell stop at its level and the target
        a sell limit at its level, linked one-updates-the-other: a fill on either shrinks the other by as
        much, so together they never sell more than is held. A flatten cancels both first (_sell_all). Within a bar the
        extreme nearer the open trades first. A stop fills at its level, or at the open when the price
        gaps through it; a target fills at its level, never better, never worse."""
        cfg = self._cfg
        if not (cfg.stop_loss or cfg.take_profit) or self._entry_px is None:
            return
        qty = self._position_qty(free=True).quantize(self.instrument.size_increment.as_decimal(), rounding=ROUND_DOWN)
        if qty < self._min_qty():
            return
        quantity = Quantity.from_decimal_dp(qty, self.instrument.size_precision)
        ids = {k: self.order_factory.generate_client_order_id() for k in ("stop_loss", "take_profit")
               if getattr(cfg, k)}
        both = len(ids) == 2

        def link(kind: str) -> dict:
            other = "take_profit" if kind == "stop_loss" else "stop_loss"
            return {"contingency_type": ContingencyType.OUO, "linked_order_ids": [ids[other]]} if both else {}

        orders = []
        now = self.clock.timestamp_ns()
        if cfg.stop_loss:
            level = self._entry_px * (1 - cfg.stop_loss)
            orders.append((StopMarketOrder(
                self.trader_id, self.strategy_id, cfg.instrument_id, ids["stop_loss"], OrderSide.SELL, quantity,
                Price(level, self.instrument.price_precision), TriggerType.DEFAULT, TimeInForce.GTC, False, False,
                UUID4(), now, **link("stop_loss")), "stop_loss", "STOP",
                f"Stop-loss: resting sell at {level:,.6g}, {cfg.stop_loss:.1%} below the {self._entry_px:,.6g} entry; "
                "fills at that level, or the open if the price gaps through",
                {"entry_px": round(self._entry_px, 8), "stop_loss": cfg.stop_loss, "trigger": round(level, 8)}))
        if cfg.take_profit:
            level = self._entry_px * (1 + cfg.take_profit)
            orders.append((LimitOrder(
                self.trader_id, self.strategy_id, cfg.instrument_id, ids["take_profit"], OrderSide.SELL, quantity,
                Price(level, self.instrument.price_precision), TimeInForce.GTC, False, False, False,
                UUID4(), now, **link("take_profit")), "take_profit", "LIMIT",
                f"Take-profit: resting sell at {level:,.6g}, {cfg.take_profit:.1%} above the {self._entry_px:,.6g} "
                "entry; fills at that level",
                {"entry_px": round(self._entry_px, 8), "take_profit": cfg.take_profit, "limit_px": round(level, 8)}))
        for order, intent, kind, reason, signal in orders:
            self.decisions[str(order.client_order_id)] = {"intent": intent, "reason": reason, "signal": signal}
            if self.runtime is not None:
                self.runtime.on_order(order_id=str(order.client_order_id), side="SELL", qty=float(qty), intent=intent,
                                      reason=reason, signal=signal, order_type=kind)
        for order, *_ in orders:
            self._sent.append(order.client_order_id)
            self.submit_order(order)

    def on_stop(self) -> None:
        for coid in list(self._maker):
            self._cancel_alert(coid)
        self.cancel_all_orders(self._cfg.instrument_id)
        if self.runtime is not None:
            self.clock.cancel_timer("sleeve-tick") if "sleeve-tick" in self.clock.timer_names() else None
            self.runtime.on_stop()
        if self.recorder is not None:
            self.recorder.close()
