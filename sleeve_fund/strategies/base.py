"""Strategy template.

Every strategy is a plain NautilusTrader Strategy (no AI in the order path) plus
an IdeaSpec that records where it came from and what it needs. The same class
runs in backtest and in paper trading. Phase 1 is spot, so positions are long or
flat: a strategy says what share of the sleeve to hold (target_weight, 0 to 1),
or simply in or out (want_long).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Any

from nautilus_trader.config import StrategyConfig
from datetime import timedelta

from nautilus_trader.model import Bar, BarType, InstrumentId, OrderSide, Price, PriceType, Quantity, TimeInForce
from nautilus_trader.trading import Strategy


@dataclass(frozen=True)
class IdeaSpec:
    """The plain-English idea and its contract with the research loop."""

    name: str
    family: str  # e.g. trend, momentum, vol-target; the idea counter groups by this
    idea: str  # the PM's words
    rules: str  # what the code actually does
    asset_class: str = "crypto_spot"
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
        warmup_bars: int = 0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        risk_per_trade: float | None = None,
        position_cap_pct: float | None = None,
        rebalance_band: float | None = None,
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
        if risk_per_trade is not None and stop_loss is None:
            raise ValueError("risk_per_trade needs a stop_loss (size = equity x risk / stop distance)")
        self.instrument_id = instrument_id
        self.bar_type = bar_type
        # Leave room for the taker fee and rounding so a full-size buy never rejects.
        self.cash_buffer = cash_buffer
        # Hard cap on the quote-currency value of any single buy (sleeve budget).
        self.max_notional = max_notional
        # Venue instruments may not carry fee rates (some venues leave them out), so sizing uses this.
        self.assumed_taker_fee = assumed_taker_fee
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


class LongFlatStrategy(Strategy):
    """Holds a share of the sleeve between 0% and 100%, never short. Subclasses implement
    want_long() for all-or-nothing, or target_weight() for anything in between."""

    def __init__(self, config: LongFlatConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self.instrument = None
        self._last_bar_ts = 0
        self._last_close = None
        self._last_tick_ns = 0
        self._mark_warned = False
        self._entry_px = None  # average entry price of the open position
        self._entry_qty = 0.0
        self._exit_lock = False  # after a stop/target exit, wait for the signal to reset before re-entering
        self._pending_exit = None  # backtest: a sell waiting for the resting stop's cancel to confirm
        self._held_w = None  # the target weight last traded to (None: not known yet, e.g. after a restart)
        # SleeveRuntime in paper/live (journal, PM controls, risk guard); None in backtest.
        # Attach with attach_runtime() before the strategy is added to a node or engine.
        self.runtime = None
        # Every order's intent, reason and signal by client order id, in backtests too, so a
        # backtest can show why each trade happened exactly as paper and live do.
        self.decisions: dict[str, dict] = {}

    def attach_runtime(self, runtime) -> "LongFlatStrategy":
        self.runtime = runtime
        return self

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self._cfg.instrument_id)
        if self.instrument is None:
            self.log.error(f"instrument {self._cfg.instrument_id} not found")
            if self.runtime is not None:  # surface on the dashboard, e.g. a pair the venue doesn't list
                self.runtime.store.event(self.runtime.name, "error", "instrument_not_found",
                                         f"{self._cfg.instrument_id} is not listed on the venue")
            self.stop()
            return
        if self._cfg.warmup_bars:
            self.request_bars(self._cfg.bar_type, limit=self._cfg.warmup_bars)
        self.subscribe_bars(self._cfg.bar_type)
        if self.runtime is not None:
            self.runtime.on_start(self._cfg.assumed_taker_fee, now=lambda: self.clock.utc_now().replace(microsecond=0))
            book = self.runtime.book
            if book["qty"] > 0 and book["entry_px"]:  # carried over from before a restart
                self._entry_px, self._entry_qty = book["entry_px"], book["qty"]
            # Trades give a fresh price for marking and the risk guard between (daily) bars.
            self.subscribe_trades(self._cfg.instrument_id)
            # Ticks are driven by market data (trades and bars, throttled) because a clock timer
            # alone did not fire in the live node; the timer stays as a backup for quiet markets.
            self.clock.set_timer("sleeve-tick", timedelta(seconds=self.runtime.tick_seconds), callback=self._on_tick)

    def on_trade(self, tick) -> None:
        self._last_close = tick.price.as_double()  # freshest price for marking between bars
        self._check_exits(self._last_close)
        self._maybe_tick()

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
        self.update_indicators(bar)
        return True

    def on_historical_bars(self, bars) -> None:
        for bar in sorted(bars, key=lambda b: b.ts_event):
            self._accept(bar)
        self.log.info(f"warmed up on {len(bars)} historical bars")

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
        if not self._accept(bar):
            return
        self.log.info(f"bar {bar}")
        self._last_close = bar.close.as_double()
        self._maybe_tick()
        if self._pending_exit is not None:
            return
        if self._check_exits(self._last_close, high=bar.high.as_double()):
            return
        raw = self.target_weight(bar)
        if raw is None:
            return
        # The risk profile's position cap is a ceiling on the weight, the same in every mode.
        w = min(max(float(raw), 0.0), 1.0, self._cap_pct())
        if w == 0:
            self._exit_lock = False
        if self.cache.orders_inflight(strategy_id=self.strategy_id):
            return
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
            side, size = OrderSide.BUY, (budget / price).quantize(step, rounding=ROUND_DOWN)
        else:
            side = OrderSide.SELL
            size = min((-diff / price).quantize(step, rounding=ROUND_DOWN), self._position_qty(free=True))
        if size <= 0 or size < self._min_qty():
            return False
        self._submit(side, size, "rebalance", reason, signal)
        return True

    def _check_exits(self, price: float, high: float | None = None) -> bool:
        """Stop-loss / take-profit against the average entry. True if an exit was sent.

        Paper and live call this on every trade, so both levels are watched tick by tick. A
        backtest only sees whole bars: its stop rests at the venue (_rest_stop), which fills at
        the level, or at the open if the bar gaps through it, and the target is checked on the
        bar's high but sold at the close. When one bar touches both, the stop wins, and a target
        exit is never priced better than the close."""
        cfg = self._cfg
        if self._entry_px is None or not (cfg.stop_loss or cfg.take_profit) or price <= 0:
            return False
        if self.cache.orders_inflight(strategy_id=self.strategy_id):
            return False
        move = price / self._entry_px - 1
        if self.runtime is None:
            peak = max(high or price, price) / self._entry_px - 1
            hit = "take_profit" if cfg.take_profit and peak >= cfg.take_profit else None
        else:
            hit = ("stop_loss" if cfg.stop_loss and move <= -cfg.stop_loss
                   else "take_profit" if cfg.take_profit and move >= cfg.take_profit else None)
        if hit is None:
            return False
        self.log.info(f"{hit} at {price} ({move:+.2%} from entry {self._entry_px})")
        if self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "info", hit, f"exit at {price:,.4f}, {move:+.2%} from entry")
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
            limits["sleeve cap"] = Decimal(str(self._cfg.max_notional))
        if self.runtime is not None:
            limits[f"{self.runtime.profile.name} risk profile cap"] = Decimal(
                str(self.runtime.position_budget(self._mark()[0])))
        elif self._cfg.position_cap_pct is not None:
            equity = self._mark()[0] or float(free.as_decimal())
            limits["risk profile cap"] = Decimal(str(equity * self._cfg.position_cap_pct))
        if self._cfg.risk_per_trade:
            equity = self._mark()[0] or float(free.as_decimal())
            limits["risk per trade"] = Decimal(str(equity * self._cfg.risk_per_trade / self._cfg.stop_loss))
        size_by = min(limits, key=limits.get)
        budget = limits[size_by]
        step = self.instrument.size_increment.as_decimal()
        qty = (budget / bar.close.as_decimal()).quantize(step, rounding=ROUND_DOWN)
        min_qty = self.instrument.min_quantity.as_decimal() if self.instrument.min_quantity else step
        if qty <= 0 or qty < min_qty:
            self.log.warning(f"buy size {qty} below minimum {min_qty}; skipping")
            return
        signal = {**(values or {}), "close": bar.close.as_double(), "sized_by": size_by,
                  "budget": round(float(budget), 2)}
        self._submit(OrderSide.BUY, qty, "entry", reason, signal)

    def _submit(self, side, qty: Decimal, intent: str, reason: str, signal: dict) -> None:
        """Send a market order, journaling it with its reason first so the record exists before the venue sees it."""
        order = self.order_factory.market(
            instrument_id=self._cfg.instrument_id,
            order_side=side,
            quantity=Quantity.from_decimal_dp(qty, self.instrument.size_precision),
            time_in_force=TimeInForce.GTC,
        )
        signal = {k: (round(v, 8) if isinstance(v, float) else v) for k, v in signal.items()}
        signal.setdefault("price", self._price())
        self.decisions[str(order.client_order_id)] = {"intent": intent, "reason": reason, "signal": signal}
        if self.runtime is not None:
            self.runtime.on_order(order_id=str(order.client_order_id), side="BUY" if side == OrderSide.BUY else "SELL",
                                  qty=float(qty), intent=intent, reason=reason, signal=signal)
        self.submit_order(order)

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

    def _is_long(self) -> bool:
        return self._position_qty() >= self._min_qty()

    def _sell_all(self, intent: str = "exit", reason: str = "Signal to be flat", values: dict | None = None) -> None:
        if self.runtime is None and self.cache.orders_open(instrument_id=self._cfg.instrument_id):
            # The resting stop holds the position; cancel it and sell once the cancel confirms.
            self._pending_exit = (intent, reason, values)
            self.cancel_all_orders(self._cfg.instrument_id)
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
        return self._last_close or 0.0

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

    def _on_tick(self, _event=None) -> None:
        self._last_tick_ns = self.clock.timestamp_ns()
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

    def on_order_rejected(self, event) -> None:
        self._order_status(event, "rejected")

    def on_order_denied(self, event) -> None:
        self._order_status(event, "denied")

    def on_order_canceled(self, event) -> None:
        self._order_status(event, "canceled")
        if self._pending_exit is not None and not self.cache.orders_open(instrument_id=self._cfg.instrument_id):
            intent, reason, values = self._pending_exit
            self._pending_exit = None
            self._sell_all(intent, reason, values)

    def on_order_expired(self, event) -> None:
        self._order_status(event, "expired")

    def on_order_filled(self, event) -> None:
        qty, px = event.last_qty.as_double(), event.last_px.as_double()
        if event.is_buy:
            cost = (self._entry_px or 0.0) * self._entry_qty + qty * px
            self._entry_qty += qty
            self._entry_px = cost / self._entry_qty
        else:
            self._entry_qty = max(self._entry_qty - qty, 0.0)
            if self._entry_qty <= 1e-12:
                self._entry_px, self._entry_qty = None, 0.0
        if self.runtime is None:
            if event.is_buy:
                self._rest_stop()
            elif self.decisions.get(str(event.client_order_id), {}).get("intent") == "stop_loss":
                self._exit_lock = True
            return
        fee = event.commission.as_double() if event.commission is not None else 0.0
        self.runtime.on_fill(
            side="BUY" if event.is_buy else "SELL",
            qty=event.last_qty.as_double(),
            price=event.last_px.as_double(),
            fee=fee,
            order_id=str(event.client_order_id),
            trade_id=str(event.trade_id),
        )

    def _rest_stop(self) -> None:
        """Backtest only: after an entry fills, rest a sell stop at the stop-loss level, as a real
        stop order would sit at the venue. Bars are matched open, high, low, close, so it fills at
        the level within the bar, or at the open when the price gaps through it."""
        if not self._cfg.stop_loss or self._entry_px is None:
            return
        qty = self._position_qty(free=True).quantize(self.instrument.size_increment.as_decimal(), rounding=ROUND_DOWN)
        if qty < self._min_qty():
            return
        level = self._entry_px * (1 - self._cfg.stop_loss)
        order = self.order_factory.stop_market(
            instrument_id=self._cfg.instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_decimal_dp(qty, self.instrument.size_precision),
            trigger_price=Price(level, self.instrument.price_precision),
            time_in_force=TimeInForce.GTC,
        )
        self.decisions[str(order.client_order_id)] = {
            "intent": "stop_loss",
            "reason": (f"Stop-loss: resting sell at {level:,.6g}, {self._cfg.stop_loss:.1%} below the "
                       f"{self._entry_px:,.6g} entry; fills at that level, or the open if the price gaps through"),
            "signal": {"entry_px": round(self._entry_px, 8), "stop_loss": self._cfg.stop_loss,
                       "trigger": round(level, 8)},
        }
        self.submit_order(order)

    def on_stop(self) -> None:
        self.cancel_all_orders(self._cfg.instrument_id)
        if self.runtime is not None:
            self.clock.cancel_timer("sleeve-tick") if "sleeve-tick" in self.clock.timer_names() else None
            self.runtime.on_stop()
