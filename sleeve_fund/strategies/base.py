"""Strategy template.

Every strategy is a plain NautilusTrader Strategy (no AI in the order path) plus
an IdeaSpec that records where it came from and what it needs. The same class
runs in backtest and in paper trading. Phase 1 is spot, so the base class is
long-or-flat with the whole sleeve in or out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Any

from nautilus_trader.config import StrategyConfig
from datetime import timedelta

from nautilus_trader.model import Bar, BarType, InstrumentId, OrderSide, PriceType, Quantity, TimeInForce
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
        assumed_taker_fee: float = 0.008,
        warmup_bars: int = 0,
        **kwargs: Any,
    ) -> None:
        unknown = set(kwargs) - _BASE_FIELDS
        if unknown:
            raise TypeError(f"unknown strategy parameters: {sorted(unknown)}")
        super().__init__()
        if not 0 <= assumed_taker_fee < 0.05:
            raise ValueError(f"assumed_taker_fee {assumed_taker_fee} outside [0, 0.05)")
        if warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")
        if not 0 <= cash_buffer < 0.5:
            raise ValueError(f"cash_buffer {cash_buffer} outside [0, 0.5)")
        if max_notional is not None and max_notional <= 0:
            raise ValueError("max_notional must be positive")
        self.instrument_id = instrument_id
        self.bar_type = bar_type
        # Leave room for the taker fee and rounding so a full-size buy never rejects.
        self.cash_buffer = cash_buffer
        # Hard cap on the quote-currency value of any single buy (sleeve budget).
        self.max_notional = max_notional
        # Venue instruments may not carry fee rates (Kraken spot doesn't), so sizing uses this.
        self.assumed_taker_fee = assumed_taker_fee
        # Live/paper only: bars to request from the venue at start to warm indicators.
        self.warmup_bars = warmup_bars


class LongFlatStrategy(Strategy):
    """Holds 100% of the sleeve or 0%. Subclasses implement want_long()."""

    def __init__(self, config: LongFlatConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self.instrument = None
        self._last_bar_ts = 0
        self._last_close = None
        # SleeveRuntime in paper/live (journal, PM controls, risk guard); None in backtest.
        # Attach with attach_runtime() before the strategy is added to a node or engine.
        self.runtime = None

    def attach_runtime(self, runtime) -> "LongFlatStrategy":
        self.runtime = runtime
        return self

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self._cfg.instrument_id)
        if self.instrument is None:
            self.log.error(f"instrument {self._cfg.instrument_id} not found")
            self.stop()
            return
        if self._cfg.warmup_bars:
            self.request_bars(self._cfg.bar_type, limit=self._cfg.warmup_bars)
        self.subscribe_bars(self._cfg.bar_type)
        if self.runtime is not None:
            self.runtime.on_start(self._cfg.assumed_taker_fee, now=lambda: self.clock.utc_now().replace(microsecond=0))
            # Trades give a fresh price for marking and the risk guard between (daily) bars.
            self.subscribe_trades(self._cfg.instrument_id)
            self.clock.set_timer("sleeve-tick", timedelta(seconds=self.runtime.tick_seconds), callback=self._on_tick)

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

    def on_bar(self, bar: Bar) -> None:
        if not self._accept(bar):
            return
        self.log.info(f"bar {bar}")
        self._last_close = bar.close.as_double()
        target = self.want_long(bar)
        if target is None:
            return
        if self.cache.orders_inflight(strategy_id=self.strategy_id):
            return
        is_long = self.portfolio.is_net_long(self._cfg.instrument_id)
        if target and not is_long:
            if self.runtime is not None and not self.runtime.can_open():
                return
            self._buy_all(bar)
        elif not target and is_long:
            self.close_all_positions(self._cfg.instrument_id)

    def _buy_all(self, bar: Bar) -> None:
        account = self.portfolio.account(self._cfg.instrument_id.venue)
        if account is None:
            self.log.warning("no account yet; skipping buy")
            return
        free = account.balance_free(self.instrument.quote_currency)
        if free is None:
            return
        budget = free.as_decimal()
        if self._cfg.max_notional is not None:
            budget = min(budget, Decimal(str(self._cfg.max_notional)))
        if self.runtime is not None:
            budget = min(budget, Decimal(str(self.runtime.position_budget(self._mark()[0]))))
        budget *= Decimal(1) - Decimal(str(self._cfg.cash_buffer)) - Decimal(str(self._cfg.assumed_taker_fee))
        step = self.instrument.size_increment.as_decimal()
        qty = (budget / bar.close.as_decimal()).quantize(step, rounding=ROUND_DOWN)
        min_qty = self.instrument.min_quantity.as_decimal() if self.instrument.min_quantity else step
        if qty <= 0 or qty < min_qty:
            self.log.warning(f"buy size {qty} below minimum {min_qty}; skipping")
            return
        order = self.order_factory.market(
            instrument_id=self._cfg.instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_decimal_dp(qty, self.instrument.size_precision),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    # --- sleeve runtime hooks (paper/live only) --------------------------------

    def _price(self) -> float:
        px = self.cache.price(self._cfg.instrument_id, PriceType.LAST)
        if px is not None:
            return px.as_double()
        return self._last_close or 0.0

    def _mark(self) -> tuple[float, float, float, float]:
        """(equity, cash, coin qty, price) in the quote currency."""
        price = self._price()
        account = self.portfolio.account(self._cfg.instrument_id.venue)
        if account is None or self.instrument is None:
            return 0.0, 0.0, 0.0, price
        cash_m = account.balance_total(self.instrument.quote_currency)
        coin_m = account.balance_total(self.instrument.base_currency)
        cash = cash_m.as_double() if cash_m else 0.0
        qty = coin_m.as_double() if coin_m else 0.0
        return cash + qty * price, cash, qty, price

    def _on_tick(self, _event=None) -> None:
        try:
            equity, cash, qty, price = self._mark()
            if price <= 0 or equity <= 0:
                return
            if self.runtime.tick(equity=equity, cash=cash, qty=qty, price=price) == "flatten":
                self.cancel_all_orders(self._cfg.instrument_id)
                self.close_all_positions(self._cfg.instrument_id)
        except Exception as exc:  # never let bookkeeping kill the sleeve silently
            self.log.error(f"sleeve tick failed: {exc!r}")
            self.runtime.store.event(self.runtime.name, "error", "tick_failed", repr(exc))

    def on_order_filled(self, event) -> None:
        if self.runtime is None:
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

    def on_stop(self) -> None:
        self.cancel_all_orders(self._cfg.instrument_id)
        if self.runtime is not None:
            self.clock.cancel_timer("sleeve-tick") if "sleeve-tick" in self.clock.timer_names() else None
            self.runtime.on_stop()
