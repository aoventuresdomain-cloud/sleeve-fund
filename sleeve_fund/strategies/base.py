"""Strategy template.

Every strategy is a plain NautilusTrader Strategy (no AI in the order path) plus
an IdeaSpec that records where it came from and what it needs. The same class
runs in backtest and in paper trading. On spot, positions are long or flat: a strategy
says what share of the sleeve to hold (target_weight, 0 to 1), or simply in or out
(want_long). On a perpetual (sleeve_fund.markets) a strategy with allow_short may also
be short: it says which side to be on (want_side, +1, 0 or -1).
"""

from __future__ import annotations

import math
import os
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal
from typing import Any

from nautilus_trader.config import StrategyConfig
from datetime import datetime, timedelta, timezone

from nautilus_trader.core import UUID4
from nautilus_trader.model import (Bar, BarType, ClientOrderId, ContingencyType, InstrumentId, LimitOrder, Money, OrderSide,
                                   OrderStatus, OrderType, Price, PriceType, Quantity, StopMarketOrder, TimeInForce,
                                   TriggerType)
from nautilus_trader.trading import Strategy

from sleeve_fund import bars as bar_rule
from sleeve_fund import markets, open_risk, risk
from sleeve_fund.data import bar_minutes
from sleeve_fund.instruments import BOOK_SHARE, lot_decimals, pair_of
from sleeve_fund.paper.runtime import WIPED_OUT, liquidation_reason
from sleeve_fund.store import DUST, OPEN_ORDER_STATUSES, replay_book
from sleeve_fund.strategies.indicators import AtrSma

# Orders the signal asks for may wait for a maker fill; protective exits (stop-loss, take-profit,
# risk halts, PM flatten) always go at market, because getting out matters more than the fee.
MAKER_INTENTS = ("entry", "exit", "rebalance")
OPENING_INTENTS = ("entry", "rebalance")  # every other order only ever reduces a position
EXIT_LEGS = ("stop_loss", "take_profit")  # the resting exits a backtest keeps through a reconcile halt
# Exits after which the side they closed isn't entered again until the signal has moved off it (_exit_lock).
LOCKING_INTENTS = (*EXIT_LEGS, "liquidation", "liquidation_cut")
MINUTE_NS = 60_000_000_000
DAY_NS = 86_400_000_000_000
SAFETY_STOP_SHARE = 0.5  # of the way from the mark to the liquidation price (_safety_stop_on_restore)
# The status reason the supervisor gives a refused start that still holds a position: it runs for its exits only.
EXITS_ONLY = "exits only"


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


_OPS = {"<=": lambda v, t: v <= t, ">=": lambda v, t: v >= t, "<": lambda v, t: v < t, ">": lambda v, t: v > t}


@dataclass(frozen=True, slots=True)
class Condition:
    """One of a model's rules on one bar, as the strategy page's Signals tab shows it: the rule in words,
    the value it reads, the threshold, how they compare (op: <=, >=, < or >), whether it holds, the unit
    (pts or %) and the span a gauge draws. exit: a rule that ends the leg the model is on, rather than one
    that opens a new one. A rule with no measure (value None) only says whether it holds.

    A model builds these in the code its decision uses (target_side, want_long), so the tab can't disagree
    with what the model does on the bar."""

    name: str
    value: float | None
    threshold: float
    op: str
    met: bool
    unit: str = "pts"
    gauge_min: float = 0.0
    gauge_max: float = 100.0
    exit: bool = False
    note: str = ""

    @classmethod
    def check(cls, name: str, value: float, op: str, threshold: float, **kw) -> "Condition":
        """The rule `value op threshold`, met as the comparison says."""
        return cls(name, value, threshold, op, _OPS[op](value, threshold), **kw)


def _condition_json(c: Condition) -> dict:
    """A condition as plain JSON (the Signals tab's store row)."""
    def r(v):
        return None if v is None or not math.isfinite(v) else round(float(v), 6)

    return {"name": c.name, "value": r(c.value), "threshold": r(c.threshold), "op": c.op, "met": bool(c.met),
            "unit": c.unit, "gauge_min": r(c.gauge_min), "gauge_max": r(c.gauge_max), "exit": c.exit, "note": c.note}


# StrategyConfig is a native type: Python passes the same keyword arguments to its
# __new__, which reads the base fields (strategy_id etc.) from them. So __init__ must
# not forward them, and we reject anything unrecognised so a typo in a sleeve file fails.
# A stop set from the market is kept between these shares of the price: a swing low at the close would
# otherwise give no room at all, and a stop half the price away protects nothing.
MIN_STOP, MAX_STOP = 0.002, 0.5
# The most of a bar's traded volume one buy may take (see LongFlatConfig.max_participation).
MAX_PARTICIPATION = 0.25
# Paper's price watchdog: minutes without a trade or a quote before it warns, and before it treats the
# feed as dead and stops reporting, so the supervisor restarts the process and it reconnects.
STALE_PRICE_WARN_MINUTES = 5
STALE_PRICE_RESTART_MINUTES = 15
# Paper: how often at most the model's conditions on the forming candle are written for the Signals tab
# (and straight after each bar). Display only: never in a backtest.
SIGNAL_WRITE_EVERY_NS = 5_000_000_000

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


def maker_orders_enabled() -> bool:
    """Post-only (maker-first) orders are switched off: every order goes at market. At this size the maker
    fill model cost more review findings than it saved in fees (PM, 4 Oct 2026), so it stays in the code
    for when a strategy proves it needs it, behind SLEEVE_MAKER_ORDERS=1."""
    return os.environ.get("SLEEVE_MAKER_ORDERS", "") == "1"


def r_target(r: float, stop: float, leg: float, side: int = 1) -> float:
    """The target, as a share of the entry price, that makes r times what the stop loses, both after
    costs: each leg pays `leg` (the taker fee and half the spread) on its notional. A long's stop-out
    loses stop + leg + (1 - stop) * leg; a target hit makes tp - leg - (1 + tp) * leg. So "2R" pays 2R,
    not 2R before costs, which on a 2.3% stop at 0.85% a leg nets about +0.7R (review round 8, R8-M1).
    A short (side -1) buys back above the entry at its stop and below it at its target, so its stop leg
    costs a little more and its target needs a little less (long/short verdict, the arithmetic for a short)."""
    return (r * loss_at_stop(stop, leg, side) + 2 * leg) / (1 - side * leg)


def loss_at_stop(stop: float, leg: float, side: int = 1) -> float:
    """1R: the share of a position's entry value lost at its stop, costs included. The exit leg pays its
    cost on the exit's value: below the entry for a long, above it for a short."""
    return stop + leg + (1 - side * stop) * leg


def gain_at_target(tp: float, leg: float, side: int = 1) -> float:
    """The share of a position's entry value a target hit makes after both legs' costs."""
    return tp - leg - (1 + side * tp) * leg


def _ns(ts: datetime) -> int:
    """A journal time as clock nanoseconds (a stored time without a zone is UTC)."""
    ts = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp()) * 1_000_000_000 + ts.microsecond * 1000


def _from_entry(stop: float, side: int = 1) -> str:
    """A stop's distance in words: "2.0% below the entry" for a long ("above" for a short), or the other
    way for a stop set from the market on a position already in profit."""
    below = (stop >= 0) == (side > 0)
    return f"{abs(stop):.1%} {'below' if below else 'above'} the entry"


def through_liquidation(side: int, price: float, liq: float | None) -> bool:
    """Whether a price is at or past a position's liquidation price on its losing side (side: +1 long, -1 short).
    The one test behind the engine's liquidation check and GAP-LIQ's (a stop filled at or past it books as a
    liquidation), in paper and in a backtest alike."""
    return liq is not None and price > 0 and (price <= liq if side > 0 else price >= liq)


def entry_liquidation(cash: float, qty: float, close: float, side: int, fee: float, maintenance: float,
                      leverage: float) -> tuple[float | None, float]:
    """The liquidation price of an entry of qty at close once it fills (the fee paid, its notional over the
    leverage as its isolated margin: markets.isolated_margin), and how far that is from close as a share of
    it (inf when none)."""
    notional = qty * close  # cash afterwards is spot-style, as the journal keeps it: the fee paid, the notional taken out
    liq = markets.isolated_liquidation(cash - side * notional * (1 + side * fee), side * qty, close, leverage, maintenance)
    return liq, (abs(liq / close - 1) if liq is not None else float("inf"))


def _utc(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _side_word(side: int) -> str:
    return "long" if side > 0 else "short" if side < 0 else "flat"


def exit_warmup(params: dict) -> int:
    """Bars an ATR or swing-low stop looks back over: an entry waits until it can set its stop."""
    if params.get("stop_atr"):
        return int(params.get("atr_bars") or 14) + 1
    return int(params.get("stop_swing_bars") or 0)


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
        stop_atr: float | None = None,
        stop_swing_bars: int | None = None,
        atr_bars: int = 14,
        take_profit_r: float | None = None,
        position_cap_pct: float | None = None,
        rebalance_band: float | None = None,
        maker_wait_minutes: int | None = None,
        max_participation: float | None = MAX_PARTICIPATION,
        volume_scale: float = 1.0,
        market: str = markets.SPOT,
        allow_short: bool = False,
        demo_mirror: bool = False,
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
        stops = [k for k, v in (("stop_loss", stop_loss), ("stop_atr", stop_atr), ("stop_swing_bars", stop_swing_bars))
                 if v is not None]
        if len(stops) > 1:
            raise ValueError(f"choose one kind of stop-loss, not {' and '.join(stops)}")
        if stop_atr is not None and not 0 < stop_atr <= 20:
            raise ValueError(f"stop_atr {stop_atr} outside (0, 20]; a multiple of the simple average true range, e.g. 2")
        for label, v in (("stop_swing_bars", stop_swing_bars), ("atr_bars", atr_bars)):
            if v is not None and (int(v) != v or not 2 <= v <= 500):
                raise ValueError(f"{label} {v} must be a whole number of bars from 2 to 500")
        if take_profit_r is not None:
            if not 0 < take_profit_r <= 50:
                raise ValueError(f"take_profit_r {take_profit_r} outside (0, 50]; a multiple of the stop distance")
            if take_profit is not None:
                raise ValueError("set the take-profit as a % or as a multiple of the stop distance, not both")
            if not stops:
                raise ValueError("a take-profit in multiples of the stop distance needs a stop-loss")
        if position_cap_pct is not None and not 0 < position_cap_pct <= 1:
            raise ValueError(f"position_cap_pct {position_cap_pct} outside (0, 1]")
        if rebalance_band is not None and not 0 <= rebalance_band < 1:
            raise ValueError(f"rebalance_band {rebalance_band} outside [0, 1)")
        if rebalance_band is not None and (stops or take_profit or take_profit_r or risk_per_trade):
            raise ValueError("stop-loss, take-profit and risk per trade work on all-or-nothing positions; "
                             "they can't be combined with rebalancing to a target weight yet")
        if maker_wait_minutes is not None and not maker_orders_enabled():
            raise ValueError("maker-first orders are switched off for now: every order goes at market, at the taker fee")
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
        if risk_per_trade is not None and not stops:
            raise ValueError("risk_per_trade needs a stop_loss (size = equity x risk / loss at the stop)")
        if market not in markets.MARKETS:
            raise ValueError(f"unknown market {market!r}; choose one of {', '.join(markets.MARKETS)}")
        if allow_short and market == markets.SPOT:
            raise ValueError("short positions need a perpetual: set market to perp (or perp-venue-fees)")
        if market != markets.SPOT:
            if maker_wait_minutes is not None:
                raise ValueError("maker-first orders aren't available on a perpetual yet: every order goes at market")
            if rebalance_band is not None:
                raise ValueError("rebalancing to a target weight isn't available on a perpetual yet")
        # A target in R is after costs by construction (r_target); a fixed one must clear them itself.
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
        # Or a stop set from the market at each entry: this many average true ranges (over atr_bars
        # bars) below the close, or at the lowest low of the last stop_swing_bars bars.
        self.stop_atr = stop_atr
        self.stop_swing_bars = int(stop_swing_bars) if stop_swing_bars is not None else None
        self.atr_bars = int(atr_bars)
        # Or a take-profit that makes this many times what the stop loses, both after costs (r_target).
        self.take_profit_r = take_profit_r
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
        # spot: a cash account, long or flat. perp / perp-venue-fees: a perpetual on margin, with funding,
        # a liquidation price and leverage limits (sleeve_fund.markets); allow_short lets it go short.
        self.market = market
        self.allow_short = bool(allow_short)
        self.perp = markets.terms({"market": market}, str(instrument_id.venue))
        # Read by the demo mirror (sleeve_fund.mirror), not by the strategy: paper is unchanged.
        self.demo_mirror = bool(demo_mirror)


# Handlers whose exceptions the engine would swallow or only log: market data, order events, the bar
# that decides, the risk check's tick and the maker order's timer (review round 8, M8-3).
REPORTED_HANDLERS = ("on_trade", "on_quote", "on_order_accepted", "on_order_rejected", "on_order_denied",
                     "on_order_canceled", "on_order_expired", "on_order_filled", "on_order_updated",
                     "on_order_modify_rejected", "on_bar", "_on_tick", "_maker_timeout")
_HANDLER_WORDS = {"on_trade": "a trade print", "on_quote": "a quote", "on_order_accepted": "an order accepted",
                  "on_order_rejected": "an order rejected", "on_order_denied": "an order denied",
                  "on_order_canceled": "an order cancelled", "on_order_expired": "an order expired",
                  "on_order_filled": "an order filled", "on_order_updated": "an order resized",
                  "on_order_modify_rejected": "a resize refused", "on_bar": "a bar", "_on_tick": "the risk check",
                  "_maker_timeout": "a post-only order's wait running out"}
MAX_KEPT_ERRORS = 100  # handler errors kept with their text; every one is counted


def handler_error_words(handler: str, exc: BaseException | str) -> str:
    """'handling an order filled: float division by zero (ZeroDivisionError)', for the PM rather than a
    Python repr. exc: the exception, or its repr as a run keeps it."""
    if isinstance(exc, BaseException):
        kind, what = type(exc).__name__, str(exc)
    else:
        m = re.fullmatch(r"(\w+)\((['\"])(.*)\2\)", exc)
        kind, what = (m.group(1), m.group(3)) if m else ("", exc)
    where = _HANDLER_WORDS.get(handler, handler.replace("_", " "))
    return f"handling {where}: {what or 'no message'}" + (f" ({kind})" if kind else "")


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
        self._entry_qty = 0.0  # its size, unsigned
        self._entry_side = 0  # +1 long, -1 short, 0 flat
        # Perp only. A signal that turns a long short (or the other way) closes first; the new side opens
        # as soon as the close has filled, as (side, bar, reason, values).
        self._flip: tuple | None = None
        # Paper on a perp: the sandbox's margin account can't be given a position at start, so a position
        # carried over a restart is bought or sold again there, at no fee and unjournaled ("restore"),
        # before anything else trades. _cash_adj then keeps the account's cash equal to the journal's:
        # the restore filled at today's price, not the entry, and funding is the journal's alone.
        self._restore: dict | None = None
        self._restore_id: str | None = None
        self._cash_adj = 0.0
        self._funding_since = None  # the last time funding was settled up to
        # Perp: the position and the last trade price held at each hour's start, as the first event past it
        # found them (_snap_settlements), so funding charges the position held at the settlement instant, not
        # at the tick that charges it (QA P1-O2). Settlements fall on the hour; the last few days are kept.
        self._held_at: dict[int, tuple[float, float]] = {}
        # A bars-only backtest's resting fill, while its funding is settled: (the fill's bar open ns, its close ns,
        # whether it filled on a gap at the open). See _intrabar_fill.
        self._intrabar: tuple[int, int, bool] | None = None
        self._opened_in_bar: list[tuple[tuple[int, int, bool], float]] = []  # _fund_opened_in_bar, once the bar is in
        # Liquidated in this process: the halt it stays in, "Position margin lost (liquidated): ..." (_margin_lost).
        self._liquidated: str | None = None
        self._snap_ns: int | None = None
        self._settled = None  # backtest: the venue's settled rates, loaded once
        self._funding_fallback_said = False  # the baseline fallback for a missing settled rate is said once
        self.funding_log: list[tuple] = []  # (time, amount) for every funding payment, for a backtest's equity
        # (time, amount) for every shortfall the venue's insurance fund took past the bankruptcy price
        self.insurance_log: list[tuple] = []
        self._rebook: str | None = None  # a stop's order filled past liquidation, to re-book as one (GAP-LIQ)
        # Backtest on a perp: the equity at the minute's worst price, for the risk guard (_intrabar_guard),
        # and that price when it went through the liquidation price.
        self._guard_equity: float | None = None
        self._guard_price: float | None = None
        # Backtest on a perp: the reduce-only stop resting at the price the risk guard would act at
        # (_rest_risk_stop), so a risk exit fills where it was judged, as a stop-loss does.
        self._risk_stop_id: str | None = None
        # This position's stop and target as shares of the entry price, fixed when it was entered
        # (from the % settings, or from the market for an ATR or swing-low stop); None when flat.
        self._stop_frac: float | None = None
        self._tp_frac: float | None = None
        self._stop_basis = ""  # how the stop was set, in words, for the journal
        # After a restart: a new ATR or swing-low stop to set from the market on the next bar, with the old
        # stop working until then (_replan), as ("edit" or "restart", the settings-change event it applies).
        self._replan_pending: tuple[str, int] | None = None
        self._plan_entry: dict | None = None  # the open position's entry order, after a restart
        self._atr = AtrSma(config.atr_bars) if config.stop_atr else None
        self._lows: deque[float] | None = deque(maxlen=config.stop_swing_bars) if config.stop_swing_bars else None
        # A short's swing stop sits at the highest high (review round 11, M11-7).
        self._highs: deque[float] | None = deque(maxlen=config.stop_swing_bars) if config.stop_swing_bars else None
        self._exit_lock = False  # after a stop/target exit, wait for the signal to reset before re-entering
        # After a restart: the journal's last entry and the exit after it that locked re-entry, which the warm-up
        # bars since are decided on again to rebuild the model's leg (_plan_resume, _replay); None once done.
        self._resume: dict | None = None
        self._pending_exit = None  # a sell waiting for every working order to close first
        self._sent: list = []  # client order ids of orders sent, until the venue has them (see _unsent)
        self._cancel_on_accept: set[str] = set()  # orders to cancel as soon as the venue has them
        self._exec_type = None  # backtest: the shorter bars the decision bars are built from
        # Volumes of the last day's decision bars, for the participation cap on buys.
        self._volumes: deque[float] = deque(maxlen=max(1, 1440 // bar_minutes(config.bar_type)))
        self._noted: set[str] = set()  # warnings already logged; each is said once until it clears
        self._journal_intents: dict[str, str] | None = None  # open orders' intents from before a restart
        self._gated_fills: set[str] = set()  # orders whose fill came in while nothing may open (CHOKE): one incident each
        self._last_market_ns: int | None = None  # the latest trade or quote, for the price watchdog
        self._held_w = None  # the target weight last traded to (None: not known yet, e.g. after a restart)
        self._maker: dict[str, dict] = {}  # working post-only orders: intent, reason and signal by client order id
        self._fallback: set[str] = set()  # post-only orders this strategy cancelled for running out of time
        # Paper: the post-only orders this strategy holds itself rather than at the simulated venue, which
        # would fill one whole on the first trade through its price. A backtest fills BOOK_SHARE of what
        # trades through, a slice at a time, so paper does too: each slice goes at market as the tape earns
        # it, charged as filled at the limit with the maker fee (_slice_maker; review round 9, M9-3). By
        # client order id: the order, why it was sent, the minute bar building, what the tape has earned it,
        # how much went as slices, and how many are in flight.
        self._kept: dict[str, dict] = {}
        self._closing: dict[str, dict] = {}  # kept orders closed with a slice still in flight
        self._slices: dict[str, str] = {}  # each slice's client order id, to its kept order's
        self._tape_last = None  # the latest trade counted (time, trade id)
        # Paper and its replay (set by the node): post-only orders are kept here and sliced. Live, they rest
        # at the venue, whose own queue decides.
        self.simulated_venue = False
        # Paper fed by its venue's market data hub (v2 P1-1, sleeve_fund.paper.hub_client): the bars are the hub's,
        # so warm-up comes from the history store the hub feeds, never from the venue (set by the node).
        self.hub_fed = False
        self.hub_status = None  # the hub's own word on its venue connection (hub_client.HubStatus), when hub-fed
        # The interim open-risk limit (sleeve_fund.open_risk): a backtest's daily ATR shares by day (set by the runner),
        # and how many of its entries the limit would have refused; a backtest doesn't gate on it.
        self._daily_atr: dict[int, float] = {}
        self.open_risk_binds = 0
        # After a restart, a position whose stop couldn't be restored, or one started for its exits only, works to a
        # safety stop, and no new entry opens (_safety_stop_on_restore).
        self._safety_stop = False
        self._safety_pending: tuple[float, float] | None = None  # (cash, qty) at the restart, until the first price
        self._exits_only = False  # started only to run a held position's exits (EXITS_ONLY)
        self._exits_why: str | None = None  # why, when the strategy found it itself (_refused_to_trade)
        self._resizing: dict[str, str] = {}  # backtest exits asked to resize, with why, until the venue confirms
        self._resize_due = False  # an entry slice filled while an exit was in flight (_resize_exits)
        self._resized_ns = None  # when the exits were last resized (_rest_exits)
        self.fee_model = None
        # SleeveRuntime in paper/live (journal, PM controls, risk guard); None in backtest.
        # Attach with attach_runtime() before the strategy is added to a node or engine.
        self.runtime = None
        # Paper/live on bars built from live trades: loads warm-up bars from the venue history store
        # (the venue can't serve those bars). Attach with attach_history(); None requests them from
        # the venue's candles instead.
        self.history_loader = None
        self.gap_loader = None  # paper: (instrument, bar_type, since_ns, until_ns) -> the venue's own closed bars
        # Paper on its own feed: (instrument id, after ns, before ns) -> the 1-minute bars the history store holds
        # closing strictly between, as (close ns, open, high, low, close, volume) (paper.node.stored_minutes).
        self.minutes_loader = None
        self._first_minute: int | None = None  # the first minute this process saw market data in
        self._first_bar_due = True  # the bar under way at the start is still to be built from the store
        self._gap_bars: list[Bar] = []  # paper: candles built while no trades arrived, held until the feed is back
        # Degraded bars (board 5a): close time (ns) -> minutes missing, for bars built with more than
        # sleeve_fund.bars.DEGRADED_ABOVE of their minutes absent. Indicators update and exits run on them;
        # no new entry is decided on one. Given by mark_degraded(); each is dropped once its bar is seen.
        self._degraded: dict[int, int] = {}
        self._no_entry_ts: int | None = None  # the close time of the degraded bar being decided on
        self._degraded_missing = 0
        # Paper on its own trade feed builds its decision bars itself: the minutes (open time, in minutes since the
        # epoch) it saw market data in, so each bar's missing minutes are counted by the store's rule (_count_minutes,
        # QA P1-D2). A backtest fed shorter execution bars is told which decision bars they build (expect_bars, P1-D1).
        self._minutes_seen: set[int] = set()
        self._built: set[int] | None = None
        # Every order's intent, reason and signal by client order id, in backtests too, so a
        # backtest can show why each trade happened exactly as paper and live do.
        self.decisions: dict[str, dict] = {}
        # Exceptions raised in these handlers, oldest first, as (handler, repr). See _reporting.
        self.handler_errors: list[tuple[str, str]] = []
        self.handler_error_count = 0
        self._failed_handlers: set[str] = set()  # handlers already journaled as failed (see _report)
        # Paper: when the Signals tab's conditions were last written (clock ns) and for which bar; off once it
        # is known the model lists none. A failed write is said once in the log, never an order's concern.
        self._signals_ns = 0
        self._signals_bar = -1
        self._signals_on = True
        self._signals_warned = False
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
                self._report(name, exc)
        run.__name__ = name
        return run

    def _report(self, name: str, exc: Exception) -> None:
        """Keep a handler's exception: count it, keep the first MAX_KEPT_ERRORS in handler_errors, log it,
        and journal it as an error event (paper) the first time each handler fails, so a handler failing
        on every tick raises one alert, not one every 30 seconds."""
        self.handler_error_count += 1
        if len(self.handler_errors) < MAX_KEPT_ERRORS:
            self.handler_errors.append((name, repr(exc)))
        self.log.error(f"strategy handler {name} failed: {exc!r}\n{traceback.format_exc()}")
        if self.runtime is not None and name not in self._failed_handlers:
            self._failed_handlers.add(name)
            try:
                # The tick's failure is the risk check's: it lists with the breaches on the Risk page.
                self.runtime.store.event(self.runtime.name, "error",
                                         "tick_failed" if name == "_on_tick" else "handler_failed",
                                         f"The strategy failed {handler_error_words(name, exc)}. Its "
                                         "orders after this may be wrong.", ts=self.runtime.now())
            except Exception:  # the journal itself failing must not hide the first error
                pass

    def attach_runtime(self, runtime) -> "LongFlatStrategy":
        self.runtime = runtime
        return self

    def attach_history(self, loader) -> "LongFlatStrategy":
        """loader(instrument, bar_type, limit) -> the latest `limit` complete bars, oldest first."""
        self.history_loader = loader
        return self

    def mark_degraded(self, bars: dict[int, int]) -> None:
        """Bars, by close time in ns, built with too many minutes missing to enter on (sleeve_fund.bars, board
        5a), with how many: the backtest passes the store's flagged bars before the run, and a live bar
        builder each one as it closes, before handing the bar over."""
        self._degraded.update({int(ts): int(m) for ts, m in bars.items()})

    def expect_bars(self, closes) -> "LongFlatStrategy":
        """Backtest on execution bars: the close times (ns) of the decision bars they build. The engine makes up any
        other one flat at the last price from nothing, and it is dropped (runner.decision_bars, QA P1-D1)."""
        self._built = {int(t) for t in closes}
        return self

    def _first_bar_from_store(self, bar: Bar) -> Bar:
        """Paper on its own feed, the bar under way when the process started (a deploy or a restart mid-bar): built
        from the minutes the history store holds for its part before the start, the same minutes research reads,
        and from this process's own data after; degraded only for minutes still missing from both (_count_minutes).
        Never the venue's candle (Independent Quant Advisor and HoE, 6 Oct, R1; QA C7). Without the store it is the
        part this process saw, degraded as such."""
        if (not self._first_bar_due or self._backtest or self.hub_fed or self.minutes_loader is None
                or str(bar.bar_type) != str(self._cfg.bar_type).split("@")[0]
                or not str(bar.bar_type).endswith("INTERNAL")):
            return bar
        self._first_bar_due = False
        minutes, first = bar_minutes(self._cfg.bar_type), self._first_minute
        end = bar.ts_event // MINUTE_NS
        if minutes <= 1 or first is None or first <= end - minutes:
            return bar  # the process saw the whole bar
        try:  # minutes opening from the bar's start up to and including the first one seen here: closing (start, first + 1]
            rows = self.minutes_loader(str(self._cfg.instrument_id), (end - minutes) * MINUTE_NS,
                                       (first + 1) * MINUTE_NS + 1)
        except Exception as exc:  # noqa: BLE001 - an unreadable store: the part bar stands, degraded as such
            self.log.warning(f"stored minutes for the first bar unavailable: {exc}")
            return bar
        rows = [r for r in rows if end - minutes < r[0] // MINUTE_NS <= first + 1]
        if not rows:
            return bar
        self._minutes_seen.update(r[0] // MINUTE_NS - 1 for r in rows)
        # The minute this process started in, when the store has it: its whole range, since a start mid-minute saw
        # only its end (QA P1-D19). Its volume stays what was seen here, which the store's minute would count twice.
        before = [r for r in rows if r[0] // MINUTE_NS <= first]
        inst = self.instrument
        return Bar(bar.bar_type, inst.make_price(rows[0][1]),
                   inst.make_price(max(bar.high.as_double(), *(r[2] for r in rows))),
                   inst.make_price(min(bar.low.as_double(), *(r[3] for r in rows))), bar.close,
                   inst.make_qty(bar.volume.as_double() + sum(r[5] for r in before)), bar.ts_event, bar.ts_init)

    def _count_minutes(self, bar: Bar) -> bool:
        """Apply the store's bar rule (sleeve_fund.bars, board 5a) where this process builds its own decision bars.
        Paper on its own trade feed counts the minutes it saw market data in: too many missing marks the bar
        degraded, and a bar with none at all, which the engine makes up flat at the last price, is not decided on
        (QA P1-D2); with a gap loader that bar is left to _hold_gap, which rebuilds it from the venue's own candles
        (PM, 5 Oct 2026). A hub-fed node takes the hub client's count; a backtest on execution bars, the runner's
        (expect_bars). True when the bar is to be dropped."""
        if self.hub_status is not None and self.hub_status.degraded:
            self._degraded.update(self.hub_status.degraded)
            self.hub_status.degraded.clear()
        if str(bar.bar_type) != str(self._cfg.bar_type).split("@")[0]:
            return False
        if self._backtest:
            if self._built is None or bar.ts_event in self._built:
                return False
            self.log.info(f"bar {bar} dropped: none of its execution bars exist, so it isn't built (board 5a)")
            return True
        if self.hub_fed or not str(bar.bar_type).endswith("INTERNAL"):
            return False
        minutes = bar_minutes(self._cfg.bar_type)
        if minutes <= 1:
            return False
        end = bar.ts_event // MINUTE_NS
        seen = sum(1 for m in self._minutes_seen if end - minutes <= m < end)
        self._minutes_seen = {m for m in self._minutes_seen if m >= end}
        if seen == 0:
            if not self._backtest and self.gap_loader is not None:
                return False
            self.log.info(f"bar {bar} dropped: no data in any of its minutes, so it isn't built (board 5a)")
            return True
        missing = minutes - seen
        if bar_rule.degraded(missing, minutes):
            self._degraded.setdefault(bar.ts_event, missing)
        return False

    def _entry_blocked(self, bar: Bar) -> bool:
        """Whether no new entry may be decided on this bar because it is degraded; says why, once per bar."""
        if bar.ts_event != self._no_entry_ts:
            return False
        if self.runtime is not None and not self.runtime.can_open():
            return True  # halted or paused: nothing would open anyway, so nothing was held back (QA P1-D5)
        minutes = bar_minutes(self._cfg.bar_type)
        missing = self._degraded_missing
        self.log.info(f"entry held back on a degraded bar: {missing} of {minutes} minutes missing")
        self._note("degraded_bar", f"Entry held back: this {minutes}-minute bar is missing {missing} of its minutes "
                   "(over 10%), so no new position is opened on it; exits still run", level="info")
        return True

    def attach_minutes(self, loader) -> "LongFlatStrategy":
        """loader(instrument id, after_ns, before_ns) -> the history store's 1-minute bars closing strictly between,
        oldest first, as (close ns, open, high, low, close, volume): the first bar after a start is built from them
        (_first_bar_from_store)."""
        self.minutes_loader = loader
        return self

    def attach_gap_loader(self, loader) -> "LongFlatStrategy":
        """loader(instrument, bar_type, since_ns, until_ns) -> the venue's own closed bars stamped in that span,
        oldest first: what a candle built while no trades reached this process should have been."""
        self.gap_loader = loader
        return self

    @property
    def _margin(self) -> bool:
        """Trading a perpetual on a margin account (may go short) rather than spot in a cash account."""
        return self._cfg.perp is not None

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
        self._plan_resume()
        if self._cfg.warmup_bars:
            if self.history_loader is not None and (str(self._cfg.bar_type).endswith("INTERNAL") or self.hub_fed):
                self._warm_from_history()
                self._finish_resume()
            else:
                self.request_bars(self._cfg.bar_type, limit=self._cfg.warmup_bars)  # resumed when they arrive
        else:
            self._finish_resume()
        self.subscribe_bars(self._cfg.bar_type)
        if self.runtime is not None:
            self.runtime.on_start(self._cfg.assumed_taker_fee, now=lambda: self.clock.utc_now().replace(microsecond=0))
            book = self.runtime.book
            if book["entry_px"] and (book["qty"] > 0 or (book["qty"] < 0 and self._margin)):  # carried over a restart
                self._entry_px, self._entry_qty = book["entry_px"], abs(book["qty"])
                self._entry_side = 1 if book["qty"] > 0 else -1
                self._restore_plan()
                self._safety_stop_on_restore(book)
                if self._margin and not self.runtime.backtest:
                    self._restore = {"qty": book["qty"], "entry": book["entry_px"]}
            if self._margin:
                # Funding owed for times the process was down while a position was held is settled on the
                # first tick, at that tick's price.
                last = self.runtime.store.funding(self.runtime.name, limit=1)
                fills = self.runtime.store.fills(self.runtime.name, limit=1) if book["qty"] else []
                marks = [r["ts"] for r in (*last, *fills)]
                self._funding_since = max(marks) if marks else self.runtime.now()
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
        self._snap_settlements(tick.ts_event)
        self._last_close = tick.price.as_double()  # freshest price for marking between bars
        self._market_seen()
        if self._restore is not None:
            self._send_restore()
            return
        if self._kept:
            self._tape(tick)
        if self._check_exits(self._last_close):
            return
        if self._margin and self._guard_breached(self._last_close):
            self._on_tick()  # leveraged: the guards act on the trade, not on the next 30-second tick
            return
        self._maybe_tick()
        self._publish_signals()

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
        if self._restore is not None:
            self._send_restore()

    def _on_exec_bar(self, bar: Bar) -> None:
        """Backtest: value the book and run the risk guard on every execution bar, so a halt or a
        daily-loss pause fires within the decision bar, as paper's 30-second ticks would."""
        self._snap_settlements(bar.ts_event, bar.close.as_double())
        self._last_close = bar.close.as_double()
        if self._margin:
            self._intrabar_guard(bar)
        if self.clock.timestamp_ns() != self._last_tick_ns:  # the decision bar at this time may have ticked
            self._on_tick()
        self._rest_risk_stop()

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
        if self._atr is not None:
            self._atr.update_raw(bar.high.as_double(), bar.low.as_double(), bar.close.as_double())
        if self._lows is not None:
            self._lows.append(bar.low.as_double())
            self._highs.append(bar.high.as_double())
        self.update_indicators(bar)
        return True

    def _hold_gap(self, bar: Bar) -> bool:
        """Paper: a candle with no volume was built while no trades reached this process, a flat candle at the
        last price (a dropped connection, or a quiet market). It is held, not decided on, until trades arrive
        again, when _fill_gap checks it against the venue's own candles (PM, 5 Oct 2026)."""
        if (self._backtest or self.gap_loader is None or bar.bar_type != self._cfg.bar_type
                or bar.ts_event <= self._last_bar_ts or bar.volume.as_double() > 0):
            return False
        self._gap_bars.append(bar)
        self.log.info(f"bar {bar} held: no trades arrived during it")
        if self.runtime is not None:
            self._maybe_tick()
        return True

    def _fill_gap(self, bar: Bar) -> Bar:
        """The first candle with trades after held ones: feed the indicators the venue's own candles for the held
        span first (or the held ones where the venue had no trades either), then return the bar to decide on,
        the venue's own when it saw more of it than this process did."""
        if not self._gap_bars:
            return bar
        held, self._gap_bars = self._gap_bars, []
        try:
            venue = {b.ts_event: b for b in self.gap_loader(self.instrument, self._cfg.bar_type, held[0].ts_event,
                                                            bar.ts_event)}
            err = ""
        except Exception as exc:  # noqa: BLE001 - an unreachable venue: the held candles stand, and it says so
            venue, err = {}, str(exc)[:200]
        # The venue's candles reach back over the whole held span, so one it lacks had no trades there either.
        covered = bool(venue) and min(venue) <= held[0].ts_event
        rebuilt = unchecked = 0
        for h in held:
            v = venue.get(h.ts_event)
            if v is not None and v.volume.as_double() > 0:
                rebuilt += 1
                self._accept(v)
            else:
                unchecked += v is None and not covered
                self._accept(h)
        own = venue.get(bar.ts_event)
        if own is not None and own.volume.as_double() >= bar.volume.as_double():
            bar = own
            self._degraded.pop(bar.ts_event, None)  # the venue's whole candle: no longer a part bar (CR minor)
        if (rebuilt or unchecked) and self.runtime is not None:
            step = bar_minutes(self._cfg.bar_type)
            first = datetime.fromtimestamp(held[0].ts_event / 1e9 - step * 60, tz=timezone.utc)
            last = datetime.fromtimestamp(held[-1].ts_event / 1e9, tz=timezone.utc)
            n = len(held)
            msg = (f"No trades reached this process for {n} candle{'s' if n != 1 else ''} ({first:%d %b %H:%M} to "
                   f"{last:%H:%M} UTC)")
            if rebuilt:
                msg += f"; {rebuilt} rebuilt from the venue's own candles before deciding"
            if unchecked:
                why = f"couldn't fetch the venue's candles: {err}" if err else "the venue's candles don't reach back"
                msg += (f"; {unchecked} couldn't be checked ({why}), so the model used them flat at the last "
                        "price")
            self.runtime.store.event(self.runtime.name, "warning", "feed_gap", msg, ts=self.runtime.now())
        return bar

    @property
    def _has_exits(self) -> bool:
        c = self._cfg
        return bool(c.stop_loss or c.stop_atr or c.stop_swing_bars or c.take_profit)

    def _plan_exits(self, close: float, side: int = 1) -> tuple[float | None, float | None, str] | None:
        """This entry's stop and target as shares of the price, and how the stop was set; None while
        an ATR or swing-low stop hasn't seen enough bars to be set."""
        c = self._cfg
        stop, basis = c.stop_loss, _from_entry(c.stop_loss, side) if c.stop_loss else ""
        if c.stop_atr:
            if not self._atr.initialized or close <= 0:
                return None
            stop, basis = c.stop_atr * self._atr.value / close, (
                f"{c.stop_atr:g} x the {c.atr_bars}-bar simple average true range ({self._atr.value:,.6g})")
        elif c.stop_swing_bars:
            if len(self._lows) < c.stop_swing_bars or close <= 0:
                return None
            if side < 0:
                high = max(self._highs)
                stop, basis = high / close - 1, f"at the highest high of the last {c.stop_swing_bars} bars ({high:,.6g})"
            else:
                low = min(self._lows)
                stop, basis = 1 - low / close, f"at the lowest low of the last {c.stop_swing_bars} bars ({low:,.6g})"
        if stop is not None and (c.stop_atr or c.stop_swing_bars):
            clamped = min(max(stop, MIN_STOP), MAX_STOP)
            if clamped != stop:
                basis += f", held to {clamped:.1%}"
            stop = clamped
        return stop, self._target_for(stop, side), basis

    def _target_for(self, stop: float | None, side: int | None = None) -> float | None:
        """The target as a share of the entry price: the fixed % one, or r_target around this stop."""
        c = self._cfg
        if c.take_profit:
            return c.take_profit
        if c.take_profit_r and stop is not None and stop > 0:
            return r_target(c.take_profit_r, stop, self._round_trip_cost(), side or self._entry_side or 1)
        return None

    def _stop_cfg(self) -> dict:
        """The stop settings, journaled with each entry, so a later edit can tell whether it changed the stop."""
        c = self._cfg
        if c.stop_atr:
            return {"stop_atr": c.stop_atr, "atr_bars": c.atr_bars}
        if c.stop_swing_bars:
            return {"stop_swing_bars": c.stop_swing_bars}
        return {"stop_loss": c.stop_loss} if c.stop_loss else {}

    def _restore_plan(self) -> None:
        """After a restart with a position open, the stop and target it works to: the plan its entry
        journaled, or the latest one set since (Store.exit_plan). When the PM has changed the stop or
        target since, the new settings apply to it (review round 8, B8-1):
        - a target-only edit keeps the stop to the tick and sets the target again around it;
        - a new % stop applies at once;
        - a new ATR or swing-low stop is set from the market on the next bar with bars enough (_replan),
          and the old stop keeps working until then.
        An entry journaled before stops were recorded gets the fixed % plan, or an ATR or swing-low stop
        set from the market once there are bars enough."""
        c, store, name = self._cfg, self.runtime.store, self.runtime.name
        opened = "SELL" if self._entry_side < 0 else "BUY"
        entry = next((o for o in store.orders(name, limit=200)
                      if o.get("intent") == "entry" and o.get("side") == opened), None)
        self._plan_entry, self._replan_pending = entry, None
        if entry is None:
            if self._has_exits and not (c.stop_atr or c.stop_swing_bars):
                self._stop_frac, self._tp_frac, self._stop_basis = (self._plan_exits(self._entry_px, self._entry_side or 1)
                                                                    or (None, None, ""))
            return
        sig = entry.get("signal") or {}
        plan = store.exit_plan(name, entry["order_id"])
        if plan is not None:
            base = (plan["stop_frac"], plan["tp_frac"], plan["basis"] or "", plan["stop_cfg"])
        elif "stop_frac" in sig or "tp_frac" in sig:
            base = (sig.get("stop_frac"), sig.get("tp_frac"), sig.get("stop_basis", ""), sig.get("stop_cfg"))
        else:
            base = None
        after = (plan or {}).get("event_id") or 0
        edits = [e for e in store.sleeve_events_since(name, ("exits_change",), after) if e["ts"] > entry["ts"]]
        if not edits:
            if base is not None:
                self._stop_frac, self._tp_frac, self._stop_basis = base[0], base[1], base[2]
            elif c.stop_atr or c.stop_swing_bars:
                last = store.last_event(name, ("exits_change",))
                self._replan_pending = ("restart", last["id"] if last else 0)
            elif self._has_exits:
                self._stop_frac, self._tp_frac, self._stop_basis = (self._plan_exits(self._entry_px, self._entry_side or 1)
                                                                    or (None, None, ""))
            return
        event_id = edits[-1]["id"]
        old_stop, old_tp, old_basis, old_cfg = base or (None, None, "", None)
        if old_cfg is not None:
            same_stop = old_cfg == self._stop_cfg()
        else:  # an entry journaled before its stop settings were: the change messages say what changed
            same_stop = base is not None and not any("Stop-loss " in e["message"] for e in edits)
        if same_stop:
            self._set_plan(old_stop, self._target_for(old_stop), old_basis, "edit", event_id)
        elif not (c.stop_atr or c.stop_swing_bars):
            stop = c.stop_loss
            self._set_plan(stop, self._target_for(stop), _from_entry(stop, self._entry_side or 1) if stop else "",
                           "edit", event_id)
        else:
            self._stop_frac, self._stop_basis = old_stop, old_basis
            self._tp_frac = c.take_profit or old_tp
            self._replan_pending = ("edit", event_id)

    def _replan(self, close: float) -> None:
        """Set an ATR or swing-low stop from the market for a position already open: as a price level
        (the close less so many ATRs, or the swing low itself), then as a share of the entry price, so
        the stop sits where its words say. It only ever tightens the stop working until now."""
        plan = self._plan_exits(close, self._entry_side or 1)
        if plan is None:
            self._note("stop_not_ready", ("The open position keeps its old stop until the new one can be set: "
                                          if self._stop_frac is not None else "The open position has no stop yet: ")
                       + f"it is set from the last {self._cfg.atr_bars if self._cfg.stop_atr else self._cfg.stop_swing_bars} "
                       "bars, and there aren't that many since the restart")
            return
        self._noted.discard("stop_not_ready")
        kind, event_id = self._replan_pending
        self._replan_pending = None
        distance, _, basis = plan
        side = self._entry_side or 1
        level = close * (1 - side * distance)
        stop = side * (1 - level / self._entry_px)
        if self._stop_frac is not None and self._stop_frac < stop:
            basis = (f"kept: the new setting ({basis}) would loosen it, to {level:,.6g}, and a stop set "
                     f"from the market only ever tightens; {self._stop_basis or _from_entry(self._stop_frac, self._entry_side or 1)}")
            stop = self._stop_frac
        self._set_plan(stop, self._target_for(stop) if (stop > 0 or not self._cfg.take_profit_r) else self._tp_frac,
                       basis, kind, event_id)
        if self._safety_stop and not self._exits_only:
            self._safety_stop = False
            self.runtime.store.event(self.runtime.name, "info", "stop_restored", "The model's own stop is set again "
                                     "for the open position, so new entries can open again.", ts=self.runtime.now())

    def _safety_stop_on_restore(self, book: dict) -> None:
        """After a restart holding a position (Independent Quant Advisor and HoE, 6 Oct; QA P1-S1, S5, S7):
        - when the model sets a stop but it couldn't be restored (none was journaled, and an ATR or swing stop needs
          more bars than there are yet), or
        - when the strategy was started for its exits only (a refused start still holding: EXITS_ONLY),
        no new entry or add opens, and a safety stop is set at the first price: half the REMAINING distance from that
        mark to the isolated liquidation price (half way to zero where there is none: a long at 1x, or spot), so it
        always sits strictly between the mark and liquidation. A stop already restored stays if it is tighter. The
        model's own stop, once set again (_replan), replaces it only if tighter, and entries open again (never for
        an exits-only start). It raises an incident, and never closes the position itself."""
        c = self._cfg
        status = self.runtime.store.sleeve(self.runtime.name)
        self._exits_only = (status.status == "paused" and (status.status_reason or "").startswith(EXITS_ONLY))
        if not self._exits_only and (why := self._refused_to_trade(status)) is not None:
            # Settings paper refuses to start (a stopless model above 1x), holding a position however it got here:
            # the same exits-only start as the supervisor gives one (QA P1-S7, R-S6).
            self._exits_only, self._exits_why = True, why
        unrestored = (self._stop_frac is None and bool(c.stop_loss or c.stop_atr or c.stop_swing_bars))
        if not (unrestored or self._exits_only):
            return
        self._safety_stop, self._safety_pending = True, (book["cash"], book["qty"])
        if self._replan_pending is None and self._stop_frac is None and (c.stop_atr or c.stop_swing_bars):
            last = self.runtime.store.last_event(self.runtime.name, ("exits_change",))
            self._replan_pending = ("restart", last["id"] if last else 0)  # the model's stop, once there are bars

    def _place_safety_stop(self, mark: float) -> None:
        """The restart's safety stop, at the first price after it (_safety_stop_on_restore)."""
        cash, qty = self._safety_pending
        self._safety_pending = None
        side, entry = self._entry_side or 1, self._entry_px
        liq = None
        if self._margin:
            lev = self.runtime.profile.max_leverage
            liq = markets.isolated_liquidation(cash, qty, entry, lev, self._cfg.perp.maintenance_margin)
        target = liq if liq is not None else 0.0
        if side * (mark - target) <= 0:
            return  # already through liquidation: the liquidation guard closes it
        level = mark - SAFETY_STOP_SHARE * (mark - target)
        restored = entry * (1 - side * self._stop_frac) if self._stop_frac is not None else None
        kept = restored is not None and side * (restored - level) >= 0  # the restored stop is the tighter one
        if not kept:
            self._stop_frac = side * (1 - level / entry)
            self._stop_basis = f"safety stop, half way from the {mark:,.6g} mark to " + (
                f"the liquidation price {liq:,.6g}" if liq is not None else "zero")
        why = (f"started for its exits only ({self._exits_reason()})" if self._exits_only else
               "its stop couldn't be restored after the restart")
        work = (f"it keeps its restored stop at {restored:,.6g}, tighter than a safety stop at {level:,.6g}" if kept
                else f"it works to a safety stop at {level:,.6g}, half way from the {mark:,.6g} mark to "
                + (f"the liquidation price {liq:,.6g}" if liq is not None else "zero"))
        self.runtime.store.event(
            self.runtime.name, "error", "incident",
            f"Incident, {self.runtime.name}: the open {_side_word(side)} position of {abs(qty):.12g} (entry "
            f"{entry:,.6g}) {why}; {work}. No new entries or adds open"
            + ("" if self._exits_only else " until the model's own stop is set again")
            + "; the position is not closed.", ts=self.runtime.now())
        if self._backtest and not kept:
            self._rest_exits()  # a backtest's stop rests at the venue, so a gap through it fills at the open

    def _exits_reason(self) -> str:
        if self._exits_why:
            return self._exits_why
        reason = self.runtime.store.sleeve(self.runtime.name).status_reason or ""
        return reason.removeprefix(EXITS_ONLY).strip(" :") or "refused"

    @staticmethod
    def _refused_to_trade(sleeve) -> str | None:
        """Why paper refuses to start this strategy's stored settings, or None (Supervisor._refused's checks)."""
        from sleeve_fund.strategies import check_perp_sizing, check_perp_stop

        try:
            check_perp_sizing(sleeve.strategy, sleeve.params)
            check_perp_stop(sleeve.strategy, sleeve.params, sleeve.risk_profile)
        except ValueError as exc:
            return str(exc)
        return None

    def _open_risk_refusal(self, side: int, qty: float, close: float, equity: float, ts_ns: int,
                           held_qty: float = 0.0, held_stop: float | None = None) -> str | None:
        """Why the interim open-risk limit refuses this entry (sleeve_fund.open_risk), or None: it gates entries and
        adds, never a start, and refuses rather than trims (Independent Quant Advisor). Paper gates on the account's
        book, counting this strategy's own position still held; a backtest only counts the entries it would have
        refused, against its own equity. A position whose stop the price has gone through counts as stopless and
        alerts (QA P1-S9)."""
        stop = close * (1 - side * self._stop_frac) if self._stop_frac else None
        if self._backtest:
            atr = self._daily_atr.get((ts_ns - 1) // DAY_NS * DAY_NS)
            try:
                risk = (open_risk.position_risk(side * qty, close, stop, atr)
                        + open_risk.position_risk(held_qty, close, held_stop, atr))
            except ValueError:
                return None
            if open_risk.check_entry(equity, 0.0, risk):
                self.open_risk_binds += 1
            return None
        from sleeve_fund.venues import venue as venue_profile

        now = self.runtime.now()

        def atr(s):
            return open_risk.history_atr_pct(venue_profile(s.venue).name, s.instrument, now)

        store, name = self.runtime.store, self.runtime.name
        try:
            book, others, through = open_risk.account_book(store, name, equity, atr)
            if open_risk.gapped(held_qty, close, held_stop):
                through.append(name)
            needs_atr = stop is None or (held_qty and (held_stop is None or name in through))
            mine_atr = atr(store.sleeve(name)) if needs_atr else None
            mine = (open_risk.position_risk(side * qty, close, stop, mine_atr)
                    + open_risk.position_risk(held_qty, close, None if name in through else held_stop, mine_atr))
        except ValueError as exc:
            return f"its open risk can't be measured: {exc}"
        if through:
            self._note("stop_gapped_open", "Open risk counts " + ", ".join(through) + " as having no stop: the price "
                       "has gone through its stop and the position is still open", level="error")
        else:
            self._noted.discard("stop_gapped_open")
        return open_risk.check_entry(book, others, mine)

    def set_daily_atr(self, lookup: dict[int, float]) -> "LongFlatStrategy":
        """Backtest: the daily ATR share known at each UTC day's start (open_risk.daily_atr_lookup)."""
        self._daily_atr = dict(lookup)
        return self

    def _set_plan(self, stop: float | None, tp: float | None, basis: str, kind: str, event_id: int) -> None:
        """Put a plan set after entry in force, journal it with the trade's R from now on, and say so."""
        self._stop_frac, self._tp_frac, self._stop_basis = stop, tp, basis
        if self.runtime is None or self._plan_entry is None:
            return
        sig = self._plan_entry.get("signal") or {}
        notional = self._entry_px * self._entry_qty
        cost = self._round_trip_cost()
        side = self._entry_side or 1
        # A looser stop risks more than the entry did, and R is measured on the larger of the two, so
        # widening a stop can't flatter the trade's R multiple (review round 8, M8-1).
        risk = max(float(sig.get("risk_amount") or 0.0),
                   notional * loss_at_stop(stop, cost, side) if stop is not None else 0.0) or None
        if self._cfg.take_profit_r and not self._cfg.take_profit and risk:
            # An R target is so many of the trade's 1R. After a tighter stop that is still the entry's
            # risk, so the target is set from it: a typed 3R records +3.00R, not less (review round 9, N2).
            tp = (self._cfg.take_profit_r * risk / notional + 2 * cost) / (1 - side * cost)
            self._tp_frac = tp
        planned = round(gain_at_target(tp, cost, side) * notional / risk, 2) if tp and risk else None
        store, name, now = self.runtime.store, self.runtime.name, self.runtime.now()
        store.set_exit_plan(name, self._plan_entry["order_id"], kind=kind, ts=now, event_id=event_id,
                            stop_frac=None if stop is None else round(stop, 6), tp_frac=None if tp is None else round(tp, 6),
                            basis=basis, stop_cfg=self._stop_cfg(), risk_amount=None if risk is None else round(risk, 2),
                            planned_r=planned)
        parts = [f"stop at {self._entry_px * (1 - side * stop):,.6g}, {_from_entry(stop, side)}" + (f" ({basis})" if basis and
                 not basis.endswith("the entry") else "") if stop is not None else "no stop",
                 f"target at {self._entry_px * (1 + side * tp):,.6g}, {tp:.1%} {'above' if side > 0 else 'below'} it"
                 if tp else "no target"]
        if kind == "edit":
            store.event(name, "info", "exits_applied", "The open position now works to the new settings: "
                        f"{'; '.join(parts)} (entry {self._entry_px:,.6g})"
                        + (f"; 1R is now {risk:,.2f}" if risk else "") + ".", ts=now)
        else:
            store.event(name, "warning", "stop_reset", "Stop for the open position set again from the market after "
                        f"a restart: {'; '.join(parts)} (entry {self._entry_px:,.6g}).", ts=now)

    def on_historical_bars(self, bars) -> None:
        for bar in sorted(bars, key=lambda b: b.ts_event):
            if self._accept(bar):
                self._replay(bar)
        self._finish_resume()
        self.log.info(f"warmed up on {len(bars)} historical bars")

    def resume_leg(self, side: int, held: int) -> None:
        """After a restart: put the model's rules back on the leg the journal's last entry opened (+1 long, -1
        short), as they stood `held` decision bars after the entry's own bar. The warm-up bars since are then
        decided on again, without trading (_replay), so the leg ends, its time stop falls and an exit lock
        clears where they would have without the restart (review round 13, E13-3/E13-4). Models whose rules
        keep a leg override this; for any other model a restart rebuilds nothing beyond its indicators."""

    def _plan_resume(self) -> None:
        """After a restart: read the journal's last entry (its side and the bar it was decided on) and the first
        exit after it that locked re-entry. The journal is the only record of them; nothing else is kept."""
        self._resume = None
        if (self.runtime is None or self.runtime.backtest
                or type(self).resume_leg is LongFlatStrategy.resume_leg):
            return
        step = bar_minutes(self._cfg.bar_type) * MINUTE_NS
        orders = self.runtime.store.orders(self.runtime.name, limit=1000)  # newest first
        at = next((i for i, o in enumerate(orders) if o["intent"] == "entry"), None)
        qty = self.runtime.book["qty"]
        held = {"side": 1 if qty > 0 else -1, "bar": None, "step": step, "lock_ns": None, "on": False} if qty else None
        if at is None:  # held with no entry in the journal's recent orders: that leg is on, from now
            self._resume = held
            return
        entry = orders[at]
        side = 1 if entry["side"] == "BUY" else -1
        lock = next((o for o in reversed(orders[:at]) if o["intent"] in LOCKING_INTENTS), None)
        if qty * side <= 0 and lock is None:
            # Not held on the entry's side, and no stop, target or liquidation closed it: its own signal, a PM
            # close or a flatten ended the leg, and the exit may be older than the warm-up, so resuming it
            # would wake the model in a trade the book doesn't hold. Only the book's own position comes back.
            self._resume = held
            return
        # Orders are stamped when sent, to the second, just after the close of the bar that decided them.
        self._resume = {"side": side, "bar": _ns(entry["ts"]) // step * step,
                        "step": step, "lock_ns": _ns(lock["ts"]) if lock is not None else None, "on": False}

    def _replay(self, bar: Bar) -> None:
        """A warm-up bar after a restart, closed since the journal's last entry: decide on it as the run did, for
        the model's state only. No order, no journal; the exit lock is set and cleared as on_bar would."""
        r = self._resume
        if r is None or r["bar"] is None or bar.ts_event <= r["bar"]:
            return
        if r["lock_ns"] is not None and bar.ts_event > r["lock_ns"]:
            self._lock_resumed(r)
        if not r["on"]:  # until the model's indicators are ready to decide, the leg's bars count on
            self.resume_leg(r["side"], int((bar.ts_event - r["bar"]) // r["step"]) - 1)
        if self._margin:
            side = self.want_side(bar)
            if side is None:
                return
            side = 0 if int(side) < 0 and not self._cfg.allow_short else int(side)
            if self._exit_lock and self._exit_lock != side:
                self._exit_lock = False
        else:
            raw = self.target_weight(bar)
            if raw is None:
                return
            if min(max(float(raw), 0.0), 1.0, self._cap_pct()) == 0:
                self._exit_lock = False
        r["on"] = True

    def _lock_resumed(self, r: dict) -> None:
        self._exit_lock = r["side"] if self._margin else True
        r["lock_ns"] = None

    def _finish_resume(self) -> None:
        """The warm-up is in: a leg no warm-up bar was decided on resumes at the bars counted to the next one,
        and an exit lock set after the last warm-up bar is set now."""
        r, self._resume = self._resume, None
        if r is None:
            return
        if r["lock_ns"] is not None:
            self._lock_resumed(r)
        if not r["on"]:
            if r["bar"] is None:
                self.resume_leg(r["side"], 0)
            else:
                step = r["step"]
                upcoming = self.clock.timestamp_ns() // step * step + step
                self.resume_leg(r["side"], max(int((upcoming - r["bar"]) // step) - 1, 0))

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
            msg = f"Loaded {len(bars)} of {want} warm-up bars from {getattr(self.history_loader, 'source', 'the history store')}"
            if len(bars) < want:  # the venue's recent candles stop short of a long look-back (m13-E4)
                level = "warning"
                short = want - len(bars)
                msg += (f"; {short} short of what the model looks back over, so its indicators are unsettled "
                        f"until {short} more bar{'s' if short != 1 else ''} close")
            step = bar_minutes(self._cfg.bar_type) * 60_000_000_000
            # Bars absent inside the window (minutes the store never got, so no bar was built): the indicators step
            # across them, so the largest hole is named (QA P1-D8).
            gaps = [(a.ts_event, b.ts_event) for a, b in zip(bars, bars[1:]) if b.ts_event - a.ts_event > step]
            if gaps:
                level = "warning"
                a, b = max(gaps, key=lambda g: g[1] - g[0])
                absent = sum((y - x) // step - 1 for x, y in gaps)
                msg += (f"; {absent} bar{'s' if absent != 1 else ''} inside the window {'are' if absent != 1 else 'is'} "
                        f"missing (no stored minutes), the largest the bars closing {_utc(a + step):%d %b %H:%M} to "
                        f"{_utc(b - step):%d %b %H:%M} UTC, and the indicators step across "
                        + ("them" if len(gaps) > 1 else "it"))
            # Bars between the last one loaded and the first live one are a hole the indicators skip.
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

    @classmethod
    def weight_sized(cls, params: dict) -> bool:
        """Whether, with these settings, the model holds a share of the capital below all of it (target_weight)
        rather than all or nothing. A model that overrides target_weight does, unless it says otherwise."""
        return cls.target_weight is not LongFlatStrategy.target_weight

    def want_side(self, bar: Bar) -> int | None:
        """The side to be on from this bar's close: +1 long, 0 flat, -1 short (taken only with allow_short);
        None = not enough data yet. The default is want_long's long or flat."""
        w = self.target_weight(bar)
        return None if w is None else (1 if w > 0 else 0)

    def explain(self, bar: Bar, target: bool) -> tuple[str, dict]:
        """Why want_long() just said `target`: one plain-English sentence and the indicator values
        behind it. Called straight after want_long() on the same bar, so it sees the same state.
        It is journaled with the order, so the reason is the one the strategy acted on. A strategy that
        decides with want_side gets the side (+1, 0 or -1) as `target`."""
        if target is not True and target is not False and target < 0:
            return "Signal to be short", {}
        return ("Signal to be long" if target else "Signal to be flat"), {}

    def conditions(self, side: int, price: float | None = None) -> list[Condition] | None:
        """The model's rules for `side` (+1 long, -1 short) on this bar, for the strategy page's Signals tab:
        those that would all have to hold for the model's decision to be that side, or, while the model is on
        that side, the ones that end it (Condition.exit). price: the forming candle's would-be close, worked
        out on a copy so nothing the model decides with changes; None reads the last closed bar. An empty
        list: the model can't say yet (its indicators are warming up). None: this model doesn't list its
        conditions. Overriding models build them in the same code their decision uses."""
        return None

    def guard_conditions(self, price: float) -> list[Condition]:
        """The open position's stop-loss and take-profit at this price, as _check_exits judges them: either
        one alone exits, at once rather than at the bar's close."""
        if self._entry_px is None or not price or price <= 0:
            return []
        side = self._entry_side or 1
        gain = side * (price / self._entry_px - 1)  # what the position made: a short gains as the price falls
        rows = []
        stop, tp = self._stop_frac, self._tp_frac
        if stop is not None:
            span = max(abs(stop) * 2, 0.002) * 100
            rows.append(Condition(f"Stop-loss {_from_entry(stop, side)}", round(gain * 100, 4), round(-stop * 100, 4),
                                  "<=", gain <= -stop, "%", -span, span, exit=True,
                                  note=f"stop at {self._entry_px * (1 - side * stop):,.6g}"))
        if tp:
            span = max(abs(tp) * 2, 0.002) * 100
            rows.append(Condition(f"Take-profit {tp:.1%} {'above' if side > 0 else 'below'} the entry",
                                  round(gain * 100, 4), round(tp * 100, 4), ">=", gain >= tp, "%", -span, span,
                                  exit=True, note=f"target at {self._entry_px * (1 + side * tp):,.6g}"))
        return rows

    def signal_state(self, price: float | None = None) -> dict | None:
        """What the Signals tab shows, as JSON-ready data: both sides' conditions on the forming candle at
        `price` (the latest price when None), and the open position's stop and target. None when the model
        doesn't list its conditions. Reads only: nothing the model trades by changes."""
        if type(self).conditions is LongFlatStrategy.conditions:
            return None
        price = price if price is not None else self._price()
        sides = {}
        for side, key in ((1, "long"), (-1, "short")):
            rows = self.conditions(side, price if price > 0 else None)
            sides[key] = None if rows is None else [_condition_json(r) for r in rows]
        held = self._entry_side if self._entry_px is not None else 0
        return {"price": price, "bar_ts": self._last_bar_ts or None, "bar_minutes": bar_minutes(self._cfg.bar_type),
                **sides, "held": held, "guards": [_condition_json(r) for r in self.guard_conditions(price)]}

    def _publish_signals(self) -> None:
        """Paper: write the model's conditions on the forming candle for the Signals tab, at most every
        SIGNAL_WRITE_EVERY_NS and straight after a new bar. Display only: never in a backtest, worked out on
        copies of the model's state, and a failure here is logged once and never reaches the order path."""
        if not self._signals_on or self.runtime is None or self._backtest:
            return
        now = self.clock.timestamp_ns()
        if self._signals_bar == self._last_bar_ts and now - self._signals_ns < SIGNAL_WRITE_EVERY_NS:
            return
        self._signals_ns, self._signals_bar = now, self._last_bar_ts
        try:
            state = self.signal_state()
            if state is None:
                self._signals_on = False
                return
            self.runtime.publish_signals(state)
            self._signals_warned = False
        except Exception as exc:  # noqa: BLE001 - the tab goes stale and says so; trading carries on
            if not self._signals_warned:
                self._signals_warned = True
                self.log.warning(f"Signals tab: couldn't write the model's conditions; it tries again: {exc!r}")

    def _on_bar_sided(self, bar: Bar) -> None:
        """A perpetual's decision: be long, short or flat, all or nothing. Turning from one side to the
        other closes first, and the new side opens once the close has filled (on_order_filled)."""
        side = self.want_side(bar)
        if side is None:
            return
        side = int(side)
        if side < 0 and not self._cfg.allow_short:
            side = 0
        if self._exit_lock and self._exit_lock != side:
            self._exit_lock = False  # the signal has moved off the side a stop or target closed
        if self._busy() or self._pending_exit is not None:
            return  # the last decision is still being carried out
        current = self._pos_side()
        if side == current:
            return
        close = bar.close.as_double()
        reason, values = self.explain(bar, side)
        values = {**values, "close": close}
        if current != 0:
            self._flip = (side, bar, reason, values) if side != 0 else None
            self._sell_all("exit", reason, values)
            return
        self._open(side, bar, reason, values)

    def _open(self, side: int, bar: Bar, reason: str, values: dict) -> None:
        """Open a position on a perpetual, all of it at market: long (side 1) or short (-1). Sized by the
        smallest of the leverage cap, the risk profile's position cap, the largest order cap, the risk per
        trade and the bar's volume; refused when its stop would sit too near the liquidation price."""
        if self._exit_lock == side:
            return
        if self._entry_blocked(bar):
            return
        if self.runtime is not None and not self.runtime.can_open():
            return
        if self._safety_stop:
            if self._entry_px is not None or self._exits_only:
                self._note("entry_held_safety_stop", "Entry held back: started for its exits only" if self._exits_only
                           else "Entry held back: the open position works to a safety stop until the model's own stop "
                           "is set again after the restart", level="info")
                return
            self._safety_stop = False  # the position it guarded has closed
        close = bar.close.as_double()
        # The position still held (a flip whose close hasn't filled, or an add) counts towards open risk too (QA P1-S8).
        held_qty = self._net_position()[0] if self.cache is not None and self.instrument is not None else 0.0
        held_stop = (self._entry_px * (1 - (self._entry_side or 1) * self._stop_frac)
                     if held_qty and self._entry_px and self._stop_frac is not None else None)
        if self._has_exits:
            plan = self._plan_exits(close, side)
            if plan is None:
                self._note("stop_not_ready", "Entry held back: the stop is set from the last "
                           f"{self._cfg.atr_bars} bars, and there aren't that many yet", level="info")
                return
            self._noted.discard("stop_not_ready")
            self._stop_frac, self._tp_frac, self._stop_basis = plan
            leg = self._round_trip_cost()
            if self._tp_frac and gain_at_target(self._tp_frac, leg, side) <= 0:
                self._note("target_below_costs", f"No take-profit on this entry: the {self._tp_frac:.2%} target "
                           "doesn't cover the round trip's fees and spread at the venue's spread now")
                self._tp_frac = None
            else:
                self._noted.discard("target_below_costs")
        equity, cash, _, _ = self._mark()
        if equity <= 0:
            self.log.warning("no equity to size from yet; skipping entry")
            return
        fee = Decimal(str(self._cfg.assumed_taker_fee))
        room = Decimal(1) - Decimal(str(self._cfg.cash_buffer)) - fee
        lev = self.runtime.profile.max_leverage if self.runtime is not None else 1.0
        limits = {f"{lev:g}x leverage cap": Decimal(str(equity * lev)) * room}
        if self.runtime is not None:
            limits[f"{self.runtime.profile.name} risk profile cap"] = Decimal(str(self.runtime.position_budget(equity)))
        elif self._cfg.position_cap_pct is not None:
            limits["risk profile cap"] = Decimal(str(equity * self._cfg.position_cap_pct))
        if self._cfg.max_notional is not None:
            limits["largest order cap"] = Decimal(str(self._cfg.max_notional))
        if self._cfg.risk_per_trade and self._stop_frac:
            limits["risk per trade"] = Decimal(str(equity * self._cfg.risk_per_trade / self._loss_at_stop(side)))
        if (cap := self._volume_cap(bar)) is not None:
            limits["share of the bar's volume"] = cap
        size_by = min(limits, key=limits.get)
        budget = limits[size_by]
        qty = (budget / bar.close.as_decimal()).quantize(self._lot(), rounding=ROUND_DOWN)
        if qty <= 0 or qty < self._min_qty():
            self._note("buy_skipped", f"Entry skipped: the {size_by} limit ({float(budget):,.2f}) comes to {qty}, "
                       f"below the smallest order the venue takes ({self._min_qty()})")
            return
        self._noted.discard("buy_skipped")
        signal = {**values, "close": close, "side": _side_word(side), "sized_by": size_by,
                  "budget": round(float(budget), 2)}
        notional = float(qty) * close
        liq, distance = entry_liquidation(cash, float(qty), close, side, self._cfg.assumed_taker_fee,
                                          self._cfg.perp.maintenance_margin, lev)
        if liq is not None:
            signal["liquidation_px"] = round(liq, 8)
            signal["leverage"] = round(notional / equity, 4)
            share = self.runtime.profile.stop_to_liquidation if self.runtime is not None else 0.5
            if self._stop_frac and self._stop_frac > share * distance:
                self._note("entry_refused_liquidation",
                           f"Entry refused: its {self._stop_frac:.2%} stop is more than {share:.0%} of the way to the "
                           f"liquidation price {liq:,.6g} ({distance:.2%} away)")
                return
            self._noted.discard("entry_refused_liquidation")
        if self._stop_frac:
            loss = self._loss_at_stop(side)
            signal["risk_amount"] = round(notional * loss, 2)
            signal["stop_frac"] = round(self._stop_frac, 6)
            if self._cfg.stop_atr:
                signal["stop_basis"] = self._stop_basis
            if self._tp_frac:
                signal["tp_frac"] = round(self._tp_frac, 6)
                signal["planned_r"] = round(gain_at_target(self._tp_frac, self._round_trip_cost(), side) / loss, 2)
        elif self._tp_frac:
            signal["tp_frac"] = round(self._tp_frac, 6)
        if self._has_exits:
            signal["stop_cfg"] = self._stop_cfg()
        if (why := self._open_risk_refusal(side, float(qty), close, equity, bar.ts_event, held_qty, held_stop)) is not None:
            self._note("entry_refused_open_risk", f"Entry refused: {why}")
            return
        self._noted.discard("entry_refused_open_risk")
        self._submit(OrderSide.BUY if side > 0 else OrderSide.SELL, qty, "entry", reason, signal)

    def on_bar(self, bar: Bar) -> None:
        self._snap_settlements(bar.ts_event, bar.close.as_double())
        self._finish_resume()  # warm-up bars asked of the venue that never came: resume without them
        self._resize_if_due()
        if self._exec_type is not None and bar.bar_type == self._exec_type:
            self._on_exec_bar(bar)
            return
        if self._opened_in_bar and str(bar.bar_type) == str(self._cfg.bar_type).split("@")[0]:
            due = [o for o in self._opened_in_bar if o[0][1] <= bar.ts_event]
            self._opened_in_bar = [o for o in self._opened_in_bar if o[0][1] > bar.ts_event]
            for window, qty in due:
                self._fund_opened_in_bar(window, qty, bar.low.as_double(), bar.high.as_double())
        bar = self._first_bar_from_store(bar)
        if self._count_minutes(bar):
            return
        if self._hold_gap(bar):
            return
        bar = self._fill_gap(bar)
        if not self._accept(bar):
            return
        missing = self._degraded.pop(bar.ts_event, None)
        if missing is None:
            self._noted.discard("degraded_bar")
        else:
            self._no_entry_ts, self._degraded_missing = bar.ts_event, missing
        self.log.info(f"bar {bar}")
        self._last_close = bar.close.as_double()
        if self._margin and self._backtest and self._exec_type is None:
            # Fed only the bars it decides on, a backtest on a perp still judges each bar at its worst price
            # (_on_exec_bar does it for every shorter execution bar), as paper judges every trade.
            self._intrabar_guard(bar)
        self._maybe_tick()
        if self._exec_type is None:
            self._rest_risk_stop()
        if self._pending_exit is not None:
            return
        if self._replan_pending is not None and self._entry_px is not None:
            self._replan(self._last_close)
        if self._check_exits(self._last_close):
            return
        if self._margin:
            if self._restore is None:
                self._apply_funding(self._last_close)
                self._on_bar_sided(bar)
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
            if self._exit_lock or self._entry_blocked(bar):
                return
            if self.runtime is not None and not self.runtime.can_open():
                return
            if self._has_exits:
                plan = self._plan_exits(close)
                if plan is None:
                    self._note("stop_not_ready", "Entry held back: the stop is set from the last "
                               f"{self._cfg.atr_bars if self._cfg.stop_atr else self._cfg.stop_swing_bars} bars, "
                               "and there aren't that many yet", level="info")
                    return
                self._noted.discard("stop_not_ready")
                self._stop_frac, self._tp_frac, self._stop_basis = plan
                leg = self._round_trip_cost()
                if self._tp_frac and self._tp_frac <= leg + (1 + self._tp_frac) * leg:
                    # A fixed target checked against the assumed spread can still fall short when the
                    # venue's spread is wider now; a hit would lose money, so this trade goes without one.
                    self._note("target_below_costs", f"No take-profit on this entry: the {self._tp_frac:.2%} target "
                               "doesn't cover the round trip's fees and spread at the venue's spread now")
                    self._tp_frac = None
                else:
                    self._noted.discard("target_below_costs")
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
                if w > self._held_w and self._entry_blocked(bar):
                    return  # adding to the position is an entry; trimming it still runs
                reason, values = self.explain(bar, True)
                if self._rebalance(bar, w, reason, {**values, "target_weight": round(float(raw), 6), "close": close}):
                    self._held_w = w

    def _cap_pct(self) -> float:
        if self.runtime is not None:
            return float(self.runtime.cap)
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
        step = self._lot()
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
        return self._submit(side, size, "rebalance", reason, signal)

    def _check_exits(self, price: float) -> bool:
        """Stop-loss / take-profit against the average entry. True if an exit was sent.

        Paper and live call this on every trade, so both levels are watched tick by tick. A
        backtest only sees whole bars, so both exits rest at the venue instead (_rest_exits)."""
        if self._safety_pending is not None and self._entry_px is not None and price > 0:
            self._place_safety_stop(price)
        stop, tp = self._stop_frac, self._tp_frac
        if self._entry_px is None or (stop is None and not tp) or price <= 0:
            return False
        if self._busy():
            return False
        if self._backtest:
            return False  # both exits rest at the venue (_rest_exits) and fill at their levels
        if self._restore is not None:
            return False  # the position isn't back at the simulated venue yet
        side = self._entry_side or 1
        move = price / self._entry_px - 1
        gain = side * move  # what the position made: a short gains as the price falls
        hit = ("stop_loss" if stop is not None and gain <= -stop else "take_profit" if tp and gain >= tp else None)
        if hit is None:
            return False
        self.log.info(f"{hit} at {price} ({move:+.2%} from entry {self._entry_px})")
        if self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "info", hit, f"exit at {price:,.4f}, {move:+.2%} from entry",
                                     ts=self.runtime.now())
        level = stop if hit == "stop_loss" else tp
        reason = (f"{'Stop-loss' if hit == 'stop_loss' else 'Take-profit'}: price {price:,.6g} is {move:+.2%} from "
                  f"the {self._entry_px:,.6g} entry, past the "
                  + (f"stop {_from_entry(level, side)}" if hit == "stop_loss" else
                     f"{level:.1%} target" + (" (below it, for a short)" if side < 0 else "")))
        values = {"entry_px": self._entry_px, "move": move, hit: level}
        if hit == "stop_loss" and self._margin:
            _, cash, qty, _ = self._mark()
            liq = self._liq(cash, qty) if qty else None
            if liq is not None:
                values["liquidation_px"] = round(liq, 8)  # GAP-LIQ: judged on its fill (_gap_liquidation)
        self._exit_lock = side if self._margin else True
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
        step = self._lot()
        qty = (budget / bar.close.as_decimal()).quantize(step, rounding=ROUND_DOWN)
        min_qty = self._min_qty()
        if qty <= 0 or qty < min_qty:
            self._note("buy_skipped", f"Buy skipped: the {size_by} limit ({float(budget):,.2f}) buys {qty}, below "
                       f"the smallest order the venue takes ({min_qty})")
            return
        self._noted.discard("buy_skipped")
        signal = {**(values or {}), "close": bar.close.as_double(), "sized_by": size_by,
                  "budget": round(float(budget), 2)}
        if self._stop_frac:
            # What this position loses if the stop is hit, costs included: one R, for the trade's R multiple.
            loss = self._loss_at_stop()
            signal["risk_amount"] = round(float(qty * bar.close.as_decimal()) * loss, 2)
            signal["stop_frac"] = round(self._stop_frac, 6)
            if self._cfg.stop_atr or self._cfg.stop_swing_bars:
                signal["stop_basis"] = self._stop_basis
            if self._tp_frac:  # what the target makes, after the same costs, in R
                tp, cost = self._tp_frac, self._round_trip_cost()
                signal["tp_frac"] = round(tp, 6)
                signal["planned_r"] = round((tp - cost - (1 + tp) * cost) / loss, 2)
        elif self._tp_frac:
            signal["tp_frac"] = round(self._tp_frac, 6)
        if self._has_exits:
            signal["stop_cfg"] = self._stop_cfg()
        self._submit(OrderSide.BUY, qty, "entry", reason, signal)

    def _loss_at_stop(self, side: int = 1) -> float:
        """The share of a position's cost lost if its stop is hit: the stop distance, plus the taker fee
        and half the spread to buy, plus the same on what is left to sell. A 6% stop at 0.8% taker and a
        0.05% half spread loses 7.65%, not 6%. A short's stop buys back above the entry (loss_at_stop)."""
        cost, stop = self._round_trip_cost(), self._stop_frac if self._stop_frac is not None else self._cfg.stop_loss
        return loss_at_stop(stop, cost, side)

    def _round_trip_cost(self) -> float:
        """Each leg's cost as a share of its notional: the taker fee and half the spread."""
        return self._cfg.assumed_taker_fee + self._half_spread()

    def _half_spread(self) -> float:
        if self._bid is not None and self._ask is not None and self._bid > 0 and self._ask >= self._bid:
            return (self._ask - self._bid) / (self._ask + self._bid)
        return self._cfg.assumed_half_spread

    def _submit(self, side, qty: Decimal, intent: str, reason: str, signal: dict, market: bool = False) -> bool:
        """Send an order, journaling it with its reason first so the record exists before the venue sees
        it. With maker_wait_minutes set, signal-driven orders rest as post-only limits first. An order that would
        open or add while nothing may (CHOKE) is not sent, and says why once; returns whether it was sent."""
        if (why := self._gated(side, intent)) is not None:
            self._note("entry_gated", f"Order not sent: it would open or add to the position, and {why}. Nothing "
                       "opens until that clears; stops and closes still run")
            return False
        if intent in OPENING_INTENTS:
            self._noted.discard("entry_gated")
        quantity = Quantity.from_decimal_dp(qty, self.instrument.size_precision)
        wait = self._cfg.maker_wait_minutes
        last, tick = self._price(), self.instrument.price_increment.as_double()
        signal = {k: (round(v, 8) if isinstance(v, float) else v) for k, v in signal.items()}
        signal.setdefault("price", last)
        maker = bool(wait and not market and intent in MAKER_INTENTS and last > 2 * tick)
        # On a perp every exit is reduce-only: whatever the strategy's own book says, the venue never lets an
        # exit open the other side (review round 11, B11-3).
        reduce = self._margin and intent not in OPENING_INTENTS
        if maker:
            px = Price(self._maker_price(side, last, tick), self.instrument.price_precision)
            order = self.order_factory.limit(instrument_id=self._cfg.instrument_id, order_side=side, quantity=quantity,
                                             price=px, time_in_force=TimeInForce.GTC, post_only=True,
                                             reduce_only=reduce)
            signal.update(order_type="maker", limit_px=px.as_double(), maker_wait_minutes=wait)
        else:
            order = self.order_factory.market(instrument_id=self._cfg.instrument_id, order_side=side,
                                              quantity=quantity, time_in_force=TimeInForce.GTC, reduce_only=reduce)
            if wait:
                signal["order_type"] = "market"
        coid = str(order.client_order_id)
        self.decisions[coid] = {"intent": intent, "reason": reason, "signal": signal}
        if self.runtime is not None:
            self.runtime.on_order(order_id=coid, side="BUY" if side == OrderSide.BUY else "SELL",
                                  qty=float(qty), intent=intent, reason=reason, signal=signal,
                                  order_type="POST-ONLY LIMIT" if maker else "MARKET")
        if maker and self.simulated_venue:
            self._kept[coid] = {"order": order, "info": self.decisions[coid], "bar": None, "earned": 0.0,
                                "sent": Decimal(0), "inflight": 0}
            if self.runtime is not None:
                self.runtime.on_order_status(coid, "accepted")
        else:
            self._sent.append(order.client_order_id)
            self.submit_order(order)
            if maker:
                self._maker[coid] = {"intent": intent, "reason": reason, "signal": signal}
        if maker:
            self.clock.set_time_alert(f"maker-{coid}", self.clock.utc_now() + timedelta(minutes=wait),
                                      callback=self._maker_timeout)
        return True

    def _maker_price(self, side, last: float, tick: float) -> float:
        """Where a post-only order rests: at the best bid (to buy) or ask (to sell), so it adds liquidity
        rather than taking it. Without quotes (a backtest on bars, or paper before its first quote) the bid
        and ask are estimated as the last trade less or plus the half spread the run assumes, and at least
        a tick away. One rule for paper and backtests: the backtest used to rest a tick inside the last
        trade, nearer than paper's bid, and filled up to 34 points more often (review round 9, M9-3)."""
        if self._bid is not None and self._ask is not None:
            return self._bid if side == OrderSide.BUY else self._ask
        away = max(tick, last * self._cfg.assumed_half_spread)
        steps = math.ceil(away / tick - 1e-9)  # whole ticks, never nearer than the half spread
        return last - steps * tick if side == OrderSide.BUY else last + steps * tick

    def _maker_timeout(self, event) -> None:
        """The post-only order has waited long enough: cancel what is left; the cancel's confirmation
        sends the rest at market (on_order_canceled), so nothing is sold or bought twice."""
        coid = event.name.removeprefix("maker-")
        if coid in self._kept:
            wait = self._cfg.maker_wait_minutes
            self._close_kept(coid, f"was not filled within {wait} minute{'s' if wait != 1 else ''}",
                             at_market=self._pending_exit is None)
            return
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
        return bool(self._maker) or bool(self._kept)

    def _cancel_alert(self, coid: str) -> None:
        name = f"maker-{coid}"
        if name in self.clock.timer_names():
            self.clock.cancel_timer(name)

    def _finish_at_market(self, coid: str, info: dict, why: str, order=None, left: Decimal | None = None) -> None:
        """Send the unfilled rest of a post-only order at market, sized to what the account can do now.
        order and left: a kept order (paper), and what of it was never sent as a slice."""
        order = order or self.cache.order(ClientOrderId(coid))
        if order is None:
            return
        step = self._lot()
        left = order.leaves_qty.as_decimal() if left is None else left
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
        """Paper: build each kept post-only order its own one-minute bars from the trades since it was sent,
        and on each bar's close earn it what a backtest's venue fills from that bar (_earn)."""
        key = (tick.ts_event, str(tick.trade_id))
        if key == self._tape_last:
            return
        self._tape_last = key
        px, size, minute = tick.price.as_double(), tick.size.as_double(), tick.ts_event // MINUTE_NS
        for coid, kept in list(self._kept.items()):
            bar = kept["bar"]
            if bar is not None and bar["minute"] != minute:
                self._earn(coid, kept)
                bar = None
            if bar is None:
                kept["bar"] = {"minute": minute, "prints": [px, px, px, px], "volume": size}
            else:
                p = bar["prints"]
                p[1], p[2], p[3] = max(p[1], px), min(p[2], px), px
                bar["volume"] += size

    def _earn(self, coid: str, kept: dict) -> None:
        """A backtest shows its venue each one-minute bar as four prints (open, high, low, close), each
        carrying a quarter of BOOK_SHARE of the bar's volume, and a resting order fills at its limit from
        each print that trades through it. Paper earns its kept order the same from the same bar, then
        sends what that adds (_slice_maker)."""
        bar, kept["bar"] = kept["bar"], None
        limit = kept["order"].price.as_double()
        through = sum(1 for p in bar["prints"] if ((p < limit) if kept["order"].is_buy else (p > limit)))
        if through:
            kept["earned"] += BOOK_SHARE * bar["volume"] / 4 * through
            self._slice_maker(coid, kept)

    def _slice_maker(self, coid: str, kept: dict) -> None:
        """Paper: send at market the part of a kept post-only order the tape has earned so far (_earn). The
        fee model charges the slice as filled at the limit with the maker fee (ScheduleFeeModel.maker_slices),
        and the journal records it on the post-only order, so paper's fills, fees and position follow the
        backtest's slice for slice."""
        if self._pending_exit is not None:
            return
        order, step = kept["order"], self._lot()
        earned = min(Decimal(str(kept["earned"])), order.quantity.as_decimal())
        qty = earned.quantize(step, rounding=ROUND_DOWN) - kept["sent"]
        limit = order.price.as_double()
        if order.is_buy:
            account = self._account()
            bal = next((b for c, b in account.balances().items() if str(c.code) in self._codes("quote")),
                       None) if account else None
            room = Decimal(1) - Decimal(str(self._cfg.cash_buffer)) - Decimal(str(self._cfg.assumed_taker_fee))
            affordable = bal.free.as_decimal() * room / Decimal(str(limit)) if bal is not None else Decimal(0)
            qty = min(qty, affordable)
        else:
            qty = min(qty, self._position_qty(free=True))
        qty = qty.quantize(step, rounding=ROUND_DOWN)
        if qty <= 0 or qty < self._min_qty():
            return
        piece = self.order_factory.market(instrument_id=self._cfg.instrument_id, order_side=order.side,
                                          quantity=Quantity.from_decimal_dp(qty, self.instrument.size_precision),
                                          time_in_force=TimeInForce.GTC)
        pid = str(piece.client_order_id)
        self._slices[pid] = coid
        self.decisions[pid] = kept["info"]
        if self.fee_model is not None:
            self.fee_model.maker_slices[pid] = (order.price.as_decimal(), order.is_buy)
        kept["sent"] += qty
        kept["inflight"] += 1
        self._sent.append(piece.client_order_id)  # working until it closes, so a stop waits for it
        self.submit_order(piece)

    def _close_kept(self, coid: str, why: str, at_market: bool = False) -> None:
        """Paper: a kept post-only order ends: filled, out of time (the rest goes at market, as a backtest
        sends it), or cancelled by a stop, a flatten, a halt or a shutdown. A slice still in flight finishes
        first, and the order's journal row closes after it (_slice_done)."""
        kept = self._kept.pop(coid, None)
        if kept is None:
            return
        self._cancel_alert(coid)
        kept["why"] = why
        if kept["inflight"]:
            self._closing[coid] = kept
        else:
            self._kept_closed(coid, kept)
        left = kept["order"].quantity.as_decimal() - kept["sent"]
        if at_market and left > 0:
            self._finish_at_market(coid, kept["info"], why, order=kept["order"], left=left)

    def _kept_closed(self, coid: str, kept: dict) -> None:
        if self.runtime is not None and kept["sent"] < kept["order"].quantity.as_decimal():
            self.runtime.on_order_status(coid, "canceled", kept["why"])

    def _slice_done(self, pid: str, unfilled: float = 0.0) -> None:
        """Paper: a slice closed. What it didn't fill (a denial, say) is the kept order's to send again."""
        coid = self._slices.pop(pid)
        if self.fee_model is not None:
            self.fee_model.maker_slices.pop(pid, None)
        kept = self._kept.get(coid) or self._closing.get(coid)
        if kept is None:
            return
        kept["inflight"] -= 1
        if unfilled > 0:
            kept["sent"] -= Decimal(str(unfilled)).quantize(self._lot(), rounding=ROUND_DOWN)
        if coid in self._closing:
            if not kept["inflight"]:
                self._kept_closed(coid, self._closing.pop(coid))
        elif kept["sent"] >= kept["order"].quantity.as_decimal() and not kept["inflight"]:
            self._close_kept(coid, "filled")  # every slice in: the journal has it filled

    def _lot(self) -> Decimal:
        """The smallest size an order carries: the instrument's step, never finer than the account can hold
        the base currency in. A venue can list 8 lot decimals for a currency the engine keeps at 6 (XRP, ADA);
        an 8-decimal buy then leaves 1e-7 behind after a full exit, and reconcile halts (review round 9, B9-1)."""
        return max(self.instrument.size_increment.as_decimal(), Decimal(10) ** -lot_decimals(self.instrument))

    def _held_qty(self, qty: float) -> float:
        """A fill's quantity as the account holds it. A venue can print and fill at 8 lot decimals in a
        currency the engine keeps at 6 (XRP, ADA): the account rounds each fill to 6, so the journal must
        too, or the two drift apart a fraction of a lot per fill until reconcile halts (round 10, B10-1)."""
        base = getattr(self.instrument, "base_currency", None)
        if base is None or self.instrument.size_precision <= base.precision:
            return qty
        return float(Money(qty, base).as_decimal())  # rounded exactly as the account rounds it

    def _min_qty(self) -> Decimal:
        step = self._lot()
        return max(self.instrument.min_quantity.as_decimal(), step) if self.instrument.min_quantity else step

    def _liq(self, cash: float, qty: float) -> float | None:
        """The open position's liquidation price on isolated margin at the risk profile's leverage cap, the
        same margin the sizing, the dashboard and the demo copy use (markets.isolated_margin). On the position's own
        entry, as the journal has it (cash is spot-style on that entry, _mark): after a restart the simulated venue
        holds it at the restart's price, which put the liquidation price far past where it is (GAP-LIQ, line 6)."""
        lev = self.runtime.profile.max_leverage if self.runtime is not None else 1.0
        entry = self._entry_px or self._net_position()[1]
        return markets.isolated_liquidation(cash, qty, entry, lev, self._cfg.perp.maintenance_margin)

    def _net_position(self) -> tuple[float, float]:
        """Perp: the signed position at the simulated venue and its average entry there."""
        net = notional = 0.0
        for p in self.cache.positions_open(instrument_id=self._cfg.instrument_id):
            q = p.signed_qty
            net += q
            notional += q * p.avg_px_open
        return net, (notional / net if net else 0.0)

    def _signed_qty(self) -> Decimal:
        """The position: positive long, negative short. On a perp, summed from the venue's own quantities:
        the float signed_qty reads 0.08838354999999999 for 0.08838355, and an exit rounded down from that
        left one lot behind, so the next entry on the other side counted as a reduction and rested no stop
        (review round 11, B11-2)."""
        if self._margin:
            net = Decimal(0)
            for p in self.cache.positions_open(instrument_id=self._cfg.instrument_id):
                q = p.quantity.as_decimal()
                net += q if p.signed_qty > 0 else -q
            return net
        return self._position_qty()

    def _pos_side(self) -> int:
        """+1 long, -1 short, 0 flat (less than the smallest order the venue takes)."""
        q, step = self._signed_qty(), self._min_qty()
        return 1 if q >= step else -1 if q <= -step else 0

    def _position_qty(self, free: bool = False) -> Decimal:
        """Quantity held, read from the account rather than positions, so a book restored from the
        journal after a restart (a balance with no position object) is still recognised. On a perp,
        the size of the position, long or short."""
        if self._margin:
            return abs(self._signed_qty())
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
        """An order is on its way to the venue or being changed there, so the last decision is still being
        carried out. The backtest's risk stop doesn't count: it is re-priced on most bars, just before the
        decision, and counting its update stopped a perp backtest from ever deciding again."""
        risk_stop = self._risk_stop_id
        return (any(str(o.client_order_id) != risk_stop for o in self.cache.orders_inflight(strategy_id=self.strategy_id))
                or any(str(o.client_order_id) != risk_stop for o in self._unsent()))

    def _is_long(self) -> bool:
        if self._margin:
            return self._pos_side() > 0
        return self._position_qty() >= self._min_qty()

    def _sell_all(self, intent: str = "exit", reason: str = "Signal to be flat", values: dict | None = None) -> None:
        self._drop_kept(earn=self._pending_exit is None)
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
        step = self._lot()
        # Spot can't sell more than the free balance, so round down; a perp's position is already whole lots.
        qty = self._position_qty(free=True).quantize(step, rounding=ROUND_HALF_EVEN if self._margin else ROUND_DOWN)
        if qty <= 0 or qty < self._min_qty():
            return
        # A short is closed by buying it back.
        side = OrderSide.BUY if self._margin and self._pos_side() < 0 else OrderSide.SELL
        self._submit(side, qty, intent, reason, dict(values or {}))

    # --- perpetuals: restore, funding, liquidation and intrabar guards ---------------

    def _send_restore(self) -> None:
        """Paper on a perp, after a restart: put the journal's position back at the simulated venue with
        one market order, at no fee and unjournaled. The journal keeps the real entry; _cash_adj takes up
        the difference from today's price when it fills (on_order_filled)."""
        r = self._restore
        if r is None or self._restore_id is not None or self.instrument is None:
            return
        qty = Decimal(repr(abs(r["qty"]))).quantize(self._lot(), rounding=ROUND_HALF_EVEN)  # a float sum, not floored
        if qty < self._min_qty():
            self._restore = None
            return
        order = self.order_factory.market(instrument_id=self._cfg.instrument_id,
                                          order_side=OrderSide.BUY if r["qty"] > 0 else OrderSide.SELL,
                                          quantity=Quantity.from_decimal_dp(qty, self.instrument.size_precision),
                                          time_in_force=TimeInForce.GTC, tags=["restore"])
        self._restore_id = str(order.client_order_id)
        if self.fee_model is not None:
            self.fee_model.free_orders.add(self._restore_id)
        if self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "info", "restore_position",
                                     f"Putting the {_side_word(1 if r['qty'] > 0 else -1)} position of {abs(r['qty']):.12g} "
                                     "back at the simulated venue after the restart (no fee, not a trade)",
                                     ts=self.runtime.now())
        self.submit_order(order)

    HOUR_NS = 3_600_000_000_000
    HELD_KEPT = 72  # hours of positions at settlement kept: a paper restart settles longer gaps at the tick

    def _snap_settlements(self, event_ns: int, event_close: float | None = None, qty: float | None = None) -> None:
        """Perp: before an event (a trade, a bar or a tick) changes anything, note the position and price held at
        each hour's start since the last event: nothing traded or filled in between, so they are what was held
        at that instant. The price is the last trade before the hour: a bar closing on the hour carries it as its
        close; a trade at or after the hour leaves the one before it (so paper and backtest mark alike). A fill
        passes `qty`, the position before it: the venue's position already includes the fill."""
        if not self._margin:
            return
        last, self._snap_ns = self._snap_ns, max(event_ns, self._snap_ns or 0)
        if last is None or event_ns <= last:
            return
        hour = (last // self.HOUR_NS + 1) * self.HOUR_NS
        if hour > event_ns:
            return
        qty = self._net_position()[0] if qty is None else qty
        prev = self._last_close or self._price()
        while hour <= event_ns:
            px = event_close if (hour == event_ns and event_close is not None) else prev
            self._held_at[hour] = (qty, px)
            hour += self.HOUR_NS
        if len(self._held_at) > self.HELD_KEPT:
            for k in sorted(self._held_at)[:-self.HELD_KEPT]:
                del self._held_at[k]

    def _settlements(self, terms, since: datetime, now: datetime) -> tuple[list[datetime], object]:
        """The settlements in (since, now], and the venue's settled rates they came from (None without): the
        venue's own times where its terms name a venue with settled rates, else the profile's fixed hours
        (markets.settlement_times)."""
        settled = None
        if terms.funding_venue is not None:
            from sleeve_fund import funding

            if self._backtest:
                if self._settled is None:
                    self._settled = funding.rates(terms.funding_venue, pair_of(self.instrument))
                settled = self._settled
            else:
                settled = funding.rates(terms.funding_venue, pair_of(self.instrument))
        return markets.settlement_times(since, now, terms.funding_hours, settled), settled

    # Paper rescans this far back, so a settlement the venue publishes late, at a time the schedule didn't
    # foresee (a change of interval), is still charged when its record arrives.
    FUNDING_LOOKBACK = timedelta(hours=1)

    def _apply_funding(self, price: float) -> None:
        """Exchange the perp's funding for every settlement since the last one charged: a long pays position x
        price x rate, a short receives it (negative rates the other way). The settlements are the venue's own
        (QA P1-O1), and each charges the position held at that instant and the last trade price before it
        (_held_at, QA P1-O2), in paper as in backtest. Booked to cash and journaled, so equity, the journal and
        reconciliation all carry it."""
        terms = self._cfg.perp
        if terms is None or price <= 0:
            return
        now = self.clock.utc_now()
        since = self._funding_since
        if since is None:
            self._funding_since = now
            return
        if int(now.timestamp()) // 3600 * 3600 <= since.timestamp():
            return  # no hour's start in (since, now]: settlements fall on the hour
        times, settled = self._settlements(terms, since, now)
        held = [(ts, *self._held_at.get(int(ts.timestamp()) * 1_000_000_000, (self._net_position()[0], price)))
                for ts in times]
        if not any(q for _, q, _ in held):
            self._funding_since = self._rescan_from(now)
            return
        for ts, qty, px in held:
            inside = self._intrabar is not None and self._intrabar[0] < int(ts.timestamp()) * 1_000_000_000 <= self._intrabar[1]
            if qty == 0 or (inside and self._intrabar[2]):  # a gap fill at the bar's open held nothing after it
                self._funding_since = ts
                continue
            rate = self._funding_rate(terms, ts, now, markets.settlement_wait(ts, settled, self.FUNDING_WAIT))
            if rate is None:  # paper, just after a settlement the venue hasn't published yet: try on the next tick
                return
            self._funding_since = ts
            amount = -qty * px * rate
            if inside and amount > 0:
                continue  # a touch at an unknown time inside the bar: a credit it may not have been held for isn't booked
            self._book_funding(ts, qty, px, rate, amount)
        self._funding_since = max(self._funding_since, self._rescan_from(now))

    def _book_funding(self, ts: datetime, qty: float, px: float, rate: float, amount: float) -> None:
        self._cash_adj += amount
        self.funding_log.append((ts, amount))
        if self.runtime is not None:
            self.runtime.store.record_funding(self.runtime.name, qty=qty, price=px, rate=rate,
                                              amount=round(amount, 8), ts=ts)
            self.runtime.store.event(self.runtime.name, "info", "funding",
                                     f"Funding {'received' if amount >= 0 else 'paid'}: {abs(amount):,.2f} on a "
                                     f"{_side_word(1 if qty > 0 else -1)} position of {abs(qty):.12g} at "
                                     f"{px:,.6g} ({rate:.4%})", ts=ts)

    def _fund_opened_in_bar(self, window: tuple[int, int, bool], qty: float, low: float, high: float) -> None:
        """Rule (c) for a resting entry or add a bars-only backtest filled inside a bar (Independent Quant Advisor, 6 Oct
        17:57): what it opened is charged the settlements inside the bar after its fill. Filled on a gap, it was held
        from the bar's open, so it pays or receives each; touched at an unknown time inside, it takes the worse
        outcome, paying those that cost it and booking no credit. The price at each is unknown inside the bar, so it
        takes the worse of the bar's low and high. Called with the bar, once it has closed. (The settlements before
        the fill charged the position held until then: _apply_funding.)"""
        terms = self._cfg.perp
        if terms is None or not qty:
            return
        lo, hi, gap = window
        since, now = (datetime.fromtimestamp(t / 1e9, tz=timezone.utc) for t in (lo, hi))
        times, settled = self._settlements(terms, since, now)
        for ts in times:
            if not lo < int(ts.timestamp()) * 1_000_000_000 <= hi:
                continue
            rate = self._funding_rate(terms, ts, now, markets.settlement_wait(ts, settled, self.FUNDING_WAIT))
            if rate is None:
                continue
            px = high if qty * rate > 0 else low  # the price at which it costs most, or credits least
            amount = -qty * px * rate
            if gap or amount < 0:
                self._book_funding(ts, qty, px, rate, amount)

    def _intrabar_fill(self, event) -> tuple[int, int, bool] | None:
        """A bars-only backtest fills a resting order somewhere inside the bar, stamped at its close, at an unknown
        time (Independent Quant Advisor, rule (c), QA P1-D9): one filled on a gap took the bar's open price, so it
        is stamped at the open and pays no settlement inside the bar; one touched inside it takes the worse outcome,
        paying a settlement inside the bar that costs the position and booking no credit. The 1-minute backtest
        stays the reference: a bars-only result differs from it only for the worse. None for anything else."""
        if not self._backtest or self._exec_type is not None:
            return None
        order = self.cache.order(event.client_order_id)
        if order is None or order.order_type == OrderType.MARKET:
            return None
        px, buy = event.last_px.as_double(), order.side == OrderSide.BUY
        if order.order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT, OrderType.MARKET_IF_TOUCHED):
            level = order.trigger_price.as_double()
            gap = px > level if buy else px < level
        else:
            level = order.price.as_double()
            gap = px < level if buy else px > level
        step = bar_minutes(self._cfg.bar_type) * MINUTE_NS
        return event.ts_event - step, event.ts_event, gap

    def _rescan_from(self, now: datetime) -> datetime:
        """Where the next charge looks from once everything up to `now` is settled: `now`, except in paper on a
        venue's settled rates, which looks back FUNDING_LOOKBACK for a settlement published late."""
        terms = self._cfg.perp
        if self._backtest or terms is None or terms.funding_venue is None:
            return now
        return max(self._funding_since, now - self.FUNDING_LOOKBACK)

    # Paper waits this long after a settlement for the venue to publish its rate before charging the baseline.
    FUNDING_WAIT = timedelta(minutes=15)

    def _funding_rate(self, terms, ts, now, wait: timedelta | None = None) -> float | None:
        """The rate settled at `ts`: the venue's own where its terms name one (sleeve_fund.funding), else the
        terms' fixed rate. A settlement the venue's records lack is charged the fixed rate as a fallback, said
        once per run; paper first waits FUNDING_WAIT for the venue to publish it (None: not yet)."""
        if terms.funding_venue is None:
            return terms.funding_rate
        import pandas as pd

        from sleeve_fund import funding

        pair = pair_of(self.instrument)
        when = pd.Timestamp(ts)
        series = funding.rates(terms.funding_venue, pair)
        rate = funding.rate_at(series, when)
        if rate is None and not self._backtest:
            # Paper asks the venue directly (the history service keeps the store, which paper only reads).
            try:
                rate = funding.rate_at(funding.fetch(terms.funding_venue, pair, when - funding.MATCH), when)
            except Exception as exc:  # noqa: BLE001 - the venue unreachable: wait, then the fallback
                self.log.warning(f"funding rates unavailable: {exc!r}")
            if rate is None and now - ts < (wait or self.FUNDING_WAIT):
                return None
        if rate is None:
            if not self._funding_fallback_said:
                self._funding_fallback_said = True
                if self.runtime is not None:
                    self.runtime.store.event(self.runtime.name, "warning", "funding_fallback",
                                             f"No settled funding rate from the venue for {pair} at {when:%d %b %Y %H:%M} "
                                             f"UTC; charged the {terms.funding_rate:.4%} baseline instead (said once)",
                                             ts=ts)
            return terms.funding_rate
        return rate

    def _risk_level(self) -> tuple[float, str] | None:
        """Backtest on a perp: the nearest price, from here, at which the risk guard or the liquidation cut
        acts on the open position, and which one: the drawdown halt (equity falls to the peak less the
        profile's drawdown), the daily-loss pause (to the day's opening equity less its daily loss), or the
        cut (the price within the profile's distance of liquidation). None when flat or already past."""
        rt = self.runtime
        equity, cash, qty, price = self._mark()
        if qty == 0 or equity <= 0 or price <= 0:
            return None
        p, side = rt.profile, (1 if qty > 0 else -1)
        peak = max(rt.peak, equity)
        day_open = rt._day_open or equity
        # The next bar opens a new UTC day: the day then opens at this equity (SleeveRuntime.tick).
        step = timedelta(minutes=bar_minutes(self._exec_type or self._cfg.bar_type))
        now = self.clock.utc_now()
        if risk.trading_day(now + step) != risk.trading_day(now):
            day_open = equity
        levels = [((peak * (1 - p.max_drawdown) - cash) / qty, "risk_halt"),
                  ((day_open * (1 - p.daily_loss) - cash) / qty, "risk_pause")]
        liq = self._liq(cash, qty)
        if liq is not None:
            d = p.min_liquidation_distance
            levels.append((liq / (1 - d) if side > 0 else liq / (1 + d), "liquidation_cut"))
        ahead = [(lv, k) for lv, k in levels if lv > 0 and (lv < price if side > 0 else lv > price)]
        if not ahead:
            return None
        return max(ahead) if side > 0 else min(ahead)

    def _rest_risk_stop(self) -> None:
        """Backtest on a perp: keep a reduce-only stop for the whole position resting at the price the risk
        guard would act at (_risk_level), re-priced each bar. Paper judges every trade and closes on the one
        that breaches; judged at a minute's worst price but closed at its close, a backtest's risk exit was
        kinder by the wick. The stop fills at its level, or at the open if the price gaps through it."""
        if not (self._backtest and self._margin and self.runtime is not None) or self.instrument is None:
            return
        stop = self._resting_exits().get("stop_loss") if self._entry_side else None
        if stop is not None:  # GAP-LIQ: the resting stop is judged against the liquidation price as of this bar
            self.decisions[str(stop.client_order_id)]["signal"].update(self._liq_signal())
        order = self.cache.order(ClientOrderId(self._risk_stop_id)) if self._risk_stop_id else None
        if order is not None and order.is_closed:
            order, self._risk_stop_id = None, None
        want = None
        if self.runtime.status == "running" and self._pending_exit is None and self._flip is None:
            want = self._risk_level()
        qty = self._position_qty().quantize(self._lot(), rounding=ROUND_HALF_EVEN)
        if want is None or qty < self._min_qty():
            if order is not None and order.status in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                self.cancel_order(order.client_order_id)
            return
        level, intent = want
        px = Price(level, self.instrument.price_precision)
        quantity = Quantity.from_decimal_dp(qty, self.instrument.size_precision)
        _, cash, net, _ = self._mark()
        liq = self._liq(cash, net)
        if order is not None:
            if order.status not in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                return  # in flight: re-priced at the next bar
            d = self.decisions.get(self._risk_stop_id, {})
            if d.get("intent") == intent:
                d["liq"] = liq
                if order.trigger_price != px or order.quantity != quantity:
                    self.modify_order(order.client_order_id, quantity=quantity, trigger_price=px)
                return
            self.cancel_order(order.client_order_id)  # the nearest limit changed: rest one in its name
        side = self._pos_side()
        exit_side, word = (OrderSide.SELL, "sell") if side > 0 else (OrderSide.BUY, "buy")
        what = {"risk_halt": "the drawdown halt", "risk_pause": "the daily-loss pause",
                "liquidation_cut": "the cut before liquidation"}[intent]
        if side == 0:
            return
        coid = self.order_factory.generate_client_order_id()
        stop = StopMarketOrder(self.trader_id, self.strategy_id, self._cfg.instrument_id, coid, exit_side, quantity,
                               px, TriggerType.DEFAULT, TimeInForce.GTC, True, False, UUID4(), self.clock.timestamp_ns())
        self._risk_stop_id = str(coid)
        # Journaled when it trades (_journal_risk_stop), at the level it traded at: re-priced every bar and
        # cancelled whenever the position closes some other way, most risk stops never trade.
        self.decisions[self._risk_stop_id] = {"intent": intent, "reason": f"Risk stop: {word} where {what} acts",
                                              "signal": {"kind": intent}, "journaled": False, "liq": liq}
        self._sent.append(coid)
        self.submit_order(stop)

    def _journal_risk_stop(self, order, price: float) -> None:
        """Journal the risk stop on its first fill, with the level it rested at then. Filled through the
        liquidation price (a gap past it), the venue took the position first: journaled as a liquidation,
        as paper's guard journals one (review round 11, M11-3)."""
        d = self.decisions.get(str(order.client_order_id))
        if d is None or d.get("journaled", True):
            return
        level, intent = order.trigger_price.as_double(), d["intent"]
        word = "sell" if order.side == OrderSide.SELL else "buy"
        held = "long" if order.side == OrderSide.SELL else "short"
        liq = d.get("liq")
        if through_liquidation(1 if held == "long" else -1, price, liq):
            d.update(journaled=True, intent="liquidation", signal={"price": price, "liquidation_px": round(liq, 8)},
                     reason=f"Liquidated: the price {price:,.6g} gapped through the liquidation price {liq:,.6g}")
            self._liquidation_events(d["reason"], price, liq)
        else:
            what = {"risk_halt": "the drawdown halt", "risk_pause": "the daily-loss pause",
                    "liquidation_cut": "the cut before liquidation"}[intent]
            d.update(journaled=True, signal={"trigger": round(level, 8), "kind": intent},
                     reason=(f"Risk stop: rested a {word} at {level:,.6g}, where {what} acts on the open {held} "
                             "(re-priced each bar); fills at that level, or the open if the price gaps through"))
        intent = d["intent"]
        self.runtime.on_order(order_id=str(order.client_order_id), side=word.upper(), qty=float(order.quantity),
                              intent=intent, reason=d["reason"], signal=d["signal"], order_type="STOP")

    def _risk_stop_filled(self, done: bool, sign: int, price: float) -> None:
        """The risk stop traded: the guard acts as paper's would on that trade. A halt or pause is the
        runtime's to set (its tick judges the equity now); a cut locks out the side it closed. Either way
        the stop-loss and target still resting are cancelled with what is left (_sell_all)."""
        if not done and self._pos_side() != 0:
            return
        intent = self.decisions.get(self._risk_stop_id, {}).get("intent")
        self._risk_stop_id = None
        if intent == "liquidation":
            self._exit_lock, self._flip = -sign, None
            self._on_tick()  # the risk check halts on what is left
        elif intent == "liquidation_cut":
            self.runtime.store.event(self.runtime.name, "warning", "liquidation_cut",
                                     f"Cut to avoid liquidation: the risk stop filled at {price:,.6g}",
                                     ts=self.runtime.now())
            self._exit_lock, self._flip = -sign, None
            self._sell_all("liquidation_cut", "Cut to avoid liquidation: the rest of the position", {"price": price})
        else:
            self._on_tick()

    def _liq_signal(self) -> dict:
        """The open position's liquidation price as the engine's check has it now, for a stop's journaled signal
        ({} when there is none: flat, spot, or a long at 1x)."""
        if not self._margin:
            return {}
        _, cash, qty, _ = self._mark()
        liq = self._liq(cash, qty) if qty else None
        return {"liquidation_px": round(liq, 8)} if liq is not None else {}

    def _gap_liquidation(self, coid: str, journal_id: str, price: float) -> bool:
        """GAP-LIQ (Independent Quant Advisor, 6 Oct): a stop whose fill, after slippage, is at or past the
        liquidation price books as a liquidation, the venue having taken the position first. Its order is journaled
        as one, with the liquidation event and an incident, so the D3 loss, the halt with X and Y, and the reset after
        liquidation follow as for any liquidation. The liquidation price is the one the engine's own check had when
        the stop fired (paper) or as of the bar it rested through (backtest); through_liquidation decides both."""
        d = self.decisions.get(coid)
        held = self._entry_side or 0
        if d is None or d.get("intent") != "stop_loss" or not held:
            return False
        liq = (d.get("signal") or {}).get("liquidation_px")
        if not through_liquidation(held, price, liq):
            return False
        reason = (f"Liquidated: the stop filled at {price:,.6g}, at or past the liquidation price {liq:,.6g}, so the "
                  "venue took the position first")
        d.update(intent="liquidation", reason=reason, signal={**d["signal"], "price": price})
        self._rebook = journal_id  # the journal re-books it once this fill is in (on_order_filled)
        self._liquidation_events(reason, price, liq)
        self._exit_lock, self._flip = held, None
        if self._entry_px is None:
            # A paper stop clears the entry as it fires (_check_exits); the liquidation's X and a later slice's
            # book need it, so it comes back from what the stop journaled.
            self._entry_px = d["signal"].get("entry_px")
        return True

    def _liquidation_events(self, reason: str, price: float, liq: float) -> None:
        """Every liquidation the engine books journals its error event; the incident (Advisor 18:17: one on every
        liquidation) is opened once the liquidation's fill is in, with X, Y and the equity left (_liquidation_incident)."""
        rt = self.runtime
        rt.store.event(rt.name, "error", "liquidation", reason, ts=rt.now())

    def _cover_shortfall(self, price: float, event=None) -> None:
        """Isolated margin: a position closed past its bankruptcy price (a gap through the liquidation price)
        loses its isolated margin and no more, plus its fees (markets.gap_loss_cap, the same figure as the Risk
        page's stress rows; Independent Quant Advisor, QA P1-D3); the venue's insurance fund takes the rest. So
        once flat, a price loss past the margin comes back to cash, journaled, and equity never ends below zero."""
        credit = 0.0
        pos = self.cache.position(event.position_id) if event is not None and event.position_id else None
        if pos is not None and pos.is_closed and pos.peak_qty.as_double() > 0:
            qty, entry = pos.peak_qty.as_double(), pos.avg_px_open
            sign = 1 if pos.entry == OrderSide.BUY else -1
            lev = self.runtime.profile.max_leverage if self.runtime is not None else 1.0
            past = sign * (entry - pos.avg_px_close) * qty - markets.isolated_margin(qty, entry, lev)
            credit = max(past, 0.0)
        equity = self._mark()[0] + credit
        credit += max(-equity, 0.0)
        if credit <= 0:
            return
        credit = math.ceil(credit * 100) / 100  # to the cent, so no float residue leaves it a fraction below zero
        self._cash_adj += credit
        now = self.clock.utc_now()
        self.insurance_log.append((now, credit))
        if self.runtime is not None:
            self.runtime.store.record_insurance(self.runtime.name, price=price, amount=round(credit, 8),
                                                ts=self.runtime.now())
            self.runtime.store.event(self.runtime.name, "error", "insurance_fund",
                                     f"Closed at {price:,.6g}, past the bankruptcy price: the venue's insurance fund "
                                     f"takes the {credit:,.2f} shortfall, as isolated margin caps the loss at the "
                                     "position's margin", ts=self.runtime.now())

    def _liquidation_figures(self, trade_id: str | None, notional: float = 0.0) -> tuple[float, float, float | None, bool]:
        """What a liquidation lost, from the journal (Advisor 18:17 point 4), as (fees, quantity taken, equity before,
        whether the fill `trade_id` is journaled yet). Its fills are those of every liquidation order since the
        position was last flat: a risk stop journaled as one can take part and the guard's close the rest (Code
        Reviewer on 6047b50). The fees: the entry fees of the position held, so a restart and a partial reduce are
        counted right, plus the liquidation's own as journaled, plus the taker rate on `notional`, what is still to
        close. The equity: the strategy's when the position was opened, the last mark before its first fill (Advisor
        20:37, P1-D20), fixed through partial reductions; or, with no mark before it, the cash the journal had then."""
        if self.runtime is None:
            return 0.0, 0.0, None, True
        store, name = self.runtime.store, self.runtime.name
        fills = sorted(store.fills(name, limit=1_000_000), key=lambda f: (f["ts"], f["id"]))
        liquidations = {o["order_id"] for o in store.orders(name, limit=100_000, intents=("liquidation",))}
        held, flat = Decimal(0), 0  # the fills since the position was last flat before now
        for i, f in enumerate(fills[:-1]):
            held += Decimal(repr(float(f["qty"]))) * (1 if f["side"] == "BUY" else -1)
            if abs(held) < DUST:
                held, flat = Decimal(0), i + 1
        window = fills[flat:]
        liq = [f for f in window if f["order_id"] in liquidations]
        rest = [f for f in window if f["order_id"] not in liquidations]
        fees = (replay_book(rest, 0.0)["entry_fees"] + sum(float(f["fee"]) for f in liq)
                + notional * self.runtime.taker_fee)
        at_entry = None
        if window:  # Y's base (Advisor 20:37): the equity at the position's first fill, flat then so all cash
            opened = window[0]["ts"]
            at_entry = replay_book(fills[:flat], self.runtime.starting_balance,
                                   store.funding_total(name, before=opened), store.insurance_total(name, before=opened)
                                   )["cash"]
        journaled = trade_id is None or any(str(f.get("trade_id")) == trade_id for f in liq)
        return fees, sum(float(f["qty"]) for f in liq), at_entry, journaled

    def _liquidation_margin(self, trade_id: str, qty: float, fee: float, held: tuple) -> tuple[str, bool]:
        """The liquidation's halt once its last fill (trade_id, qty, fee) is in: X over every slice (not the last), and
        whether that fill is journaled. One that isn't yet (a write still to retry) is added from the event itself,
        so X is never short of it (Code Reviewer on eb737bd)."""
        fees, taken, before, journaled = self._liquidation_figures(trade_id)
        if not journaled:
            fees, taken = fees + fee, taken + qty
        return self._margin_lost(max(taken, held[0]), held[1], fees, before), journaled

    def _margin_lost(self, qty: float, entry: float, fees: float = 0.0, before: float | None = None) -> str:
        """The halt for a liquidation (Independent Quant Advisor, 6 Oct 17:57, Y as of 20:37): "Position margin lost
        (liquidated): X, Y% of strategy equity at entry", X the position's isolated margin plus its entry and
        liquidation fees (18:17 point 4), Y its share of the strategy's equity when the position was opened, uncapped,
        to one decimal under 10% so a small loss never reads 0% (QA P1-D17, P1-D20)."""
        lev = self.runtime.profile.max_leverage if self.runtime is not None else 1.0
        lost = markets.isolated_margin(qty, entry, lev) + fees
        if before is None:
            before = (self.runtime._last_equity or self.runtime.peak) if self.runtime is not None else 0.0
        share = (f"{lost / before:.1%}" if lost / before < 0.1 else f"{lost / before:.0%}") if before and before > 0 \
            else "all"
        return f"{WIPED_OUT}: {lost:,.2f}, {share} of strategy equity at entry"

    def _liquidation_incident(self) -> None:
        """The incident the engine opens on every liquidation, once the position is gone (Advisor 18:17; 20:37: with
        the equity left). A reset after liquidation needs its note (RAL)."""
        rt = self.runtime
        left = self._mark()[0]
        rt.store.event(rt.name, "error", "incident",
                       f"Incident, {rt.name}: {self._liquidated or WIPED_OUT}; {max(left, 0.0):,.2f} of equity left. It "
                       "stays halted until you reset it after liquidation, which needs a note on why the "
                       "half-liquidation stop did not protect the position.", ts=rt.now())

    def _wiped_out_why(self, shortfall: float = 0.0) -> str:
        """shortfall: an open position's equity below zero, which the insurance fund will cover once it closes, so
        the halt says how much before the close journals it (fix re-check, mF-1)."""
        covered = sum(a for _, a in self.insurance_log)
        if not covered and self.runtime is not None:  # since a restart: the journal has it
            covered = self.runtime.store.insurance_total(self.runtime.name)
        why = self._liquidated or (self.runtime.liquidated if self.runtime is not None else None) or WIPED_OUT
        if covered > 0:
            return liquidation_reason(why, covered)
        if shortfall > 0:
            return why + f"; the venue's insurance fund covers the shortfall, about {shortfall:,.2f} at this price"
        return why

    def _liquidation_guard(self, cash: float, qty: float, price: float) -> None:
        """The position closes at market once the price reaches its liquidation price (the venue would
        take it), or comes within the risk profile's minimum distance of it (cut back before the venue
        does), on the position's isolated margin (_liq)."""
        terms = self._cfg.perp
        if terms is None or qty == 0 or price <= 0 or self._pending_exit is not None or self._busy():
            return
        liq = self._liq(cash, qty)
        if liq is None:
            return
        side = 1 if qty > 0 else -1
        crossed = through_liquidation(side, price, liq)
        distance = abs(price - liq) / price
        floor = self.runtime.profile.min_liquidation_distance if self.runtime is not None else 0.0
        if not crossed and distance >= floor:
            return
        intent = "liquidation" if crossed else "liquidation_cut"
        reason = (f"Liquidated: the price {price:,.6g} reached the liquidation price {liq:,.6g}" if crossed else
                  f"Cut to avoid liquidation: the price {price:,.6g} is {distance:.1%} from the liquidation price "
                  f"{liq:,.6g}, inside the {floor:.0%} the risk profile keeps")
        if self.runtime is not None and crossed:
            self._liquidation_events(reason, price, liq)
        elif self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "warning", intent, reason, ts=self.runtime.now())
        self._exit_lock = side
        self._flip = None
        self._sell_all(intent, reason, {"price": price, "liquidation_px": round(liq, 8), "distance": round(distance, 6)})

    def _guard_breached(self, price: float) -> bool:
        """Paper on a perp, on every trade: would the risk guard or the liquidation guard act at this price?
        With leverage, waiting for the next 30-second tick lets a fast move run through both."""
        if self.runtime is None or self._entry_px is None or self.runtime.status != "running":
            return False
        equity, cash, qty, _ = self._mark()
        if equity <= 0:
            return True
        day_open = self.runtime._day_open or equity
        if risk.check(self.runtime.profile, equity, max(self.runtime.peak, equity), day_open) is not None:
            return True
        liq = self._liq(cash, qty)
        return liq is not None and abs(price - liq) / price < self.runtime.profile.min_liquidation_distance

    def _intrabar_guard(self, bar: Bar) -> None:
        """Backtest on a perp: the risk guard judges each minute at its worst price for the position (the
        low for a long, the high for a short), not its close, so a wick that recovers inside the minute
        still halts it, as it would on paper's trades (long/short verdict, L3)."""
        qty = self._net_position()[0]
        if qty == 0 or self.runtime is None:
            self._guard_equity = None
            return
        worst = bar.low.as_double() if qty > 0 else bar.high.as_double()
        _, cash, _, _ = self._mark()
        self._guard_equity = cash + qty * worst
        liq = self._liq(cash, qty)
        if liq is not None and (worst <= liq if qty > 0 else worst >= liq):
            self._guard_price = worst

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
        base, quote = pair_of(self.instrument).split("/")
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
        if self._margin:
            # A margin account holds realised money only; the journal keeps spot-style cash (a short holds
            # its sale's proceeds), so take the open position's cost back out: cash = balance - qty x entry.
            # _cash_adj carries the restore and funding, which the simulated venue knows nothing of.
            qty, avg = self._net_position()
            cash = cash - qty * avg + self._cash_adj
            return cash + qty * price, cash, qty, price
        qty = sum(totals.get(c, 0.0) for c in self._codes("base"))
        return cash + qty * price, cash, qty, price

    def _market_seen(self) -> None:
        self._last_market_ns = self.clock.timestamp_ns()
        if not self._backtest and not self.hub_fed:
            self._minutes_seen.add(self._last_market_ns // MINUTE_NS)
            if self._first_minute is None:
                self._first_minute = self._last_market_ns // MINUTE_NS
        if self.runtime is not None and not self._backtest:
            self.runtime.market_seen()
        if self._noted & {"stale_price", "feed_dead", "hub_venue_down"}:
            self._noted -= {"stale_price", "feed_dead", "hub_venue_down"}
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
        now = self.clock.timestamp_ns()
        minutes = (now - self._last_market_ns) / 60e9
        if minutes >= STALE_PRICE_WARN_MINUTES:
            self._note("stale_price", f"No trade or quote from the venue for {minutes:.0f} minutes; marks and the "
                       f"risk guard are using the last price, {self._price():,.6g}")
        if minutes >= STALE_PRICE_RESTART_MINUTES and self.hub_status is not None and self.hub_status.venue_down(now):
            # The hub is up and has lost the venue itself: a restart reconnects to the same hub and can't help, and
            # each one costs the bar under way (QA P1-C9). Wait for the hub to get the venue back.
            self._note("hub_venue_down", "The market data hub is running but has lost its venue connection; "
                       "waiting for it to come back rather than restarting this strategy")
            return False
        if minutes >= STALE_PRICE_RESTART_MINUTES:
            self._note("feed_dead", f"No market data for {minutes:.0f} minutes: the price feed looks dead, so this "
                       "process stops reporting and the supervisor restarts it to reconnect", level="error")
            return True
        return False

    def _on_tick(self, _event=None) -> None:
        self._last_tick_ns = self.clock.timestamp_ns()
        self._snap_settlements(self._last_tick_ns)
        if self._feed_dead():
            return  # no heartbeat, so the supervisor restarts the process
        self._publish_signals()  # a quiet market still shows the lights (display only)
        if self._restore is not None:
            # The carried-over position isn't back at the simulated venue yet: nothing to mark or guard
            # against the journal until it is, but the strategy is alive.
            self.runtime.store.heartbeat(self.runtime.name)
            return
        try:
            if self._margin:
                self._apply_funding(self._price())
            equity, cash, qty, price = self._mark()
            # A perp position valued at or below zero equity is past its liquidation price (a gap the guards
            # couldn't trade inside), not a book that can't be valued: liquidate it, and the risk check halts.
            underwater = self._margin and qty != 0 and price > 0 and equity <= 0
            # Flat with nothing left after a gap past the bankruptcy price (the insurance fund took the rest): a
            # strategy that was wiped out, not one that can't be valued. It is marked at zero and halted
            # (review round 12, B12-1: it kept its last mark before the gap and stayed running).
            ruined = (self._margin and qty == 0 and price > 0 and equity <= 0
                      and self._account() is not None and self.instrument is not None)
            # Flat after a wipe-out the PM resumed, with what wasn't margined kept (P1-D3): still a wipe-out, so it
            # halts again, marked at what it kept (HoE and QA, 6 Oct).
            kept = (self._margin and qty == 0 and price > 0 and equity > 0
                    and (self.runtime.wiped_out or self._liquidated is not None))
            if price <= 0 or (equity <= 0 and not underwater and not ruined):
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
                tol = 2 * float(self._lot())  # two units of the base currency (review rounds 9 and 10, B9-1, B10-1)
                if not self.runtime.reconcile(cash=cash, qty=qty, qty_tolerance=tol):
                    # Halted: nothing new trades and nothing is flattened, but a resting stop-loss and target
                    # stay, so the position isn't left unguarded until the PM acts (review round 10, m10-1).
                    self._drop_kept()
                    for order in self.cache.orders_open(strategy_id=self.strategy_id):
                        if self.decisions.get(str(order.client_order_id), {}).get("intent") not in EXIT_LEGS:
                            self.cancel_order(order)
            guard, self._guard_equity = self._guard_equity, None
            worst, self._guard_price = self._guard_price, None
            if self._margin and qty != 0:
                # Through the liquidation price: the venue takes the position whatever our own risk check says,
                # so the liquidation goes first and is journaled as one (review round 11, M11-3).
                probe = price if underwater or worst is None else worst
                liq = self._liq(cash, qty)
                if underwater or through_liquidation(1 if qty > 0 else -1, probe, liq):
                    self._liquidation_guard(cash, qty, probe)
            self.runtime.close_floor = float(max(self._lot(), self._min_qty()))
            wiped = None
            if underwater and self._liquidated is None:
                # Its liquidation fee at the taker rate on this mark, until the fill gives the fee charged.
                fees, taken, before, _ = self._liquidation_figures(None, abs(qty) * price)
                self._liquidated = self._margin_lost(abs(qty) + taken, self._entry_px or price, fees, before)
            if underwater or ruined:  # isolated margin: the strategy can't lose more than it has
                wiped = self._wiped_out_why(max(-equity, 0.0) if underwater else 0.0)
                equity = 0.0
                cash = 0.0 if ruined else cash
            elif kept and (self.runtime.status != "halted" or self.runtime.liquidated is None):
                wiped = self._wiped_out_why()
            liquidating = any(self.decisions.get(str(o.client_order_id), {}).get("intent") == "liquidation"
                              for o in self._working())
            if self.runtime.tick(equity=equity, cash=cash, qty=qty, price=price, guard_equity=guard,
                                 busy=bool(self._working()), ruined=wiped, liquidating=liquidating) == "flatten":
                self.cancel_all_orders(self._cfg.instrument_id)
                self._flip = None
                intent, reason = self.runtime.flatten_why or ("pm_flatten", "Flattened")
                self._sell_all(intent, reason, {"equity": equity, "peak": self.runtime.peak})
            elif self._margin:
                self._liquidation_guard(cash, qty, worst or price)
            self._cancel_resting_entries()
        except Exception as exc:  # never let bookkeeping kill the sleeve silently: counted, kept and journaled
            self._report("_on_tick", exc)

    def _cancel_resting_entries(self) -> None:
        """While nothing may open (CHOKE: liquidated until a reset after liquidation, Advisor 17:57 rule (c), QA P1-D21;
        halted, paused or stopped), every entry or rebalance still resting at the venue, or the unfilled rest of one,
        is cancelled. Without it a resting entry filled later and opened a position on a halted strategy. Closing
        orders (stops, exits, the liquidation itself) are left alone."""
        open_ = [o for o in self.cache.orders_open(strategy_id=self.strategy_id) if o.status != OrderStatus.PENDING_CANCEL]
        if self._journal_intents is None and any(str(o.client_order_id) not in self.decisions for o in open_):
            # An order this process has no decision for was sent before a restart: the journal has its intent. Read
            # once; such an order's intent never changes (Code Reviewer on 81e6d6f).
            rt = self.runtime
            self._journal_intents = {o["order_id"]: o["intent"]
                                     for o in rt.store.orders(rt.name, statuses=OPEN_ORDER_STATUSES, limit=1000)}

        def intent(coid: str):
            return self.decisions.get(coid, {}).get("intent") or (self._journal_intents or {}).get(coid)
        resting = [o for o in open_ if intent(str(o.client_order_id)) in OPENING_INTENTS]
        if resting and self.runtime.entry_blocked()[0]:
            for order in resting:
                self.cancel_order(order.client_order_id)

    def _gated_fill(self, coid: str, qty: float, px: float) -> None:
        """CHOKE at fill: an entry or rebalance that fills while nothing may open (sent before the gate closed, its
        cancel too late) is kept, with its stop placed for what filled as on any entry, and never flattened; it opens
        an incident, once per order, for the PM to decide."""
        if (self.runtime is None or coid in self._gated_fills
                or self.decisions.get(coid, {}).get("intent") not in OPENING_INTENTS):
            return
        blocked, why = self.runtime.entry_blocked()
        if not blocked:
            return
        self._gated_fills.add(coid)
        rt = self.runtime
        rt.store.event(rt.name, "error", "incident",
                       f"Incident, {rt.name}: an order that adds to the position filled while nothing may open ({why}): "
                       f"{qty:g} at {px:,.6g}. It is kept with its stop, not closed; you decide what to do with it.",
                       ts=rt.now())

    def _adds(self, side) -> bool:
        """Whether an order on `side` would make the position bigger: on a perp, a buy when flat or long and a sell
        when flat or short (a reversal's opening leg included, sent once its close has filled); on spot, a buy."""
        if not self._margin:
            return side == OrderSide.BUY
        net = self._net_position()[0] if self.cache is not None and self.instrument is not None else 0.0
        return net >= 0 if side == OrderSide.BUY else net <= 0

    def _gated(self, side, intent: str) -> str | None:
        """CHOKE at submit: why an order that would open or add may not be sent now (runtime.entry_blocked), else
        None. An entry always opens or adds (a reversal's opening leg is sent once its close has filled); a rebalance
        only when it buys more. Stops, exits, closes and liquidations are never gated."""
        if self.runtime is None or not (intent == "entry" or (intent == "rebalance" and self._adds(side))):
            return None
        return self.runtime.entry_blocked()[1]

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
            return
        self._resize_if_due()

    def _resume_exit(self, coid: str) -> None:
        """An order closed: if a sell was waiting for every working order to close, send it now."""
        self._cancel_on_accept.discard(coid)
        if self._pending_exit is not None and not self._working():
            intent, reason, values = self._pending_exit
            self._pending_exit = None
            self._sell_all(intent, reason, values)

    def on_order_rejected(self, event) -> None:
        self._order_status(event, "rejected")
        if str(event.client_order_id) == self._restore_id:
            self._restore_id = None  # sent again on the next price
            return
        if self._slice_ended(event):
            return
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
        if self._pos_side() == 0:
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
        self._exit_lock = self._pos_side() if self._margin else True
        self._sell_all("stop_loss", reason, {**signal, "price": price})

    def _drop_kept(self, earn: bool = False) -> None:
        """Paper: the decision behind every kept post-only order no longer stands; nothing more is sent.
        earn: first earn each its minute so far. A backtest's venue fills a resting order from the prints
        of the bar its stop fires in, before the stop sells it all with them (review round 9, M9-3)."""
        for coid in list(self._kept):
            if earn and self._kept[coid]["bar"] is not None:
                self._earn(coid, self._kept[coid])
            self._close_kept(coid, "")

    def _slice_ended(self, event) -> bool:
        """A slice closed without filling all of it (denied, rejected, cancelled or expired): settle its
        kept order. True if it was a slice."""
        pid = str(event.client_order_id)
        if pid not in self._slices:
            return False
        order = self.cache.order(event.client_order_id)
        self._slice_done(pid, order.leaves_qty.as_double() if order is not None else 0.0)
        self._resume_exit(pid)
        return True

    def on_order_denied(self, event) -> None:
        self._order_status(event, "denied")
        if str(event.client_order_id) == self._restore_id:
            self._restore_id = None
            return
        if self._slice_ended(event):
            return
        self._resume_exit(str(event.client_order_id))

    def on_order_canceled(self, event) -> None:
        self._order_status(event, "canceled")
        if self._slice_ended(event):
            return
        coid = str(event.client_order_id)
        info = self._maker.pop(coid, None)
        self._cancel_alert(coid)
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
        if self._slice_ended(event):
            return
        self._resume_exit(str(event.client_order_id))

    def on_order_filled(self, event) -> None:
        intrabar = self._intrabar_fill(event)  # a bars-only backtest's resting order filled inside a bar, or None
        held = (self._entry_qty, self._entry_px)  # the position before this fill
        if self._margin:  # a fill just after the hour, with no trade, bar or tick between: held before it
            fill = float(event.last_qty) * (1 if event.is_buy else -1)
            self._snap_settlements(event.ts_event, qty=self._net_position()[0] - fill)
            if self._restore is None and str(event.client_order_id) != self._restore_id:
                # Settle funding owed up to the fill before booking it, on the position held until then: a stop or a
                # liquidation filled on a gap pays the settlements it was held through, and the insurance fund's
                # share is then reckoned on that cash.
                self._intrabar = intrabar
                try:
                    self._apply_funding(self._last_close or float(event.last_px))
                finally:
                    self._intrabar = None
        coid = str(event.client_order_id)
        order = self.cache.order(event.client_order_id)
        done = order is None or order.is_closed
        if coid == self._restore_id:
            # The carried-over position is back at the simulated venue, at today's price: keep the journal's
            # cash by taking up the difference from the journal's entry. Not a trade, so not journaled.
            q = event.last_qty.as_double() * (1 if event.is_buy else -1)
            self._cash_adj += q * (event.last_px.as_double() - self._restore["entry"])
            if done:
                if self.runtime is not None:
                    self.runtime.store.event(self.runtime.name, "info", "restore_filled",
                                             f"Position back at the simulated venue at {event.last_px.as_double():,.6g}; "
                                             f"the journal keeps its entry of {self._restore['entry']:,.6g}",
                                             ts=self.runtime.now())
                self._restore = self._restore_id = None
            return
        if coid in self._maker and done:
            self._maker.pop(coid)
            self._cancel_alert(coid)
        # The quantity's own decimal, as the nearest float: as_double() can land a float away (1015.315545 read
        # 1015.3155449999999), and those differences summed into merged round trips (round 11, M10-4).
        qty, px = self._held_qty(float(event.last_qty.as_decimal())), event.last_px.as_double()
        # Paper: a slice of a kept post-only order is the order's own fill, at its limit and the maker fee
        # (the fee model charged it so; ScheduleFeeModel.maker_slices).
        journal_id, fee = coid, (event.commission.as_double() if event.commission is not None else 0.0)
        kept_id = self._slices.get(coid)
        if kept_id is not None:
            px = float(self.fee_model.maker_slices[coid][0]) if self.fee_model is not None else px
            journal_id = kept_id
            fee = qty * px * float(self.fee_model.fees.maker) if self.fee_model is not None else fee
        sign = 1 if event.is_buy else -1
        if self._margin and self.runtime is not None and self._gap_liquidation(coid, journal_id, px):
            held = (held[0], self._entry_px)  # the entry a paper stop cleared as it fired, put back
        if self._margin:
            opening = self._track_entry(sign, event.last_qty.as_decimal(), qty, px)
        else:
            opening = self._entry_side in (0, sign) and sign > 0
            if opening:
                cost = (self._entry_px or 0.0) * self._entry_qty + qty * px
                self._entry_qty += qty
                self._entry_px = cost / self._entry_qty
                self._entry_side = sign
            else:
                self._entry_qty = max(self._entry_qty - qty, 0.0)
                if self._entry_qty < float(self._lot()) / 2:  # less than half a lot is flat: no order can trade it
                    self._entry_px, self._entry_qty, self._entry_side = None, 0.0, 0
                    self._stop_frac = self._tp_frac = None
                    self._replan_pending = self._plan_entry = None
        if opening:
            self._gated_fill(coid, qty, px)
        if self._backtest:
            if opening and self._pending_exit is None:
                # On every entry fill, not only the last: a post-only entry can fill in slices through its
                # maker wait, and paper guards each slice from its first trade (review round 9, M9-4).
                self._rest_exits()
                self._rest_risk_stop()
            elif self.decisions.get(str(event.client_order_id), {}).get("intent") in ("stop_loss", "take_profit"):
                # As in paper: no re-entry until the signal has moved off the side that was closed.
                self._exit_lock = -sign if self._margin else True
                # Paper's stop sells the whole position and cancels an entry still working (_sell_all); so does
                # this. A slice that filled after the stop last grew is sold at market with what is left.
                intent = self.decisions[str(event.client_order_id)]["intent"]
                if done and self._entry_qty > 1e-12 and self._pos_side() != 0:
                    what = "Stop-loss" if intent == "stop_loss" else "Take-profit"
                    self._sell_all(intent, f"{what}: the rest of the position, which filled after the resting "
                                           "order was last sized", {"price": px})
                else:
                    entry_side = OrderSide.SELL if sign > 0 else OrderSide.BUY  # the side the closed position opened on
                    for order in self._working():
                        if order.side == entry_side and order.status not in (OrderStatus.PENDING_CANCEL,):
                            if order.status in (OrderStatus.INITIALIZED, OrderStatus.SUBMITTED):
                                self._cancel_on_accept.add(str(order.client_order_id))
                            else:
                                self.cancel_order(order.client_order_id)
        if self.runtime is not None:
            if coid == self._risk_stop_id and order is not None:
                self._journal_risk_stop(order, px)
            # A fill on a gap took the bar's open price, so its record carries the open's time (QA P1-D12).
            at = (datetime.fromtimestamp(intrabar[0] / 1e9, tz=timezone.utc)
                  if intrabar is not None and intrabar[2] else None)
            self.runtime.on_fill(side="BUY" if event.is_buy else "SELL", qty=qty, price=px, fee=fee,
                                 order_id=journal_id, trade_id=str(event.trade_id), ts=at)
            if self._rebook is not None:
                d = self.decisions[coid]
                self.runtime.store.rebook_liquidation(self._rebook, d["reason"], d["signal"], ts=at or self.runtime.now())
                self._rebook = None
        if self._margin and opening and intrabar is not None and coid != self._restore_id:
            self._opened_in_bar.append((intrabar, sign * qty))  # charged with the bar's range in on_bar
        if self._margin and self._entry_side == 0:
            if self.decisions.get(coid, {}).get("intent") == "liquidation" and held[0] and held[1]:
                margin, journaled = self._liquidation_margin(str(event.trade_id), qty, fee, held)
                if not journaled:
                    margin = self._liquidated or margin  # its fill isn't in the journal yet: keep the estimate
                elif self._liquidated is not None and margin != self._liquidated and self.runtime is not None:
                    # Halted on the mark with the liquidation fee estimated: once the whole position has gone, the
                    # halt gives the X the journal has, from all its slices (QA P1-D18).
                    old, self._liquidated = self._liquidated, margin
                    reason = self.runtime.store.sleeve(self.runtime.name).status_reason or ""
                    if self.runtime.status == "halted" and reason.startswith(old):
                        self.runtime.liquidated = margin
                        self.runtime._set("halted", reason.replace(old, margin, 1))
                self._liquidated = margin
            self._cover_shortfall(px, event)
            if self.decisions.get(coid, {}).get("intent") == "liquidation" and self.runtime is not None:
                self._liquidation_incident()
        if coid == self._risk_stop_id:
            self._risk_stop_filled(done, sign, px)
        if kept_id is not None and done:
            self._slice_done(coid)
        if done:
            self._resume_exit(coid)
            if self._flip is not None and self._pending_exit is None and self._pos_side() == 0:
                # Turning from one side to the other: the close has filled, so open the new side now.
                side, bar, reason, values = self._flip
                self._flip = None
                self._open(side, bar, reason, values)

    def _track_entry(self, sign: int, filled: Decimal, qty: float, px: float) -> bool:
        """Perp: the entry book after a fill, read from the venue's position (already updated when the fill
        arrives), never kept up incrementally: one exit that left a lot behind once put the incremental
        book out of phase for good, booking every later short as a reduction of a phantom long, so no
        short ever rested a stop (review round 11, B11-2). Less than half a lot is flat. Returns whether
        the fill opened or added to the position (and so needs its exits rested)."""
        post = self._signed_qty()
        half = self._lot() / 2
        side = 0 if abs(post) < half else (1 if post > 0 else -1)
        pre = post - sign * filled
        if side == 0:
            self._entry_px, self._entry_qty, self._entry_side = None, 0.0, 0
            self._stop_frac = self._tp_frac = None
            self._replan_pending = self._plan_entry = None
            return False
        opening = side == sign and abs(post) > abs(pre)
        if opening and self._entry_side == side and self._entry_px is not None:
            # Added to the same side: the average entry of what was held and this fill.
            held = float(abs(post)) - qty
            self._entry_px = (self._entry_px * held + px * qty) / float(abs(post))
        elif opening or self._entry_side != side or self._entry_px is None:
            self._entry_px = px  # opened from flat, or this fill went through flat and opened the other side
        self._entry_qty, self._entry_side = float(abs(post)), side
        return opening

    def _rest_exits(self) -> None:
        """Backtests: after an entry fills, rest the exits at the venue as real orders would sit there.
        (Paper and live watch every trade instead.) The stop is a sell stop at its level and the target
        a sell limit at its level, linked one-updates-the-other: a fill on either shrinks the other by as
        much, so together they never sell more than is held. A flatten cancels both first (_sell_all). Within a bar the
        extreme nearer the open trades first. A stop fills at its level, or at the open when the price
        gaps through it; a target fills at its level, never better, never worse."""
        cfg = self._cfg
        plan = {"stop_loss": self._stop_frac, "take_profit": self._tp_frac}
        if not any(plan.values()) or self._entry_px is None:
            return
        if self.runtime is not None and self.runtime.status != "running" and not (
                self._exits_only and self.runtime.status == "paused"):
            return  # halted, paused or flattening: nothing new rests (review round 11, B11-3); exits-only rests its own
        resting = self._resting_exits()
        if resting:
            # At most one resize per moment of the run. The simulated venue matches its resting orders against
            # the current bar again on every command, without using up what they took, so a resize there
            # fills the post-only entry once more, whose fill would ask for another resize: a chain that
            # took the whole entry from one bar. A later slice at the same moment waits for the next event.
            now = self.clock.timestamp_ns()
            if now == self._resized_ns:
                self._resize_due = True
                return
            self._resized_ns = now
            self._resize_exits(resting, plan)
            return
        qty = self._position_qty(free=True).quantize(self._lot(), rounding=ROUND_DOWN)
        if qty < self._min_qty():
            return
        quantity = Quantity.from_decimal_dp(qty, self.instrument.size_precision)
        ids = {k: self.order_factory.generate_client_order_id() for k in ("stop_loss", "take_profit") if plan[k]}
        both = len(ids) == 2
        side = self._entry_side or 1
        exit_side, exit_word = (OrderSide.SELL, "sell") if side > 0 else (OrderSide.BUY, "buy")

        def link(kind: str) -> dict:
            other = "take_profit" if kind == "stop_loss" else "stop_loss"
            return {"contingency_type": ContingencyType.OUO, "linked_order_ids": [ids[other]]} if both else {}

        orders = []
        now = self.clock.timestamp_ns()
        if plan["stop_loss"]:
            stop = plan["stop_loss"]
            level = self._entry_px * (1 - side * stop)
            orders.append((StopMarketOrder(
                self.trader_id, self.strategy_id, cfg.instrument_id, ids["stop_loss"], exit_side, quantity,
                Price(level, self.instrument.price_precision), TriggerType.DEFAULT, TimeInForce.GTC, self._margin, False,
                UUID4(), now, **link("stop_loss")), "stop_loss", "STOP",
                f"Stop-loss: resting {exit_word} at {level:,.6g}, {stop:.1%} {'below' if side > 0 else 'above'} the "
                f"{self._entry_px:,.6g} entry"
                + (f" (set {self._stop_basis})" if cfg.stop_atr or cfg.stop_swing_bars else "")
                + "; fills at that level, or the open if the price gaps through",
                {"entry_px": round(self._entry_px, 8), "stop_loss": round(stop, 6), "trigger": round(level, 8),
                 **self._liq_signal()}))
        if plan["take_profit"]:
            tp = plan["take_profit"]
            level = self._entry_px * (1 + side * tp)
            orders.append((LimitOrder(
                self.trader_id, self.strategy_id, cfg.instrument_id, ids["take_profit"], exit_side, quantity,
                Price(level, self.instrument.price_precision), TimeInForce.GTC, False, self._margin, False,
                UUID4(), now, **link("take_profit")), "take_profit", "LIMIT",
                f"Take-profit: resting {exit_word} at {level:,.6g}, {tp:.1%} {'above' if side > 0 else 'below'} the "
                f"{self._entry_px:,.6g} entry"
                + (f" ({cfg.take_profit_r:g}R after costs)" if cfg.take_profit_r else "")
                + "; fills at that level",
                {"entry_px": round(self._entry_px, 8), "take_profit": round(tp, 6), "limit_px": round(level, 8)}))
        for order, intent, kind, reason, signal in orders:
            self.decisions[str(order.client_order_id)] = {"intent": intent, "reason": reason, "signal": signal}
            if self.runtime is not None:
                self.runtime.on_order(order_id=str(order.client_order_id), side="SELL" if side > 0 else "BUY",
                                      qty=float(qty), intent=intent, reason=reason, signal=signal, order_type=kind)
        for order, *_ in orders:
            self._sent.append(order.client_order_id)
            self.submit_order(order)

    def _resting_exits(self) -> dict:
        """The backtest's stop and target resting at the venue, by intent."""
        out = {}
        exit_side = OrderSide.BUY if self._entry_side < 0 else OrderSide.SELL
        for order in self._working():
            intent = self.decisions.get(str(order.client_order_id), {}).get("intent")
            if order.side == exit_side and intent in ("stop_loss", "take_profit"):
                out[intent] = order
        return out

    def _resize_exits(self, resting: dict, plan: dict) -> None:
        """A later slice of the entry filled: grow the resting stop and target to the whole position, at
        levels from the new average entry. An exit the venue hasn't answered yet (sent, or mid-resize) is
        resized once it does (_resize_if_due): skipping it left most of a position with no stop."""
        qty = self._position_qty().quantize(self._lot(), rounding=ROUND_DOWN)
        if qty < self._min_qty():
            return
        quantity = Quantity.from_decimal_dp(qty, self.instrument.size_precision)
        for intent, order in resting.items():
            frac = plan.get(intent)
            if not frac:
                continue
            if order.status not in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                self._resize_due = True  # in flight or mid-resize: resized again once the venue answers
                continue
            side = self._entry_side or 1
            level = self._entry_px * (1 - side * frac if intent == "stop_loss" else 1 + side * frac)
            px = Price(level, self.instrument.price_precision)
            if ((order.quantity == quantity or (intent == "take_profit" and "stop_loss" in resting))
                    and (order.trigger_price if intent == "stop_loss" else order.price) == px):
                continue
            if intent == "stop_loss":
                self.modify_order(order.client_order_id, quantity=quantity, trigger_price=px)
            elif "stop_loss" in resting:
                # Linked one-updates-the-other, the venue gives the target the stop's new quantity itself;
                # sending it here too makes the two resizes undo each other, back and forth.
                if order.price != px:
                    self.modify_order(order.client_order_id, price=px)
            else:
                self.modify_order(order.client_order_id, quantity=quantity, price=px)
            what = "Stop-loss" if intent == "stop_loss" else "Take-profit"
            self._resizing[str(order.client_order_id)] = (f"{what} resized to {qty.normalize():f} at {level:,.6g} as "
                                                          f"more of the entry filled (average entry {self._entry_px:,.6g})")

    def on_order_updated(self, event) -> None:
        """The venue took a resize (_resize_exits): journal the new quantity once it holds, not when asked,
        since the order can fill before the venue gets to it."""
        coid = str(event.client_order_id)
        why = self._resizing.pop(coid, None)
        exit_ = self.decisions.get(coid, {}).get("intent") in ("stop_loss", "take_profit")
        if (why is not None or exit_) and self.runtime is not None:  # a linked target follows its stop's size
            self.runtime.store.update_order(coid, qty=event.quantity.as_double(), message=why)
        self._resize_if_due()

    def on_order_modify_rejected(self, event) -> None:
        self._resizing.pop(str(event.client_order_id), None)
        self._resize_if_due()

    def _resize_if_due(self) -> None:
        """Backtests: an entry slice filled while an exit couldn't be resized; size it to the position now."""
        if self._resize_due and self._backtest and self._pending_exit is None:
            self._resize_due = False
            self._rest_exits()

    def on_stop(self) -> None:
        self._drop_kept()
        for coid in list(self._maker):
            self._cancel_alert(coid)
        self.cancel_all_orders(self._cfg.instrument_id)
        if self.runtime is not None:
            self.clock.cancel_timer("sleeve-tick") if "sleeve-tick" in self.clock.timer_names() else None
            self.runtime.on_stop()
        if self.recorder is not None:
            self.recorder.close()
