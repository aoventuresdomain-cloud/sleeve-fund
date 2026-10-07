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
from nautilus_trader.model import (Bar, BarType, ClientOrderId, InstrumentId, Money, OrderSide,
                                   OrderStatus, OrderType, Price, PriceType, Quantity, StopMarketOrder, TimeInForce,
                                   TriggerType)
from nautilus_trader.trading import Strategy

from sleeve_fund import bars as bar_rule
from sleeve_fund import markets, open_risk, risk
from sleeve_fund.data import bar_minutes
from sleeve_fund.instruments import BOOK_SHARE, lot_decimals, pair_of, taker_slippage, target_fill_px
from sleeve_fund.paper.runtime import ENTRY_CANCELLED, EXITS_ONLY, RACED_FILL, RESUMABLE, WIPED_OUT, block_codes, liquidation_reason
from sleeve_fund.store import DUST, OPEN_ORDER_STATUSES, replay_book
from sleeve_fund.strategies.indicators import AtrSma, warmup_for
from sleeve_fund.strategies.timeframes import Candle, SlowerCandles, bar_spec, span

# Orders the signal asks for may wait for a maker fill; protective exits (stop-loss, take-profit,
# risk halts, PM flatten) always go at market, because getting out matters more than the fee.
MAKER_INTENTS = ("entry", "exit", "rebalance")
OPENING_INTENTS = ("entry", "rebalance")  # every other order only ever reduces a position
EXIT_LEGS = ("stop_loss", "take_profit")  # the resting exits a backtest keeps through a reconcile halt
# Exits after which the side they closed isn't entered again until the signal has moved off it (_exit_lock).
LOCKING_INTENTS = (*EXIT_LEGS, "liquidation", "liquidation_cut")
# The resting stops a backtest keeps: the model's own, and the risk guard's (_rest_risk_stop).
STOP_INTENTS = ("stop_loss", "risk_halt", "risk_pause", "liquidation_cut", "liquidation")
MINUTE_NS = 60_000_000_000
DAY_NS = 86_400_000_000_000
SAFETY_STOP_SHARE = 0.5  # of the way from the mark to the liquidation price (_safety_stop_on_restore)
# The status reason the supervisor gives a refused start that still holds a position: it runs for its exits only
# (EXITS_ONLY, from the runtime).


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
    # False keeps a model off the dashboard's pickers and the Development tab: the rule builder needs a
    # definition, and until the form can pick one a bare entry would only start a strategy that refuses to run.
    listed: bool = True


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
STALE_AT_START = "no trade or quote yet since this process started. It clears when data resumes"
# P1-SG15 (Advisor 7 Oct 05:01): while an opening order rests at the venue, the gate is read this often whatever the feed
# does, so a block's cancel goes out within it: a resting entry fills on at most a print inside it, never past 1 s.
GATE_WATCH_SECONDS = 0.5
GATE_WATCH = "entry-gate-watch"
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


def _aware(ts: datetime) -> datetime:
    """A journal time with its zone (a stored time without one is UTC)."""
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


# A decision this long after its bar's close (a bar missed over a restart, m13-E3) opens or adds to nothing: the
# price has moved on from the signal. Exits and reductions always run, however late (Independent Quant Advisor,
# 5 Oct 2026). The same limit the hub client puts on a refilled bar.
LATE_DECISION_NS = 90 * 1_000_000_000
# Paper: this long with no trade while in a position is venue time no price reached the strategy for (QA P1-L1).
# A quiet market's pause that long only replays minutes whose prices the live checks saw already.
UNSEEN_GAP_NS = 15 * 1_000_000_000
MISSING_KEEP_NS = 6 * 60 * 60_000_000_000  # how long a minute the feed missed is still replayed if it lands
HOLD_NS = 20 * 1_000_000_000  # hub_client.HOLD_SECONDS: a minute this late was held for a refill that didn't come


def late_bar(bars: list, now_ns: int, step_ns: int, last_alive: datetime | None, last_order_at: datetime | None):
    """m13-E3: the latest warm-up bar, when it closed while the strategy was down and is still the latest
    closed bar, so it is decided on once rather than only warmed up on. last_alive: the previous process's last
    heartbeat. None (warm-up only) for a first start, when that process was still alive at the close (it saw
    the bar), when an order was journaled at or after the close (it acted on it), or when the bar is a bar or
    more old (a newer bar decides instead). Decided more than LATE_DECISION_NS after its close, it can only
    exit or reduce (_late_entry)."""
    if not bars or last_alive is None:
        return None
    last = bars[-1]
    if now_ns - last.ts_event >= step_ns or _ns(last_alive) >= last.ts_event:
        return None
    if last_order_at is not None and _ns(last_order_at) >= last.ts_event:
        return None
    return last


def outage_fill_note(decision: dict | None, px: float) -> str | None:
    """For the first fill of an exit sent because its level was crossed while the strategy or its market data
    was away (_replay_missed): the price it is booked at, from that level, and the market's price on return.
    None for any other fill, or a later one."""
    signal = (decision or {}).get("signal") or {}
    level = signal.pop("outage_level", None)
    if level is None:
        return None
    what = decision.get("intent", "exit").replace("_", "-")
    back = signal.get("market_on_return")
    return (f"The {what} is booked at {px:,.6g}, {px / level - 1:+.2%} from its {level:,.6g} level, as the venue "
            f"would have filled it in the minute to {signal.get('breached_at', '?')}, reached "
            f"{signal.pop('outage_while', 'while the strategy was down')}"
            + (f"; the market was {back:,.6g} on return" if back else ""))


# Advisor NA-1 to NA-3 (6 Oct 2026, 16:53): minutes no price reached the strategy for are replayed as the venue's
# resting orders would have traded them. Within one minute the adverse side goes first: the venue's liquidation
# and resting stop, then this strategy's own risk guard, and the target only after them.
_VENUE_FIRST = ("liquidation", "stop_loss")
_GUARD_ORDER = ("risk_halt", "risk_pause", "liquidation_cut")  # as the tick judges: the halt or pause, then the cut


def replay_missed(minutes, side: int, venue: dict, guards: dict, target: float | None,
                  target_px: float | None = None):
    """The first exit over `minutes` ((close ns, open, high, low), in time order) for a position on `side`, as
    (intent, booked price, level, close ns, the minute's worst price for the position), or None.

    venue: {"liquidation": price, "stop_loss": price}, what the venue does on its own; guards: {"risk_halt",
    "risk_pause", "liquidation_cut": price}, the levels where this strategy's risk guard acts; target: the
    take-profit price, booked at target_px (its level less the taker's slippage, Advisor L12; the level itself
    when not given). In each minute: an open already past the liquidation price or the stop fills there at the
    open (a gap); an open at or through the target takes the target. Then the low (long) or high (short): the stop,
    which always sits short of liquidation, at its level, or else the liquidation price; else the guard, at the
    level of the first one the tick would act on. Then the target, when the price reached it (a market order on
    touch), never better than target_px."""
    target_px = target if target_px is None else target_px
    for close, o, h, lo in minutes:
        worst, best = (lo, h) if side > 0 else (h, lo)
        for intent in _VENUE_FIRST:
            level = venue.get(intent)
            if level is not None and side * (o - level) <= 0:
                return intent, o, level, close, worst
        if target is not None and side * (o - target) >= 0:
            return "take_profit", target_px, target, close, worst
        for intent in ("stop_loss", "liquidation"):
            level = venue.get(intent)
            if level is not None and side * (worst - level) <= 0:
                return intent, level, level, close, worst
        for intent in _GUARD_ORDER:
            level = guards.get(intent)
            if level is not None and side * (worst - level) <= 0:
                return intent, level, level, close, worst
        if target is not None and side * (best - target) >= 0:
            return "take_profit", target_px, target, close, worst
    return None


def _hhmm(ns: int) -> str:
    return f"{datetime.fromtimestamp(ns / 1e9, tz=timezone.utc):%H:%M}"


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
        trade_from: int | None = None,
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
        if market not in markets.MARKETS and market != markets.PERP_FUNDING_STRESS:  # research only
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
        # Backtest only: no trade on a bar that closes before this time (ns), so a study's out-of-sample window starts
        # flat (Advisor, P1-D13 19:00 and 19:16). The bars before it only warm the model up: its indicators and its own
        # state move on, so a crossover rule still waits for its next cross after it.
        self.trade_from = trade_from
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


def pd_ts(t):
    import pandas as pd

    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def funding_snap_note(settled, ts) -> str:
    from sleeve_fund import funding

    return funding.snap_note(settled, ts)


class LongFlatStrategy(Strategy):
    """Holds a share of the sleeve between 0% and 100%, never short. Subclasses implement
    want_long() for all-or-nothing, or target_weight() for anything in between."""

    REENTER_AFTER_EXIT_LEG = False  # True: a stop or target ends the model's leg, not locks its side (_lock_exit_leg)

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
        self._opened_seq = -1  # backtests: the venue's bar (fee_model.bar_seq) the last entry or add filled in
        self._entry_qty = 0.0  # its size, unsigned
        self._entry_side = 0  # +1 long, -1 short, 0 flat
        # Perp only. A signal that turns a long short (or the other way) closes first; the new side opens
        # as soon as the close has filled, as (side, bar, reason, values).
        self._flip: tuple | None = None
        self._last_submitted: str | None = None  # the client order id _submit last sent (_exit_at_market links it)
        # Paper on a perp: the sandbox's margin account can't be given a position at start, so a position
        # carried over a restart is bought or sold again there, at no fee and unjournaled ("restore"),
        # before anything else trades. _cash_adj then keeps the account's cash equal to the journal's:
        # the restore filled at today's price, not the entry, and funding is the journal's alone.
        self._restore: dict | None = None
        self._restore_id: str | None = None
        # The open perp position's liquidation price, worked out once at each entry or add from the position's own
        # average entry and posted margin, journaled on that order and read back after a restart: never from the
        # restore's price (Advisor 22:36, QA P1-L22).
        self._liq_px: float | None = None
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
        self._settled_idx = None  # backtest: their snapped settlement times (markets.snapped), once
        self._funding_fallback_said = False  # backtest: the baseline for a missing settled rate is said once
        # Paper: the settlement whose rate the venue hadn't published by its check, watched until it arrives, and
        # when it was last asked for (funding_stale / funding_stale_cleared, per instrument, once per episode).
        self._funding_missing: set = set()  # every settlement charged the baseline whose rate hasn't come yet
        self._funding_last_settled = None  # paper: the newest settlement charged at the venue's own rate
        self._funding_paid: dict = {}  # paper: settlement -> (qty, price, rate, amount) charged the baseline
        self._funding_recheck = None
        # (time, amount, kind) for every funding payment, for a backtest's equity: kind "settled" (the venue's rate) or
        # "baseline" (missing, charged adversely).
        self.funding_log: list[tuple] = []
        self.funding_notes: dict = {}  # settlement -> its audit note (a snapped record), for BacktestResult.funding
        # (time, the venue's rate missing, a position held) for every settlement: how much of a backtest's funding is
        # the venue's own (Advisor, 6 Oct 2026, QA P1-O17). Flat settlements are marked in backtests only.
        self.funding_marks: list[tuple] = []
        # (time, amount) for every credit that only brought a flat book's equity back to zero (_cover_shortfall)
        self.insurance_log: list[tuple] = []
        # GAP-LIQ-CAP: each liquidation order's booking, {order id: (bankruptcy price, its liquidation fees)}, which a
        # backtest's fills report takes in place of the venue's (research.runner); and the one being closed now
        self.liquidation_books: dict[str, tuple[float, float]] = {}
        self._liq_closing: dict | None = None
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
        self._slower: list[SlowerCandles] = []  # slower candles the model reads (slower(), v2 P1-4)
        self._short_history: str | None = None  # why the slower candles' warm-up isn't met yet: no new entries
        # After a restart: the journal's last entry and the exit after it that locked re-entry, which the warm-up
        # bars since are decided on again to rebuild the model's leg (_plan_resume, _replay); None once done.
        self._resume: dict | None = None
        self._resume_entry_ns: int | None = None  # set by _plan_resume
        self._pending_exit = None  # a sell waiting for every working order to close first
        self._sent: list = []  # client order ids of orders sent, until the venue has them (see _unsent)
        self._cancel_on_accept: set[str] = set()  # orders to cancel as soon as the venue has them
        self._exec_type = None  # backtest: the shorter bars the decision bars are built from
        # Volumes of the last day's decision bars, for the participation cap on buys.
        self._volumes: deque[float] = deque(maxlen=max(1, 1440 // bar_minutes(config.bar_type)))
        self._noted: set[str] = set()  # warnings already logged; each is said once until it clears
        self._journal_intents: dict[str, str] | None = None  # open orders' intents from before a restart
        self._gated_fills: set[str] = set()  # orders whose fill came in while nothing may open (CHOKE): one incident each
        # Paper's stop as the journal shows it: (journal order id, (side, qty, level)) of the open stop_loss row that
        # stands for the stop watched in the process (_sync_watched_stop), or None.
        self._watched: tuple[str, tuple] | None = None
        self._watched_n = 0  # watched-stop rows written by this process, for their order ids
        self._last_market_ns: int | None = None  # the latest trade or quote, for the price watchdog
        self._market_since_start = False  # whether this process has had a trade or quote yet (P1-SG21)
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
        # how many of its entries the limit would have refused, and the largest share of equity one of them put at
        # risk; a backtest doesn't gate on it.
        self._daily_atr: dict[int, float] = {}
        self.open_risk_binds = 0
        self.open_risk_max = 0.0
        # After a restart, a position whose stop couldn't be restored, or one started for its exits only, works to a
        # safety stop, and no new entry opens (_safety_stop_on_restore).
        self._safety_stop = False
        self._safety_why = ""  # why the safety stop was set, for its incident
        self._safety_pending: tuple[float, float] | None = None  # (cash, qty) at the restart, until the first price
        self._exits_only = False  # started only to run a held position's exits (EXITS_ONLY)
        self._exits_why: str | None = None  # why, when the strategy found it itself (_refused_to_trade)
        self._resizing: dict[str, str] = {}  # backtest exits asked to resize, with why, until the venue confirms
        self._resize_due = False  # an entry slice filled while an exit was in flight (_resize_exits)
        self._resized_ns = None  # when the exits were last resized (_rest_exits)
        self.fee_model = None
        # Backtests on bars only, or on execution bars longer than a minute: what traded first inside a bar is unknown,
        # so resting exits are booked pessimistically (ScheduleFeeModel.exit_price) and the result is labelled when that
        # was relied on (P1-D13). Set by the runner.
        self.pessimistic = False
        self._entry_fill_ns = None  # when the position last opened or added (a resting entry may fill inside a bar)
        self._book_stop_at = None  # an entry stopped out inside its own bar: the price its stop-out is booked at
        self._book_at: dict[str, float] = {}  # those stop-outs' orders and their prices, for the fee model
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
        # Every bar with minutes missing, degraded or not: close time (ns) -> minutes missing, so the slower candles
        # built from it count them (Independent Quant Advisor 6 Oct 16:40, 4.2). Given by mark_missing().
        self._bar_missing: dict[int, int] = {}
        self._missed_said: dict[int, int] = {}  # per slower candles (index): the latest missing one journaled
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
        # Paper (v2 P1-2): the decision bar's close and arrival (ns) while on_bar decides, for each order's timing.
        self._deciding: tuple[int, int] | None = None
        self._late: Bar | None = None  # paper, m13-E3: a bar that closed while the strategy was down (late_bar)
        self._lag: int | None = None  # ns: this decision came more than LATE_DECISION_NS after its bar's close
        self._late_skips: list[int] = []  # the closes of the late bars an opening was skipped on, this run
        self._last_alive: datetime | None = None  # paper: the previous process's last heartbeat
        self._last_seen: datetime | None = None  # paper: the last market data the previous process saw
        self._outage_check = False  # paper: check the open position's stop and target over the outage first
        # Paper: the minutes no price reached the strategy for are replayed for the open position (_replay_missed).
        # The booking the next exit carries (book_px and the market on return), until _submit takes it; the
        # close of the latest minute replayed or seen in a position (nothing at or before it is replayed again);
        # the latest trade's venue time; spans of venue time with no trade while in a position, (from, to); and,
        # after such a span, (its end, when to stop waiting) while its minutes are awaited before the live checks.
        self._outage_book: dict | None = None
        self._replayed_to = 0
        self._trade_ns: int | None = None
        self._unseen: list[tuple[int, int]] = []
        self._awaiting: tuple[int, int] | None = None
        # Paper, hub-fed 1-minute bars (QA P1-L14, L15, L17): a minute that came after missing ones, held from the
        # indicators and decisions until they land (_hold_for_missing); the missing minutes' closes; and those a
        # minute was decided on without (no entries until they are refilled and out of the indicators' reach).
        self._waiting: Bar | None = None
        self._waiting_after = 0  # the last minute decided on before the held one
        self._missing: set[int] = set()
        self._gap: set[int] = set()
        self._lost_told: int | None = None  # the first of the hub's lost minutes said (HubStatus.lost)
        self._released: int | None = None
        # Funding settlements after this time are skipped: (when, credits only). QA P1-L19: an outage's replayed
        # exit closed the position then, though its order goes later (all skipped); P1-L11: the bar a backtest's
        # target traded in, when is unknown (credits only, D9).
        self._funding_skip: tuple[datetime, bool] | None = None
        # An outage's replayed exit: when the venue's order closed the position (_held_at).
        self._replayed_close: datetime | None = None
        self._fills_read: tuple[int, list] | None = None  # (last fill id, the journal's fills then): _position_at
        self._filling = 0.0  # a fill the venue has booked and the journal not yet: on_order_filled, _apply_funding
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
        # Backtest (set by the runner): decision bars fed before the window, without trading, as paper's history
        # warm-up is; and the warm-up the model says its indicators need to settle (warmup_needed), so a decision
        # taken on fewer bars is flagged "unsettled" (Independent Quant Advisor, 6 Oct, 5.1).
        self.preload: list | None = None
        self.settle_bars_needed: int | None = None
        self._decision_bars = 0
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

    def mark_missing(self, bars: dict[int, int]) -> None:
        """Bars, by close time in ns, built with some of their minutes missing, degraded or not, with how many: the
        slower candles built from them add these up (v2 P1-4, Advisor 4.2). Entries are held only on the degraded
        ones (mark_degraded)."""
        if self._slower:  # only slower candles read them
            self._bar_missing.update({int(ts): int(m) for ts, m in bars.items() if int(m) > 0})

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
        if self.hub_status is not None and self.hub_status.missing:
            if self._slower:
                self._bar_missing.update(self.hub_status.missing)
            self.hub_status.missing.clear()
        if str(bar.bar_type) != str(self._cfg.bar_type).split("@")[0]:
            return False
        if self._backtest:
            if self._built is None or bar.ts_event in self._built:
                return False
            self.log.info(f"bar {bar} dropped: none of its execution bars exist, so it isn't built (board 5a)")
            return True
        if self.hub_fed or not str(bar.bar_type).endswith("INTERNAL"):
            return False
        minutes = bar_minutes(self._cfg.bar_type)  # 1-minute bars too: a minute with no data isn't built (P1-4 5a)
        end = bar.ts_event // MINUTE_NS
        seen = sum(1 for m in self._minutes_seen if end - minutes <= m < end)
        self._minutes_seen = {m for m in self._minutes_seen if m >= end}
        if seen == 0:
            if not self._backtest and self.gap_loader is not None:
                return False
            self.log.info(f"bar {bar} dropped: no data in any of its minutes, so it isn't built (board 5a)")
            return True
        missing = minutes - seen
        if missing > 0 and self._slower:
            self._bar_missing.setdefault(bar.ts_event, missing)
        if bar_rule.degraded(missing, minutes):
            self._degraded.setdefault(bar.ts_event, missing)
        return False

    def _cannot_open(self, what: str) -> bool:
        """Whether the strategy's status holds every entry. In paper, the entry the signal wanted is journaled as a
        refused order, one decision row each, naming why (Advisor 22:29 (3)); a backtest has no PM to read them."""
        if self.runtime is None or self.runtime.can_open():
            return False
        if not self.runtime.backtest:
            why = self.runtime.entry_blocked()[1]
            self.runtime.refused(why or f"it is {self.runtime.status}", f"{what} would open the position")
        return True

    def _entry_blocked(self, bar: Bar, what: str = "an entry") -> bool:
        """Whether no new entry may be decided on this bar because it is degraded; says why, once per bar."""
        if bar.ts_event != self._no_entry_ts:
            return False
        if self.runtime is not None and not self.runtime.can_open():
            # Halted or paused: nothing would open anyway, so no degraded-bar note (QA P1-D5); the refusal's decision
            # row lists every cause, the degraded candle with the halt (Advisor 00:20 (b)).
            return self._cannot_open(what)
        minutes = bar_minutes(self._cfg.bar_type)
        missing = self._degraded_missing
        self.log.info(f"entry held back on a degraded bar: {missing} of {minutes} minutes missing")
        self._note("degraded_bar", f"Entry held back: this {minutes}-minute bar is missing {missing} of its minutes "
                   "(over 10%), so no new position is opened on it; exits still run", level="info")
        if self.runtime is not None and not self.runtime.backtest:  # its decision row (Advisor 22:29 (3))
            self.runtime.refused(f"the {minutes}-minute candle is degraded ({missing} of its minutes missing)",
                                 "an entry would open the position")
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
        if self.runtime is not None and not self._backtest:
            # P1-SG21 (Advisor STALE-5MIN): a process has no fresh price until its first trade or quote, so a restart on
            # a dead feed holds entries from its first moment instead of opening for 5 minutes on a stale one.
            self.runtime.holds["stale_data"] = STALE_AT_START
        if self.runtime is not None and not getattr(self.runtime, "backtest", False):  # before this process writes a heartbeat
            self._last_alive = self.runtime.store.sleeve(self.runtime.name).heartbeat_at
            self._last_seen = self.runtime.store.last_feed(self.runtime.name)  # its last market data
        self._plan_resume()
        if self.preload:
            preload, self.preload = self.preload, None
            self.on_historical_bars(preload)
        if self._slower and self.history_loader is not None:
            self._warm_slower()  # before the decision bars, which then complete the slower candle forming now
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
                self._outage_check = not self.runtime.backtest and self._last_alive is not None
                if self._margin and not self.runtime.backtest:
                    self._restore = {"qty": book["qty"], "entry": book["entry_px"]}
            if self._margin:
                # Funding owed for times the process was down while a position was held is settled on the
                # first tick, at that tick's price.
                self._funding_since = self._funding_resumes_from(book["qty"])
                if not self.runtime.backtest:
                    self._replayed_close = self._journaled_replayed_close()
            if self.runtime.backtest:
                # A backtest marks and guards from its bars: every execution bar when it is fed shorter
                # bars than it decides on (paper does every 30 s), otherwise every decision bar.
                if self._cfg.bar_type.is_composite():
                    self._exec_type = self._cfg.bar_type.composite()
                    self.subscribe_bars(self._exec_type)
                return
            if self._margin:
                self._rebuild_funding_missing()
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
        self._note_trade(int(tick.ts_event))
        if not self._backtest and self._resting_openers():
            # P1-SG15: a Stop (or any block) accepted between ticks cancels a resting entry on the first trade or quote
            # after it; asked only while an opening order rests, so a strategy without one reads nothing more per trade.
            self._cancel_resting_entries()
        if self._restore is not None:
            self._send_restore()
            return
        if self._outage_check:
            self._outage_check = False
            if self._check_outage_exits():
                self._late = None  # out of the position: the missed bar is warm-up only, nothing to exit
                return
        if self._late is not None:
            self._decide_late()
        if self._kept:
            self._tape(tick)
        if self._still_awaiting():
            self._maybe_tick()  # the stop and target wait for the missed minutes' replay
            self._publish_signals()
            return
        try:
            if self._check_exits(self._last_close):
                return
        finally:
            if self._pending_exit is None:
                self._outage_book = None  # a fallback flag the live check didn't use
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
        if not self._backtest and self._resting_openers():
            self._cancel_resting_entries()  # P1-SG15: as on a trade (on_trade); a quote can fill a resting order too
        if self.runtime is not None:
            self.runtime.on_quote(bid, ask, venue=str(self._cfg.instrument_id.venue))
        if self._restore is not None:
            self._send_restore()

    def _on_exec_bar(self, bar: Bar) -> None:
        """Backtest: value the book and run the risk guard on every execution bar, so a halt or a
        daily-loss pause fires within the decision bar, as paper's 30-second ticks would."""
        self._snap_settlements(bar.ts_event, bar.close.as_double())
        self._last_close = bar.close.as_double()
        if self.pessimistic:
            self._levels_in_bar(bar)
            if self._stopped_in_entry_bar(bar):
                return
        if self._margin:
            self._intrabar_guard(bar)
        if self.clock.timestamp_ns() != self._last_tick_ns:  # the decision bar at this time may have ticked
            self._on_tick()
        self._bar_target(bar)
        self._rest_risk_stop()

    def _maybe_tick(self) -> None:
        if self.runtime is None:
            return
        now = self.clock.timestamp_ns()
        if now - self._last_tick_ns >= self.runtime.tick_seconds * 1_000_000_000:
            self._on_tick()

    def update_indicators(self, bar: Bar) -> None:
        """Override to feed indicators. Called once per bar, historical or live, in time order."""

    def slower(self, minutes: int, *blocks) -> SlowerCandles:
        """Slower candles of this instrument for the model to read, e.g. self.slower(240, Sma(50)) for a 4-hour
        trend average (v2 P1-4): built from the decision bars, each fed to `blocks` once closed, before the
        decision on the bar that closed it. Warm-up loads them from the history store at their own size. Call it in
        __init__ (as rsi_cross does): missing minutes are only counted once a slower candle exists (mark_missing)."""
        from sleeve_fund.venues import VENUES

        profile = VENUES.get(self._cfg.instrument_id.venue.value)
        s = SlowerCandles(minutes, bar_minutes(self._cfg.bar_type), blocks,
                          profile.daily_anchor_minutes if profile is not None else 0)
        self._slower.append(s)
        return s

    def _accept(self, bar: Bar) -> bool:
        # Indicators are fed by hand rather than registered, so warm-up bars and live
        # bars can't double-count. Anything at or before the last bar seen is ignored.
        if bar.ts_event <= self._last_bar_ts:
            return False
        self._last_bar_ts = bar.ts_event
        self._decision_bars += 1
        self._volumes.append(bar.volume.as_double() / self._cfg.volume_scale)
        if self._atr is not None:
            self._atr.update_raw(bar.high.as_double(), bar.low.as_double(), bar.close.as_double())
        if self._lows is not None:
            self._lows.append(bar.low.as_double())
            self._highs.append(bar.high.as_double())
        missing = self._bar_missing.pop(bar.ts_event, None)
        if self._bar_missing and min(self._bar_missing) < bar.ts_event:  # bars never decided on (dropped): forgotten
            self._bar_missing = {t: m for t, m in self._bar_missing.items() if t > bar.ts_event}
        if self._slower:
            missing = self._degraded.get(bar.ts_event, 0) if missing is None else missing
            for s in self._slower:
                s.handle_bar(bar, missing)
            self._journal_missed()
        self.update_indicators(bar)
        return True

    def _journal_missed(self) -> None:
        """Each slower candle with no decision candles at all is recorded missing as it is found, never made up
        (Advisor 5 Oct 19:48; 6 Oct 16:40, 4.5): a journal event, the source of truth, and the Signals tab."""
        for k, s in enumerate(self._slower):
            said = self._missed_said.get(k, 0)
            new = [end for end in s.missed if end > said]
            if not new:
                continue
            self._missed_said[k] = new[-1]
            if self.runtime is None or self._backtest:
                continue
            when = ", ".join(f"{_hhmm(end - s.period)}-{_hhmm(end)}" for end in new[-6:])
            more = f" (and {len(new) - 6} earlier)" if len(new) > 6 else ""
            self.runtime.store.event(self.runtime.name, "warning", "slower_candle_missing",
                                     f"Recorded missing: no {span(s.minutes)} candle {when} UTC{more}. No trades "
                                     "reached the strategy in it, so none is made up and its indicators skip it",
                                     ts=self.runtime.now())

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
            self._bar_missing.pop(bar.ts_event, None)
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
        if self._margin:  # the liquidation price set when it was entered or last added to (_set_liq), if journaled
            # Filled entries only: a later entry that never filled (an expired post-only) must not hide the held
            # entry's price (CR #146).
            recent, filled = store.orders(name, limit=200), {f["order_id"] for f in store.fills(name, limit=500)}
            held = next((o for o in recent if o.get("intent") == "entry" and o.get("side") == opened
                         and o["order_id"] in filled), None)
            self._liq_px = next((o["signal"]["position_liquidation_px"] for o in recent
                                 if held is not None and o["ts"] >= held["ts"] and o.get("side") == opened
                                 and o.get("intent") in OPENING_INTENTS and o["order_id"] in filled
                                 and (o.get("signal") or {}).get("position_liquidation_px") is not None), None)
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
        # Held while nothing may open (halted, paused, stopped, liquidated: a fill that raced the block's cancel):
        # never left unwatched (P1-U35, Advisor 20:56), so it gets the safety stop and an incident too.
        orphan, held = self.runtime.entry_blocked()
        if orphan and set(block_codes(held)) <= {"stale_data"}:
            # Stale data alone is not a block a position was held under: a process holds entries from its first moment
            # until its first trade or quote (P1-SG21), so every restart would otherwise swap the model's restored stop
            # for a safety stop and raise an incident before the feed has had a second to arrive.
            orphan = False
        if not (unrestored or self._exits_only or orphan):
            return
        self._safety_why = ("its stop couldn't be restored after the restart" if unrestored else
                            f"is held while nothing may open ({held})")
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
        # A stop never loosens (QA SG4): a safety stop an earlier restart set for this same position (no fill since)
        # stays when it is tighter than one measured from today's mark, even if the price is already through it.
        before = self.runtime.last_watched_stop() if not self._backtest else None
        held = before is not None and side * (before - level) > 0
        if held:
            level = before
        restored = entry * (1 - side * self._stop_frac) if self._stop_frac is not None else None
        kept = restored is not None and side * (restored - level) >= 0  # the restored stop is the tighter one
        if not kept:
            self._stop_frac = side * (1 - level / entry)
            self._stop_basis = (f"safety stop at {level:,.6g}, set before the last restart" if held else
                                f"safety stop, half way from the {mark:,.6g} mark to "
                                + (f"the liquidation price {liq:,.6g}" if liq is not None else "zero"))
        why = (f"started for its exits only ({self._exits_reason()})" if self._exits_only else self._safety_why)
        work = (f"it keeps its restored stop at {restored:,.6g}, tighter than a safety stop at {level:,.6g}" if kept
                else f"it keeps the safety stop at {level:,.6g} set before this restart (a stop never loosens)"
                if held else f"it works to a safety stop at {level:,.6g}, half way from the {mark:,.6g} mark to "
                + (f"the liquidation price {liq:,.6g}" if liq is not None else "zero"))
        head = f"Incident, {self.runtime.name}: the open {_side_word(side)} position of {abs(qty):.12g} (entry "
        self.runtime.incident_once(
            head, f"{head}{entry:,.6g}) {why}; {work}. No new entries or adds open"
            + ("" if self._exits_only else " until the model's own stop is set again")
            + "; the position is not closed.")
        self._sync_watched_stop()
        if self._backtest and not kept:
            self._rest_exits()  # a backtest's stop rests at the venue, so a gap through it fills at the open

    def _sync_watched_stop(self) -> None:
        """Paper watches its stop on every trade in the process (_check_exits), so no stop order rests at its venue.
        The journal still shows it, as the resting stop_loss order it stands for (P1-U35: a position is never shown
        unwatched): one open row while a position is held with a stop, replaced when the level or size changes, and
        cancelled when the position closes (a stop that fires sends its own market stop-loss, journaled as usual). A
        backtest's stop is a real resting order (_rest_exits)."""
        if self._backtest or self.runtime is None:
            return
        want = None
        if self._entry_px is not None and self._stop_frac is not None and self._entry_qty > 1e-12:
            side = self._entry_side or 1
            want = ("SELL" if side > 0 else "BUY", round(self._entry_qty, 12),
                    round(self._entry_px * (1 - side * self._stop_frac), 8))
        if (self._watched[1] if self._watched else None) == want:
            return
        store, name = self.runtime.store, self.runtime.name
        if self._watched is not None:
            store.update_order(self._watched[0], status="canceled",
                               message="replaced by the stop as it is now" if want else "the position it guarded closed")
        self._watched = None
        if want is not None:
            # Unique within the 64 characters an order id has, however often it is replaced in one instant.
            self._watched_n += 1
            oid = f"{name[:24]}-watched-stop-{time.time_ns() // 1000}-{self._watched_n}"
            incident = self.runtime.position_incident()  # the safety stop's, or a raced fill's, when there is one
            store.record_order(name, order_id=oid, side=want[0], qty=want[1], intent="stop_loss",
                               reason=f"Stop at {want[2]:,.6g}, watched on every trade: paper keeps it in the process and "
                               "sends a market stop-loss when the price reaches it",
                               signal={"stop_px": want[2], "watched": True,
                                       **({"incident": incident} if incident is not None else {})},
                               order_type="STOP (watched)",
                               ts=self.runtime.now())
            store.update_order(oid, status="accepted")
            self._watched = (oid, want)

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
            if equity > 0:
                self.open_risk_max = max(self.open_risk_max, risk / equity)
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
        self._resume_entry_ns = None  # the close of the candle the journal's last entry was decided on
        if (self.runtime is None or self.runtime.backtest
                or type(self).resume_leg is LongFlatStrategy.resume_leg):
            return
        step = bar_minutes(self._cfg.bar_type) * MINUTE_NS
        orders = [o for o in self.runtime.store.orders(self.runtime.name, limit=1000)  # newest first
                  if not (o.get("signal") or {}).get("watched")]  # paper's watched stop is no venue order
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
        self._resume_entry_ns = _ns(entry["ts"]) // step * step
        self._resume = {"side": side, "bar": self._resume_entry_ns,
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

    def _lock_exit_leg(self, lock) -> None:
        """A stop or target closed the position inside a candle: the side it closed waits for its signal to move
        off it (_exit_lock), unless the model ends its leg there instead (exit_leg_closed), as the rule builder
        does: it may then enter again at the candle's close (Advisor 22:30), and a backtest counts that."""
        if self.REENTER_AFTER_EXIT_LEG:
            self.exit_leg_closed()
        else:
            self._exit_lock = lock

    def exit_leg_closed(self) -> None:
        """For a model with REENTER_AFTER_EXIT_LEG: its stop or target closed the position."""

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

    def _warm_slower(self) -> None:
        """Each slower timeframe's warm-up (v2 P1-4): its blocks' look-back in candles of its own size from the
        history store, up to the last closed one; the decision bars loaded after this carry on from there. When
        the store can't cover it the model opens and adds nothing until the candles have closed live
        (_short_history, _entry_held), rather than enter on a filter that isn't settled, and says why. Stops,
        targets and exits still run: a position held across a restart is still managed (Independent Quant
        Advisor, 5 Oct)."""
        short = []
        for s in self._slower:
            s.need = need = warmup_for(s.blocks)
            if not need:
                continue
            if s.anchor:
                short.append(f"its {span(s.minutes)} candles start at the venue's day start, and the history store "
                             "builds them from 00:00 UTC, so they warm up live")
                continue
            bar_type = BarType.from_str(f"{self._cfg.instrument_id}-{bar_spec(s.minutes)}")
            try:
                bars, why = self.history_loader(self.instrument, bar_type, need), "the history store has fewer"
            except Exception as exc:  # noqa: BLE001 - said below, with what is missing
                bars, why = [], str(exc)
            s.seed(Candle(b.open.as_double(), b.high.as_double(), b.low.as_double(), b.close.as_double(),
                          b.volume.as_double(), b.ts_event) for b in sorted(bars, key=lambda b: b.ts_event))
            if len(bars) < need:
                short.append(f"its {span(s.minutes)} candles need {need} closed ones of history and {len(bars)} "
                             f"loaded ({why})")
        if short:
            self._short_history = "; ".join(short)
            msg = (f"No new entries: {self._short_history}. Stops, targets and exits still run; entries start once "
                   "those candles have closed live, or after a restart once the history store covers them.")
            self.log.error(msg)
            if self.runtime is not None:
                self.runtime.store.event(self.runtime.name, "error", "warmup_short", msg)

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
        if bars and self.runtime is not None and not self.runtime.backtest:
            last = self.runtime.store.orders(self.runtime.name, limit=1)
            self._late = late_bar(bars, time.time_ns(), bar_minutes(self._cfg.bar_type) * 60_000_000_000,
                                  self._last_alive, last[0]["ts"] if last else None)
            if self._late is not None:  # decided on with the first trade, once the position is back (_decide_late)
                bars = bars[:-1]
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

    @classmethod
    def slower_needs(cls, params: dict) -> dict[int, int]:
        """The slower candles the model reads with these settings (slower()), as {minutes: closed candles its
        warm-up needs}, so a strategy the history store can't warm up is refused when it is created (P1-4)."""
        return {}

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
        doesn't list its conditions and reads no slower candles; a model that only reads slower candles sends their
        notes alone (v2 P1-4). Reads only: nothing the model trades by changes."""
        if type(self).conditions is LongFlatStrategy.conditions:
            if not self._slower:
                return None
            return {"price": None, "bar_ts": self._last_bar_ts or None, "bar_minutes": bar_minutes(self._cfg.bar_type),
                    "long": None, "short": None, "held": 0, "guards": [], "notes": self._slower_notes()}
        price = price if price is not None else self._price()
        sides = {}
        for side, key in ((1, "long"), (-1, "short")):
            rows = self.conditions(side, price if price > 0 else None)
            sides[key] = None if rows is None else [_condition_json(r) for r in rows]
        held = self._entry_side if self._entry_px is not None else 0
        return {"price": price, "bar_ts": self._last_bar_ts or None, "bar_minutes": bar_minutes(self._cfg.bar_type),
                **sides, "held": held, "guards": [_condition_json(r) for r in self.guard_conditions(price)],
                "notes": self._slower_notes()}

    def _slower_notes(self) -> list[str]:
        """The Signals tab's lines on the slower candles (v2 P1-4): the latest closed one when it is degraded, and
        the latest recorded missing in the last day (Advisor 4.1, 4.5)."""
        notes = []
        for s in self._slower:
            last, size = s.last, span(s.minutes)
            if last is not None and last.degraded(s.minutes):
                notes.append(f"The latest {size} candle, to {_hhmm(last.end)} UTC, is missing {last.missing} of its "
                             f"{s.minutes} minutes (over 10%): no entries or additions until a fuller one closes")
            recent = [end for end in s.missed if self._last_bar_ts - end < 86_400_000_000_000]
            if recent:
                notes.append(f"The {size} candle {_hhmm(recent[-1] - s.period)}-{_hhmm(recent[-1])} UTC had no trades: "
                             "recorded missing, none made up" + (f" ({len(recent)} in the last day)"
                                                                  if len(recent) > 1 else ""))
        return notes

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
            # Late, a reversal only closes: the new side would open on a stale signal.
            flip = side != 0 and not self._entry_held(bar, f"{_side_word(side)} entry after closing the "
                                                           f"{_side_word(current)}")
            self._flip = (side, bar, reason, values) if flip else None
            self._late_exit(bar)
            self._sell_all("exit", reason, values)
            return
        if self._entry_held(bar, f"{_side_word(side)} entry"):
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
        if self._cannot_open("a long entry" if side > 0 else "a short entry"):
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
        self._deciding = None
        try:
            self._on_bar(bar)
        finally:
            self._deciding = self._lag = None
        if self._waiting is not None and not self._still_missing():
            self._decide_waiting()  # every minute missing before it has landed: decide on it now, in order

    def _hold_for_missing(self, bar: Bar) -> bool:
        """Paper, hub-fed 1-minute bars (QA P1-L14, L15). A minute after missing ones that the hub client released
        unfilled (HOLD_SECONDS after its close: the hub announced a refill that hasn't come) is held from the
        indicators and decisions until the refill lands (the client sends it late), so the minutes reach the
        strategy in order and the stop is replayed over them, or until the next minute comes. A refill landing
        after that is replayed for the exits only (_late_minute). The live trades keep checking the stop
        meanwhile. True if this bar is held or taken here."""
        if (self._backtest or not self.hub_fed or bar.bar_type != self._cfg.bar_type
                or bar_minutes(self._cfg.bar_type) != 1 or not self._last_bar_ts):
            return False
        ts = int(bar.ts_event)
        if ts == self._released:
            return False  # the held minute, decided on now (_decide_waiting)
        if ts in self._missing:
            self._missing.discard(ts)
            if ts > self._last_bar_ts:
                return False  # in order, before the minute held for it: decided on as usual
            self._late_minute(bar)
            return True
        if ts <= self._last_bar_ts:
            return False  # delivered before: _accept drops it
        if self._waiting is not None:
            self._decide_waiting()  # a newer minute: wait no longer
        if ts - self._last_bar_ts > MINUTE_NS:
            keep = ts - MISSING_KEEP_NS
            self._missing = {t for t in self._missing if t > keep} | set(range(self._last_bar_ts + MINUTE_NS, ts,
                                                                               MINUTE_NS))
            if int(bar.ts_init) - ts >= HOLD_NS:
                # The hub client held it for an announced refill that hasn't come (hub_client.HOLD_SECONDS): wait
                # on. A minute on time after missing ones is after minutes with no trade (or a lost gap message):
                # decided on now, as a backtest does, and a refill of them that lands later is replayed for exits.
                self._waiting, self._waiting_after = bar, self._last_bar_ts
                return True
        return False

    def _still_missing(self) -> bool:
        return any(self._waiting_after < t < self._waiting.ts_event for t in self._missing)

    def _decide_waiting(self) -> None:
        """Decide on the minute held for missing ones. With some still missing it is decided on as it is, but
        opens nothing (QA P1-L17: every degraded state blocks new entries)."""
        if self._still_missing():
            gone = sorted(t for t in self._missing if self._waiting_after < t < self._waiting.ts_event)
            self._gap.update(gone)
            if self.runtime is not None:
                self.runtime.store.event(
                    self.runtime.name, "warning", "data_gap",
                    f"Market data gap: {len(gone)} minute(s) closing {_hhmm(gone[0])}"
                    + (f" to {_hhmm(gone[-1])}" if len(gone) > 1 else "") + " haven't arrived, not even from the "
                    "hub's refill. No new entries until they are refilled and the indicators no longer read them; "
                    "the open position's stop is still checked on every trade")
        bar, self._waiting = self._waiting, None
        self._released = int(bar.ts_event)
        self.on_bar(bar)

    def _blind(self, bar: Bar) -> bool:
        """QA P1-L17 and the L16 condition (HoE, deferring L16 to the DA): minutes the feed missed and the hub hasn't
        refilled (the hub client's HubStatus.lost, or minutes a held one was decided on without), or ones the
        indicators still read without (their warm-up, and the minute before the one decided on, which a cross or
        a change compares with): no new entries until they are refilled and out of that reach. Exits are not
        held."""
        lost = self.hub_status.unfilled(str(self._cfg.instrument_id)) if self.hub_status is not None else []
        if lost and self.runtime is not None and self._lost_told != lost[0]:
            self.runtime.store.event(
                self.runtime.name, "warning", "data_gap",
                f"Market data gap: {len(lost)} minute(s) from {_hhmm(lost[0])}"
                + (f" to {_hhmm(lost[-1])}" if len(lost) > 1 else "") + " were lost while the market data hub was "
                "away and haven't been refilled. No new entries until they are; the open position's stop is still "
                "checked on every trade")
        if lost:
            self._lost_told = lost[0]
            return True
        reach = int(bar.ts_event) - (max(self._cfg.warmup_bars, 1) + 1) * MINUTE_NS  # warm-up, and the minute before
        if any(t in self._missing or t > reach for t in self._gap):
            return True
        if (self._gap or self._lost_told is not None) and self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "info", "gap_cleared",
                                     "New entries allowed again: the minutes the market data feed missed are "
                                     "refilled and no longer read by the indicators")
        self._gap, self._lost_told = set(), None
        return False

    def _late_minute(self, bar: Bar) -> None:
        """A minute the hub refilled after the strategy had decided on a later one: too late for the indicators,
        replayed for the open position's exits as the venue would have traded it (Advisor NA-1)."""
        self.log.info(f"late minute {bar}: replayed for the exits")
        if (self._entry_px is None or self._pending_exit is not None or self._busy() or self.runtime is None
                or not self._replay_watches()):
            return
        self._replay_missed([bar], "while the market data feed was away", since=0)

    def _decide_late(self) -> None:
        """m13-E3: decide on the bar that closed while the strategy was down, once, if it is still the latest.
        A bar since makes it warm-up only. Past LATE_DECISION_NS it only exits or reduces (_late_entry)."""
        late, self._late = self._late, None
        lag = self._now_ns() - late.ts_event
        if lag >= bar_minutes(self._cfg.bar_type) * 60_000_000_000:
            self.on_historical_bars([late])
            return
        self.runtime.store.event(self.runtime.name, "info", "late_bar",
                                 f"Decided on the {_hhmm(late.ts_event)} candle {lag / 1e9:.0f} s after its close: it "
                                 "closed while the strategy was down")
        self.on_bar(late)

    def _now_ns(self) -> int:
        return self.clock.timestamp_ns()

    def _entry_held(self, bar: Bar, what: str) -> bool:
        """True when no entry or addition may be decided on this bar: its slower candles' warm-up isn't met yet
        (v2 P1-4), the latest closed slower candle is degraded (over 10% of its minutes missing: Advisor 6 Oct
        16:40, 4.1), or the decision is late (_late_entry). Exits and reductions are never held."""
        if self._short_history is not None:
            if any(s.count < s.need for s in self._slower):
                self._note("entry_held", f"Skipped a {what} on the {_hhmm(bar.ts_event)} candle: "
                           f"{self._short_history}. Said once until entries open again")
                return True
            self._short_history = None
            self._noted.discard("entry_held")
            if self.runtime is not None:
                self.runtime.store.event(self.runtime.name, "info", "warmup_met",
                                         "Entries open again: its slower candles now have the closed ones they need")
        thin = next((s for s in self._slower if s.last is not None and s.last.degraded(s.minutes)), None)
        if thin is not None:
            self._note("slower_degraded", f"Skipped a {what} on the {_hhmm(bar.ts_event)} candle: the latest "
                       f"{span(thin.minutes)} candle, to {_hhmm(thin.last.end)}, is missing {thin.last.missing} of its "
                       f"{thin.minutes} minutes (over 10%), so nothing new is opened on it; exits still run",
                       level="info")
            return True
        self._noted.discard("slower_degraded")
        return self._late_entry(bar, what)

    def _late_entry(self, bar: Bar, what: str) -> bool:
        """True when this decision is too late to open or add (LATE_DECISION_NS after its bar's close). The first
        skip of a run of late bars is said with its lag, and the run's extent once a bar is on time again
        (_end_late_run), so the fills-against-model check before G2 can count what was skipped (QA P1-L4)."""
        if self._blind(bar):
            if self.runtime is not None:
                self.runtime.store.event(self.runtime.name, "warning", "entry_skipped_gap",
                                         f"Skipped {'an' if what[:1] in 'aeiou' else 'a'} {what} on the "
                                         f"{_hhmm(bar.ts_event)} minute: minutes the market data feed missed haven't "
                                         "arrived, so the indicators lack them (exits still run)")
            return True
        if self._lag is None:
            return False
        self._late_skips.append(int(bar.ts_event))
        if self.runtime is not None and len(self._late_skips) == 1:
            article = "an" if what[:1] in "aeiou" else "a"
            self.runtime.store.event(self.runtime.name, "warning", "late_entry_skipped",
                                     f"Skipped {article} {what} on the {_hhmm(bar.ts_event)} candle: decided "
                                     f"{self._lag / 1e9:.1f} s after its close, past the {LATE_DECISION_NS // 10**9} s "
                                     f"limit for opening (candle close {bar.close.as_double():,.6g}, price now "
                                     f"{self._price():,.6g}); later skips in this run of late candles are summed up "
                                     "once the feed is current")
        return True

    def _end_late_run(self) -> None:
        """An on-time decision after a run of late bars that skipped openings: say the run's extent once."""
        skips, self._late_skips = self._late_skips, []
        if len(skips) > 1 and self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "info", "late_entries_summary",
                                     f"Skipped opening on {len(skips)} late candles, {_hhmm(skips[0])} to "
                                     f"{_hhmm(skips[-1])}; the feed is current again")

    def _late_exit(self, bar: Bar) -> None:
        """An exit or reduction runs however late; a late one is said with its lag and both prices."""
        if self._lag is not None and self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "warning", "late_exit",
                                     f"Exiting on the {_hhmm(bar.ts_event)} candle {self._lag / 1e9:.0f} s after its "
                                     f"close (candle close {bar.close.as_double():,.6g}, price now "
                                     f"{self._price():,.6g})")

    def _check_outage_exits(self) -> bool:
        """Paper, on the first trade after a restart with a position open: the process that was down watched
        nothing, so replay the stored minutes since it last saw market data (_outage_since) as the venue's
        resting orders would have traded them (_replay_missed). Minutes that can't be loaded leave the price on
        return to decide, at market, with the exit flagged as that fallback (Advisor NA-1). True if an exit was
        sent."""
        if self._entry_px is None or self.history_loader is None or self._busy() or not self._replay_watches():
            return False
        since = self._outage_since()
        minute = BarType.from_str(f"{self._cfg.instrument_id}-1-MINUTE-LAST-INTERNAL")
        try:
            bars = self.history_loader(self.instrument, minute, int((time.time_ns() - since) // 60_000_000_000) + 2)
        except Exception as exc:  # noqa: BLE001 - said; the price on return decides, flagged
            self.runtime.store.event(self.runtime.name, "warning", "outage_unchecked",
                                     f"Couldn't replay the minutes the strategy was down for: {exc}; the stop and "
                                     "target are checked at the price now instead, and an exit is flagged so")
            return self._fallback_exit("the minutes the strategy was down for couldn't be loaded")
        self._replayed_to = max(self._replayed_to, since)
        return self._replay_missed(bars, "while the strategy was down")

    def _outage_since(self) -> int:
        """Where a restart's replay starts: the last market data the previous process saw, or its last heartbeat
        if earlier (QA P1-L3: a process keeps its heartbeat through a hub outage until its watchdog gives up).
        With no market data on record, the open position's last fill, the earliest its stop could matter."""
        name = self.runtime.name
        fills = self.runtime.store.fills(name, limit=1)
        fill = _ns(fills[0]["ts"]) if fills else None
        seen = self._last_seen
        latest = min(_ns(self._last_alive), _ns(seen)) if seen is not None else None
        if latest is None:
            return fill if fill is not None else _ns(self._last_alive)
        return max(latest, fill) if fill is not None else latest

    def _replay_watches(self) -> bool:
        """The open position has something a replay can act on: a stop or target, or on a perp its liquidation
        price and the risk guard."""
        return self._margin or self._stop_frac is not None or bool(self._tp_frac)

    def _note_trade(self, ts: int) -> None:
        """Paper: a trade at venue time ts. A stretch over UNSEEN_GAP_NS with no trade while in a position is
        venue time no price reached the strategy for (the hub or its venue away): its minutes are replayed when
        they arrive (_check_unseen_exits), and the live stop and target wait for them until LATE_DECISION_NS
        has passed (QA P1-L1, Advisor NA-1)."""
        last = self._trade_ns
        self._trade_ns = ts if last is None else max(ts, last)
        if last is None or self._backtest or self._entry_px is None or ts - last <= UNSEEN_GAP_NS:
            return
        self._unseen.append((last, ts))
        if self._replay_watches():
            self._awaiting = (ts, self._now_ns() + LATE_DECISION_NS)

    def _still_awaiting(self) -> bool:
        """True while the minutes of a stretch with no trade are awaited, so the live checks wait for the replay.
        Past the wait, the live checks go ahead at the price now, and an exit they send is flagged as that
        fallback (Advisor NA-1)."""
        if self._awaiting is None:
            return False
        if self._entry_px is None:
            self._awaiting = None
            return False
        if self._now_ns() < self._awaiting[1]:
            return True
        self._awaiting = None
        self.runtime.store.event(self.runtime.name, "warning", "outage_unchecked",
                                 f"The minutes the market data feed missed didn't arrive within "
                                 f"{LATE_DECISION_NS // 10**9} s; the stop and target are checked at the price "
                                 "now instead, and an exit is flagged so")
        self._outage_book = {"outage_fallback": "market on return: the missed minutes didn't arrive"}
        return False

    def _fallback_exit(self, why: str) -> bool:
        """The live check at the price now, its exit flagged as the market-on-return fallback (Advisor NA-1)."""
        self._outage_book = {"outage_fallback": f"market on return: {why}"}
        try:
            return self._check_exits(self._last_close)
        finally:
            if self._pending_exit is None:
                self._outage_book = None

    def _check_unseen_exits(self, bar: Bar) -> bool:
        """QA P1-L1: a decision bar holding venue time no price reached the strategy for (a stretch with no
        trade, _note_trade, or one still open as the bar arrives), late or not. Its minutes are replayed
        (_replay_missed): from the history store on bars longer than a minute when it has them all, else the
        bar itself. True if an exit was sent."""
        if self._backtest or self.runtime is None or self._entry_px is None or self._restore is not None:
            return False
        end = int(bar.ts_event)
        start = end - bar_minutes(self._cfg.bar_type) * 60_000_000_000
        spans = list(self._unseen)
        if self._trade_ns is not None and end - self._trade_ns > UNSEEN_GAP_NS:
            spans.append((self._trade_ns, end))
        self._unseen = [(a, b) for a, b in self._unseen if b > end]
        if self._awaiting is not None and end >= self._awaiting[0] - UNSEEN_GAP_NS:
            # Its minutes are all here now, bar a stretch after the bar shorter than UNSEEN_GAP_NS: ordinary trade
            # spacing, not time unseen. A sparse trade landing just before each minute's bar would otherwise re-arm
            # the wait for ever (QA FD-F8).
            self._awaiting = None
        if not any(b > start and a < end for a, b in spans) or self._busy() or not self._replay_watches():
            return False
        return self._replay_missed(self._unseen_minutes(bar, start, end) or [bar],
                                   "while the market data feed was away")

    def _unseen_minutes(self, bar: Bar, start: int, end: int) -> list:
        """The stored minutes of a bar longer than a minute, when the history store has every one of them."""
        if self.history_loader is None or end - start <= 60_000_000_000:
            return []
        minute = BarType.from_str(f"{self._cfg.instrument_id}-1-MINUTE-LAST-INTERNAL")
        try:
            got = [b for b in self.history_loader(self.instrument, minute, (end - start) // 60_000_000_000 + 2)
                   if start < b.ts_event <= end]
        except Exception:  # noqa: BLE001 - the bar itself is replayed instead
            return []
        return got if len(got) == (end - start) // 60_000_000_000 else []

    def _replay_missed(self, bars: list, while_: str, since: int | None = None) -> bool:
        """Advisor NA-1 to NA-3: replay minutes no price reached the strategy for as the venue's resting orders
        would have traded them (replay_missed), and send the first exit they make, booked at the replayed price
        with the market's price now beside it. The stop, target and liquidation close the position; the risk
        guard acts through the tick, as it would have on the worst price. True if an exit was sent."""
        side, entry = self._entry_side or 1, self._entry_px
        venue, guards = {}, {}
        if self._stop_frac is not None:
            venue["stop_loss"] = entry * (1 - side * self._stop_frac)
        target = entry * (1 + side * self._tp_frac) if self._tp_frac else None
        cash = qty = 0.0
        if self._margin and self.runtime is not None:
            equity, cash, qty, _ = self._mark()
            liq = self._liq(cash, qty) if qty else None
            if liq is not None:
                d = self.runtime.profile.min_liquidation_distance
                venue["liquidation"] = liq
                guards["liquidation_cut"] = liq / (1 - d) if side > 0 else liq / (1 + d)
            if qty:
                p, day_open = self.runtime.profile, self.runtime._day_open or equity
                guards["risk_halt"] = (max(self.runtime.peak, equity) * (1 - p.max_drawdown) - cash) / qty
                guards["risk_pause"] = (day_open * (1 - p.daily_loss) - cash) / qty
                guards = {k: v for k, v in guards.items() if v > 0}
        after = self._replayed_to if since is None else since  # since: a late minute, older than those replayed
        rows = sorted((int(b.ts_event), b.open.as_double(), b.high.as_double(), b.low.as_double())
                      for b in bars if int(b.ts_event) > after)
        if rows:
            self._replayed_to = max(self._replayed_to, rows[-1][0])
        # The half spread a backtest charges (the run's measured or assumed one), not the quotes at return: replay ==
        # backtest (Advisor 22:36). The spread at return goes in the journal beside it, as a diagnostic only.
        spread = self._cfg.assumed_half_spread
        book = float(target_fill_px(target, side > 0, spread)) if target is not None else None
        hit = replay_missed(rows, side, venue, guards, target, book)
        if hit is None:
            return False
        intent, px, level, at, worst = hit
        gapped = px != level  # the minute opened past the level (replay_missed books the open), not a touch
        if intent == "stop_loss":
            # Advisor 20:42 (NA-1 replay slippage): a replayed stop is a modelled fill, as the backtest's: its level
            # (or the price that gapped through it) less the taker's slippage, max(half spread, 0.05%), adverse.
            px = float(Decimal(str(px)) * (1 - side * taker_slippage(spread)))
        now = self._price()
        words = {"stop_loss": "stop", "take_profit": "target", "liquidation": "liquidation price",
                 "liquidation_cut": "cut before liquidation", "risk_halt": "drawdown halt's level",
                 "risk_pause": "daily-loss pause's level"}[intent]
        gap = " (it opened past it)" if gapped and intent != "take_profit" else ""
        filled = {"take_profit": "as a market order on touch would have, less the taker's slippage",
                  "stop_loss": "as the venue's stop would have filled it, less the taker's slippage"}.get(
                      intent, "as the venue would have filled it")
        reason = (f"{intent.replace('_', '-').capitalize()} reached {while_}: the price passed the {level:,.6g} "
                  f"{words} in the minute to {_hhmm(at)}{gap}; booked at {px:,.6g}, {filled}, and closing now at "
                  f"about {now:,.6g}")
        self.runtime.store.event(self.runtime.name, "warning", "outage_exit", reason)
        # The booked price is the replay's model of the venue, not a venue fill: fills-against-model leaves it out.
        self._outage_book = {"book_px": round(px, 8), "price_source": "replay_model", "market_on_return": now,
                             "outage_level": round(level, 8), "outage_while": while_, "breached_at": _hhmm(at),
                             "half_spread_booked": spread, "half_spread_live": self._half_spread()}
        closed = datetime.fromtimestamp(at / 1e9, tz=timezone.utc)  # the close of the minute the venue's order filled in
        if intent in EXIT_LEGS or intent == "liquidation":  # read back after a restart (_position_at, CR #179)
            self._outage_book["replayed_close"] = closed.isoformat()
        if intent in EXIT_LEGS or intent == "liquidation":
            # The venue's order closed the position in that minute: no settlement after it is the position's,
            # though the order goes now (QA P1-L19).
            self._funding_skip = (closed, False)
            self._reverse_funding_after(self._funding_skip[0])
            self._replayed_close = self._funding_skip[0]
        try:
            if intent in EXIT_LEGS:
                self._lock_exit_leg(side if self._margin else True)
                self._entry_px = None  # don't fire again while the order is in flight
                self._exit_at_market(intent, reason, {"entry_px": entry, intent: self._stop_frac
                                                      if intent == "stop_loss" else self._tp_frac}, px)
            elif intent == "liquidation":
                self._liquidation_guard(cash, qty, px)
            else:  # the risk guard: the tick judges the equity at the minute's worst price, as it would have
                self._guard_equity = cash + qty * worst
                self._guard_price = px if intent == "liquidation_cut" else None
                self._on_tick()
        finally:
            if self._pending_exit is None:
                self._outage_book = None  # taken by the order, or nothing was sent
        return True

    def _on_bar(self, bar: Bar) -> None:
        if self._late is not None and bar.ts_event > self._late.ts_event:  # a newer bar first: warm-up only
            late, self._late = self._late, None
            self.on_historical_bars([late])
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
        if self._hold_for_missing(bar) or self._hold_gap(bar):
            return
        bar = self._fill_gap(bar)
        if not self._accept(bar):
            return
        missing = self._degraded.pop(bar.ts_event, None)
        if missing is None:
            self._noted.discard("degraded_bar")
            if self.runtime is not None:
                self.runtime.holds.pop("degraded_candle", None)
        else:
            self._no_entry_ts, self._degraded_missing = bar.ts_event, missing
            if self.runtime is not None:
                # CHOKE (Advisor 22:29): while the latest candle is degraded nothing opens, and resting entries are
                # cancelled at once, as for a halt; stops and exits still run on the last good data.
                self.runtime.holds["degraded_candle"] = (
                    f"the latest {bar_minutes(self._cfg.bar_type)}-minute candle is missing {missing} of its minutes. "
                    "The next whole candle clears it")
                self._cancel_resting_entries()
        self._deciding = (int(bar.ts_event), int(bar.ts_init))
        lag = 0 if self._backtest or self.runtime is None else self._now_ns() - bar.ts_event
        self._lag = lag if lag > LATE_DECISION_NS else None
        if self._lag is None and self._late_skips:
            self._end_late_run()
        self.log.info(f"bar {bar}")
        self._last_close = bar.close.as_double()
        if self.pessimistic and self._exec_type is None:
            self._levels_in_bar(bar)
            if self._stopped_in_entry_bar(bar):
                return
        if self._cfg.trade_from is not None and bar.ts_event < self._cfg.trade_from:
            self.want_side(bar) if self._margin else self.target_weight(bar)  # seen, never traded
            return
        if self._margin and self._backtest and self._exec_type is None:
            # Fed only the bars it decides on, a backtest on a perp still judges each bar at its worst price
            # (_on_exec_bar does it for every shorter execution bar), as paper judges every trade.
            self._intrabar_guard(bar)
        if self._exec_type is None and self._target_level(bar) is not None:
            # QA P1-L11 (Advisor D9): the target traded somewhere in this bar, when is unknown, so a settlement in
            # it is charged when it is a cost and not credited when it would be paid to the position.
            self._funding_skip = (self._bar_start(bar), True)
        try:
            self._maybe_tick()
        finally:
            if self._funding_skip is not None and self._funding_skip[1]:
                self._funding_skip = None
        if self._exec_type is None:
            self._rest_risk_stop()
        if self._pending_exit is not None:
            return
        if self._replan_pending is not None and self._entry_px is not None:
            self._replan(self._last_close)
        held = self._pos_side()
        if self._check_unseen_exits(bar):
            return
        if self._exec_type is None and self._bar_target(bar):
            self._flip_after_target(bar, held)
            return
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
            if self._exit_lock or self._entry_blocked(bar) or self._entry_held(bar, "long entry"):
                return
            if self._cannot_open("a long entry"):
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
            self._late_exit(bar)
            self._sell_all("exit", reason, {**values, "close": close})
            self._held_w = 0.0
        elif w > 0 and is_long and self._cfg.rebalance_band is not None:
            if self._held_w is None:  # e.g. after a restart: start from what is actually held
                equity, _, qty, _ = self._mark()
                self._held_w = qty * close / equity if equity > 0 else w
            if abs(w - self._held_w) > self._cfg.rebalance_band * self._held_w:
                if w > self._held_w and (self._entry_blocked(bar, "an addition") or self._entry_held(bar, "addition")):
                    return  # adding to the position is an entry; trimming it still runs
                if w < self._held_w:
                    self._late_exit(bar)
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

    def _bar_target(self, bar: Bar) -> bool:
        """Backtests: the target, judged on a bar the venue has already matched. The stop rests at the venue
        (_rest_exits), so a bar that reached it has sold the position before this runs: within a bar the stop
        goes before the target (Advisor NA-2, as the outage replay does), whichever extreme was nearer the
        open, unless the bar opened through the target (_rebook_as_target). A bar that traded through the target
        sells at market, booked at the target's level, as the resting limit filled, with the taker fee paper's
        market exit pays (the fee model carries the difference: ScheduleFeeModel.booked). True if sent."""
        level = self._target_level(bar)
        if level is None:
            return False
        tp, side = self._tp_frac, self._entry_side or 1
        spread = self.fee_model.half_spread if self.fee_model is not None else self._half_spread()
        book = float(target_fill_px(level, side > 0, spread))
        if self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "info", "take_profit", f"exit at {level:,.4f}, "
                                     f"{side * tp:+.2%} from entry", ts=self.runtime.now())
        reason = (f"Take-profit: the {level:,.6g} target, {tp:.1%} {'above' if side > 0 else 'below'} the "
                  f"{self._entry_px:,.6g} entry"
                  + (f" ({self._cfg.take_profit_r:g}R after costs)" if self._cfg.take_profit_r else "")
                  + f", was reached, so it {'sells' if side > 0 else 'buys'} at market, booked at {book:,.6g}: the "
                  f"target less the taker's {float(taker_slippage(spread)):.2%} slippage (a bar that also reached "
                  "the stop takes the stop first)")
        values = {"entry_px": self._entry_px, "take_profit": tp, "target_px": level}
        self._lock_exit_leg(side if self._margin else True)
        self._entry_px = None
        self._outage_book = {"book_px": book}
        try:
            self._sell_all("take_profit", reason, values)
        finally:
            if self._pending_exit is None:
                self._outage_book = None  # taken by the order, or nothing was sent
        return True

    def _bar_start(self, bar: Bar) -> datetime:
        return datetime.fromtimestamp((int(bar.ts_event) - bar_minutes(self._cfg.bar_type) * MINUTE_NS) / 1e9,
                                      tz=timezone.utc)

    def _target_level(self, bar: Bar) -> float | None:
        """Backtests: the open position's target level when this bar reached it and it is judged on this bar
        (_bar_target), else None."""
        tp, side = self._tp_frac, self._entry_side or 1
        if (not self._backtest or not tp or self._entry_px is None or self._pending_exit is not None
                or self._busy() or self._pos_side() == 0):
            return None
        if self.fee_model is not None and self._opened_seq == self.fee_model.bar_seq:
            return None  # filled inside this bar: its extremes may have come before the fill, so judge from the next
        level = Price(self._entry_px * (1 + side * tp), self.instrument.price_precision).as_double()
        if not (bar.high.as_double() >= level if side > 0 else bar.low.as_double() <= level):
            return None  # a market order on touch: reaching the level is enough (Advisor 19:40)
        return level

    def _flip_after_target(self, bar: Bar, held: int) -> None:
        """QA P1-L10: a perp whose target traded in this bar is flat by its close in paper, which then decides from
        flat and opens the other side on this close if the signal has turned. The backtest sells at this close, so
        the turn waits for that sale, as a reversal's close does (_flip), rather than for the next bar."""
        if not self._margin or self._pending_exit is None or held == 0:
            return
        side = self.want_side(bar)
        if side is None or int(side) != -held or (side < 0 and not self._cfg.allow_short):
            return
        if self._entry_held(bar, f"{_side_word(int(side))} entry after the target"):
            return  # an entry like any other: held while late or while the slower candles are short of history
        reason, values = self.explain(bar, int(side))
        self._flip = (int(side), bar, reason, {**values, "close": bar.close.as_double()})

    def _check_exits(self, price: float) -> bool:
        """Stop-loss / take-profit against the average entry. True if an exit was sent.

        Paper and live call this on every trade, so both levels are watched tick by tick. A
        backtest only sees whole bars: the stop rests at the venue (_rest_exits) and the target is
        judged on each bar after it (_bar_target)."""
        if self._safety_pending is not None and self._entry_px is not None and price > 0:
            self._place_safety_stop(price)
        stop, tp = self._stop_frac, self._tp_frac
        if self._entry_px is None or (stop is None and not tp) or price <= 0:
            return False
        if self._busy():
            return False
        if self._backtest:
            return False  # the stop rests at the venue, the target is judged on the bar (_bar_target)
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
        if hit == "take_profit":  # the level beside the real fill, for fills-against-model (Advisor L12)
            values["target_px"] = round(self._entry_px * (1 + side * tp), 8)
        self._lock_exit_leg(side if self._margin else True)
        self._entry_px = None  # don't fire again while the sell is in flight
        self._exit_at_market(hit, reason, values, price)
        return True

    def _exit_at_market(self, intent: str, reason: str, values: dict, price: float) -> None:
        """Send an exit at market. When it is the stop firing, the journal's watched stop ends "triggered", linked to
        the market stop-loss sent for it (Head of QA and HoE, 7 Oct); it ends "canceled" only when the position closes
        some other way (_sync_watched_stop)."""
        watched = self._watched if intent == "stop_loss" and self.runtime is not None and not self._backtest else None
        if watched is not None:
            self._watched = None  # not the position closing some other way
        self._last_submitted = None
        self._sell_all(intent, reason, values)
        if watched is not None:
            # The order _submit sent, by its id: _sell_all prunes _sent as it goes, so its length can't say (CR on #182).
            oid = self._last_submitted
            self.runtime.store.update_order(
                watched[0], status="triggered",
                message=f"triggered at {price:,.6g}: " + (f"the market stop-loss {oid} closes the position" if oid else
                                                          "its market stop-loss goes once the order in flight is done"))
            if oid is not None:
                self.runtime.store.merge_order_signal(watched[0], {"triggered_order": oid})

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
        # Never more than the journal's cash either, when an exit booked at a replayed price left it lower.
        limits = {"free cash": (free.as_decimal() + Decimal(str(min(self._cash_adj, 0.0)))) * room}
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
        """Send an order with its reason journaled. An order that opens or adds is on record before the venue sees
        it, or not sent (a failed write raises here); paper's queued journal sends an exit or stop whatever
        happens to its row, and an unwritten row is an incident (paper.queued, QA P1-L5). With
        maker_wait_minutes set, signal-driven orders rest as post-only limits first. An order that would open or
        add while nothing may (CHOKE) is not sent, and says why once; returns whether it was sent."""
        if (why := self._gated(side, intent)) is not None:
            self.runtime.refused(why, f"{intent} {side.name.lower()} {qty} would open or add to the position")
            return False
        decided = self.clock.timestamp_ns()
        quantity = Quantity.from_decimal_dp(qty, self.instrument.size_precision)
        wait = self._cfg.maker_wait_minutes
        last, tick = self._price(), self.instrument.price_increment.as_double()
        if self._outage_book is not None and intent not in OPENING_INTENTS:
            signal, self._outage_book = {**signal, **self._outage_book}, None  # a replayed exit's booking
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
        if intent == "stop_loss" and self._book_stop_at is not None and not maker:
            self._book_at[coid], self._book_stop_at = self._book_stop_at, None
            signal["booked_at"] = round(self._book_at[coid], 8)
        self.decisions[coid] = {"intent": intent, "reason": reason, "signal": signal}
        if self._unsettled():
            # Decided before the model's indicators could have settled: kept as the model trades it, and flagged.
            self.decisions[coid]["unsettled"] = signal["unsettled"] = True
        if intent == "entry" and signal.get("breakout_slippage_bp") and self._backtest and self.fee_model is not None:
            # A backtest's fills are at the candle's price; a breakout entry pays this much more (P1-5, board 9b B3).
            self.fee_model.slippage[coid] = Decimal(str(signal["breakout_slippage_bp"])) / 10_000
        if self._backtest and "book_px" in signal and self.fee_model is not None:
            self.fee_model.booked[coid] = (Decimal(str(signal["book_px"])), side == OrderSide.BUY)
        bar_close, bar_recv = self._deciding or (None, None)
        if self.runtime is not None:
            # A decision on a bar is timed from its order's own journal row (a risk stop or restore is not).
            timing = {"bar_close": bar_close, "bar_recv": bar_recv, "decided": decided} if bar_close else None
            self.runtime.on_order(order_id=coid, side="BUY" if side == OrderSide.BUY else "SELL",
                                  qty=float(qty), intent=intent, reason=reason, signal=signal,
                                  order_type="POST-ONLY LIMIT" if maker else "MARKET", timing=timing)
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
        if self.runtime is not None and coid not in self._kept:
            self.runtime.on_timing(coid, sent=self.clock.timestamp_ns())
        if maker:
            self.clock.set_time_alert(f"maker-{coid}", self.clock.utc_now() + timedelta(minutes=wait),
                                      callback=self._maker_timeout)
        self._last_submitted = coid
        return True

    def _unsettled(self) -> bool:
        """Whether a decision now comes before the warm-up the model's indicators need (settle_bars_needed): today's
        hand-coded models trade once an indicator is `initialized`, which can be before it has settled."""
        return self.settle_bars_needed is not None and self._decision_bars < self.settle_bars_needed

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
        why = self._gated(order.side, kept["info"].get("intent"))
        if why:  # P1-SG15: nothing opens from the moment the gate closes, not from the next tick
            self._part_filled(coid, float(kept["sent"]), order.quantity.as_double(), why)
            self._close_kept(coid, f"cancelled: nothing may open now. {why}")
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
        same margin the sizing, the dashboard and the demo copy use (markets.isolated_margin): the one set when
        it was entered (_set_liq), else worked out from its own average entry."""
        if qty and self._liq_px is not None:
            return self._liq_px
        return self._liq_from_entry(cash, qty)

    def _liq_from_entry(self, cash: float, qty: float) -> float | None:
        """From the position's average entry as the strategy (and the journal) keep it, never the simulated
        venue's, which after a restart is the restore's price (QA P1-L22)."""
        lev = self.runtime.profile.max_leverage if self.runtime is not None else 1.0
        entry = self._entry_px if self._entry_px else self._net_position()[1]
        return markets.isolated_liquidation(cash, qty, entry, lev, self._cfg.perp.maintenance_margin)

    def _set_liq(self, order_id: str) -> None:
        """After a perp entry or add fills: the position's liquidation price, journaled on that order beside the
        decision's own estimate from the close (liquidation_px), which stays as it was decided."""
        _, cash, qty, _ = self._mark()
        self._liq_px = self._liq_from_entry(cash, qty) if qty else None
        if self._liq_px is not None and self.runtime is not None and not self._backtest:
            self.runtime.store.merge_order_signal(order_id, {"position_liquidation_px": round(self._liq_px, 8)})

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
                    # fixed for the run, so snapped once, not every hour (QA P1-O17a-17)
                    self._settled_idx = markets.snapped(self._settled) if self._settled is not None and len(self._settled) else None
                settled = self._settled
            else:
                settled = funding.rates(terms.funding_venue, pair_of(self.instrument))
        published = None
        if terms.funding_venue is not None:  # the venue's own interval for the instrument, where it gives one (DA-11)
            from sleeve_fund.venues import venue

            interval = getattr(venue(terms.funding_venue), "funding_interval", None)
            published = interval(pair_of(self.instrument)) if callable(interval) else None
        idx = getattr(self, "_settled_idx", None) if self._backtest else None
        return markets.settlement_times(since, now, terms.funding_hours, settled, published, idx), settled

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
        if self._funding_missing:  # before the deferral below, so trades far apart still clear or reopen (QA P1-O17a-12)
            self._watch_funding_recovery(terms, self.clock.utc_now())
        now = self.clock.utc_now()
        deadline = False
        if not self._backtest:
            # No trade is reaching the strategy, or the minutes it missed are still to come: they may show the
            # venue's stop closed the position before a settlement in them. Settled once they are replayed (QA
            # P1-L19); a settlement after the replayed exit is not the position's (_funding_skip). Held flat or
            # not, as a settlement is owed on the position held at it (_position_at). Missed minutes awaited are
            # always waited for (they land within LATE_DECISION_NS, _still_awaiting); a market only trading
            # sparsely is held to FUNDING_DEFER_MAX (Advisor 7 Oct 00:40, QA FD-F2).
            if self._awaiting is not None:
                return
            if self._trade_ns is not None and self._now_ns() - self._trade_ns > UNSEEN_GAP_NS:
                if not self._funding_overdue(terms, now):
                    return
                deadline = True
        since = self._funding_since
        if since is None:
            self._funding_since = now
            return
        if int(now.timestamp()) // 3600 * 3600 <= since.timestamp():
            return  # no hour's start in (since, now]: settlements fall on the hour
        times, settled = self._settlements(terms, since, now)
        held = [(ts, *self._held_at.get(int(ts.timestamp()) * 1_000_000_000, (self._net_position()[0], price)))
                for ts in times]
        if not self._backtest and held:
            # Paper: a settlement this process saw pass keeps the position it noted then (_snap_settlements); one it
            # didn't (a restart since) is read from the journal; and one the outage replay found the venue's order
            # had closed the position before is held flat (QA FD-F1, P1-L19).
            current = self._net_position()[0] - self._filling  # the journal has none of a fill being booked (FD-F9)
            held = [(ts, self._position_at(ts, current, q if int(ts.timestamp()) * 1_000_000_000 in self._held_at
                                           else None), px) for ts, q, px in held]
        if not any(q for _, q, _ in held):
            self._mark_flat(terms, times, now)
            self._funding_since = self._rescan_from(now)
            return
        for ts, qty, px in held:
            if deadline and now - ts < self.FUNDING_DEFER_MAX:
                return  # a later settlement keeps its own deadline: held until it is that old too (CR #179)
            inside = self._intrabar is not None and self._intrabar[0] < int(ts.timestamp()) * 1_000_000_000 <= self._intrabar[1]
            if qty == 0 or (inside and self._intrabar[2]):  # a gap fill at the bar's open held nothing after it
                self._mark_flat(terms, [ts], now)  # marked flat, so a batch mixing flat and held leaves no hole (CR)
                self._funding_since = ts
                continue
            wait = markets.settlement_wait(ts, settled, self.FUNDING_WAIT, terms.funding_hours)
            rate = self._funding_rate(terms, ts, now, wait)
            if rate is None:  # paper, just after a settlement the venue hasn't published yet: try on the next tick
                return
            rate, baseline = rate
            self._funding_since = ts
            # A missing rate never helps a result: both sides pay the baseline, so a short isn't credited on
            # data the venue never gave (Advisor, 6 Oct 2026, D9; QA P1-O17).
            amount = -abs(qty) * px * abs(rate) if baseline else -qty * px * rate
            if inside and amount > 0:
                continue  # a touch at an unknown time inside the bar: a credit it may not have been held for isn't booked
            if self._funding_skip is not None and ts > self._funding_skip[0] and (amount > 0 or
                                                                                  not self._funding_skip[1]):
                continue  # closed before it (an outage's replayed exit), or a credit the target may have missed
            if amount > 0 and ts == self._replayed_close:
                # The replay's exit filled at an unknown time inside the minute ending at the settlement: the worse
                # outcome, as the backtest of the same minutes books it (rule (c)): a cost is paid, a credit is not
                # (QA FD-F10).
                continue
            self.funding_marks.append((ts, baseline, True))
            note = "" if baseline or settled is None else funding_snap_note(settled, ts)
            self._book_funding(ts, qty, px, rate, amount, "baseline" if baseline else "settled", note)
            if deadline and self.runtime is not None:
                self.runtime.store.event(
                    self.runtime.name, "info", "funding_deadline_booked",
                    f"The {ts:%H:%M} settlement was booked {(now - ts).total_seconds() // 60:.0f} minutes after it, "
                    f"though no trade had reached the strategy in the last {UNSEEN_GAP_NS // 10**9} s: a quiet market "
                    "holds funding back no longer than that", ts=self.runtime.now())
        self._funding_since = max(self._funding_since, self._rescan_from(now))

    def _mark_flat(self, terms, times, now) -> None:
        """Backtest: flat settlements still mark where the venue's rate is missing, so flat time neither breaks a
        stretch without it nor adds to it (funding.baseline_summary; Advisor, 6 Oct 2026)."""
        if self._backtest:
            self.funding_marks += [(ts, self._funding_rate(terms, ts, now, held=False)[1], False) for ts in times]

    def _mark_held(self, ts, baseline: bool) -> None:
        """A settlement charged to what a bar's resting fill opened (_fund_opened_in_bar): marked held, in place of
        the flat mark the position before the fill left, so it is counted once."""
        for i in range(len(self.funding_marks) - 1, -1, -1):
            if self.funding_marks[i][0] == ts:
                if not self.funding_marks[i][2]:
                    self.funding_marks[i] = (ts, baseline, True)
                return
            if self.funding_marks[i][0] < ts:
                break
        self.funding_marks.append((ts, baseline, True))

    def _position_at(self, ts: datetime, current: float, noted: float | None = None) -> float:
        """Paper: the position held at the settlement instant `ts`: `noted` when this process noted it as the
        settlement passed, else the position now less the journal's fills from then on (a close, a reduction or a
        reversal since). Flat when the outage replay found the venue's order closed the position before `ts`, though
        ours, journaled after the settlement, went on return (QA P1-L19)."""
        if self.runtime is None:
            return current if noted is None else noted
        held, last = Decimal(repr(float(current))), None
        store, name = self.runtime.store, self.runtime.name
        seen = store.last_fill_id(name)  # read the journal again only when a fill has been added (CR #179)
        if self._fills_read is None or self._fills_read[0] != seen:
            self._fills_read = (seen, store.fills(name, limit=10_000))
        for f in self._fills_read[1]:  # newest first
            at = f["ts"] if f["ts"].tzinfo else f["ts"].replace(tzinfo=timezone.utc)
            if at < ts:  # a fill at the settlement instant is after it, as the snapshot has it (_snap_settlements)
                last = at
                break
            held -= Decimal(repr(float(f["qty"]))) * (1 if f["side"] == "BUY" else -1)
        closed = self._replayed_close
        if closed is not None and closed < ts and (last is None or last <= closed):
            return 0.0
        if noted is not None:
            return noted
        return 0.0 if abs(held) < self._lot() / 2 else float(held)  # under half a lot is flat

    def _funding_resumes_from(self, qty: float) -> datetime:
        """After a restart: where funding is settled from. The last settlement booked, or the fill that opened the
        position held since (from flat) if later: a settlement between it and the restart that wasn't booked yet is
        charged on the position the journal shows held at it (_position_at), though a fill came after it, or the
        book is flat now (QA FD-F7). Before, the last fill was used, past such a settlement."""
        store, name = self.runtime.store, self.runtime.name
        last = store.funding(name, limit=1)
        booked = _aware(last[0]["ts"]) if last else None
        fills = store.fills(name, limit=10_000)  # newest first
        running, half, opened = Decimal(repr(float(qty))), self._lot() / 2, None
        for f in fills:
            at = _aware(f["ts"])
            if booked is not None and at <= booked:
                break
            before = running - Decimal(repr(float(f["qty"]))) * (1 if f["side"] == "BUY" else -1)
            if abs(before) < half <= abs(running):
                opened = at
                break
            running = before
        if opened is None and booked is None and qty and fills:
            opened = _aware(fills[0]["ts"])  # the journal doesn't reach the opening: its last fill, as before
        marks = [t for t in (booked, opened) if t is not None]
        return max(marks) if marks else self.runtime.now()

    def _journaled_replayed_close(self) -> datetime | None:
        """After a restart: when the last outage replay found the venue's order closed the position, from the replayed
        exit's journaled order (_outage_book), so a settlement after it is still held flat (_position_at, CR #179)."""
        for o in self.runtime.store.orders(self.runtime.name, limit=200):  # newest first
            at = (o.get("signal") or {}).get("replayed_close")
            if at:
                return datetime.fromisoformat(at)
        return None

    def _book_funding(self, ts: datetime, qty: float, px: float, rate: float, amount: float,
                      kind: str = "settled", note: str = "") -> None:
        self._cash_adj += amount
        self.funding_log.append((ts, amount, kind))
        if note:
            self.funding_notes[ts] = note.strip(" ()")  # the snapped record's own stamp, kept for the audit
        if kind == "baseline" and not self._backtest:
            self._funding_paid[ts] = (qty, px, rate, amount)  # reversed if the venue shows it was no settlement
        if self.runtime is not None:
            self.runtime.store.record_funding(self.runtime.name, qty=qty, price=px, rate=rate,
                                              amount=round(amount, 8), ts=ts, kind=kind)
            self.runtime.store.event(self.runtime.name, "info", "funding",
                                     f"Funding {'received' if amount >= 0 else 'paid'}: {abs(amount):,.2f} on a "
                                     f"{_side_word(1 if qty > 0 else -1)} position of {abs(qty):.12g} at "
                                     f"{px:,.6g} ({rate:.4%}){note}", ts=ts)

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
            wait = markets.settlement_wait(ts, settled, self.FUNDING_WAIT, terms.funding_hours)
            rate = self._funding_rate(terms, ts, now, wait)
            if rate is None:
                continue
            rate, baseline = rate
            if baseline:  # a missing rate is paid whichever side is held, at the price where it costs most
                px, amount = high, -abs(qty) * high * abs(rate)
            else:
                px = high if qty * rate > 0 else low  # the price at which it costs most, or credits least
                amount = -qty * px * rate
            if gap or amount < 0:
                self._mark_held(ts, baseline)
                self._book_funding(ts, qty, px, rate, amount, "baseline" if baseline else "settled")

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
    # ...and holds a settlement back for missed minutes at most this long (DA 7 Oct): on a market that trades less often
    # than UNSEEN_GAP_NS the hold would otherwise never end. The same 15 minutes, so a missing rate is alerted, charged
    # and blocks entries at due + 15 minutes however quiet the market (Advisor, 7 Oct 03:13, QA P1-O17a-13).
    FUNDING_DEFER_MAX = FUNDING_WAIT

    def _reverse_funding_after(self, closed: datetime) -> None:
        """Paper: a replay found the venue's order closed the position at `closed`, but a settlement after it was
        already charged (held past FUNDING_DEFER_MAX). Journal it as funding_charged_while_flat, which fills-against-
        model counts, and reverse it with a separate correcting entry at the same settlement time; the original row is
        never edited (Advisor 7 Oct 00:40). A settlement already corrected nets to zero and is left alone. Only a
        perp whose funding was settled past `closed` (this process, or the journal's last row before a restart) can
        have one."""
        since = self._funding_since
        if since is not None and since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        if not self._margin or since is None or since <= closed:
            return
        store = self.runtime.store
        net: dict = {}
        for r in store.funding(self.runtime.name):
            ts = r["ts"] if r["ts"].tzinfo else r["ts"].replace(tzinfo=timezone.utc)
            if ts > closed:
                was = net.get(ts)
                net[ts] = (r if was is None else was[0], (0.0 if was is None else was[1]) + r["amount"])
        for ts, (row, amount) in sorted(net.items(), key=lambda kv: kv[0]):
            if abs(amount) < 1e-9:
                continue
            self._cash_adj -= amount
            # Its own kind, so O17b never trues up a reversed baseline and the books read it as a correction (QA M-1)
            self.funding_log.append((ts, -amount, "reversal"))
            store.record_funding(self.runtime.name, qty=row["qty"], price=row["price"], rate=row["rate"],
                                 amount=round(-amount, 8), ts=ts, kind="reversal")
            # Nothing of the strategy's is owed at it now: it leaves the watch, and the hub alone judges the rate (M-1)
            when = pd_ts(ts)
            self._funding_missing.discard(when)
            self._funding_paid.pop(when, None)
            store.event(self.runtime.name, "warning", "funding_charged_while_flat",
                        f"Funding of {abs(amount):,.2f} {'paid' if amount < 0 else 'received'} at the {ts:%H:%M} "
                        f"settlement was booked before the replay found the position closed at {closed:%H:%M}: "
                        f"reversed by a separate correcting entry of {-amount:+,.2f}", ts=self.runtime.now())

    def _funding_overdue(self, terms, now: datetime) -> bool:
        """A settlement held back for missed minutes is charged anyway once FUNDING_DEFER_MAX has passed and a trade
        from after it has reached the strategy (the feed is back, and its replay has had that long to run)."""
        since = self._funding_since
        due = self._settlements(terms, since, now)[0][:1] if since is not None else []
        return bool(due) and now - due[0] >= self.FUNDING_DEFER_MAX and (
            self._trade_ns is not None and self._trade_ns > int(due[0].timestamp()) * 1_000_000_000)

    def _funding_rate(self, terms, ts, now, wait: timedelta | None = None,
                      held: bool = True) -> tuple[float, bool] | None:
        """The rate settled at `ts` and whether it is the baseline for a missing one: the venue's own where its
        terms name one (sleeve_fund.funding). A settlement the venue's records lack, and every settlement of a
        simulated perp (no venue rates), is charged the baseline (Advisor, 6 Oct 2026). Paper first waits `wait`
        (FUNDING_WAIT by default) for the venue to publish it (None: not yet), then alerts the instrument as stale.
        A flat settlement (`held` False, backtest only) is looked up for the record and charges nothing, so it says
        nothing."""
        if terms.funding_venue is None:
            return markets.baseline_rate(terms), True
        import pandas as pd

        from sleeve_fund import funding

        pair = pair_of(self.instrument)
        when = pd.Timestamp(ts)
        cap = funding.cap_of(terms.funding_venue, pair)
        rate = funding.rate_at(funding.rates(terms.funding_venue, pair), when, cap)
        if rate is None and not self._backtest:
            rate = self._venue_rate(terms, pair, when)
            if rate is None and now - ts < (wait or self.FUNDING_WAIT):
                return None
            if rate is None:
                self._funding_missing.add(when)
                self._funding_missed(terms, pair, when, f"No settled funding rate from the venue for {pair} at "
                                     f"{when:%d %b %Y %H:%M} UTC, {int((wait or self.FUNDING_WAIT).total_seconds() // 60)} "
                                     "minutes after it settled; charging the baseline, whichever side is held, until it "
                                     "arrives")
            elif rate is not None and (self._funding_last_settled is None or when > self._funding_last_settled):
                self._funding_last_settled = when  # a settlement the venue published: earlier missing ones may be lost
        if rate is None:
            if self._backtest and held and not self._funding_fallback_said and self.runtime is not None:
                self._funding_fallback_said = True
                self.runtime.store.event(self.runtime.name, "warning", "funding_fallback",
                                         f"No settled funding rate from the venue for {pair} at {when:%d %b %Y %H:%M} "
                                         f"UTC; charged the {abs(markets.baseline_rate(terms)):.4%} baseline instead, "
                                         "paid whichever side is held (said once)", ts=ts)
            return markets.baseline_rate(terms), True
        return rate, False

    # Paper asks the venue at most this often whether a missing settlement's rate has arrived.
    FUNDING_RECHECK = timedelta(minutes=1)

    def _venue_rate(self, terms, pair: str, when) -> float | None:
        """Paper asks the venue directly (the history service keeps the store, which paper only reads)."""
        from sleeve_fund import funding

        try:
            return funding.rate_at(funding.fetch(terms.funding_venue, pair, when - funding.MATCH), when,
                                   funding.cap_of(terms.funding_venue, pair))
        except Exception as exc:  # noqa: BLE001 - the venue unreachable: wait, then the baseline
            self.log.warning(f"funding rates unavailable: {exc!r}")
            return None

    def _watch_funding_recovery(self, terms, now) -> None:
        """Paper, while settlements charged the baseline are still missing (Advisor, 7 Oct 2026; QA P1-O17a-8, -10,
        -11): one whose rate arrives leaves the watch (O17b trues it up); one still missing a day after it was due,
        once a later one is published, is never published (marked once; its baseline stays) and leaves it too. Every
        one still missing is in an open episode (one closed meanwhile, or never opened, is opened again on it), and an
        episode closes once every settlement in it, from the one it opened on, is kept or never published: so one
        arriving while another is missing keeps it open (CR, #163), and one never published can't hold it open."""
        if self._funding_recheck is not None and now - self._funding_recheck < self.FUNDING_RECHECK:
            return
        self._funding_recheck = now
        import pandas as pd

        from sleeve_fund import funding

        pair, rt = pair_of(self.instrument), self.runtime
        venue = getattr(terms, "funding_venue", None) or ""
        asked: dict = {}

        def arrived(when) -> bool:
            if when not in asked:
                asked[when] = self._venue_rate(terms, pair, when) is not None
            return asked[when]

        if terms is not None:
            self._reverse_unsettled(terms)
        came = {when for when in sorted(self._funding_missing) if arrived(when)}
        self._funding_missing -= came
        try:
            state = funding.journal_state(rt.store, funding.stale_tag(venue, pair))
        except Exception:  # noqa: BLE001 - no database (locally), or a stub inbox: one episode, as before
            state = None
        tag = funding.stale_tag(venue, pair)
        if state is None:
            self._funding_watch_unread(pair, came)
            return
        stamp = pd.Timestamp(now)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
        later = max([t for t in (getattr(self, "_funding_last_settled", None), self._newest_kept(terms, pair))
                     if t is not None],
                    default=None)
        for when in sorted(self._funding_missing):
            if when in state["never"] or (stamp - when >= funding.NEVER_PUBLISHED_AFTER and later is not None
                                          and later > when + funding.MATCH):
                if when not in state["never"]:
                    funding.mark(rt.store, tag, "funding_never_published", when, ts=rt.now())
                    state["never"].add(when)
                self._funding_missing.discard(when)
        for when in sorted(self._funding_missing):
            # Still charging the baseline: an episode closed meanwhile is opened again on it, so the inbox never reads
            # clear while one is missing.
            self._funding_missed(terms, pair, when, f"No settled funding rate from the venue for {pair} at "
                                 f"{when:%d %b %Y %H:%M} UTC yet; charging the baseline, whichever side is held, "
                                 "until it arrives", state=state)
        # From the settlement each opened on (one written before episodes named it opened at its alert, inside the
        # interval after it)
        hours = getattr(terms, "funding_hours", None) or (0, 8, 16)
        gap = pd.Timedelta(markets.funding_interval(hours)) - funding.MATCH
        for o in sorted(k for k in state["open"] if k is not None):
            waiting = [t for t in state["missing"] if t > o - gap and t not in state["never"]
                       and (t in self._funding_missing or not arrived(t))
                       and (terms is None or self._is_settlement(terms, t))]
            if not waiting:
                rt.store.event(None, "info", "funding_stale_cleared", f"{tag} The settled funding rate for {pair} has "
                               f"arrived from the venue for every settlement missing {funding.from_words(o)}"
                               + (f" ({len(came)} came in now)" if came else ""), ts=rt.now())

    def _is_settlement(self, terms, when) -> bool:
        """Whether `when` is still one of the venue's settlements as its records now read (markets.settlement_times)."""
        from sleeve_fund import funding

        times = self._settlements(terms, when - timedelta(minutes=2), when + timedelta(minutes=2))[0]
        return any(abs(pd_ts(t) - pd_ts(when)) <= funding.MATCH for t in times)

    def _reverse_unsettled(self, terms) -> None:
        """Paper: a settlement charged the baseline that the venue's newer records show was none (its interval
        lengthened, so the time foreseen from the shorter step never settled) has the charge reversed by its own
        journaled correction, kind "reversal", never by editing the original row, and leaves the watch (Advisor,
        7 Oct 2026, QA P1-O17a-13). O17b never trues up a reversed baseline."""
        for when in sorted(self._funding_missing):
            if self._is_settlement(terms, when):
                continue
            self._funding_missing.discard(when)
            paid = self._funding_paid.pop(when, None)
            if paid is None:
                continue
            qty, px, rate, amount = paid
            self._cash_adj -= amount
            self.funding_log.append((when, -amount, "reversal"))
            if self.runtime is not None:
                self.runtime.store.record_funding(self.runtime.name, qty=qty, price=px, rate=rate,
                                                  amount=round(-amount, 8), ts=when, kind="reversal")
                self.runtime.store.event(self.runtime.name, "info", "funding",
                                         f"Funding reversed: {abs(amount):,.2f} back; the venue's records show no "
                                         f"settlement at {when:%d %b %Y %H:%M} UTC (its interval lengthened), so the "
                                         "baseline charged for it is refunded", ts=self.runtime.now())

    def _newest_kept(self, terms, pair):
        from sleeve_fund import funding

        if getattr(terms, "funding_venue", None) is None:
            return None
        try:
            kept = funding.rates(terms.funding_venue, pair).index
        except Exception:  # noqa: BLE001 - no store here: the venue's answers alone say what was published
            return None
        return kept[-1] if len(kept) else None

    def _funding_missed(self, terms, pair: str, when, message: str, state: dict | None = None) -> None:
        """Paper: settlement `when` is charged the baseline, its rate missing. Marked once (funding_missing), and
        alerted unless an open episode already holds it: one opened on or before it with every settlement since then
        missing too (funding.episode_of), so one outage alerts once and a later one after a published rate alerts
        again (Advisor, 7 Oct 2026, QA P1-O17a-11)."""
        from sleeve_fund import funding

        rt = self.runtime
        if rt is None:
            return
        tag = funding.stale_tag(getattr(terms, "funding_venue", None) or "", pair)
        try:
            state = state if state is not None else funding.journal_state(rt.store, tag)
        except Exception:  # noqa: BLE001 - no database (locally), or a stub inbox: one episode, as before
            self._funding_episode(pair, True, message)
            return
        if when not in state["missing"]:
            kept = self._newest_kept(terms, pair)
            funding.mark(rt.store, tag, "funding_missing", when, ts=rt.now(), inferred=kept is not None)
            state["missing"].add(when)
        starts = [k for k in state["open"] if k is not None and k <= when]
        due = self._settlements(terms, min(starts) - timedelta(seconds=1), when)[0] if starts and terms is not None else []
        if funding.episode_of(state, when, due) is None:
            event = {"kind": "funding_stale", "ts": rt.now()}
            rt.store.event(None, "warning", "funding_stale", f"{tag} {message}; {funding.from_words(when)}",
                           ts=event["ts"])
            state["open"][when] = event

    def _funding_watch_unread(self, pair: str, came: set) -> None:
        """The journal unreadable: the instrument's one episode closes once nothing is missing."""
        if came and not self._funding_missing:
            last = max(came)
            self._funding_episode(pair, False, f"The settled funding rate for {pair} at {last:%d %b %Y %H:%M} UTC "
                                  "has arrived from the venue" + (f", with {len(came) - 1} earlier" if len(came) > 1 else ""))

    def _rebuild_funding_missing(self) -> None:
        """Paper, on start: the settlements of the instrument's open staleness episode still charged the baseline,
        from the journal's baseline rows, so a restart keeps watching them (and the episode open) until their rates
        arrive rather than leaving the inbox clear while one is missing (QA P1-O17a-10)."""
        import pandas as pd

        from sleeve_fund import funding

        terms = self._cfg.perp
        if terms is None or terms.funding_venue is None:
            return
        try:
            state = funding.journal_state(self.runtime.store, funding.stale_tag(terms.funding_venue, pair_of(self.instrument)))
        except Exception:  # noqa: BLE001 - no database (locally): nothing to carry on watching
            return
        starts = [k for k in state["open"] if k is not None]
        if not starts:
            return
        # Back to the settlement the oldest opened on (an episode written before they named it opened at its alert),
        # one of the instrument's own intervals as its stored rates show it, else the profile's (CR minor 4)
        step = markets.latest_interval(funding.rates(terms.funding_venue, pair_of(self.instrument)))
        opened = min(starts) - pd.Timedelta(step or markets.funding_interval(terms.funding_hours))
        rows: dict = {}
        for row in self.runtime.store.funding(self.runtime.name):
            ts = pd.Timestamp(row["ts"])
            rows.setdefault(ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC"), []).append(row)
        for ts, booked in rows.items():
            # A settlement already reversed (its own reversal row, or rows netting to zero) is never watched again,
            # so a restart can't refund it twice (CR, #163)
            amounts = [r.get("amount") for r in booked]
            if any(r.get("kind") == "reversal" for r in booked) or (
                    None not in amounts and len(amounts) > 1 and abs(sum(amounts)) < 1e-9):
                continue
            row = next((r for r in booked if r.get("kind") == "baseline"), None)
            if row is not None and ts > opened and ts not in state["never"]:
                self._funding_missing.add(ts)
                if row.get("amount") is not None:
                    self._funding_paid[ts] = (row.get("qty"), row.get("price"), row.get("rate"), row["amount"])

    def _funding_episode(self, pair: str, stale: bool, message: str, ts=None) -> None:
        """Open (funding_stale, a warning) or close (funding_stale_cleared) the instrument's staleness episode, once
        whichever strategy on it, or the collector, notices first: the journal's latest such event for the instrument
        says whether one is open (Advisor, 6 Oct 2026: per instrument, once per episode; CR, #163)."""
        from sleeve_fund import funding

        rt = self.runtime
        if rt is None:
            return
        perp = getattr(getattr(self, "_cfg", None), "perp", None)
        tag = funding.stale_tag(getattr(perp, "funding_venue", None) or "", pair)
        open_ = funding.stale_open(rt.store, tag)
        if stale and open_ is not True:
            rt.store.event(None, "warning", "funding_stale", f"{tag} {message}", ts=ts or rt.now())
        elif not stale and open_ is not False:
            rt.store.event(None, "info", "funding_stale_cleared", f"{tag} {message}", ts=rt.now())

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
            d.update(journaled=True, intent="liquidation", signal={"price": price, "liquidation_px": round(liq, 8),
                                                                   "market_px": round(price, 8)},
                     reason=f"Liquidated: the price {price:,.6g} gapped through the liquidation price {liq:,.6g}")
            self._liquidation_events(d["reason"], price, liq)
        else:
            what = {"risk_halt": "the drawdown halt", "risk_pause": "the daily-loss pause",
                    "liquidation_cut": "the cut before liquidation"}[intent]
            d.update(journaled=True, signal={"trigger": round(level, 8), "kind": intent},
                     reason=(f"Risk stop: rested a {word} at {level:,.6g}, where {what} acts on the open {held} "
                             "(re-priced each bar); fills at that level, or the open if the price gaps through"
                             + ("; on bars that can't show what traded first, at the bar's worst price"
                                if self.pessimistic else "")))
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
        d.update(intent="liquidation", reason=reason, signal={**d["signal"], "price": price, "market_px": round(price, 8)})
        self._rebook = journal_id  # the journal re-books it once this fill is in (on_order_filled)
        self._liquidation_events(reason, price, liq)
        self._on_liquidation(price, liq, reason)  # a gapped stop is a liquidation like any other (QA P1-L22 pins)
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

    def _book_liquidation(self, coid: str, sign: int, qty: float, px: float, fee: float,
                          entry: float) -> tuple[float, float]:
        """GAP-LIQ-CAP (Independent Quant Advisor 6 Oct 23:42, 7 Oct 00:19): a liquidation's fill of qty (sign: the
        fill's side) is booked at the bankruptcy price, where the price move loses exactly the posted margin, gapped
        past it or not, and its fee is qty x the liquidation (trigger) price x the taker rate, its own line. So every
        liquidation loses exactly X, the margin plus the entry and liquidation fees. The simulated venue filled it at
        the market's price px with the fee charged on that: the difference goes to cash beside it (_cash_adj), never
        as P&L. How far the market went past bankruptcy (the venue's insurance fund's) or stopped short of it (margin
        the venue kept) is journaled once the position is gone (_liquidation_diagnostic). Returns (price, fee)."""
        lev = self.runtime.profile.max_leverage if self.runtime is not None else 1.0
        held = -sign
        bankrupt = markets.bankruptcy_price(held * qty, entry, lev)
        trigger = (self.decisions[coid].get("signal") or {}).get("liquidation_px") or px
        rate = self.runtime.taker_fee if self.runtime is not None else self._cfg.assumed_taker_fee
        booked_fee = qty * float(trigger) * rate
        self._cash_adj += sign * qty * (px - bankrupt) + (fee - booked_fee)
        c = self._liq_closing or {"qty": 0.0, "market": 0.0, "past": 0.0, "bankrupt": bankrupt}
        c.update(qty=c["qty"] + qty, market=c["market"] + qty * px, past=c["past"] + held * qty * (bankrupt - px))
        self._liq_closing = c
        _, fees = self.liquidation_books.get(coid, (bankrupt, 0.0))
        self.liquidation_books[coid] = (bankrupt, fees + booked_fee)
        return bankrupt, booked_fee

    def _liquidation_diagnostic(self) -> None:
        """Once a liquidation has closed the position: where the market took it beside the bankruptcy price it was
        booked at, journaled as a diagnostic only (never P&L): past it, the venue's insurance fund covered the
        difference; short of it, the venue kept the margin in between (Advisor 7 Oct 00:19 (1), (2))."""
        c, self._liq_closing = self._liq_closing, None
        if c is None or self.runtime is None or not c["qty"]:
            return
        market, past = c["market"] / c["qty"], c["past"]
        rt = self.runtime
        if past >= 0.005:
            rt.store.event(rt.name, "warning", "insurance_fund",
                           f"The market closed the position at {market:,.6g}, past the bankruptcy price "
                           f"{c['bankrupt']:,.6g}: the {past:,.2f} beyond it is covered by the venue's insurance fund. "
                           "Booked at the bankruptcy price: the strategy loses its margin and fees, no more",
                           ts=rt.now())
        elif past <= -0.005:
            rt.store.event(rt.name, "info", "liquidation_forfeit",
                           f"The market closed the position at {market:,.6g}, short of the bankruptcy price "
                           f"{c['bankrupt']:,.6g}: the venue kept the {-past:,.2f} of margin in between. Booked at the "
                           "bankruptcy price: the strategy loses its margin and fees", ts=rt.now())

    def _cover_shortfall(self, price: float, event=None) -> None:
        """Once flat, equity never ends below zero: a liquidation is booked at the bankruptcy price, so it loses
        its isolated margin and fees and no more (_book_liquidation); should this book's own arithmetic still leave
        it below zero, the difference comes back to cash, journaled as the venue's insurance fund's."""
        credit = max(-self._mark()[0], 0.0)
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
                                     f"Closed at {price:,.6g} with equity {credit:,.2f} below zero: the venue's "
                                     "insurance fund takes the shortfall, as isolated margin caps the loss at the "
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
        book = rt.store.journal_book(rt.name, rt.starting_balance)
        # The journal's own cash once the position is gone, to the cent the PM reads elsewhere (QA SG8); the mark
        # only while the journal still holds part of it.
        left = book["cash"] if abs(book["qty"]) < 1e-12 else self._mark()[0]
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
        self._sell_all(intent, reason, {"price": price, "liquidation_px": round(liq, 8), "distance": round(distance, 6)}
                       | ({"market_px": round(price, 8)} if crossed else {}))
        if crossed:
            self._on_liquidation(price, liq, reason)

    def _on_liquidation(self, price: float, liq: float, reason: str) -> None:
        """A liquidation the strategy closed at: on a live trade, or found by the outage replay (_replay_missed,
        which books it in the minute the price reached it). The liquidation incident and halt (Advisor 17:57 and
        18:17, QA P1-L18) hang on the liquidation order's fill, not here: the incident names the equity left once the
        position is gone (_liquidation_incident), and the halt is the tick's (_wiped_out_why). Both run for a
        replayed liquidation exactly as for a live one, so this hook only marks the moment; it stays the one place a
        caller is told of a liquidation."""

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
        cash += self._cash_adj  # an exit booked at a replayed price (Advisor NA-1): the journal's cash
        return cash + qty * price, cash, qty, price

    def _market_seen(self) -> None:
        self._last_market_ns = self.clock.timestamp_ns()
        self._market_since_start = True
        if not self._backtest and not self.hub_fed:
            self._minutes_seen.add(self._last_market_ns // MINUTE_NS)
            if self._first_minute is None:
                self._first_minute = self._last_market_ns // MINUTE_NS
        if self.runtime is not None and not self._backtest:
            self.runtime.market_seen()
            self.runtime.holds.pop("stale_data", None)
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
            if self.runtime is not None:
                # Stale data holds every entry and add, and cancels resting entries, until a trade or quote arrives
                # (Advisor 22:29 (1), 00:20 (b)); stops and exits still run on the last price.
                age = (now - self._last_market_ns) / 1e9
                self.runtime.holds["stale_data"] = (
                    f"last price {age:.0f} s old. It clears when data resumes" if self._market_since_start else
                    f"no trade or quote since this process started {age:.0f} s ago. It clears when data resumes")
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
                # Its liquidation fee at the taker rate on the liquidation price, where the venue books it
                # (_book_liquidation), until the fill gives the fee charged.
                at = self._liq(cash, qty) if self._margin else None
                fees, taken, before, _ = self._liquidation_figures(None, abs(qty) * (at or price))
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
            self._sync_watched_stop()
        except Exception as exc:  # never let bookkeeping kill the sleeve silently: counted, kept and journaled
            self._report("_on_tick", exc)

    def _resting_openers(self) -> tuple:
        """The opening orders that rest, as (orders at the venue, kept post-only client order ids), or () when none
        does. An opening order is an entry, or a rebalance that adds (as _gated reads it at submit)."""
        open_ = [o for o in self.cache.orders_open(strategy_id=self.strategy_id) if o.status != OrderStatus.PENDING_CANCEL]
        if self._journal_intents is None and any(str(o.client_order_id) not in self.decisions for o in open_):
            # An order this process has no decision for was sent before a restart: the journal has its intent. Read
            # once; such an order's intent never changes (Code Reviewer on 81e6d6f).
            rt = self.runtime
            self._journal_intents = {o["order_id"]: o["intent"]
                                     for o in rt.store.orders(rt.name, statuses=OPEN_ORDER_STATUSES, limit=1000)}

        def intent(coid: str):
            return self.decisions.get(coid, {}).get("intent") or (self._journal_intents or {}).get(coid)
        def opens(what, side) -> bool:  # an entry, or a rebalance that adds (as _gated reads it at submit)
            return what == "entry" or (what == "rebalance" and self._adds(side))
        resting = [o for o in open_ if opens(intent(str(o.client_order_id)), o.side)]
        kept = [c for c, k in self._kept.items() if opens(k["info"].get("intent"), k["order"].side)]
        return (resting, kept) if resting or kept else ()

    def _watch_gate(self) -> None:
        """P1-SG15 (Advisor 7 Oct 05:01): read the gate every GATE_WATCH_SECONDS while an opening order rests at the
        venue, so a block accepted between prints (a Stop) has its cancel sent within that time, not on the next print
        or tick: a slow feed can't widen the window in which a resting entry may still fill."""
        if GATE_WATCH not in self.clock.timer_names() and self._resting_openers():
            self.clock.set_timer(GATE_WATCH, timedelta(seconds=GATE_WATCH_SECONDS), callback=self._on_gate_watch)

    def _on_gate_watch(self, event) -> None:
        try:
            if self._resting_openers():
                self._cancel_resting_entries()
            elif GATE_WATCH in self.clock.timer_names():
                self.clock.cancel_timer(GATE_WATCH)  # nothing opening rests: nothing to watch until one is accepted
        except Exception as exc:  # never let bookkeeping kill the sleeve silently: counted, kept and journaled
            self._report("_on_gate_watch", exc)

    def _cancel_resting_entries(self) -> None:
        """While nothing may open (CHOKE: liquidated until a reset after liquidation, Advisor 17:57 rule (c), QA P1-D21;
        halted, paused or stopped), every entry or rebalance still resting at the venue, or the unfilled rest of one,
        is cancelled. Without it a resting entry filled later and opened a position on a halted strategy. Closing
        orders (stops, exits, the liquidation itself) are left alone."""
        resting, kept = self._resting_openers() or ([], [])
        blocked, why = self.runtime.entry_blocked()  # asked every tick, so the block episode is journaled
        if not blocked:
            return
        for order in resting:
            self._part_filled(str(order.client_order_id), order.filled_qty.as_double(), order.quantity.as_double(), why)
            self.cancel_order(order.client_order_id)
        if resting and not self._backtest:
            # Advisor 7 Oct 05:47: how long after the block (a Stop: its acceptance) the cancel went out, on record.
            rt, ms = self.runtime, self._ms_since_block(why)
            rt.store.event(rt.name, "info", ENTRY_CANCELLED, f"{len(resting)} resting opening order"
                           f"{'s' if len(resting) != 1 else ''} cancelled {ms} ms after nothing could open any more "
                           f"({', '.join(block_codes(why)) or why}).", ts=rt.now())
        for coid in kept:  # paper's kept post-only entry: no more slices of it go at market
            k = self._kept[coid]
            self._part_filled(coid, float(k["sent"]), k["order"].quantity.as_double(), why)
            self._close_kept(coid, f"cancelled: nothing may open now. {why}")

    def _ms_since_block(self, why) -> int:
        """Milliseconds from the block's start (a Stop: its acceptance) to now, on the engine's own clock."""
        now = datetime.fromtimestamp(self.clock.timestamp_ns() / 1e9, tz=timezone.utc)
        return max(0, round((now - self.runtime.block_began(why, now)).total_seconds() * 1000))

    def _part_filled(self, coid: str, filled: float, qty: float, why: str) -> None:
        """Advisor 20:56: an entry part filled when the block starts has its rest cancelled and keeps what filled, with
        its stop, never flattened; an incident says so, once per order, for the PM to decide."""
        if filled <= 0 or coid in self._gated_fills:
            return
        self._gated_fills.add(coid)
        rt = self.runtime
        rt.store.event(rt.name, "error", "incident",
                       f"Incident, {rt.name}: an entry was part filled ({filled:g} of {qty:g}) when nothing could open "
                       f"any more. {why} Its rest is cancelled; what filled is kept with its stop, not closed; you "
                       "decide what to do with it.", ts=rt.now())

    def _gated_fill(self, coid: str, qty: float, px: float) -> None:
        """CHOKE at fill: an entry or rebalance that fills while nothing may open (sent before the gate closed, its
        cancel too late) is kept, with its stop placed for what filled as on any entry, and never flattened; it opens
        an incident, once per order: a kept post-only order's slices count as that one order."""
        order = self._slices.get(coid, coid)
        if (self.runtime is None or order in self._gated_fills
                or self.decisions.get(coid, {}).get("intent") not in OPENING_INTENTS):
            return
        blocked, why = self.runtime.entry_blocked()
        if not blocked:
            return
        self._gated_fills.add(order)
        rt = self.runtime
        # Advisor 23:05 (SG7): a raced fill is treated as the block treats a position already held. A halt, a
        # liquidation and the daily pause flatten, so it is sold at once; Stop, stale data, funding and retire keep it.
        sells = bool({*RESUMABLE, "liquidated", "daily_pause"} & set(getattr(why, "codes", ())))
        if sells:
            rt.raced = why
        rt.store.event(rt.name, "error", "incident",
                       f"Incident, {rt.name}: an order that adds to the position filled while nothing may open: "
                       f"{qty:g} at {px:,.6g}. {why} "
                       + ("It is sold at once through the exit path, as that block flattens what it holds." if sells else
                          "It is kept with its stop, not closed; you decide what to do with it."),
                       ts=rt.now())
        # Advisor 7 Oct 05:01: each raced fill is on record with how long after the block it filled, for fills-vs-model.
        ms = self._ms_since_block(why)
        rt.store.event(rt.name, "info", RACED_FILL, f"Raced fill: {qty:g} at {px:,.6g}, {ms} ms after nothing could "
                       f"open any more ({', '.join(block_codes(why)) or why}).", ts=rt.now())

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
        if self.runtime is not None:
            self.runtime.on_timing(coid, accepted=int(event.ts_init))
            if not self._backtest:
                self._watch_gate()
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
        self._lock_exit_leg(self._pos_side() if self._margin else True)
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
                self._intrabar, self._filling = intrabar, fill
                try:
                    self._apply_funding(self._last_close or float(event.last_px))
                finally:
                    self._intrabar, self._filling = None, 0.0
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
        book = ((self.decisions.get(journal_id) or {}).get("signal") or {}).get("book_px")
        if self._backtest and self.fee_model is not None and coid in self.fee_model.rebooked:
            book = self._rebook_as_target(coid, *self.fee_model.rebooked.pop(coid), px)
        if book and kept_id is None and self._backtest:
            # A backtest's target (_bar_target): the fee model charged the difference from its level with the fee
            # (ScheduleFeeModel.booked), so the account already holds the target's cash. Journal the fee alone.
            fee -= sign * qty * (book - px)
            px = book
        elif book and kept_id is None:
            # A replayed exit (Advisor NA-1): journaled at the price the venue's resting order would have had. The
            # account's cash keeps the journal's, as for a restore; the fee is the venue's rate on that price.
            charged, fee = fee, (fee * book / px if px else fee)
            self._cash_adj += sign * qty * (px - book) + (charged - fee)  # the fee too, as journaled (QA P1-D25)
            px = book
        queued = self.fee_model.booked_fills.get(coid) if self.fee_model is not None else None
        booked = queued.pop(0) if queued else None
        if booked is not None and not book:
            # A backtest's taker fill, booked by the fee model at the ask or bid, or a resting stop with its slippage
            # (ScheduleFeeModel.exit_price): the journal keeps the price it was booked at and the rest of the charge as
            # the fee (the venue's, with the rounding carried from earlier fees), as the fills report does
            # (runner._spread_into_prices).
            px, fee = booked[0], fee - booked[1]
        note = outage_fill_note(self.decisions.get(journal_id), px)
        if note is not None and self.runtime is not None:
            self.runtime.store.event(self.runtime.name, "warning", "outage_exit_filled", note)
        if self._margin and self.runtime is not None and self._gap_liquidation(coid, journal_id, px):
            held = (held[0], self._entry_px)  # the entry a paper stop cleared as it fired, put back
        if self.runtime is not None and coid == self._risk_stop_id and order is not None:
            self._journal_risk_stop(order, px)  # first: a risk stop gapped through liquidation is booked as one below
        if (self._margin and self.decisions.get(coid, {}).get("intent") == "liquidation" and held[1]
                and self._entry_side == -sign):
            px, fee = self._book_liquidation(coid, sign, qty, px, fee, held[1])
        if self._margin:
            opening = self._track_entry(sign, event.last_qty.as_decimal(), qty, px)
            if opening:
                self._set_liq(journal_id)
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
        if opening and self.fee_model is not None:
            self._opened_seq = self.fee_model.bar_seq
        if opening:  # minutes before this fill are no one's to replay for this position
            self._replayed_to = max(self._replayed_to, int(event.ts_event))
        if opening:
            self._entry_fill_ns = self.clock.timestamp_ns()
            if self._has_exits and self._stop_frac is None and not self._tp_frac:
                # A resting entry (not sent by _open, which plans before it sends): its exits are planned from its fill,
                # so its stop rests the moment it fills, as an entry at market's does.
                plan = self._plan_exits(px, self._entry_side or sign)
                if plan is not None:
                    self._stop_frac, self._tp_frac, self._stop_basis = plan
            self._gated_fill(coid, qty, px)
        self._sync_watched_stop()  # an add or a partial close resizes the journal's stop in the same step (QA)
        if self._backtest:
            if opening and self._pending_exit is None:
                # On every entry fill, not only the last: a post-only entry can fill in slices through its
                # maker wait, and paper guards each slice from its first trade (review round 9, M9-4).
                self._rest_exits()
                self._rest_risk_stop()
            elif self.decisions.get(str(event.client_order_id), {}).get("intent") in ("stop_loss", "take_profit"):
                # As in paper: no re-entry until the signal has moved off the side that was closed.
                self._lock_exit_leg(-sign if self._margin else True)
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
            # A fill on a gap took the bar's open price, so its record carries the open's time (QA P1-D12).
            at = (datetime.fromtimestamp(intrabar[0] / 1e9, tz=timezone.utc)
                  if intrabar is not None and intrabar[2] else None)
            self.runtime.on_fill(side="BUY" if event.is_buy else "SELL", qty=qty, price=px, fee=fee,
                                 order_id=journal_id, trade_id=str(event.trade_id), ts=at)
            self.runtime.on_timing(journal_id, fill=int(event.ts_init), venue_ts=int(event.ts_event))
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
            self._liquidation_diagnostic()
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
            self._stop_frac = self._tp_frac = self._liq_px = None
            self._funding_skip = None
            self._replan_pending = self._plan_entry = None
            return False
        opening = side == sign and abs(post) > abs(pre)
        if opening and self._entry_side == side and self._entry_px is not None:
            # Added to the same side: the average entry of what was held and this fill.
            held = float(abs(post)) - qty
            self._entry_px = (self._entry_px * held + px * qty) / float(abs(post))
        elif opening or self._entry_side != side or self._entry_px is None:
            self._entry_px = px  # opened from flat, or this fill went through flat and opened the other side
        if self._entry_side != side:
            self._liq_px = None  # went through flat: the old side's price is no longer the position's
        self._entry_qty, self._entry_side = float(abs(post)), side
        return opening

    def _rest_exits(self) -> None:
        """Backtests: after an entry fills, rest the stop at the venue as a real order would sit there (paper and
        live watch every trade instead): a sell stop at its level, filling there, or at the open when the price
        gaps through it. A flatten cancels it first (_sell_all). The target doesn't rest: within a bar the adverse
        side trades first (Advisor NA-2), so the venue takes the stop on the bar and the target is judged after it,
        on what the bar left (_bar_target). The open trades before either: a stop filled in a bar that opened through
        the target is booked as the target (ScheduleFeeModel.open_targets, _rebook_as_target). The fee model books the
        stop with its slippage, and on bars that can't show what traded first inside them, at the bar's worst price
        (ScheduleFeeModel.exit_price, P1-D13)."""
        cfg = self._cfg
        plan = {"stop_loss": self._stop_frac}
        # A stop moved from the market can sit at the entry price (0) or past it, in profit (below 0).
        if plan["stop_loss"] is None or self._entry_px is None:
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
        side = self._entry_side or 1
        exit_side, exit_word = (OrderSide.SELL, "sell") if side > 0 else (OrderSide.BUY, "buy")
        orders = []
        now = self.clock.timestamp_ns()
        if plan["stop_loss"] is not None:
            stop = plan["stop_loss"]
            level = self._entry_px * (1 - side * stop)
            orders.append((StopMarketOrder(
                self.trader_id, self.strategy_id, cfg.instrument_id, self.order_factory.generate_client_order_id(),
                exit_side, quantity, Price(level, self.instrument.price_precision), TriggerType.DEFAULT,
                TimeInForce.GTC, self._margin, False, UUID4(), now), "stop_loss", "STOP",
                f"Stop-loss: resting {exit_word} at {level:,.6g}, {_from_entry(stop, side).replace(' the entry', '')} "
                f"the {self._entry_px:,.6g} entry"
                + (f" (set {self._stop_basis})" if cfg.stop_atr or cfg.stop_swing_bars else "")
                + "; fills at that level, or the open if the price gaps through"
                + ("; on bars that can't show what traded first, at the bar's worst price" if self.pessimistic else ""),
                {"entry_px": round(self._entry_px, 8), "stop_loss": round(stop, 6), "trigger": round(level, 8),
                 **self._liq_signal()}))
        for order, intent, kind, reason, signal in orders:
            self.decisions[str(order.client_order_id)] = {"intent": intent, "reason": reason, "signal": signal}
            if self.runtime is not None:
                self.runtime.on_order(order_id=str(order.client_order_id), side="SELL" if side > 0 else "BUY",
                                      qty=float(qty), intent=intent, reason=reason, signal=signal, order_type=kind)
        for order, *_ in orders:
            self._open_target(str(order.client_order_id))
            self._sent.append(order.client_order_id)
            self.submit_order(order)

    def submit_order(self, order, *args, **kwargs) -> None:
        """Nothing reaches the venue before trade_from, whatever sends it (a model's own resting entry too): a study's
        out-of-sample window starts flat."""
        if self._cfg.trade_from is not None and self.clock.timestamp_ns() < self._cfg.trade_from:
            self.decisions.pop(str(order.client_order_id), None)
            return
        super().submit_order(order, *args, **kwargs)

    def _exit_booking(self, order) -> dict | None:
        """For the backtest's fee model (ScheduleFeeModel.exit_info): what a filling order is, if it is a resting stop
        (the model's or the risk guard's) or a stop-out to be booked elsewhere (_book_at). The target is booked by
        _bar_target."""
        coid = str(order.client_order_id)
        side = -1 if order.side == OrderSide.BUY else 1  # the position the exit closes
        if coid in self._book_at:
            return {"kind": "stop", "side": side, "base": self._book_at[coid]}
        intent = self.decisions.get(coid, {}).get("intent")
        mine = self.cache.order(ClientOrderId(coid)) or order
        now = self.clock.timestamp_ns()
        if mine.order_type == OrderType.STOP_MARKET and intent in STOP_INTENTS:
            return {"kind": "stop", "side": side, "trigger": mine.trigger_price.as_double(), "rested": mine.ts_init < now,
                    "liq": intent in ("liquidation_cut", "liquidation"),
                    # A model stop carries its liquidation price in its signal (GAP-LIQ), so a gapped one is a
                    # liquidation close to the fee model and pays no floor (D13-F2).
                    "liq_px": (self.decisions[coid].get("liq") or (self.decisions[coid].get("signal") or {})
                               .get("liquidation_px")) if intent != "liquidation" else None,
                    "liquidation": intent == "liquidation"}
        if intent == "liquidation":  # the market close at the liquidation price (_check_liquidation)
            return {"kind": "liquidation", "side": side}
        return None

    def _levels_in_bar(self, bar: Bar) -> None:
        """Bars too coarse to say what traded first: note whether any resting level (a stop, target or entry, or the
        liquidation price) lay inside this bar's range, traded or not. The result is labelled when one did (P1-D13
        19:00); the fee model notes the ones that filled."""
        if self.fee_model is None:
            return
        low, high = bar.low.as_double(), bar.high.as_double()
        for order in self._working():
            if getattr(order, "is_post_only", False):
                continue  # a maker order fills only on a trade through its price, at it: no order within the bar matters
            level = (order.trigger_price if order.order_type == OrderType.STOP_MARKET else
                     order.price if order.order_type == OrderType.LIMIT else None)
            if level is not None and low <= level.as_double() <= high:
                intent = self.decisions.get(str(order.client_order_id), {}).get("intent")
                self.fee_model.intrabar.add("liq" if intent in ("liquidation_cut", "liquidation") else "fill")
        if self._tp_frac and self._entry_px is not None and self._pos_side():
            target = self._entry_px * (1 + (self._entry_side or 1) * self._tp_frac)  # judged on the bar (_bar_target)
            if low <= target <= high:
                self.fee_model.intrabar.add("fill")
        if self._margin and self._net_position()[0]:
            _, cash, qty, _ = self._mark()
            liq = self._liq(cash, qty)
            if liq is not None and low <= liq <= high:
                self.fee_model.intrabar.add("liq")

    def _stopped_in_entry_bar(self, bar: Bar) -> bool:
        """Bars too coarse to say what traded first: an entry that filled inside this bar (a resting entry), whose stop
        the bar also traded through, filled and was then stopped out (Advisor 18:36), at the bar's adverse extreme
        with the stop's slippage. The venue only saw the stop once the entry had filled, so it may not have."""
        if self._entry_fill_ns != self.clock.timestamp_ns() or not self._stop_frac or self._entry_px is None:
            return False
        side = self._entry_side or 1
        if self._pos_side() != side or self._pending_exit is not None:
            return False
        level = self._entry_px * (1 - side * self._stop_frac)
        adverse = bar.low.as_double() if side > 0 else bar.high.as_double()
        if not (adverse <= level if side > 0 else adverse >= level):
            return False
        self._book_stop_at = adverse
        self._exit_lock = side if self._margin else True
        self._sell_all("stop_loss", f"Stop-loss: the entry filled inside this bar and the bar also traded through the "
                       f"{level:,.6g} stop, to {adverse:,.6g}; with only the bar to go on, the entry filled first and was "
                       "then stopped out there", {"entry_px": self._entry_px, "trigger": round(level, 8),
                                                   "stop_loss": self._stop_frac})
        return True

    def _rebook_as_target(self, coid: str, level: float, book: float, px: float) -> float:
        """Backtests: the venue filled the resting stop in a bar that opened through the target. The open trades
        first, so the target took it (Advisor NA-2), booked at its level less the taker's slippage, never the open
        (L12); the fee model charged it so, and the order is journaled as the take-profit."""
        decision = self.decisions.get(coid)
        if decision is not None and decision.get("intent") == "stop_loss":
            reason = (f"Take-profit: the bar opened through the {level:,.6g} target, so the target sold at the open, "
                      f"before the price reached the stop at {px:,.6g} later in the bar; booked at {book:,.6g}, the "
                      "target less the taker's slippage, never the better open")
            decision.update(intent="take_profit", reason=reason)
            decision["signal"] = {**(decision.get("signal") or {}), "book_px": book, "target_px": level}
            if self.runtime is not None:
                self.runtime.store.update_order(coid, intent="take_profit", message=reason)
        return book

    def _open_target(self, stop_id: str, resized: bool = False) -> None:
        """Backtests: tell the fee model where this position's target is, so a stop the venue fills in a bar that
        opened through the target is booked at the target instead (Advisor NA-2: the open trades first)."""
        if self.fee_model is None:
            return
        if self._tp_frac and self._entry_px is not None:
            level = self._entry_px * (1 + (self._entry_side or 1) * self._tp_frac)
            # Set during the bar the entry filled in, it counts from the next bar's open: that bar's open came
            # before the position did. So does a resize's (a later slice of the entry filled): the stop now covers
            # a slice that wasn't held at this bar's open, so none of it is booked as this bar's target (QA P1-L21;
            # the slices held at the open are stopped with it, the conservative side).
            _, since = self.fee_model.open_targets.get(stop_id, (None, self.fee_model.bar_seq))
            if resized:
                since = self.fee_model.bar_seq
            self.fee_model.open_targets[stop_id] = (Price(level, self.instrument.price_precision).as_decimal(), since)
        else:
            self.fee_model.open_targets.pop(stop_id, None)

    def _resting_exits(self) -> dict:
        """The backtest's stop resting at the venue, by intent."""
        out = {}
        exit_side = OrderSide.BUY if self._entry_side < 0 else OrderSide.SELL
        for order in self._working():
            intent = self.decisions.get(str(order.client_order_id), {}).get("intent")
            if order.side == exit_side and intent == "stop_loss":
                out[intent] = order
        return out

    def _resize_exits(self, resting: dict, plan: dict) -> None:
        """A later slice of the entry filled: grow the resting stop to the whole position, at its level
        from the new average entry. An exit the venue hasn't answered yet (sent, or mid-resize) is
        resized once it does (_resize_if_due): skipping it left most of a position with no stop."""
        qty = self._position_qty().quantize(self._lot(), rounding=ROUND_DOWN)
        if qty < self._min_qty():
            return
        quantity = Quantity.from_decimal_dp(qty, self.instrument.size_precision)
        for intent, order in resting.items():
            frac = plan.get(intent)
            if frac is None or (intent == "take_profit" and not frac):
                continue
            if order.status not in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                self._resize_due = True  # in flight or mid-resize: resized again once the venue answers
                continue
            side = self._entry_side or 1
            level = self._entry_px * (1 - side * frac)
            px = Price(level, self.instrument.price_precision)
            if order.quantity == quantity and order.trigger_price == px:
                continue
            self.modify_order(order.client_order_id, quantity=quantity, trigger_price=px)
            self._open_target(str(order.client_order_id), resized=True)  # it moves with the average entry too
            self._resizing[str(order.client_order_id)] = (f"Stop-loss resized to {qty.normalize():f} at {level:,.6g} as "
                                                          f"more of the entry filled (average entry {self._entry_px:,.6g})")

    def on_order_updated(self, event) -> None:
        """The venue took a resize (_resize_exits): journal the new quantity once it holds, not when asked,
        since the order can fill before the venue gets to it."""
        coid = str(event.client_order_id)
        why = self._resizing.pop(coid, None)
        exit_ = self.decisions.get(coid, {}).get("intent") == "stop_loss"
        if (why is not None or exit_) and self.runtime is not None:
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
            for timer in ("sleeve-tick", GATE_WATCH):
                self.clock.cancel_timer(timer) if timer in self.clock.timer_names() else None
            self.runtime.on_stop()
        if self.recorder is not None:
            self.recorder.close()
