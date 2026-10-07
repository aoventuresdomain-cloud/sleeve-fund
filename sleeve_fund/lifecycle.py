"""A strategy's lifecycle (P2-6): Active -> Winding down -> Retired. The pure core; the runtime, supervisor, store and
dashboard route call it and keep no rule of their own. Nothing here reads the database, the clock or the environment;
time is an argument (tz-aware UTC). Rulings: Independent Quant Advisor 6 Oct 14:50/14:55 (advisor-rulings.md), the
Data Architect's columns (sleeves.lifecycle, wind_down_deadline) and the day-0 interface note v4's money rules.

- Only the PM moves a strategy. Allowed: active -> winding_down, and to retired from either, but only when flat.
  "Close now" on a strategy holding a position winds it down with its deadline now: it is flattened at the next bar
  close and retired once flat.
- A winding-down strategy opens nothing new; its exits, trims and risk stops all keep running (an entry can be
  skipped, an exit cannot).
- The deadline is 2 x the p95 hold, taking the larger of the backtest's and the strategy's own paper holds. With
  fewer than 20 backtest trades it is 2 x the longest hold seen. Floor 3 days, cap 30 days. It is fixed when wind-down
  starts and never moved, except to now by "close now".
- Notice at 75% of the window; at the first bar close at or after the deadline the position is flattened, on stale
  data too, retried each bar until flat. Retired only when confirmed flat.
- At wind-down start one reduce-only stop is placed: the strategy's own stop_atr, else 3 x Wilder ATR(14) on the
  decision timeframe, from the price at wind-down start (not the entry), never further than half way to the
  liquidation price. Placed once, never trailed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from sleeve_fund.money import money, scale

ACTIVE, WINDING_DOWN, RETIRED = "active", "winding_down", "retired"
STATES = (ACTIVE, WINDING_DOWN, RETIRED)
PM = "pm"  # the only actor that may move a strategy

HOLD_MULTIPLE = 2  # deadline = this x the p95 (or fallback) hold
HOLD_QUANTILE = 0.95
MIN_TRADES = 20  # backtest trades below which the p95 isn't trusted and the longest hold is used
WINDOW_FLOOR = timedelta(days=3)
WINDOW_CAP = timedelta(days=30)
NOTICE_AT = 0.75  # share of the window at which the PM is told the deadline is coming
FALLBACK_STOP_ATRS = 3.0  # Wilder ATR(14) multiples for a strategy with no stop_atr of its own

# What due() asks the supervisor to do at a bar close.
NONE, NOTICE, FLATTEN, RETIRE = "none", "notice", "flatten", "retire"


class LifecycleError(ValueError):
    """A move the lifecycle doesn't allow. `reason` is the sentence for the journal and the strategy page."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class NotPM(PermissionError):
    """Anyone but the PM tried to move a strategy."""


def _utc(name: str, t: datetime) -> datetime:
    if not isinstance(t, datetime) or t.tzinfo is None or t.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be a tz-aware UTC datetime, not {t!r}")
    return t


def _state(name: str, s: str) -> str:
    if s not in STATES:
        raise ValueError(f"{name} must be one of {STATES}, not {s!r}")
    return s


def _pm(actor: str) -> None:
    if actor != PM:
        raise NotPM(f"only the PM can change a strategy's lifecycle, not {actor!r}")


def move(current: str, to: str, actor: str, *, flat: bool) -> str:
    """The new lifecycle, after checking the move. Raises NotPM for any actor but the PM, and LifecycleError for a
    move that isn't allowed, including retiring a strategy that still holds a position (use close_now)."""
    _state("current", current)
    _state("to", to)
    _pm(actor)
    if current == RETIRED:
        raise LifecycleError("This strategy is retired; a retired strategy can't be moved.")
    if to == current:
        raise LifecycleError(f"This strategy is already {current.replace('_', ' ')}.")
    if to == ACTIVE:
        raise LifecycleError("A strategy that is winding down can't be made active again.")
    if to == RETIRED and not flat:
        raise LifecycleError("This strategy still holds a position, so it can't be retired yet: "
                             "close it now to flatten it and retire it once flat.")
    return to


@dataclass(frozen=True)
class Move:
    lifecycle: str
    deadline: datetime | None  # set while winding down


def close_now(current: str, actor: str, now: datetime, *, flat: bool) -> Move:
    """The PM's "close now", from active or winding down. Flat: retired at once. Holding: winding down with the
    deadline now, so due() flattens it at the next bar close and retires it once flat."""
    _utc("now", now)
    if flat:
        return Move(move(current, RETIRED, actor, flat=True), None)
    _state("current", current)
    _pm(actor)
    if current == RETIRED:
        raise LifecycleError("This strategy is retired; a retired strategy can't be moved.")
    return Move(WINDING_DOWN, now)


def _holds(name: str, holds: Sequence[timedelta]) -> list[timedelta]:
    out = sorted(holds)
    if any(not isinstance(h, timedelta) or h < timedelta(0) for h in out):
        raise ValueError(f"{name} must be non-negative timedeltas")
    return out


def _p95(holds: list[timedelta]) -> timedelta:
    """The nearest-rank 95th percentile: the smallest hold at least 95% of holds are no longer than."""
    return holds[math.ceil(HOLD_QUANTILE * len(holds)) - 1]


def window(backtest_holds: Sequence[timedelta], paper_holds: Sequence[timedelta]) -> timedelta:
    """How long a strategy may wind down for, from its trades' holding times."""
    bt, paper = _holds("backtest_holds", backtest_holds), _holds("paper_holds", paper_holds)
    if len(bt) >= MIN_TRADES:
        hold = max(_p95(bt), _p95(paper) if paper else timedelta(0))
    else:  # too few backtest trades to trust a p95: the longest hold seen anywhere
        hold = max(bt + paper, default=timedelta(0))
    return min(max(HOLD_MULTIPLE * hold, WINDOW_FLOOR), WINDOW_CAP)


def deadline(started: datetime, backtest_holds: Sequence[timedelta], paper_holds: Sequence[timedelta]) -> datetime:
    """When a strategy winding down from `started` must be flat. Fixed at the start, never moved."""
    return _utc("started", started) + window(backtest_holds, paper_holds)


def due(lifecycle: str, started: datetime | None, deadline_at: datetime | None, now: datetime, *, flat: bool,
        noticed: bool) -> str:
    """What the supervisor does for a strategy at the bar close `now`: NONE, NOTICE (once, at 75% of the window),
    FLATTEN (every bar from the deadline until flat: the retry) or RETIRE (winding down and flat)."""
    _state("lifecycle", lifecycle)
    _utc("now", now)
    if lifecycle != WINDING_DOWN:
        return NONE
    if started is None or deadline_at is None:
        raise ValueError("a strategy winding down has a start and a deadline")
    _utc("started", started)
    _utc("deadline", deadline_at)
    if deadline_at < started:
        raise ValueError("a wind-down's deadline can't be before its start")
    if flat:
        return RETIRE
    if now >= deadline_at:
        return FLATTEN
    if not noticed and now - started >= (deadline_at - started) * NOTICE_AT:
        return NOTICE
    return NONE


def can_open(lifecycle: str) -> bool:
    """Whether a strategy may open or add to a position. Exits never ask."""
    return _state("lifecycle", lifecycle) == ACTIVE


@dataclass(frozen=True)
class WindDownStop:
    level: Decimal  # the stop's trigger price, before rounding to the tick (round it toward `price`)
    distance: Decimal  # |price - level|
    atrs: float  # the ATR multiple asked for
    clamped: bool  # held at half way to liquidation
    reason: str  # for the journal and the strategy page


def wind_down_stop(side: int, price: Decimal | int | str, atr: Decimal | int | str, *, stop_atr: float | None = None,
                   liquidation: Decimal | int | str | None = None) -> WindDownStop:
    """The one reduce-only stop placed when a strategy holding a position starts winding down. side: 1 long, -1 short;
    price: the price at wind-down start; atr: Wilder ATR(14) on the decision timeframe, in price; liquidation: the
    position's liquidation price, None when nothing liquidates it."""
    if side not in (1, -1):
        raise ValueError(f"side must be 1 (long) or -1 (short), not {side!r}")
    price, atr = money(price, "price"), money(atr, "atr")
    if price <= 0 or atr <= 0:
        raise ValueError("a wind-down stop needs a positive price and a positive ATR")
    k = FALLBACK_STOP_ATRS if stop_atr is None else stop_atr
    if isinstance(k, bool) or not isinstance(k, (float, int)) or not math.isfinite(k) or k <= 0:
        raise ValueError(f"stop_atr must be a positive number, not {stop_atr!r}")
    own = "its own stop" if stop_atr is not None else "the fallback stop"
    distance = scale(atr, float(k))
    clamped = False
    if liquidation is not None:
        liq = money(liquidation, "liquidation")
        if (liq - price) * side >= 0:
            raise ValueError("the liquidation price must be on the losing side of the price")
        half = abs(price - liq) / 2
        if distance > half:
            distance, clamped = half, True
    level = price - side * distance
    if level <= 0:
        raise ValueError("the ATR is too large for a stop below this price")
    where = "below" if side > 0 else "above"
    reason = f"Wind-down stop: {own}, {k:g} average true ranges {where} the price at wind-down start"
    if clamped:
        reason += ", held at half the distance to liquidation"
    return WindDownStop(level=level, distance=distance, atrs=float(k), clamped=clamped, reason=reason + ".")
