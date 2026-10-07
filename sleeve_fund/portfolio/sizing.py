"""Central sizing (v2 P2-1a): one pure function sizes every entry, in backtest and paper alike. Rebuilt on main from
#165 (Independent Quant Advisor, 7 Oct 17:03), to the frozen day-0 interface note v4: money and quantities are Decimal,
ratios are float, and nothing here reads a clock, a venue or the database.

The risk budget is the strategy's allocated equity x its risk per trade (per side, B1) x the
regime weight fixed at entry x the definition's fraction of full size (A8). The quantity is that budget over what one
unit loses if the stop is hit, costs and the stop's slippage included. Volatility targeting (B2) sizes the notional
from the instrument's volatility instead, and still declares a stop; with both, the smaller size is taken, never their
product. Then the caps, smallest first wins:

- position cap: on a perpetual the margin is at most the profile's cap x allocated equity, and notional = margin x
  leverage (the PM's perp rule); on spot the cap applies to the notional;
- leverage cap: notional at most allocated equity x leverage, less room for the fee;
- largest order cap, and a share of the bar's volume;
- on a perpetual, the PM's liquidation rule: the stop sits no further than half the way to the liquidation price
  (or the profile's tighter share). The size shrinks until it does; the leverage is never widened to make it fit.

The Advisor's guarantees (MUST FIX, 7 Oct 17:03), each pinned in tests/test_sizing.py:
1. Nothing is sized without a stop. A definition that declares none gets the fallback, ATR_STOP_MULTIPLE x Wilder's
   ATR(14), placed as a real stop by the caller; with no ATR either, the entry is skipped. So a central entry never
   carries more than the stopless 1x: it carries none.
2. Open risk of a position with no placed stop is notional x max(10%, 3 daily ATR), from open_risk.stopless_move, the
   one formula; risk_per_unit() uses it.
3. On a perpetual the stop is never further than half the distance to liquidation.

A stop fills as a market order, past its price: the loss per unit carries a named stop slippage, by default the larger
of half the spread and 0.05% (D13). Below the venue's smallest order, the entry rounds up to it only when the risk then
stays within 1.5 x the budget and no cap is broken; otherwise it is skipped, saying why."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from types import MappingProxyType
from collections.abc import Mapping

from sleeve_fund.money import money, scale

ATR_STOP_MULTIPLE = 2.5  # the fallback stop, in Wilder ATR(14)s, when a definition declares none
ROUND_UP_RISK = 1.5  # the venue minimum may be bought only while its risk stays within this many budgets
ROUND_UP_FLAG = 0.2  # a strategy rounding up on more than this share of its entries is too small for the instrument
DEFAULT_STOP_SLIPPAGE = 0.0005  # a stop fills past its price by the larger of half the spread and this (D13)
STOP_TO_LIQUIDATION = 0.5  # the PM's rule: a stop sits at most half the way to liquidation
OVERLAYS = ("stop", "vol_target")

_MONEY = ("allocated_equity", "price", "lot", "min_qty", "atr", "max_notional", "volume_notional")
_RATIOS = ("leg_cost", "half_spread", "risk_per_trade", "position_cap_pct", "leverage", "maintenance_margin",
           "stop_frac", "stop_slippage", "regime_weight", "fraction", "vol_target", "instrument_vol", "vol_floor",
           "stop_to_liquidation", "risk_long", "risk_short")


def rounds_up_too_often(entries: int, rounded_up: int) -> bool:
    """A strategy whose entries round up to the venue's minimum MORE than ROUND_UP_FLAG of the time is too small for
    the instrument at its risk per trade (Advisor, 6 Oct 16:45)."""
    return entries > 0 and rounded_up / entries > ROUND_UP_FLAG


def loss_at_stop(stop: float, leg: float, side: int = 1) -> float:
    """1R: the share of a position's entry value lost at its stop, costs included. The exit leg pays its cost on the
    exit's value: below the entry for a long, above it for a short."""
    return stop + leg + (1 - side * stop) * leg


@dataclass(frozen=True)
class SizingInputs:
    allocated_equity: Decimal  # allocation + this strategy's own P&L since the last rebalance (+ min(0, unrealised))
    price: Decimal  # the expected fill: close x (1 ± half spread) for the side, the same number as Intent.price
    side: int  # 1 long, -1 short
    lot: Decimal
    min_qty: Decimal
    leg_cost: float  # taker fee + half spread, share of notional
    half_spread: float
    risk_per_trade: float  # share of allocated equity lost at the stop, both sides unless overridden below
    position_cap_pct: float  # profile cap, on margin (perp) or notional (spot)
    leverage: float = 1.0
    perp: bool = False
    maintenance_margin: float = 0.0
    stop_frac: float | None = None  # the definition's stop; None means the fallback stop (2.5 x Wilder ATR)
    atr: Decimal | None = None
    stop_slippage: float | None = None  # None: max(half spread, 0.05%), the D13 rule
    regime_weight: float = 1.0
    fraction: float = 1.0
    overlay: str = "stop"
    vol_target: float | None = None
    instrument_vol: float | None = None
    vol_floor: float | None = None
    max_notional: Decimal | None = None
    volume_notional: Decimal | None = None
    stop_to_liquidation: float | None = None  # None: the PM's half way
    risk_long: float | None = None  # B1: per-side risk overrides, each defaulting to risk_per_trade
    risk_short: float | None = None

    def __post_init__(self) -> None:
        # Money is refused as a float (TypeError) and as NaN or Infinity; ratios are refused as NaN or Infinity.
        for name in _MONEY:
            v = getattr(self, name)
            if v is not None or name in ("allocated_equity", "price", "lot", "min_qty"):
                object.__setattr__(self, name, money(v, name))
        for name in _RATIOS:
            v = getattr(self, name)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{name} is a ratio: pass a number, not {type(v).__name__}")
            if not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number, got {v}")


@dataclass(frozen=True)
class Sizing:
    qty: Decimal  # 0 when skipped
    sized_by: str  # the limit that set the size
    stop_frac: float | None  # the stop actually placed (the fallback included)
    risk_budget: Decimal
    risk_amount: Decimal
    limits: Mapping[str, Decimal] = field(default_factory=dict)  # every limit as a notional, by name
    rounded_up: bool = False
    skipped: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "limits", MappingProxyType(dict(self.limits)))

    @property
    def ok(self) -> bool:
        return self.qty > 0 and not self.skipped


def _risk(i: SizingInputs) -> tuple[float, str]:
    if i.side > 0 and i.risk_long is not None:
        return i.risk_long, "risk per trade, long"
    if i.side < 0 and i.risk_short is not None:
        return i.risk_short, "risk per trade, short"
    return i.risk_per_trade, "risk per trade"


def _stop(i: SizingInputs) -> tuple[float | None, str]:
    if i.stop_frac is not None:
        return i.stop_frac, ""
    # No stop declared: the fallback, when there is an ATR to set it from; a zero or unknown one skips the entry.
    if i.atr is not None and i.atr > 0:
        # The stop is a ratio (a share of the price), so it leaves Decimal here; it is never stored as money (SZ-F1).
        return float(i.atr * Decimal(repr(ATR_STOP_MULTIPLE)) / i.price), \
            f"stop {ATR_STOP_MULTIPLE:g} x ATR(14), none declared"
    return None, ""


def _validate(i: SizingInputs) -> None:
    if i.side not in (1, -1):
        raise ValueError(f"side is 1 or -1, got {i.side}")
    if i.overlay not in OVERLAYS:
        raise ValueError(f"overlay is one of {OVERLAYS}, got {i.overlay!r}")
    if not 0 < i.regime_weight <= 1:
        raise ValueError(f"the regime weight is above 0 and at most 1, got {i.regime_weight}")
    if not 0 <= i.fraction <= 1:
        raise ValueError(f"the fraction of full size is between 0 and 1, got {i.fraction}")
    if i.price <= 0 or i.lot <= 0 or i.min_qty < 0:
        raise ValueError(f"price and lot must be above 0 and min_qty not below it, got {i.price}, {i.lot}, {i.min_qty}")
    for name in ("leg_cost", "half_spread", "risk_per_trade", "position_cap_pct", "maintenance_margin", "risk_long",
                 "risk_short"):
        if getattr(i, name) is None:
            continue
        if getattr(i, name) < 0:
            raise ValueError(f"{name} must not be negative, got {getattr(i, name)}")
    if not i.leverage > 0:
        raise ValueError(f"leverage must be above 0, got {i.leverage}")
    if i.stop_frac is not None and not 0 < i.stop_frac < 1:
        # A declared stop of 0 is refused, never sized or replaced by the fallback (Advisor, 6 Oct 16:45).
        raise ValueError(f"a declared stop must be above 0 and below 1, got {i.stop_frac}")
    if not i.perp and i.leverage != 1:
        raise ValueError("leverage applies only to a perpetual; spot is 1")
    if i.stop_to_liquidation is not None and not 0 < i.stop_to_liquidation <= STOP_TO_LIQUIDATION:
        raise ValueError(f"a stop may sit at most {STOP_TO_LIQUIDATION:.0%} of the way to liquidation, got "
                         f"{i.stop_to_liquidation}")


def stop_slippage(i: SizingInputs) -> float:
    """The named stop slippage the loss per unit carries: the venue's own, or max(half spread, 0.05%) (D13)."""
    return max(i.half_spread, DEFAULT_STOP_SLIPPAGE) if i.stop_slippage is None else i.stop_slippage


def size_entry(i: SizingInputs) -> Sizing:
    """The quantity for one new entry, the limit that set it, and its risk, or why it is skipped."""
    _validate(i)
    zero = Decimal(0)
    risk_pct, risk_name = _risk(i)
    stop, stop_note = _stop(i)
    equity = i.allocated_equity
    if stop is None:
        return Sizing(zero, "", None, zero, zero, skipped="no stop declared and no ATR yet: no sizing without a stop")
    if equity <= 0:
        return Sizing(zero, "", stop, zero, zero, skipped="no allocated equity to size from")
    loss = loss_at_stop(stop, i.leg_cost, i.side) + (1 - i.side * stop) * stop_slippage(i)  # per unit of notional
    budget = scale(equity, risk_pct * i.regime_weight * i.fraction)
    limits: dict[str, Decimal] = {risk_name: budget / Decimal(repr(loss))}
    if i.overlay == "vol_target":
        # No volatility, or no floor under it, skips the entry rather than sizing on a guess (Advisor, 16:45).
        known = [v for v in (i.instrument_vol, i.vol_floor) if v is not None]
        vol = max(known) if len(known) == 2 else 0.0
        if not i.vol_target or vol <= 0:
            return Sizing(zero, "", stop, budget, zero, skipped="volatility targeting needs a target, the instrument's "
                          "volatility and its one-year floor")
        limits["volatility target"] = scale(equity, i.vol_target / vol * i.regime_weight * i.fraction)
    limits["margin cap" if i.perp else "position cap"] = scale(equity, i.position_cap_pct * i.leverage)
    if i.perp:
        limits[f"{i.leverage:g}x leverage cap"] = scale(equity, i.leverage * (1 - i.leg_cost))
        # The whole account margins the position: the price can move about equity / notional - maintenance margin
        # before liquidation, so the stop fits within its share of that while notional stays under this.
        share = STOP_TO_LIQUIDATION if i.stop_to_liquidation is None else i.stop_to_liquidation
        limits[f"liquidation rule (stop within {share:.0%} of the way)"] = (
            equity / Decimal(repr(stop / share + i.maintenance_margin)))
    if i.max_notional is not None:
        limits["largest order cap"] = i.max_notional
    if i.volume_notional is not None:
        limits["share of the bar's volume"] = i.volume_notional
    sized_by = min(limits, key=limits.__getitem__)  # first named on a tie
    notional = max(limits[sized_by], zero)
    if stop_note:
        sized_by = f"{sized_by} ({stop_note})"
    per_unit_loss = scale(i.price, loss)
    qty = (notional / i.price).quantize(i.lot, rounding=ROUND_DOWN)
    if qty >= i.min_qty and qty > 0:
        return Sizing(qty, sized_by, stop, budget, qty * per_unit_loss, limits)
    # Below the venue's smallest order: buy the minimum only if its risk stays near the budget and no cap breaks.
    least = (i.min_qty / i.lot).to_integral_value(rounding=ROUND_UP) * i.lot if i.min_qty > 0 else i.lot
    least_notional = least * i.price
    least_risk = least * per_unit_loss
    hard = {k: v for k, v in limits.items() if k not in ("volatility target", risk_name)}  # caps, not sizes
    broken = [k for k, v in hard.items() if least_notional > v]
    if budget > 0 and least_risk <= scale(budget, ROUND_UP_RISK) and not broken:
        return Sizing(least, f"venue minimum (rounded up from {qty})", stop, budget, least_risk, limits,
                      rounded_up=True)
    head = f"entry skipped: {sized_by} comes to {qty}, below the venue's smallest order ({i.min_qty}), and "
    why = (f"the minimum would break the {broken[0]} ({hard[broken[0]]:,.2f})" if broken else
           f"rounding up would take its risk {least_risk:,.2f} over {ROUND_UP_RISK:g}x the {budget:,.2f} budget")
    return Sizing(zero, sized_by, stop, budget, zero, limits, skipped=head + why)


def margin_per_unit(i: SizingInputs) -> Decimal:
    """What one unit posts: price / leverage on a perpetual, the price on spot (the gate's Intent.margin_per_unit)."""
    return i.price / Decimal(repr(i.leverage)) if i.perp else i.price


def risk_per_unit(s: Sizing, i: SizingInputs, atr_pct: float | None = None) -> Decimal:
    """What one unit risks for the gate's Intent.risk_per_unit, by open_risk.position_risk's rule (note v4 section 2), so
    the gate counts an entry as it counts the position once held. With a stop: the distance from the expected fill to
    it, |price - stop price|. With none: the stopless measure, price x max(10%, 3 x the daily ATR share), from
    open_risk.stopless_move (Advisor guarantee 2). Costs and slippage are sizing's, in Sizing.risk_amount."""
    if s.stop_frac is not None:
        return scale(i.price, s.stop_frac)
    from sleeve_fund.open_risk import stopless_move

    if atr_pct is None or not math.isfinite(atr_pct):
        raise ValueError("the daily ATR isn't known, so a position with no stop can't be measured")
    return scale(i.price, stopless_move(atr_pct))


@dataclass(frozen=True)
class Step:
    qty: Decimal  # the signed order: the target less what is held; 0 when skipped or already there
    target: Decimal  # the fraction of full size this step aims at, in units
    skipped: str = ""  # why no order goes out, in words


def step_order(full_qty: Decimal, held_qty: Decimal, fraction: float, lot: Decimal, min_qty: Decimal) -> Step:
    """A8: the order that moves a position to `fraction` of its full size, fixed once at first entry (Advisor 6 Oct
    18:33). Only the difference is sent. A step smaller than the venue's minimum is skipped, saying why, and the target
    never drifts: the next step is still measured from what is held. Going to 0 always closes in full."""
    full_qty, held_qty = money(full_qty, "full_qty"), money(held_qty, "held_qty")
    lot, min_qty = money(lot, "lot"), money(min_qty, "min_qty")
    if not 0 <= fraction <= 1:
        raise ValueError(f"the fraction of full size is between 0 and 1, got {fraction}")
    exact = scale(full_qty, fraction) / lot
    # 1/3 as a float is a hair under a third: a target within a millionth of a step of a whole step is that step,
    # anything else rounds down, as every size does.
    steps = exact.to_integral_value() if abs(exact - exact.to_integral_value()) < Decimal("1e-6") else \
        exact.to_integral_value(rounding=ROUND_DOWN)
    target = steps * lot
    order = target - held_qty
    if order == 0:
        return Step(Decimal(0), target)
    if target != 0 and abs(order) < min_qty:
        return Step(Decimal(0), target, skipped=f"the step to {fraction:.0%} of full size is {abs(order)}, below the "
                                                f"venue's smallest order ({min_qty})")
    return Step(order, target)
