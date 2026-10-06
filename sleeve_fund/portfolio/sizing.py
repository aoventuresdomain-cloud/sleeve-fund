"""Central sizing (v2 P2-1, with A8, B1 and B2): one pure function sizes every entry, in backtest and paper alike.

The risk budget is the strategy's allocated equity × its risk per trade (per side, B1) × the regime weight fixed at
entry × the definition's fraction of full size (A8). The quantity is that budget over what one unit loses if the
stop is hit, costs included. Volatility targeting (B2) sizes the notional from the instrument's volatility instead,
and still declares a stop. Then the caps, smallest first wins:

- margin cap: margin ≤ the profile's position cap × allocated equity; on a perpetual, notional = margin × leverage
  (the PM's perp rule);
- leverage cap: notional ≤ allocated equity × leverage, less room for the fee;
- largest order cap, and a share of the bar's volume;
- on a perpetual, the PM's liquidation rule: the stop sits no further than stop_to_liquidation of the way to the
  liquidation price. The size shrinks until it does; the leverage is never widened to make it fit.

With both a stop and a volatility target, both sizes are worked out and the smaller is taken, never their product.
The instrument's volatility is floored (its trailing one-year 25th percentile, from the caller), so a quiet stretch
doesn't size up into the caps. A stop fills as a market order, often past its price: the loss per unit carries a
named stop slippage, one extra half spread until fills are measured against the model (Independent Quant Advisor,
6 Oct 2026).

Below the venue's smallest order, the entry rounds up to it only when the risk then stays within 1.5 × the budget
and no cap is broken; otherwise it is skipped, saying why. There is no sizing without a stop: a definition that
declares none falls back to ATR_STOP_MULTIPLE × ATR(14), and with no ATR either the entry is skipped.

Nothing here reads a clock, a venue or the database, so every case is a plain function call to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_UP, Decimal

ATR_STOP_MULTIPLE = 2.5  # the fallback stop, in ATR(14)s, when a definition declares none (spec: 2-3x)
ROUND_UP_RISK = 1.5  # the venue minimum may be bought only while its risk stays within this many budgets
ROUND_UP_FLAG = 0.2  # a strategy rounding up on more than this share of its entries is too small for the instrument
# A stop fills as a market order, past its price: by the venue's named setting, or by default the larger of half the
# spread and this (Independent Quant Advisor, 6 Oct 2026, 16:45).
DEFAULT_STOP_SLIPPAGE = 0.0005
OVERLAYS = ("stop", "vol_target")


def loss_at_stop(stop: float, leg: float, side: int = 1) -> float:
    """1R: the share of a position's entry value lost at its stop, costs included. The exit leg pays its cost on
    the exit's value: below the entry for a long, above it for a short."""
    return stop + leg + (1 - side * stop) * leg


@dataclass(frozen=True)
class SizingInputs:
    allocated_equity: float  # the strategy's allocated equity now: its allocation, floating with its own P&L
    price: float  # the entry price the order is sized at (the bar's close)
    side: int  # 1 long, -1 short
    leg_cost: float  # taker fee + half the spread, as a share of a leg's notional
    half_spread: float  # half the bid-ask spread, the stop slippage's default
    risk_per_trade: float  # share of allocated equity lost at the stop, both sides unless set per side below
    position_cap_pct: float  # the profile's cap, applied to the margin (on spot, to the notional)
    lot: Decimal  # the venue's quantity step
    min_qty: Decimal  # the venue's smallest order
    stop_frac: float | None = None  # the definition's stop distance as a share of the price
    atr: float | None = None  # ATR(14) in price units, for the fallback stop
    risk_long: float | None = None  # B1: per-side risk, each defaulting to risk_per_trade
    risk_short: float | None = None
    regime_weight: float = 1.0  # fixed at entry; floor above zero (north star rule 1)
    fraction: float = 1.0  # A8: the definition's target fraction of full size
    leverage: float = 1.0  # the profile's leverage cap; 1 on spot
    perp: bool = False
    max_notional: float | None = None  # the largest order cap
    volume_notional: float | None = None  # the most a share of the bar's volume is worth
    overlay: str = "stop"  # B2: "stop" (risk to the stop) or "vol_target"
    vol_target: float | None = None  # target volatility of the position, per bar, as a share of allocated equity
    instrument_vol: float | None = None  # the instrument's volatility per bar (ATR / price, or the std of returns)
    vol_floor: float | None = None  # the floor under instrument_vol: its trailing one-year 25th percentile
    stop_slippage: float | None = None  # past the stop price, as a share; None: max(half spread, 0.05%)
    maintenance_margin: float = 0.0  # a perpetual's maintenance margin rate, for the liquidation rule
    stop_to_liquidation: float | None = None  # the profile's share of the way to liquidation a stop may sit


@dataclass
class Sizing:
    qty: Decimal  # 0 when skipped
    sized_by: str  # the limit that set the size, for the journal
    stop_frac: float | None
    risk_budget: float  # what the trade may lose at its stop
    risk_amount: float  # what this quantity loses at its stop, costs included
    limits: dict = field(default_factory=dict)  # every limit as a notional, by name
    rounded_up: bool = False
    skipped: str = ""  # why no order goes out, in words

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
        # A declared or computed stop is used as it is: one of 0 (or not a number) skips the entry, never falling
        # back silently to the ATR stop (Advisor, 16:45).
        return (i.stop_frac, "") if i.stop_frac == i.stop_frac and i.stop_frac > 0 else (None, "")
    if i.atr is not None and i.atr > 0 and i.price > 0:
        return ATR_STOP_MULTIPLE * i.atr / i.price, f"stop {ATR_STOP_MULTIPLE:g} x ATR(14), none declared"
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
    for name in ("price", "leg_cost", "risk_per_trade", "position_cap_pct", "leverage"):
        v = getattr(i, name)
        if not v == v or v < 0 or (name in ("price", "leverage") and v <= 0):
            raise ValueError(f"{name} must be a positive number, got {v}")
    if not i.perp and i.leverage != 1:
        raise ValueError("leverage applies only to a perpetual; spot is 1")


def size_entry(i: SizingInputs) -> Sizing:
    """The quantity for one new entry, the limit that set it, and its risk, or why it is skipped."""
    _validate(i)
    risk_pct, risk_name = _risk(i)
    stop, stop_note = _stop(i)
    equity = i.allocated_equity
    if stop is None:
        why = (f"its stop came to {i.stop_frac} of the price, so the entry can't be sized" if i.stop_frac is not None
               else "no stop declared and no ATR yet: no sizing without a stop")
        return Sizing(Decimal(0), "", None, 0.0, 0.0, skipped=why)
    if equity <= 0:
        return Sizing(Decimal(0), "", stop, 0.0, 0.0, skipped="no allocated equity to size from")
    slip = max(i.half_spread, DEFAULT_STOP_SLIPPAGE) if i.stop_slippage is None else i.stop_slippage
    loss = loss_at_stop(stop, i.leg_cost, i.side) + (1 - i.side * stop) * slip  # per unit of notional
    budget = equity * risk_pct * i.regime_weight * i.fraction
    limits: dict[str, float] = {risk_name: budget / loss if loss > 0 else float("inf")}
    if i.overlay == "vol_target":
        # No volatility, or no floor under it, skips the entry rather than sizing on a guess (Advisor, 16:45).
        known = [v for v in (i.instrument_vol, i.vol_floor) if v is not None and v == v]
        vol = max(known) if len(known) == 2 else 0.0
        if not i.vol_target or vol <= 0:
            return Sizing(Decimal(0), "", stop, budget, 0.0, skipped="volatility targeting needs a target, the "
                          "instrument's volatility and its one-year floor")
        limits["volatility target"] = i.vol_target * equity / vol * i.regime_weight * i.fraction
    limits["margin cap" if i.perp else "position cap"] = i.position_cap_pct * equity * i.leverage
    if i.perp:
        limits[f"{i.leverage:g}x leverage cap"] = equity * i.leverage * (1 - i.leg_cost)
    if i.perp and i.stop_to_liquidation:
        # The whole account margins the position: the price can move about equity / notional - maintenance
        # margin before liquidation, so the stop fits within its share of that while notional stays under this.
        limits[f"liquidation rule (stop within {i.stop_to_liquidation:.0%} of the way)"] = (
            equity / (stop / i.stop_to_liquidation + i.maintenance_margin))
    if i.max_notional is not None:
        limits["largest order cap"] = i.max_notional
    if i.volume_notional is not None:
        limits["share of the bar's volume"] = i.volume_notional
    sized_by = min(limits, key=limits.get)
    if stop_note:
        sized_by = f"{sized_by} ({stop_note})"
    notional = min(limits.values())
    price = Decimal(str(i.price))
    qty = (Decimal(str(max(notional, 0.0))) / price).quantize(i.lot, rounding=ROUND_DOWN)
    out = Sizing(qty, sized_by, stop, budget, float(qty * price) * loss, limits)
    if qty >= i.min_qty and qty > 0:
        return out
    # Below the venue's smallest order: buy the minimum only if its risk stays near the budget and no cap breaks.
    least = i.min_qty.quantize(i.lot, rounding=ROUND_UP) if i.min_qty > 0 else i.lot
    least_notional = float(least * price)
    least_risk = least_notional * loss
    hard = {k: v for k, v in limits.items() if k not in ("volatility target", risk_name)}  # caps, not sizes
    broken = [k for k, v in hard.items() if least_notional > v]
    if budget > 0 and least_risk <= ROUND_UP_RISK * budget and not broken:
        return Sizing(least, f"venue minimum (rounded up from {qty})", stop, budget, least_risk, limits,
                      rounded_up=True)
    why = (f"the {broken[0]} ({hard[broken[0]]:,.2f})" if broken else
           f"its risk {least_risk:,.2f} is over {ROUND_UP_RISK:g}x the {budget:,.2f} budget")
    return Sizing(Decimal(0), sized_by, stop, budget, 0.0, limits,
                  skipped=f"entry skipped: {sized_by} comes to {qty}, below the venue's smallest order ({i.min_qty}), "
                          f"and the minimum would break {why}" if broken else
                          f"entry skipped: {sized_by} comes to {qty}, below the venue's smallest order ({i.min_qty}), "
                          f"and rounding up would take {why}")


@dataclass
class Step:
    qty: Decimal  # the signed order: the target less what is held; 0 when skipped or already there
    target: Decimal  # the fraction of full size this step aims at, in units
    skipped: str = ""  # why no order goes out, in words


def step_order(full_qty: Decimal, held_qty: Decimal, fraction: float, lot: Decimal, min_qty: Decimal) -> Step:
    """A8: the order that moves a position to `fraction` of its full size, fixed once at first entry (Advisor
    18:33). Only the difference is sent. A step smaller than the venue's minimum is skipped, saying why, and the
    target never drifts: the next step is still measured from what is held. Going to 0 always closes in full."""
    if not 0 <= fraction <= 1:
        raise ValueError(f"the fraction of full size is between 0 and 1, got {fraction}")
    exact = full_qty * Decimal(repr(fraction)) / lot
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
