"""Central sizing (v2 P2-1, with A8, B1 and B2): one pure function sizes every entry, in backtest and paper alike.

The risk budget is the strategy's allocated equity × its risk per trade (per side, B1) × the regime weight fixed at
entry × the definition's fraction of full size (A8). The quantity is that budget over what one unit loses if the
stop is hit, costs included. Volatility targeting (B2) sizes the notional from the instrument's volatility instead,
and still declares a stop. Then the caps, smallest first wins:

- margin cap: margin ≤ the profile's position cap × allocated equity; on a perpetual, notional = margin × leverage
  (the PM's perp rule);
- leverage cap: notional ≤ allocated equity × leverage, less room for the fee;
- largest order cap, and a share of the bar's volume.

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
    if i.stop_frac is not None and i.stop_frac > 0:
        return i.stop_frac, ""
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
        return Sizing(Decimal(0), "", None, 0.0, 0.0, skipped="no stop declared and no ATR yet: no sizing without "
                      "a stop")
    if equity <= 0:
        return Sizing(Decimal(0), "", stop, 0.0, 0.0, skipped="no allocated equity to size from")
    loss = loss_at_stop(stop, i.leg_cost, i.side)  # per unit of notional
    budget = equity * risk_pct * i.regime_weight * i.fraction
    limits: dict[str, float] = {}
    if i.overlay == "vol_target":
        if not i.vol_target or not i.instrument_vol or i.instrument_vol <= 0:
            return Sizing(Decimal(0), "", stop, budget, 0.0, skipped="volatility targeting needs a target and the "
                          "instrument's volatility")
        limits["volatility target"] = i.vol_target * equity / i.instrument_vol * i.regime_weight * i.fraction
    else:
        limits[risk_name] = budget / loss if loss > 0 else float("inf")
    limits["margin cap" if i.perp else "position cap"] = i.position_cap_pct * equity * i.leverage
    if i.perp:
        limits[f"{i.leverage:g}x leverage cap"] = equity * i.leverage * (1 - i.leg_cost)
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
    hard = {k: v for k, v in limits.items() if k not in ("volatility target", risk_name)}
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
