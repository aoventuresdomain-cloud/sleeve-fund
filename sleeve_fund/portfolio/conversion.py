"""P2-1b: a weight-sized model (Donchian, the vol_target trend filter) traded on a perpetual as entries and exits,
sized once at entry by central sizing (Advisor's spec, v2/p2-1b-weight-to-entry-spec.md, and rulings of 6 Oct 17:09).
Pure rules, no engine: the strategy applies them bar by bar, and the tests pin them on synthetic series."""

from __future__ import annotations

from decimal import Decimal


def _sign(w: float) -> int:
    return 1 if w > 0 else -1 if w < 0 else 0


def trades_from_weights(weights, stopped=()) -> list[tuple[int, str, int]]:
    """The trades a weight series makes, as (bar, "entry" | "exit", side): an entry when the weight leaves 0, an exit
    when it returns to 0, and a sign change an exit then an entry on the same bar (rule 1, ruling 4). A change of size
    within one sign trades nothing. `stopped`: bars a stop filled on; after one the model re-enters only once the
    weight has been 0 and turned on again (ruling 6)."""
    out, held, wait = [], 0, False
    for i, w in enumerate(weights):
        side = _sign(w)
        if held and i in stopped:
            out.append((i, "exit", held))
            held, wait = 0, side != 0
            continue
        if held and side != held:
            out.append((i, "exit", held))
            held = 0
        if wait:
            wait = side != 0
            continue
        if not held and side:
            out.append((i, "entry", side))
            held = side
    return out


def band_target(central_qty: Decimal, weight: float, full_weight: float) -> Decimal:
    """The band variant's target size: central sizing at this close times the weight as a share of full weight
    (ruling 5)."""
    if not full_weight:
        return Decimal(0)
    return central_qty * abs(Decimal(repr(float(weight)))) / abs(Decimal(repr(float(full_weight))))


def band_order(target_qty: Decimal, held_qty: Decimal, band: Decimal = Decimal("0.25")) -> Decimal:
    """The signed order the band variant sends: the difference to the target once it is more than `band` of the size
    held away from it, else 0 (ruling 5: exactly 25% trades nothing). From flat it is the target."""
    held = abs(held_qty)
    if held == 0:
        return target_qty
    diff = abs(target_qty) - held
    return diff if abs(diff) > band * held else Decimal(0)
