"""P2-8 two legs: one Strategy holding two positions at once, such as spot + perp carry or a pair (spec
pe2-plans/p2-8-spec.md; Independent Quant Advisor, 13 Oct 23:55 UK).

Each leg is a sleeves row of its own, the second "<name>~2" under the first (Store.create_two_legs), so every money path
(fills, the journal's book, marks, funding, liquidation, the reconcile, the start check and the nightly recon) works per
leg unchanged, and both legs' orders carry one parent_trade_id. The gate decides a two-leg entry as one position
(portfolio.gate.check_pair). This module holds the rules a pair is made under.

A leg is a dict: instrument, params (its market and settings), venue, side (+1 long, -1 short), ratio (its notional per
unit of the first leg's; the first's is 1) and, on a perp, leverage (1 by default, 2 at most)."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from sleeve_fund import markets

MAX_LEG_LEVERAGE = Decimal(2)  # Advisor B: the perp leg 1x by default, 2x hard cap
CENT = Decimal("0.01")


class TwoLegError(ValueError):
    """A two-leg Strategy that can't be made, with the reason."""


def _dec(x, what: str) -> Decimal:
    try:
        d = Decimal(str(x))
    except Exception:  # noqa: BLE001 - any unreadable figure is refused
        raise TwoLegError(f"{what} must be a number, not {x!r}") from None
    if not d.is_finite():
        raise TwoLegError(f"{what} must be a number, not {x!r}")
    return d


def check_legs(first: dict, second: dict) -> tuple[dict, dict]:
    """Each leg's side, ratio and leverage checked and filled in; raises TwoLegError on a pair that isn't allowed."""
    out = []
    for n, leg in (("first", first), ("second", second)):
        if not leg.get("instrument"):
            raise TwoLegError(f"the {n} leg names no instrument")
        side = leg.get("side")
        if side not in (1, -1):
            raise TwoLegError(f"the {n} leg's side must be 1 (long) or -1 (short), not {side!r}")
        perp = markets.is_perp(leg.get("params"))
        if side == -1 and not perp:
            raise TwoLegError(f"the {n} leg is a short on spot: a short leg must be a perp (no spot borrowing)")
        ratio = Decimal(1) if n == "first" else _dec(leg.get("ratio", 1), f"the {n} leg's ratio")
        if n == "first" and "ratio" in leg and _dec(leg["ratio"], "the first leg's ratio") != 1:
            raise TwoLegError("the first leg's ratio is 1: the second leg's ratio is per unit of the first's")
        if ratio <= 0:
            raise TwoLegError(f"the {n} leg's ratio must be above 0, not {ratio}")
        lev = _dec(leg.get("leverage", 1), f"the {n} leg's leverage")
        if not perp and lev != 1:
            raise TwoLegError(f"the {n} leg is spot, which has no leverage")
        if not Decimal(1) <= lev <= MAX_LEG_LEVERAGE:
            raise TwoLegError(f"the {n} leg's leverage must be from 1x to {MAX_LEG_LEVERAGE}x, not {lev}x")
        out.append({**leg, "side": side, "ratio": ratio, "leverage": lev})
    if first["instrument"] == second["instrument"] and markets.is_perp(first.get("params")) == markets.is_perp(
            second.get("params")):
        raise TwoLegError("the two legs are the same instrument on the same market")
    if (first.get("venue") or "").upper() != (second.get("venue") or "").upper():
        raise TwoLegError("both legs trade on one venue: a pair across venues waits for the router")
    return out[0], out[1]


def split_capital(total, legs: tuple[dict, dict]) -> tuple[Decimal, Decimal]:
    """The Strategy's balance split by what each leg needs at its ratio: its notional on spot, its notional over its
    leverage as margin on a perp. To the cent, the first leg's share rounded down and the rest the second's, so the
    two add up to the balance exactly."""
    first, second = check_legs(*legs)
    total = _dec(total, "the starting balance")
    if total <= 0:
        raise TwoLegError(f"the starting balance must be above 0, not {total}")
    need = [leg["ratio"] / (leg["leverage"] if markets.is_perp(leg.get("params")) else 1) for leg in (first, second)]
    one = (total * need[0] / (need[0] + need[1])).quantize(CENT, rounding=ROUND_DOWN)
    return one, total - one


def leg_params(leg: dict) -> dict:
    """A checked leg's row params: its own market settings, with its side, ratio and leverage kept beside them."""
    return {**(leg.get("params") or {}), "leg": {"side": leg["side"], "ratio": str(leg["ratio"]),
                                                 "leverage": str(leg["leverage"])}}
