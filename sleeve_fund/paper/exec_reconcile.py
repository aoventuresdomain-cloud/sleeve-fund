"""EXEC-CLIENT restart reconcile (PE2, head 3; ships OFF, nothing calls it while EXEC_MODE is off): at start, one
account's position and open orders on one instrument, as the venue reports them, set against the journal's, per Strategy
tag.

- Equal (the venue's net is the sum of the tags' journal nets, exactly, and every open order at the venue is a journaled
  open order with the same remaining quantity, and the other way round): each tag keeps its journal quantity.
- Anything else is a gap, named. Several Strategies on one account can only be split back to their tags from the
  journal, so a gap never splits or reassigns anything by guess (DRILL-VENUE MUST 3): it holds new entries on every tag
  on that account and instrument, with an error event the alerts send. Exits, stops, the liquidation cut and Stop are
  never held. The hold lifts only on a clean re-run, never on a timer.

Venue figures are the fields of GET /v5/position/list and GET /v5/order/realtime (SOURCES), the same ones
sleeve_fund.mirror reads on the demo account. A figure that can't be read is a gap, never an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation

SOURCES = {"position": "https://bybit-exchange.github.io/docs/v5/position",
           "open orders": "https://bybit-exchange.github.io/docs/v5/order/open-order"}
HOLD, CLEARED = "exec_reconcile_hold", "exec_reconcile_cleared"
_SIGN = {"Buy": 1, "Sell": -1}


class Unreadable(ValueError):
    """A venue figure the reconcile can't read; the caller holds."""


@dataclass(frozen=True)
class Result:
    instrument: str
    tags: tuple[str, ...]
    gaps: tuple[str, ...]
    per_tag: dict[str, Decimal] = field(default_factory=dict)  # empty unless clean: nothing is split by guess

    @property
    def clean(self) -> bool:
        return not self.gaps


def _qty(raw, what: str) -> Decimal:
    try:
        q = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError):
        raise Unreadable(f"{what} {raw!r} isn't a number") from None
    if not q.is_finite() or q < 0:
        raise Unreadable(f"{what} {raw!r} isn't a quantity")
    return q


def venue_net(rows: list[dict]) -> Decimal:
    """The account's signed position from /v5/position/list rows: Buy long, Sell short, side "" flat. One-way mode
    only (positionIdx 0): a hedge-mode pair is refused, never netted."""
    net = Decimal(0)
    for r in rows:
        if r.get("positionIdx", 0) not in (0, "0"):
            raise Unreadable("the account isn't in one-way mode (positionIdx is not 0)")
        if "size" not in r:
            raise Unreadable("a position row has no size")
        size, side = _qty(r["size"], "position size"), r.get("side") or ""
        if side in _SIGN:
            net += _SIGN[side] * size
        elif side not in ("", "None") or size:
            raise Unreadable(f"position side {side!r} with size {size}")
    return net


def venue_open(rows: list[dict]) -> dict[str, Decimal]:
    """The account's open orders from /v5/order/realtime rows, by our own order id (orderLinkId) to what's left."""
    out: dict[str, Decimal] = {}
    for r in rows:
        link = (r.get("orderLinkId") or "").strip()
        if not link:
            raise Unreadable(f"open order {r.get('orderId')!r} has no orderLinkId of ours")
        out[link] = _qty(r.get("leavesQty"), f"open order {link} remaining qty")
    return out


def reconcile(instrument: str, journal: dict[str, Decimal], journal_open: dict[str, Decimal],
              venue_position: list[dict], venue_orders: list[dict]) -> Result:
    """`journal`: each Strategy tag's net on `instrument` on this account; `journal_open`: its open orders there, by
    our order id to what's left. The venue rows as the two endpoints return them."""
    tags = tuple(journal)
    try:
        net, open_ = venue_net(venue_position), venue_open(venue_orders)
    except Unreadable as e:
        return Result(instrument, tags, (f"the venue's figures couldn't be read: {e}",))
    gaps = []
    ours = sum(journal.values(), Decimal(0))
    if net != ours:
        gaps.append(f"the account holds {net} {instrument}; the journal's Strategies hold {ours} between them")
    for link in sorted(set(open_) - set(journal_open)):
        gaps.append(f"open order {link} on the account isn't in the journal")
    for link in sorted(set(journal_open) - set(open_)):
        gaps.append(f"journaled open order {link} isn't open on the account")
    for link in sorted(set(open_) & set(journal_open)):
        if open_[link] != journal_open[link]:
            gaps.append(f"open order {link} has {open_[link]} left on the account, {journal_open[link]} in the journal")
    if gaps:
        return Result(instrument, tags, tuple(gaps))
    return Result(instrument, tags, (), dict(journal))


def record(store, account: str, instrument: str, result: Result, now: datetime) -> None:
    """The episode in the journal, per tag: a gap opens a hold (an error, which the alerts send) on every tag not
    already held; a clean run clears every tag that is. A clean run on a tag never held writes nothing."""
    from sleeve_fund.wording import no_venues

    for tag in result.tags:
        held = hold(store, tag) is not None
        if not result.clean and not held:
            store.event(tag, "error", HOLD, no_venues(
                f"Restart check on account {account}: {'; '.join(result.gaps)}. No new entries on {instrument} for "
                f"{', '.join(result.tags)} until a restart check ties; exits and stops still go out. Check it now"),
                ts=now)
        elif result.clean and held:
            store.event(tag, "info", CLEARED, f"Restart check on account {account} ties on {instrument}: entries "
                        "are allowed again", ts=now)


def hold(store, tag: str) -> str | None:
    """Why `tag` may not open or add after a restart check that didn't tie, or None. Read from the journal (the latest
    hold or clear), so a restart neither lifts nor repeats it. A journal that can't be read holds (fail closed)."""
    try:
        last = store.last_event(tag, (HOLD, CLEARED))
    except Exception:  # noqa: BLE001 - fail closed, never an error out of the entry gate
        return "whether the restart check tied couldn't be read from the journal."
    if last is None or last["kind"] != HOLD:
        return None
    return last["message"]
