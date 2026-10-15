"""Advisor STOP-EXEC (15 Oct 2026): one position's resting reduce-only stop at the venue, moved by PLACE-THEN-CANCEL.

The exchange client's amend drops a new trigger price without a word (findings:
/mnt/project-files/sleeve-fund/qd/exec-client-stop-findings.md), so a level change is always a new stop placed first,
and the old one is cancelled only once the venue has accepted the new one. Two reduce-only stops can never over-close:
whichever triggers first closes, and the other can only reduce what is left. The planner keeps no clock and sends
nothing: each call returns the actions for its owner to send, and the owner feeds back the venue's answers.

Actions: ("place", order_id, level, qty), ("retry", order_id, level, qty) (a place sent after the limiter's back-off),
("cancel", order_id), ("amend", order_id, qty), ("lookup", order_id), ("close", qty, why) (a reduce-only market close
of this Strategy's quantity) and ("alert", "critical" | "warning", words). Built OFF: nothing calls it until the client is switched on."""

from __future__ import annotations

import uuid
from decimal import Decimal

from sleeve_fund.paper.exec_client import stop_change

MAX_RESTING = 2  # never more than two resting stops per position (STOP-EXEC)
ID_PREFIX = "sf-"  # the fund's namespace on a shared account (Advisor STOP-CLAIM): only these are ever ours to cancel


def _d(x) -> Decimal:
    return Decimal(str(x))




def new_order_id() -> str:
    """A venue order id in the fund's namespace that fits its 36 characters (EXEC-SPEC MUST 5): the prefix and a UUID's
    32 hex digits."""
    return ID_PREFIX + uuid.uuid4().hex


def ours(order_id: str) -> bool:
    return order_id.startswith(ID_PREFIX)


class StopMover:
    def __init__(self, *, side: str, qty) -> None:
        self.side = side
        self.step = _d(qty).as_tuple().exponent  # quantities keep the venue's step they came with
        self.qty = self._s(_d(qty))  # what the position holds, so what every resting stop covers at most
        self.current: str | None = None  # the accepted stop the Strategy counts on
        self.levels: dict[str, str] = {}  # every stop placed and not known gone: order id -> level
        self.pending: str | None = None  # placed, not yet accepted
        self.cancelling: set[str] = set()  # cancel sent (or failed, or unanswered), not confirmed gone
        self.waiting: str | None = None  # the latest level asked for while two stops rest
        self.retried: str | None = None  # the order id of a transient refusal's one retry

    def _s(self, x: Decimal) -> str:
        return "0" if x == 0 else format(x.quantize(Decimal(1).scaleb(self.step)), "f")

    def resting(self) -> set[str]:
        return set(self.levels)

    # the Strategy's asks --------------------------------------------------------------------------------------------
    def move(self, level) -> list[tuple]:
        """The stop should be at `level`. Places a new stop (the first, or a replacement) or, with two resting or a
        replacement unconfirmed, waits for the earlier cancel; the latest level asked for wins."""
        level = str(level)
        if _d(self.qty) <= 0:
            return []
        if self.current is not None and self.pending is None and not self.cancelling and \
                stop_change((self.levels[self.current], self.qty), (level, self.qty)) == "none":
            return []
        if self.pending is not None or len(self.levels) >= MAX_RESTING:
            self.waiting = level
            return []
        oid = new_order_id()
        self.pending, self.levels[oid] = oid, level
        return [("place", oid, level, self.qty)]

    def resize(self, qty) -> list[tuple]:
        """A quantity change alone (an add, a partial close): the client's amend carries a quantity."""
        self.qty = self._s(_d(qty))
        return [("amend", oid, self.qty) for oid in sorted(self.levels) if oid not in self.cancelling]

    # the venue's answers --------------------------------------------------------------------------------------------
    def on_accepted(self, oid: str) -> list[tuple]:
        if oid != self.pending:
            return []
        old, self.current, self.pending = self.current, oid, None
        if old is None:
            return self._release()
        self.cancelling.add(old)
        return [("cancel", old)]

    def on_rejected(self, oid: str, why: str, kind: str = "other") -> list[tuple]:
        """The venue refused a place, by reject class (Advisor STOP-RETRY; exec_client.reject_class): "transient" is
        placed again once after the limiter's back-off, then left to the next stop update, never looped; "crossed"
        (the price is already past the new level) means the Strategy's stop is hit, so its quantity closes at market
        now; anything else is not retried. Every refusal alerts, CRITICAL when no stop is left resting."""
        level = self.levels.pop(oid, None)
        if oid == self.pending:
            self.pending = None
        retried, self.retried = oid == self.retried, None
        if kind == "crossed":
            return [("close", self.qty, "stop level already crossed"),
                    ("alert", "warning", f"The new stop at {level} was already crossed ({why}): closing at market")]
        if kind == "transient" and not retried and level is not None:
            again = new_order_id()
            self.pending, self.levels[again], self.retried = again, level, again
            return [("retry", again, level, self.qty)]
        kept = f"; the stop at {self.levels[self.current]} stays" if self.current in self.levels else ""
        return [("alert", "critical" if self.current not in self.levels else "warning",
                 f"The venue refused the new stop at {level}: {why}{kept}")] + self._release()

    def on_cancelled(self, oid: str) -> list[tuple]:
        self.cancelling.discard(oid)
        self.levels.pop(oid, None)
        return self._release()

    def on_cancel_failed(self, oid: str, why: str) -> list[tuple]:
        return [("lookup", oid), ("alert", "warning", (f"Cancelling the old stop {oid} failed ({why}): looking it up "
                                                        "and retrying; two stops rest meanwhile"))]

    def on_cancel_unknown(self, oid: str) -> list[tuple]:
        return [("lookup", oid), ("alert", "warning", (f"No answer to cancelling the old stop {oid}: looking it up; "
                                                        "two stops rest meanwhile"))]

    def on_lookup(self, oid: str, *, resting: bool) -> list[tuple]:
        if resting:
            return [("cancel", oid)] if oid in self.cancelling else []
        return self.on_cancelled(oid)

    def on_filled(self, oid: str | None, filled) -> list[tuple]:
        """A stop (part-)filled, or (oid None) a market close of this position filled: the position is smaller, so every
        other resting stop covers only what is left, and none is left resting once it is flat."""
        left = max(_d(self.qty) - _d(filled), Decimal(0))
        self.qty = self._s(left)
        others = sorted(o for o in self.levels if o != oid)
        if left == 0:
            self.levels.pop(oid, None)
            self.current, self.waiting = None, None
            out = []
            for o in others:
                self.cancelling.add(o)
                out.append(("cancel", o))
            return out
        return [("amend", o, self.qty) for o in others]

    def _release(self) -> list[tuple]:
        if self.waiting is None or self.pending is not None or len(self.levels) >= MAX_RESTING:
            return []
        level, self.waiting = self.waiting, None
        return self.move(level)

    # a restart ------------------------------------------------------------------------------------------------------
    @staticmethod
    def reconcile(venue_resting: list[str], journal_current: str | None) -> list[tuple]:
        """Every resting stop at the venue IN THE FUND'S NAMESPACE that the journal doesn't own as its current stop is
        cancelled (STOP-EXEC, narrowed by STOP-CLAIM); one outside it is never touched and raises a CRITICAL alert."""
        out = []
        for oid in venue_resting:
            if not ours(oid):
                out.append(("alert", "critical", f"Order {oid} on the account is not the fund's: left alone, entries "
                                                 "held until the account reconciles"))
            elif oid != journal_current:
                out.append(("cancel", oid))
        return out
