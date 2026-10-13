"""ORPHAN-GUARD (Advisor 8 Oct 06:15 and 06:20 UK; HoQA cells og1-og5): a resting entry never outlives its decision bar
unwatched, whatever happens to the process that sent it. The layers, first to last:

- og1 First line: sent good-till-date, expiring at its decision bar's next close, where the venue supports it.
- W2  Its own process cancels it at that close (LongFlatStrategy._cancel_gated_at_close).
- og2 The supervisor cancels a dead owner's resting entries within 60 s (the DA's cancel command, migration 0015).
- og3 A restart cancels leftovers before its first new entry.
- og4 An entry still resting at its bar's close + 2 min alerts once and blocks that strategy's entries (the supervisor's
      check, so a dead owner can't silence it).
- og5 An orphan that fills anyway is closed by one reduce-only market order through the exit path.

This module holds the pure parts. The journal's half (og2 to og5) builds on 0015's Store.request_cancel and
apply_cancel (data-architect/0015-orphan-guard-design.md)."""

from __future__ import annotations

from datetime import datetime, timedelta

from nautilus_trader.model import TimeInForce

GTD_MARGIN = timedelta(seconds=30)  # og1: past the venue's minimum lead, so a short bar's expiry is never refused
OVERDUE_AFTER = timedelta(minutes=2)  # og4: past its bar's close by this, a resting entry is an orphan (Advisor 06:15)
NO_GTD_NOTE = ("Sent good-till-cancelled: this venue takes no good-till-date orders, so only the backstops cover this "
               "resting entry (its process's cancel at the next bar close, the supervisor's cancel if that process "
               "goes down, the restart's reconcile and the alert 2 minutes past the bar's close)")


def resting_entry_order(*, gtd_supported: bool, decision_close: datetime, bar: timedelta,
                        now: datetime | None = None,
                        min_lead: timedelta = timedelta(0)) -> tuple[TimeInForce, datetime | None, str | None]:
    """og1: a resting entry's time in force, expiry and journal note. Good-till-date to the decision bar's next close
    where the venue supports it, or, when that is nearer than the venue takes (its minimum lead, e.g. a 1-minute bar),
    to now + that lead + GTD_MARGIN (Advisor 13 Oct: never refuse resting entries on short bars; the bar-close cancel
    still takes it at the close). With no good-till-date, good-till-cancelled with a note that only the backstops
    cover it. It is sent either way."""
    if bar <= timedelta(0):
        raise ValueError(f"a bar is a positive length, not {bar}")
    if not gtd_supported:
        return TimeInForce.GTC, None, NO_GTD_NOTE
    expires = decision_close + bar
    if now is not None:
        expires = max(expires, now + min_lead + GTD_MARGIN)
    return TimeInForce.GTD, expires, None


def overdue_at(decision_close: datetime, bar: timedelta) -> datetime:
    """og4: when a resting entry decided at `decision_close` becomes an orphan, if still resting: its next close plus
    OVERDUE_AFTER."""
    return decision_close + bar + OVERDUE_AFTER


def is_overdue(decision_close: datetime, bar: timedelta, now: datetime) -> bool:
    """og4: whether it is past overdue_at at `now` (at the instant itself, not yet)."""
    return now > overdue_at(decision_close, bar)
