"""ORPHAN-GUARD (Advisor 8 Oct 06:15 and 06:20 UK; HoQA cells og1-og5): a resting entry never outlives its decision bar
unwatched, whatever happens to the process that sent it. The layers, first to last:

- og1 First line: sent good-till-date, expiring at its decision bar's next close, where the venue supports it.
- W2  Its own process cancels it at that close (LongFlatStrategy._cancel_gated_at_close).
- og2 The supervisor cancels a dead owner's resting entries within 60 s (the DA's cancel command, migration 0015).
- og3 A restart cancels leftovers before its first new entry.
- og4 An entry still resting at its bar's close + 2 min alerts once and blocks that strategy's entries (the supervisor's
      check, so a dead owner can't silence it).
- og5 An orphan that fills anyway is closed by one reduce-only market order through the exit path.
- og6 A cancel the venue answers "not found" is never released on silence (Advisor 13 Oct 17:00 UK): it holds the
      reservation and blocks that strategy until the venue's lookup (the order by client id and its fill history)
      settles it, and a fill found then is booked once, kept if before the bar's close, else closed as og5.

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


# --- the journal's half (og2-og5), on the DA's 0015 cancel command ----------------------------------------------------

OWNER_DOWN, OWNER_UP = "owner_down", "owner_up"
ORPHANED, ORPHAN_CANCEL, ORPHAN_CLOSED = "resting_entry_orphaned", "orphan_cancel", "orphan_fill_closed"
SANDBOX_ENDED, PREVIOUS_UNCONFIRMED = "sandbox_ended", "previous_process_unconfirmed"
DEAD_OWNER_CANCEL_WITHIN = timedelta(seconds=60)  # og2: a dead owner's resting entries are asked to cancel within this


def owner_down(store, name: str, at: datetime, how: str) -> None:
    """og2: the supervisor saw `name`'s process go down at `at` (how: crash, stop or refused). Its resting entries are
    then the supervisor's to cancel (supervisor_pass) until the process is back (owner_up)."""
    store.event(name, "warning", OWNER_DOWN, f"the strategy's process went down ({how}); the supervisor cancels its "
                "resting entries", ts=at)


def owner_up(store, name: str, at: datetime) -> None:
    store.event(name, "info", OWNER_UP, "the strategy's process is running again", ts=at)


def down_since(store, name: str) -> datetime | None:
    """When `name`'s process went down, if it hasn't come back since; else None."""
    last = store.last_event(name, (OWNER_DOWN, OWNER_UP))
    return last["ts"] if last is not None and last["kind"] == OWNER_DOWN else None


def gated_open(store, name: str | None = None) -> list[dict]:
    """Gated entries still open (sent, resting or part filled): order id, strategy, side, filled, and the bar their
    decision was on, with that strategy's bar length."""
    from sqlalchemy import select

    from sleeve_fund.data import spec_minutes
    from sleeve_fund.store import (
        OPEN_ORDER_STATUSES,
        gate_decisions_t,
        gate_reservations_t,
        orders_t,
    )

    o, d, r = orders_t, gate_decisions_t, gate_reservations_t
    q = (select(o.c.order_id, o.c.sleeve, o.c.side, o.c.qty, o.c.filled_qty, d.c.bar_ts)
         .join(d, d.c.order_id == o.c.order_id).join(r, r.c.decision_id == d.c.id)
         .where(o.c.status.in_(OPEN_ORDER_STATUSES)).order_by(o.c.id))
    if name is not None:
        q = q.where(o.c.sleeve == name)
    with store.engine.connect() as c:
        rows = [dict(row._mapping) for row in c.execute(q)]
    bars = {}
    for row in rows:
        if row["sleeve"] not in bars:
            bars[row["sleeve"]] = timedelta(minutes=spec_minutes(store.sleeve(row["sleeve"]).bar_spec))
        row["bar"], row["bar_ts"] = bars[row["sleeve"]], _aware(row["bar_ts"])
    return rows


def resting_block(store, name: str, now: datetime) -> str | None:
    """og3 and og4: why `name` may open nothing because of a resting entry, else None. A leftover the restart asked to
    cancel blocks until the venue confirms it; an entry still resting past its bar's close + 2 min (an orphan) blocks
    until it is cancelled or filled; a late fill blocks until it is closed flat (A2-DW4). Only this strategy is blocked:
    another Strategy on the instrument is held by the compliance tests on the real book."""
    asked = store.open_cancels(name)
    if asked:
        return (f"{len(asked)} resting entr{'y' if len(asked) == 1 else 'ies'} left from before are being cancelled; "
                "nothing new opens until the venue confirms")
    late = [r["order_id"] for r in gated_open(store, name) if is_overdue(r["bar_ts"], r["bar"], now)]
    if late:
        return (f"resting entr{'y' if len(late) == 1 else 'ies'} {', '.join(late)} still resting past the bar's close "
                f"+ {OVERDUE_AFTER.seconds // 60} min; nothing new opens until it is cancelled or filled")
    held = unflat(store, name)
    if held:  # A2-DW4 MUST 5: the venue's confirmed end is not enough while a late fill is still held
        return (f"late fill{'' if len(held) == 1 else 's'} on {', '.join(x['order_id'] for x in held)} after a cancel "
                "still being closed; nothing new opens until it is flat")
    return unprotected_block(store)


def unprotected_block(store) -> str | None:
    """HEAD3-DW line 6 (b): the fund-wide hold while any Strategy's late fill sits with its owner down (an episode the
    supervisor opened, unprotected_pass), else None. One last-event read per Strategy, as F-RX-2's block."""
    for s in store.sleeves():
        last = store.last_event(s.name, (UNPROTECTED, PROTECTED))
        if last is not None and last["kind"] == UNPROTECTED:
            return (f"{s.name} holds a late fill with its process down, so no stop guards it: nothing opens anywhere in "
                    "the fund until its process is back and the late fill is flat")
    return None


def unprotected_pass(store, now: datetime) -> None:
    """The supervisor's half of line 6 (b): an episode opens, alerted CRITICAL once, when a late fill isn't flat and its
    owner is down; it closes once the owner is back and every late fill of it is flat."""
    open_now = {x["sleeve"] for x in late_fills(store) if x["late"] > x["closed"]}
    for s in store.sleeves():
        name = s.name
        last = store.last_event(name, (UNPROTECTED, PROTECTED))
        is_open = last is not None and last["kind"] == UNPROTECTED
        down = down_since(store, name) is not None
        if not is_open and name in open_now and down:
            store.event(name, "error", UNPROTECTED, "CRITICAL: a late fill after cancel is held while the Strategy's "
                        "process is down, so no stop guards it; nothing opens anywhere in the fund until the process "
                        "is back and the late fill is closed flat", ts=now)
        elif is_open and name not in open_now and not down:
            store.event(name, "info", PROTECTED, "The late fill is closed flat and the Strategy's process is back: the "
                        "fund opens entries again", ts=now)


def in_process_kind(venue) -> str | None:
    """What kind of in-process venue `venue` is, by its type (Advisor 13 Oct 17:30 UK, condition 1): "sandbox" for the
    paper node's Sandbox execution client (its factory), "backtest" for a BacktestEngine's simulated exchange (a
    replay), else None: a venue outside the process (testnet, live), which only its own order query can answer."""
    from nautilus_trader.adapters.sandbox import SandboxExecutionClientFactory
    from nautilus_trader.backtest import BacktestEngine

    if isinstance(venue, SandboxExecutionClientFactory):
        return "sandbox"
    if isinstance(venue, BacktestEngine):
        return "backtest"
    return None


def restart_reconcile(store, name: str, now: datetime, venue_open: list[str], send=None,
                      in_process: str | None = None, sole_instance: bool | None = None) -> list[str]:
    """og3: before a restarted process's first new entry, every gated entry the venue still has open for it is asked to
    cancel (journaled, idempotent); entries stay blocked (resting_block) until each cancel is confirmed. send(order_id)
    sends the cancel to the venue. Returns the ids asked.

    in_process (in_process_kind): the venue ran inside the process before this one, so it ended with that process
    (Advisor 13 Oct 17:30 UK and 17:35 addendum): its exit, journalled as owner_down, is the venue's confirmed end and
    the journal's fills its whole fill history. Every gated entry it left (open, or with its cancel open) is settled
    now: what filled is kept, or closed if it filled after its bar's close or its owner went down (og5), and the rest
    released as 'cancel'. Outside the process (None: testnet, live) a cancel the venue doesn't find stays open, and
    the strategy blocked, until the venue's own lookup settles it (venue_lookup).

    sole_instance: this process holds the strategy's single-instance lock (instance_lock). With no owner_down journalled
    (the supervisor restarted too), only that proves the process before this one, and its sandbox, are gone (Advisor
    13 Oct 17:55 UK); without it nothing is released, each leftover is asked to cancel (so entries stay blocked) and
    it alerts once."""
    if in_process is not None:
        left = {r["order_id"] for r in gated_open(store, name)} | {c["order_id"] for c in store.open_cancels(name)}
        left -= set(venue_open)
        if left:
            down = store.last_event(name, (OWNER_DOWN, OWNER_UP))
            if (down is None or down["kind"] != OWNER_DOWN) and sole_instance is not True:
                # The supervisor didn't see it go, and nothing proves it went: it may still run, and its sandbox fill.
                asking = {c["order_id"] for c in store.open_cancels(name)}
                for oid in sorted(left - asking):
                    store.request_cancel(name, oid, "the process before this one may still be running")
                told = store.last_event(name, (PREVIOUS_UNCONFIRMED, OWNER_UP))
                if told is None or told["kind"] != PREVIOUS_UNCONFIRMED:  # once while it lasts
                    store.event(name, "error", PREVIOUS_UNCONFIRMED, f"{len(left)} resting entr"
                                f"{'y' if len(left) == 1 else 'ies'} left from before can't be released: the process "
                                "before this one isn't confirmed gone (its lock is held or unknown); entries stay "
                                "blocked until a person checks", ts=now)
                return []
            if down is None or down["kind"] != OWNER_DOWN:  # the supervisor didn't see it go (it restarted too)
                owner_down(store, name, now, "restart: the process before this one is gone (its lock is free)")
                down = store.last_event(name, (OWNER_DOWN,))
            for oid in sorted(left):
                sandbox_ended(store, name, oid, down, now)
    owner_up(store, name, now)
    if in_process is None:  # a leftover the venue no longer lists may have filled: unknown until its lookup (og6)
        asking = {c["order_id"] for c in store.open_cancels(name)}
        for r in gated_open(store, name):
            if r["order_id"] not in venue_open and r["order_id"] not in asking:
                store.request_cancel(name, r["order_id"], "not listed by the venue after the restart: its state is "
                                     "unknown until the venue's lookup")
    for c in store.open_cancels(name):  # asked while it was down, and the venue no longer has it: "not found"
        if c["order_id"] not in venue_open:
            cancel_confirmed(store, c["order_id"], not_found=True, at=now)
    asked = []
    mine = {r["order_id"] for r in gated_open(store, name)}
    for oid in venue_open:
        if oid not in mine:
            continue
        store.request_cancel(name, oid, "left resting from before the restart")
        if send is not None:
            send(oid)
        asked.append(oid)
    return asked


def sandbox_ended(store, name: str, order_id: str, down: dict, at: datetime) -> None:
    """One gated entry ended with its in-process venue (restart_reconcile), journalled with the owner_down event and
    the journal's fill total (Advisor 13 Oct 17:30 UK, condition 3): its cancel, if open, applied as "cancelled", else
    the order ended; either way what no fill used of its reservation is released as 'cancel'."""
    why = (f"sandbox ended with owner_down (event {down['id']}); journal fills {_journal_filled(store, order_id):g}: "
           f"{order_id} ended with the process that sent it")
    store.event(name, "info", SANDBOX_ENDED, why, ts=at)
    open_ = [c for c in store.open_cancels(name) if c["order_id"] == order_id]
    if open_:
        store.apply_cancel(open_[0]["id"], "cancelled", at=at)
    else:
        store.update_order(order_id, status="canceled", message=why)


def sandbox_gone(store, name: str, now: datetime) -> list[str]:
    """CR MUST 2 (HoE ruling, CR option 1, 15 Oct): the caller has proved `name`'s process gone (no process of ours
    alive and its single-instance lock taken) on the simulated paper client, whose venue lived in that process. Every
    gated entry it left, open or with its cancel open, ended with it: settled as sandbox_ended, as a restart's
    reconcile does, so a cancel nothing can ever answer doesn't wedge the PM's Reset. Returns the ids settled."""
    left = sorted({r["order_id"] for r in gated_open(store, name)} | {c["order_id"] for c in store.open_cancels(name)})
    if not left:
        return []
    down = store.last_event(name, (OWNER_DOWN, OWNER_UP))
    if down is None or down["kind"] != OWNER_DOWN:
        owner_down(store, name, now, "its process is gone (its lock is free)")
        down = store.last_event(name, (OWNER_DOWN,))
    for oid in left:
        sandbox_ended(store, name, oid, down, now)
    return left


def supervisor_pass(store, now: datetime, send=None) -> list[str]:
    """One supervisor pass of ORPHAN-GUARD at `now`; returns the order ids asked to cancel in it.

    - og2: a dead owner's resting entries are asked to cancel within DEAD_OWNER_CANCEL_WITHIN of it going down, the
      cancel journaled (Store.request_cancel, idempotent) and its reservation kept until the venue answers.
    - og4/og4b: an entry still resting past its bar's close + 2 min alerts once (its process may be dead).
    - og5: a late fill's close is journaled and alerted once (close_orphans); its owner sends it (send_pending).
    send(order_id) sends a cancel to the venue. A late fill's close is journaled and alerted here, never sent: its
    strategy's own process sends it (send_pending), so there is exactly one sender."""
    store.__dict__["_supervisor_now"] = now  # the venue's answers between passes are stamped on this clock
    asked = []
    told = {e["message"].split(" ", 1)[0] for e in _events_of(store, ORPHANED)}
    for r in gated_open(store):
        name, oid = r["sleeve"], r["order_id"]
        if is_overdue(r["bar_ts"], r["bar"], now) and oid not in told:
            store.event(name, "error", ORPHANED, f"{oid} still resting {OVERDUE_AFTER.seconds // 60} min past its bar's "
                        "close: its entries are blocked until it is cancelled or filled", ts=now)
        down = down_since(store, name)
        if down is None or now < down:
            continue
        opened = {c["order_id"] for c in store.open_cancels(name)}
        if oid not in opened:
            store.request_cancel(name, oid, "its strategy's process is down")
            store.event(name, "warning", ORPHAN_CANCEL, f"{oid}: cancel asked, its strategy's process is down", ts=now)
            if send is not None:
                send(oid)
            asked.append(oid)
    close_orphans(store, now)  # journaled and alerted; the owner sends it (send_pending)
    unprotected_pass(store, now)
    realert_held(store, now)
    settle_cancels(store, now)
    return asked


def settle_cancels(store, now: datetime) -> None:
    """Close the cancels whose order the journal shows ended in fills (DA 0015 "filled"), once og5 has closed what
    filled late. Nothing is released on silence (Advisor 13 Oct 17:00 UK): a cancel answered "not found" stays open,
    and its strategy blocked, until the venue's lookup settles it (venue_lookup)."""
    filled = {o["order_id"] for o in store.orders(limit=100_000) if float(o.get("filled_qty") or 0) > 0}
    for c in store.open_cancels():
        if c["order_id"] in filled and c["order_id"] not in {r["order_id"] for r in gated_open(store, c["sleeve"])}:
            store.apply_cancel(c["id"], "filled", at=now)


def bar_close_cancel(store, name: str, at: datetime) -> list[str]:
    """W2's cancel at the bar close, journalled (og6): every gated entry of `name` still resting that was decided
    before `at` gets a cancel command (Store.request_cancel, idempotent) before its process sends the cancel, so the
    venue's answer settles it like any other (answer_cancel, venue_lookup). Returns the ids to send."""
    from sleeve_fund.paper.portfolio_gate import resting_entries

    due = resting_entries(store, name, at)
    for oid in due:
        store.request_cancel(name, oid, "cancelled at the next bar close: an entry never rests past the bar after it "
                             "was decided", actor="strategy")
    return due


def answer_cancel(store, order_id: str, answer: str, at: datetime) -> str | None:
    """The venue's answer to a cancel, received at `at`: 'canceled' (or 'expired'), 'not_found' or 'filled'. A
    "not found" releases nothing; a "filled" whose fills aren't journalled yet waits for the lookup to book them."""
    if answer in ("canceled", "expired"):
        return cancel_confirmed(store, order_id, at=at)
    if answer == "not_found":
        return cancel_confirmed(store, order_id, not_found=True, at=at)
    if answer != "filled":
        raise ValueError(f"a venue's answer to a cancel is canceled, expired, not_found or filled, not {answer!r}")
    open_ = [c for c in store.open_cancels() if c["order_id"] == order_id]
    if not open_ or not _journal_filled(store, order_id):
        return None
    close_orphans(store, at)
    return store.apply_cancel(open_[0]["id"], "filled", at=at)


def venue_lookup(store, order_id: str, *, terminal: str | None, fills: list, at: datetime) -> str | None:
    """og6: the venue's lookup of `order_id` by client id plus its fill history settles its open cancel. `fills` are
    (qty, ts, price, fee[, trade_id]) as the venue reports them. Whatever of them the journal doesn't hold is booked
    (the journal's own fills counted first, so nothing books twice); then a terminal 'canceled' or 'expired' releases
    the unfilled remainder as 'cancel', 'filled' applies it, and None (the venue doesn't know it yet) leaves it open
    and its strategy blocked. A fill after its bar's close, or after its owner went down, is closed (og5)."""
    o = next((o for o in store.orders(limit=100_000) if o["order_id"] == order_id), None)
    if o is None:
        raise ValueError(f"no order {order_id!r}")
    have = _journal_filled(store, order_id)
    seen = 0.0
    for i, f in enumerate(sorted(fills, key=lambda f: f[1])):
        qty, ts, price, fee = float(f[0]), f[1], float(f[2]), float(f[3])
        new = min(qty, seen + qty - have) if seen + qty > have else 0.0
        seen += qty
        if new > 1e-12:
            store.book_fill(o["sleeve"], side=o["side"], qty=new, price=price, fee=fee * new / qty, order_id=order_id,
                            trade_id=f[4] if len(f) > 4 and f[4] else f"{order_id}-lookup-{i}", ts=ts)
    if terminal is None:
        return None
    if terminal in ("canceled", "expired"):
        got = cancel_confirmed(store, order_id, not_found=False, at=at)
        close_orphans(store, at)
        return got
    if terminal == "filled":
        return answer_cancel(store, order_id, "filled", at)
    raise ValueError(f"a venue's terminal state is canceled, expired or filled, not {terminal!r}")


def order_ended(store, order_id: str, status: str, at: datetime) -> None:
    """The owner's process saw its venue end an order (canceled or expired): the cancel it journalled, if any, is
    settled with it (og6), so a confirmed cancel never leaves its strategy blocked."""
    if any(c["order_id"] == order_id for c in store.open_cancels()):
        cancel_confirmed(store, order_id, at=at)


def _journal_filled(store, order_id: str) -> float:
    from sqlalchemy import func, select

    from sleeve_fund.store import fills_t

    with store.engine.connect() as c:
        return float(c.execute(select(func.coalesce(func.sum(fills_t.c.qty), 0)).where(fills_t.c.order_id == order_id))
                     .scalar())


def cancel_confirmed(store, order_id: str, *, not_found: bool = False, at: datetime | None = None) -> str | None:
    """The venue's answer to a cancel the journal holds open: Store.apply_cancel, stamped `at` (the answering
    process's clock; by default the supervisor's last pass). A "not found" for an order that the journal shows filled
    is a fill (Advisor 06:20 UK): nothing is released as 'cancel', and og5 closes it."""
    open_ = [c for c in store.open_cancels() if c["order_id"] == order_id]
    if not open_:  # its own process's cancel (the bar-close cancel): the order's end settles its reservation (W2)
        store.update_order(order_id, status="expired" if not_found else "canceled")
        return "applied"
    if at is None:
        at = store.__dict__.get("_supervisor_now")  # the supervisor's clock, which confirm_cancels' quiet window runs on
    got = store.apply_cancel(open_[0]["id"], "not_found" if not_found else "cancelled", at=at)
    if got in ("filled", "partly_filled"):  # a fill, not a cancel: og5 closes what filled while its owner was down
        close_orphans(store, utcnow_())
    return got


CLOSE_PREFIX = "orphan-close-"  # a late fill's close: orphan-close-<entry id>, then -2, -3 for later fills of it
LATE_HELD = "late_fill_held"
# Advisor HEAD3-DW line 6 (b): a late fill whose owner is down has no S3 (the owner places it; paper's venue is in that
# process): CRITICAL at once, and the whole fund opens nothing until the owner is back and it is flat.
UNPROTECTED, PROTECTED = "late_fill_unprotected", "late_fill_protected"  # a late fill whose close can't go yet (its owner is down): alerted, re-alerted
REALERT_EVERY = timedelta(minutes=30)  # A2-DW4 follow-up: while the block holds, as F-RX-2 re-alerts
ENDED = ("filled", "canceled", "expired", "rejected", "denied")


def _down_at(store, name: str):
    """A test of whether `name`'s process was down at a time: the owner_down/owner_up history, so a fill made while it
    was down stays late after it is back (owner_up), and the close owed for it is never forgotten."""
    from sqlalchemy import select

    from sleeve_fund.store import events_t

    with store.engine.connect() as c:
        marks = [(_aware(ts), kind) for ts, kind in c.execute(
            select(events_t.c.ts, events_t.c.kind).where(events_t.c.sleeve == name,
                                                         events_t.c.kind.in_((OWNER_DOWN, OWNER_UP)))
            .order_by(events_t.c.ts, events_t.c.id))]

    def down(ts: datetime) -> bool:
        state = False
        for at, kind in marks:
            if at > ts:
                break
            state = kind == OWNER_DOWN
        return state

    return down, any(k == OWNER_DOWN for _, k in marks)


def late_fills(store, name: str | None = None) -> list[dict]:
    """og5 and A2-DW4: every gated entry with a late fill, as order id, strategy, side, `late` (what filled while its
    strategy's process was down, or after its bar's close: the bar-close cancel raced it), `sent` (closes still out,
    plus what ended ones filled: a close that ended unfilled, with its process or refused, no longer counts) and
    `closed` (what its closes filled). Exact: fills are unique by venue fill id (fills_sleeve_order_trade), so a
    duplicate fill event counts once. Only gated entries: a reduce-only exit is never gated, so its late fill is a
    valid exit, booked as any fill. What filled before both is the strategy's to keep. Paper and replay only: a
    backtest has no resting-cancel race and never journals a gated entry, so this finds nothing there."""
    from decimal import Decimal

    from sqlalchemy import select

    from sleeve_fund.data import spec_minutes
    from sleeve_fund.store import (
        fills_t,
        gate_decisions_t,
        gate_reservations_t,
        orders_t,
    )

    q = select(orders_t).where(orders_t.c.order_id.like(f"{CLOSE_PREFIX}%"))
    if name is not None:
        q = q.where(orders_t.c.sleeve == name)
    closes: dict[str, list[dict]] = {}
    with store.engine.connect() as c:
        for row in c.execute(q.order_by(orders_t.c.id)):
            o = dict(row._mapping)
            of = (o.get("signal") or {}).get("orphan_of")
            if of:
                closes.setdefault(of, []).append(o)
    d, r = gate_decisions_t, gate_reservations_t
    q = (select(orders_t.c.order_id, orders_t.c.sleeve, orders_t.c.side, orders_t.c.order_type, d.c.bar_ts)
         .join(d, d.c.order_id == orders_t.c.order_id).join(r, r.c.decision_id == d.c.id)
         .where(orders_t.c.filled_qty > 0))
    if name is not None:
        q = q.where(orders_t.c.sleeve == name)
    with store.engine.connect() as c:
        rows = c.execute(q).all()
    out, downs = [], {}
    for oid, sleeve, side, kind, bar_ts in rows:
        if sleeve not in downs:
            downs[sleeve] = _down_at(store, sleeve)
        was_down, ever_down = downs[sleeve]
        end = None
        if kind != "MARKET" and bar_ts is not None:  # a resting entry: its bar-close cancel races its fills
            end = _aware(bar_ts) + timedelta(minutes=spec_minutes(store.sleeve(sleeve).bar_spec))
        mine = closes.get(oid, [])
        if not ever_down and end is None and not mine:
            continue
        with store.engine.connect() as c:
            fills = c.execute(select(fills_t.c.qty, fills_t.c.ts).where(fills_t.c.order_id == oid)).all()
        late, after = Decimal(0), None
        for qty, ts in fills:
            ts = _aware(ts)
            if ever_down and was_down(ts):
                late, after = late + Decimal(str(qty)), "while its strategy's process was down"
            elif end is not None and ts > end:
                late, after = late + Decimal(str(qty)), after or "after its bar's close"
        if late <= 0 and not mine:
            continue
        filled = [Decimal(str(o["filled_qty"] or 0)) for o in mine]
        out.append({"order_id": oid, "sleeve": sleeve, "side": side, "late": late, "after": after,
                    "sent": sum((f if o["status"] in ENDED else Decimal(str(o["qty"])) for o, f in zip(mine, filled)),
                                Decimal(0)),
                    "closed": sum(filled, Decimal(0)), "closes": mine})
    return out


def unflat(store, name: str) -> list[dict]:
    """A2-DW4 MUST 5: the late fills of `name` not yet closed flat (closes still out, or not yet sent)."""
    return [x for x in late_fills(store, name) if x["late"] > x["closed"]]


def close_orphans(store, now: datetime, submit=None, name: str | None = None) -> list[str]:
    """og5 and A2-DW4: a late fill (late_fills) is closed by one reduce-only market order on the opposite side for
    exactly what filled late less what closes already have out, through the exit path (never gated: brakes, halts and
    the entry block never stop an exit), journaled and alerted as a late fill after cancel, closed. Reduce-only, so it
    never overshoots into the other side; a pass with nothing new journals nothing. Each close is journaled once under
    its own id (orphan-close-<entry>[-n], unique in orders), so two writers racing write one row.

    Exactly one sender (HoE, 15 Oct): only the Strategy's own process sends a close to its venue
    (submit(name, side, qty, close_id, reason)); the venue is in that process on paper. The supervisor passes no
    submit: it journals the close owed and alerts, and the owner sends what it finds journaled and unsent
    (send_pending). Returns the close ids journaled now."""
    from sqlalchemy.exc import IntegrityError

    out = []
    for x in late_fills(store, name):
        qty = x["late"] - x["sent"]
        if qty <= 0:
            continue
        oid, sleeve = x["order_id"], x["sleeve"]
        exit_side = "SELL" if x["side"] == "BUY" else "BUY"
        n = len(x["closes"])
        cid = f"{CLOSE_PREFIX}{oid}" + (f"-{n + 1}" if n else "")
        what = f"Late fill after cancel, closed: {oid} filled {qty:f} {x['after']}"
        try:
            store.record_order(sleeve, order_id=cid, side=exit_side, qty=float(qty), intent="exit",
                               order_type="MARKET", reason=what, signal={"reduce_only": True, "orphan_of": oid}, ts=now)
        except IntegrityError:  # the other writer journaled it first: one row, one close
            continue
        if submit is not None:
            submit(sleeve, exit_side, qty, cid, what)
        held = "" if submit is not None else (
            "; its strategy's process sends it, and if that process is down it can't go until it restarts: nothing new "
            "opens on this Strategy until it is flat (fails safe)")
        store.event(sleeve, "error", ORPHAN_CLOSED, f"Orphan fill closed. {what}; closed at market reduce-only ({cid})"
                    f"{held}", ts=now)
        out.append(cid)
    return out


def send_pending(store, name: str, now: datetime, submit, sent: set[str]) -> list[str]:
    """The owner's half of exactly one sender (HoE, 15 Oct): its late fills' closes are journaled (close_orphans, by
    this process or the supervisor) and then sent here, each once: `sent` holds the close ids this process has already
    sent (its own cache), so a pass sends nothing twice. A close the process before this one sent ended with it (an
    in-process venue's restart ends its open orders), so it no longer counts and a fresh one is journaled for what is
    still open: a crash loop never sends two for one quantity. Returns the close ids sent now."""
    from decimal import Decimal

    close_orphans(store, now, name=name)
    out = []
    for x in late_fills(store, name):
        for o in x["closes"]:
            if o["status"] in ENDED or o["order_id"] in sent:
                continue
            submit(name, o["side"], Decimal(str(o["qty"])), o["order_id"], o["reason"])
            out.append(o["order_id"])
    return out


def unsent(store, name: str, sent: set[str]) -> bool:
    """Whether a close journaled for `name` (by the supervisor, say) is still out and not one this process sent: one
    indexed read, cheap enough for every tick."""
    from sqlalchemy import select

    from sleeve_fund.store import OPEN_ORDER_STATUSES, orders_t

    with store.engine.connect() as c:
        ids = c.execute(select(orders_t.c.order_id).where(orders_t.c.sleeve == name,
                                                          orders_t.c.order_id.like(f"{CLOSE_PREFIX}%"),
                                                          orders_t.c.status.in_(OPEN_ORDER_STATUSES))).scalars().all()
    return any(i not in sent for i in ids)


def realert_held(store, now: datetime) -> None:
    """A2-DW4 follow-up: while a Strategy is blocked by a late fill not yet flat, it is re-alerted every 30 minutes,
    as F-RX-2 does."""
    for x in late_fills(store):
        if x["late"] <= x["closed"]:
            continue
        name = x["sleeve"]
        last = store.last_event(name, (LATE_HELD, ORPHAN_CLOSED))
        if last is not None and now - _aware(last["ts"]) < REALERT_EVERY:
            continue
        store.event(name, "error", LATE_HELD, f"Still blocked: {x['order_id']} filled {x['late'] - x['closed']:f} "
                    f"{x['after']} and is not yet closed; nothing new opens on this Strategy until it is flat",
                    ts=now)


def qa_resting_entry(store, name: str, order_id: str, decision_close: datetime, side: str = "BUY") -> int:
    """QA's set-up (HoQA cells): an acked resting entry for `name` decided at `decision_close`, holding the gate's
    reservation, through the paper gate and the order's own write. Returns the reservation id."""
    from decimal import Decimal

    from sleeve_fund.gate_ledger import DbLedger
    from sleeve_fund.paper.portfolio_gate import paper_gate
    from sleeve_fund.portfolio.gate import PortfolioState

    led = DbLedger(store, lambda: None)
    st = led.state()
    marked = decision_close + timedelta(days=1)  # the fund marked through the cell: no stale-book block in its way
    if st.mark_ts is None or st.mark_ts < marked:
        eq = st.equity or Decimal(20_000)
        led.set_state(PortfolioState(mark_ts=marked, equity=eq, hwm=st.hwm or eq, day=decision_close.date(),
                                     day_start_equity=st.day_start_equity or eq))
    s = 1 if side == "BUY" else -1
    intent = {"side": s, "qty": Decimal("0.1"), "close": Decimal(60_000), "price": Decimal(60_000), "stop_frac": 0.02,
              "what": "entry", "step": Decimal("0.001"), "min_qty": Decimal("0.001"), "instrument": "BTC",
              "leverage": None, "atr_pct": None, "taker": Decimal(0),  # a fee-less set-up, as PE1's a256490d
              "bar": int(decision_close.timestamp()) * 1_000_000_000}
    g = paper_gate(store)(name, intent, int(decision_close.timestamp()) * 1_000_000_000 + 1_000_000)
    if g.checked.reservation is None:
        raise RuntimeError(f"set-up: the gate didn't approve {name}'s entry: {g.decision}")
    store.record_order(name, order_id=order_id, side=side, qty=0.1, intent="entry", reason="QA resting entry",
                       order_type="LIMIT", ts=decision_close, reservation=g.checked.reservation)
    store.update_order(order_id, status="accepted")
    return g.checked.reservation


def _events_of(store, kind: str) -> list[dict]:
    from sqlalchemy import select

    from sleeve_fund.store import events_t

    with store.engine.connect() as c:
        return [dict(r._mapping) for r in c.execute(select(events_t).where(events_t.c.kind == kind))]


def utcnow_() -> datetime:
    from sleeve_fund.store import utcnow

    return utcnow()


def _aware(ts: datetime) -> datetime:
    from datetime import timezone

    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
