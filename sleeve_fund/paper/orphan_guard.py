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


# --- the journal's half (og2-og5), on the DA's 0015 cancel command ----------------------------------------------------

OWNER_DOWN, OWNER_UP = "owner_down", "owner_up"
ORPHANED, ORPHAN_CANCEL, ORPHAN_CLOSED = "resting_entry_orphaned", "orphan_cancel", "orphan_fill_closed"
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
    from sleeve_fund.store import OPEN_ORDER_STATUSES, gate_decisions_t, gate_reservations_t, orders_t

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
    until it is cancelled or filled. Only this strategy is blocked."""
    asked = store.open_cancels(name)
    if asked:
        return (f"{len(asked)} resting entr{'y' if len(asked) == 1 else 'ies'} left from before are being cancelled; "
                "nothing new opens until the venue confirms")
    late = [r["order_id"] for r in gated_open(store, name) if is_overdue(r["bar_ts"], r["bar"], now)]
    if late:
        return (f"resting entr{'y' if len(late) == 1 else 'ies'} {', '.join(late)} still resting past the bar's close "
                f"+ {OVERDUE_AFTER.seconds // 60} min; nothing new opens until it is cancelled or filled")
    return None


def restart_reconcile(store, name: str, now: datetime, venue_open: list[str], send=None) -> list[str]:
    """og3: before a restarted process's first new entry, every gated entry the venue still has open for it is asked to
    cancel (journaled, idempotent); entries stay blocked (resting_block) until each cancel is confirmed. send(order_id)
    sends the cancel to the venue. Returns the ids asked."""
    owner_up(store, name, now)
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


def supervisor_pass(store, now: datetime, send=None, close=None) -> list[str]:
    """One supervisor pass of ORPHAN-GUARD at `now`; returns the order ids asked to cancel in it.

    - og2: a dead owner's resting entries are asked to cancel within DEAD_OWNER_CANCEL_WITHIN of it going down, the
      cancel journaled (Store.request_cancel, idempotent) and its reservation kept until the venue answers.
    - og4/og4b: an entry still resting past its bar's close + 2 min alerts once (its process may be dead).
    - og5: a dead owner's entry that filled anyway is closed once, through the exit path (close_orphans).
    send(order_id) sends a cancel to the venue; close(name, side, qty, order_id) sends the closing order."""
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
    close_orphans(store, now, close)
    settle_cancels(store, now)
    return asked


def settle_cancels(store, now: datetime) -> None:
    """Close what the venue's answers leave open (DA 0015): a "not found" with no fill in the quiet window after it is a
    cancel (Store.confirm_cancels releases it as 'cancel'); a cancel whose order filled is applied once its fills are
    booked and og5 has closed what filled after the owner went down ('filled': nothing more released)."""
    if hasattr(store, "confirm_cancels"):
        store.confirm_cancels(now)
    filled = {o["order_id"] for o in store.orders(limit=100_000) if float(o.get("filled_qty") or 0) > 0}
    for c in store.open_cancels():
        if c["order_id"] in filled and c["order_id"] not in {r["order_id"] for r in gated_open(store, c["sleeve"])}:
            try:
                store.apply_cancel(c["id"], "filled")
            except ValueError:  # 0015 without the "filled" outcome yet: the cancel stays open, its strategy blocked
                pass


def cancel_confirmed(store, order_id: str, *, not_found: bool = False, close=None,
                     at: datetime | None = None) -> str | None:
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
        close_orphans(store, utcnow_(), close)
    return got


def close_orphans(store, now: datetime, close=None) -> list[str]:
    """og5: an entry that filled after its strategy's process went down is closed once: one reduce-only market order
    on the opposite side for what filled, through the exit path (never gated), journaled and alerted as an orphan fill
    closed. Returns the closing order ids."""
    from sqlalchemy import select

    from sleeve_fund.store import fills_t, gate_decisions_t, gate_reservations_t, orders_t

    done = {(o.get("signal") or {}).get("orphan_of") for o in store.orders(limit=100_000) if o["intent"] == "exit"}
    out = []
    d, r = gate_decisions_t, gate_reservations_t
    with store.engine.connect() as c:
        rows = c.execute(select(orders_t.c.order_id, orders_t.c.sleeve, orders_t.c.side)
                         .join(d, d.c.order_id == orders_t.c.order_id).join(r, r.c.decision_id == d.c.id)
                         .where(orders_t.c.filled_qty > 0)).all()
    for oid, name, side in rows:
        down = down_since(store, name)
        if down is None or oid in done:
            continue
        with store.engine.connect() as c:
            late = c.execute(select(fills_t.c.qty).where(fills_t.c.order_id == oid, fills_t.c.ts >= down)).scalars().all()
        qty = sum(float(q) for q in late)
        if qty <= 0:
            continue
        exit_side = "SELL" if side == "BUY" else "BUY"
        cid = f"orphan-close-{oid}"
        store.record_order(name, order_id=cid, side=exit_side, qty=qty, intent="exit", order_type="MARKET",
                           reason=f"Orphan fill closed: {oid} filled {qty:g} after its strategy's process went down",
                           signal={"reduce_only": True, "orphan_of": oid}, ts=now)
        if close is not None:
            close(name, exit_side, qty, cid)
        store.event(name, "error", ORPHAN_CLOSED, f"Orphan fill closed: {oid} filled {qty:g} after its strategy's "
                    f"process went down; closed at market reduce-only ({cid})", ts=now)
        out.append(cid)
    return out


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
              "leverage": None, "atr_pct": None, "bar": int(decision_close.timestamp()) * 1_000_000_000}
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
