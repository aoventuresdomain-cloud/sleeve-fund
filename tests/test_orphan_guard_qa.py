"""QA Done-when cells for ORPHAN-GUARD (Head of QA, 8 Oct). Source: Advisor 06:15 UK 8 Oct, advisor-rulings.md, a MUST
before any Strategy that uses resting entries starts on paper. PE2 builds it after W2.

Every cell is a strict xfail on main. The ADAPTERS name what the build must provide, and PE2 renames them to the real
entry points. The assertions change only through the HoQA. Run from a worktree's tests/ beside test_gate_ledger.py,
on SQLite and on PG.

  og1  First line. A resting entry is sent good-till-date, expiring at its decision bar's next close, where the venue
       supports it. Where it doesn't, the order is still sent, and the run journals that only the backstops cover it.
  og2  Backstop. Within 60 s of the owner going down (crash, stop, refused start), something alive (the supervisor's
       'cancel' command) cancels each unfilled resting entry. The cancel is journalled. The reservation is kept until
       the cancel is confirmed, then released as 'cancel'. A "not found" alone releases nothing (Advisor 17:00 UK
       13 Oct): only the venue's lookup by client id plus its fill history, showing a zero-fill terminal state, does.
  og2c A "not found" plus a journalled fill for that order, in either order, converts the reservation once (released
       as 'fill', never 'cancel') and triggers og5 (Advisor 06:20 UK).
  og3  Restart reconcile cancels any leftover resting entry before the first new entry. Entries stay blocked until
       every leftover's cancel is confirmed.
  og4  An acked entry still resting at its bar close + 2 min raises one alert and blocks that Strategy's entries
       (CHOKE names it) until it is cleared. Before + 2 min there is no alert. Other Strategies are not blocked.
  og4b The + 2 min check runs in the supervisor and fires with the owner dead (Advisor 06:20 UK).
  og6  Advisor 17:00 UK 13 Oct, the bar-close cancel. A "not found" holds the reservation and blocks that Strategy's
       new entries (others not) until the venue lookup resolves it; silence never releases, and at bar close + 2 min
       it alerts once. A fill found by the lookup is booked and treated as a race. In a race the fill wins: filled at
       or before the bar close is kept; filled after it is an orphan closed at market (og5). The cancel record never
       stays open or blocks. A partial fill keeps the filled part under the same rule and releases only the confirmed
       unfilled remainder, as 'cancel'.
  og5  An orphan that fills anyway, long or short, is closed by one reduce-only market order on the opposite side,
       sent by the supervisor through the exit path (never gated). It is journalled as "orphan fill closed" and alerted,
       once.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from test_gate_ledger import _drop_schemas, _store, _strategies  # noqa: F401

from sleeve_fund.paper.runtime import entry_blocked

BAR = timedelta(hours=1)
CLOSE = datetime(
    2026, 10, 8, 9, 0, tzinfo=UTC
)  # the decision bar's close; the order rests into the next bar


def cell(why: str):
    return pytest.mark.xfail(strict=True, reason=f"ORPHAN-GUARD: {why}")


# === ADAPTERS (PE2: rename to what you build; nothing below this block changes) =====================================


def resting_entry_order(
    *, gtd_supported: bool, decision_close: datetime, bar: timedelta
):
    """The resting (limit / post-only) entry order the engine builds for a decision at `decision_close`, on a venue
    with or without good-till-date. Returns (time_in_force, expire_time or None, journal note or None)."""
    from sleeve_fund.paper import orphan_guard

    return orphan_guard.resting_entry_order(
        gtd_supported=gtd_supported, decision_close=decision_close, bar=bar
    )


def resting_entry_order_at(
    *, decision_close: datetime, bar: timedelta, now: datetime, min_lead: timedelta
):
    """As resting_entry_order on a venue with good-till-date whose minimum expiry lead is `min_lead`, sent at `now`."""
    from sleeve_fund.paper import orphan_guard

    return orphan_guard.resting_entry_order(
        gtd_supported=True, decision_close=decision_close, bar=bar, now=now, min_lead=min_lead
    )


def owner_down(store, name: str, at: datetime, how: str) -> None:
    """The supervisor sees `name`'s process go down at `at` (how: 'crash', 'stop' or 'refused')."""
    from sleeve_fund.paper import orphan_guard

    orphan_guard.owner_down(store, name, at, how)


def supervisor_pass(store, now: datetime) -> list[str]:
    """One supervisor pass at `now`. Returns the order ids it asked the venue to cancel in this pass."""
    from sleeve_fund.paper import orphan_guard

    return orphan_guard.supervisor_pass(store, now)


def cancel_confirmed(store, order_id: str, *, not_found: bool = False) -> None:
    """The venue's answer to a cancel: cancelled, or no such order."""
    from sleeve_fund.paper import orphan_guard

    orphan_guard.cancel_confirmed(store, order_id, not_found=not_found)


def bar_close_cancel(store, name: str, at: datetime) -> list[str]:
    """The owner (alive) cancels its unfilled resting entries at the resting bar's close. Returns the ids sent."""
    from sleeve_fund.paper import orphan_guard

    return orphan_guard.bar_close_cancel(store, name, at)


def answer_cancel(store, order_id: str, answer: str, at: datetime) -> None:
    """The venue's answer to a cancel, received at `at`: 'canceled', 'not_found' or 'filled'."""
    from sleeve_fund.paper import orphan_guard

    orphan_guard.answer_cancel(store, order_id, answer, at)


def venue_lookup(
    store, order_id: str, *, terminal: str | None, fills: list[tuple[float, datetime]], at: datetime
) -> None:
    """The venue's answer to the reconcile lookup of `order_id` by client id plus its fill history: the terminal
    state ('canceled', 'expired', 'filled', or None while the order is unknown) and every fill (qty, fill time). The
    build books any fill it has not journalled."""
    from sleeve_fund.paper import orphan_guard

    orphan_guard.venue_lookup(store, order_id, terminal=terminal, at=at,
                              fills=[(q, t, 60000.0, 0.0) for q, t in fills])  # the cells' price, no fee


def restart_reconcile(
    store, name: str, now: datetime, venue_open: list[str]
) -> list[str]:
    """The owner's restart reconciliation, with the venue's open orders for it. Returns the ids it cancelled."""
    from sleeve_fund.paper import orphan_guard

    return orphan_guard.restart_reconcile(store, name, now, venue_open)


def resting_entry(
    store, name: str, order_id: str, decision_close: datetime, side: str = "BUY"
) -> int:
    """An acked resting entry for `name` (a long's BUY or a short's SELL), decided at `decision_close`, holding its
    reservation. Returns that id."""
    from sleeve_fund.paper import orphan_guard

    return orphan_guard.qa_resting_entry(
        store, name, order_id, decision_close, side=side
    )


def orphan_fills(
    store, name: str, order_id: str, qty: float, at: datetime, side: str = "BUY"
) -> None:
    """The venue fills an orphaned resting entry."""
    store.book_fill(
        name,
        side=side,
        qty=qty,
        price=60000.0,
        fee=0.1,
        order_id=order_id,
        trade_id="orphan-1",
        ts=at,
    )


def is_reduce_only(store, order_id: str) -> bool:
    """Whether the close was sent reduce-only, as journalled (default: the order row's signal)."""
    o = next(o for o in store.orders() if o["order_id"] == order_id)
    return bool((o.get("signal") or {}).get("reduce_only"))


def reservation_reason(store, rid: int) -> str | None:
    from sqlalchemy import select

    from sleeve_fund.store import gate_reservations_t

    with store.engine.connect() as c:
        return c.execute(
            select(gate_reservations_t.c.release_reason).where(
                gate_reservations_t.c.decision_id == rid
            )
        ).scalar()


# =====================================================================================================================


@pytest.fixture
def store():
    s = _store()
    _strategies(s)
    return s


def _events(store, kind, name=None):
    from sqlalchemy import select

    from sleeve_fund.store import events_t

    q = select(events_t).where(events_t.c.kind == kind)
    with store.engine.connect() as c:
        return [r for r in c.execute(q) if name is None or r.sleeve == name]


def test_og1_a_resting_entry_is_sent_good_till_its_decision_bars_next_close():
    tif, expires, note = resting_entry_order(
        gtd_supported=True, decision_close=CLOSE, bar=BAR
    )
    assert str(tif).upper().endswith("GTD") and expires == CLOSE + BAR and note is None
    tif, expires, note = resting_entry_order(
        gtd_supported=False, decision_close=CLOSE, bar=BAR
    )
    assert not str(tif).upper().endswith("GTD") and expires is None
    assert note and "backstop" in note.lower()


LEAD = timedelta(seconds=600)  # an assumed venue minimum; the cells hold for any lead


def test_og1b_on_a_short_bar_the_expiry_is_clamped_to_now_plus_the_venue_minimum_plus_30_s():
    """Advisor 13 Oct 15:30 UK: expiry = max(next close, now + venue minimum lead + 30 s); never refused for it."""
    now = CLOSE + timedelta(seconds=2)
    tif, expires, note = resting_entry_order_at(
        decision_close=CLOSE, bar=timedelta(minutes=1), now=now, min_lead=LEAD
    )
    assert str(tif).upper().endswith("GTD") and expires == now + LEAD + timedelta(seconds=30)


def test_og1b_on_a_4h_bar_early_in_the_bar_the_expiry_is_the_next_close():
    now = CLOSE + timedelta(seconds=5)
    tif, expires, note = resting_entry_order_at(
        decision_close=CLOSE, bar=timedelta(hours=4), now=now, min_lead=LEAD
    )
    assert str(tif).upper().endswith("GTD") and expires == CLOSE + timedelta(hours=4) and note is None


@pytest.mark.parametrize("how", ["crash", "stop", "refused"])
def test_og2_within_60_s_of_the_owner_going_down_its_resting_entries_are_cancelled_by_the_supervisor(
    store, how
):
    rid = resting_entry(store, "a", "o-rest", CLOSE)
    other = resting_entry(
        store, "b", "o-b", CLOSE
    )  # another Strategy, alive: untouched
    down = CLOSE + timedelta(minutes=10)
    owner_down(store, "a", down, how)
    asked = []
    for s in range(0, 61, 5):
        asked += supervisor_pass(store, down + timedelta(seconds=s))
    assert asked.count("o-rest") >= 1 and "o-b" not in asked
    assert reservation_reason(store, rid) is None  # kept until the venue confirms
    assert _events(store, "orphan_cancel", "a")  # journalled
    cancel_confirmed(store, "o-rest")
    assert (
        reservation_reason(store, rid) == "cancel"
        and reservation_reason(store, other) is None
    )


def test_og2b_not_found_releases_only_on_a_zero_fill_lookup_and_an_unanswered_cancel_never_releases_blind(
    store,
):
    gone = resting_entry(store, "a", "o-gone", CLOSE)
    silent = resting_entry(store, "a", "o-silent", CLOSE)
    down = CLOSE + timedelta(minutes=10)
    owner_down(store, "a", down, "crash")
    for s in range(0, 600, 5):
        supervisor_pass(store, down + timedelta(seconds=s))
        if s == 30:
            cancel_confirmed(store, "o-gone", not_found=True)
        if s == 300:
            assert reservation_reason(store, gone) is None  # "not found" alone is no proof (Advisor 17:00 UK 13 Oct)
            venue_lookup(store, "o-gone", terminal="canceled", fills=[], at=down + timedelta(seconds=s))
    assert reservation_reason(store, gone) == "cancel"  # released only on the venue's zero-fill terminal state
    assert (
        reservation_reason(store, silent) is None
    )  # ten minutes unanswered: still held


@pytest.mark.parametrize("fill_first", [True, False])
def test_og2c_not_found_with_a_journalled_fill_converts_once_and_closes_the_orphan_never_releases(
    store, fill_first
):
    """Advisor 06:20 UK: a "not found" for an order that the journal shows filled is a fill. The reservation turns
    into the holding once (released as 'fill', never 'cancel'), and og5 closes it. This holds whichever arrives first."""
    rid = resting_entry(store, "a", "o-rest", CLOSE)
    down = CLOSE + timedelta(minutes=10)
    owner_down(store, "a", down, "crash")
    supervisor_pass(store, down + timedelta(seconds=5))
    steps = [
        lambda: orphan_fills(store, "a", "o-rest", 0.1, down + timedelta(seconds=10)),
        lambda: cancel_confirmed(store, "o-rest", not_found=True),
    ]
    for step in steps if fill_first else steps[::-1]:
        step()
    for s in range(15, 90, 5):
        supervisor_pass(store, down + timedelta(seconds=s))
    assert reservation_reason(store, rid) == "fill"
    closes = [o for o in store.orders("a") if o["order_id"] != "o-rest"]
    assert (
        len(closes) == 1
        and closes[0]["side"] == "SELL"
        and float(closes[0]["qty"]) == 0.1
    )
    assert len(_events(store, "orphan_fill_closed", "a")) == 1


def test_og3_restart_reconcile_cancels_leftovers_before_any_new_entry(store):
    rid = resting_entry(store, "a", "o-left", CLOSE)
    now = CLOSE + timedelta(minutes=20)
    assert restart_reconcile(store, "a", now, venue_open=["o-left"]) == ["o-left"]
    blocked, why = entry_blocked(store, "a", now)
    assert blocked and "resting" in why.lower()
    cancel_confirmed(store, "o-left")
    assert reservation_reason(store, rid) == "cancel"
    assert not entry_blocked(store, "a", now + timedelta(seconds=1))[0]


def test_og4_an_entry_still_resting_at_bar_close_plus_2_min_alerts_once_and_blocks_only_its_strategy(
    store,
):
    resting_entry(store, "a", "o-rest", CLOSE)
    due = CLOSE + BAR + timedelta(minutes=2)
    supervisor_pass(store, due - timedelta(seconds=5))
    assert not _events(store, "resting_entry_orphaned")
    for s in range(0, 30, 5):
        supervisor_pass(store, due + timedelta(seconds=s))
    assert len(_events(store, "resting_entry_orphaned", "a")) == 1
    blocked, why = entry_blocked(store, "a", due + timedelta(seconds=31))
    assert blocked and "resting" in why.lower()
    assert not entry_blocked(store, "b", due + timedelta(seconds=31))[0]
    cancel_confirmed(store, "o-rest")
    supervisor_pass(store, due + timedelta(seconds=40))
    assert not entry_blocked(store, "a", due + timedelta(seconds=41))[0]


def test_og4b_the_bar_plus_2_min_check_runs_in_the_supervisor_and_fires_with_the_owner_dead(
    store,
):
    """Advisor 06:20 UK: the check is the supervisor's, so a dead owner can't silence it."""
    resting_entry(store, "a", "o-rest", CLOSE)
    owner_down(store, "a", CLOSE + timedelta(minutes=5), "crash")
    due = CLOSE + BAR + timedelta(minutes=2)
    for s in range(0, 30, 5):
        supervisor_pass(store, due + timedelta(seconds=s))
    assert len(_events(store, "resting_entry_orphaned", "a")) == 1
    blocked, why = entry_blocked(store, "a", due + timedelta(seconds=31))
    assert blocked and "resting" in why.lower()


END = CLOSE + BAR  # the resting bar's close, where the owner cancels what is still resting


def _closes(store, name="a", entry="o-rest"):
    return [o for o in store.orders(name) if o["order_id"] != entry]


def test_og6a_not_found_holds_blocks_only_its_strategy_and_a_found_fill_before_close_is_booked_and_kept(store):
    rid = resting_entry(store, "a", "o-rest", CLOSE)
    assert "o-rest" in bar_close_cancel(store, "a", END)
    answer_cancel(store, "o-rest", "not_found", END + timedelta(seconds=1))
    for s in range(5, 600, 5):  # ten minutes of silence: nothing releases, the Strategy stays blocked
        supervisor_pass(store, END + timedelta(seconds=s))
    assert reservation_reason(store, rid) is None
    blocked, why = entry_blocked(store, "a", END + timedelta(seconds=601))
    assert blocked and "resting" in why.lower()
    assert not entry_blocked(store, "b", END + timedelta(seconds=601))[0]
    venue_lookup(store, "o-rest", terminal="filled", fills=[(0.1, END - timedelta(seconds=30))],
                 at=END + timedelta(seconds=605))
    supervisor_pass(store, END + timedelta(seconds=610))
    assert reservation_reason(store, rid) == "fill"
    assert sum(float(f["qty"]) for f in store.fills("a")) == pytest.approx(0.1)  # booked once
    assert _closes(store) == []  # filled before the bar close: the Strategy keeps it
    assert not entry_blocked(store, "a", END + timedelta(seconds=611))[0]


def test_og6b_not_found_alerts_once_at_bar_close_plus_2_min_and_releases_only_on_a_zero_fill_lookup(store):
    rid = resting_entry(store, "a", "o-rest", CLOSE)
    bar_close_cancel(store, "a", END)
    answer_cancel(store, "o-rest", "not_found", END + timedelta(seconds=1))
    supervisor_pass(store, END + timedelta(minutes=2) - timedelta(seconds=5))
    assert not _events(store, "resting_entry_orphaned")
    for s in range(0, 60, 5):
        supervisor_pass(store, END + timedelta(minutes=2, seconds=s))
    assert len(_events(store, "resting_entry_orphaned", "a")) == 1
    assert reservation_reason(store, rid) is None
    at = END + timedelta(minutes=4)
    venue_lookup(store, "o-rest", terminal="canceled", fills=[], at=at)
    supervisor_pass(store, at + timedelta(seconds=5))
    assert reservation_reason(store, rid) == "cancel"
    assert store.fills("a") == [] and _closes(store) == []
    assert not entry_blocked(store, "a", at + timedelta(seconds=6))[0]


@pytest.mark.parametrize("after_close", [False, True])
def test_og6c_a_cancel_racing_a_fill_books_it_once_keeps_it_before_close_and_closes_it_at_market_after(
    store, after_close
):
    rid = resting_entry(store, "a", "o-rest", CLOSE)
    bar_close_cancel(store, "a", END)
    filled_at = END + timedelta(seconds=2) if after_close else END - timedelta(seconds=2)
    orphan_fills(store, "a", "o-rest", 0.1, filled_at)
    answer_cancel(store, "o-rest", "filled", END + timedelta(seconds=3))
    for s in range(5, 90, 5):
        supervisor_pass(store, END + timedelta(seconds=s))
    assert reservation_reason(store, rid) == "fill"  # converted once, never 'cancel'
    closes = _closes(store)
    if after_close:  # an orphan: one reduce-only market close through the exit path
        assert len(closes) == 1
        c = closes[0]
        assert (c["side"], c["order_type"], float(c["qty"])) == ("SELL", "MARKET", 0.1)
        assert is_reduce_only(store, c["order_id"])
        assert len(_events(store, "orphan_fill_closed", "a")) == 1
    else:
        assert closes == [] and not _events(store, "orphan_fill_closed", "a")
    assert not entry_blocked(store, "a", END + timedelta(seconds=91))[0]  # the cancel record never stays open


def test_og6d_a_partial_fill_then_a_confirmed_cancel_keeps_the_filled_part_and_releases_only_the_remainder(store):
    rid = resting_entry(store, "a", "o-rest", CLOSE)  # reserves 0.1
    orphan_fills(store, "a", "o-rest", 0.04, END - timedelta(seconds=20))
    bar_close_cancel(store, "a", END)
    answer_cancel(store, "o-rest", "canceled", END + timedelta(seconds=1))
    venue_lookup(store, "o-rest", terminal="canceled", fills=[(0.04, END - timedelta(seconds=20))],
                 at=END + timedelta(seconds=2))
    for s in range(5, 60, 5):
        supervisor_pass(store, END + timedelta(seconds=s))
    assert sum(float(f["qty"]) for f in store.fills("a")) == pytest.approx(0.04)  # the lookup books nothing twice
    assert _closes(store) == []  # filled before the close: kept
    assert reservation_reason(store, rid) == "cancel"  # the unfilled remainder
    assert not entry_blocked(store, "a", END + timedelta(seconds=61))[0]


@pytest.mark.parametrize(
    "entry,close", [("BUY", "SELL"), ("SELL", "BUY")]
)  # a long's and a short's orphan
def test_og5_an_orphan_fill_is_closed_at_market_reduce_only_journalled_and_alerted_once(
    store, entry, close
):
    resting_entry(store, "a", "o-rest", CLOSE, side=entry)
    down = CLOSE + timedelta(minutes=10)
    owner_down(store, "a", down, "crash")
    orphan_fills(store, "a", "o-rest", 0.1, down + timedelta(seconds=20), side=entry)
    for s in range(25, 90, 5):
        supervisor_pass(store, down + timedelta(seconds=s))
    closes = [o for o in store.orders("a") if o["order_id"] != "o-rest"]
    assert len(closes) == 1, closes
    c = closes[0]
    assert (c["side"], c["order_type"], float(c["qty"])) == (close, "MARKET", 0.1)
    assert c["intent"] in (
        "exit",
        "pm_flatten",
        "risk_halt",
    )  # the exit path, never an entry
    assert is_reduce_only(store, c["order_id"])
    assert len(_events(store, "orphan_fill_closed", "a")) == 1
    assert any(
        "orphan fill closed" in e.message.lower()
        for e in _events(store, "orphan_fill_closed", "a")
    )
