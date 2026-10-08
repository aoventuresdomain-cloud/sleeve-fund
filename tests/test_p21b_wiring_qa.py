"""QA Done-when cells for P2-1b W1 and W2, the portfolio gate on paper (Head of QA, 8 Oct; PE2 plan
pe2-plans/p2-1b-paper-wiring.md). They are written before the build under the money-path rule, and each is a strict
xfail on main 9d5ee60 for the reason it names.

Run them from a worktree's tests/ beside test_gate_ledger.py and test_portfolio_gate.py, on SQLite and on PG
(TEST_DATABASE_URL).

PE2 may rename the ADAPTERS block (marked) to the entry points as built. The assertions are QA's: change them only
through the HoQA.

W1 (the supervisor marks the fund; no fill-path change)
  w1a  a pass marks portfolio_state with fund_equity at that time.
  w1b  book_marks gets one row per minute of 5 s passes, plus one row at the pass that changes the status.
  w1c  a fund_equity that raises (the CASH-2 stub): the pass doesn't crash, nothing is marked, and after 60 s the
       gate and CHOKE both block entries as stale, with one alert per spell.
  w1d  a halt halts every strategy (CHOKE names the portfolio halt) and flattens each one that holds, through the
       exit path (one flatten command each, never repeated on later passes). A flat strategy gets no sale.
  w1e  a pause blocks entries only: CHOKE names it until 00:00 UTC, with no flatten and no strategy halted; it clears
       at 00:00.
  w1f  sweep: a past-TTL reservation whose order row is finished is released as 'ttl'; one sent but never acked
       is kept, with its cancel asked for (the TTL covers send to ack only: Advisor 8 Oct 06:10 UK).
  w1g  an acked resting order's reservation outlives the TTL, hours on: never cancelled or expired by time.
W2 (entry path)
  w2a  an approved entry's order row carries its reservation id.
  w2b  that order's fills reduce the reservation in book_fill's own transaction (partial: exact share; full:
       released 'fill'); reject and cancel release it with their reasons.
  w2c  the release is atomic with the fill: when the reservation write fails, the fill isn't booked either.
  w2f  an unfilled resting entry is cancelled by its own process at its next bar close; the confirmed cancel
       releases the reservation (a partial fill keeps what filled).
  w2d  the paper GateFn does no mark per check: portfolio_state.marked_at is unchanged by a check.
  w2e  exits, stops, the PM's close and liquidations never call check_order (HoQA F208-1): no decision row, no
       reservation, and the gate never blocks them, even while halted.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest
from sqlalchemy import select

from sleeve_fund.money import money
from sleeve_fund.paper.runtime import entry_blocked
from sleeve_fund.portfolio import gate as gate_mod
from sleeve_fund.portfolio.gate import check_order, mark_book
from sleeve_fund.risk import PORTFOLIO
from sleeve_fund.store import book_marks_t, events_t, gate_decisions_t, gate_reservations_t, portfolio_state_t
from test_gate_ledger import CellLedger, _drop_schemas, _store, _strategies  # noqa: F401
from test_portfolio_gate import _buy

T0 = datetime(2026, 10, 8, 9, 0, 3, tzinfo=timezone.utc)
EQ = D("20000")


def cell(slice_: str, why: str):
    return pytest.mark.xfail(strict=True, reason=f"P2-1b {slice_}: {why}")


# === ADAPTERS (PE2: rename these to what you build; nothing below this block changes) ===============================

def portfolio_pass(store, now: datetime, equity) -> None:
    """One supervisor poll's portfolio work at `now`: mark the fund with fund_equity (`equity`: a value, or a
    callable that raises like the CASH-2 stub), write book_marks, act on a halt or pause, sweep orphans. No strategy
    process is started."""
    from sleeve_fund import supervisor

    sup = supervisor.Supervisor(store)
    fn = equity if callable(equity) else (lambda *a, **k: D(equity))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(supervisor, "fund_equity", fn, raising=False)
        sup.mark_portfolio(now)


def paper_gate(store):
    """The paper GateFn over DbLedger: gate(strategy, intent: dict, ts_ns) -> Gated, as research.portfolio's."""
    from sleeve_fund.paper import portfolio_gate  # W2's module

    return portfolio_gate.paper_gate(store)


def order_reservation(store, order_id: str):
    """The reservation id the order row carries (W2)."""
    from sleeve_fund.store import orders_t

    with store.engine.connect() as c:
        return c.execute(select(orders_t.c.reservation_id).where(orders_t.c.order_id == order_id)).scalar()


def record_entry_order(store, strategy: str, order_id: str, qty: D, reservation: int) -> None:
    """The engine's write of a gated entry's order row, with the reservation it holds."""
    store.record_order(strategy, order_id=order_id, side="BUY", qty=float(qty), intent="entry", reason="qa",
                       reservation=reservation)


def cancel_resting_entries_at_bar_close(store, strategy: str, bar_close: datetime) -> None:
    """The strategy's own process at its next bar close: cancel its unfilled resting entry, and the venue confirms."""
    from sleeve_fund.paper import portfolio_gate

    portfolio_gate.cancel_resting_entries(store, strategy, bar_close)


# =====================================================================================================================

def _ledger(store):
    """A DbLedger on the cell's journal that tags each intent as the engine does (test_gate_ledger's harness)."""
    return CellLedger(equity=EQ, store=store)


def _state(store):
    with store.engine.connect() as c:
        return c.execute(select(portfolio_state_t).where(portfolio_state_t.c.id == 1)).first()


def _marks(store):
    with store.engine.connect() as c:
        return list(c.execute(select(book_marks_t).order_by(book_marks_t.c.ts)))


def _alerts(store, kind):
    with store.engine.connect() as c:
        return list(c.execute(select(events_t).where(events_t.c.kind == kind)))


def _res(store, rid):
    with store.engine.connect() as c:
        return c.execute(select(gate_reservations_t).where(gate_reservations_t.c.decision_id == rid)).one()


def _decisions(store):
    with store.engine.connect() as c:
        return list(c.execute(select(gate_decisions_t)))


@pytest.fixture
def store():
    s = _store()
    _strategies(s)
    return s


def _hold(store, name="a", qty=0.1, px=60000.0):
    store.record_order(name, order_id=f"{name}-held", side="BUY", qty=qty, intent="entry", reason="qa")
    store.book_fill(name, side="BUY", qty=qty, price=px, fee=1.0, order_id=f"{name}-held", trade_id="t1", ts=T0)


def _flattens(store, name):
    return [c for c in store.pending_commands(name) if c["command"] == "flatten"]


def _entry_intent(qty="0.1"):
    return {"side": 1, "qty": D(qty), "close": D(60000), "price": D(60000), "stop_frac": 0.02, "what": "entry",
            "step": D("0.001"), "min_qty": D("0.001"), "instrument": "BTC", "leverage": None, "atr_pct": None}


def _ns(t: datetime) -> int:
    return int(t.timestamp()) * 1_000_000_000 + t.microsecond * 1000


# --- W1 ---------------------------------------------------------------------------------------------------------------

def test_w1a_a_pass_marks_the_fund_at_its_time(store):
    portfolio_pass(store, T0, EQ)
    st = _state(store)
    assert money(st.book_equity) == EQ and st.marked_at.replace(tzinfo=timezone.utc) == T0


def test_w1b_one_book_mark_per_minute_and_one_on_each_status_change(store):
    t = T0
    for _ in range(37):  # 5 s passes over 3 minutes
        portfolio_pass(store, t, EQ)
        t += timedelta(seconds=5)
    rows = _marks(store)
    minutes = {r.ts.replace(second=0, microsecond=0) for r in rows}
    assert len(rows) == len(minutes) == 4 and {r.status for r in rows} == {"ok"}
    at = t + timedelta(seconds=7)  # mid-minute: a 3% daily loss pauses the fund
    portfolio_pass(store, at, EQ * D("0.96"))
    rows = _marks(store)
    assert rows[-1].status == "paused" and rows[-1].ts.replace(tzinfo=timezone.utc) == at
    assert money(rows[-1].equity) == EQ * D("0.96")


def test_w1c_an_equity_read_that_fails_blocks_entries_as_stale_and_never_crashes_the_pass(store):
    portfolio_pass(store, T0, EQ)

    def stub(*a, **k):
        raise NotImplementedError("fund_equity waits on CASH-2")

    for s in range(5, 125, 5):
        portfolio_pass(store, T0 + timedelta(seconds=s), stub)  # never raises out of the pass
    assert _state(store).marked_at.replace(tzinfo=timezone.utc) == T0  # nothing marked from a failed read
    late = T0 + timedelta(seconds=125)
    blocked, why = entry_blocked(store, "a", late)
    assert blocked and "marked" in why.lower()
    c = check_order(_ledger(store), "a", _buy("0.1"), PORTFOLIO, late)
    assert (c.decision.outcome, c.decision.limit_hit, c.reservation) == ("rejected", "portfolio_state_stale", None)
    assert len(_alerts(store, "portfolio_state_stale")) == 1


def test_w1d_a_halt_halts_every_strategy_and_flattens_each_holder_once_through_the_exit_path(store):
    _hold(store, "a")
    portfolio_pass(store, T0, EQ)
    portfolio_pass(store, T0 + timedelta(seconds=5), EQ * D("0.84"))  # 16% under the HWM: the 15% halt
    for name in ("a", "b"):
        blocked, why = entry_blocked(store, name, T0 + timedelta(seconds=6))
        assert blocked and "portfolio halted" in why.lower(), (name, why)
    assert len(_flattens(store, "a")) == 1 and not _flattens(store, "b")  # a flat strategy sells nothing
    for s in (10, 15, 20):
        portfolio_pass(store, T0 + timedelta(seconds=s), EQ * D("0.80"))
    assert len(_flattens(store, "a")) == 1  # once per halt, not once per pass
    assert len(_alerts(store, "portfolio_halt")) == 1


def test_w1e_a_pause_blocks_entries_only_until_midnight(store):
    _hold(store, "a")
    portfolio_pass(store, T0, EQ)
    portfolio_pass(store, T0 + timedelta(seconds=5), EQ * D("0.965"))  # 3.5% down on the day
    blocked, why = entry_blocked(store, "a", T0 + timedelta(seconds=6))
    assert blocked and "portfolio paused" in why.lower()
    assert not _flattens(store, "a") and store.sleeve("a").status != "halted"
    midnight = datetime(2026, 10, 9, 0, 0, 0, 1, tzinfo=timezone.utc)
    portfolio_pass(store, midnight, EQ * D("0.965"))
    blocked, why = entry_blocked(store, "a", midnight + timedelta(seconds=1))
    assert not blocked or "portfolio" not in (why or "").lower()


def test_w1f_past_the_ttl_a_finished_order_is_released_and_an_unacked_one_has_its_cancel_asked(store):
    """The TTL covers send to venue ack only (Advisor 8 Oct 06:10 UK, MUST)."""
    portfolio_pass(store, T0, EQ)
    led = _ledger(store)
    gone = check_order(led, "a", _buy("0.1"), PORTFOLIO, T0 + timedelta(seconds=1)).reservation
    unacked = check_order(led, "b", _buy("0.1"), PORTFOLIO, T0 + timedelta(seconds=2)).reservation
    for rid, oid, status in ((gone, "o-gone", "canceled"), (unacked, "o-sent", None)):
        led.attach_order(rid, oid)  # the harness writes the order row ('submitted'), as the engine's send does
        if status:
            store.update_order(oid, status=status)
    later = T0 + gate_mod.RESERVATION_TTL + timedelta(minutes=1)
    for s in range(0, 15, 5):
        portfolio_pass(store, later + timedelta(seconds=s), EQ)
    assert _res(store, gone).release_reason == "ttl"
    kept = _res(store, unacked)
    assert kept.released_at is None and kept.cancel_sent_at is not None


@pytest.mark.parametrize("status", ["accepted", "partially_filled"])
def test_w1g_an_acked_resting_orders_reservation_outlives_the_ttl_and_is_never_cancelled_by_time(store, status):
    """Once the venue acks, the reservation lives with the order: released on a confirmed cancel, converted on a fill,
    never expired by time (Advisor 8 Oct 06:10 UK, MUST)."""
    portfolio_pass(store, T0, EQ)
    led = _ledger(store)
    rid = check_order(led, "a", _buy("0.1"), PORTFOLIO, T0 + timedelta(seconds=1)).reservation
    led.attach_order(rid, "o-rest")
    store.update_order("o-rest", status=status)
    for later in (T0 + gate_mod.RESERVATION_TTL + timedelta(minutes=1), T0 + timedelta(hours=6)):
        for s in range(0, 15, 5):
            portfolio_pass(store, later + timedelta(seconds=s), EQ)
    r = _res(store, rid)
    assert r.released_at is None and r.cancel_sent_at is None
    assert not _alerts(store, "reservation_expired") and not _alerts(store, "reservation_order_cancelled")
    assert rid in {x.id for x in led.reservations()}  # still counted in the book


# --- W2 ---------------------------------------------------------------------------------------------------------------

def _approved(store, strategy="a", qty="0.1", oid="o-1"):
    portfolio_pass(store, T0, EQ)
    g = paper_gate(store)(strategy, _entry_intent(qty), _ns(T0 + timedelta(seconds=1)))
    assert g.decision.outcome == "approved"
    rid = g.checked.reservation
    record_entry_order(store, strategy, oid, D(qty), rid)
    return rid


@cell("W2", "there is no paper GateFn and the order row carries no reservation")
def test_w2a_an_approved_entrys_order_row_carries_its_reservation(store):
    rid = _approved(store)
    assert rid is not None and order_reservation(store, "o-1") == rid


@cell("W2", "book_fill and update_order don't touch the reservation")
def test_w2b_fills_reduce_and_reject_or_cancel_release_the_reservation(store):
    rid = _approved(store)
    store.book_fill("a", side="BUY", qty=0.03, price=60000.0, fee=0.1, order_id="o-1", trade_id="t1", ts=T0)
    r = _res(store, rid)
    assert r.released_at is None and money(r.remaining_qty) == D("0.07")
    store.book_fill("a", side="BUY", qty=0.07, price=60000.0, fee=0.1, order_id="o-1", trade_id="t2", ts=T0)
    assert _res(store, rid).release_reason == "fill"
    for oid, status, reason in (("o-2", "rejected", "reject"), ("o-3", "canceled", "cancel")):
        g = paper_gate(store)("b", _entry_intent(), _ns(T0 + timedelta(seconds=2)))
        record_entry_order(store, "b", oid, D("0.1"), g.checked.reservation)
        store.update_order(oid, status=status)
        assert _res(store, g.checked.reservation).release_reason == reason


@cell("W2", "nothing cancels an unfilled resting entry at the strategy's next bar close")
def test_w2f_an_unfilled_resting_entry_is_cancelled_at_the_next_bar_close_and_its_reservation_released(store):
    """Advisor 8 Oct 06:10 UK: the strategy's own process cancels it at its next bar close; the confirmed cancel
    releases the reservation as 'cancel'. A partly filled one keeps what filled, and its rest is released."""
    rid = _approved(store)
    store.update_order("o-1", status="accepted")
    rid2 = _approved(store, strategy="b", oid="o-2")
    store.update_order("o-2", status="accepted")
    store.book_fill("b", side="BUY", qty=0.04, price=60000.0, fee=0.1, order_id="o-2", trade_id="t1", ts=T0)
    for name in ("a", "b"):
        cancel_resting_entries_at_bar_close(store, name, T0 + timedelta(hours=1))
    assert _res(store, rid).release_reason == "cancel"
    assert _res(store, rid2).release_reason == "cancel" and money(_res(store, rid2).remaining_qty) == D("0.06")
    assert {o["order_id"]: o["status"] for o in store.orders()}["o-1"] == "canceled"


@cell("W2", "book_fill doesn't reduce the reservation, so there's no shared transaction")
def test_w2c_a_failed_reservation_write_leaves_the_fill_unbooked(store, monkeypatch):
    from sleeve_fund.gate_ledger import DbLedger

    rid = _approved(store)

    def boom(self, *a, **k):
        raise RuntimeError("reservation write failed")

    monkeypatch.setattr(DbLedger, "reduce", boom)
    with pytest.raises(Exception):
        store.book_fill("a", side="BUY", qty=0.1, price=60000.0, fee=0.1, order_id="o-1", trade_id="t1", ts=T0)
    assert not [f for f in store.fills("a") if f["order_id"] == "o-1"]
    assert _res(store, rid).released_at is None and money(_res(store, rid).remaining_qty) == D("0.1")


@cell("W2", "there is no paper GateFn")
def test_w2d_a_check_never_marks_the_book(store):
    portfolio_pass(store, T0, EQ)
    paper_gate(store)("a", _entry_intent(), _ns(T0 + timedelta(seconds=30)))
    assert _state(store).marked_at.replace(tzinfo=timezone.utc) == T0


@cell("W2", "there is no paper GateFn")
@pytest.mark.parametrize("what", ["exit", "stop_loss", "take_profit", "pm_flatten", "risk_halt", "liquidation"])
def test_w2e_exits_stops_close_now_and_liquidations_never_reach_check_order(store, monkeypatch, what):
    portfolio_pass(store, T0, EQ)
    portfolio_pass(store, T0 + timedelta(seconds=5), EQ * D("0.80"))  # halted: entries refused, exits never
    calls = []
    real = gate_mod.check_order
    spy = lambda *a, **k: calls.append(a) or real(*a, **k)  # noqa: E731
    gate = paper_gate(store)
    monkeypatch.setattr(gate_mod, "check_order", spy)
    for mod in {getattr(gate, "__module__", None), type(gate).__module__} - {None}:
        import sys
        if hasattr(sys.modules.get(mod), "check_order"):
            monkeypatch.setattr(sys.modules[mod], "check_order", spy)
    before = len(_decisions(store))
    g = gate("a", {**_entry_intent(), "side": -1, "what": what, "reduce_only": True}, _ns(T0 + timedelta(seconds=6)))
    assert not calls and len(_decisions(store)) == before
    # Not gated at all (None), or passed through whole with no reservation: never refused, even while halted
    assert g is None or (g.decision.outcome == "approved" and g.checked.reservation is None)
