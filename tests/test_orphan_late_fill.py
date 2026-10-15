"""A2-DW4 (Advisor, 15 Oct): a late fill after a cancel is closed at market, exactly and only once per quantity, and the
Strategy's entry block lifts only when the venue has confirmed the order's end AND every late-fill quantity is flat.
Paper and replay only: a backtest has no resting-cancel race, so the guard never runs there."""
from datetime import UTC, datetime, timedelta

import pytest

from sleeve_fund.paper import orphan_guard as og

CLOSE = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
DOWN = CLOSE + timedelta(minutes=10)


@pytest.fixture
def jstore():
    from test_gate_ledger import _store, _strategies

    s = _store()
    _strategies(s)
    return s


def _fill(store, trade_id, qty, at, order_id="o-rest", side="BUY", price=60000.0, fee=0.1):
    store.book_fill("a", side=side, qty=qty, price=price, fee=fee, order_id=order_id, trade_id=trade_id, ts=at)


def _closes(store, of="o-rest"):
    return [o for o in store.orders("a") if (o.get("signal") or {}).get("orphan_of") == of]


def _down(store):
    og.qa_resting_entry(store, "a", "o-rest", CLOSE)
    og.owner_down(store, "a", DOWN, "crash")


def test_dw4_2_each_late_fill_is_closed_by_exactly_what_is_still_open_partial_and_duplicate_fills_included(jstore):
    """MUST 2: close qty = the late fills (unique by venue fill id) less what is already closed. A second partial fill
    after the first close went out is closed too, by its own quantity; the same fill again closes nothing more; a pass
    with nothing new sends nothing."""
    _down(jstore)
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1))
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))
    assert [float(c["qty"]) for c in _closes(jstore)] == [pytest.approx(0.02)]

    _fill(jstore, "t2", 0.01, DOWN + timedelta(seconds=8))
    _fill(jstore, "t2", 0.01, DOWN + timedelta(seconds=8))  # the venue's duplicate event: booked once
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=10))
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=15))
    closes = _closes(jstore)
    assert sorted(float(c["qty"]) for c in closes) == [pytest.approx(0.01), pytest.approx(0.02)]
    assert len({c["order_id"] for c in closes}) == 2
    assert all(c["side"] == "SELL" and c["intent"] == "exit" and (c.get("signal") or {}).get("reduce_only")
               for c in closes)


def test_dw4_5_the_block_lifts_only_when_the_venue_confirmed_the_end_and_every_late_fill_is_flat(jstore):
    """MUST 5 (and Done-when 3): the venue's confirmed end alone is not enough while a late fill is still held; the block
    lifts once the close has filled, never on a timer."""
    _down(jstore)
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))
    assert jstore.open_cancels("a")
    _fill(jstore, "t1", 0.03, DOWN + timedelta(seconds=6))
    og.cancel_confirmed(jstore, "o-rest", at=DOWN + timedelta(seconds=7))
    (close,) = _closes(jstore)
    assert not jstore.open_cancels("a")  # the venue confirmed the order's end
    later = DOWN + timedelta(hours=3)
    assert og.resting_block(jstore, "a", later) is not None  # but the late fill is not flat yet: no timer release

    _fill(jstore, "c1", 0.03, DOWN + timedelta(seconds=9), order_id=close["order_id"], side="SELL")
    assert og.resting_block(jstore, "a", later) is None


class _Owner:
    """The Strategy's process as send_pending sees it: what it has sent (its cache) and every submit it made."""

    def __init__(self):
        self.sent, self.submits = set(), []

    def submit(self, name, side, qty, cid, reason):
        self.sent.add(cid)
        self.submits.append((name, side, float(qty), cid))


def test_one_sender_an_alive_owner_and_supervisor_passes_give_exactly_one_close(jstore):
    """HoE condition 1: the supervisor journals and alerts, never sends; the owner sends what is journaled, once,
    whichever of them journals it first."""
    og.qa_resting_entry(jstore, "a", "o-rest", CLOSE)
    end = CLOSE + timedelta(hours=1)  # the QA Strategies' 1-hour bar: a fill after it is late
    _fill(jstore, "t1", 0.05, end + timedelta(seconds=2))
    owner = _Owner()
    og.supervisor_pass(jstore, end + timedelta(seconds=3))
    og.send_pending(jstore, "a", end + timedelta(seconds=4), owner.submit, owner.sent)
    og.supervisor_pass(jstore, end + timedelta(seconds=5))
    og.send_pending(jstore, "a", end + timedelta(seconds=6), owner.submit, owner.sent)
    assert [s[1:3] for s in owner.submits] == [("SELL", pytest.approx(0.05))]
    assert len(_closes(jstore)) == 1


def test_a_crash_loop_sends_one_close_per_quantity_and_a_close_lost_with_its_process_is_replaced_once(jstore):
    """HoE condition 3: a restarted process sends the journaled close once; when the process before it had sent it
    and its in-process venue ended with it (the restart ends its open orders), one fresh close goes for what is still
    open, never two."""
    _down(jstore)
    _fill(jstore, "t1", 0.04, DOWN + timedelta(seconds=1))
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))  # journaled while down: can't be sent
    (first,) = _closes(jstore)
    first_proc = _Owner()
    og.send_pending(jstore, "a", DOWN + timedelta(minutes=1), first_proc.submit, first_proc.sent)
    og.send_pending(jstore, "a", DOWN + timedelta(minutes=1, seconds=5), first_proc.submit, first_proc.sent)
    assert [s[3] for s in first_proc.submits] == [first["order_id"]]

    jstore.update_order(first["order_id"], status="canceled")  # it crashed again; its sandbox took the close with it
    second_proc = _Owner()
    for s in (0, 5, 10):
        og.send_pending(jstore, "a", DOWN + timedelta(minutes=2, seconds=s), second_proc.submit, second_proc.sent)
    assert [(s[1], s[2]) for s in second_proc.submits] == [("SELL", pytest.approx(0.04))]
    assert second_proc.submits[0][3] != first["order_id"]


def test_a_fill_made_while_its_process_was_down_stays_owed_after_the_process_is_back(jstore):
    """MUST 5 with a restart between: owner_up does not forget a fill made while the process was down."""
    _down(jstore)
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1))
    og.owner_up(jstore, "a", DOWN + timedelta(minutes=1))
    later = DOWN + timedelta(hours=2)
    assert og.resting_block(jstore, "a", later) is not None
    owner = _Owner()
    og.send_pending(jstore, "a", later, owner.submit, owner.sent)
    assert [s[1:3] for s in owner.submits] == [("SELL", pytest.approx(0.02))]


def test_a_held_late_fill_alerts_that_it_fails_safe_and_re_alerts_every_30_minutes(jstore):
    """HoE condition 2 and the Advisor's follow-up: a dead owner's close can't go, so the Strategy stays blocked; the
    alert says so, and while it holds it is said again every 30 minutes, not every pass."""
    _down(jstore)
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1))
    for m in range(0, 65, 1):
        og.supervisor_pass(jstore, DOWN + timedelta(seconds=5, minutes=m))
    closed = [e for e in jstore.events("a", limit=500) if e["kind"] == og.ORPHAN_CLOSED]
    held = [e for e in jstore.events("a", limit=500) if e["kind"] == og.LATE_HELD]
    assert len(closed) == 1 and "fails safe" in closed[0]["message"]
    assert len(held) == 2  # at 30 and 60 minutes


def test_dw4_1_a_late_fill_on_a_reduce_only_exit_is_booked_as_an_exit_never_closed_and_blocks_nothing(jstore):
    """MUST 1: the guard is for opening and adding orders only. An exit (never gated) that fills after its owner went
    down is a valid exit: booked as any fill, with no close sent for it (that would open the position again) and no
    block. Its stop then follows the position left (the runtime's _sync_watched_stop on any partial close)."""
    _down(jstore)
    _fill(jstore, "t1", 0.03, DOWN - timedelta(minutes=1))  # the entry filled while its process ran: the Strategy's
    jstore.record_order("a", order_id="tp-1", side="SELL", qty=0.03, intent="take_profit", order_type="LIMIT",
                        reason="resting take-profit", signal={"reduce_only": True}, ts=DOWN - timedelta(seconds=30))
    _fill(jstore, "tp-f1", 0.01, DOWN + timedelta(seconds=3), order_id="tp-1", side="SELL")
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))
    assert og.late_fills(jstore, "a") == [] and _closes(jstore) == [] and _closes(jstore, of="tp-1") == []
    assert sum(float(f["qty"]) * (1 if f["side"] == "BUY" else -1) for f in jstore.fills("a")) == pytest.approx(0.02)
    assert og.unflat(jstore, "a") == []


def test_dw4_4_the_late_fill_and_its_close_are_the_strategys_with_fees_on_both_legs_to_the_cent(jstore):
    """MUST 4: both legs are journaled to the owning Strategy with their own fees; the book after the close is flat and
    its cash is the start less both fees and the price move, to the cent."""
    from decimal import Decimal

    _down(jstore)
    start = jstore.journal_book("a", 0)["cash"]
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1), price=60_000.0, fee=0.72)
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))
    (close,) = _closes(jstore)
    assert close["sleeve"] == "a" and close["reason"].startswith("Late fill after cancel, closed: o-rest filled")
    _fill(jstore, "c1", 0.02, DOWN + timedelta(seconds=6), order_id=close["order_id"], side="SELL", price=59_950.0,
          fee=0.71)
    book = jstore.journal_book("a", 0)
    assert book["qty"] == pytest.approx(0)
    got = Decimal(str(book["cash"])) - Decimal(str(start))
    want = Decimal("0.02") * (Decimal("59950") - Decimal("60000")) - Decimal("0.72") - Decimal("0.71")
    assert got.quantize(Decimal("0.01")) == want.quantize(Decimal("0.01"))  # -2.43
    assert {f["sleeve"] for f in jstore.fills("a")} == {"a"}


# ---- MUST 3 (A2-DW4 grace correction, Advisor 15 Oct ~10:16 UK): never unprotected ---------------------------------

def _late(status="ACCEPTED", grace=1000, sent=True, margin=False, liq=None, entry=60_000.0, side=1):
    from types import SimpleNamespace

    from nautilus_trader.model import OrderStatus

    t = [10**18]
    order = None if status is None else SimpleNamespace(status=getattr(OrderStatus, status))
    strat = SimpleNamespace(_late_sent={"orphan-close-o-rest": t[0]} if sent else {},
                            cache=SimpleNamespace(order=lambda cid: order), _margin=margin,
                            _mark=lambda: (0, 1_000.0, 0.1 * side, 0), _liq_now=lambda cash, qty: liq,
                            _entry_px=entry, _entry_side=side, clock=SimpleNamespace(timestamp_ns=lambda: t[0]),
                            _cfg=SimpleNamespace(backstop_grace_ms=grace))
    return strat, t


def test_dw4_3_the_safety_stop_goes_on_once_the_close_is_unfilled_for_backstop_grace_ms():
    from sleeve_fund.strategies.base import LongFlatStrategy

    strat, t = _late()
    assert LongFlatStrategy._late_safety_due(strat, 59_900.0) is None
    t[0] += 999_000_000
    assert LongFlatStrategy._late_safety_due(strat, 59_900.0) is None
    t[0] += 1_000_000
    assert "1,000 ms" in LongFlatStrategy._late_safety_due(strat, 59_900.0)  # 1 s on: S3


@pytest.mark.parametrize("status", ["REJECTED", "DENIED", None])
def test_dw4_3_the_safety_stop_goes_on_at_once_when_the_close_is_refused_or_unknown(status):
    from sleeve_fund.strategies.base import LongFlatStrategy

    strat, _ = _late(status=status)
    assert LongFlatStrategy._late_safety_due(strat, 59_900.0) is not None


def test_dw4_3_the_safety_stop_goes_on_at_once_past_half_the_way_to_liquidation():
    from sleeve_fund.strategies.base import LongFlatStrategy

    strat, _ = _late(margin=True, liq=50_000.0)  # half of 60,000 to 50,000 is 5,000
    assert LongFlatStrategy._late_safety_due(strat, 55_100.0) is None
    assert "liquidation" in LongFlatStrategy._late_safety_due(strat, 54_900.0)
    short, _ = _late(margin=True, liq=70_000.0, side=-1)
    assert LongFlatStrategy._late_safety_due(short, 65_100.0) is not None


def test_dw4_3_nothing_is_due_with_no_close_out_or_once_it_filled():
    from sleeve_fund.strategies.base import LongFlatStrategy

    strat, t = _late(sent=False)
    t[0] += 10**10
    assert LongFlatStrategy._late_safety_due(strat, 59_900.0) is None
    filled, t = _late(status="FILLED")
    t[0] += 10**10
    assert LongFlatStrategy._late_safety_due(filled, 59_900.0) is None


def test_dw4_3_a_grace_of_30000_ms_is_refused_at_load(instrument):
    """No new 30 s constant: the close's grace is backstop_grace_ms, 0 to 3,000."""
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies.base import LongFlatConfig

    with pytest.raises(ValueError, match="backstop_grace_ms"):
        LongFlatConfig(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-INTERNAL"),
                       assumed_taker_fee=0.001, backstop_grace_ms=30_000)


def test_dw4_replay_parity_reads_no_late_fill_close_as_a_gate_decision(jstore):
    """Paper and replay (A2-DW4 follow-up): a late fill's close is an exit, never gated, so paper's replayable gate
    decisions are the same rows with or without it, and replay parity has no decision of its own to compare for it.
    The backtest side is a no-op (orphan_guard and _guarding say so): it has no resting-cancel race."""
    from decimal import Decimal

    from sleeve_fund.gate_ledger import DbLedger
    from sleeve_fund.research.replay_parity import from_journal

    ledger = DbLedger(jstore, lambda: (Decimal(0), ()))
    _down(jstore)
    before = from_journal(ledger.replayable())
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1))
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))
    (close,) = _closes(jstore)
    rows = ledger.replayable()
    assert all(r.intent_id != close["order_id"] for r in rows)
    assert from_journal(rows) == before


def test_dw4_6_a_dead_owners_late_fill_is_critical_at_once_and_holds_entries_fund_wide_until_back_and_flat(jstore):
    """Advisor HEAD3-DW, line 6 option (b): S3 is the owner's (paper's venue lives in its process), so a late fill held
    by a dead owner is a CRITICAL alert on the first supervisor pass, not after 30 minutes, and no Strategy in the fund
    opens until that owner is back AND the late fill is flat. One alert per episode; the 30-minute re-alert stands."""
    _down(jstore)
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1))
    assert og.resting_block(jstore, "b", DOWN + timedelta(seconds=2)) is None  # nothing seen yet
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=10))
    crit = [e for e in jstore.events("a", limit=500) if e["kind"] == og.UNPROTECTED]
    assert len(crit) == 1 and crit[0]["level"] == "error" and crit[0]["message"].startswith("CRITICAL")
    assert "fund" in og.resting_block(jstore, "b", DOWN + timedelta(seconds=11))

    og.owner_up(jstore, "a", DOWN + timedelta(minutes=1))  # back, but the late fill isn't flat yet
    og.supervisor_pass(jstore, DOWN + timedelta(minutes=1, seconds=5))
    assert og.resting_block(jstore, "b", DOWN + timedelta(minutes=1, seconds=6)) is not None

    (close,) = _closes(jstore)
    _fill(jstore, "c1", 0.02, DOWN + timedelta(minutes=1, seconds=7), order_id=close["order_id"], side="SELL")
    og.supervisor_pass(jstore, DOWN + timedelta(minutes=1, seconds=10))
    assert og.resting_block(jstore, "b", DOWN + timedelta(minutes=1, seconds=11)) is None
    og.cancel_confirmed(jstore, "o-rest", at=DOWN + timedelta(minutes=1, seconds=12))  # its own MUST 5 end
    assert og.resting_block(jstore, "a", DOWN + timedelta(minutes=1, seconds=13)) is None
    assert jstore.last_event("a", (og.UNPROTECTED, og.PROTECTED))["kind"] == og.PROTECTED


def test_dw4_6_a_live_owners_late_fill_holds_only_its_own_strategy(jstore):
    """With the owner alive its S3 is in force (MUST 3), so the hold stays the Strategy's own (A2-DW4 follow-up)."""
    og.qa_resting_entry(jstore, "a", "o-rest", CLOSE)
    end = CLOSE + timedelta(hours=1)
    _fill(jstore, "t1", 0.05, end + timedelta(seconds=2))
    og.supervisor_pass(jstore, end + timedelta(seconds=3))
    assert og.resting_block(jstore, "a", end + timedelta(seconds=4)) is not None
    assert og.resting_block(jstore, "b", end + timedelta(seconds=4)) is None
    assert not [e for e in jstore.events("a", limit=500) if e["kind"] == og.UNPROTECTED]
