"""v2 P2-2: the portfolio gate on the in-memory ledger: reservations and first-come order, the reservation living as
long as its order (Advisor MUST FIX 17:20 UK, cell G11), failing closed, the portfolio's entry blocks (halt, day
pause, stale book mark) and the supervisor's marking (Advisor ruling 6 Oct ~21:54; the 60 s rule, PM yes, board
23:03). PE1's cells with QA's names and Decimal money, plus PE2's G11."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from sleeve_fund.portfolio import Holding, Intent
from sleeve_fund.portfolio.gate import (
    MARK_STALE,
    RESERVATION_TTL,
    MemoryLedger,
    apply_flow,
    check_order,
    entry_block,
    mark_book,
    next_utc_midnight,
    resume_after_halt,
    sweep,
)
from sleeve_fund.risk import PORTFOLIO

T0 = datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc)
PX = D("60000")


def _buy(qty, side=1):
    return Intent("BTC", side, D(qty), PX, PX / 5, PX * D("0.005"), D("0.001"), D("0.001"))


def _marked(equity="20000", at=T0):
    ledger = MemoryLedger(equity=D(equity))
    mark_book(ledger, D(equity), at, PORTFOLIO)
    return ledger


def _nothing_live(order_id):
    return False


def _no_cancel(order_id):
    raise AssertionError("nothing should be cancelled")


def test_an_approved_entry_is_reserved_so_the_next_strategy_sees_the_room_it_took():
    """Net 0.5x of 20,000 = 10,000. A takes 0.1 (6,000); B asks 0.1 too and gets the 4,000 left, 0.066."""
    ledger = _marked()
    a = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    b = check_order(ledger, "b", _buy("0.1"), PORTFOLIO, T0)
    assert (a.decision.outcome, a.decision.approved_qty) == ("approved", D("0.1"))
    assert (b.decision.outcome, b.decision.approved_qty, b.decision.limit_hit) == ("trimmed", D("0.066"),
                                                                                   "net_instrument")
    assert [r.seq for r in ledger.records] == [1, 2] and [r.strategy for r in ledger.records] == ["a", "b"]
    ledger.release(a.reservation, "cancel")  # A's entry was cancelled: its room comes back
    assert check_order(ledger, "c", _buy("0.06"), PORTFOLIO, T0).decision.outcome == "approved"
    assert ledger.released == [(a.reservation, "cancel")]


def test_a_rejected_entry_reserves_nothing_and_is_still_recorded():
    ledger = _marked()
    ledger.held = [Holding("BTC", D(10_000), D(2_000), D(0))]
    c = check_order(ledger, "a", _buy("0.01"), PORTFOLIO, T0)
    assert (c.decision.outcome, c.reservation, c.seq) == ("rejected", None, 1) and not ledger.reserved


def test_the_check_fails_closed_when_the_ledger_fails():
    class Broken(MemoryLedger):
        def positions(self):
            raise ConnectionError("database gone")

    ledger = Broken(equity=D(20_000))
    mark_book(ledger, D(20_000), T0, PORTFOLIO)
    c = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    assert (c.decision.outcome, c.decision.approved_qty, c.seq) == ("rejected", D(0), None)
    assert "couldn't run (ConnectionError: database gone)" in c.decision.reason
    assert ledger.alerts[-1][0] == "portfolio_check_failed"


def test_float_money_from_the_ledger_fails_closed_as_a_rejected_entry():
    """Day-0 note v4: a float for money is a TypeError, which the gate turns into no entry plus an event."""
    ledger = _marked()
    ledger.equity = 20_000.0
    c = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    assert (c.decision.outcome, c.reservation) == ("rejected", None) and "TypeError" in c.decision.reason


def test_g11_a_resting_entry_keeps_its_reservation_past_the_ttl_until_its_order_is_gone():
    """Advisor MUST FIX (17:20 UK), cell G11: a reservation lives as long as its order. Past the TTL the sweep
    cancels a live order and keeps the reservation until the cancel lands; meanwhile no other strategy can take its
    headroom. Once the order is gone it is released, as "cancel" by its own event or as "ttl" by the sweep."""
    ledger = _marked()
    a = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)  # 6,000 of the 10,000 net room
    ledger.attach_order(a.reservation, "O-1")
    live, cancelled = {"O-1"}, []
    late = T0 + RESERVATION_TTL + timedelta(seconds=1)
    mark_book(ledger, D(20_000), late, PORTFOLIO)
    assert sweep(ledger, late, live.__contains__, cancelled.append) == []
    assert cancelled == ["O-1"] and a.reservation in ledger.reserved  # cancelled, still reserved
    assert ledger.alerts[-1][0] == "reservation_order_cancelled"
    b = check_order(ledger, "b", _buy("0.1"), PORTFOLIO, late)
    assert (b.decision.outcome, b.decision.approved_qty) == ("trimmed", D("0.066"))  # A's room isn't free
    live.clear()  # the cancel landed but its event was lost: the next sweep finds an orphan
    gone = sweep(ledger, late + timedelta(seconds=5), live.__contains__, _no_cancel)
    assert [r.id for r in gone] == [a.reservation] and (a.reservation, "ttl") in ledger.released
    assert ledger.alerts[-1][0] == "reservation_expired"


def test_g11_a_cancel_that_fails_keeps_the_reservation():
    ledger = _marked()
    a = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    ledger.attach_order(a.reservation, "O-1")

    def refuse(order_id):
        raise ConnectionError("venue unreachable")

    assert sweep(ledger, T0 + RESERVATION_TTL * 2, lambda o: True, refuse) == []
    assert a.reservation in ledger.reserved and ledger.alerts[-1][0] == "reservation_cancel_failed"


def test_a_reservation_whose_process_died_before_sending_is_swept_with_an_alert():
    ledger = _marked()
    check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)  # never attached to an order
    assert sweep(ledger, T0 + timedelta(minutes=5), _nothing_live, _no_cancel) == []
    assert len(sweep(ledger, T0 + timedelta(minutes=11), _nothing_live, _no_cancel)) == 1 and not ledger.reserved
    assert ledger.alerts[-1][0] == "reservation_expired"


def test_a_partial_fill_reduces_the_reservation_and_the_rest_fills_it_away():
    ledger = _marked()
    a = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    ledger.reduce(a.reservation, D("0.04"))
    r = ledger.reserved[a.reservation]
    assert r.qty == D("0.06") and r.holding.notional == D("3600") and r.holding.margin == D("720")
    ledger.reduce(a.reservation, D("0.06"))
    assert not ledger.reserved and ledger.released == [(a.reservation, "fill")]


def test_a_release_names_its_reason():
    ledger = _marked()
    a = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    with pytest.raises(ValueError, match="release reason"):
        ledger.release(a.reservation, "because")


def test_a_book_mark_older_than_60_s_blocks_entries_and_alerts_once_per_spell():
    ledger = _marked()
    assert entry_block(ledger, T0 + MARK_STALE) is None  # exactly 60 s passes
    late = T0 + MARK_STALE + timedelta(seconds=1)
    for _ in range(3):
        kind, why = entry_block(ledger, late)
        assert kind == "portfolio_state_stale" and "last marked 61 s ago" in why
    assert [a[0] for a in ledger.alerts] == ["portfolio_state_stale"]
    mark_book(ledger, D(20_000), late, PORTFOLIO)  # the supervisor is back
    assert entry_block(ledger, late) is None
    entry_block(ledger, late + 2 * MARK_STALE)  # a new spell, a new alert
    assert [a[0] for a in ledger.alerts] == ["portfolio_state_stale"] * 2


def test_a_book_never_marked_blocks_entries_with_one_alert():
    ledger = MemoryLedger(equity=D(20_000))
    assert entry_block(ledger, T0)[0] == entry_block(ledger, T0)[0] == "portfolio_state_stale"
    assert len(ledger.alerts) == 1


def test_a_blocked_portfolio_rejects_the_entry_outright_never_trims_it():
    """CHOKE: a trim is never a way round a halt, a pause or a stale book."""
    ledger = _marked()
    stale = T0 + MARK_STALE + timedelta(seconds=1)
    c = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, stale)
    assert (c.decision.outcome, c.decision.approved_qty, c.decision.limit_hit) == ("rejected", D(0),
                                                                                   "portfolio_state_stale")
    assert c.reservation is None and c.seq == 1


def test_15_percent_below_the_reference_halts_until_the_pm_resumes_then_rebases():
    ledger = _marked()
    assert mark_book(ledger, D(21_000), T0 + timedelta(seconds=5), PORTFOLIO) is None  # a new high
    assert mark_book(ledger, D(17_850), T0 + timedelta(seconds=10), PORTFOLIO) == "halt"  # 15% below 21,000
    assert mark_book(ledger, D(17_800), T0 + timedelta(seconds=15), PORTFOLIO) is None  # already halted: once
    kind, why = entry_block(ledger, T0 + timedelta(seconds=15))
    assert kind == "portfolio_halt" and "only the PM clears that" in why
    resume_after_halt(ledger)
    st = ledger.state()
    assert (st.halt_reference, st.hwm) == (D(17_800), D(21_000))  # re-based; the true HWM is kept
    # the day lost over 3% (from 20,000), so the pause, its own condition, takes over at the next mark
    assert mark_book(ledger, D(17_800), T0 + timedelta(seconds=20), PORTFOLIO) == "pause"
    assert entry_block(ledger, T0 + timedelta(seconds=20))[0] == "portfolio_pause"


def test_3_percent_down_on_the_day_pauses_entries_until_the_next_midnight():
    ledger = _marked()
    assert mark_book(ledger, D(19_400), T0 + timedelta(seconds=5), PORTFOLIO) == "pause"
    st = ledger.state()
    assert st.paused_until == next_utc_midnight(T0) == datetime(2026, 10, 8, tzinfo=timezone.utc)
    assert entry_block(ledger, T0 + timedelta(seconds=6))[0] == "portfolio_pause"
    midnight = datetime(2026, 10, 8, 0, 0, 5, tzinfo=timezone.utc)
    mark_book(ledger, D(19_400), midnight, PORTFOLIO)  # a new day starts at the last mark before 00:00
    assert ledger.state().day_start_equity == D(19_400) and entry_block(ledger, midnight) is None


def test_a_deposit_is_not_a_gain_and_a_withdrawal_not_a_loss():
    ledger = _marked()
    apply_flow(ledger, D(-1_000))
    assert mark_book(ledger, D(19_000), T0 + timedelta(seconds=5), PORTFOLIO) is None  # not a 5% day loss
    assert ledger.state().hwm == D(19_000) and ledger.state().day_start_equity == D(19_000)
