"""v2 P2-2: the portfolio gate on the in-memory ledger: reservations and first-come order, failing closed, the
portfolio's entry blocks (halt, day pause, stale book mark) and the supervisor's marking (Advisor ruling 6 Oct ~21:54;
the 60 s rule, PM yes, board 23:03)."""

from datetime import datetime, timedelta, timezone

from sleeve_fund.portfolio import Holding, Intent
from sleeve_fund.portfolio.gate import (
    MARK_STALE,
    MemoryLedger,
    apply_flow,
    check_order,
    entry_block,
    mark_book,
    resume_after_halt,
    sweep,
)
from sleeve_fund.risk import PORTFOLIO

T0 = datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc)
PX = 60_000.0


def _buy(qty, side=1):
    return Intent("BTC", side, qty, PX, PX / 5, PX * 0.005, 0.001, 0.001)


def _marked(equity=20_000.0, at=T0):
    ledger = MemoryLedger(equity=equity)
    mark_book(ledger, equity, at, PORTFOLIO)
    return ledger


def test_an_approved_entry_is_reserved_so_the_next_strategy_sees_the_room_it_took():
    """Net 0.5x of 20,000 = 10,000. A takes 0.1 (6,000); B asks 0.1 too and gets the 4,000 left, 0.066."""
    ledger = _marked()
    a = check_order(ledger, "a", _buy(0.1), PORTFOLIO, T0)
    b = check_order(ledger, "b", _buy(0.1), PORTFOLIO, T0)
    assert (a.decision.outcome, a.decision.qty) == ("approved", 0.1)
    assert (b.decision.outcome, b.decision.qty, b.decision.limit) == ("trimmed", 0.066, "net")
    assert [r.seq for r in ledger.records] == [1, 2] and [r.strategy for r in ledger.records] == ["a", "b"]
    ledger.release(a.reservation)  # A's entry was cancelled: its room comes back
    assert check_order(ledger, "c", _buy(0.06), PORTFOLIO, T0).decision.outcome == "approved"


def test_a_refused_entry_reserves_nothing_and_is_still_recorded():
    ledger = _marked()
    ledger.held = [Holding("BTC", 10_000.0, 2_000.0, 0.0)]
    c = check_order(ledger, "a", _buy(0.01), PORTFOLIO, T0)
    assert (c.decision.outcome, c.reservation, c.seq) == ("refused", None, 1) and not ledger.reserved


def test_the_check_fails_closed_when_the_ledger_fails():
    class Broken(MemoryLedger):
        def positions(self):
            raise ConnectionError("database gone")

    c = check_order(Broken(equity=20_000.0), "a", _buy(0.1), PORTFOLIO, T0)
    assert (c.decision.outcome, c.decision.qty, c.seq) == ("refused", 0.0, None)
    assert "couldn't run (ConnectionError: database gone)" in c.decision.why


def test_a_reservation_its_process_never_released_is_swept_with_an_alert():
    ledger = _marked()
    check_order(ledger, "a", _buy(0.1), PORTFOLIO, T0)
    assert sweep(ledger, T0 + timedelta(minutes=5)) == []
    assert len(sweep(ledger, T0 + timedelta(minutes=11))) == 1 and not ledger.reserved
    assert ledger.alerts[-1][0] == "reservation_swept"


def test_a_book_mark_older_than_60_s_blocks_entries_and_alerts_once_per_spell():
    ledger = _marked()
    assert entry_block(ledger, T0 + MARK_STALE) is None  # exactly 60 s passes
    late = T0 + MARK_STALE + timedelta(seconds=1)
    for _ in range(3):
        kind, why = entry_block(ledger, late)
        assert kind == "book_mark_stale" and "last marked 61 s ago" in why
    assert [a[0] for a in ledger.alerts] == ["book_mark_stale"]
    mark_book(ledger, 20_000.0, late, PORTFOLIO)  # the supervisor is back
    assert entry_block(ledger, late) is None
    entry_block(ledger, late + 2 * MARK_STALE)  # a new spell, a new alert
    assert [a[0] for a in ledger.alerts] == ["book_mark_stale"] * 2


def test_a_book_never_marked_blocks_entries_with_one_alert():
    ledger = MemoryLedger(equity=20_000.0)
    assert entry_block(ledger, T0)[0] == entry_block(ledger, T0)[0] == "book_mark_stale"
    assert len(ledger.alerts) == 1


def test_15_percent_below_the_reference_halts_until_the_pm_resumes_then_rebases():
    ledger = _marked()
    assert mark_book(ledger, 21_000.0, T0 + timedelta(seconds=5), PORTFOLIO) is None  # a new high
    assert mark_book(ledger, 17_850.0, T0 + timedelta(seconds=10), PORTFOLIO) == "halt"  # 15% below 21,000
    assert mark_book(ledger, 17_800.0, T0 + timedelta(seconds=15), PORTFOLIO) is None  # already halted: once
    kind, why = entry_block(ledger, T0 + timedelta(seconds=15))
    assert kind == "portfolio_halt" and "only the PM clears that" in why
    resume_after_halt(ledger)
    assert ledger.state().reference == 17_800.0  # the next halt is 15% below the book at the resume
    # the day lost over 3% (from 20,000), so the pause, its own condition, takes over at the next mark
    assert mark_book(ledger, 17_800.0, T0 + timedelta(seconds=20), PORTFOLIO) == "pause"
    assert entry_block(ledger, T0 + timedelta(seconds=20))[0] == "portfolio_pause"


def test_3_percent_down_on_the_day_pauses_entries_until_the_next_midnight():
    ledger = _marked()
    assert mark_book(ledger, 19_400.0, T0 + timedelta(seconds=5), PORTFOLIO) == "pause"
    st = ledger.state()
    assert st.paused_until == datetime(2026, 10, 8, tzinfo=timezone.utc)
    assert entry_block(ledger, T0 + timedelta(seconds=6))[0] == "portfolio_pause"
    midnight = datetime(2026, 10, 8, 0, 0, 5, tzinfo=timezone.utc)
    mark_book(ledger, 19_400.0, midnight, PORTFOLIO)  # a new day starts at the last mark before 00:00
    assert ledger.state().day_start == 19_400.0 and entry_block(ledger, midnight) is None


def test_a_deposit_is_not_a_gain_and_a_withdrawal_not_a_loss():
    ledger = _marked()
    apply_flow(ledger, -1_000.0)
    assert mark_book(ledger, 19_000.0, T0 + timedelta(seconds=5), PORTFOLIO) is None  # not a 5% day loss
    assert ledger.state().reference == 19_000.0 and ledger.state().day_start == 19_000.0
