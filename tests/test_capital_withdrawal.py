"""F-C3B-A (spec qd/f-c3b-a-spec.md; Advisor 14 Oct 09:45 UK; HoE OK to invert HALT-REF-1's C2): a Strategy put away
for good (archived, retired, or archived by a clean slate) takes its whole equity out of the fund's book as a flow, and
one brought back puts it in again. A flow scales the book's high-water mark, the halt's reference and the day's start by
(book + flow) / book, so a withdrawal is never a loss and the drawdown is the same before and after. A reset is still no
flow. Strategies archived before this landed count as before until the one-off withdraw-archived command takes each out
with its own flow."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D

import pytest
from test_halt_ref import T0, _flows, _four, _make, _mark, _reset, _state, _trade, store  # noqa: F401

from sleeve_fund import supervisor
from sleeve_fund.gate_ledger import DbLedger


def _dd(st):
    return (st.hwm - st.equity) / st.hwm


def _day(st):
    return (st.day_start_equity - st.equity) / st.day_start_equity


# --- W1, W2: archive withdraws the whole equity, and keeps the drawdown ---------------------------------------------

def test_w1_archiving_after_a_gain_takes_its_equity_out_with_a_flow_not_a_loss(store):
    _make(store, "a", 16_000)
    _make(store, "b", 4_000)
    _trade(store, "b", 4_000)
    assert _four(_mark(store, T0)) == (D(24_000),) * 4
    store.archive("b")
    assert _four(_state(store)) == (D(24_000),) * 4 and store.withdrawn() == {}  # waits for the poll (F286-CR1)
    assert store.withdrawal_pending() == ["b"] and _flows(store) == []
    assert _four(_mark(store, T0 + timedelta(seconds=5))) == (D(16_000),) * 4  # 24,000 x 16,000 / 24,000
    assert supervisor.fund_equity(store) == D(16_000) and store.withdrawn() == {"b": D(8_000)}
    assert len(_flows(store)) == 1 and "-8,000.00" in _flows(store)[0] and "b put away" in _flows(store)[0]
    st = _mark(store, T0 + timedelta(minutes=1))
    assert _four(st) == (D(16_000),) * 4 and not st.halted and st.paused_until is None


def test_w2_archiving_after_a_loss_keeps_the_remaining_books_drawdown(store):
    _make(store, "a", 14_000)
    _make(store, "b", 6_000)
    _mark(store, T0)
    _trade(store, "a", -500, T0 + timedelta(minutes=1))
    _trade(store, "b", -1_500, T0 + timedelta(minutes=1))
    before = _mark(store, T0 + timedelta(minutes=2))
    assert before.equity == D(18_000) and _dd(before) == D("0.1") and not before.halted
    store.archive("b")  # its 4,500, under the 6,000 it started with
    assert _state(store) == before  # nothing leaves until the poll's own mark (F286-CR1)
    st = _mark(store, T0 + timedelta(minutes=2, seconds=5))
    assert st.equity == D(13_500) and st.hwm == D(15_000)  # 20,000 x 13,500 / 18,000
    assert _dd(st) == _dd(before) and _day(st) == _day(before)
    assert (st.halted, st.paused_until) == (before.halted, before.paused_until)  # the withdrawal alone sets nothing
    after = _mark(store, T0 + timedelta(minutes=3))
    assert after.equity == D(13_500) and _dd(after) == D("0.1") and not after.halted


# --- W3: unarchive is the undo -----------------------------------------------------------------------------------------

def test_w3_unarchiving_deposits_its_equity_then_and_the_round_trip_nets_to_nothing(store):
    _make(store, "a", 15_000)
    _make(store, "b", 5_000)
    _trade(store, "b", 1_000)
    before = _four(_mark(store, T0))
    store.archive("b")
    _mark(store, T0 + timedelta(seconds=5))
    store.unarchive("b")
    assert _four(_state(store)) == before and store.withdrawn() == {}
    assert supervisor.fund_equity(store) == D(21_000)
    assert [("-6,000.00" in f, "+6,000.00" in f) for f in _flows(store)] in ([(True, False), (False, True)],
                                                                             [(False, True), (True, False)])


# --- W4, W5: retire and a clean slate are archives -----------------------------------------------------------------------

def test_w4_retiring_withdraws_in_the_lifecycle_moves_transaction(store, monkeypatch):
    _make(store, "a", 15_000)
    _make(store, "b", 5_000)
    _mark(store, T0)
    store.set_lifecycle("b", "retire", "pm", "done with it", now=T0)
    assert store.withdrawal_pending() == ["b"] and store.withdrawn() == {}
    assert _four(_mark(store, T0 + timedelta(seconds=5))) == (D(15_000),) * 4 and store.withdrawn() == {"b": D(5_000)}
    # A put-away that fails rolls the retire back: the row stays active and nothing is withdrawn or queued.
    _make(store, "c", 5_000)
    import sleeve_fund.portfolio.gate as gate

    monkeypatch.setattr(gate, "apply_flow", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no flow")))
    monkeypatch.setattr(store, "withdrawal_blocker", lambda name: (_ for _ in ()).throw(RuntimeError("no decision")))
    with pytest.raises(RuntimeError):
        store.set_lifecycle("c", "retire", "pm", "r", now=T0)
    assert store.sleeve("c").lifecycle == "active" and "c" not in store.withdrawn() and "c" not in store.archived()
    assert "c" not in store.withdrawal_pending()


def test_w5_a_clean_slate_withdraws_each_strategy_it_archives(store, tmp_path):
    _make(store, "a", 10_000)
    _make(store, "b", 10_000)
    _trade(store, "a", 500)
    _mark(store, T0)
    slate = tmp_path / "clear.toml"
    slate.write_text('[[clear]]\nid = "s1"\nreason = "fresh book"\nkeep = ["b"]\n')
    assert supervisor.clear(store, str(slate)) == ["a"]
    assert store.withdrawal_pending() == ["a"]
    assert _four(_mark(store, T0 + timedelta(seconds=5))) == (D(10_000),) * 4 and store.withdrawn() == {"a": D(10_500)}
    assert len(_flows(store)) == 1


# --- W6: a reset is still no flow --------------------------------------------------------------------------------------

@pytest.mark.parametrize("pnl", [1_000, -1_000])
def test_w6_a_reset_is_never_a_flow(store, pnl):
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", pnl)
    before = _mark(store, T0 + timedelta(minutes=1))
    _reset(store, "a")
    assert _four(_mark(store, T0 + timedelta(minutes=2)))[:3] == _four(before)[:3] and _flows(store) == []
    assert store.withdrawn() == {}


# --- W7, W8: the archives made before this landed ------------------------------------------------------------------------

def _archived_before(store, name):
    """An archive as it was written before F-C3B-A: the row only, no withdrawal and no flow."""
    from sqlalchemy import insert

    from sleeve_fund.store import sleeve_archive_t

    with store.engine.begin() as c:
        c.execute(insert(sleeve_archive_t).values(sleeve=name, archived_at=T0))


def test_w7_an_archive_made_before_this_landed_counts_until_withdraw_archived_runs(store):
    _make(store, "a", 15_000)
    _make(store, "b", 5_000)
    _archived_before(store, "b")
    assert supervisor.fund_equity(store) == D(20_000) and store.withdrawn() == {}
    assert _four(_mark(store, T0)) == (D(20_000),) * 4 and _flows(store) == []


def test_w8_withdraw_archived_takes_each_out_once_and_refuses_while_one_holds(store):
    _make(store, "a", 15_000)
    _make(store, "b", 3_000)
    _make(store, "c", 2_000)
    _archived_before(store, "b")
    _archived_before(store, "c")
    _mark(store, T0)
    done = supervisor.withdraw_archived(store, "operator", "abc1234")
    assert sorted(done) == ["b", "c"] and _flows(store) == [] and sorted(store.withdrawal_pending()) == ["b", "c"]
    assert supervisor.withdraw_archived(store, "operator", "abc1234") == []  # already queued
    assert _four(_mark(store, T0 + timedelta(seconds=5))) == (D(15_000),) * 4 and len(_flows(store)) == 2
    assert supervisor.withdraw_archived(store, "operator", "abc1234") == [] and len(_flows(store)) == 2
    # One archived while still holding (before holders were flattened first): nothing is withdrawn until it is flat.
    _make(store, "d", 1_000)
    store.record_fill("d", side="BUY", qty=0.5, price=1_000.0, fee=0.0, order_id="dh", trade_id="dh", ts=T0)
    _archived_before(store, "d")
    with pytest.raises(RuntimeError, match="d holds"):
        supervisor.withdraw_archived(store, "operator", "abc1234")
    assert "d" not in store.withdrawn() and "d" not in store.withdrawal_pending()
    assert len(_flows(store)) == 3  # d's create, a deposit; no withdrawal


# --- W9: a mark that read the book before a withdrawal landed ----------------------------------------------------------

def test_w9_a_mark_that_read_the_book_before_a_withdrawal_landed_is_skipped(store, monkeypatch):
    _make(store, "a", 15_000)
    _make(store, "b", 5_000)
    _mark(store, T0)
    real = supervisor.fund_equity

    def read_then_archive(s):
        seen = real(s)  # 20,000: read before the archive lands
        store.archive("b")
        return seen

    monkeypatch.setattr(supervisor, "fund_equity", read_then_archive)
    st = _mark(store, T0 + timedelta(minutes=1))
    assert st.equity == D(15_000) and not st.halted and st.paused_until is None  # not 20,000, not a phantom gain
    monkeypatch.setattr(supervisor, "fund_equity", real)
    assert _four(_mark(store, T0 + timedelta(minutes=2))) == (D(15_000),) * 4


# --- W10, W11: the smaller book is the denominator ------------------------------------------------------------------------

def test_w10_a_withdrawal_that_leaves_the_live_book_over_a_limit_is_allowed(store):
    """Allowed (Advisor 09:45 UK): the gate blocks new entries as it does today, nothing is reduced, the halt stands."""
    _make(store, "a", 10_000)
    _make(store, "b", 10_000)
    _mark(store, T0)
    store.archive("b")
    st = _mark(store, T0 + timedelta(seconds=5))
    assert st.equity == D(10_000) and not st.halted
    # 14,000 of gross was 0.7x of 20,000; it is 1.4x of 10,000 now, so a 2,000 entry would be over the 1.5x cap.
    from sleeve_fund.risk import PORTFOLIO

    assert D(14_000) / st.equity + D(2_000) / st.equity > D(str(PORTFOLIO.gross))


def test_w11_a_withdrawn_strategys_capital_is_not_capital_put_in(store):
    _make(store, "a", 15_000)
    _make(store, "b", 5_000)
    _mark(store, T0)
    store.archive("b")
    assert supervisor.capital_in(store) == D(20_000)  # in the book until the poll takes it out
    _mark(store, T0 + timedelta(seconds=5))
    assert supervisor.capital_in(store) == D(15_000)


def test_the_withdrawal_is_journaled_as_capital_not_pnl(store):
    _make(store, "a", 16_000)
    _make(store, "b", 4_000)
    _mark(store, T0)
    store.archive("b")
    _mark(store, T0 + timedelta(seconds=5))
    led = DbLedger(store, lambda: ())
    assert led.state().equity == D(16_000)
    assert "isn't a gain or a loss" in _flows(store)[0]


# --- W12, W13: put away twice, and a reset's run (DA schema review, 14 Oct) ------------------------------------------------

def test_w12_putting_away_again_keeps_the_withdrawal_and_adds_no_second_flow(store):
    """Archive then retire, or archive twice: already out of the book, so the row keeps its amount and when it went,
    and nothing flows again."""
    _make(store, "a", 15_000)
    _make(store, "b", 5_000)
    _trade(store, "b", 1_000)
    _mark(store, T0)
    store.archive("b")
    went = store.archived()["b"]
    _trade(store, "a", 500)  # the book moves on; b's figure must not be read again
    marked = _four(_mark(store, T0 + timedelta(minutes=1)))
    store.archive("b")
    store.set_lifecycle("b", "retire", "pm", "done with it", now=T0 + timedelta(minutes=2))
    assert store.withdrawn() == {"b": D(6_000)} and store.archived()["b"] == went
    assert len(_flows(store)) == 1 and _four(_state(store)) == marked and marked[0] == D(15_500)
    assert supervisor.fund_equity(store) == D(15_500)


def test_w13_withdraw_archived_keys_on_reset_runs_so_never_takes_a_runs_cash_out(store):
    """A reset's run is archived with withdrawn NULL, the same as an archive from before F-C3B-A; withdraw-archived
    tells them apart by store.reset_runs() and leaves the run's cash in the book."""
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", 1_000)
    _mark(store, T0 + timedelta(minutes=1))
    run = _reset(store, "a")
    assert run in store.archived() and run in store.reset_runs() and run not in store.withdrawn()
    before = _four(_mark(store, T0 + timedelta(minutes=2)))
    assert supervisor.withdraw_archived(store, "operator", "abc1234") == []
    assert _flows(store) == [] and store.withdrawn() == {}
    assert _four(_mark(store, T0 + timedelta(minutes=3))) == before



def test_w14_withdraw_archived_keeps_when_an_old_archive_went_so_it_stays_an_earlier_books(store, tmp_path):
    """DA MUST (14 Oct): previous_book reads archived_at against the clean slate's start, so the one-off writes only
    the amount on an archive from before 0023, never a new date."""
    from sqlalchemy import insert

    from sleeve_fund.store import sleeve_archive_t

    _make(store, "a", 15_000)
    _make(store, "b", 5_000)
    went = T0 - timedelta(days=30)
    with store.engine.begin() as c:
        c.execute(insert(sleeve_archive_t).values(sleeve="b", archived_at=went))
    slate = tmp_path / "clear.toml"
    slate.write_text('[[clear]]\nid = "s1"\nreason = "fresh book"\nkeep = ["a"]\n')
    supervisor.clear(store, str(slate))
    assert "b" in store.previous_book()
    _mark(store, T0)
    assert supervisor.withdraw_archived(store, "operator", "abc1234") == ["b"]
    assert store.archived()["b"] == went and "b" in store.previous_book() and store.withdrawn() == {}
    _mark(store, T0 + timedelta(seconds=5))  # the poll takes it out (F286-CR1), keeping when it went
    assert store.archived()["b"] == went and "b" in store.previous_book() and store.withdrawn() == {"b": D(5_000)}


# --- E1-E4: a withdrawal that empties the book (Advisor 14 Oct 10:00 UK, on QD's proposal) ------------------------------

def test_e1_emptying_the_book_leaves_it_unmarked_and_the_next_capital_starts_it_afresh(store):
    """Within a cent of 0, with nothing open and nothing that could trade: the state is left unmarked, a poll of the
    empty book neither marks nor halts it, the next create brings no flow, and its first mark starts the book at it."""
    _make(store, "a", 10_000)
    _trade(store, "a", -1_000)
    _mark(store, T0)
    store.archive("a")
    st = _mark(store, T0 + timedelta(minutes=1))
    assert (st.equity, st.hwm, st.reference(), st.day_start_equity, st.mark_ts) == (None,) * 5 and not st.halted
    assert store.withdrawn() == {"a": D(9_000)} and supervisor.fund_equity(store) == 0
    _make(store, "b", 5_000)
    assert len(_flows(store)) == 1  # b's create on an unmarked book: no flow, the mark reads it
    assert _four(_mark(store, T0 + timedelta(minutes=2))) == (D(5_000),) * 4
    assert not _state(store).halted


def test_e2_the_journal_keeps_the_old_high_water_mark_reference_and_day_start(store):
    _make(store, "a", 10_000)
    _trade(store, "a", 2_000)
    _mark(store, T0)
    _trade(store, "a", -300, T0 + timedelta(minutes=1))
    _mark(store, T0 + timedelta(minutes=1))  # 11,700 against a 12,000 HWM; the day started at 12,000
    store.archive("a")
    _mark(store, T0 + timedelta(minutes=1, seconds=5))
    (flow,) = _flows(store)
    assert "-11,700.00" in flow and "empties it" in flow
    assert "high-water mark 12,000.00" in flow and "reference 12,000.00" in flow and "day's start 12,000.00" in flow


def test_e3_a_halt_in_force_stays_through_the_empty_book_until_the_pm_resumes(store):
    _make(store, "a", 10_000)
    _mark(store, T0)
    led = DbLedger(store, lambda: ())
    with led.lock():
        led.set_state(replace(led.state(), halted="the test's halt"))
    store.archive("a")
    st = _mark(store, T0 + timedelta(minutes=1))
    assert st.equity is None and st.halted == "the test's halt"
    from sleeve_fund.portfolio.gate import resume_after_halt

    _make(store, "b", 5_000)
    assert _mark(store, T0 + timedelta(minutes=2)).halted == "the test's halt"  # capital back: still halted
    assert not resume_after_halt(led).halted  # only the PM's Resume clears it
    assert _four(_mark(store, T0 + timedelta(minutes=3))) == (D(5_000),) * 4


def test_e3b_a_days_pause_keeps_the_book_marked_so_emptying_it_waits_for_midnight(store):
    """A pause follows a mark (the schema's portfolio_state_marked): the PM's retire is done, and the withdrawal that
    would empty the book waits, said once in plain words, with the pause kept (F286-CR1: only the poll withdraws)."""
    _make(store, "a", 10_000)
    _mark(store, T0)
    led = DbLedger(store, lambda: ())
    with led.lock():
        led.set_state(replace(led.state(), paused="the test's pause", paused_until=T0 + timedelta(hours=2)))
    store.set_lifecycle("a", "retire", "pm", "done with it", now=T0 + timedelta(hours=1))
    assert store.sleeve("a").lifecycle == "retired" and store.withdrawal_pending() == ["a"]
    sup = supervisor.Supervisor(store)
    sup.mark_portfolio(T0 + timedelta(hours=1))
    sup.mark_portfolio(T0 + timedelta(hours=1, minutes=1))
    st = _state(store)
    assert st.equity == D(10_000) and st.paused_until == T0 + timedelta(hours=2) and store.withdrawn() == {}
    (waits,) = store.events_of(("withdrawal_waits",))
    assert "paused for the day (the test's pause)" in waits["message"]
    assert "try again after 12:00 UK (11:00 UTC)" in waits["message"]


def test_e3c_a_pause_that_has_ended_doesnt_hold_the_withdrawal_and_goes_with_the_marks(store):
    """DA MUST (14 Oct): paused_until stays on the row after the pause ends; only one still in force refuses."""
    from sqlalchemy import select

    from sleeve_fund.store import portfolio_state_t

    _make(store, "a", 10_000)
    _mark(store, T0)
    led = DbLedger(store, lambda: ())
    with led.lock():
        led.set_state(replace(led.state(), paused="the test's pause", paused_until=T0 + timedelta(hours=2)))
    store.set_lifecycle("a", "retire", "pm", "done with it", now=T0 + timedelta(days=1))  # the pause ended at 00:00
    _poll(store, T0 + timedelta(days=1))
    st = _state(store)
    assert st.equity is None and st.paused_until is None and st.paused is None and store.withdrawn() == {"a": D(10_000)}
    with store.engine.connect() as c:
        row = c.execute(select(portfolio_state_t)).one()
    assert row.status == "ok" and row.marked_at is None and row.book_equity is None and row.hwm is None


@pytest.mark.parametrize("why", ["could trade", "holds 0.5", "below nothing", "an order"])
def test_e4_a_withdrawal_to_empty_is_refused_while_anything_is_open_and_below_nothing_always(store, why):
    from sleeve_fund.portfolio.gate import apply_flow

    _make(store, "a", 10_000)
    _mark(store, T0)
    led = DbLedger(store, lambda: ())
    if why == "could trade":
        store.set_desired_state("a", "running")
    elif why == "holds 0.5":
        store.record_fill("a", side="BUY", qty=0.5, price=1_000.0, fee=0.0, order_id="h", trade_id="h", ts=T0)
    amount = D("-10000.01") if why == "below nothing" else D(-10_000)
    said = {"could trade": "Not done: .* while Strategy a can still trade", "an order": "an order still holds room"}
    with pytest.raises(ValueError, match=said.get(why, why)):
        if why == "an order":
            led.reservations = lambda: ["an entry's order, still resting"]  # what the gate holds room for
            apply_flow(led, amount, lambda: None)
        else:
            with store._book_move(lambda: amount, "the test's withdrawal"):
                pass
    assert _state(store).equity == D(10_000) and _flows(store) == []


def test_e1b_a_residue_under_a_cent_is_empty(store):
    from sleeve_fund.portfolio.gate import apply_flow

    _make(store, "a", 10_000)
    _mark(store, T0)
    led = DbLedger(store, lambda: ())
    assert apply_flow(led, D("-9999.995"), lambda: None).equity == D(10_000) and _state(store).equity is None



# --- H4: an emptied book keeps its drawdown and the day's change for the next capital (MUST-F286-H4) -----------------
# Advisor 14 Oct 13:35 UK; DA's four carried columns on 0023; HoQA's H4 cells.

YESTERDAY = T0 - timedelta(days=1)


def _carried(st):
    return st.carried_hwm_drawdown, st.carried_reference_drawdown, st.carried_day_return, st.carried_day


def _emptied_at_a_loss(store, pnl=-1_400, name="a"):
    """`name` alone in the book, down `pnl` on 10,000 the day before (its pause ended at 00:00), put away and withdrawn
    at T0, so the book is empty with its drawdown carried and the day's change 0 on T0's day."""
    _make(store, name, 10_000)
    _mark(store, YESTERDAY)
    _trade(store, name, pnl, YESTERDAY + timedelta(minutes=1))
    _mark(store, YESTERDAY + timedelta(minutes=1))
    _mark(store, T0)  # a new day starts at the book
    store.archive(name)
    st = _poll(store, T0 + timedelta(seconds=5))
    assert st.equity is None and store.withdrawn() == {name: D(10_000 + pnl)}
    return st


def test_h4a_archive_then_unarchive_keeps_the_14_percent_drawdown(store):
    st = _emptied_at_a_loss(store)
    assert _carried(st) == (D("0.14"), None, D(0), T0.date())
    assert "keeps its drawdown of 14.0000%" in _flows(store)[0]
    store.unarchive("a")
    st = _mark(store, T0 + timedelta(minutes=1))
    assert _four(st) == (D(8_600), D(10_000), D(10_000), D(8_600)) and _dd(st) == D("0.14")
    assert _carried(st) == (None,) * 4


def test_h4b_another_strategys_capital_keeps_the_drawdown_too(store):
    _emptied_at_a_loss(store, pnl=-1_000)
    _make(store, "b", 9_000)
    st = _mark(store, T0 + timedelta(minutes=1))
    assert _four(st) == (D(9_000), D(10_000), D(10_000), D(9_000)) and _carried(st) == (None,) * 4


def test_h4c_the_days_change_is_kept_on_the_same_day_and_a_new_day_starts_at_the_capital(store):
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", -200, T0 + timedelta(minutes=1))
    _mark(store, T0 + timedelta(minutes=1))  # 2% down on the day: no pause
    store.archive("a")
    assert _carried(_poll(store, T0 + timedelta(minutes=1, seconds=5))) == (D("0.02"), None, D("-0.02"), T0.date())
    _make(store, "b", 4_900)
    st = _mark(store, T0 + timedelta(minutes=2))
    assert (st.hwm, st.day_start_equity) == (D(5_000), D(5_000)) and st.paused_until is None
    _trade(store, "b", -60, T0 + timedelta(minutes=3))  # 1.2% more: 3.12% down on the day, so the day's pause
    st = _mark(store, T0 + timedelta(minutes=3))
    assert st.paused_until is not None and _day(st) == D("0.032")


def test_h4c2_crossing_midnight_while_empty_the_day_starts_at_the_capital_and_the_drawdown_stays(store):
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", -200, T0 + timedelta(minutes=1))
    _mark(store, T0 + timedelta(minutes=1))
    store.archive("a")
    _poll(store, T0 + timedelta(minutes=1, seconds=5))
    _make(store, "b", 4_900)
    st = _mark(store, T0 + timedelta(days=1))
    assert _four(st) == (D(4_900), D(5_000), D(5_000), D(4_900)) and st.day == (T0 + timedelta(days=1)).date()


def test_h4d_a_book_reset_while_empty_drops_the_carried_ratios(store):
    _emptied_at_a_loss(store)
    store.book_reset("a fresh start")
    assert _carried(_state(store)) == (None,) * 4
    _make(store, "b", 5_000)
    assert _four(_mark(store, T0 + timedelta(minutes=1))) == (D(5_000),) * 4


def test_h4e_a_clean_slates_emptying_carries_nothing(store, tmp_path):
    _make(store, "a", 10_000)
    _mark(store, YESTERDAY)
    _trade(store, "a", -1_000, YESTERDAY + timedelta(minutes=1))
    _mark(store, YESTERDAY + timedelta(minutes=1))
    _mark(store, T0)
    slate = tmp_path / "clear.toml"
    slate.write_text('[[clear]]\nid = "s1"\nreason = "fresh book"\n')
    assert supervisor.clear(store, str(slate)) == ["a"]
    st = _poll(store, T0 + timedelta(seconds=5))
    assert st.equity is None and _carried(st) == (None,) * 4 and "starts it afresh" in _flows(store)[0]
    _make(store, "b", 5_000)
    assert _four(_mark(store, T0 + timedelta(minutes=1))) == (D(5_000),) * 4


def test_h4f_a_fresh_stack_carries_nothing(store):
    _make(store, "a", 10_000)
    st = _mark(store, T0)
    assert _four(st) == (D(10_000),) * 4 and _carried(st) == (None,) * 4


def test_h4g_da_pin_10_percent_down_emptied_then_capital_c_marks_its_hwm_at_c_over_0_9(store):
    _emptied_at_a_loss(store, pnl=-1_000)
    _make(store, "b", 7_777)
    st = _mark(store, T0 + timedelta(minutes=1))
    assert abs(st.hwm - D(7_777) / D("0.9")) < D("1e-9") and st.halt_reference is None  # SQLite keeps ~17 digits


def test_h4h_advisor_pin_capital_into_an_emptied_book_trades_only_after_its_mark_against_the_kept_hwm(store):
    from sleeve_fund.portfolio.gate import entry_block

    _emptied_at_a_loss(store, pnl=-1_000)
    _make(store, "b", 9_000)
    led = DbLedger(store, lambda: ())
    assert entry_block(led, T0 + timedelta(seconds=30))[0] == "portfolio_state_stale"  # unmarked: no entry yet
    st = _mark(store, T0 + timedelta(minutes=1))
    assert entry_block(led, T0 + timedelta(minutes=1)) is None and st.hwm == D(10_000)
    _trade(store, "b", -500, T0 + timedelta(minutes=2))  # 15% under the kept HWM of 10,000, not 5.6% under 9,000
    assert _mark(store, T0 + timedelta(minutes=2)).halted


def test_h4i_a_halt_reference_below_the_hwm_is_kept_as_its_own_drawdown(store):
    from sleeve_fund.portfolio.gate import resume_after_halt

    _make(store, "a", 10_000)
    _mark(store, YESTERDAY)
    led = DbLedger(store, lambda: ())
    _trade(store, "a", -1_600, YESTERDAY + timedelta(minutes=1))
    assert _mark(store, YESTERDAY + timedelta(minutes=1)).halted
    resume_after_halt(led)  # the reference re-based to 8,400
    _trade(store, "a", -420, YESTERDAY + timedelta(minutes=2))
    _mark(store, YESTERDAY + timedelta(minutes=2))  # 7,980: 5% under the reference, 20.2% under the HWM
    _mark(store, T0)
    store.archive("a")
    st = _poll(store, T0 + timedelta(seconds=5))
    assert _carried(st)[:2] == (D("0.202"), D("0.05"))
    _make(store, "b", 1_000)
    st = _mark(store, T0 + timedelta(minutes=1))
    assert abs(st.halt_reference - D(1_000) / D("0.95")) < D("1e-9") and not st.halted


def test_h4j_a_poll_of_an_emptied_book_that_reads_nothing_restores_nothing(store):
    _emptied_at_a_loss(store)
    st = _mark(store, T0 + timedelta(minutes=1))
    assert st.equity is None and _carried(st) == (D("0.14"), None, D(0), T0.date())


def test_h4k_a_mark_reading_nothing_while_something_could_trade_leaves_the_carried_book_alone(store, monkeypatch):
    """The F282-3 no-op needs nothing open; a carried empty book is never restored against ~0 either way."""
    _emptied_at_a_loss(store)
    _make(store, "b", 5_000)
    store.set_desired_state("b", "running")
    monkeypatch.setattr(supervisor, "fund_equity", lambda s: D(0))
    st = _mark(store, T0 + timedelta(minutes=1))
    assert st.equity is None and _carried(st) == (D("0.14"), None, D(0), T0.date())
    assert not store.events_of(("supervisor_error",))

@pytest.mark.parametrize("bad", ["hwm", "reference", "day"])
def test_h4l_carried_ratios_that_cant_be_restored_fail_closed_and_alert_once(store, monkeypatch, bad):
    """Advisor MUST (14 Oct 14:45 UK): restored only while 1 - d > 0 and 1 + r > 0 (the CHECKs keep them so; this is
    the guard behind them): otherwise the book stays unmarked, entries blocked, and one alert names it."""
    from sleeve_fund.portfolio.gate import entry_block

    _emptied_at_a_loss(store)
    _make(store, "b", 5_000)
    sup = supervisor.Supervisor(store)
    real = sup.ledger.state
    field = {"hwm": "carried_hwm_drawdown", "reference": "carried_reference_drawdown", "day": "carried_day_return"}
    value = D(-1) if bad == "day" else D(1)
    monkeypatch.setattr(sup.ledger, "state", lambda: replace(real(), **{field[bad]: value}))
    sup.mark_portfolio(T0 + timedelta(minutes=1))
    sup.mark_portfolio(T0 + timedelta(minutes=2))
    st = _state(store)
    assert st.equity is None and st.hwm is None
    assert entry_block(DbLedger(store, lambda: ()), T0 + timedelta(minutes=2))[0] == "portfolio_state_stale"
    (told,) = store.events_of(("portfolio_carried_invalid",))
    assert "can't be restored" in told["message"] and "entries stay blocked" in told["message"]
    assert {"hwm": "high-water mark 100.00%", "reference": "reference 100.00%", "day": "change -100.00%"}[bad] in told["message"]


def test_h4m_a_withdrawal_waiting_for_a_days_pause_runs_after_the_old_days_last_mark_and_the_next_day_starts_at_c(store):
    """Advisor 14 Oct 14:45 UK (the 8 Oct order): the pause ends at 00:00 UTC; that poll's mark is the old day's last,
    then the waiting withdrawal; the next capital's first mark on the new day starts the day at it (r = 0)."""
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", -400, T0 + timedelta(minutes=1))
    assert _mark(store, T0 + timedelta(minutes=1)).paused_until is not None  # 4% down on the day
    store.set_lifecycle("a", "retire", "pm", "done with it", now=T0 + timedelta(minutes=2))
    sup = supervisor.Supervisor(store)
    sup.mark_portfolio(T0 + timedelta(minutes=3))
    assert store.withdrawn() == {} and store.events_of(("withdrawal_waits",))
    midnight = (T0 + timedelta(days=1)).replace(hour=0, minute=0)
    sup.mark_portfolio(midnight)
    st = _state(store)
    assert store.withdrawn() == {"a": D(9_600)} and st.equity is None and st.carried_hwm_drawdown == D("0.04")
    _make(store, "b", 4_800)
    st = _mark(store, midnight + timedelta(minutes=1))
    assert (st.hwm, st.day_start_equity, st.day) == (D(5_000), D(4_800), midnight.date()) and st.paused_until is None

# --- D1-D8: archived while not flat, withdrawn once flat (Advisor 14 Oct 10:45 UK, option b) -----------------------------

def _open(store, name, qty=0.5, price=1_000.0, ts=T0):
    store.record_fill(name, side="BUY", qty=qty, price=price, fee=0.0, order_id=f"{name}-in", trade_id=f"{name}-in",
                      ts=ts)
    store.record_equity(name, equity=float(store.sleeve(name).starting_balance), cash=0.0, qty=qty, price=price,
                        benchmark=0.0, ts=ts)


def _close(store, name, qty=0.5, price=1_000.0, fee=0.0, ts=T0):
    store.record_fill(name, side="SELL", qty=qty, price=price, fee=fee, order_id=f"{name}-out",
                      trade_id=f"{name}-out", ts=ts)
    _settle(store, name, ts)


def _settle(store, name, ts):
    cash = float(store.journal_book(name, store.sleeve(name).starting_balance)["cash"])
    store.record_equity(name, equity=cash, cash=cash, qty=0.0, price=1_000.0, benchmark=cash, ts=ts)


def _poll(store, now):
    return _mark(store, now)


def test_d1_archiving_a_holder_moves_nothing_and_it_counts_in_full(store):
    _make(store, "a", 20_000)
    _make(store, "b", 10_000)
    _open(store, "b")
    before = _four(_mark(store, T0))
    store.archive("b")
    assert _four(_state(store)) == before and _flows(store) == [] and store.withdrawn() == {}
    assert store.withdrawal_pending() == ["b"] and "b" in store.archived() and store.nothing_withdrawn() == {}
    assert supervisor.fund_equity(store) == D(30_000)
    st = _poll(store, T0 + timedelta(minutes=1))  # still holding: the poll takes nothing out
    assert st.equity == D(30_000) and _flows(store) == [] and store.withdrawal_pending() == ["b"]


def test_d2_d5_once_flat_one_withdrawal_of_its_settled_equity_after_the_flattens_loss_is_marked(store):
    """d2: one flow, the settled 9,950. d5: the poll marks the 50 lost first (29,950 against the 30,000 high-water
    mark), then scales: had the flow come first, the HWM would read 20,050."""
    _make(store, "a", 20_000)
    _make(store, "b", 10_000)
    _open(store, "b")
    _mark(store, T0)
    store.archive("b")
    _close(store, "b", price=900.0, ts=T0 + timedelta(minutes=1))  # 50 lost on the flatten
    st = _poll(store, T0 + timedelta(minutes=1))
    assert store.withdrawn() == {"b": D(9_950)} and store.withdrawal_pending() == []
    (flow,) = _flows(store)
    assert "-9,950.00" in flow and "30,000.00 -> 20,033.39" in flow
    assert st.equity == D(20_000) and st.hwm.quantize(D("0.01")) == D("20033.39")
    assert float((st.hwm - st.equity) / st.hwm) == pytest.approx(50 / 30_000)
    _poll(store, T0 + timedelta(minutes=2))
    assert len(_flows(store)) == 1


def test_d3_a_restart_between_going_flat_and_the_poll_withdraws_once(store):
    _make(store, "a", 20_000)
    _make(store, "b", 10_000)
    _open(store, "b")
    _mark(store, T0)
    store.archive("b")
    _close(store, "b", ts=T0 + timedelta(minutes=1))
    for minute in (1, 2, 3):  # each poll a fresh supervisor, as after a restart
        _poll(store, T0 + timedelta(minutes=minute))
    assert len(_flows(store)) == 1 and store.withdrawn() == {"b": D(10_000)}


def test_d4_the_poll_racing_withdraw_archived_withdraws_once(store):
    _make(store, "a", 20_000)
    _make(store, "b", 10_000)
    _open(store, "b")
    _mark(store, T0)
    store.archive("b")
    assert supervisor.withdraw_archived(store, "operator", "abc1234") == []  # holding: it waits for the poll
    _close(store, "b", ts=T0 + timedelta(minutes=1))
    assert supervisor.withdraw_archived(store, "operator", "abc1234") == []  # waiting already: the poll's (F286-CR1)
    _poll(store, T0 + timedelta(minutes=1))
    with store._put_away("b", T0 + timedelta(minutes=2), "the poll, late"):  # the loser of the race, under the lock
        pass
    assert len(_flows(store)) == 1 and store.withdrawn() == {"b": D(10_000)}


def test_d6_a_settled_equity_of_nothing_or_less_is_never_withdrawn_and_is_journaled(store):
    _make(store, "a", 20_000)
    _make(store, "b", 1_000)
    _open(store, "b", qty=1.0)
    _mark(store, T0)
    store.archive("b")
    _close(store, "b", qty=1.0, price=1.0, fee=5.0, ts=T0 + timedelta(minutes=1))  # 1,000 - 999 - 5: -4
    st = _poll(store, T0 + timedelta(minutes=1))
    assert _flows(store) == [] and store.withdrawn() == {} and store.withdrawal_pending() == []
    said = [e["message"] for e in store.events_of(("withdrawal_none",))]
    assert len(said) == 1 and "-4.00" in said[0] and "never negative" in said[0]
    assert st.equity == D(19_996)  # it stays in the book at that
    assert store.nothing_withdrawn() == {"b": D(-4)}
    _poll(store, T0 + timedelta(minutes=2))
    assert len(store.events_of(("withdrawal_none",))) == 1


def test_d7_emptying_the_book_under_a_pause_waits_for_midnight(store):
    _make(store, "b", 10_000)
    _open(store, "b")
    _mark(store, T0)
    store.archive("b")
    led = DbLedger(store, lambda: ())
    with led.lock():
        led.set_state(replace(led.state(), paused="the test's pause", paused_until=T0 + timedelta(hours=2)))
    _close(store, "b", ts=T0 + timedelta(minutes=1))
    sup = supervisor.Supervisor(store)  # one running supervisor: it says the wait once
    sup.mark_portfolio(T0 + timedelta(minutes=1))
    sup.mark_portfolio(T0 + timedelta(minutes=2))
    assert _flows(store) == [] and store.withdrawal_pending() == ["b"]
    waits = store.events_of(("withdrawal_waits",))
    assert len(waits) == 1 and "try again after 12:00 UK (11:00 UTC)" in waits[0]["message"]
    sup.mark_portfolio(T0 + timedelta(days=1))
    assert len(_flows(store)) == 1 and _state(store).equity is None and store.withdrawn() == {"b": D(10_000)}


def test_d8_a_withdrawn_strategy_holding_blocks_every_entry_and_raises_an_alert(store):
    from sleeve_fund.paper import portfolio_gate

    _make(store, "a", 20_000)
    _make(store, "b", 10_000)
    _mark(store, T0)
    store.archive("b")
    _poll(store, T0 + timedelta(seconds=5))
    _open(store, "b", ts=T0 + timedelta(minutes=1))  # broken invariant: a fill after its equity left
    with pytest.raises(ValueError, match="every entry is refused"):
        portfolio_gate.fund_holdings(store, atr_pct=lambda s: 0.02)
    _poll(store, T0 + timedelta(minutes=1))
    _poll(store, T0 + timedelta(minutes=2))
    alerts = store.events_of(("withdrawn_holds",))
    assert len(alerts) >= 1 and all(e["level"] == "error" for e in alerts)


def test_d8b_dust_left_after_a_withdrawal_blocks_nothing_and_is_named_while_a_real_position_blocks(store):
    """Advisor 11:10 UK: the invariant uses the withdrawal's own flat (Store.lifecycle_flat), so dust is no breach."""
    from sleeve_fund.paper import portfolio_gate

    _make(store, "a", 20_000)
    _make(store, "b", 10_000)
    _open(store, "b")
    _mark(store, T0)
    store.archive("b")
    store.record_fill("b", side="SELL", qty=0.4995, price=1_000.0, fee=0.0, order_id="b-out", trade_id="b-out",
                      ts=T0 + timedelta(minutes=1))  # 0.0005 left at 1,000: 0.50 of notional, under any order
    _settle(store, "b", T0 + timedelta(minutes=1))
    sup = supervisor.Supervisor(store)
    sup.mark_portfolio(T0 + timedelta(minutes=1))
    (flow,) = _flows(store)
    assert "b" in store.withdrawn() and "dust of 0.0005" in flow
    assert portfolio_gate.fund_holdings(store, atr_pct=lambda s: 0.02) == ()
    sup.mark_portfolio(T0 + timedelta(minutes=2))
    assert store.events_of(("withdrawn_holds",)) == []
    store.record_fill("b", side="BUY", qty=0.5, price=1_000.0, fee=0.0, order_id="b-again", trade_id="b-again",
                      ts=T0 + timedelta(minutes=3))
    with pytest.raises(ValueError, match="every entry is refused"):
        portfolio_gate.fund_holdings(store, atr_pct=lambda s: 0.02)
    sup.mark_portfolio(T0 + timedelta(minutes=3))
    assert len(store.events_of(("withdrawn_holds",))) == 1


# --- F286-1 (HoQA): the Strategy being put away is not what holds the book open ---------------------------------------


def test_f286_1a_the_pm_retiring_the_only_strategy_while_it_runs_empties_the_book(store):
    """Retired, it is stopped, but its process may not have exited by the poll: its own "could trade" never holds
    open the book its withdrawal empties (supervisor.nothing_open's leaving)."""
    _make(store, "a", 10_000)
    store.set_desired_state("a", "running")
    _mark(store, T0)
    store.set_lifecycle("a", "retire", "pm", "done with it", now=T0 + timedelta(minutes=1))
    store.set_status("a", "running")  # its process hasn't seen the stop yet
    assert store.sleeve("a").lifecycle == "retired" and store.withdrawal_pending() == ["a"]
    st = _poll(store, T0 + timedelta(minutes=1))
    assert store.withdrawn() == {"a": D(10_000)} and st.equity is None and len(_flows(store)) == 1
    assert store.events_of(("withdrawal_waits",)) == []


def test_f286_1b_a_winding_down_strategy_going_flat_retires_and_the_poll_empties_the_book(store):
    _make(store, "a", 10_000)
    store.set_desired_state("a", "running")
    _mark(store, T0)
    store.set_lifecycle("a", "wind_down", "pm", "done with it", now=T0)
    assert store.complete_retire("a", T0 + timedelta(minutes=1)) is True
    store.set_status("a", "running")  # its process hasn't seen the stop yet
    assert store.sleeve("a").lifecycle == "retired" and store.withdrawal_pending() == ["a"]
    assert _poll(store, T0 + timedelta(minutes=1)).equity is None and store.withdrawn() == {"a": D(10_000)}


def test_f286_1c_another_strategy_that_can_still_trade_holds_the_empty_book_and_says_so_plainly(store):
    """Only what isn't leaving holds the book open: here a test-account Strategy, outside the book, still enabled."""
    _make(store, "a", 10_000)
    _make(store, "t", 5_000, instrument="BTC/USD", venue="KRAKEN")  # on "paper", its venue's test account
    store.set_desired_state("t", "running")
    _mark(store, T0)
    store.set_lifecycle("a", "retire", "pm", "done with it", now=T0 + timedelta(minutes=1))
    _poll(store, T0 + timedelta(minutes=1))
    assert store.withdrawn() == {} and _state(store).equity == D(10_000) and store.withdrawal_pending() == ["a"]
    (waits,) = store.events_of(("withdrawal_waits",))
    assert ("Not done: this would leave the fund with no capital while Strategy t can still trade. Nothing is lost by "
            "waiting: try again once that has changed.") in waits["message"]


def test_f286_1d_a_winding_down_strategy_going_flat_under_the_days_pause_retires_and_waits_for_midnight(store):
    """The bar close's retire once flat never meets the pause: it only queues the withdrawal (F286-CR1), which the
    poll makes once the pause has ended."""
    _make(store, "a", 10_000)
    _mark(store, T0)
    store.set_lifecycle("a", "wind_down", "pm", "done with it", now=T0)
    led = DbLedger(store, lambda: ())
    with led.lock():
        led.set_state(replace(led.state(), paused="the test's pause", paused_until=T0 + timedelta(hours=2)))
    assert store.complete_retire("a", T0 + timedelta(hours=1)) is True and store.sleeve("a").lifecycle == "retired"
    _poll(store, T0 + timedelta(hours=1))
    assert store.withdrawn() == {} and store.withdrawal_pending() == ["a"]
    _poll(store, T0 + timedelta(days=1))
    assert store.withdrawn() == {"a": D(10_000)} and _state(store).equity is None


# --- F286-CR1 (CR): the withdrawal is taken at the poll's own mark, so P&L since the last mark is judged first --------


def test_f286_cr1a_a_loss_since_the_last_mark_lands_on_the_book_it_was_made_in_not_on_the_strategies_left(store):
    """b loses 400 after the mark and is archived. Taken against the stale mark, the remaining book would read a
    3.8% day's loss and pause; judged at the poll's own mark first, the day's change is the fund's true 2%."""
    _make(store, "a", 10_000)
    _make(store, "b", 10_000)
    _mark(store, T0)
    _trade(store, "b", -400, T0 + timedelta(seconds=30))
    store.archive("b")
    st = _poll(store, T0 + timedelta(minutes=1))
    assert store.withdrawn() == {"b": D(9_600)} and st.equity == D(10_000)
    assert _day(st) == pytest.approx(D("0.02"), abs=D("1e-12")) and st.paused_until is None and not st.halted
    st = _mark(store, T0 + timedelta(minutes=2))
    assert _day(st) == pytest.approx(D("0.02"), abs=D("1e-12")) and st.paused_until is None and not st.halted


def test_f286_cr1b_the_last_strategy_archived_after_a_loss_never_falsely_halts_the_fund(store):
    """Taken against the stale mark, 500 would be left in the book and the next mark would read 0 against it: a false
    halt. Judged first, the 5% day's loss is real and pauses the fund, and the withdrawal that empties the book waits
    for the pause to end (E3b)."""
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", -500, T0 + timedelta(seconds=30))
    store.archive("a")
    st = _poll(store, T0 + timedelta(minutes=1))
    assert st.equity == D(9_500) and not st.halted and st.paused_until is not None and store.withdrawn() == {}
    st = _poll(store, T0 + timedelta(days=1))
    assert store.withdrawn() == {"a": D(9_500)} and st.equity is None and not st.halted


def test_f286_cr1c_a_gain_since_the_last_mark_keeps_the_days_change_and_never_inflates_the_book(store):
    """b gains 1,000 after the mark and is archived: the fund was 5% up on the day, and a's book keeps that 5%, not the
    11% a stale-mark withdrawal would leave it."""
    _make(store, "a", 10_000)
    _make(store, "b", 10_000)
    _mark(store, T0)
    _trade(store, "b", 1_000, T0 + timedelta(seconds=30))
    store.archive("b")
    st = _poll(store, T0 + timedelta(minutes=1))
    assert store.withdrawn() == {"b": D(11_000)} and st.equity == D(10_000) and st.hwm == D(10_000)
    assert st.equity / st.day_start_equity - 1 == pytest.approx(D("0.05"), abs=D("1e-12"))


def test_f286_cr1d_the_last_strategy_archived_after_a_gain_is_not_refused(store):
    """Taken against the stale mark, 10,500 out of a book of 10,000 is below nothing and refused."""
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", 500, T0 + timedelta(seconds=30))
    store.archive("a")
    st = _poll(store, T0 + timedelta(minutes=1))
    assert store.withdrawn() == {"a": D(10_500)} and st.equity is None and not st.halted
    assert store.events_of(("withdrawal_waits",)) == []


# --- F286-2 (CR, HoQA, DA): dust no order can close doesn't block withdraw-archived ----------------------------------


def test_f286_2_withdraw_archived_takes_old_dust_out_and_refuses_only_a_real_holding_by_name(store):
    """Two archives from before 0023: d holds dust (0.0005 at 1,000: 0.50 of notional, under any order), h a real
    position. Refused, nothing written, naming h and its size; once h is flat both go, d's dust with it (Advisor D8b),
    each keeping when it was archived."""
    _make(store, "a", 20_000)
    _make(store, "d", 5_000)
    _make(store, "h", 5_000)
    store.record_fill("d", side="BUY", qty=0.0005, price=1_000.0, fee=0.0, order_id="d1", trade_id="d1", ts=T0)
    _settle(store, "d", T0)
    _open(store, "h")
    _archived_before(store, "d")
    _archived_before(store, "h")
    went = store.archived()
    _mark(store, T0)
    with pytest.raises(RuntimeError, match=r"refused, nothing written: h holds 0\.5 ") as refused:
        supervisor.withdraw_archived(store, "operator", "abc1234")
    assert "d holds" not in str(refused.value) and store.withdrawal_pending() == []
    _close(store, "h", ts=T0 + timedelta(minutes=1))
    assert sorted(supervisor.withdraw_archived(store, "operator", "abc1234")) == ["d", "h"]
    _poll(store, T0 + timedelta(minutes=1))
    assert set(store.withdrawn()) == {"d", "h"} and len([f for f in _flows(store) if "put away" in f]) == 2
    assert {n: store.archived()[n] for n in ("d", "h")} == {n: went[n] for n in ("d", "h")}
    (dust,) = [f for f in _flows(store) if f.startswith("d put away")]
    assert "dust of 0.0005" in dust


# --- F286-3 (HoQA H6, HoE M2): the open-risk limit's book skips a withdrawn Strategy -----------------------------------


def test_f286_3_the_open_risk_book_is_the_funds_book_once_a_strategy_is_withdrawn(store):
    from sleeve_fund import open_risk

    _make(store, "a", 15_000)
    _make(store, "b", 5_000)
    account = store.account_of("a")
    _mark(store, T0)
    store.archive("b")
    assert open_risk.account_equity(store, account) == 20_000  # waiting for the poll: still in the book
    _poll(store, T0 + timedelta(minutes=1))
    assert store.withdrawn() == {"b": D(5_000)}
    assert open_risk.account_equity(store, account) == 15_000 == float(supervisor.fund_equity(store))
    book, _, _ = open_risk.account_book(store, "a", 15_000.0, lambda s: 0.02)
    assert book == 15_000
