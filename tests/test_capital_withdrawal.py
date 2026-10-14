"""F-C3B-A (spec qd/f-c3b-a-spec.md; Advisor 14 Oct 09:45 UK; HoE OK to invert HALT-REF-1's C2): a Strategy put away
for good (archived, retired, or archived by a clean slate) takes its whole equity out of the fund's book as a flow, and
one brought back puts it in again. A flow scales the book's high-water mark, the halt's reference and the day's start by
(book + flow) / book, so a withdrawal is never a loss and the drawdown is the same before and after. A reset is still no
flow. Strategies archived before this landed count as before until the one-off withdraw-archived command takes each out
with its own flow."""

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
    assert _four(_state(store)) == (D(16_000),) * 4  # 24,000 x 16,000 / 24,000
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
    st = _state(store)
    assert st.equity == D(13_500) and st.hwm == D(15_000)  # 20,000 x 13,500 / 18,000
    assert _dd(st) == _dd(before) and _day(st) == _day(before)
    assert (st.halted, st.paused_until) == (before.halted, before.paused_until)  # the archive alone sets nothing
    after = _mark(store, T0 + timedelta(minutes=3))
    assert after.equity == D(13_500) and _dd(after) == D("0.1") and not after.halted


# --- W3: unarchive is the undo -----------------------------------------------------------------------------------------

def test_w3_unarchiving_deposits_its_equity_then_and_the_round_trip_nets_to_nothing(store):
    _make(store, "a", 15_000)
    _make(store, "b", 5_000)
    _trade(store, "b", 1_000)
    before = _four(_mark(store, T0))
    store.archive("b")
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
    assert _four(_state(store)) == (D(15_000),) * 4 and store.withdrawn() == {"b": D(5_000)}
    # A flow that fails rolls the retire back: the row stays active and nothing is withdrawn.
    _make(store, "c", 5_000)
    import sleeve_fund.portfolio.gate as gate

    monkeypatch.setattr(gate, "apply_flow", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no flow")))
    with pytest.raises(RuntimeError):
        store.set_lifecycle("c", "retire", "pm", "r", now=T0)
    assert store.sleeve("c").lifecycle == "active" and "c" not in store.withdrawn() and "c" not in store.archived()


def test_w5_a_clean_slate_withdraws_each_strategy_it_archives(store, tmp_path):
    _make(store, "a", 10_000)
    _make(store, "b", 10_000)
    _trade(store, "a", 500)
    _mark(store, T0)
    slate = tmp_path / "clear.toml"
    slate.write_text('[[clear]]\nid = "s1"\nreason = "fresh book"\nkeep = ["b"]\n')
    assert supervisor.clear(store, str(slate)) == ["a"]
    assert _four(_state(store)) == (D(10_000),) * 4 and store.withdrawn() == {"a": D(10_500)}
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
    assert sorted(done) == ["b", "c"] and len(_flows(store)) == 2
    assert _four(_state(store)) == (D(15_000),) * 4
    assert supervisor.withdraw_archived(store, "operator", "abc1234") == [] and len(_flows(store)) == 2
    # One archived while still holding (before holders were flattened first): nothing is withdrawn until it is flat.
    _make(store, "d", 1_000)
    store.record_fill("d", side="BUY", qty=0.5, price=1_000.0, fee=0.0, order_id="dh", trade_id="dh", ts=T0)
    _archived_before(store, "d")
    with pytest.raises(RuntimeError, match="d holds"):
        supervisor.withdraw_archived(store, "operator", "abc1234")
    assert "d" not in store.withdrawn() and len(_flows(store)) == 3  # d's create, a deposit; no withdrawal


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
    st = _state(store)
    assert st.equity == D(10_000) and not st.halted
    # 14,000 of gross was 0.7x of 20,000; it is 1.4x of 10,000 now, so a 2,000 entry would be over the 1.5x cap.
    from sleeve_fund.risk import PORTFOLIO

    assert D(14_000) / st.equity + D(2_000) / st.equity > D(str(PORTFOLIO.gross))


def test_w11_a_withdrawn_strategys_capital_is_not_capital_put_in(store):
    _make(store, "a", 15_000)
    _make(store, "b", 5_000)
    _mark(store, T0)
    store.archive("b")
    assert supervisor.capital_in(store) == D(15_000)


def test_the_withdrawal_is_journaled_as_capital_not_pnl(store):
    _make(store, "a", 16_000)
    _make(store, "b", 4_000)
    _mark(store, T0)
    store.archive("b")
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
