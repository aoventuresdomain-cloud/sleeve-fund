"""CAP5-RESET (Advisor, 14 Oct ~21:35 UK; bridge Drop A): the PM's reset of the paper book (reset-book.yml). The clean
slate puts every Strategy away, and only onto an empty book is the fund's fresh deposit written, as a re-base. Cells
(a), (c) and (e) of the ruling, the day's pause, the dry run, and a second reset once the fund holds the book. The
deposit is the DA's fund_flow; the poll that applies it, and what empty means, are PE2's (empty_book_pockets)."""

from datetime import timedelta
from decimal import Decimal as D

import pytest
from test_capital_withdrawal import _archived_before, _carried
from test_halt_ref import T0, _four, _make, _mark, _reset, _trade, store  # noqa: F401

from sleeve_fund import reset_book, supervisor
from sleeve_fund.store import Store

SLATE = "2026-10-21-reset"


def _book(store):
    """A book down from its high-water mark (20,100 against 20,400), so a carried state would show."""
    _make(store, "ping-pong-ls-binance", 10_000)
    _make(store, "rsi-bands-ls-binance", 10_000)
    _mark(store, T0)
    _trade(store, "ping-pong-ls-binance", 400, T0)
    _mark(store, T0 + timedelta(minutes=1))
    _trade(store, "ping-pong-ls-binance", -300, T0 + timedelta(minutes=2))
    st = _mark(store, T0 + timedelta(minutes=3))
    assert st.hwm > st.equity and supervisor.fund_equity(store) == D(20_100)


def _flows(store):
    """The fund's flows as (kind, amount, actor, rebase), in journal order."""
    return [(f["kind"], D(str(f["amount"])), f["actor"], f["rebase"]) for f in store.fund_flows()]


def _old_book(store):
    """The book as the PE1 probe found it, before the fund holds it: two Strategies, one archived before F-C3B-A
    (5,000, still counted), and a reset run holding 250 of cash, the book 150 below its high-water mark (W13: withdraw-archived never takes it)."""
    _make(store, "a", 10_000)
    _make(store, "b", 10_000)
    _make(store, "old", 5_000)
    _archived_before(store, "old")
    _mark(store, T0)
    _trade(store, "b", 400, T0)
    _mark(store, T0 + timedelta(minutes=1))
    _trade(store, "b", -150, T0 + timedelta(minutes=1))
    run = _reset(store, "b")
    st = _mark(store, T0 + timedelta(minutes=2))
    assert supervisor.fund_equity(store) == D(25_250) and st.hwm > st.equity
    return run


def _until_done(store, now):
    """The operator's runs, one after each supervisor mark, as long as it says to run again."""
    for i in range(6):
        code, lines = reset_book.run(store, "operator", "abc1234", SLATE, D(100_000), now=now + timedelta(minutes=i))
        if code != reset_book.WAIT:
            return code, lines
        _mark(store, now + timedelta(minutes=i, seconds=30))
    raise AssertionError(f"still waiting: {lines}")


def test_a1_the_old_book_leaves_whole_and_one_re_base_deposit_is_written(store):
    run = _old_book(store)
    code, lines = _until_done(store, T0 + timedelta(minutes=3))
    assert code == reset_book.DONE and _flows(store) == [("deposit", D(100_000), "pm", True)]
    assert set(store.withdrawn()) == {"a", "b", "old", run} and store.withdrawn()[run] == D(250)
    assert reset_book.strategies(store) == [] and supervisor.empty_book_pockets(store) == []
    assert "written as fund flow 1" in lines[-1]


def test_a2_the_pe1_case_ends_at_100000_fresh(store):
    _old_book(store)
    code, _ = _until_done(store, T0 + timedelta(minutes=3))
    st = _mark(store, T0 + timedelta(minutes=20))
    assert code == reset_book.DONE and supervisor.fund_equity(store) == D(100_000)
    assert _four(st) == (D(100_000),) * 4 and _carried(st) == (None,) * 4 and not st.halted


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="PE2: fund_book counts a flow applied at the same poll "
                   "as the re-base (applied_at >= since) when it is before it in id order: the fund's withdrawal "
                   "lands in the new book (500 instead of 100,000)")
def test_a3_a_second_reset_once_the_fund_holds_the_book_starts_it_afresh_again(store):
    _old_book(store)
    _until_done(store, T0 + timedelta(minutes=3))
    _mark(store, T0 + timedelta(minutes=20))
    assert supervisor.bridge_start(store) is not None
    _make(store, "c", 0)
    _trade(store, "c", -500, T0 + timedelta(minutes=21))
    _mark(store, T0 + timedelta(minutes=22))
    for i in range(6):
        code, lines = reset_book.run(store, "operator", "abc1234", "second", D(100_000), now=T0 + timedelta(minutes=23 + i))
        if code != reset_book.WAIT:
            break
        _mark(store, T0 + timedelta(minutes=23 + i, seconds=30))
    st = _mark(store, T0 + timedelta(minutes=40))
    assert code == reset_book.DONE and [(f[0], f[3]) for f in _flows(store)] == [
        ("deposit", True), ("withdraw", False), ("deposit", True)]  # the fund's cash leaves ahead of the fresh deposit
    assert _flows(store)[1][1] == D(99_500) and all(f["applied_at"] is not None for f in store.fund_flows())
    assert supervisor.fund_equity(store) == D(100_000) and _four(st) == (D(100_000),) * 4 and _carried(st) == (None,) * 4


def test_c1_a_reset_run_that_lost_is_a_pocket_no_withdrawal_takes_so_it_is_refused_by_name(store):
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", -120, T0)
    run = _reset(store, "a")
    before = (store.decisions(limit=10_000), store.archived())
    code, lines = reset_book.run(store, "operator", "abc1234", SLATE, D(100_000), now=T0 + timedelta(minutes=1))
    assert code == reset_book.REFUSED and _flows(store) == [] and (store.decisions(limit=10_000), store.archived()) == before
    assert lines == [f"refused, nothing written: the book can't be emptied while {run} -120.00, a loss no withdrawal "
                     "takes out"]


def test_c1b_a_pocket_still_in_the_book_at_the_deposit_is_refused_by_name(store, monkeypatch):
    _book(store)
    monkeypatch.setattr(supervisor, "empty_book_pockets", lambda st: ["left-over 250.00"])
    code, lines = _until_done(store, T0 + timedelta(minutes=4))
    assert code == reset_book.REFUSED and _flows(store) == []
    assert lines[-1] == "refused, no deposit written: the book isn't empty; still in it: left-over 250.00"


def test_c2_an_archive_still_holding_is_refused_by_name_and_nothing_is_written(store):
    _make(store, "a", 10_000)
    _make(store, "old", 5_000)
    store.record_fill("old", side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id="o1", trade_id="t1", ts=T0)
    store.record_equity("old", equity=5_000.0, cash=-1_000.0, qty=0.1, price=60_000.0, benchmark=0.0, ts=T0)
    _archived_before(store, "old")
    _mark(store, T0)
    before = (store.decisions(limit=10_000), store.archived())
    code, lines = reset_book.run(store, "operator", "abc1234", SLATE, D(100_000), now=T0 + timedelta(minutes=1))
    assert code == reset_book.REFUSED and _flows(store) == [] and (store.decisions(limit=10_000), store.archived()) == before
    assert lines == ["refused, nothing written: the book can't be emptied while old holds 0.1"]


def test_e_a_restart_between_the_put_aways_and_the_deposit_carries_on_without_repeating_a_step(store):
    _book(store)
    code, lines = reset_book.run(store, "operator", "abc1234", SLATE, D(100_000), now=T0 + timedelta(minutes=4))
    assert code == reset_book.WAIT and any("next mark" in line for line in lines) and _flows(store) == []
    # the supervisor and the operator's shell both restart: a fresh Store on the same database
    again = Store(engine=store.engine, portfolio_gate=True)
    _mark(again, T0 + timedelta(minutes=5))
    code, lines = reset_book.run(again, "operator", "abc1234", SLATE, D(100_000), now=T0 + timedelta(minutes=6))
    assert code == reset_book.DONE and _flows(again) == [("deposit", D(100_000), "pm", True)]
    assert len(again.decisions(action=reset_book.RESET_ACTION)) == 1  # only the deposit's row
    assert len([d for d in again.decisions(action="clear") if d["reason"].startswith(SLATE)]) == 1
    code, lines = reset_book.run(again, "operator", "abc1234", SLATE, D(100_000), now=T0 + timedelta(minutes=7))
    assert code == reset_book.DONE and lines == [f"reset {SLATE}: deposited already, nothing to do"]
    assert len(_flows(again)) == 1


def test_a_holder_is_flattened_first_and_the_reset_waits_for_it(store):
    _make(store, "a", 10_000)
    store.record_fill("a", side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id="o1", trade_id="t1", ts=T0)
    store.record_equity("a", equity=10_000.0, cash=4_000.0, qty=0.1, price=60_000.0, benchmark=0.0, ts=T0)
    _mark(store, T0)
    code, lines = reset_book.run(store, "operator", "abc1234", SLATE, D(100_000), now=T0 + timedelta(minutes=1))
    assert code == reset_book.WAIT and lines[-1] == "waiting on a flatten before the slate can finish: a holds 0.1"
    assert [c["command"] for c in store.pending_commands("a")] == ["flatten"] and _flows(store) == []


def test_the_days_pause_holds_the_put_aways_until_midnight_and_says_so(store):
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", -400, T0 + timedelta(minutes=1))  # 4% down on the day: the fund's day pause
    st = _mark(store, T0 + timedelta(minutes=1))
    assert st.paused_until is not None
    code, lines = reset_book.run(store, "operator", "abc1234", SLATE, D(100_000), now=T0 + timedelta(minutes=2))
    assert code == reset_book.WAIT and _flows(store) == []
    assert lines[-1] == ("The fund is paused for the day (the book has lost 4.0% since 00:00 UTC, at the 3% daily "
                         "limit), so what was put away leaves the book only once the pause ends at 01:00 UK (00:00 UTC)."
                         " Nothing is lost by waiting: run the reset again after then.")
    _mark(store, T0 + timedelta(minutes=3))  # still the same day: the poll can't empty it
    assert reset_book.run(store, "operator", "abc1234", SLATE, D(100_000), now=T0 + timedelta(minutes=4))[0] == \
        reset_book.WAIT
    midnight = (T0 + timedelta(days=1)).replace(hour=0, minute=0)
    code, _ = _until_done(store, midnight + timedelta(seconds=5))
    assert code == reset_book.DONE and _flows(store) == [("deposit", D(100_000), "pm", True)]


def test_the_dry_run_names_what_it_would_do_and_writes_nothing(store, monkeypatch, capfd):
    _book(store)
    store.record_fill("rsi-bands-ls-binance", side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id="o1",
                      trade_id="t1", ts=T0 + timedelta(minutes=3))
    store.record_equity("rsi-bands-ls-binance", equity=10_000.0, cash=4_000.0, qty=0.1, price=60_000.0, benchmark=0.0,
                        ts=T0 + timedelta(minutes=3))
    monkeypatch.setattr(supervisor, "Store", lambda **k: store)
    before = (store.decisions(limit=10_000), store.withdrawn(), store.archived(), store.pending_commands("rsi-bands-ls-binance"))
    args = ["reset-book", "--actor", "operator", "--commit", "abc1234", "--id", SLATE, "--dry-run"]
    assert supervisor.main(args) == reset_book.DONE
    out = capfd.readouterr().out.splitlines()
    assert (store.decisions(limit=10_000), store.withdrawn(), store.archived(), store.pending_commands("rsi-bands-ls-binance")) == before
    assert _flows(store) == [] and out[:3] == ["dry run, nothing written:", f"reset {SLATE}: not run", "the book now: 20,100.00"]
    assert sorted(out[3].removeprefix("would put away: ").split(", ")) == ["ping-pong-ls-binance", "rsi-bands-ls-binance"]
    assert out[4:] == ["would flatten first: rsi-bands-ls-binance holds 0.1",
                       "then deposit 100,000.00 into the fund's cash, once the book is empty, as its fresh start"]


def test_the_dry_run_names_what_leaves_the_old_book(store):
    run = _old_book(store)
    lines = reset_book.plan(store, SLATE, D(100_000))
    (taken,) = [line for line in lines if line.startswith("would take out of the old book")]
    assert sorted(taken.split(": ", 1)[1].split(", ")) == sorted(["old 5,000.00", f"{run} 250.00"])
