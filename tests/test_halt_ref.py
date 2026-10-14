"""HALT-REF-1 (HoE 14 Oct 08:53 UK, a MUST before the first test; HoQA plan quant-review/runs/halt-ref-1, cells C1-C12):
every capital move into or out of the fund's book goes through the flow path (portfolio.gate.apply_flow) in the same
transaction as the move, so the book's high-water mark, the halt's reference and the day's start move with it and a
deposit is never a gain nor a withdrawal a loss. The flows follow what fund_equity counts, so the book and its reference
always move together: a create (+ its capital) and a move onto or off a test account (F229-6, -/+ its equity) move the
book; a reset (its run keeps its cash in the book, its successor the capital) and an archive (an archived Strategy's
cash is still the fund's) move nothing, so neither has a flow. And the state paper halted on is put back on the true
book once, by the operator's one-off command (supervisor rebase), under the Advisor's conditions (14 Oct 09:00, 09:10).

The paper incident: Strategies created after the first mark raised the high-water mark as if they were gains, and test
accounts later left the book (F229-6) with no flow, so the book read 81.8% under its reference and halted."""

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pytest
from sqlalchemy import select

from sleeve_fund import supervisor
from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.store import (BACKTEST_PREFIX, Store, decisions_t, equity_t, fills_t, metadata, orders_t,
                               portfolio_state_t, sleeves_t)

T0 = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "sqlite":
        s = Store(f"sqlite:///{tmp_path}/h.db", portfolio_gate=True)
    else:
        from sleeve_fund.store import make_engine

        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("Postgres's advisory lock and NUMERIC: CI runs these against it (TEST_DATABASE_URL)")
        engine = make_engine(url)
        metadata.drop_all(engine)
        s = Store(engine=engine, portfolio_gate=True)
    s.create_account("desk", "paper")  # a paper account no venue names as a test account: in the book
    return s


def _make(store, name, balance, *, instrument="BTC/USDT", venue="BINANCE"):
    store.create_sleeve(name=name, strategy="ping_pong", instrument=instrument, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=balance, venue=venue, desired_state="stopped")


def _trade(store, name, pnl, ts=T0):
    """A round trip that moves `name`'s cash by `pnl`, fee free, then a mark of it as the journal has it (flat)."""
    n = len(store.fills(name, limit=10_000))
    store.record_fill(name, side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id=f"o{n}", trade_id=f"t{n}", ts=ts)
    store.record_fill(name, side="SELL", qty=1.0, price=1_000.0 + pnl, fee=0.0, order_id=f"o{n + 1}",
                      trade_id=f"t{n + 1}", ts=ts)
    _settle(store, name, ts)


def _settle(store, name, ts=T0):
    cash = float(store.journal_book(name, store.sleeve(name).starting_balance)["cash"])
    store.record_equity(name, equity=cash, cash=cash, qty=0.0, price=1_000.0, benchmark=cash, ts=ts)


def _mark(store, now):
    supervisor.Supervisor(store).mark_portfolio(now)
    return _state(store)


def _state(store):
    return DbLedger(store, lambda: ()).state()


def _four(st):
    return st.equity, st.hwm, st.reference(), st.day_start_equity


def _flows(store):
    return [e["message"] for e in store.events(None, limit=100) if e["kind"] == "portfolio_flow"]


def _incident(store, monkeypatch=None):
    """Paper's state at the halt: the reference far above the book, and the halt it set."""
    led = DbLedger(store, lambda: ())
    with led.lock():
        led.set_state(led.state().__class__(equity=D(19_860), mark_ts=T0, hwm=D("109108.90"), day=T0.date(),
                                            day_start_equity=D(30_000), halted="the book is 81.8% below its reference"))
    if monkeypatch is not None:
        monkeypatch.setattr(supervisor, "Store", lambda **k: store)  # the command's journal is this one


# --- C1, C11: a create is a deposit -----------------------------------------------------------------------------------

def test_c1_a_strategy_created_after_the_first_mark_is_a_deposit_not_a_gain(store):
    _make(store, "a", 20_000)
    assert _four(_mark(store, T0)) == (D(20_000),) * 4
    _make(store, "b", 5_000)
    assert _four(_state(store)) == (D(25_000),) * 4  # moved with the Strategy's row, before any mark
    assert len(_flows(store)) == 1 and "Strategy b created with 5,000.00: the fund's book moves by +5,000.00 and its " \
        "high-water mark, the halt's reference and the day's start scale with it (the reference 20,000.00 -> " \
        "25,000.00)" in _flows(store)[0]
    st = _mark(store, T0 + timedelta(minutes=1))
    assert _four(st) == (D(25_000),) * 4 and not st.halted and st.paused_until is None


def test_c11_a_saved_backtest_made_or_pruned_never_moves_the_book(store):
    """The only delete is prune_backtests, and a backtest is never in the fund's book: neither is a flow."""
    _make(store, "a", 20_000)
    _mark(store, T0)
    _make(store, f"{BACKTEST_PREFIX}run1", 50_000)
    store.prune_backtests(keep=0)
    assert _flows(store) == [] and _four(_mark(store, T0 + timedelta(minutes=1))) == (D(20_000),) * 4


def test_c11_no_path_deletes_a_strategy_in_the_book(store):
    """A delete is a flow only if it moves fund_equity (Advisor, 09:30 UK): the one delete of a Strategy's row is
    prune_backtests, which takes backtests only, and a backtest is never in the book. A source check pins that."""
    import inspect
    import re

    import sleeve_fund.store as store_mod

    src = inspect.getsource(store_mod)
    deletes = [m.start() for m in re.finditer(r"delete\(sleeves_t\)", src)]
    assert len(deletes) == 1 and src.rfind("def ", 0, deletes[0]) == src.index("def prune_backtests")
    _make(store, "a", 20_000)
    _mark(store, T0)
    store.prune_backtests(keep=0)
    assert [s.name for s in store.sleeves()] == ["a"] and supervisor.fund_equity(store) == D(20_000)


def test_before_the_first_mark_there_is_nothing_to_move_and_no_state_is_made(store):
    _make(store, "a", 10_000)
    with store.engine.connect() as c:
        assert c.execute(select(portfolio_state_t)).first() is None
    assert _flows(store) == []


# --- Flows keep percentages (Advisor MUST, 14 Oct 09:45 UK) ----------------------------------------------------------

def _dd(st):
    return 1 - st.equity / st.reference(), 1 - st.equity / st.day_start_equity


def test_a_deposit_on_a_book_10_percent_down_stays_10_percent_down(store):
    """The Advisor's example: 90,000 against 100,000, and 100,000 comes in. Shifted, it would read 190,000 against
    200,000, 5%; scaled by 190/90 the HWM, the reference and the day's start keep it 10% down."""
    _make(store, "a", 100_000)
    _mark(store, T0)
    _trade(store, "a", -10_000)
    before = _mark(store, T0 + timedelta(minutes=1))
    assert before.equity == D(90_000) and _dd(before) == (D("0.1"), D("0.1"))
    _make(store, "b", 100_000)
    st = _state(store)
    assert st.equity == D(190_000) and abs(st.hwm - D("211111.111111111111111111")) < D("1e-9")
    assert st.reference() == st.hwm == st.day_start_equity and all(abs(x - D("0.1")) < D("1e-12") for x in _dd(st))
    st = _mark(store, T0 + timedelta(minutes=2))
    assert all(abs(x - D("0.1")) < D("1e-12") for x in _dd(st)) and not st.halted


def test_a_withdrawal_of_10k_after_a_1k_loss_on_20k_stays_5_percent_down(store):
    """The Advisor's other example, as a test account leaving the book (F229-6): 19,000 against 20,000, 5%, and
    10,000 goes. Shifted it would read 9,000 against 10,000, 10%; scaled by 9/19 it stays 5%."""
    _make(store, "fund", 10_000)
    _make(store, "test", 10_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "desk")
    _mark(store, T0)
    _trade(store, "fund", -1_000)
    assert _mark(store, T0 + timedelta(minutes=1)).equity == D(19_000)
    store.assign_account("test", "paper")
    st = _state(store)
    assert st.equity == D(9_000) and abs(st.hwm - D("9473.684210526315789474")) < D("1e-9")
    assert all(abs(x - D("0.05")) < D("1e-12") for x in _dd(st))
    st = _mark(store, T0 + timedelta(minutes=2))
    assert all(abs(x - D("0.05")) < D("1e-12") for x in _dd(st)) and not st.halted


def test_an_empty_fund_marked_at_0_takes_its_first_strategy_as_its_starting_book(store):
    """A fresh stack's supervisor marks the empty fund at 0 before any Strategy exists: the first create starts the
    book at its capital (no HWM, so no drawdown to keep) rather than failing closed for ever. The halt that empty mark
    set (its figures aren't usable) is the gate's own and stays, as before, for the PM's Resume: no flow clears one."""
    assert _mark(store, T0).equity == D(0)
    _make(store, "a", 10_000)
    st = _state(store)
    assert _four(st) == (D(10_000),) * 4 and "figures aren't usable" in st.halted
    st = _mark(store, T0 + timedelta(minutes=1))
    assert _four(st) == (D(10_000),) * 4 and "figures aren't usable" in st.halted


def test_a_deposit_into_a_book_at_0_below_its_hwm_is_refused(store):
    """A book wiped to 0 under a HWM is a real 100% loss: no ratio keeps it, so the deposit fails closed."""
    _make(store, "a", 10_000)
    _mark(store, T0)
    led = DbLedger(store, lambda: ())
    with led.lock():
        led.set_state(replace(led.state(), equity=D(0)))
    with pytest.raises(ValueError, match="can't keep the book's percentages"):
        _make(store, "b", 5_000)
    assert "b" not in {s.name for s in store.sleeves()}


def test_a_flow_that_leaves_no_book_is_refused_and_the_move_with_it(store):
    """Every Strategy in the book leaving it: book + flow is 0, so there is no ratio to keep, and the move fails
    closed (ValueError) with its flow: the account stays as it was and nothing is journaled."""
    _make(store, "test", 20_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "desk")
    _mark(store, T0)
    with pytest.raises(ValueError, match="can't keep the book's percentages"):
        store.assign_account("test", "paper")
    assert store.account_of("test") == "desk" and _flows(store) == [] and _state(store).equity == D(20_000)


# --- C2, C3: archive and reset move nothing, as fund_equity counts them ------------------------------------------------

def test_c2_an_archived_strategys_cash_stays_in_the_book_so_archiving_is_no_flow(store):
    """fund_equity counts an archived Strategy in full (its cash is still the fund's): archiving one with 4,000 made
    neither moves the book nor its reference, and the next mark is the same book, with no halt."""
    _make(store, "a", 16_000)
    _make(store, "b", 4_000)
    _trade(store, "b", 4_000)
    assert _four(_mark(store, T0)) == (D(24_000),) * 4
    store.archive("b")
    st = _mark(store, T0 + timedelta(minutes=1))
    assert _four(st) == (D(24_000),) * 4 and not st.halted and st.paused_until is None and _flows(store) == []


def _reset(store, name):
    store.request_reset(name, "testing")
    run = store.split_run(store.pending_reset(name))
    assert store.sleeve(run).starting_balance == store.sleeve(name).starting_balance
    return run


def test_c3a_a_reset_after_a_gain_leaves_the_book_where_it_was_at_no_drawdown(store):
    """The run keeps its 1,000 gain in the book and the Strategy restarts at its 10,000: the book is still 11,000, at
    its HWM, so 0% down; there is no flow, as nothing came in or went out."""
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", 1_000)
    assert _four(_mark(store, T0 + timedelta(minutes=1)))[:2] == (D(11_000), D(11_000))
    _reset(store, "a")
    st = _mark(store, T0 + timedelta(minutes=2))
    assert (st.equity, st.hwm, st.reference()) == (D(11_000),) * 3 and _flows(store) == []
    assert supervisor.fund_equity(store) == D(11_000)


def test_c3b_a_reset_after_a_loss_keeps_the_drawdown(store):
    _make(store, "a", 10_000)
    _mark(store, T0)
    _trade(store, "a", -1_000)
    before = _mark(store, T0 + timedelta(minutes=1))
    assert (before.equity, before.hwm) == (D(9_000), D(10_000))  # 10% down, a real loss (and the day's pause)
    _reset(store, "a")
    st = _mark(store, T0 + timedelta(minutes=2))
    assert _four(st) == _four(before) and _flows(store) == []
    assert (st.hwm - st.equity) / st.hwm == D("0.1")


def test_c2_a_clean_slate_archives_without_a_flow(store, tmp_path):
    _make(store, "a", 10_000)
    _trade(store, "a", 500)
    _mark(store, T0)
    slate = tmp_path / "clear.toml"
    slate.write_text('[[clear]]\nid = "s1"\nreason = "fresh book"\n')
    assert supervisor.clear(store, str(slate)) == ["a"]
    assert _four(_mark(store, T0 + timedelta(minutes=1))) == (D(10_500),) * 4 and _flows(store) == []


# --- C4: onto or off a test account ------------------------------------------------------------------------------------

def test_c4_a_strategy_moving_out_of_the_book_is_a_withdrawal_not_a_loss(store):
    """The incident: a Strategy on the test account its venue names leaves the book (F229-6). Without its flow the
    book reads 2/3 down from its reference and halts; with it, nothing halts and the reference follows the book."""
    _make(store, "fund", 10_000)
    _make(store, "test", 20_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "desk")  # into the book from the paper account (KRAKEN's named test account)
    assert _mark(store, T0).equity == D(30_000)
    store.assign_account("test", "paper")  # back onto the test account: out of the book
    assert _four(_state(store)) == (D(10_000),) * 4
    assert "test moved out of the fund's book" in _flows(store)[0] and "-20,000.00" in _flows(store)[0]
    st = _mark(store, T0 + timedelta(minutes=1))
    assert st.equity == D(10_000) and not st.halted and st.paused_until is None


def test_c4_a_strategy_moving_into_the_book_is_a_deposit_at_its_equity_then(store):
    _make(store, "fund", 10_000)
    _make(store, "test", 5_000, instrument="BTC/USD", venue="KRAKEN")  # on the paper account: outside the book
    _trade(store, "test", 250)
    _mark(store, T0)
    store.assign_account("test", "desk")
    assert _four(_state(store)) == (D(15_250),) * 4
    assert "test moved into the fund's book" in _flows(store)[0] and "+5,250.00" in _flows(store)[0]


def test_c4_a_move_that_stays_in_the_book_or_a_dashboard_create_onto_a_test_account_nets_to_nothing(store):
    _make(store, "fund", 10_000)
    _mark(store, T0)
    store.assign_account("fund", "desk")  # paper to paper, a quote held to 8 places: in the book on both
    assert _flows(store) == []
    # The dashboard creates on the paper account, then assigns the one picked: here the test account it was on.
    _make(store, "test", 5_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "paper")
    assert _state(store).equity == _state(store).hwm == D(10_000) and _flows(store) == []


# --- C5, C6, C7: a flow never masks a loss, never races a mark, never applies twice -----------------------------------

def test_c5_a_drop_of_15_percent_from_the_flowed_hwm_halts(store):
    _make(store, "a", 20_000)
    _mark(store, T0)
    _make(store, "b", 5_000)
    _trade(store, "a", -3_800)  # 21,200 against 25,000: 15.2%
    st = _mark(store, T0 + timedelta(minutes=1))
    assert st.equity == D(21_200) and st.halted


def test_c5_a_day_drop_of_3_percent_from_the_flowed_start_pauses_and_a_loss_on_the_flows_mark_counts(store):
    """The loss lands before the create and both before the next mark: 24,000 against the flowed 25,000 is 4% down
    on the day. Without the flow it would read as a 4,000 gain."""
    _make(store, "a", 20_000)
    _mark(store, T0)
    _trade(store, "a", -1_000)
    _make(store, "b", 5_000)
    st = _mark(store, T0 + timedelta(minutes=1))
    assert (st.equity, st.hwm, st.day_start_equity) == (D(24_000), D(25_000), D(25_000))
    assert st.paused_until is not None and not st.halted


def test_c6_a_mark_that_read_the_book_before_a_flow_landed_is_skipped_not_a_phantom_loss(store, monkeypatch):
    """The race: the supervisor reads the book (20,000), a 5,000 Strategy is created and its flow lands, then the
    supervisor takes the gate's lock to mark its stale read. Marked, 20,000 against the flowed 25,000 would be a 20%
    phantom loss and halt; it is skipped, and the next mark reads the whole book."""
    _make(store, "a", 20_000)
    _mark(store, T0)
    real = supervisor.fund_equity

    def read_then_move(s):
        equity = real(s)
        if len(s.sleeves()) == 1:  # once: the move lands after this read, before the mark's lock
            _make(store, "b", 5_000)
        return equity

    monkeypatch.setattr(supervisor, "fund_equity", read_then_move)
    st = _mark(store, T0 + timedelta(minutes=1))
    assert _four(st) == (D(25_000),) * 4 and not st.halted and st.mark_ts == T0  # skipped: the last mark stands
    monkeypatch.setattr(supervisor, "fund_equity", real)
    st = _mark(store, T0 + timedelta(minutes=2))
    assert _four(st) == (D(25_000),) * 4 and not st.halted and st.paused_until is None


def test_c7_a_restart_or_a_repeated_move_applies_no_flow_twice(store):
    _make(store, "fund", 10_000)
    _make(store, "test", 20_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "desk")
    _mark(store, T0)
    store.assign_account("test", "paper")
    store.assign_account("test", "paper")  # the same command again: already out of the book
    again = Store(engine=store.engine, portfolio_gate=True)  # a restart: a fresh store and supervisor on the journal
    st = _mark(again, T0 + timedelta(minutes=1))
    assert _four(st) == (D(10_000),) * 4 and len(_flows(store)) == 1 and not st.halted


# --- C9: what the dashboard and the deploy log show ----------------------------------------------------------------------

def test_c9_a_flow_is_capital_not_pnl_in_the_book_figures(store):
    """The dashboard's and the deploy log's book figures come from each Strategy's equity against its starting
    balance, never from the gate's state: a create raises both by its capital, so it shows as no P&L."""
    _make(store, "a", 20_000)
    _trade(store, "a", 300)
    _mark(store, T0)
    assert supervisor.book_figures(store).startswith("from 20,000.00, equity 20,300.00,")
    _make(store, "b", 5_000)
    assert supervisor.book_figures(store).startswith("from 25,000.00, equity 25,300.00,")


# --- C12: a starting balance is never edited -----------------------------------------------------------------------------

def test_c12_a_starting_balance_is_never_edited_so_it_never_moves_the_book_unflowed(store):
    _make(store, "a", 20_000)
    with pytest.raises(ValueError, match="never edited"):
        store._update_sleeve("a", starting_balance=25_000)
    assert store.sleeve("a").starting_balance == 20_000


# --- C8: the one-off re-base -------------------------------------------------------------------------------------------

def _rows(store):
    with store.engine.connect() as c:
        return [c.execute(select(t)).all() for t in (sleeves_t, fills_t, orders_t, equity_t)]


def test_c8_the_rebase_puts_the_incidents_state_on_the_true_book_once_journaled(store, monkeypatch):
    """The paper incident's state: the reference far above the book and the halt it set. The re-base sets the book,
    the HWM, the reference and the day's start to the true book, clears that halt, and journals a warning event and a
    decision row with the old figures (the old HWM kept there), the drawdown and the commit. It writes only the state
    row and its journal; a dry run writes nothing; the next mark is clean; a second run is refused."""
    _make(store, "a", 30_000)
    _trade(store, "a", -500)  # 29,500: 1.67% under the 30,000 put in
    _mark(store, T0)
    _incident(store, monkeypatch)
    journal = _rows(store)
    assert supervisor.main(["rebase", "--actor", "HoE", "--commit", "abc1234"]) == 0  # a dry run
    assert _state(store).halted and not store.decisions(action="portfolio_rebase")
    before, after, words = supervisor.rebase_halt_ref_1(store, "HoE", "abc1234", now=T0 + timedelta(minutes=1))
    assert before.hwm == D("109108.90") and before.halted
    st = _state(store)
    assert _four(st) == (D(29_500),) * 4 and st.halted is None and st.paused_until is None
    (event,) = [e for e in store.events(None, limit=50) if e["kind"] == "portfolio_rebased"]
    assert event["level"] == "warning" and event["message"] == words
    for part in ("HALT-REF-1: portfolio re-based to the true book at commit abc1234",
                 "high-water mark 109,108.90 -> 29,500.00 (the old one kept here)", "1.67% drawdown against the "
                 "30,000.00 put in", "the halt (the book is 81.8% below its reference) is cleared"):
        assert part in words
    with store.engine.connect() as c:
        rows = c.execute(select(decisions_t).where(decisions_t.c.action == "portfolio_rebase")).all()
    assert [(r.actor, r.reason) for r in rows] == [("HoE", words)]
    assert _rows(store) == journal  # nothing but the state row and its journal
    st = _mark(store, T0 + timedelta(minutes=2))
    assert not st.halted and st.paused_until is None and _four(st) == (D(29_500),) * 4
    with pytest.raises(RuntimeError, match="has run already"):
        supervisor.rebase_halt_ref_1(store, "HoE", "abc1234")


def test_c8_the_command_writes_then_shows_the_next_marks_clean_checks(store, monkeypatch, capfd):
    _make(store, "a", 30_000)
    _mark(store, T0)
    _incident(store, monkeypatch)
    assert supervisor.main(["rebase", "--actor", "HoE", "--commit", "abc1234", "--yes"]) == 0
    out = capfd.readouterr().out
    assert "journaled: HALT-REF-1:" in out and "next mark: book 30000" in out and "halted no, paused no" in out
    assert supervisor.main(["rebase", "--actor", "HoE", "--commit", "abc1234", "--yes"]) == 1
    assert "refused: the HALT-REF-1 re-base has run already" in capfd.readouterr().out


@pytest.mark.parametrize("loss, ok", [(870, True), (900, False)])
def test_c8a_a_rebase_at_2_9_percent_runs_and_at_3_percent_refuses_naming_the_drawdown(store, loss, ok):
    _make(store, "a", 30_000)
    _trade(store, "a", -loss)
    _mark(store, T0)
    _incident(store)
    if ok:
        _, after, words = supervisor.rebase_halt_ref_1(store, "HoE", "abc1234")
        assert after.equity == D(29_130) and "2.90% drawdown against the 30,000.00 put in" in words
    else:
        with pytest.raises(RuntimeError, match=r"the book 29,100.00 is 3.00% under the 30,000.00 put in, a real drawdown"):
            supervisor.rebase_halt_ref_1(store, "HoE", "abc1234")
        assert _state(store).halted and _state(store).hwm == D("109108.90")  # untouched


def test_c8b_the_rebase_refuses_while_a_strategy_runs_holds_or_the_books_dont_reconcile(store):
    _make(store, "run", 10_000)
    _make(store, "held", 10_000)
    _make(store, "off", 10_000)
    store.set_desired_state("run", "running")
    store.record_fill("held", side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id="h1", trade_id="h1", ts=T0)
    store.record_equity("held", equity=10_000.0, cash=9_000.0, qty=1.0, price=1_000.0, benchmark=10_000.0, ts=T0)
    _trade(store, "off", 100)
    store.record_equity("off", equity=10_000.0, cash=10_000.0, qty=0.0, price=1_000.0, benchmark=10_000.0,
                        ts=T0 + timedelta(seconds=1))  # the dashboard's mark 100 short of the journal
    _mark(store, T0 + timedelta(seconds=2))
    _incident(store)
    with pytest.raises(RuntimeError) as refused:
        supervisor.rebase_halt_ref_1(store, "HoE", "abc1234")
    why = str(refused.value)
    assert "run could trade once the halt clears (running, stopped): pause it first" in why
    assert "held holds 1" in why and "the book must be flat" in why
    assert "the books don't reconcile: off:" in why
    assert _state(store).halted and not store.decisions(action="portfolio_rebase")


def test_c8_the_rebase_needs_an_actor_and_the_commit(store):
    with pytest.raises(ValueError, match="actor and the deployed commit"):
        supervisor.rebase_halt_ref_1(store, " ", "abc1234")


def test_c8_the_halt_is_cleared_only_by_the_command_line_rebase():
    """Advisor (09:00 UK): the halt clear lives only in the one-off re-base; no engine code, dashboard route or API
    reaches it, and the gate offers no re-base of its own."""
    import pathlib

    from sleeve_fund.portfolio import gate

    root = pathlib.Path(supervisor.__file__).parent
    callers = sorted(str(p.relative_to(root)) for p in root.rglob("*.py") if "rebase_halt_ref_1" in p.read_text())
    assert callers == ["supervisor.py"] and not hasattr(gate, "rebase_book")


def _commands(store, name):
    return [(c["command"], c["reason"]) for c in store.pending_commands(name)]


def test_c8_the_command_pauses_what_could_trade_journaled_then_rebases(store, monkeypatch, capfd):
    """Paper's Strategies are flat but enabled under the halt; once it clears they could trade. --yes pauses each one
    that could (a PM pause, journaled) and waits for its process to apply it; one halted on its own already can't, and
    gets no command. The dry run before it says what --yes would pause and writes nothing."""
    _make(store, "halted", 10_000)
    _make(store, "live", 20_000)
    for name, status in (("halted", "halted"), ("live", "running")):
        store.set_desired_state(name, "running")
        store.set_status(name, status)
    _mark(store, T0)
    _incident(store, monkeypatch)
    assert supervisor.main(["rebase", "--actor", "HoE", "--commit", "abc1234"]) == 0
    assert "--yes pauses live first" in capfd.readouterr().out and _commands(store, "live") == []

    def process_applies(_):  # what live's process does at its next poll
        for c in store.pending_commands("live"):
            store.set_status("live", "paused", f"paused by PM: {c['reason']}")
            store.mark_applied(c["id"])

    monkeypatch.setattr(supervisor.time, "sleep", process_applies)
    assert supervisor.main(["rebase", "--actor", "HoE", "--commit", "abc1234", "--yes", "--wait", "5"]) == 0
    out = capfd.readouterr().out
    assert "paused: live" in out and "paused: halted" not in out and "halted no, paused no" in out
    (pause,) = store.decisions(action="pause")
    assert (pause["actor"], pause["sleeve"], pause["reason"]) == ("HoE", "live", supervisor.REBASE_PAUSE)
    assert _commands(store, "halted") == [] and store.sleeve("live").status == "paused"
    assert _state(store).equity == D(30_000) and not _state(store).halted


def test_c8_a_pause_not_applied_in_time_writes_nothing_and_the_pause_stays(store, monkeypatch, capfd):
    _make(store, "live", 30_000)
    store.set_desired_state("live", "running")
    _mark(store, T0)
    _incident(store, monkeypatch)
    assert supervisor.main(["rebase", "--actor", "HoE", "--commit", "abc1234", "--yes", "--wait", "0"]) == 1
    assert "live didn't apply the pause within 0 s; the pauses stay" in capfd.readouterr().out
    assert _commands(store, "live") == [("pause", supervisor.REBASE_PAUSE)]
    assert _state(store).halted and not store.decisions(action="portfolio_rebase")


def test_c8_a_refusal_a_pause_cant_answer_pauses_nothing(store, monkeypatch, capfd):
    _make(store, "live", 30_000)
    store.set_desired_state("live", "running")
    store.record_fill("live", side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id="h1", trade_id="h1", ts=T0)
    store.record_equity("live", equity=30_000.0, cash=29_000.0, qty=1.0, price=1_000.0, benchmark=30_000.0, ts=T0)
    _mark(store, T0)
    _incident(store, monkeypatch)
    assert supervisor.main(["rebase", "--actor", "HoE", "--commit", "abc1234", "--yes"]) == 1
    assert "not written; nothing paused" in capfd.readouterr().out and _commands(store, "live") == []


# --- CR pre-review pins (R1-R3) ------------------------------------------------------------------------------------------

def _dust(store, name, ts=T0):
    """0.000001 at 60,000: 0.06 of notional, under DUST_NOTIONAL, so no venue takes an order for it (F229-7)."""
    store.record_fill(name, side="BUY", qty=0.000001, price=60_000.0, fee=0.0, order_id="d1", trade_id="d1", ts=ts)
    book = store.journal_book(name, store.sleeve(name).starting_balance)
    store.record_equity(name, equity=float(book["cash"]) + 0.06, cash=float(book["cash"]), qty=0.000001,
                        price=60_000.0, benchmark=30_000.0, ts=ts)


def test_r1_dust_a_strategy_holds_is_flat_to_the_rebase(store):
    _make(store, "a", 30_000)
    _dust(store, "a")
    _mark(store, T0 + timedelta(seconds=1))
    _incident(store)
    _, after, words = supervisor.rebase_halt_ref_1(store, "HoE", "abc1234")
    assert after.equity == D(30_000) and not after.halted
    assert "dust counted as flat: a 0.000001 worth 0.0600;" in words  # Advisor 09:55 (2): every holding journaled


def _dust_of(store, name, qty):
    store.record_fill(name, side="BUY", qty=qty, price=60_000.0, fee=0.0, order_id="d1", trade_id="d1", ts=T0)
    book = store.journal_book(name, store.sleeve(name).starting_balance)
    store.record_equity(name, equity=float(book["cash"]) + qty * 60_000, cash=float(book["cash"]), qty=qty,
                        price=60_000.0, benchmark=0.0, ts=T0)


def test_r1_dust_over_a_tenth_of_a_percent_of_the_book_refuses_naming_it(store):
    """Advisor 09:55 (1): 0.000015 at 60,000 is 0.90, dust (under 1.00 of notional), but 0.18% of a 500 book."""
    _make(store, "a", 500)
    _dust_of(store, "a", 0.000015)
    _mark(store, T0 + timedelta(seconds=1))
    _incident(store)
    with pytest.raises(RuntimeError, match=r"dust worth 0.90 across the book is 0.180% of it, over the 0.1%"):
        supervisor.rebase_halt_ref_1(store, "HoE", "abc1234")
    assert _state(store).halted


def test_r1_a_strategy_holding_only_dust_that_could_trade_still_refuses(store):
    """Advisor 09:55 (3): dust is flat, but a Strategy that could trade is still running."""
    _make(store, "a", 30_000)
    _dust(store, "a")
    store.set_desired_state("a", "running")
    _mark(store, T0 + timedelta(seconds=1))
    _incident(store)
    with pytest.raises(RuntimeError, match="a could trade once the halt clears") as refused:
        supervisor.rebase_halt_ref_1(store, "HoE", "abc1234")
    assert "must be flat" not in str(refused.value)


def test_r1_and_a_dust_reset_run_put_away_is_too_and_capital_in_leaves_the_run_out(store):
    """A reset that put its run away with dust (split_run dust_ok) leaves that run in store.sleeves(): it is flat to
    the re-base, and the capital put in counts the Strategy once (the successor's), so a 500 loss is 1.67%, not a
    false 3%+ refusal against 60,000."""
    _make(store, "a", 30_000)
    _trade(store, "a", -500)
    _dust(store, "a", T0 + timedelta(seconds=1))
    _mark(store, T0 + timedelta(seconds=2))
    store.request_reset("a", "testing")
    run = store.split_run(store.pending_reset("a"), dust_ok=True)
    assert run in {s.name for s in store.sleeves()} and supervisor.capital_in(store) == D(30_000)
    _mark(store, T0 + timedelta(seconds=3))
    _incident(store)
    _, after, words = supervisor.rebase_halt_ref_1(store, "HoE", "abc1234")
    assert abs(after.equity - D("29499.94")) < D("0.01") and "1.67% drawdown against the 30,000.00 put in" in words


def test_the_dry_run_fails_when_a_pause_cant_answer_the_refusal(store, monkeypatch, capfd):
    _make(store, "a", 30_000)
    store.record_fill("a", side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id="h1", trade_id="h1", ts=T0)
    store.record_equity("a", equity=30_000.0, cash=29_000.0, qty=1.0, price=1_000.0, benchmark=30_000.0, ts=T0)
    _mark(store, T0)
    _incident(store, monkeypatch)
    assert supervisor.main(["rebase", "--actor", "HoE", "--commit", "abc1234"]) == 1
    assert "refused: a holds 1" in capfd.readouterr().out


def test_a_move_whose_own_write_fails_leaves_no_flow_and_no_event(store):
    """_book_move: the flow is applied only after the caller's writes, in their transaction, so a write that raises
    rolls the whole move back: no flow on the state, no portfolio_flow event."""
    _make(store, "fund", 10_000)
    _make(store, "test", 5_000, instrument="BTC/USD", venue="KRAKEN")  # outside the book on the paper account
    before = _four(_mark(store, T0))
    with pytest.raises(ValueError, match="no account called"):
        store.assign_account("test", "nope")
    assert _four(_state(store)) == before and _flows(store) == [] and store.account_of("test") == "paper"


def test_a_two_leg_strategy_created_after_the_first_mark_is_one_deposit_of_its_balance(store):
    """P2-8: both legs' shares come in together, with the rows, as one flow of the Strategy's whole balance."""
    _make(store, "a", 20_000)
    _mark(store, T0)
    store.create_two_legs(name="carry", strategy="ping_pong", bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000,
                          first={"instrument": "BTC/USDT", "params": {}, "venue": "BINANCE", "side": 1},
                          second={"instrument": "BTC/USDT", "params": {"market": "perp"}, "venue": "BINANCE",
                                  "side": -1}, desired_state="stopped")
    assert _four(_state(store)) == (D(30_000),) * 4
    assert len(_flows(store)) == 1 and "two-leg Strategy carry created with 10,000.00" in _flows(store)[0]
    st = _mark(store, T0 + timedelta(minutes=1))
    assert _four(st) == (D(30_000),) * 4 and not st.halted
