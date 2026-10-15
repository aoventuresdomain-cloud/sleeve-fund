"""RESET-NEG (Advisor ruling, option A with the HoE's correction, 15 Oct 03:10 UK): the PM's reset found two reset runs
of the old book below zero (a loss no withdrawal takes out), so the book could never be emptied and the reset refused.
Before the empty-book check, each such run is closed out: one journaled row credits the run up to exactly 0 and debits
the old book's pooled cash, the Strategy it was reset from (its capital is that Strategy's, split_run), by the same
amount. A net-zero transfer inside the old book, never a new loss: the book's total and its identity don't move, and
the old book then leaves whole (withdrawn to the cent), so the fresh deposit lands on an empty book as before."""

from datetime import timedelta
from decimal import Decimal as D

from test_halt_ref import T0, _make, _mark, _reset, _trade, store  # noqa: F401

from sleeve_fund import reset_book, supervisor

SLATE = "2026-10-15-reset"
BOOK = D("25137.79")  # 10,000 + 10,000 + 5,000, less 29.70 and 82.51, plus 250


def _old_book(store):
    """As the dry run found it on 15 Oct: two Strategies, each with an earlier run reset below zero (-29.70 and
    -82.51), and a third whose earlier run made 250 (a positive pocket, which a close-out never touches)."""
    _make(store, "rsi-bands", 10_000)
    _make(store, "ping-pong", 10_000)
    _make(store, "carry", 5_000)
    _mark(store, T0)
    _trade(store, "rsi-bands", -29.70, T0)
    _trade(store, "ping-pong", -82.51, T0)
    _trade(store, "carry", 250, T0)
    runs = [_reset(store, n) for n in ("rsi-bands", "ping-pong", "carry")]
    _mark(store, T0 + timedelta(minutes=1))
    assert supervisor.fund_equity(store) == BOOK
    assert [store.strategy_cash_pnl(r) for r in runs] == [D("-29.70"), D("-82.51"), D(250)]
    return runs


def _pass(store, i):
    return reset_book.run(store, "operator", "abc1234", SLATE, D(100_000), now=T0 + timedelta(minutes=2 + i))


def _close_out_rows(store):
    from sqlalchemy import select

    from sleeve_fund.store import insurance_t

    i = insurance_t.c
    with store.engine.connect() as c:
        return [(r.sleeve, D(str(r.amount))) for r in c.execute(
            select(i.sleeve, i.amount).where(i.order_id == reset_book.CLOSE_OUT).order_by(i.id))]


def _until_done(store, steps):
    """The operator's passes, one after each supervisor mark; `steps` gets the book's total and its identity gap after
    every pass and every mark."""
    for i in range(6):
        code, lines = _pass(store, i)
        steps.append((supervisor.fund_equity(store), supervisor.identity_gap(store)))
        if code != reset_book.WAIT:
            return code, lines
        _mark(store, T0 + timedelta(minutes=2 + i, seconds=30))
        steps.append((supervisor.fund_equity(store), supervisor.identity_gap(store)))
    raise AssertionError(f"still waiting: {lines}")


def test_the_dry_run_lists_the_close_outs_and_no_longer_refuses(store):
    _old_book(store)
    lines = reset_book.plan(store, SLATE, D(100_000))
    assert not any(line.startswith("would refuse") for line in lines), lines
    assert lines[1] == f"the book now: {BOOK:,.2f}"
    assert ("would close out first, from the old book's cash (no effect on its total): rsi-bands −29.70, "
            "ping-pong −82.51; 112.21 in all") in lines, lines
    assert _close_out_rows(store) == []  # the dry run writes nothing


def test_each_negative_run_ends_at_exactly_0_and_the_old_book_leaves_whole_to_the_cent(store):
    r1, r2, r3 = _old_book(store)
    steps = []
    code, lines = _until_done(store, steps)
    assert code == reset_book.DONE, lines
    assert store.strategy_cash_pnl(r1) == 0 and store.strategy_cash_pnl(r2) == 0
    rows = _close_out_rows(store)
    assert sorted(rows) == sorted([(r1, D("29.70")), ("rsi-bands", D("-29.70")),
                                   (r2, D("82.51")), ("ping-pong", D("-82.51"))])
    assert sum(a for _, a in rows if a > 0) == D("112.21") and sum(a for _, a in rows) == 0
    # The old book withdraws its whole figure, the total before any close-out, to the cent; nothing left behind
    assert sum(store.withdrawn().values()) == BOOK
    assert supervisor.empty_book_pockets(store) == []
    # A positive run is never touched: it leaves with its own 250
    assert r3 not in {s for s, _ in rows} and store.withdrawn()[r3] == D(250)
    # Identity at every step: the close-outs move nothing in the book and the gap stays 0
    assert steps[0] == (BOOK, 0), steps
    assert all(gap == 0 for _, gap in steps), steps


def test_each_close_out_is_journaled_by_strategy_name_with_no_effect_on_the_old_books_total(store):
    r1, r2, _ = _old_book(store)
    _until_done(store, [])
    said = [d["reason"] for d in store.decisions(action=reset_book.RESET_ACTION, limit=10_000)]
    for name, loss in (("rsi-bands", "29.70"), ("ping-pong", "82.51")):
        assert any(r.endswith(
            f"Old book close-out: {name} ended at −{loss} from paper losses before the 14 Oct restart; covered "
            "from the old book's cash. No effect on the old book total.") for r in said), said
    assert not any(r1 in r or r2 in r for r in said if "close-out" in r)  # a run's id names its venue: never shown


def test_the_reset_says_where_the_old_book_closed_for_the_pm(store):
    _old_book(store)
    code, lines = _until_done(store, [])
    assert code == reset_book.DONE
    assert (f"Old book closed at {BOOK:,.2f}, including 112.21 of losses on two retired Strategies that had run below "
            "zero; the new fund starts at 100,000.") in lines, lines


def test_a_rerun_is_a_no_op(store):
    _old_book(store)
    code, _ = _pass(store, 0)
    assert code == reset_book.WAIT
    rows = _close_out_rows(store)
    assert len(rows) == 4
    _pass(store, 1)  # again before the mark: the runs are at 0, nothing more is closed out
    assert _close_out_rows(store) == rows
    assert not any("close out" in line for line in reset_book.plan(store, SLATE, D(100_000)))
    _until_done(store, [])
    code, lines = _pass(store, 9)
    assert code == reset_book.DONE and lines == [f"reset {SLATE}: deposited already, nothing to do"]
    assert _close_out_rows(store) == rows


def test_a_loss_the_strategy_it_came_from_cant_carry_is_still_refused_by_name_and_nothing_is_written(store):
    """The pooled cash must cover it: a run whose Strategy would itself go below zero (here one reset at 100 that
    lost 120 before) is not closed out but refused, by name, as before."""
    _make(store, "small", 100)
    _mark(store, T0)
    _trade(store, "small", -120, T0)
    run = _reset(store, "small")
    before = (store.decisions(limit=10_000), store.archived())
    assert any(line.startswith("would refuse") for line in reset_book.plan(store, SLATE, D(100_000)))
    code, lines = _pass(store, 0)
    assert code == reset_book.REFUSED and _close_out_rows(store) == []
    assert (store.decisions(limit=10_000), store.archived()) == before
    assert lines == [f"refused, nothing written: the book can't be emptied while {run} -120.00, a loss no withdrawal "
                     "takes out"]


def test_a_close_out_is_never_a_trades_pnl_in_attribution(store):
    """HoQA cell 9: the close-out's cash rows are a transfer, not trading. A trip open across one (here a close-out
    written mid-trip, the worst case) keeps the P&L of its fills alone; the rows the trades are paired with
    (Store.insurance, the Trades tab's and attribution's) leave them out, while the book's two cash readers count them."""
    from sleeve_fund.research.metrics import trades

    _make(store, "rsi-bands", 10_000)
    _mark(store, T0)
    _trade(store, "rsi-bands", -29.70, T0)
    run = _reset(store, "rsi-bands")
    t1, t2, t3 = (T0 + timedelta(minutes=m) for m in (5, 6, 7))
    store.record_fill("rsi-bands", side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id="b1", trade_id="b1", ts=t1)
    assert store.close_out_run(run, "rsi-bands", D("29.70"), reset_book.CLOSE_OUT, t2, "operator",
                               reset_book.RESET_ACTION, "close-out")
    store.record_fill("rsi-bands", side="SELL", qty=1.0, price=1_010.0, fee=0.0, order_id="s1", trade_id="s1", ts=t3)
    for name, cash in (("rsi-bands", D("-19.70")), (run, D(0))):
        assert store.insurance(name, limit=None) == []
        assert store.strategy_cash_pnl(name) == cash  # the book still counts the transfer
    fills = sorted(store.fills("rsi-bands", limit=None), key=lambda f: f["ts"])
    trips = trades(fills, True, None, store.insurance("rsi-bands", limit=None))
    assert [round(t["pnl"], 2) for t in trips] == [10.0]
    assert not store.close_out_run(run, "rsi-bands", D("29.70"), reset_book.CLOSE_OUT, t3, "operator",
                                   reset_book.RESET_ACTION, "again")  # once only


def _fund_held(store):
    """The old book reset to DONE and its re-base deposit applied: the fund holds the book (bridge_start set)."""
    _old_book(store)
    code, _ = _until_done(store, [])
    _mark(store, T0 + timedelta(minutes=20))
    assert code == reset_book.DONE and supervisor.bridge_start(store) is not None


def _holds(store):
    return store.decisions(action=supervisor.IDENTITY_HOLD_ACTION, limit=100)


def test_a_close_out_on_a_fund_held_book_is_never_an_insurance_break_or_a_held_book(store):
    """DA / HoE RESET-NEG cell 10: the close-out's rows are insurance rows, and once the fund holds the book an
    insurance credit inside it is a break that latches the identity hold, which only the PM lifts. A close-out isn't
    one; a real credit (no order named, as before 0013) still is."""
    _fund_held(store)
    _make(store, "x", 0)
    _trade(store, "x", -50, T0 + timedelta(minutes=21))
    run = _reset(store, "x")
    at = T0 + timedelta(minutes=22)
    assert store.close_out_run(run, "x", D(50), reset_book.CLOSE_OUT, at, "operator", reset_book.RESET_ACTION, "c")
    sup = supervisor.Supervisor(store)
    assert supervisor.insurance_breaks(store, supervisor.bridge_start(store)) == []
    sup._check_identity(at + timedelta(seconds=1))
    assert _holds(store) == [] and not sup.ledger.identity_hold()
    from sqlalchemy import insert

    from sleeve_fund.store import insurance_t

    with store.engine.begin() as c:  # a credit from before 0013: no fill named
        c.execute(insert(insurance_t).values(sleeve="x", ts=at + timedelta(seconds=2), price=0, amount=D(5)))
    assert len(supervisor.insurance_breaks(store, supervisor.bridge_start(store))) == 1


def test_a_loss_whose_strategy_has_already_left_the_book_is_refused_not_closed_out(store):
    """CR F-RN-2: the Strategy a run came from, already withdrawn, has no cash in the book to cover it."""
    r1, _, _ = _old_book(store)
    store.archive("rsi-bands")
    _mark(store, T0 + timedelta(minutes=1, seconds=30))
    assert "rsi-bands" in store.withdrawn()
    assert [run for run, _, _ in reset_book._close_outs(store)] == [r for r in store.reset_runs() if "ping-pong" in r]
    assert any(line.startswith("would refuse") and r1 in line for line in reset_book.plan(store, SLATE, D(100_000)))


def test_a_close_out_that_fails_part_way_writes_nothing(store):
    """CR F-RN-3: the credit, the debit and the journal row are one transaction: a failure after the cash rows (here
    the journal row, with no action) leaves neither."""
    r1, _, _ = _old_book(store)
    try:
        store.close_out_run(r1, "rsi-bands", D("29.70"), reset_book.CLOSE_OUT, T0 + timedelta(minutes=2), "operator",
                            None, "c")
    except Exception:  # noqa: BLE001 - the database's refusal, whichever engine's
        pass
    else:
        raise AssertionError("set-up: the journal row was written with no action")
    assert _close_out_rows(store) == [] and store.strategy_cash_pnl(r1) == D("-29.70")
