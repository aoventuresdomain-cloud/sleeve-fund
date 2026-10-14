"""F-ASSIGN-STALE (CR on #286, HoE 14 Oct): a Strategy moved off the book (onto the test account its venue names,
F229-6) is withdrawn at its equity NOW, but apply_flow scales the book's figures as of the LAST MARK. P&L it made or
lost since that mark is in neither, so the withdrawal takes it from the rest of the book: a loss since the mark reads
as a deeper drawdown of the rest (and can halt the fund falsely), a gain as a gain of the rest.

The right figures are the ones a mark taken at the move, then the withdrawal, would give: the book's real P&L while
the Strategy was in it is kept, as a percentage, as apply_flow keeps it (Advisor MUST, 14 Oct 09:45 UK). That is what
#286 does for a put-away Strategy (Supervisor._withdraw_flat after a mark); the fix reuses it. Red first on 22d36f8."""

from datetime import timedelta
from decimal import ROUND_HALF_EVEN, Decimal as D, localcontext

from sleeve_fund import supervisor
from tests.test_halt_ref import T0, _flows, _four, _make, _mark, _trade, store  # noqa: F401, F811


def _moved_off(store, pnl):  # noqa: F811
    """fund 10,000 and test 20,000 in the book, marked at 30,000; then `test` makes `pnl` (unmarked) and moves onto
    KRAKEN's named test account; then the next mark. Returns that mark's state."""
    _make(store, "fund", 10_000)
    _make(store, "test", 20_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "desk")  # in the book
    assert _mark(store, T0).equity == D(30_000)
    _trade(store, "test", pnl, ts=T0 + timedelta(seconds=30))  # since the mark: not in the book's figures
    store.assign_account("test", "paper")  # off the book
    return _mark(store, T0 + timedelta(minutes=1))


def _same(got, want):
    """Figure for figure to 1e-8: SQLite keeps a NUMERIC to about 16 digits, Postgres to the column's 18 places."""
    return all(abs(D(g) - D(w)) < D("1e-8") for g, w in zip(got, want, strict=True))


def _kept(peak, at_move, after):
    """A figure scaled as apply_flow scales it: peak x after / at_move, to the column's 18 places."""
    with localcontext() as ctx:
        ctx.prec = 60
        return (D(peak) * D(after) / D(at_move)).quantize(D("1e-18"), ROUND_HALF_EVEN)


def test_a_gain_since_the_mark_leaves_with_the_strategy_and_the_rest_reads_unchanged(store):  # noqa: F811
    st = _moved_off(store, 5_000)  # the book really was 35,000 at the move; 25,000 of it leaves
    # The high-water mark and the reference were 35,000 at the move (a new high), so the rest reads at its peak. The
    # day opened at 30,000 and the whole book is up 16.7% today: apply_flow keeps that percentage, as for any flow.
    assert _same(_four(st), (D(10_000), D(10_000), D(10_000), _kept(30_000, 35_000, 10_000))), _four(st)
    assert not st.halted and st.paused_until is None


def test_a_loss_since_the_mark_is_the_books_real_drawdown_not_a_deeper_one(store):  # noqa: F811
    """The book really fell 30,000 -> 27,000 (10%) while `test` was in it; 17,000 leaves. The rest reads 10% down,
    under the 15% halt. Today (the flow scaled from the 30,000 mark): the rest reads 23% down."""
    st = _moved_off(store, -3_000)
    kept = _kept(30_000, 27_000, 10_000)
    assert _same(_four(st), (D(10_000), kept, kept, kept)), _four(st)
    assert not st.halted  # 10% is under the 15% halt; it is a real 10% day loss, so the day's pause is right
    assert st.paused_until is not None


def test_a_halt_from_a_big_loss_since_the_mark_states_the_real_drawdown(store):  # noqa: F811
    """30,000 -> 22,000 while it was in the book is a real 26.7% drawdown, and halts. It must read 26.7%, not the
    44.4% the stale flow makes of it."""
    st = _moved_off(store, -8_000)
    assert st.halted and "26.7%" in st.halted and "44.4%" not in st.halted, st.halted


# --- F-ASSIGN-STALE (Advisor ADV-ASSIGN, 14 Oct 17:41 UK): only the poll moves capital on a marked book -----------------

def test_a_strategy_whose_move_waits_has_its_entries_refused_until_the_poll_lands_it(store):  # noqa: F811
    from sleeve_fund.paper.runtime import entry_blocked

    _make(store, "fund", 10_000)
    _mark(store, T0)
    _make(store, "new", 5_000)  # created on a marked book: it joins at the next poll
    blocked, why = entry_blocked(store, "new", T0, starting=True)
    assert blocked and "book_moving" in why.codes
    assert "Entry refused: this Strategy is moving to paper at the next mark." in why
    assert "book_moving" not in getattr(entry_blocked(store, "fund", T0, starting=True)[1], "codes", ())
    _mark(store, T0 + timedelta(seconds=5))
    why = entry_blocked(store, "new", T0 + timedelta(seconds=5), starting=True)[1]
    assert "book_moving" not in getattr(why, "codes", ())


def test_a_move_off_the_book_waits_on_its_old_account_and_its_entries_name_the_new_one(store):  # noqa: F811
    from sleeve_fund.paper.runtime import entry_blocked

    _make(store, "fund", 10_000)
    _make(store, "test", 20_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "desk")
    _mark(store, T0)
    store.assign_account("test", "paper")
    assert store.account_of("test") == "desk" and store.book_moves_pending()["test"]["account"] == "paper"
    why = entry_blocked(store, "test", T0, starting=True)[1]
    assert "Entry refused: this Strategy is moving to paper at the next mark." in why
    store.assign_account("test", "desk")  # asked again before the poll: same side as it is, made at once
    assert store.book_moves_pending() == {} and store.account_of("test") == "desk"
    assert "book_moving" not in getattr(entry_blocked(store, "test", T0, starting=True)[1], "codes", ())


def test_a_later_request_replaces_a_waiting_one_and_says_so(store):  # noqa: F811
    _make(store, "fund", 10_000)
    _make(store, "test", 20_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "desk")
    _mark(store, T0)
    store.create_account("desk-2", "paper")
    store.assign_account("test", "paper")
    _make(store, "new", 5_000)
    store.assign_account("new", "desk-2")  # its create still waits: it now joins on desk-2
    assert store.book_moves_pending()["new"] == {**store.book_moves_pending()["new"], "account": "desk-2",
                                                 "kind": "create"}
    said = [e["message"] for e in store.events("new", limit=10) if e["kind"] == "book_move_waiting"]
    assert len(said) == 2 and "replacing its move to account paper" in said[0]
    st = _mark(store, T0 + timedelta(minutes=1))
    assert st.equity == D(15_000) and store.account_of("new") == "desk-2" and store.account_of("test") == "paper"
    assert store.book_moves_pending() == {}


def test_a_create_while_the_book_is_halted_lands_at_the_poll_and_stays_halted(store):  # noqa: F811
    from sleeve_fund.paper.runtime import entry_blocked

    _make(store, "fund", 10_000)
    _mark(store, T0)
    _trade(store, "fund", -2_000)  # 20% down: the poll halts
    assert _mark(store, T0 + timedelta(seconds=5)).halted
    _make(store, "new", 5_000)
    st = _mark(store, T0 + timedelta(seconds=10))
    assert st.halted and st.equity == D(13_000) and not store.joining("new")
    assert abs(1 - st.equity / st.reference() - D("0.2")) < D("1e-12")  # still 20% down: a deposit isn't a gain
    why = entry_blocked(store, "new", T0 + timedelta(seconds=10), starting=True)[1]
    assert "portfolio_halt" in why.codes and "book_moving" not in why.codes


def test_a_move_still_waiting_after_three_polls_is_alerted_once(store, monkeypatch):  # noqa: F811
    _make(store, "fund", 10_000)
    _mark(store, T0)
    _make(store, "new", 5_000)

    def refuse(name, ts=None):
        raise ValueError("the gate's lock timed out")

    monkeypatch.setattr(store, "land_move", refuse)
    sup = supervisor.Supervisor(store)
    for i in range(5):
        sup.mark_portfolio(T0 + timedelta(seconds=5 * (i + 1)))
    stuck = [e["message"] for e in store.events(None, limit=50) if e["kind"] == "book_move_stuck"]
    assert len(stuck) == 1 and "waited 3 polls" in stuck[0] and "the gate's lock timed out" in stuck[0]
    waits = [e for e in store.events("new", limit=50) if e["kind"] == "book_move_waits"]
    assert len(waits) == 1 and store.joining("new")


def test_an_unmarked_book_still_moves_at_once_with_no_flow(store):  # noqa: F811
    _make(store, "fund", 10_000)
    _make(store, "test", 20_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "desk")
    assert store.account_of("test") == "desk" and store.book_moves_pending() == {} and _flows(store) == []
    assert not store.joining("fund") and _mark(store, T0).equity == D(30_000)


# The same stale scale hits any capital move made outside the poll, not only the moving Strategy's own P&L (FE prep,
# fe-plans/f-assign-stale-plan.md): what the rest of the book made or lost since the mark skews the ratio too. The
# right figures are again those of a mark at the move, then the flow.


def _dd(st):
    return (1 - st.equity / st.reference()).quantize(D("1e-6"))


def test_a_create_after_an_unmarked_loss_keeps_the_books_real_drawdown(store):  # noqa: F811
    """fund 10,000 marked, then loses 1,000 unmarked: the book really is 10% down. A new 10,000 Strategy joins it.
    Right: 19,000 against a reference of 21,111.11, 10% down. Today (scaled from the 10,000 mark): 19,000 against
    20,000, 5%: the drawdown understated, so the halt would come late."""
    _make(store, "fund", 10_000)
    assert _mark(store, T0).equity == D(10_000)
    _trade(store, "fund", -1_000, ts=T0 + timedelta(seconds=30))
    _make(store, "new", 10_000)
    st = _mark(store, T0 + timedelta(minutes=1))
    kept = _kept(10_000, 9_000, 19_000)
    assert _four(st) == (D(19_000), kept, kept, kept), _four(st)
    assert _dd(st) == D("0.1"), _dd(st)


def test_a_move_off_the_book_when_only_the_rest_moved_keeps_the_books_real_drawdown(store):  # noqa: F811
    """fund 10,000 and test 20,000 marked at 30,000; then fund loses 1,000 unmarked (the book really is 3.3% down)
    and test, flat and unchanged, moves onto the test account. Right: 9,000 against 9,310.34, 3.3%. Today: 9,000
    against 10,000, 10%."""
    _make(store, "fund", 10_000)
    _make(store, "test", 20_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "desk")
    assert _mark(store, T0).equity == D(30_000)
    _trade(store, "fund", -1_000, ts=T0 + timedelta(seconds=30))
    store.assign_account("test", "paper")
    st = _mark(store, T0 + timedelta(minutes=1))
    kept = _kept(30_000, 29_000, 9_000)
    assert _four(st) == (D(9_000), kept, kept, kept), _four(st)
    assert not st.halted and st.paused_until is None


def test_a_move_into_the_book_after_an_unmarked_loss_keeps_the_rests_drawdown(store):  # noqa: F811
    """fund 10,000 marked alone (test 20,000 sits on the test account, outside); then fund loses 1,000 unmarked (10%
    down) and test moves into the book. Right: 29,000 against 32,222.22, 10%. Today: 29,000 against 30,000, 3.3%."""
    _make(store, "fund", 10_000)
    _make(store, "test", 20_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "paper")  # outside the book
    assert _mark(store, T0).equity == D(10_000)
    _trade(store, "fund", -1_000, ts=T0 + timedelta(seconds=30))
    store.assign_account("test", "desk")  # into the book
    st = _mark(store, T0 + timedelta(minutes=1))
    kept = _kept(10_000, 9_000, 29_000)
    assert _four(st) == (D(29_000), kept, kept, kept), _four(st)
    assert _dd(st) == D("0.1"), _dd(st)


def test_a_book_never_marked_still_moves_at_once_with_no_flow(store):  # noqa: F811
    """Guard, green today and after the fix: with nothing marked there is no scale to keep (Store._book_move), so a
    move lands at once, journals no flow, waits for no poll, and the first mark reads the book as it is."""
    _make(store, "fund", 10_000)
    _make(store, "test", 20_000, instrument="BTC/USD", venue="KRAKEN")
    store.assign_account("test", "desk")
    _trade(store, "test", 5_000)
    store.assign_account("test", "paper")
    assert store.account_of("test") == "paper"  # moved now, not at a poll
    assert [e for e in store.events(None, limit=100) if e["kind"] == "portfolio_flow"] == []
    assert _four(_mark(store, T0)) == (D(10_000),) * 4
