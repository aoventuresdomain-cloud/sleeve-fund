"""F-ASSIGN-STALE (CR on #286, HoE 14 Oct): a Strategy moved off the book (onto the test account its venue names,
F229-6) is withdrawn at its equity NOW, but apply_flow scales the book's figures as of the LAST MARK. P&L it made or
lost since that mark is in neither, so the withdrawal takes it from the rest of the book: a loss since the mark reads
as a deeper drawdown of the rest (and can halt the fund falsely), a gain as a gain of the rest.

The right figures are the ones a mark taken at the move, then the withdrawal, would give: the book's real P&L while
the Strategy was in it is kept, as a percentage, as apply_flow keeps it (Advisor MUST, 14 Oct 09:45 UK). That is what
#286 does for a put-away Strategy (Supervisor._withdraw_flat after a mark); the fix reuses it. Red first on 22d36f8."""

from datetime import timedelta
from decimal import ROUND_HALF_EVEN, Decimal as D, localcontext

from tests.test_halt_ref import T0, _four, _make, _mark, _trade, store  # noqa: F401, F811


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


def _kept(peak, at_move, after):
    """A figure scaled as apply_flow scales it: peak x after / at_move, to the column's 18 places."""
    with localcontext() as ctx:
        ctx.prec = 60
        return (D(peak) * D(after) / D(at_move)).quantize(D("1e-18"), ROUND_HALF_EVEN)


def test_a_gain_since_the_mark_leaves_with_the_strategy_and_the_rest_reads_unchanged(store):  # noqa: F811
    st = _moved_off(store, 5_000)  # the book really was 35,000 at the move; 25,000 of it leaves
    assert _four(st) == (D(10_000),) * 4, _four(st)
    assert not st.halted and st.paused_until is None


def test_a_loss_since_the_mark_is_the_books_real_drawdown_not_a_deeper_one(store):  # noqa: F811
    """The book really fell 30,000 -> 27,000 (10%) while `test` was in it; 17,000 leaves. The rest reads 10% down,
    under the 15% halt. Today (the flow scaled from the 30,000 mark): the rest reads 23% down."""
    st = _moved_off(store, -3_000)
    kept = _kept(30_000, 27_000, 10_000)
    assert _four(st) == (D(10_000), kept, kept, kept), _four(st)
    assert not st.halted and st.paused_until is None


def test_a_halt_from_a_big_loss_since_the_mark_states_the_real_drawdown(store):  # noqa: F811
    """30,000 -> 22,000 while it was in the book is a real 26.7% drawdown, and halts. It must read 26.7%, not the
    44.4% the stale flow makes of it."""
    st = _moved_off(store, -8_000)
    assert st.halted and "26.7%" in st.halted and "44.4%" not in st.halted, st.halted
