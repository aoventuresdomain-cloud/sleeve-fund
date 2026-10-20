"""Head 4 step 6 gaps 2 and 3, the Advisor's rulings PAGE-DD and PAGE-SIZE (20 Oct, through the HoE).

PAGE-DD: inside the fund, Home's and Risk's drawdown headline is the one the fund's 15% halt (T5) acts on: the gate's
last mark against its reference, as portfolio.limits.book_breach measures it, "Drawdown from peak x% (halt at 15%)".
The daily-close figure stays as a labelled second line, never unlabelled. QD2's case B: the gate at 99,000 under a
high-water mark of 101,000 is 1.98%, where the pages read 1.0% from the daily closes.

PAGE-SIZE: the Strategy page's "Position size 33% of the fund" (the profile's max_position_pct, which doesn't size a
fund-held entry) becomes the entry's own figures from its decision journal, never worked again. Head 4's journal holds
the order's quantity, the close it decided at and the fund equity it was sized off (F-FC-4), so the position value's
share of the fund shows, with margin leverage on a perp. T1's room and the loss at the stop aren't journaled until
B-24, so those two lines are left out. On Postgres too (TEST_DATABASE_URL)."""

import re
from dataclasses import replace
from decimal import Decimal

from test_fe_b20_fund_held_wording import (  # noqa: F401
    ENTRY,
    FEE,
    MARK,
    PERP,
    QTY,
    _hold,
    _long,
    _mark,
    _summaries,
    _words,
    c,
)
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)


def _gate(store):  # noqa: F811
    from sleeve_fund.paper.portfolio_gate import PaperGate

    return PaperGate(store).ledger.state()


def test_fund_held_the_drawdown_headline_is_the_halts_own(c, store):  # noqa: F811
    _hold(store)
    _long(store)
    _mark(store, 99_000, hwm=101_000)
    for path in ("/", "/risk"):
        words, _ = _words(c, path)
        assert "Drawdown from peak 1.98% (halt at 15%)" in words, (path, words[:600])
        daily = re.search(r"from daily-close peak (\d+\.\d)%", words)
        assert daily, (path, "the daily-close figure is labelled")
        assert not re.search(r"Book drawdown \d", words) and not re.search(r"Drawdown \d", words), path


def test_the_headline_equals_the_gates_value_where_the_halt_fires(c, store):  # noqa: F811
    """The page's figure is the one book_breach halts at: a limit at the figure fires, a hair above doesn't."""
    from sleeve_fund.portfolio.limits import book_breach
    from sleeve_fund.risk import PORTFOLIO

    _hold(store)
    _long(store)
    _mark(store, 99_000, hwm=101_000)
    st = _gate(store)
    words, _ = _words(c, "/")
    shown = Decimal(re.search(r"Drawdown from peak (\d+\.\d\d)%", words).group(1)) / 100
    assert shown == round(1 - st.equity / max(st.reference(), st.equity), 4)
    at = book_breach(replace(PORTFOLIO, drawdown=float(shown), daily_loss=None), st.equity, st.reference(), st.equity)
    above = book_breach(replace(PORTFOLIO, drawdown=float(shown + Decimal("0.0001")), daily_loss=None), st.equity,
                        st.reference(), st.equity)
    assert at is not None and at.action == "halt" and above is None


def test_not_held_the_drawdown_reads_as_before(c, store):  # noqa: F811
    _long(store, "own-long", balance=10_000)
    words, _ = _words(c, "/")
    assert "Drawdown from peak" not in words and re.search(r"Drawdown \d+\.\d%", words)


def _journaled_long(store, name="sized-long", fund_equity=100_000.0, sized=True):  # noqa: F811
    """A fund-held perp long whose entry journals what it was sized off, as base.py's entry does."""
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0,
                        params={"rise": 0.01, "dip": 0.005, **PERP})
    signal = {"close": ENTRY, "side": "long", "sized_by": "risk profile cap", "budget": 6_000.0,
              **({"fund_equity": fund_equity} if sized else {})}
    store.record_order(name, order_id=f"{name}-1", side="BUY", qty=QTY, intent="entry", reason="dipped 0.5%",
                       signal=signal)
    store.update_order(f"{name}-1", fill_qty=QTY, fill_px=ENTRY, fee=FEE)
    store.record_fill(name, side="BUY", qty=QTY, price=ENTRY, fee=FEE, order_id=f"{name}-1", trade_id=f"{name}-T")
    cash = -QTY * ENTRY - FEE
    store.record_equity(name, equity=cash + QTY * MARK, cash=cash, qty=QTY, price=MARK, benchmark=0)
    return store.sleeve(name)


def _size_cells(c, name):  # noqa: F811
    _, page = _words(c, f"/sleeves/{name}")
    side = re.search(r"<dt>Position size</dt><dd>(.*?)</dd></div>", page, re.DOTALL)
    table = re.search(r'<td data-position-size>(.*?)</td></tr>', page, re.DOTALL)
    assert side and table
    plain = [" ".join(re.sub(r"<[^>]+>", " ", m.group(1)).split()) for m in (side, table)]
    return plain


def test_fund_held_the_position_size_is_the_entrys_journaled_share_of_the_fund(c, store):  # noqa: F811
    from sleeve_fund.dashboard import trading

    _hold(store)
    s = _journaled_long(store, fund_equity=75_000.0)  # not the fund's equity now: the figure is the journal's
    side, table = _size_cells(c, s.name)
    share = QTY * ENTRY / 75_000.0  # 8%: the order's quantity at the close it decided at, over its fund equity
    for cell in (side, table):
        assert cell.startswith(f"Position value: {share:.0%} of fund equity (share of fund {share:.2f}×)"), cell
        assert "33%" not in cell
        assert "room T1 allows" not in cell and "if the stop is hit" not in cell  # not journaled in Head 4 (B-24)
    xs, _ = _summaries(store)  # the margin leverage is the position's own, as the position card shows it
    pos = trading.book_positions(store, xs)["rows"][0]
    assert f"margin leverage {pos['leverage']:.2f}×" in side and f"margin leverage {pos['leverage']:.2f}×" in table


def test_fund_held_an_entry_journaled_without_its_fund_equity_shows_no_size_figure(c, store):  # noqa: F811
    _hold(store)
    s = _journaled_long(store, sized=False)
    for cell in _size_cells(c, s.name):
        assert "33%" not in cell and "Position value" not in cell and "isn't in its entry's record" in cell, cell


def test_not_held_the_position_size_reads_as_before(c, store):  # noqa: F811
    s = _long(store, "own-long", balance=10_000)
    side, table = _size_cells(c, s.name)
    assert side.startswith("33% of the fund") and table == "33% of the fund"
