"""v2 P2-2: the portfolio limits as pure functions (sleeve_fund.portfolio.limits), against the Independent Quant
Advisor's ruling of 6 Oct ~21:54: trim to the largest quantity within every limit, reject below the venue minimum,
bind only when the order raises a figure past its limit, and the book's halt and pause. Money is exact Decimal
(day-0 interface note v4); these are PE1's cells with QA's names."""

from decimal import Decimal as D

import pytest

from sleeve_fund.portfolio import Book, Holding, Intent, book_breach, decide
from sleeve_fund.risk import PORTFOLIO

BOOK = D("20000")
PX = D("60000")


def _buy(qty, *, side=1, lev="2", risk="0.02", underlying="BTC", step="0.001", min_qty="0.001"):
    """An order at 60,000: margin at `lev`, risk `risk` of the price per unit."""
    return Intent(underlying, side, D(qty), PX, PX / D(lev), PX * D(risk), D(step), D(min_qty))


def _held(notional, *, lev="2", risk="0", underlying="BTC"):
    n = D(notional)
    return Holding(underlying, n, abs(n) / D(lev), D(risk))


def test_an_order_within_every_limit_is_approved_whole():
    d = decide(Book(BOOK), _buy("0.1"), PORTFOLIO)  # 6,000 notional = 0.3x; margin 15%; risk 0.6%
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("approved", D("0.1"), None)


def test_net_per_instrument_binds_first_on_one_underlying_and_trims_to_the_step():
    """Advisor EXPECT: 10,000 net on a 20,000 book. 8,000 long already: 2,000 of room, 0.0333 BTC, down to 0.033."""
    d = decide(Book(BOOK, (_held("8000"),)), _buy("0.1"), PORTFOLIO)
    assert (d.outcome, d.limit_hit, d.approved_qty) == ("trimmed", "net_instrument", D("0.033"))
    assert d.figures["net_instrument"]["after"] <= 0.5
    assert "net exposure in BTC would be 0.70x of the book with the whole order, above the 0.50x limit" in d.reason


@pytest.mark.parametrize("held, intent, limit", [
    # gross 1.5x = 30,000: 29,000 across three underlyings, at 5x so margin has room (at 2x margin binds first)
    ((_held("9000", lev="5"), _held("-9000", lev="5", underlying="ETH"), _held("11000", lev="5", underlying="SOL")),
     _buy("0.1", lev="5"), "gross"),
    # margin 50% = 10,000: spot positions count their full notional
    ((_held("9500", lev="1", underlying="ETH"),), _buy("0.1", lev="1"), "margin"),
    # open risk 5% = 1,000: 900 already
    ((_held("1000", risk="900", underlying="ETH"),), _buy("0.1", risk="0.05"), "open_risk"),
])
def test_each_limit_binds_in_turn(held, intent, limit):
    d = decide(Book(BOOK, held), intent, PORTFOLIO)
    assert d.limit_hit == limit and d.outcome in ("trimmed", "rejected")
    assert d.figures[limit]["after"] <= d.figures[limit]["cap"] + 1e-12


def test_an_order_against_the_net_is_never_held_back_by_the_net_limit():
    """9,000 long: a 0.25 BTC short (15,000) takes the net to -6,000, inside 10,000 in size, so it passes whole."""
    d = decide(Book(BOOK, (_held("9000", lev="5"),)), _buy("0.25", side=-1, lev="5"), PORTFOLIO)
    assert (d.outcome, d.approved_qty) == ("approved", D("0.25"))
    assert d.figures["net_instrument"]["after"] == pytest.approx(-0.3)


def test_marks_alone_past_a_limit_force_nothing_down_but_an_order_that_raises_it_waits():
    over = Book(BOOK, (_held("12000"),))  # 0.6x net from marks
    assert decide(over, _buy("0.05", side=-1), PORTFOLIO).limit_hit is None  # brings it down
    d = decide(over, _buy("0.01"), PORTFOLIO)
    assert (d.outcome, d.limit_hit, d.approved_qty) == ("rejected", "net_instrument", D(0))


def test_exactly_at_a_limit_passes():
    d = decide(Book(BOOK, (_held("4000"),)), _buy("0.1"), PORTFOLIO)  # 4,000 + 6,000 = 10,000 = 0.5x
    assert (d.outcome, d.approved_qty) == ("approved", D("0.1"))


def test_room_below_the_venue_minimum_is_rejected():
    d = decide(Book(BOOK, (_held("9990"),)), _buy("0.1", min_qty="0.001"), PORTFOLIO)  # 10 of room: 0.000166 BTC
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("rejected", D(0), "net_instrument")
    assert "below the venue minimum" in d.reason


@pytest.mark.parametrize("book, intent", [
    (Book(D(0)), _buy("0.1")), (Book(BOOK), _buy("0")), (Book(BOOK), _buy("0.1", side=0)),
    (Book(BOOK), _buy("0.1", risk="-0.01")),
])
def test_unusable_inputs_fail_closed(book, intent):
    d = decide(book, intent, PORTFOLIO)
    assert (d.outcome, d.approved_qty) == ("rejected", D(0))


@pytest.mark.parametrize("build", [
    lambda: Book(20_000.0), lambda: Holding("BTC", 1_000.0, D(500), D(0)),
    lambda: Intent("BTC", 1, D("0.1"), 60_000.0, PX, PX, D("0.001"), D("0.001")),
])
def test_float_money_is_refused_with_a_type_error(build):
    """Day-0 note v4 (PE2 and QD, 17:49 UK): money is Decimal, int or str; a float never gets in silently."""
    with pytest.raises(TypeError, match="money"):
        build()


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_are_refused_at_the_boundary(bad):
    with pytest.raises(ValueError, match="finite"):
        Holding("BTC", D(bad), D(0), D(0))


def test_a_tie_names_the_same_limit_every_time():
    # gross room 30,000 - 21,000 = 9,000 and net room 10,000 - 1,000 = 9,000: gross comes first in LIMITS
    held = (_held("1000", lev="5"), _held("10000", lev="5", underlying="ETH"),
            _held("-10000", lev="5", underlying="SOL"))
    assert decide(Book(BOOK, held), _buy("0.5", lev="5"), PORTFOLIO).limit_hit == "gross"


def test_the_book_halts_at_15_percent_below_its_reference_and_pauses_at_3_percent_on_the_day():
    assert book_breach(PORTFOLIO, D(17_000), D(20_000), D(17_100)).action == "halt"
    assert book_breach(PORTFOLIO, D(19_400), D(20_000), D(20_000)).action == "pause"
    assert book_breach(PORTFOLIO, D(19_500), D(20_000), D(20_000)) is None
    # after a PM Resume the reference is the book at the resume: no halt again at once
    assert book_breach(PORTFOLIO, D(16_900), D(16_900), D(16_900)) is None
    assert book_breach(PORTFOLIO, D("NaN"), D(20_000), D(20_000)).action == "halt"


@pytest.mark.parametrize("held, margin_per_unit, risk_per_unit", [
    (Holding("ETH", D(1_000), D(100), D(500)), D(100), D(0)),  # open risk at 5% already; risk per unit 0
    (Holding("ETH", D(1_000), D(6_000), D(0)), D(0), D(10)),  # margin at 60% already; margin per unit 0
])
def test_cr208_1_a_zero_per_unit_margin_or_risk_is_refused_never_a_limit_switched_off(held, margin_per_unit,
                                                                                        risk_per_unit):
    """CR208-1: every order that raises a position has margin and risk above 0, so a zero is a broken input. Taken
    as is, it would drop that limit from the check and approve the whole order past the PM's 5% or 50%."""
    intent = Intent("BTC", 1, D(10), D(100), margin_per_unit, risk_per_unit, D(1), D(1))
    d = decide(Book(D(10_000), (held,)), intent, PORTFOLIO)
    assert (d.outcome, d.approved_qty) == ("rejected", D(0)) and "can't be used" in d.reason


def test_cr208_2_float_money_into_the_book_check_halts_not_raises():
    """CR208-2: a float for money is a TypeError at the boundary; the book check turns it into the halt it documents,
    like NaN, rather than raising past the supervisor's marking."""
    b = book_breach(PORTFOLIO, 17_000.0, D(20_000), D(20_000))
    assert b.action == "halt" and "aren't usable" in b.reason


def test_cr208_3_a_request_under_the_venue_minimum_is_rejected_below_min_even_when_nothing_binds():
    """Before: an order of 0.0005 against a 0.001 minimum, with room under every limit, was approved and sent."""
    d = decide(Book(BOOK), _buy("0.0005", step="0.0001"), PORTFOLIO)
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("rejected", D("0"), "below_min")
    assert "below the venue minimum of 0.001" in d.reason
    assert decide(Book(BOOK), _buy("0.001"), PORTFOLIO).outcome == "approved"  # exactly the minimum passes


def test_qa_f211_3_an_unbound_order_is_floored_to_the_step_too():
    """Before: with nothing binding, an off-step 0.1005 was approved as asked; only the trimmed path floored."""
    d = decide(Book(BOOK), _buy("0.1005"), PORTFOLIO)
    assert (d.outcome, d.approved_qty, d.requested_qty, d.limit_hit) == ("approved", D("0.100"), D("0.1005"), None)
    d = decide(Book(BOOK), _buy("0.0004", step="0.001", min_qty="0"), PORTFOLIO)  # floors to nothing
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("rejected", D("0"), "below_min")
