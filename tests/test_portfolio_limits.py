"""v2 P2-2: the portfolio limits as pure functions (sleeve_fund.portfolio.limits), against the Independent Quant
Advisor's ruling of 6 Oct ~21:54: trim to the largest quantity within every limit, refuse below the venue minimum,
bind only when the order raises a figure past its limit, and the book's halt and pause."""

import math

import pytest

from sleeve_fund.portfolio import Book, Holding, Intent, book_breach, decide
from sleeve_fund.risk import PORTFOLIO

BOOK = 20_000.0
PX = 60_000.0


def _buy(qty, *, side=1, lev=2.0, risk=0.02, underlying="BTC", step=0.001, min_qty=0.001):
    """An order at 60,000: margin at `lev`, risk `risk` of the price per unit."""
    return Intent(underlying, side, qty, PX, PX / lev, PX * risk, step, min_qty)


def _held(notional, *, lev=2.0, risk=0.0, underlying="BTC"):
    return Holding(underlying, notional, abs(notional) / lev, risk)


def test_an_order_within_every_limit_is_approved_whole():
    d = decide(Book(BOOK), _buy(0.1), PORTFOLIO)  # 6,000 notional = 0.3x; margin 15%; risk 0.6%
    assert (d.outcome, d.qty, d.limit) == ("approved", 0.1, None)


def test_net_per_instrument_binds_first_on_one_underlying_and_trims_to_the_step():
    """Advisor EXPECT: 10,000 net on a 20,000 book. 8,000 long already: 2,000 of room, 0.0333 BTC, down to 0.033."""
    d = decide(Book(BOOK, (_held(8_000),)), _buy(0.1), PORTFOLIO)
    assert (d.outcome, d.limit, d.qty) == ("trimmed", "net", 0.033)
    assert d.figures["net"]["after"] <= 0.5
    assert "net exposure in BTC would be 0.70x of the book with the whole order, above the 0.50x limit" in d.why


@pytest.mark.parametrize("held, intent, limit", [
    # gross 1.5x = 30,000: 29,000 across three underlyings, at 5x so margin has room (at 2x margin binds first)
    ((_held(9_000, lev=5), _held(-9_000, lev=5, underlying="ETH"), _held(11_000, lev=5, underlying="SOL")),
     _buy(0.1, lev=5), "gross"),
    # margin 50% = 10,000: spot positions count their full notional
    ((_held(9_500, lev=1.0, underlying="ETH"),), _buy(0.1, lev=1.0), "margin"),
    # open risk 5% = 1,000: 900 already
    ((_held(1_000, risk=900.0, underlying="ETH"),), _buy(0.1, risk=0.05), "open_risk"),
])
def test_each_limit_binds_in_turn(held, intent, limit):
    d = decide(Book(BOOK, held), intent, PORTFOLIO)
    assert d.limit == limit and d.outcome in ("trimmed", "refused")
    assert d.figures[limit]["after"] <= d.figures[limit]["cap"] + 1e-12


def test_an_order_against_the_net_is_never_held_back_by_the_net_limit():
    """9,000 long: a 0.25 BTC short (15,000) takes the net to -6,000, inside 10,000 in size, so it passes whole."""
    d = decide(Book(BOOK, (_held(9_000, lev=5),)), _buy(0.25, side=-1, lev=5), PORTFOLIO)
    assert (d.outcome, d.qty) == ("approved", 0.25) and d.figures["net"]["after"] == pytest.approx(-0.3)


def test_marks_alone_past_a_limit_force_nothing_down_but_an_order_that_raises_it_waits():
    over = Book(BOOK, (_held(12_000),))  # 0.6x net from marks
    assert decide(over, _buy(0.05, side=-1), PORTFOLIO).limit is None  # brings it down
    d = decide(over, _buy(0.01), PORTFOLIO)
    assert (d.outcome, d.limit, d.qty) == ("refused", "net", 0.0)


def test_exactly_at_a_limit_passes():
    d = decide(Book(BOOK, (_held(4_000),)), _buy(0.1), PORTFOLIO)  # 4,000 + 6,000 = 10,000 = 0.5x
    assert (d.outcome, d.qty) == ("approved", 0.1)


def test_room_below_the_venue_minimum_is_refused():
    d = decide(Book(BOOK, (_held(9_990),)), _buy(0.1, min_qty=0.001), PORTFOLIO)  # 10 of room: 0.000166 BTC
    assert (d.outcome, d.qty, d.limit) == ("refused", 0.0, "net") and "below the venue minimum" in d.why


@pytest.mark.parametrize("book, intent", [
    (Book(0.0), _buy(0.1)), (Book(math.nan), _buy(0.1)), (Book(BOOK, (_held(math.inf),)), _buy(0.1)),
    (Book(BOOK), _buy(0.0)), (Book(BOOK), _buy(0.1, side=0)), (Book(BOOK), _buy(0.1, risk=math.nan)),
])
def test_bad_inputs_fail_closed(book, intent):
    d = decide(book, intent, PORTFOLIO)
    assert (d.outcome, d.qty) == ("refused", 0.0)


def test_a_tie_names_the_same_limit_every_time():
    # gross room 30,000 - 21,000 = 9,000 and net room 10,000 - 1,000 = 9,000: gross comes first in LIMITS
    held = (_held(1_000, lev=5), _held(10_000, lev=5, underlying="ETH"), _held(-10_000, lev=5, underlying="SOL"))
    assert decide(Book(BOOK, held), _buy(0.5, lev=5), PORTFOLIO).limit == "gross"


def test_the_book_halts_at_15_percent_below_its_reference_and_pauses_at_3_percent_on_the_day():
    assert book_breach(PORTFOLIO, 17_000.0, 20_000.0, 17_100.0).action == "halt"
    assert book_breach(PORTFOLIO, 19_400.0, 20_000.0, 20_000.0).action == "pause"
    assert book_breach(PORTFOLIO, 19_500.0, 20_000.0, 20_000.0) is None
    # after a PM Resume the reference is the book at the resume: no halt again at once
    assert book_breach(PORTFOLIO, 16_900.0, 16_900.0, 16_900.0) is None
    assert book_breach(PORTFOLIO, math.nan, 20_000.0, 20_000.0).action == "halt"
