"""QA #206 (PE1 asked: a cell for the "add" branch of _restored_rounding). No model on main can add to a PERP position
(target-weight rebalancing is spot-only, and _restore exists only on margin), so the branch is unreachable end to end
today; it is pinned here at the method, on the property it exists for:

after a restore at the venue's price, an ADD, then closes (partial, then the rest), the venue's cent-rounded profit
plus every adjustment booked (the restore's gap, then each close's rounding difference) equals the cent-rounded profit
of the run straight through, which never restarted (average from the journal's entry and the add)."""

from types import SimpleNamespace

import pytest
from nautilus_trader.model import Currency, Money

from sleeve_fund.strategies.base import LongFlatStrategy

USD = Currency.from_str("USD")


def _cent(x):
    return Money(x, USD).as_double()


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("entry, restore_px, held, add_px, add_qty, closes", [
    (60_000.0, 60_731.37, 0.0123, 61_017.91, 0.0071, [(0.0101, 61_377.77), (0.0093, 60_903.13)]),
    (100.0, 101.0, 1.0, 103.0, 1.0, [(2.0, 104.37)]),
    (2_500.0, 2_487.135, 0.333, 2_490.007, 0.111, [(0.1, 2_511.119), (0.344, 2_477.777)]),
])
def test_an_add_to_a_restored_position_then_closes_book_the_straight_through_cents(side, entry, restore_px, held,
                                                                                  add_px, add_qty, closes):
    s = SimpleNamespace(_cash_adj=0.0, instrument=SimpleNamespace(settlement_currency=USD, quote_currency=USD))
    # What the restore booked (base.py on the restore's fill): its gap from the journal's entry, unrounded.
    s._cash_adj += side * held * (restore_px - entry)
    s._restored = {"side": side, "n": held, "avg": restore_px, "gap": restore_px - entry}
    venue_avg, straight_avg, n = restore_px, entry, held
    LongFlatStrategy._restored_rounding(s, side, add_qty, add_px)  # the add
    venue_avg = (venue_avg * n + add_px * add_qty) / (n + add_qty)
    straight_avg = (straight_avg * n + add_px * add_qty) / (n + add_qty)
    n += add_qty
    assert s._restored["avg"] == pytest.approx(venue_avg, abs=1e-9)
    assert s._restored["gap"] == pytest.approx(venue_avg - straight_avg, abs=1e-9)
    venue_booked = straight_booked = 0.0
    for q, px in closes:
        LongFlatStrategy._restored_rounding(s, -side, q, px)
        venue_booked += _cent(side * q * (px - venue_avg))  # the venue's own per-fill cent rounding
        straight_booked += _cent(side * q * (px - straight_avg))  # the run straight through
    assert venue_booked + s._cash_adj == pytest.approx(straight_booked, abs=1e-9), \
        (venue_booked, s._cash_adj, straight_booked)
    assert s._restored is None  # fully closed: the restored state is cleared
