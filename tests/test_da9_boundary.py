"""DA-9: the paper engine's one write boundary (EngineJournal) and the float view (Advisor 7 Oct 20:32 and 21:17 UK).
A float becomes the decimal it prints as; a quantity must already lie on the lot, and one that doesn't is refused
loudly (ValueError and an error event), never rounded; the float view only changes what is read."""

from decimal import Decimal

import pytest

from sleeve_fund.exact import EngineJournal, FloatView
from sleeve_fund.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path}/t.db")
    s.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                    starting_balance=10_000)
    return s


def _order(j, qty):
    j.record_order("s1", order_id="O-1", side="BUY", qty=qty, intent="entry", reason="test")


def test_a_quantity_off_the_lot_is_refused_and_raised_never_rounded(store):
    j = EngineJournal(store)
    j.set_grid(lot_decimals=3)
    with pytest.raises(ValueError, match="not a whole number of lots"):
        _order(j, 0.1234)
    assert store.orders("s1") == []  # nothing written, so nothing rounded
    (e,) = [e for e in store.events("s1", min_level="error") if e["kind"] == "qty_off_lot"]
    assert "0.1234" in e["message"]


def test_a_venue_quantity_read_as_a_float_is_kept_as_the_venue_wrote_it(store):
    """303.82504499 on an 8-decimal lot can reach the engine as 303.82504499000004: a few float steps off, so it is
    the lot multiple, and the engine's quantity is unchanged."""
    j = EngineJournal(store)
    j.set_grid(lot_decimals=8)
    _order(j, 303.82504499000004)
    assert store.orders("s1")[0]["qty"] == Decimal("303.82504499")


def test_a_float_becomes_the_decimal_it_prints_as(store):
    j = EngineJournal(store)
    j.set_grid(lot_decimals=3)
    _order(j, 0.1)
    j.book_fill("s1", side="BUY", qty=0.1, price=60_000.1, fee=0.3, order_id="O-1", trade_id="T-1")
    (f,) = store.fills("s1")
    assert (f["qty"], f["price"], f["fee"]) == (Decimal("0.1"), Decimal("60000.1"), Decimal("0.3"))
    assert store.orders("s1")[0]["filled_qty"] == Decimal("0.1")


def test_reads_through_the_engine_journal_and_the_float_view_are_floats(store):
    j = EngineJournal(store)
    j.set_grid(lot_decimals=3)
    _order(j, 0.1)
    for view in (j, FloatView(store)):
        (o,) = view.orders("s1")
        assert type(o["qty"]) is float and o["qty"] == 0.1


def test_the_float_view_passes_a_write_through_untouched():
    seen = []

    class Spy:
        def record_equity(self, *args, **kwargs):
            seen.append((args, kwargs))

    exact = Decimal("10000.123456789012345678")
    FloatView(Spy()).record_equity("s1", equity=exact)
    assert seen == [(("s1",), {"equity": exact})] and type(seen[0][1]["equity"]) is Decimal


def test_an_off_lot_fill_on_an_order_names_the_order(store):
    j = EngineJournal(store)
    j.set_grid(lot_decimals=3)
    _order(j, 0.1)
    with pytest.raises(ValueError, match="on order O-1"):
        j.update_order("O-1", fill_qty=0.0505, fill_px=60_000.0, fee=0.1)
    assert store.orders("s1")[0]["filled_qty"] == 0


def test_the_cores_get_a_plain_decimal_from_a_journal_figure():
    """money() strips the journal's float-tolerant Money, so a core mixing in a float still fails loudly (CR207-1)."""
    from sleeve_fund.money import Money, money

    d = money(Money("1.5"))
    assert type(d) is Decimal and d == Decimal("1.5")
    with pytest.raises(TypeError):
        d + 0.1


def test_a_journal_figure_compares_with_none_like_any_value():
    from sleeve_fund.money import Money

    assert (Money("1") == None) is False and (Money("1") != None) is True  # noqa: E711 - the comparison under test
    assert Money("1") not in [None] and Money("1") in [None, Money("1")]
