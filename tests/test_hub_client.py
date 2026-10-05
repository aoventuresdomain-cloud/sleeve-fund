"""A paper node fed by its venue's market data hub (v2 P1-1): the bar types it decides on, what it makes of the
hub's messages, and where it finds the hub."""

import pytest

from sleeve_fund.paper.hub_client import Decoder, hub_bar_spec, instrument_from
from sleeve_fund.paper.node import hub_address

BTC = "BTCUSDT-PERP.BINANCE"
T0 = 1_791_223_020_000_000_000


@pytest.mark.parametrize("spec, fed", [
    ("1-MINUTE-LAST-INTERNAL", "1-MINUTE-LAST-EXTERNAL"),  # the hub's own bars, as they come
    ("15-MINUTE-LAST-INTERNAL", "15-MINUTE-LAST-INTERNAL@1-MINUTE-EXTERNAL"),  # built in the node from them
    ("1-HOUR-LAST-INTERNAL", "1-HOUR-LAST-INTERNAL@1-MINUTE-EXTERNAL"),
    ("15-MINUTE-LAST-INTERNAL@1-MINUTE-EXTERNAL", "15-MINUTE-LAST-INTERNAL@1-MINUTE-EXTERNAL"),
])
def test_a_strategys_bars_are_the_hubs_minutes_or_built_from_them(spec, fed):
    assert hub_bar_spec(spec) == fed


def test_bars_the_venue_builds_itself_cant_come_from_the_hub():
    with pytest.raises(ValueError, match="1-DAY-LAST-EXTERNAL"):
        hub_bar_spec("1-DAY-LAST-EXTERNAL")


def _bar(ts, c="60000.10", refilled=False):
    return {"t": "bar", "id": BTC, "o": "60000.00", "h": "60010.00", "l": "59990.00", "c": c, "v": "1.250",
            "ts": ts, "recv": ts, "refilled": refilled}


def test_messages_become_nautilus_data_and_each_bar_is_delivered_once_in_order():
    d = Decoder()
    t = d({"t": "trade", "id": BTC, "px": "60000.10", "qty": "0.005", "side": "SELL", "tid": "7", "ts": T0,
           "recv": T0}, T0 + 1)
    assert (str(t.price), str(t.size), str(t.aggressor_side), t.ts_event, t.ts_init) == ("60000.10", "0.005", "SELL",
                                                                                        T0, T0 + 1)
    q = d({"t": "quote", "id": BTC, "bid": "60000.00", "ask": "60000.10", "bid_qty": "2.0", "ask_qty": "1.5",
           "ts": T0, "recv": T0}, T0 + 1)
    assert (str(q.bid_price), str(q.ask_price)) == ("60000.00", "60000.10")
    b = d(_bar(T0), T0 + 2)
    assert str(b.bar_type) == f"{BTC}-1-MINUTE-LAST-EXTERNAL" and b.ts_event == T0 and str(b.close) == "60000.10"
    assert d(_bar(T0, c="1.00"), T0 + 3) is None  # the same minute again
    assert d(_bar(T0 - 60_000_000_000, refilled=True), T0 + 4) is None  # a refill landing after the live bar moved on
    assert d(_bar(T0 + 60_000_000_000, refilled=True), T0 + 5).ts_event == T0 + 60_000_000_000  # an in-order refill
    assert d({"t": "gap", "id": BTC, "since": T0, "until": T0}, T0) is None and d({"t": "new-kind"}, T0) is None


def test_a_bar_refilled_long_after_its_close_is_not_traded_on():
    """After a hub restart its refill of the missed minutes arrives late: history for the store, not a signal."""
    d, minute = Decoder(), 60_000_000_000
    assert d(_bar(T0, refilled=True), T0 + 91 * 1_000_000_000) is None and d.late == 1
    assert d(_bar(T0 + minute, refilled=True), T0 + minute + 2_000_000_000) is not None  # the latest, just refilled
    assert d.late == 1


def test_the_instrument_comes_from_its_definition():
    from nautilus_trader.model import CryptoPerpetual, Currency, InstrumentId, Price, Quantity, Symbol

    usdt = Currency.from_str("USDT")
    inst = CryptoPerpetual(InstrumentId.from_str(BTC), Symbol("BTCUSDT"), Currency.from_str("BTC"), usdt, usdt, False,
                           2, 3, Price.from_str("0.10"), Quantity.from_str("0.001"), 0, 0)
    assert instrument_from(inst.to_dict()) == inst


def test_the_hub_is_found_from_its_variable_and_absent_means_a_venue_connection():
    assert hub_address("binance", {"HUB_BINANCE": "hub-binance:7700"}) == ("hub-binance", 7700)
    assert hub_address("binance", {}) is None and hub_address("kraken", {"HUB_BINANCE": "x:1"}) is None
    with pytest.raises(ValueError, match="host:port"):
        hub_address("binance", {"HUB_BINANCE": "hub-binance"})
