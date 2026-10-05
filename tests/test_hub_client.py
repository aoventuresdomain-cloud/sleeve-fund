"""A paper node fed by its venue's market data hub (v2 P1-1): the bar types it decides on, what it makes of the
hub's messages, and where it finds the hub."""

import pytest

from sleeve_fund.paper.hub_client import Decoder, hub_bar_spec, instrument_from
from sleeve_fund.paper.node import hub_address

BTC = "BTCUSDT-PERP.BINANCE"
T0 = 1_791_223_020_000_000_000


@pytest.mark.parametrize("spec, fed", [
    ("1-MINUTE-LAST-INTERNAL", "1-MINUTE-LAST-EXTERNAL"),  # the hub's own minutes, as they come
    ("15-MINUTE-LAST-INTERNAL", "15-MINUTE-LAST-EXTERNAL"),  # built by the hub client from them
    ("1-HOUR-LAST-INTERNAL", "1-HOUR-LAST-EXTERNAL"),
])
def test_a_strategys_bars_are_the_hubs_minutes_or_built_from_them(spec, fed):
    assert hub_bar_spec(spec) == fed


def test_bars_the_venue_builds_itself_cant_come_from_the_hub():
    with pytest.raises(ValueError, match="1-DAY-LAST-EXTERNAL"):
        hub_bar_spec("1-DAY-LAST-EXTERNAL")


def _bar(ts, c="60000.10", refilled=False, o="60000.00", h="60010.00", l="59990.00", v="1.250"):  # noqa: E741
    return {"t": "bar", "id": BTC, "o": o, "h": h, "l": l, "c": c, "v": v, "ts": ts, "recv": ts, "refilled": refilled}


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


M = 60_000_000_000
E = 1_791_223_500_000_000_000  # 18:05 on 5 Oct 2026: a 5-minute boundary


def test_longer_bars_are_the_minutes_closing_in_them_sent_with_their_last_minute():
    d = Decoder("5-MINUTE-LAST-EXTERNAL")
    assert d(_bar(E - M, c="60000.50"), E - M + 5) is None  # 18:04: the bar to 18:05 began before the node did
    assert d(_bar(E), E + 5) is None  # its last minute: a part bar, not sent
    for k, (h, lo) in enumerate([("60050.00", "59990.00"), ("60010.00", "59900.00"), ("60010.00", "59990.00"),
                                 ("60010.00", "59990.00")], 1):
        assert d(_bar(E + k * M, o=f"6000{k}.00", h=h, l=lo, c=f"6000{k}.50"), E + k * M + 5) is None
    b = d(_bar(E + 5 * M, c="60005.50", v="2.000"), E + 5 * M + 5)  # 18:10 closes the bar to 18:10
    assert str(b.bar_type) == f"{BTC}-5-MINUTE-LAST-EXTERNAL" and b.ts_event == E + 5 * M
    assert (str(b.open), str(b.high), str(b.low), str(b.close), str(b.volume)) == (
        "60001.00", "60050.00", "59900.00", "60005.50", "7.000")


def test_a_bar_whose_last_minute_never_came_is_sent_with_the_next_minute_while_current():
    d = Decoder("5-MINUTE-LAST-EXTERNAL")
    for k in range(1, 6):
        d(_bar(E - 5 * M + k * M), E - 5 * M + k * M + 5)  # a whole bar to 18:05, sent
    for k in range(1, 5):
        d(_bar(E + k * M), E + k * M + 5)  # 18:06 to 18:09; 18:10 never comes
    b = d(_bar(E + 6 * M), E + 6 * M + 5)  # 18:11 arrives a minute after the bar should have closed
    assert b.ts_event == E + 5 * M and str(b.volume) == "5.000"  # stamped at its close, four minutes in it
    assert d.building[BTC].end == E + 10 * M


def test_one_minute_bars_go_through_as_they_come():
    d = Decoder()
    b = d(_bar(E), E + 5)
    assert str(b.bar_type) == f"{BTC}-1-MINUTE-LAST-EXTERNAL" and b.ts_event == E and str(b.close) == "60000.10"


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
