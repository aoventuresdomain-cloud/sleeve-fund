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


def test_a_bar_refilled_long_after_its_close_is_sent_stamped_late_for_the_strategys_late_rule():
    """After a hub restart its refill of the missed minutes arrives late. Each is still sent, so indicators see
    every bar a backtest's do and a stop breached in it runs; its receive stamp shows the lag, and the strategy
    only exits on it (QA P1-C1, #146)."""
    d, minute = Decoder(), 60_000_000_000
    b = d(_bar(T0, refilled=True), T0 + 91 * 1_000_000_000)
    assert b.ts_event == T0 and b.ts_init - b.ts_event == 91 * 1_000_000_000 and d.late == 1
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


def test_a_minute_refilled_late_joins_the_bar_still_being_built():
    """A late minute is history as a bar of its own, but the longer bar it closes in must hold it, or that bar
    differs from the one a backtest builds from the store."""
    d = Decoder("5-MINUTE-LAST-EXTERNAL")
    for k in range(1, 6):
        d(_bar(E - 5 * M + k * M), E - 5 * M + k * M + 5)  # a whole bar to 18:05, sent
    d(_bar(E + M), E + M + 5)
    d(_bar(E + 3 * M), E + 3 * M + 5)  # 18:08 live; the hub missed 18:07
    assert d(_bar(E + 2 * M, h="60100.00", v="3.000", refilled=True), E + 3 * M + 40 * 10**9) is None  # 100 s late
    d(_bar(E + 4 * M), E + 4 * M + 5)
    b = d(_bar(E + 5 * M), E + 5 * M + 5)
    assert (str(b.high), str(b.volume)) == ("60100.00", "8.000") and d.late == 0  # five minutes, the refill in them
    d(_bar(E + 5 * M + M), E + 6 * M + 5)
    assert d(_bar(E + 5 * M, refilled=True), E + 6 * M + 10) is None  # a minute of a bar already sent: history
    assert d.building[BTC].minutes.keys() == {E + 6 * M}


def test_minutes_refilled_in_order_after_an_outage_build_the_bar_and_a_late_bar_is_sent_and_told():
    said = []
    d = Decoder("5-MINUTE-LAST-EXTERNAL", lambda *a: said.append(a))
    for k in range(1, 6):
        d(_bar(E - 5 * M + k * M), E - 5 * M + k * M + 5)
    d(_bar(E + M), E + M + 5)  # 18:06; then the hub is away until 18:13
    back = E + 8 * M
    sent = [b for k in range(2, 8) if (b := d(_bar(E + k * M, refilled=True), back)) is not None]  # 18:07 to 18:12
    assert [b.ts_event for b in sent] == [E + 5 * M] and str(sent[0].volume) == "6.250"  # the bar to 18:10, whole
    assert d.late == 1  # complete only at 18:13: sent for exits only, and told as it is sent
    assert said == [("warning", "bar_late", f"{BTC}: the bar closing 05 Oct 18:10 UTC was complete only 180 s after "
                     "its close, over the 90 s limit (refilled after the feed was away): sent for exits only, no new "
                     "entries")]
    assert d.building[BTC].minutes.keys() == {E + 6 * M, E + 7 * M}  # the next bar's refilled minutes kept
    d(_bar(E + 8 * M), back + 5)
    d(_bar(E + 9 * M), E + 9 * M + 5)
    b = d(_bar(E + 10 * M), E + 10 * M + 5)
    assert b.ts_event == E + 10 * M and str(b.volume) == "6.250"  # whole: two refilled minutes and three live
    assert len(said) == 1  # a run of one late bar: nothing more to say once the feed is current


def test_a_run_of_late_bars_is_told_as_it_starts_and_its_extent_once_the_feed_is_current():
    said = []
    d = Decoder("1-MINUTE-LAST-EXTERNAL", lambda *a: said.append(a))
    back = E + 10 * M
    for k in range(0, 3):  # refilled 18:05 to 18:07, all over 90 s late
        assert d(_bar(E + k * M, refilled=True), back) is not None
    assert [k for _, k, _ in said] == ["bar_late"] and "18:05" in said[0][2]  # on record even if nothing follows
    assert d(_bar(E + 3 * M), E + 3 * M + 5) is not None  # 18:08, current
    assert said[1] == ("warning", "bar_late", f"{BTC}: 3 bars closing 05 Oct 18:05 UTC to 05 Oct 18:07 UTC came over "
                       "90 s after their close, exits only; the feed is current again")


def test_a_bar_sent_with_minutes_missing_is_told():
    said = []
    d = Decoder("5-MINUTE-LAST-EXTERNAL", lambda *a: said.append(a))
    for k in (1, 2, 4, 5):
        d(_bar(E - 5 * M + k * M), E - 5 * M + k * M + 5)
    d(_bar(E - 5 * M + 3 * M), E + 5)  # 18:03 refilled after the bar to 18:05 was sent: too late for it
    assert said == [("warning", "bar_incomplete",
                     f"{BTC}: the bar closing 05 Oct 18:05 UTC was sent missing 1 of its 5 minutes")]


def test_the_client_reconnects_whatever_broke_its_stream_and_says_so_once(monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace

    from sleeve_fund.paper import hub_client

    monkeypatch.setattr(hub_client, "RECONNECT_SECONDS", (0,))
    monkeypatch.setattr(hub_client, "HEALTHY_SECONDS", 0)  # back as soon as it flows

    def stream(*lines):
        r = asyncio.StreamReader()
        for line in lines:
            r.feed_data(line + b"\n")
        r.feed_eof()
        return r

    trade = json.dumps({"t": "trade", "id": BTC, "px": "60000.10", "qty": "0.005", "side": "SELL", "tid": "7",
                        "ts": T0, "recv": T0}).encode()
    got, said, opens = [], [], []

    async def _open():
        opens.append(1)
        if len(opens) == 1:
            raise ConnectionRefusedError("the hub is restarting")
        if len(opens) == 2:
            return stream(trade), {}
        client._closing = True
        raise OSError("stopped")

    client = SimpleNamespace(_closing=False, clock=SimpleNamespace(timestamp_ns=lambda: T0 + 1), decode=Decoder(),
                             _handle_data=got.append, report=lambda *a: said.append(a), _open=_open,
                             last_heartbeat_ns=0, venue_up=False)
    async def run():
        await hub_client.HubDataClient._read(client, stream(b"{not json"))

    asyncio.run(run())
    assert len(got) == 1 and got[0].ts_event == T0 and len(opens) == 3
    assert [(lvl, kind) for lvl, kind, _ in said] == [("warning", "hub"), ("info", "hub"), ("warning", "hub")]
    assert "JSONDecodeError" in said[0][2] and "Reconnected" in said[1][2] and "closed the connection" in said[2][2]


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
