"""P2-8 step 5 (HoE and Advisor, 14 Oct 05:02 UK): a two-leg Strategy's second leg trades a clone of its venue in the
same paper node, fed by the hub client exactly what the instrument it stands for is fed. With no alias the client
hands data on exactly as before; with one, each definition, trade, quote and bar is followed at once by its clone's,
equal but for the id; a clone is stale and blind exactly when its source is."""

import pytest
from nautilus_trader.live.clients import MarketDataClient
from nautilus_trader.model import InstrumentId

from sleeve_fund.paper import hub_client
from sleeve_fund.paper.hub_client import Decoder, HubDataClient, HubDataClientConfig, HubStatus, re_id

BTC = "BTCUSDT-PERP.BINANCE"
CLONE = "BTCUSDT-PERP.BINANCE_P2"
T0 = 1_791_223_020_000_000_000


def _feed():
    """A trade, a quote and three 1-minute bars as the hub's messages become them."""
    d = Decoder()
    out = [d({"t": "trade", "id": BTC, "px": "60000.10", "qty": "0.005", "side": "SELL", "tid": "7", "ts": T0,
              "recv": T0}, T0 + 3),
           d({"t": "quote", "id": BTC, "bid": "60000.00", "ask": "60000.20", "bid_qty": "1.0", "ask_qty": "2.0",
              "ts": T0 + 1, "recv": T0 + 1}, T0 + 4)]
    for i in range(3):
        ts = T0 + (i + 1) * 60_000_000_000
        out.append(d({"t": "bar", "id": BTC, "o": "60000.00", "h": "60010.00", "l": "59990.00", "c": "60000.10",
                      "v": "1.250", "ts": ts, "recv": ts, "refilled": False}, ts + 5))
    return [x for x in out if x is not None]


def _client(monkeypatch, aliases=()):
    got = []
    monkeypatch.setattr(MarketDataClient, "_handle_data", lambda self, data: got.append(data))
    monkeypatch.setattr(MarketDataClient, "_handle_instrument", lambda self, inst: got.append(inst))
    client = object.__new__(HubDataClient)
    client._aliases = {}
    for a, b in aliases:
        client._aliases.setdefault(a, []).append(InstrumentId.from_str(b))
    return client, got


def test_with_no_alias_the_client_hands_on_exactly_what_it_did(monkeypatch):
    client, got = _client(monkeypatch)
    feed = _feed()
    for d in feed:
        client._handle_data(d)
    assert len(got) == len(feed) and all(a is b for a, b in zip(got, feed))


def test_each_item_is_followed_at_once_by_its_clones_equal_but_for_the_id(monkeypatch):
    client, got = _client(monkeypatch, ((BTC, CLONE),))
    feed = _feed()
    for d in feed:
        client._handle_data(d)
    assert len(got) == 2 * len(feed)
    for i, d in enumerate(feed):
        src, copy = got[2 * i], got[2 * i + 1]
        assert src is d and type(copy) is type(d)
        assert (copy.ts_event, copy.ts_init) == (d.ts_event, d.ts_init)  # no added delay, no reordering
        if hasattr(d, "bar_type"):
            assert str(copy.bar_type) == str(d.bar_type).replace(BTC, CLONE)
            assert (copy.open, copy.high, copy.low, copy.close, copy.volume) == (d.open, d.high, d.low, d.close,
                                                                                 d.volume)
        else:
            assert str(copy.instrument_id) == CLONE
            assert copy.to_dict() | {"instrument_id": BTC} == d.to_dict()


def test_the_instruments_definition_is_cloned_too(monkeypatch):
    from nautilus_trader.model import CryptoPerpetual, Currency, Price, Quantity, Symbol

    usdt = Currency.from_str("USDT")
    inst = CryptoPerpetual(InstrumentId.from_str(BTC), Symbol("BTCUSDT-PERP"), Currency.from_str("BTC"), usdt, usdt,
                           False, 2, 3, Price.from_str("0.10"), Quantity.from_str("0.001"), 0, 0)
    client, got = _client(monkeypatch, ((BTC, CLONE),))
    client._handle_instrument(inst)
    assert got[0] is inst and str(got[1].id) == CLONE
    assert got[1].to_dict() | {"id": BTC} == inst.to_dict()


@pytest.mark.parametrize("pair", [("ETHUSDT-PERP.BINANCE", "ETHUSDT-PERP.BINANCE_P2"),  # not received
                                  (BTC, "ETHUSDT-PERP.BINANCE_P2"),  # another symbol
                                  (BTC, BTC)])  # its own venue
def test_an_alias_is_a_clone_of_an_instrument_the_client_receives(pair):
    with pytest.raises(ValueError):
        HubDataClientConfig(venue="BINANCE", instrument_ids=(BTC,), aliases=(pair,))


def test_a_clone_is_stale_and_blind_exactly_when_its_source_is():
    """The second leg reads the hub's status through a view: the hub's heartbeat and venue, and the source's lost
    and kept minutes by the clone's id; bars sent missing minutes reach each strategy for it to take on its own."""
    status = HubStatus()
    view = status.view({CLONE: BTC})
    status.lost[BTC] = {T0}
    status.minutes[BTC] = {T0 + 60_000_000_000: [(T0 + 60_000_000_000, 1.0, 1.0, 1.0, 1.0)]}
    assert view.unfilled(CLONE) == status.unfilled(BTC) == [T0]
    assert view.minutes_of(CLONE, T0, T0 + 60_000_000_000) == status.minutes_of(BTC, T0, T0 + 60_000_000_000)
    status.heartbeat_ns, status.venue_up = T0, False
    assert view.venue_down(T0 + 1) and status.venue_down(T0 + 1)
    status.note_bar(T0, 7, True)
    assert status.degraded == view.degraded == {T0: 7} and status.missing == view.missing == {T0: 7}
    status.degraded.clear()  # the first leg took its copy
    assert view.degraded == {T0: 7}  # the second leg still has its own


def test_the_decoder_tells_a_bar_sent_missing_minutes_to_every_strategy_reading_its_status():
    status = HubStatus()
    view = status.view({CLONE: BTC})
    d = Decoder("15-MINUTE-LAST-EXTERNAL", status=status)
    for m in (1, 2, 15):  # minutes 1, 2 and 15 of the bar: 12 missing
        ts = T0 // (15 * 60_000_000_000) * 15 * 60_000_000_000 + m * 60_000_000_000
        d({"t": "bar", "id": BTC, "o": "1", "h": "1", "l": "1", "c": "1", "v": "1", "ts": ts, "recv": ts,
           "refilled": False}, ts + 5)
    assert status.missing and status.missing == view.missing and status.degraded == view.degraded


def test_hub_client_keeps_its_old_names():
    assert hub_client.HubStatus and hub_client.Decoder and re_id
