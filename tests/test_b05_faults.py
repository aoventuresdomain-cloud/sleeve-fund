"""B-05 M3 (drill-prep MUST 2, spec section 5): the order client under faults, against the fault-injecting stand-in
(tests/fault_proxy.py). Never a second send of one order, never a release on silence: a create the venue refused or
rate-limited is sent once and ends rejected; a create whose answer came too late is sent once and is found at the
venue by its id before anything else is sent; a cancel whose answer came too late leaves the order working (the A2
path stays open), as does a dropped WebSocket, after which a fill is still booked once by its trade id."""

from __future__ import annotations

import pytest
from fault_proxy import REJECTS, FaultVenue, run

from sleeve_fund.paper import exec_client


def _limit(strategy, sent):
    from nautilus_trader.model import (
        ClientOrderId,
        InstrumentId,
        OrderSide,
        Price,
        Quantity,
        TimeInForce,
    )

    order = strategy.order_factory.limit(InstrumentId.from_str("BTCUSDT-LINEAR.BYBIT"), OrderSide.BUY,
                                         Quantity.from_str("0.010"), Price.from_str("60000.0"),
                                         time_in_force=TimeInForce.GTC,
                                         client_order_id=ClientOrderId(exec_client.order_id()))
    sent.append(str(order.client_order_id))
    strategy.submit_order(order)


def _kinds(events, oid):
    return [type(e).__name__ for e in events if str(e.client_order_id) == oid]


@pytest.mark.parametrize("why", list(REJECTS))
def test_a_refused_create_is_sent_once_and_ends_rejected_with_the_venues_words(why):
    venue, sent = FaultVenue(), []
    venue.inject("order/create", ("reject", why))
    events = run(venue, {3: lambda s: _limit(s, sent)})
    [oid] = sent
    assert len(venue.bodies("order/create")) == 1  # never retried by the client (max_retries=0)
    assert "OrderRejected" in _kinds(events, oid) and "OrderAccepted" not in _kinds(events, oid)
    [rejected] = [e for e in events if type(e).__name__ == "OrderRejected"]
    assert str(REJECTS[why][0]) in str(rejected.reason) or REJECTS[why][1] in str(rejected.reason)


def test_a_429_is_sent_once_and_never_resent_by_the_client():
    venue, sent = FaultVenue(), []
    venue.inject("order/create", ("http", 429))
    events = run(venue, {3: lambda s: _limit(s, sent)})
    [oid] = sent
    assert len(venue.bodies("order/create")) == 1
    assert "OrderAccepted" not in _kinds(events, oid)


def test_a_create_answered_too_late_is_sent_once_and_found_at_the_venue_by_its_id():
    """The order went in, but the client gave up waiting: the node hears neither accepted nor rejected, so nothing
    is released. Nothing is sent again, and the owner's lookup by our id finds it working: no resend."""
    from sleeve_fund.paper.exec_client import Limiter, may_resend
    from sleeve_fund.paper.venue_stops import VenueStops

    venue, sent, looked = FaultVenue(), [], {}

    def lookup(_):
        stops = VenueStops(venue.http_url, symbol="BTCUSDT", api_key="k", api_secret="s", limiter=Limiter())
        looked["row"] = stops.lookup(sent[0])

    venue.inject("order/create", ("timeout", 2.5))
    events = run(venue, {3: lambda s: _limit(s, sent), 10: lookup}, http_timeout_secs=1, until=14)
    [oid] = sent
    assert len(venue.bodies("order/create")) == 1  # one send only
    assert _kinds(events, oid) == ["OrderInitialized", "OrderSubmitted"]  # silence: neither accepted nor rejected
    assert looked["row"]["orderLinkId"] == oid and looked["row"]["orderStatus"] == "Untriggered"
    assert not may_resend(looked["row"])


@pytest.mark.parametrize("row, ok", [
    (None, True),
    ({"orderStatus": "Rejected", "cumExecQty": "0"}, True),
    ({"orderStatus": "Rejected", "cumExecQty": "0.004"}, False),
    ({"orderStatus": "New", "cumExecQty": "0"}, False),
    ({"orderStatus": "PartiallyFilled", "cumExecQty": "0.004"}, False),
    ({"orderStatus": "Filled", "cumExecQty": "0.010"}, False),
    ({"orderStatus": "Cancelled", "cumExecQty": "0"}, False),
])
def test_only_the_venues_word_that_nothing_is_working_lets_a_resend_go(row, ok):
    from sleeve_fund.paper.exec_client import may_resend

    assert may_resend(row) is ok


def test_a_cancel_answered_too_late_leaves_the_order_working_and_unreleased():
    venue, sent = FaultVenue(), []

    def cancel(s):
        from nautilus_trader.model import ClientOrderId

        venue.inject("order/cancel", ("timeout", 2.5))
        s.cancel_order(ClientOrderId(sent[0]))

    events = run(venue, {3: lambda s: _limit(s, sent), 6: cancel}, http_timeout_secs=1, until=14)
    [oid] = sent
    assert len(venue.bodies("order/cancel")) == 1
    assert "OrderCanceled" not in _kinds(events, oid)  # silence is never a cancel: the A2 path stays open


def test_a_websocket_dropped_mid_fill_reconnects_and_the_fill_is_booked_once_by_its_trade_id(tmp_path):
    """The private stream drops after the order is accepted; the client reconnects, and the venue's fill arrives
    (twice: once replayed after the reconnect). The node may hear it twice; the journal books it once by its trade id
    (Store.record_fill: "same" for the same fill again)."""
    import test_b05_orders as orders

    from sleeve_fund.store import Store

    venue, sent = FaultVenue(), []

    def fill(_):
        for message in orders._pushes(venue, sent[0], "x-drop-1", fee="0.33"):
            venue.push(message)

    events = run(venue, {3: lambda s: _limit(s, sent), 5: lambda _: venue.drop(), 9: fill, 10: fill}, until=14)
    [oid] = sent
    fills = [e for e in events if type(e).__name__ == "OrderFilled" and str(e.client_order_id) == oid]
    assert fills, "the fill never reached the node after the reconnect"
    assert len(venue.bodies("order/create")) == 1
    store = Store(f"sqlite:///{tmp_path}/j.db")
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, desired_state="stopped")
    said = [store.record_fill("s1", side="BUY", qty=f.last_qty.as_double(), price=f.last_px.as_double(),
                              fee=f.commission.as_double(), order_id=oid, trade_id=str(f.trade_id)) for f in fills]
    assert said[0] == "new" and all(x == "same" for x in said[1:])
    assert len({str(f.trade_id) for f in fills}) == 1
