"""B-05 M3 (drill-prep MUST 2, spec section 5): the order client under faults, against the fault-injecting stand-in
(tests/fault_proxy.py). Never a second send of one order, never a release on silence: a create the venue refused is
sent once and ends rejected; one rate-limited (HTTP 429) is sent once and ends in silence, like one whose answer came
too late: either is looked up by its id, in open orders and in history, past the venue's visibility window, before
anything is sent again (exec_client.resend_check); a cancel whose answer came too late leaves the order working (the A2
path stays open), as does a dropped WebSocket, after which a fill is still booked once by its trade id."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

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


def test_a_429_is_sent_once_ends_in_silence_and_only_the_lookup_lets_it_go_again():
    """QD review item 3: the end state pinned. The client hears an HTTP 429 as no answer: neither accepted nor
    rejected, nothing released. The venue took nothing, so after the window both lookups find nothing: resend."""
    from sleeve_fund.paper.venue_stops import VenueStops

    venue, sent, verdicts = FaultVenue(), [], []
    t0 = datetime(2026, 10, 20, 20, 0, tzinfo=UTC)

    def check(_):
        stops = VenueStops(venue.http_url, symbol="BTCUSDT", api_key="k", api_secret="s", limiter=exec_client.Limiter())
        for after in (timedelta(seconds=1), exec_client.VISIBILITY_WINDOW):
            verdicts.append(exec_client.resend_check(stops, sent[0], sent_at=t0, now=t0 + after))

    venue.inject("order/create", ("http", 429))
    events = run(venue, {3: lambda s: _limit(s, sent), 8: check}, until=12)
    [oid] = sent
    assert len(venue.bodies("order/create")) == 1
    assert _kinds(events, oid) == ["OrderInitialized", "OrderSubmitted"]  # silence: no reject, nothing released
    assert verdicts == ["wait", "resend"]


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
    (exec_client.NOT_FOUND, True),
    (None, False),  # no answer: never a resend (QD review item 1: its own sentinel, not "not found")
    ({"orderStatus": "Rejected", "cumExecQty": "0"}, True),
    ({"orderStatus": "Rejected", "cumExecQty": "0.000"}, True),
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


class _Venue:
    """The two lookups as the venue may answer them: open orders (realtime) and history, each a row, None (not there)
    or ConnectionError (no usable answer)."""

    def __init__(self, realtime=None, history=None):
        self.answers = {"realtime": realtime, "history": history}

    def _read(self, which):
        a = self.answers[which]
        if a is ConnectionError:
            raise ConnectionError("no answer")
        return a

    def lookup(self, oid):
        return self._read("realtime")

    def history(self, oid):
        return self._read("history")


T0 = datetime(2026, 10, 20, 20, 0, tzinfo=UTC)
PAST = T0 + exec_client.VISIBILITY_WINDOW


def test_a_filled_order_that_shows_only_in_history_is_never_sent_again():
    """HoE MUST (a): open orders no longer list a filled order; history does."""
    filled = {"orderLinkId": "sf-x", "orderStatus": "Filled", "cumExecQty": "0.010"}
    assert exec_client.resend_check(_Venue(realtime=None, history=filled), "sf-x", sent_at=T0, now=PAST) == "keep"


def test_an_order_not_yet_visible_is_waited_for_and_never_sent_again_once_it_shows():
    """HoE MUST (b): just after the create neither lookup shows it yet; inside the window nothing is decided."""
    working = {"orderLinkId": "sf-x", "orderStatus": "New", "cumExecQty": "0"}
    early = exec_client.resend_check(_Venue(), "sf-x", sent_at=T0, now=T0 + timedelta(seconds=2))
    late = exec_client.resend_check(_Venue(realtime=working), "sf-x", sent_at=T0, now=PAST)
    assert (early, late) == ("wait", "keep")


@pytest.mark.parametrize("realtime, history", [(ConnectionError, None), (None, ConnectionError)])
def test_either_lookup_unanswered_is_no_answer_never_a_resend(realtime, history):
    assert exec_client.resend_check(_Venue(realtime, history), "sf-x", sent_at=T0, now=PAST) == "no answer"


def test_a_refused_lookup_is_no_answer_not_an_empty_list():
    """A lookup the venue refused (rate limit) used to read as an empty list, so "not found": now no answer."""
    from exec_stub import VenueStub

    from sleeve_fund.paper.venue_stops import VenueStops

    class Refuses(VenueStub):
        def route(self, method, path, body):
            if path.startswith(("/v5/order/realtime", "/v5/order/history")):
                return {"retCode": 10006, "retMsg": "Too many visits!", "result": {}}
            return super().route(method, path, body)

    venue = Refuses().start()
    try:
        stops = VenueStops(venue.http_url, symbol="BTCUSDT", api_key="k", api_secret="s", limiter=exec_client.Limiter())
        assert exec_client.resend_check(stops, "sf-x", sent_at=T0, now=PAST) == "no answer"
    finally:
        venue.stop()


def test_only_both_lookups_empty_past_the_window_or_a_clean_refusal_lets_it_go():
    refused = {"orderLinkId": "sf-x", "orderStatus": "Rejected", "cumExecQty": "0"}
    assert exec_client.resend_check(_Venue(), "sf-x", sent_at=T0, now=PAST) == "resend"
    assert exec_client.resend_check(_Venue(history=refused), "sf-x", sent_at=T0, now=PAST) == "resend"
