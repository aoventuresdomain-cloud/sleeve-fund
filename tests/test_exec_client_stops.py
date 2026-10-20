"""EXEC-CLIENT drill prep (QD): resting reduce-only stops and cancels through the framework's own exchange client, on
recorded responses (tests/exec_stub.py). Built OFF: nothing here runs in paper. The first cells pin what the client
actually sends (findings: /mnt/project-files/sleeve-fund/qd/exec-client-stop-findings.md): an amend never carries a new
trigger price, and a stop is created with no trigger price type. The rest pin our own guard: a stop's level only ever
moves by cancel-and-replace, and the venue's resting stop is checked against the Strategy's after every change."""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest
from exec_stub import VenueStub

from sleeve_fund.paper.exec_client import stop_change, stop_drift

LEVEL, QTY = "60000.0", "0.010"


def _row(**kw):
    row = VenueStub.lookup_row({"orderId": "1", "orderLinkId": "s1", "qty": QTY, "side": "Sell", "orderStatus":
                                "Untriggered", "triggerPrice": LEVEL, "reduceOnly": True, "triggerDirection": 2})
    return {**row, **kw}


# our guard ---------------------------------------------------------------------------------------------------------
def test_a_resting_stop_that_matches_the_strategy_has_no_drift():
    assert stop_drift(_row(), side="SELL", qty=QTY, level=LEVEL) == []


@pytest.mark.parametrize("field, value, why", [
    ("triggerPrice", "59000.0", "level"), ("triggerBy", "MarkPrice", "trigger type"),
    ("reduceOnly", False, "reduce-only"), ("qty", "0.005", "quantity"), ("side", "Buy", "side"),
    ("orderStatus", "Cancelled", "not resting"), ("orderStatus", "Rejected", "not resting")])
def test_b_every_way_the_venue_stop_differs_is_named(field, value, why):
    drift = stop_drift(_row(**{field: value}), side="SELL", qty=QTY, level=LEVEL)
    assert len(drift) == 1 and why in drift[0]


def test_c_no_stop_at_the_venue_at_all_is_drift():
    assert stop_drift(None, side="SELL", qty=QTY, level=LEVEL) == ["no resting stop at the venue"]


def test_d_figures_compare_as_numbers_not_text():
    assert stop_drift(_row(triggerPrice="60000.00", qty="0.0100"), side="SELL", qty="0.01", level="60000") == []


@pytest.mark.parametrize("old, new, how", [
    ((LEVEL, QTY), (LEVEL, QTY), "none"),
    ((LEVEL, QTY), (LEVEL, "0.020"), "amend"),  # a quantity change alone: the client's amend carries it
    ((LEVEL, QTY), ("61000.0", QTY), "replace"),  # a level change: the client's amend would drop it
    ((LEVEL, QTY), ("61000.0", "0.020"), "replace"),
    ((LEVEL, QTY), ("60000.00", "0.0100"), "none")])
def test_e_a_level_change_is_always_a_replace(old, new, how):
    assert stop_change(old, new) == how


# what the client actually sends (pins the adapter; if an upgrade changes it, these say so) ---------------------------
@pytest.fixture(scope="module")
def venue_run():
    """One node run: a reduce-only LAST_PRICE stop is placed, its quantity and level are amended, then it is cancelled,
    all through the framework's exchange client in DEMO mode, against the stand-in venue."""
    from nautilus_trader.adapters import bybit
    from nautilus_trader.common import Environment, LoggerConfig, LogLevel
    from nautilus_trader.live import LiveNode
    from nautilus_trader.model import (
        InstrumentId,
        OrderSide,
        Price,
        Quantity,
        TimeInForce,
        TraderId,
        TriggerType,
    )
    from nautilus_trader.trading import Strategy

    venue = VenueStub().start()
    iid = InstrumentId.from_str("BTCUSDT-LINEAR.BYBIT")
    done = threading.Event()
    events = []

    class Steps(Strategy):
        def on_start(self):
            self.n = 0
            self.clock.set_timer("step", timedelta(milliseconds=500), callback=self.step)

        def step(self, _):
            self.n += 1
            if self.n == 2:
                self.stop_order = self.order_factory.stop_market(
                    iid, OrderSide.SELL, Quantity.from_str(QTY), Price.from_str(LEVEL),
                    trigger_type=TriggerType.LAST_PRICE, time_in_force=TimeInForce.GTC, reduce_only=True)
                self.submit_order(self.stop_order)
            elif self.n == 4:
                self.modify_order(self.stop_order.client_order_id, Quantity.from_str("0.020"), None,
                                  Price.from_str("61000.0"))
            elif self.n == 6:
                self.cancel_order(self.stop_order.client_order_id)
            elif self.n == 8:
                done.set()

        def on_order_event(self, event):
            events.append(type(event).__name__)

    def client(cls, **urls):
        return cls(product_types=[bybit.BybitProductType.LINEAR], environment=bybit.BybitEnvironment.DEMO,
                   api_key="stub", api_secret="stub", base_url_http=venue.http_url, max_retries=0, **urls)

    node = (LiveNode.builder("EXECSTUB", TraderId.from_str("EXECSTUB-001"), Environment.LIVE)
            .with_logging(LoggerConfig(stdout_level=LogLevel.ERROR)).with_reconciliation(reconciliation=False)
            .add_data_client(None, bybit.BybitDataClientFactory(),
                             client(bybit.BybitDataClientConfig, base_url_ws_public=venue.ws_url,
                                    base_url_ws_private=venue.ws_url))
            .add_exec_client(None, bybit.BybitExecutionClientFactory(),
                             client(bybit.BybitExecutionClientConfig, base_url_ws_private=venue.ws_url,
                                    base_url_ws_trade=venue.ws_url))
            .build())
    node.add_strategy(Steps())
    handle = node.handle()
    threading.Thread(target=lambda: (done.wait(30), handle.stop()), daemon=True).start()
    try:
        node.run()
    finally:
        venue.stop()
    node.dispose()
    assert done.is_set(), "the node never reached the last step"
    return venue, events


def test_f_the_stop_reaches_the_venue_reduce_only_at_its_level_with_no_trigger_type(venue_run):
    venue, _ = venue_run
    [create] = venue.bodies("order/create")
    assert create["reduceOnly"] is True and create["triggerPrice"] == LEVEL and create["qty"] == QTY
    assert create["orderType"] == "Market" and create["side"] == "Sell"
    assert "triggerBy" not in create  # FINDING 1: LAST_PRICE is not sent; the venue's default applies


def test_g_an_amend_carries_the_quantity_and_silently_drops_the_new_level(venue_run):
    venue, events = venue_run
    [amend] = venue.bodies("order/amend")
    assert amend["qty"] == "0.020"
    assert "triggerPrice" not in amend  # FINDING 2: a level move by amend never reaches the venue
    assert "OrderModifyRejected" not in events  # and nothing says so
    [stop] = venue.orders.values()
    assert stop["triggerPrice"] == LEVEL  # the venue's stop stayed where it was


def test_h_a_cancel_goes_out_once_by_our_order_id(venue_run):
    venue, _ = venue_run
    [cancel] = venue.bodies("order/cancel")
    [create] = venue.bodies("order/create")
    assert cancel["orderLinkId"] == create["orderLinkId"]
    assert next(iter(venue.orders.values()))["orderStatus"] == "Cancelled"


@pytest.mark.xfail(strict=True, reason="Advisor STOP-EXEC MUST 1: the framework's client never sends the last-price "
                                       "trigger; resting stops through it stay refused ON until this passes")
def test_i_must_1_every_stop_create_sends_the_last_price_trigger_explicitly(venue_run):
    venue, _ = venue_run
    [create] = venue.bodies("order/create")
    assert create.get("triggerBy") == "LastPrice"


# STOP-PATH 2e: does the framework client recognise a stop we create with our own signed call? -----------------------
def _order_push(venue, link, status, cum="0", avg="0", fee="0"):
    """The venue's private order stream for one order (SOURCES["ws order"])."""
    o = venue.orders[link]
    filled = status == "Filled"
    return {"topic": "order", "id": "o", "creationTime": 1792066007495, "data": [{
        "category": "linear", "symbol": "BTCUSDT", "orderId": o["orderId"], "orderLinkId": link, "blockTradeId": "",
        "side": o["side"], "positionIdx": 0, "orderStatus": status, "cancelType": "UNKNOWN",
        "rejectReason": "EC_NoError", "timeInForce": "GTC", "isLeverage": "", "price": "0", "qty": o["qty"],
        "avgPrice": avg, "leavesQty": "0" if filled else o["qty"], "leavesValue": "0", "cumExecQty": cum,
        "cumExecValue": "0", "cumExecFee": fee, "orderType": "Market", "stopOrderType": "Stop", "orderIv": "",
        "triggerPrice": o["triggerPrice"], "takeProfit": "0", "stopLoss": "0", "triggerBy": "LastPrice",
        "tpTriggerBy": "", "slTriggerBy": "", "triggerDirection": 2, "placeType": "", "lastPriceOnCreated": "60100",
        "closeOnTrigger": False, "reduceOnly": True, "smpGroup": 0, "smpType": "None", "smpOrderId": "",
        "slLimitPrice": "0", "tpLimitPrice": "0", "tpslMode": "", "createType": "CreateByUser", "marketUnit": "",
        "createdTime": "1792066007495", "updatedTime": "1792066007495", "feeCurrency": ""}]}


def _exec_push(venue, link, exec_id):
    """The venue's private execution stream for one fill (SOURCES["ws execution"])."""
    o = venue.orders[link]
    return {"topic": "execution", "id": "e", "creationTime": 1792066007495, "data": [{
        "category": "linear", "symbol": "BTCUSDT", "isLeverage": "", "orderId": o["orderId"], "orderLinkId": link,
        "side": o["side"], "orderPrice": "0", "orderQty": o["qty"], "leavesQty": "0",
        "createType": "CreateByStopOrder", "orderType": "Market", "stopOrderType": "Stop", "execFee": "0.33",
        "execId": exec_id, "execPrice": "59990.0", "execQty": o["qty"], "execType": "Trade", "execValue": "599.9",
        "execTime": "1792066007495", "feeCurrency": "USDT", "isMaker": False, "feeRate": "0.00055", "tradeIv": "",
        "markIv": "", "markPrice": "59990", "indexPrice": "59990", "underlyingPrice": "", "blockTradeId": "",
        "closedSize": o["qty"], "seq": 1, "extraFees": ""}]}


def _outside_run(claim: bool):
    """Two reduce-only stops are created at the venue outside the framework client: ours (QD-own-1) and one another
    process on the shared account placed (QD-other-1). Both trigger and fill. What does our Strategy see?"""
    import json
    import urllib.request

    from nautilus_trader.adapters import bybit
    from nautilus_trader.common import Environment, LoggerConfig, LogLevel
    from nautilus_trader.live import LiveNode
    from nautilus_trader.model import InstrumentId, TraderId
    from nautilus_trader.trading import Strategy, StrategyConfig

    venue = VenueStub().start()
    done, seen, owners = threading.Event(), [], {}

    def create(link):
        body = {"symbol": "BTCUSDT", "side": "Sell", "orderType": "Market", "qty": QTY, "triggerDirection": 2,
                "triggerPrice": LEVEL, "triggerBy": "LastPrice", "orderLinkId": link, "reduceOnly": True,
                "category": "linear"}
        req = urllib.request.Request(venue.http_url + "/v5/order/create", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        urllib.request.urlopen(req).read()

    class Steps(Strategy):
        def on_start(self):
            self.n = 0
            self.clock.set_timer("step", timedelta(milliseconds=500), callback=self.step)

        def step(self, _):
            self.n += 1
            if self.n == 2:
                create("QD-own-1")
                create("QD-other-1")
            elif self.n == 3:
                for link in ("QD-own-1", "QD-other-1"):
                    venue.push(_order_push(venue, link, "Untriggered"))
            elif self.n == 5:
                for k, link in enumerate(("QD-own-1", "QD-other-1")):
                    venue.push(_exec_push(venue, link, f"e-{k}"))
                    venue.push(_order_push(venue, link, "Filled", QTY, "59990.0", "0.33"))
            elif self.n == 8:
                owners.update({str(o.client_order_id): str(o.strategy_id) for o in self.cache.orders()})
                done.set()

        def on_order_event(self, event):
            seen.append((type(event).__name__, str(event.client_order_id)))

    def client(cls, **urls):
        return cls(product_types=[bybit.BybitProductType.LINEAR], environment=bybit.BybitEnvironment.DEMO,
                   api_key="stub", api_secret="stub", base_url_http=venue.http_url, max_retries=0, **urls)

    iid = InstrumentId.from_str("BTCUSDT-LINEAR.BYBIT")
    node = (LiveNode.builder("EXECSTUB", TraderId.from_str("EXECSTUB-001"), Environment.LIVE)
            .with_logging(LoggerConfig(stdout_level=LogLevel.OFF)).with_reconciliation(reconciliation=False)
            .add_data_client(None, bybit.BybitDataClientFactory(),
                             client(bybit.BybitDataClientConfig, base_url_ws_public=venue.ws_url,
                                    base_url_ws_private=venue.ws_url))
            .add_exec_client(None, bybit.BybitExecutionClientFactory(),
                             client(bybit.BybitExecutionClientConfig, base_url_ws_private=venue.ws_url,
                                    base_url_ws_trade=venue.ws_url))
            .build())
    node.add_strategy(Steps(StrategyConfig(external_order_instrument_ids=[iid] if claim else None)))
    handle = node.handle()
    threading.Thread(target=lambda: (done.wait(30), handle.stop()), daemon=True).start()
    try:
        node.run()
    finally:
        venue.stop()
    node.dispose()
    assert done.is_set()
    return seen, owners


def test_j_unclaimed_a_stop_made_outside_the_client_is_a_foreign_order_our_strategy_never_hears_of():
    seen, owners = _outside_run(claim=False)
    assert seen == []
    assert owners == {"QD-own-1": "EXTERNAL", "QD-other-1": "EXTERNAL"}


def test_k_claimed_our_strategy_hears_each_once_but_also_another_process_s_stop_on_the_shared_account():
    """With the instrument claimed the Strategy gets accepted and filled once per order (STOP-PATH 2e can hold), but it
    is handed every outside order on the instrument, another Strategy's too: the journal must own the id before a fill
    is booked (the filter the ON path needs)."""
    seen, owners = _outside_run(claim=True)
    for link in ("QD-own-1", "QD-other-1"):
        assert [k for k, c in seen if c == link] == ["OrderInitialized", "OrderAccepted", "OrderFilled"]
    assert len(set(owners.values())) == 1 and "EXTERNAL" not in owners.values()  # both booked to our Strategy
