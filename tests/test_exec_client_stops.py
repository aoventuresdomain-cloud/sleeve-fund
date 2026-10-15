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
    import nautilus_trader.adapters.bybit as bybit
    from nautilus_trader.common import Environment, LoggerConfig, LogLevel
    from nautilus_trader.live import LiveNode
    from nautilus_trader.model import (InstrumentId, OrderSide, Price, Quantity, TimeInForce, TraderId,
                                       TriggerType)
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
    assert list(venue.orders.values())[0]["orderStatus"] == "Cancelled"
