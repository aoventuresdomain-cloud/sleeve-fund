"""B-05 M2: orders through the exchange client (drill-prep MUST 1, 4, 6; EXEC-SPEC MUST 3 and 5). Built OFF: nothing
here runs in paper until the drills. ON, every order id the fund sends is in its namespace and fits the venue's 36
characters, a late fill's close id is deterministic, and a fill books the venue's own fee, alerted when more than a
cent off FEE-LIVE's. The node run goes through the framework's Bybit client against the stand-in venue
(tests/exec_stub.py): nothing reaches the exchange."""

from __future__ import annotations

import re
import threading
from datetime import timedelta
from types import SimpleNamespace

import pytest
from exec_stub import VenueStub

from sleeve_fund.paper import exec_client
from sleeve_fund.paper.stop_mover import ours
from sleeve_fund.paper.venue_stops import classify

DEMO_ENV = {"EXEC_MODE": "demo", "EXEC_HOST": "api-demo.bybit.com", "BYBIT_DEMO_API_KEY": "stub",
            "BYBIT_DEMO_API_SECRET": "stub"}


# ids ----------------------------------------------------------------------------------------------------------------
def test_an_order_id_on_fits_the_venue_and_is_the_funds():
    ids = {exec_client.order_id() for _ in range(200)}
    assert len(ids) == 200
    for oid in ids:
        assert exec_client.link_id_ok(oid) and len(oid) <= 36 and ours(oid)
        assert not oid.startswith(exec_client.CLOSE_ID_PREFIX)  # never taken for a late fill's close
        assert classify(oid, own={oid}, fund={oid}) == "book" and classify(oid, own=set(), fund=set()) == "orphan"


def test_todays_paper_ids_do_not_fit_which_is_why_on_has_its_own():
    """The MUST 5 finding: the simulated client's ids and the A2 close ids pass 36 characters."""
    paper = "O-20261015-121800-PAPER-CANDLE-STEP-BTCUSDT-CANDLESTEP-2C1F0A9B-12"
    assert not exec_client.link_id_ok(paper) and not exec_client.link_id_ok(f"orphan-close-{paper}")


def test_a_close_id_on_is_deterministic_short_and_one_per_close():
    entry = "sf-" + "a" * 32
    first, again, second = exec_client.close_id(entry), exec_client.close_id(entry), exec_client.close_id(entry, 1)
    assert first == again != second  # two writers racing journal one row; a second close of it is its own
    for cid in (first, second):
        assert len(cid) == 36 and exec_client.link_id_ok(cid) and ours(cid)
        assert re.fullmatch(r"sf-z[0-9a-f]{32}", cid)
    assert exec_client.close_id("sf-" + "b" * 32) != first


def test_the_strategys_ids_are_its_factorys_off_and_the_funds_on():
    from sleeve_fund.strategies.base import LongFlatStrategy

    off = SimpleNamespace(_exec_on=lambda: False)
    on = SimpleNamespace(_exec_on=lambda: True)
    assert LongFlatStrategy._ids(off) == {}  # OFF: the factory's own ids, so paper stays byte-identical
    [(key, cid)] = LongFlatStrategy._ids(on).items()
    assert key == "client_order_id" and ours(str(cid)) and exec_client.link_id_ok(str(cid))


def test_no_order_in_the_model_is_made_without_the_id_switch():
    """Pins the call sites: every order the base model makes takes its id from _ids or _new_coid."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "sleeve_fund/strategies/base.py").read_text()
    made = re.findall(r"self\.order_factory\.(market|limit|stop_market)\(", src)
    assert len(made) == src.count("**self._ids())") + src.count("client_order_id=ClientOrderId(close_id)")
    assert src.count("self.order_factory.generate_client_order_id()") == 1  # inside _new_coid, its OFF branch


# the A2 close under either scheme -----------------------------------------------------------------------------------
@pytest.mark.parametrize("on", [False, True])
def test_a_late_fill_is_closed_once_under_the_id_scheme_in_force(tmp_path, on):
    from sleeve_fund.paper import orphan_guard
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/j.db")
    _late_fill(store)
    [cid] = orphan_guard.close_orphans(store, _NOW, on=on)
    assert cid.startswith(exec_client.CLOSE_ID_PREFIX if on else orphan_guard.CLOSE_PREFIX)
    assert exec_client.link_id_ok(cid) is on or not on
    assert orphan_guard.close_orphans(store, _NOW, on=on) == []  # journaled once: the next pass finds it sent
    [x] = orphan_guard.late_fills(store, "s1")
    assert [o["order_id"] for o in x["closes"]] == [cid]
    assert orphan_guard.unsent(store, "s1", set()) and not orphan_guard.unsent(store, "s1", {cid})


# the venue's fee ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fee, gap", [(33.0, False), (33.009, False), (33.02, True), (0.0, True)])
def test_a_venue_fee_more_than_a_cent_off_fee_live_is_named(fee, gap):
    why = exec_client.fee_gap(qty=1.0, price=60_000.0, fee=fee, rate=0.00055)
    assert (why is not None) is gap
    if gap:
        assert "the venue's is booked" in why


def test_no_rate_to_compare_is_no_alert():
    assert exec_client.fee_gap(qty=1.0, price=60_000.0, fee=12.0, rate=None) is None


# the client's configs -----------------------------------------------------------------------------------------------
def test_demo_and_live_build_through_one_function_and_never_retry_an_order_on_their_own():
    from nautilus_trader.adapters import bybit

    data, execution = exec_client.node_configs(exec_client.exec_settings(DEMO_ENV), DEMO_ENV)
    assert execution.environment == bybit.BybitEnvironment.DEMO and execution.max_retries == 0
    assert execution.base_url_http == "https://api-demo.bybit.com" and data.base_url_http == execution.base_url_http
    assert execution.margin_mode == bybit.BybitMarginMode.ISOLATED_MARGIN
    assert execution.product_types == [bybit.BybitProductType.LINEAR]
    live = {"EXEC_MODE": "live", "EXEC_HOST": "api.bybit.com", "BYBIT_TRADE_API_KEY": "k",
            "BYBIT_TRADE_API_SECRET": "s"}
    _, live_exec = exec_client.node_configs(exec_client.exec_settings(live), live)
    assert live_exec.environment == bybit.BybitEnvironment.MAINNET and live_exec.max_retries == 0


def test_a_host_or_key_mismatch_builds_no_client():
    bad = {**DEMO_ENV, "EXEC_HOST": "api.bybit.com"}
    with pytest.raises(Exception, match="refused"):
        exec_client.node_configs(exec_client.exec_settings(bad), bad)
    with pytest.raises(ValueError, match="OFF"):
        exec_client.node_configs(exec_client.exec_settings({}), {})


def test_the_paper_node_still_refuses_on():
    """Drill-prep: ON stays refused at the node until every MUST line is green and the PM's go."""
    from sleeve_fund.paper.safety import PaperSafetyError

    with pytest.raises(PaperSafetyError):
        exec_client.assert_off(DEMO_ENV)


# one order through the framework's client, against the stand-in venue -----------------------------------------------
_NOW = None


def _late_fill(store):
    """A resting entry that filled after its bar's close: one late fill to close (A2)."""
    from datetime import UTC, datetime

    from sleeve_fund.paper import orphan_guard

    global _NOW
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, desired_state="stopped")
    t0 = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
    orphan_guard.qa_resting_entry(store, "s1", "e1", t0)
    store.book_fill("s1", side="BUY", qty=0.01, price=60_000.0, fee=0.33, order_id="e1", trade_id="t1",
                      ts=t0 + timedelta(minutes=2))
    _NOW = t0 + timedelta(minutes=3)


def _pushes(venue, link, exec_id, *, fee):
    o = venue.orders[link]
    common = {"category": "linear", "symbol": "BTCUSDT", "orderId": o["orderId"], "orderLinkId": link,
              "side": o["side"], "isLeverage": ""}
    order = {"topic": "order", "id": "o", "creationTime": 1792066007495, "data": [{
        **common, "blockTradeId": "", "positionIdx": 0, "orderStatus": "Filled", "cancelType": "UNKNOWN",
        "rejectReason": "EC_NoError", "timeInForce": "GTC", "price": "60000.0", "qty": o["qty"], "avgPrice": "60000",
        "leavesQty": "0", "leavesValue": "0", "cumExecQty": o["qty"], "cumExecValue": "600", "cumExecFee": fee,
        "orderType": "Limit", "stopOrderType": "", "orderIv": "", "triggerPrice": "0", "takeProfit": "0",
        "stopLoss": "0", "triggerBy": "", "tpTriggerBy": "", "slTriggerBy": "", "triggerDirection": 0,
        "placeType": "", "lastPriceOnCreated": "60000", "closeOnTrigger": False, "reduceOnly": False, "smpGroup": 0,
        "smpType": "None", "smpOrderId": "", "slLimitPrice": "0", "tpLimitPrice": "0", "tpslMode": "",
        "createType": "CreateByUser", "marketUnit": "", "createdTime": "1792066007495",
        "updatedTime": "1792066007495", "feeCurrency": ""}]}
    execution = {"topic": "execution", "id": "e", "creationTime": 1792066007495, "data": [{
        **common, "orderPrice": "0", "orderQty": o["qty"], "leavesQty": "0", "createType": "CreateByUser",
        "orderType": "Limit", "stopOrderType": "", "execFee": fee, "execId": exec_id, "execPrice": "60000.0",
        "execQty": o["qty"], "execType": "Trade", "execValue": "600", "execTime": "1792066007495",
        "feeCurrency": "USDT", "isMaker": False, "feeRate": "0.00055", "tradeIv": "", "markIv": "",
        "markPrice": "60000", "indexPrice": "60000", "underlyingPrice": "", "blockTradeId": "", "closedSize": "0",
        "seq": 1, "extraFees": ""}]}
    return order, execution


@pytest.fixture(scope="module")
def one_order():
    from nautilus_trader.common import Environment, LoggerConfig, LogLevel
    from nautilus_trader.live import LiveNode
    from nautilus_trader.model import (
        ClientOrderId,
        InstrumentId,
        OrderSide,
        Price,
        Quantity,
        TimeInForce,
        TraderId,
    )
    from nautilus_trader.trading import Strategy

    venue = VenueStub().start()
    iid = InstrumentId.from_str("BTCUSDT-LINEAR.BYBIT")
    done, events, sent = threading.Event(), [], {}

    class One(Strategy):
        def on_start(self):
            self.n = 0
            self.clock.set_timer("step", timedelta(milliseconds=500), callback=self.step)

        def step(self, _):
            self.n += 1
            if self.n == 3:
                # A limit order: the risk engine refuses a market order with no price in the cache, and the
                # stand-in has no public feed (the Strategy's own market data on the venue is MUST 9, the hub).
                order = self.order_factory.limit(iid, OrderSide.BUY, Quantity.from_str("0.010"),
                                                 Price.from_str("60000.0"), time_in_force=TimeInForce.GTC,
                                                 client_order_id=ClientOrderId(exec_client.order_id()))
                sent["id"] = str(order.client_order_id)
                self.submit_order(order)
            elif self.n == 5 and sent.get("id") in venue.orders:
                for message in _pushes(venue, sent["id"], "x-1", fee="0.33"):
                    venue.push(message)
            elif self.n == 9:
                done.set()

        def on_order_event(self, event):
            events.append(event)

    data, execution = exec_client.node_configs(
        exec_client.exec_settings(DEMO_ENV), DEMO_ENV,
        urls={"base_url_http": venue.http_url, "base_url_ws_public": venue.ws_url,
              "base_url_ws_private": venue.ws_url, "base_url_ws_trade": venue.ws_url})
    from nautilus_trader.adapters import bybit

    node = (LiveNode.builder("B05", TraderId.from_str("B05-001"), Environment.LIVE)
            .with_logging(LoggerConfig(stdout_level=LogLevel.ERROR)).with_reconciliation(reconciliation=False)
            .add_data_client(None, bybit.BybitDataClientFactory(), data)
            .add_exec_client(None, bybit.BybitExecutionClientFactory(), execution)
            .build())
    node.add_strategy(One())
    handle = node.handle()
    threading.Thread(target=lambda: (done.wait(30), handle.stop()), daemon=True).start()
    try:
        node.run()
    finally:
        venue.stop()
    node.dispose()
    assert done.is_set(), "the node never reached the last step"
    return venue, events, sent["id"]


def test_the_order_reaches_the_venue_once_under_the_funds_id(one_order):
    venue, _, oid = one_order
    [create] = venue.bodies("order/create")
    assert create["orderLinkId"] == oid and exec_client.link_id_ok(oid) and ours(oid)
    assert create["side"] == "Buy" and create["orderType"] == "Limit" and create["qty"] == "0.010"


def test_the_fill_carries_the_venues_own_fee(one_order):
    _, events, oid = one_order
    fills = [e for e in events if type(e).__name__ == "OrderFilled"]
    assert len(fills) == 1 and str(fills[0].client_order_id) == oid
    assert fills[0].commission.as_double() == pytest.approx(0.33)  # what the strategy books (on_order_filled)


def test_finding_the_client_sets_the_accounts_margin_mode_when_it_connects(one_order):
    """FINDING for MUST 3 (Advisor to rule): with margin_mode set, the framework's client posts the account's margin
    mode at connect, before any order. MUST 3 says never auto-fix; on a shared account this is an account-wide write.
    Pinned so a change in the client is seen; the connect check (M4) reads the mode instead of setting it."""
    venue, _, _ = one_order
    [body] = venue.bodies("account/set-margin-mode")
    assert body == {"setMarginMode": "ISOLATED_MARGIN"}
    paths = [p.split("?")[0] for _, p, _ in venue.seen]
    assert paths.index("/v5/account/set-margin-mode") < paths.index("/v5/order/create")
