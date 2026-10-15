"""EXEC-CLIENT drill prep: a local stand-in for the exchange's REST and private WebSocket, so the framework's own Bybit
execution client runs inside a LiveNode against recorded responses (spec §5-§6). Every request the client sends is
kept in `seen`, verbatim, so a cell asserts exactly what would have reached the venue. Never points at a real host.

Each response shape names the venue documentation page it follows (SOURCES). The docs host could not be reached from
the box these were written on (15 Oct 2026), so each page is to be checked by PE3 or QA before QA signs the cells off;
nothing undocumented is recorded here (HEAD3 (1))."""

from __future__ import annotations

import asyncio
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DOCS = "https://bybit-exchange.github.io/docs/v5"
SOURCES = {
    "order/create": f"{DOCS}/order/create-order",
    "order/amend": f"{DOCS}/order/amend-order",
    "order/cancel": f"{DOCS}/order/cancel-order",
    "order/realtime": f"{DOCS}/order/open-order",
    "order/history": f"{DOCS}/order/order-list",
    "instruments-info": f"{DOCS}/market/instrument",
    "fee-rate": f"{DOCS}/account/fee-rate",
    "wallet-balance": f"{DOCS}/account/wallet-balance",
    "ws auth": f"{DOCS}/ws/connect",
    "ws order": f"{DOCS}/websocket/private/order",
    "ws execution": f"{DOCS}/websocket/private/execution",
}
SYMBOL = "BTCUSDT"
INSTRUMENT = {"symbol": SYMBOL, "contractType": "LinearPerpetual", "status": "Trading", "baseCoin": "BTC",
              "quoteCoin": "USDT", "launchTime": "1585526400000", "deliveryTime": "0", "deliveryFeeRate": "",
              "priceScale": "2", "leverageFilter": {"minLeverage": "1", "maxLeverage": "100.00", "leverageStep": "0.01"},
              "priceFilter": {"minPrice": "0.10", "maxPrice": "1999999.80", "tickSize": "0.10"},
              "lotSizeFilter": {"maxOrderQty": "1190.000", "minOrderQty": "0.001", "qtyStep": "0.001",
                                "postOnlyMaxOrderQty": "1190.000", "maxMktOrderQty": "500.000",
                                "minNotionalValue": "5"},
              "unifiedMarginTrade": True, "fundingInterval": 480, "settleCoin": "USDT", "copyTrading": "both",
              "upperFundingRate": "0.00375", "lowerFundingRate": "-0.00375", "isPreListing": False,
              "preListingInfo": None, "riskParameters": {"priceLimitRatioX": "0.01", "priceLimitRatioY": "0.02"}}
WALLET = {"totalEquity": "100000", "accountIMRate": "0", "totalMarginBalance": "100000", "totalInitialMargin": "0",
          "accountType": "UNIFIED", "totalAvailableBalance": "100000", "accountMMRate": "0", "totalPerpUPL": "0",
          "totalWalletBalance": "100000", "accountLTV": "0", "totalMaintenanceMargin": "0",
          "coin": [{"availableToBorrow": "", "bonus": "0", "accruedInterest": "0", "availableToWithdraw": "100000",
                    "totalOrderIM": "0", "equity": "100000", "totalPositionMM": "0", "usdValue": "100000",
                    "spotHedgingQty": "0", "unrealisedPnl": "0", "collateralSwitch": True, "borrowAmount": "0",
                    "totalPositionIM": "0", "walletBalance": "100000", "cumRealisedPnl": "0", "locked": "0",
                    "marginCollateral": True, "coin": "USDT"}]}


def _ok(result: dict) -> dict:
    return {"retCode": 0, "retMsg": "OK", "result": result, "retExtInfo": {}, "time": 1735809771618}


class VenueStub:
    """REST and private WebSocket on 127.0.0.1. Orders are kept by orderLinkId; create, amend and cancel change them as
    the venue would, and the order lookups return them. `seen` holds (method, path, body) of every REST request."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, str]] = []
        self.ws_seen: list[str] = []
        self.headers: list[dict] = []  # each REST request's headers, for the signing cells
        self.orders: dict[str, dict] = {}  # orderLinkId -> the venue's view of it
        self._ids = itertools.count(1321003749386327552)
        self._lock = threading.Lock()
        self._http: ThreadingHTTPServer | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_port: int | None = None
        self._ws_stop: asyncio.Event | None = None
        self._clients: set = set()

    # what a cell reads ---------------------------------------------------------------------------------------------
    def bodies(self, endpoint: str) -> list[dict]:
        return [json.loads(b) for m, p, b in self.seen if m == "POST" and p.startswith(f"/v5/{endpoint}")]

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self._http.server_address[1]}"

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self._ws_port}"

    # the venue ----------------------------------------------------------------------------------------------------
    def route(self, method: str, path: str, body: str) -> dict:
        if "instruments-info" in path:
            return _ok({"category": "linear", "list": [INSTRUMENT], "nextPageCursor": ""})
        if "fee-rate" in path:
            return _ok({"category": "linear",
                        "list": [{"symbol": SYMBOL, "takerFeeRate": "0.00055", "makerFeeRate": "0.0002"}]})
        if "wallet-balance" in path:
            return _ok({"list": [WALLET]})
        with self._lock:
            if path.startswith("/v5/order/create"):
                b = json.loads(body)
                if len(b.get("orderLinkId", "")) > 36:  # SOURCES["order/create"]: orderLinkId max 36 characters
                    return {"retCode": 10001, "retMsg": "params error: orderLinkId too long", "result": {},
                            "retExtInfo": {}, "time": 1735809771618}
                oid = str(next(self._ids))
                self.orders[b["orderLinkId"]] = dict(b, orderId=oid, orderStatus="Untriggered")
                return _ok({"orderId": oid, "orderLinkId": b["orderLinkId"]})
            if path.startswith(("/v5/order/amend", "/v5/order/cancel")):
                b = json.loads(body)
                o = self.orders.get(b.get("orderLinkId", ""))
                if o is None:
                    return {"retCode": 110001, "retMsg": "order not exists or too late to cancel", "result": {},
                            "retExtInfo": {}, "time": 1735809771618}
                if path.startswith("/v5/order/amend"):
                    o.update({k: v for k, v in b.items() if k in ("qty", "triggerPrice", "triggerBy")})
                else:
                    o["orderStatus"] = "Cancelled"
                return _ok({"orderId": o["orderId"], "orderLinkId": o["orderLinkId"]})
            if path.startswith(("/v5/order/realtime", "/v5/order/history")):
                from urllib.parse import parse_qs, urlparse

                link = parse_qs(urlparse(path).query).get("orderLinkId", [None])[0]
                rows = [self.lookup_row(o) for o in self.orders.values() if link in (None, o["orderLinkId"])]
                return _ok({"category": "linear", "list": rows, "nextPageCursor": ""})
        return _ok({"category": "linear", "list": [], "nextPageCursor": ""})

    @staticmethod
    def lookup_row(o: dict) -> dict:
        """One order as the venue's order lookup reports it (SOURCES["order/realtime"]). A conditional order created
        without triggerBy reports the venue's default, assumed LastPrice here: the drill confirms it on the demo."""
        return {"orderId": o["orderId"], "orderLinkId": o["orderLinkId"], "blockTradeId": "", "symbol": SYMBOL,
                "price": "0", "qty": o["qty"], "side": o["side"], "isLeverage": "", "positionIdx": 0,
                "orderStatus": o["orderStatus"], "cancelType": "UNKNOWN", "rejectReason": "EC_NoError",
                "avgPrice": "0", "leavesQty": o["qty"], "leavesValue": "0", "cumExecQty": "0", "cumExecValue": "0",
                "cumExecFee": "0", "timeInForce": "GTC", "orderType": o.get("orderType", "Market"),
                "stopOrderType": "Stop" if o.get("triggerPrice") else "", "orderIv": "",
                "triggerPrice": o.get("triggerPrice", "0"), "takeProfit": "0", "stopLoss": "0", "tpTriggerBy": "",
                "slTriggerBy": "", "triggerDirection": o.get("triggerDirection", 0),
                "triggerBy": o.get("triggerBy", "LastPrice"), "lastPriceOnCreated": "",
                "reduceOnly": bool(o.get("reduceOnly")), "closeOnTrigger": False, "smpType": "None", "smpGroup": 0,
                "smpOrderId": "", "tpslMode": "", "tpLimitPrice": "", "slLimitPrice": "", "placeType": "",
                "createdTime": "1792066007495", "updatedTime": "1792066007495"}

    # servers ------------------------------------------------------------------------------------------------------
    def start(self) -> VenueStub:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self):
                n = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(n).decode() if n else ""
                stub.seen.append((self.command, self.path, body))
                stub.headers.append(dict(self.headers))
                out = json.dumps(stub.route(self.command, self.path, body)).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            do_GET = do_POST = _reply

            def log_message(self, *args):
                pass

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        ready = threading.Event()
        threading.Thread(target=self._serve_ws, args=(ready,), daemon=True).start()
        if not ready.wait(10):
            raise RuntimeError("the WebSocket stand-in did not start")
        return self

    def _serve_ws(self, ready: threading.Event) -> None:
        from websockets.asyncio.server import serve

        async def handler(ws):
            self._clients.add(ws)
            async for raw in ws:
                self.ws_seen.append(raw)
                try:
                    op = json.loads(raw).get("op")
                except (ValueError, AttributeError):
                    continue
                if op in ("auth", "subscribe", "ping"):  # SOURCES["ws auth"]
                    await ws.send(json.dumps({"success": True, "ret_msg": "pong" if op == "ping" else "", "op": op,
                                              "conn_id": "stub"}))

        async def main():
            self._ws_loop, self._ws_stop = asyncio.get_running_loop(), asyncio.Event()
            async with serve(handler, "127.0.0.1", 0) as server:
                self._ws_port = server.sockets[0].getsockname()[1]
                ready.set()
                await self._ws_stop.wait()

        asyncio.run(main())

    def push(self, message: dict) -> None:
        """Send `message` to every connected private WebSocket, as the venue's order or execution stream would."""
        raw = json.dumps(message)

        async def send():
            for ws in list(self._clients):
                try:
                    await ws.send(raw)
                except Exception:  # noqa: BLE001 - a closed client is dropped
                    self._clients.discard(ws)

        asyncio.run_coroutine_threadsafe(send(), self._ws_loop).result(5)

    def stop(self) -> None:
        if self._http is not None:
            self._http.shutdown()
        if self._ws_loop is not None:
            self._ws_loop.call_soon_threadsafe(self._ws_stop.set)
