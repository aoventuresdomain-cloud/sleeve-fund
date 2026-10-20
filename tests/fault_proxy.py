"""B-05 M3 (drill-prep MUST 2, spec section 5): the stand-in venue with faults injected, for the framework's own Bybit
client in a LiveNode. A fault is queued per endpoint and taken by the next request to it: a reject code, an HTTP
status (429), or a timeout (the venue answers late, after the client gave up, and the request still took effect, so
the order's state is unknown to the client). drop() closes the WebSockets mid-run. Recorded responses only: never a
real host. Each reject code names its source page (exec_stub.SOURCES, the venue's error-code table)."""

from __future__ import annotations

import threading
import time
from datetime import timedelta

from exec_stub import DOCS, VenueStub

ERROR_CODES = f"{DOCS}/error"  # the venue's error-code table. The docs host is unreachable from build boxes (15 and 20 Oct):
# each code and wording below is to be confirmed on that page before QA signs these cells off (drill-prep MUST 17).
REJECTS = {
    "insufficient margin": (110007, "ab not enough for new order"),
    "reduce-only would increase": (110017, "Reduce-only rule not satisfied"),
    "qty below min": (10001, "The number of contracts is below min order"),
    "rate limit": (10006, "Too many visits!"),
}


class FaultVenue(VenueStub):
    def __init__(self) -> None:
        super().__init__()
        self.faults: dict[str, list[tuple]] = {}
        self.late: list[threading.Thread] = []

    def inject(self, endpoint: str, *faults: tuple) -> None:
        self.faults.setdefault(endpoint, []).extend(faults)

    def route(self, method: str, path: str, body: str) -> dict:
        for endpoint, queued in self.faults.items():
            if queued and path.startswith(f"/v5/{endpoint}"):
                kind, *arg = queued.pop(0)
                if kind == "reject":
                    code, msg = REJECTS[arg[0]]
                    return {"retCode": code, "retMsg": msg, "result": {}, "retExtInfo": {}, "time": 1735809771618}
                if kind == "http":
                    return {"__status__": arg[0], "retCode": -1, "retMsg": "Too Many Requests"}
                if kind == "timeout":  # it takes effect, but the answer comes after the client's timeout
                    answer = super().route(method, path, body)
                    time.sleep(arg[0])
                    return answer
        return super().route(method, path, body)


def run(venue: FaultVenue, steps: dict, *, http_timeout_secs: int = 1, until: int = 12) -> list:
    """One LiveNode on the framework's client against `venue`: steps[n](strategy) runs on the n-th half second.
    Returns the Strategy's order events."""
    from nautilus_trader.adapters import bybit
    from nautilus_trader.common import Environment, LoggerConfig, LogLevel
    from nautilus_trader.live import LiveNode
    from nautilus_trader.model import TraderId
    from nautilus_trader.trading import Strategy

    from sleeve_fund.paper import exec_client

    venue.start()
    env = {"EXEC_MODE": "demo", "EXEC_HOST": "api-demo.bybit.com", "BYBIT_DEMO_API_KEY": "stub",
           "BYBIT_DEMO_API_SECRET": "stub"}
    done, events = threading.Event(), []

    class Steps(Strategy):
        def on_start(self):
            self.n = 0
            self.clock.set_timer("step", timedelta(milliseconds=500), callback=self.step)

        def step(self, _):
            self.n += 1
            if self.n in steps:
                steps[self.n](self)
            if self.n >= until:
                done.set()

        def on_order_event(self, event):
            events.append(event)

    data, execution = exec_client.node_configs(
        exec_client.exec_settings(env), env, http_timeout_secs=http_timeout_secs,
        urls={"base_url_http": venue.http_url, "base_url_ws_public": venue.ws_url,
              "base_url_ws_private": venue.ws_url, "base_url_ws_trade": venue.ws_url})
    node = (LiveNode.builder("B05F", TraderId.from_str("B05F-001"), Environment.LIVE)
            .with_logging(LoggerConfig(stdout_level=LogLevel.OFF)).with_reconciliation(reconciliation=False)
            .add_data_client(None, bybit.BybitDataClientFactory(), data)
            .add_exec_client(None, bybit.BybitExecutionClientFactory(), execution)
            .build())
    node.add_strategy(Steps())
    handle = node.handle()
    threading.Thread(target=lambda: (done.wait(60), handle.stop()), daemon=True).start()
    try:
        node.run()
    finally:
        venue.stop()
    node.dispose()
    assert done.is_set(), "the node never reached the last step"
    return events
