"""A paper node's market data from its venue's hub (v2 P1-1) instead of a venue connection of its own: a
Nautilus data client that reads the hub's stream (sleeve_fund.hub.protocol) and hands the node the same trades,
quotes and closed 1-minute bars every other strategy on the instrument gets, and the instrument's definition.

Bars arrive as `<instrument>-1-MINUTE-LAST-EXTERNAL`; a strategy on longer bars subscribes to a composite type
(`15-MINUTE-LAST-INTERNAL@1-MINUTE-EXTERNAL`) and the node builds them from these, as a backtest builds them
from the store's 1-minute bars. A bar at or before the last one delivered (a refill landing after the live bars
moved on) is not delivered again. When the hub goes away the client reconnects; until it does, no trades reach
the strategy and its stale-feed guard holds new entries."""

from __future__ import annotations

import asyncio
import json
import time

from nautilus_trader.live import DataClientConfig
from nautilus_trader.live.clients import DataClientFactory, MarketDataClient
from nautilus_trader.model import (
    AggressorSide,
    Bar,
    BarType,
    InstrumentId,
    Price,
    Quantity,
    QuoteTick,
    TradeId,
    TradeTick,
    Venue,
)

from sleeve_fund.hub import protocol

RECONNECT_SECONDS = (1, 2, 5, 10, 30)
CONNECT_TIMEOUT = 90  # the hub picks up an instrument it doesn't relay yet within a minute
LATE_BAR_SECONDS = 90  # a bar refilled this long after its close is history, not a signal


def bar_type(instrument_id: str) -> BarType:
    return BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")


def hub_bar_spec(spec: str) -> str:
    """A strategy's bar spec as fed by the hub: 1-minute bars as they come, anything longer built in the node
    from them. 15-MINUTE-LAST-INTERNAL -> 15-MINUTE-LAST-INTERNAL@1-MINUTE-EXTERNAL."""
    if spec == "1-MINUTE-LAST-EXTERNAL" or spec.endswith("@1-MINUTE-EXTERNAL"):
        return spec
    if spec == "1-MINUTE-LAST-INTERNAL":
        return "1-MINUTE-LAST-EXTERNAL"
    if not spec.endswith("-LAST-INTERNAL"):
        raise ValueError(f"bar spec {spec!r} can't be built from the hub's 1-minute bars")
    return f"{spec}@1-MINUTE-EXTERNAL"


def instrument_from(defn: dict):
    from nautilus_trader import model

    return getattr(model, defn["type"]).from_dict(defn)


class Decoder:
    """Hub messages -> Nautilus data, delivering each instrument's bars in order only once, and only while
    current: a bar that closed more than LATE_BAR_SECONDS before it arrived (the hub refilling minutes it missed)
    is stored by the hub for warm-ups but not traded on here, as a strategy building its own bars never saw
    the minutes its feed missed either."""

    def __init__(self) -> None:
        self.last_bar: dict[str, int] = {}
        self.types: dict[str, BarType] = {}
        self.late = 0

    def __call__(self, m: dict, now_ns: int):
        t = m.get("t")
        if t == "trade":
            return TradeTick(InstrumentId.from_str(m["id"]), Price.from_str(m["px"]), Quantity.from_str(m["qty"]),
                             AggressorSide.from_str(m["side"]), TradeId(m["tid"]), m["ts"], now_ns)
        if t == "quote":
            return QuoteTick(InstrumentId.from_str(m["id"]), Price.from_str(m["bid"]), Price.from_str(m["ask"]),
                             Quantity.from_str(m["bid_qty"]), Quantity.from_str(m["ask_qty"]), m["ts"], now_ns)
        if t == "bar":
            if m["ts"] <= self.last_bar.get(m["id"], 0):
                return None
            if now_ns - m["ts"] > LATE_BAR_SECONDS * 1_000_000_000:
                self.late += 1
                return None
            self.last_bar[m["id"]] = m["ts"]
            bt = self.types.get(m["id"]) or self.types.setdefault(m["id"], bar_type(m["id"]))
            return Bar(bt, Price.from_str(m["o"]), Price.from_str(m["h"]), Price.from_str(m["l"]),
                       Price.from_str(m["c"]), Quantity.from_str(m["v"]), m["ts"], now_ns)
        return None  # heartbeats, gaps and anything newer: read for their effect, not passed on


class HubDataClientConfig(DataClientConfig):
    def __init__(self, *, venue: str, instrument_ids: tuple[str, ...], host: str = "hub", port: int = 7700,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.venue, self.instrument_ids, self.host, self.port = venue, tuple(instrument_ids), host, port


class HubDataClient(MarketDataClient):
    def __init__(self, *, name: str, config: HubDataClientConfig, cache, clock) -> None:
        super().__init__(name=name, config=config, cache=cache, clock=clock, venue=Venue(config.venue))
        self.cfg = config
        self.decode = Decoder()
        self.last_heartbeat_ns = 0
        self.venue_up = False
        self.connects = 0
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task = None
        self._closing = False

    async def _open(self) -> tuple[asyncio.StreamReader, dict]:
        reader, writer = await asyncio.open_connection(self.cfg.host, self.cfg.port, limit=16 * 1024 * 1024)
        writer.write(protocol.encode(protocol.subscription(list(self.cfg.instrument_ids))))
        await writer.drain()
        hello = protocol.decode(await reader.readline())
        if hello.get("t") != "hello":
            writer.close()
            raise ConnectionError(f"hub refused: {hello.get('message', hello)}")
        self._writer = writer
        self.connects += 1
        return reader, hello

    async def _connect(self) -> None:
        """Waits until the hub serves every instrument asked for, with its definition, so the strategy finds it
        in the cache when it starts."""
        deadline = time.monotonic() + CONNECT_TIMEOUT
        while True:
            reader, hello = await self._open()
            defs = {d["id"]: d for d in hello.get("instruments", [])}
            if set(self.cfg.instrument_ids) <= set(defs):
                break
            self._writer.close()
            if time.monotonic() > deadline:
                missing = sorted(set(self.cfg.instrument_ids) - set(defs))
                raise ConnectionError(f"the hub doesn't serve {', '.join(missing)}")
            await asyncio.sleep(5)
        for d in defs.values():
            self._handle_instrument(instrument_from(d))
        self._reader_task = self.create_task(self._read(reader), name="hub-read")

    async def _read(self, reader: asyncio.StreamReader) -> None:
        attempt = 0
        while not self._closing:
            try:
                while line := await reader.readline():
                    m = json.loads(line)
                    now = self.clock.timestamp_ns()
                    if m.get("t") == "hb":
                        self.last_heartbeat_ns, self.venue_up = now, bool(m.get("venue_up"))
                        continue
                    late = self.decode.late
                    data = self.decode(m, now)
                    if data is not None:
                        self._handle_data(data)
                    elif self.decode.late != late:
                        print(f"hub: skipped {m['id']} bar closing {m['ts']}, {(now - m['ts']) / 1e9:.0f}s late")
                    attempt = 0
            except (ConnectionError, OSError, ValueError):
                pass
            if self._closing:
                return
            await asyncio.sleep(RECONNECT_SECONDS[min(attempt, len(RECONNECT_SECONDS) - 1)])
            attempt += 1
            try:
                reader, _ = await self._open()
            except (ConnectionError, OSError, ValueError):
                reader = asyncio.StreamReader()
                reader.feed_eof()

    async def _disconnect(self) -> None:
        self._closing = True
        if self._writer is not None:
            self._writer.close()

    # The stream carries everything for the instruments asked for at connection; subscriptions change nothing.
    async def _subscribe_trades(self, command) -> None:
        pass

    async def _subscribe_quotes(self, command) -> None:
        pass

    async def _subscribe_bars(self, command) -> None:
        pass

    async def _subscribe_instrument(self, command) -> None:
        pass

    async def _unsubscribe_trades(self, command) -> None:
        pass

    async def _unsubscribe_quotes(self, command) -> None:
        pass

    async def _unsubscribe_bars(self, command) -> None:
        pass


class HubDataClientFactory(DataClientFactory):
    @staticmethod
    def create(*, name: str, config: HubDataClientConfig, cache, clock) -> HubDataClient:
        return HubDataClient(name=name, config=config, cache=cache, clock=clock)
