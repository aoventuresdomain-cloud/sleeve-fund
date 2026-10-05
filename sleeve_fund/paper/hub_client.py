"""A paper node's market data from its venue's hub (v2 P1-1) instead of a venue connection of its own: a
Nautilus data client that reads the hub's stream (sleeve_fund.hub.protocol) and hands the node the same trades,
quotes and closed 1-minute bars every other strategy on the instrument gets, and the instrument's definition.

Bars reach the strategy as `<instrument>-<its bar>-LAST-EXTERNAL`. A strategy on longer bars gets them built here
from the hub's minutes, as a backtest builds them from the store's: a 15-minute bar is the minutes closing in it,
sent as the one closing at its end arrives and stamped at that close. Building them here rather than on the
node's clock means a bar never closes before its last minute has arrived. A bar at or before the last one
delivered (a refill landing after the live bars moved on) is not delivered again. When the hub goes away the
client reconnects; until it does, nothing arrives, so no bar closes and the strategy decides nothing."""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal

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

RECONNECT_SECONDS = (1, 2, 5)  # the hub is on the same host: back within seconds of it
CONNECT_TIMEOUT = 90  # the hub picks up an instrument it doesn't relay yet within a minute
LATE_BAR_SECONDS = 90  # a bar refilled this long after its close is history, not a signal


MINUTE_NS = 60_000_000_000


def hub_bar_spec(spec: str) -> str:
    """A strategy's bar spec as fed by the hub: 15-MINUTE-LAST-INTERNAL -> 15-MINUTE-LAST-EXTERNAL."""
    if not spec.endswith("-LAST-INTERNAL"):
        raise ValueError(f"bar spec {spec!r} can't be built from the hub's 1-minute bars")
    return spec.replace("-INTERNAL", "-EXTERNAL")


def instrument_from(defn: dict):
    from nautilus_trader import model

    return getattr(model, defn["type"]).from_dict(defn)


class _Building:
    """A longer bar being built from minutes: ends at `end` (ns), whole when its first minute was seen."""

    def __init__(self, end: int, whole: bool, m: dict) -> None:
        self.end, self.whole = end, whole
        self.o, self.h, self.l, self.c, self.v = (Decimal(m[k]) for k in ("o", "h", "l", "c", "v"))

    def add(self, m: dict) -> None:
        self.h, self.l = max(self.h, Decimal(m["h"])), min(self.l, Decimal(m["l"]))
        self.c, self.v = Decimal(m["c"]), self.v + Decimal(m["v"])


class Decoder:
    """Hub messages -> Nautilus data, delivering each instrument's minutes in order only once, and only while
    current: a minute that closed more than LATE_BAR_SECONDS before it arrived (the hub refilling minutes it
    missed) is stored by the hub for warm-ups but not traded on here, as a strategy building its own bars never
    saw the minutes its feed missed either. bar_spec: the strategy's bars, e.g. 15-MINUTE-LAST-EXTERNAL, built
    from those minutes; the first, begun before the node started, is a part bar and is not sent."""

    def __init__(self, bar_spec: str = "1-MINUTE-LAST-EXTERNAL") -> None:
        self.bar_spec = bar_spec
        self.period = int(BarType.from_str(f"X.Y-{bar_spec}").spec.timedelta.total_seconds()) * 1_000_000_000
        self.last_bar: dict[str, int] = {}
        self.types: dict[str, BarType] = {}
        self.building: dict[str, _Building] = {}
        self.sent: set[str] = set()  # instruments with a bar sent: a part bar after that is a gap, still sent
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
            return self._build(m, now_ns)
        return None  # heartbeats, gaps and anything newer: read for their effect, not passed on

    def _build(self, m: dict, now_ns: int):
        iid, close = m["id"], m["ts"]
        end = -(-close // self.period) * self.period  # the close of the bar this minute is part of
        b, out = self.building.get(iid), None
        if b is not None and b.end != end:  # the last bar's closing minute never came (the hub was away)
            del self.building[iid]
            out = self._bar(iid, b, now_ns) if now_ns - b.end <= LATE_BAR_SECONDS * 1_000_000_000 else None
            b = None
        if b is None:
            b = self.building[iid] = _Building(end, close - MINUTE_NS == end - self.period, m)
        else:
            b.add(m)
        if close == end:
            del self.building[iid]
            out = self._bar(iid, b, now_ns)
        return out

    def _bar(self, iid: str, b: _Building, now_ns: int):
        if not b.whole and iid not in self.sent:
            return None
        self.sent.add(iid)
        bt = self.types.get(iid) or self.types.setdefault(iid, BarType.from_str(f"{iid}-{self.bar_spec}"))
        return Bar(bt, Price.from_str(str(b.o)), Price.from_str(str(b.h)), Price.from_str(str(b.l)),
                   Price.from_str(str(b.c)), Quantity.from_str(str(b.v)), b.end, now_ns)


class HubDataClientConfig(DataClientConfig):
    def __init__(self, *, venue: str, instrument_ids: tuple[str, ...], host: str = "hub", port: int = 7700,
                 bar_spec: str = "1-MINUTE-LAST-EXTERNAL", **kwargs) -> None:
        super().__init__(**kwargs)
        self.venue, self.instrument_ids, self.host, self.port = venue, tuple(instrument_ids), host, port
        self.bar_spec = bar_spec


class HubDataClient(MarketDataClient):
    def __init__(self, *, name: str, config: HubDataClientConfig, cache, clock) -> None:
        super().__init__(name=name, config=config, cache=cache, clock=clock, venue=Venue(config.venue))
        self.cfg = config
        self.decode = Decoder(config.bar_spec)
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
