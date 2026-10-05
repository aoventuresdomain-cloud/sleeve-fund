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
from datetime import datetime, timezone
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
    """A longer bar being built from minutes: the minutes closing in (end - period, end], kept by their close so a
    refilled minute can join out of order."""

    def __init__(self, end: int, period: int) -> None:
        self.end, self.start = end, end - period
        self.minutes: dict[int, dict] = {}

    @property
    def whole(self) -> bool:  # its first minute was seen: not a bar begun before the node started
        return self.start + MINUTE_NS in self.minutes

    def ohlcv(self) -> tuple[Decimal, ...]:
        ms = [self.minutes[ts] for ts in sorted(self.minutes)]
        return (Decimal(ms[0]["o"]), max(Decimal(m["h"]) for m in ms), min(Decimal(m["l"]) for m in ms),
                Decimal(ms[-1]["c"]), sum((Decimal(m["v"]) for m in ms), Decimal(0)))


def _hhmm(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%d %b %H:%M UTC")


class Decoder:
    """Hub messages -> Nautilus data, delivering each instrument's bars in order, once, and only while current.

    bar_spec: the strategy's bars, e.g. 15-MINUTE-LAST-EXTERNAL, built from the hub's minutes as a backtest builds
    them from the store's. Every minute goes into the bar it closes in, including one the hub refilled late or out
    of order while that bar is still being built, so a bar holds the same minutes the store's does. A minute for a
    bar already sent is history only. A bar is sent when its last minute arrives, or with the first minute after it
    if that never comes, and only within LATE_BAR_SECONDS of its close: a bar complete only later (the hub
    refilling minutes it missed) is history for warm-ups, not a signal, as a strategy on its own feed never saw
    those minutes either. The first bar, begun before the node started, is a part bar and is not sent.

    report(level, kind, message): told of bars skipped as late (one event per run of them, when the next bar is
    sent) and of bars sent with minutes missing."""

    def __init__(self, bar_spec: str = "1-MINUTE-LAST-EXTERNAL", report=None) -> None:
        self.bar_spec = bar_spec
        self.period = int(BarType.from_str(f"X.Y-{bar_spec}").spec.timedelta.total_seconds()) * 1_000_000_000
        self.report = report or (lambda level, kind, message: None)
        self.last_bar: dict[str, int] = {}
        self.types: dict[str, BarType] = {}
        self.building: dict[str, _Building] = {}
        self.sent: set[str] = set()  # instruments with a bar sent: a part bar after that is a gap, still sent
        self.late = 0  # bars skipped as late
        self._skipped: dict[str, list[int]] = {}  # instrument -> the closes of a run of skipped bars

    def __call__(self, m: dict, now_ns: int):
        t = m.get("t")
        if t == "trade":
            return TradeTick(InstrumentId.from_str(m["id"]), Price.from_str(m["px"]), Quantity.from_str(m["qty"]),
                             AggressorSide.from_str(m["side"]), TradeId(m["tid"]), m["ts"], now_ns)
        if t == "quote":
            return QuoteTick(InstrumentId.from_str(m["id"]), Price.from_str(m["bid"]), Price.from_str(m["ask"]),
                             Quantity.from_str(m["bid_qty"]), Quantity.from_str(m["ask_qty"]), m["ts"], now_ns)
        if t == "bar":
            return self._minute(m, now_ns)
        return None  # heartbeats, gaps and anything newer: read for their effect, not passed on

    def _minute(self, m: dict, now_ns: int):
        iid, close = m["id"], m["ts"]
        end = -(-close // self.period) * self.period  # the close of the bar this minute is part of
        b = self.building.get(iid)
        if close <= self.last_bar.get(iid, 0):
            # Delivered before, or a refill landing after the live minutes moved on: it still joins the bar
            # being built if that bar lacks it.
            if b is not None and b.end == end:
                b.minutes.setdefault(close, m)
            return None
        self.last_bar[iid] = close
        out = None
        if b is not None and b.end != end:  # the last bar's closing minute never came (the hub was away)
            del self.building[iid]
            out = self._bar(iid, b, now_ns)
            b = None
        if b is None:
            b = self.building[iid] = _Building(end, self.period)
        b.minutes[close] = m
        if close == end:  # can't coincide with a bar sent just above: that one would be a whole period late
            del self.building[iid]
            out = self._bar(iid, b, now_ns)
        return out

    def _bar(self, iid: str, b: _Building, now_ns: int):
        if not b.whole and iid not in self.sent:
            return None
        if now_ns - b.end > LATE_BAR_SECONDS * 1_000_000_000:
            self.late += 1
            self._skipped.setdefault(iid, []).append(b.end)
            return None
        self.sent.add(iid)
        if skipped := self._skipped.pop(iid, None):
            n = len(skipped)
            self.report("warning", "bar_skipped",
                        f"{iid}: skipped {n} bar{'s' * (n > 1)} closing {_hhmm(skipped[0])}"
                        f"{f' to {_hhmm(skipped[-1])}' if n > 1 else ''}: complete only over {LATE_BAR_SECONDS} s "
                        "after the close (refilled after the feed was away), so not decided on")
        if (missing := self.period // MINUTE_NS - len(b.minutes)) > 0:
            self.report("warning", "bar_incomplete",
                        f"{iid}: the bar closing {_hhmm(b.end)} was sent missing {missing} of its "
                        f"{self.period // MINUTE_NS} minutes")
        bt = self.types.get(iid) or self.types.setdefault(iid, BarType.from_str(f"{iid}-{self.bar_spec}"))
        o, h, lo, c, v = b.ohlcv()
        return Bar(bt, Price.from_str(str(o)), Price.from_str(str(h)), Price.from_str(str(lo)), Price.from_str(str(c)),
                   Quantity.from_str(str(v)), b.end, now_ns)


class HubDataClientConfig(DataClientConfig):
    def __init__(self, *, venue: str, instrument_ids: tuple[str, ...], host: str = "hub", port: int = 7700,
                 bar_spec: str = "1-MINUTE-LAST-EXTERNAL", report=None, **kwargs) -> None:
        """report(level, kind, message): where skipped and incomplete bars and connection losses are told,
        e.g. the strategy's journal events; by default the node's log."""
        super().__init__(**kwargs)
        self.venue, self.instrument_ids, self.host, self.port = venue, tuple(instrument_ids), host, port
        self.bar_spec, self.report = bar_spec, report


class HubDataClient(MarketDataClient):
    def __init__(self, *, name: str, config: HubDataClientConfig, cache, clock) -> None:
        super().__init__(name=name, config=config, cache=cache, clock=clock, venue=Venue(config.venue))
        self.cfg = config
        self._report = config.report or self._log_report
        self.decode = Decoder(config.bar_spec, self.report)
        self.last_heartbeat_ns = 0
        self.venue_up = False
        self.connects = 0
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task = None
        self._closing = False

    def report(self, level: str, kind: str, message: str) -> None:
        try:
            self._report(level, kind, message)
        except Exception as exc:  # noqa: BLE001 - telling must never stop the feed
            self._log.error(f"couldn't record {kind} ({message}): {exc!r}")

    def _log_report(self, level: str, kind: str, message: str) -> None:
        (self._log.warning if level != "info" else self._log.info)(f"{kind}: {message}")

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
        """Until the node stops: hands the stream's data to the node, and on losing it for any reason (the hub
        restarting, a message it can't read) reconnects, saying once when it went and when it came back."""
        attempt, lost = 0, None
        while not self._closing:
            try:
                while line := await reader.readline():
                    m = json.loads(line)
                    now = self.clock.timestamp_ns()
                    if m.get("t") == "hb":
                        self.last_heartbeat_ns, self.venue_up = now, bool(m.get("venue_up"))
                        continue
                    data = self.decode(m, now)
                    if data is not None:
                        self._handle_data(data)
                    attempt = 0
                    if lost is not None:  # said once the stream flows again, so a stream that breaks at once is quiet
                        self.report("info", "hub", f"Reconnected to the market data hub after "
                                                   f"{time.monotonic() - lost:.0f} s")
                        lost = None
                reason = "the hub closed the connection"
            except Exception as exc:  # noqa: BLE001 - whatever broke the stream, the cure is to reconnect
                reason = f"{type(exc).__name__}: {exc}"
            if self._closing:
                return
            if lost is None:
                lost = time.monotonic()
                self.report("warning", "hub", f"Lost the market data hub ({reason}); reconnecting, no bars until then")
            await asyncio.sleep(RECONNECT_SECONDS[min(attempt, len(RECONNECT_SECONDS) - 1)])
            attempt += 1
            try:
                reader, _ = await self._open()
            except Exception:  # noqa: BLE001 - still away: try again
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
