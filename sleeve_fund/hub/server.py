"""The hub's fan-out: one TCP server, on its own thread and event loop, sending each message to every client
subscribed to its instrument (sleeve_fund.hub.protocol). The venue side (sleeve_fund.hub.relay) publishes from
the Nautilus node's thread through publish(), which only hands the message over; nothing here can hold up the
venue connection.

A client that falls behind by more than MAX_QUEUE messages is cut off rather than slowing everyone else: it
sees its feed go quiet, its stale-feed guard stops new entries, and it reconnects."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Callable

from sleeve_fund.hub import protocol

MAX_QUEUE = 20_000


class _Client:
    def __init__(self, writer: asyncio.StreamWriter, ids: set[str]) -> None:
        self.writer, self.ids = writer, ids
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(MAX_QUEUE)
        self.dropped = False
        self.task: asyncio.Task | None = None


class Fanout:
    """known: the instrument ids this hub relays (a callable, so instruments added while it runs count).
    want: called with ids a client asked for that aren't relayed yet, so the relay can add them.
    instruments: the definitions (Instrument.to_dict) of the ids asked for, sent in the hello."""

    def __init__(self, venue: str, known: Callable[[], set[str]], want: Callable[[set[str]], None] | None = None,
                 venue_up: Callable[[], bool] = lambda: True, heartbeat: float = protocol.HEARTBEAT_SECONDS,
                 instruments: Callable[[set[str]], list[dict]] = lambda ids: [], log=print) -> None:
        self.venue, self.known, self.want, self.venue_up = venue, known, want, venue_up
        self.instruments = instruments
        self.heartbeat, self.log = heartbeat, log
        self.clients: set[_Client] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.port: int | None = None
        self.cut_off = 0  # clients dropped for falling behind, since start
        self._ready = threading.Event()
        self._server: asyncio.base_events.Server | None = None

    # --- from any thread ---------------------------------------------------------------

    def publish(self, msg: dict) -> None:
        """Send msg to the clients subscribed to msg["id"] (to every client when it has none). Thread-safe."""
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._send, msg)

    def start(self, host: str = "0.0.0.0", port: int = 0) -> int:
        """Serve on a thread of its own; returns the port (port 0 picks a free one)."""
        threading.Thread(target=self._run, args=(host, port), name="hub-fanout", daemon=True).start()
        if not self._ready.wait(10):
            raise RuntimeError("the hub's server didn't start")
        return self.port

    def stop(self) -> None:
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)

    # --- on the server's loop ----------------------------------------------------------

    def _run(self, host: str, port: int) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._server = self.loop.run_until_complete(asyncio.start_server(self._serve, host, port,
                                                                         limit=protocol.MAX_LINE))
        self.port = self._server.sockets[0].getsockname()[1]
        self.loop.create_task(self._beat())
        self._ready.set()
        try:
            self.loop.run_forever()
        finally:
            self._server.close()
            for c in list(self.clients):
                c.writer.close()
            pending = asyncio.all_tasks(self.loop)
            for t in pending:
                t.cancel()
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.close()

    def _send(self, msg: dict) -> None:
        line, inst = protocol.encode(msg), msg.get("id")
        for c in list(self.clients):
            if inst is not None and inst not in c.ids:
                continue
            try:
                c.queue.put_nowait(line)
            except asyncio.QueueFull:
                self._cut(c, "fell behind")

    def _cut(self, c: _Client, why: str) -> None:
        if c in self.clients:
            self.clients.discard(c)
            self.cut_off += 1
            c.dropped = True
            self.log(f"hub {self.venue}: client cut off ({why})")
            if c.task is not None:
                c.task.cancel()  # it may be waiting on its queue; its finally closes the socket

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat)
            self._send({"t": "hb", "ts": time.time_ns(), "venue_up": bool(self.venue_up())})

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), 10)
            ids = set(protocol.check_subscription(protocol.decode(line)))
        except (ValueError, asyncio.TimeoutError, asyncio.LimitOverrunError) as exc:
            with contextlib.suppress(OSError):
                writer.write(protocol.encode({"t": "error", "v": protocol.VERSION, "message": str(exc)}))
                await writer.drain()
            writer.close()
            return
        missing = ids - self.known()
        if missing and self.want is not None:
            self.want(missing)  # relayed from the next pass of the relay's timer
        c = _Client(writer, ids)
        c.task = asyncio.current_task()
        self.clients.add(c)
        writer.write(protocol.encode({"t": "hello", "v": protocol.VERSION, "venue": self.venue,
                                      "pending": sorted(missing), "instruments": self.instruments(ids)}))
        try:
            while not c.dropped:
                line = await c.queue.get()
                writer.write(line)
                await writer.drain()
        except (ConnectionError, OSError, asyncio.CancelledError):
            pass
        finally:
            self.clients.discard(c)
            writer.close()
