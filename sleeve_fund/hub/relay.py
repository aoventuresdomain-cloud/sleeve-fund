"""The hub's venue side (v2 P1-1): one Nautilus node per venue, holding the venue's one public market-data
connection, relaying every trade and quote and building each 1-minute bar once, stamped at the bar's close.
Everything goes to the fan-out (sleeve_fund.hub.server); closed bars also go to `sink`, the storage side's hook
(the history store), which this module leaves to its owner.

A trade counts in the minute of its venue time, as the venue's own candle counts it, not the minute it reached
the hub in (P1-1-DELAY): a minute closes BAR_GRACE_SECONDS after its end, so a trade from its last moments that
arrives just after the boundary still lands in it. Nautilus's own time-bar aggregator buckets by arrival time, and
its build delay only shifts that window (and the bar's stamp), so the hub does not use it.

A minute the hub missed (the venue connection dropped, or the hub was down) is announced as a gap and refilled
from the venue's REST candles, flagged as refilled, on a worker thread so the venue connection never waits.
Writes to the store run on a thread of their own too, in the order the bars were seen."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import InstrumentId, Quantity

from sleeve_fund.hub import protocol

MINUTE_NS = 60_000_000_000
REFRESH_SECONDS = 60  # how often the relay picks up instruments asked for since it started
VENUE_QUIET_SECONDS = 60  # no trade or quote for this long: the heartbeat says the venue is down
BAR_GRACE_SECONDS = 2  # a minute's bar is built this long after the minute ends; a trade later than that is late


class Minutes:
    """Per instrument, the 1-minute bars under way, built from trades by their venue time. A minute is keyed by its
    close: a trade at 12:00:59.9 is in the minute closing 12:01, one at exactly 12:01:00 in the next (the venue's
    candle opens at its open time). Bars come out of close() once the minute is over; a trade for a minute already
    closed is refused (add returns False), never put in a later one."""

    def __init__(self) -> None:
        self.bars: dict[str, dict[int, list]] = {}  # instrument id -> {close ns: [open, high, low, close, volume]}
        self.closed: dict[str, int] = {}  # instrument id -> close of the last minute built
        self.last_px: dict[str, object] = {}  # instrument id -> the last closed bar's close, for a minute without trades
        self.no_qty: dict[str, Quantity] = {}  # instrument id -> zero at its size decimals

    def add(self, instrument_id: str, ts_ns: int, px, qty) -> bool:
        close = (ts_ns // MINUTE_NS + 1) * MINUTE_NS
        if close <= self.closed.get(instrument_id, 0):
            return False
        self.no_qty.setdefault(instrument_id, Quantity(0, qty.precision))
        b = self.bars.setdefault(instrument_id, {}).get(close)
        if b is None:
            self.bars[instrument_id][close] = [px, px, px, px, qty]
        else:
            b[1], b[2], b[3], b[4] = max(b[1], px), min(b[2], px), px, b[4] + qty
        return True

    def close(self, now_ns: int, grace_ns: int) -> list[tuple]:
        """(instrument id, close ns, open, high, low, close, volume) for every minute over by now_ns - grace_ns, in
        time order per instrument. A minute with no trades between two built ones comes out flat at the last close
        with no volume, as Nautilus builds one (the relay stands the venue's candle in for it)."""
        upto = (now_ns - grace_ns) // MINUTE_NS * MINUTE_NS
        out = []
        for iid in sorted(set(self.bars) | set(self.last_px)):
            open_ = self.bars.get(iid, {})
            due = sorted(c for c in open_ if c <= upto)
            prev = self.closed.get(iid)
            start = prev + MINUTE_NS if prev is not None else (due[0] if due else None)
            if start is None:
                continue
            for close in range(start, upto + 1, MINUTE_NS):
                b = open_.pop(close, None)
                if b is None:
                    px = self.last_px[iid]
                    b = [px, px, px, px, self.no_qty[iid]]
                out.append((iid, close, *b))
                self.last_px[iid], self.closed[iid] = b[3], close
        return out


class Gaps:
    """Per instrument, the close of the last bar seen; says which minutes a new bar skipped over."""

    def __init__(self) -> None:
        self.last: dict[str, int] = {}

    def see(self, instrument_id: str, close_ns: int) -> tuple[int, int] | None:
        """(first, last) close of the minutes missing before this bar, or None. A bar at or before the last
        one seen (a refill landing late) changes nothing."""
        prev = self.last.get(instrument_id)
        if prev is not None and close_ns <= prev:
            return None
        self.last[instrument_id] = close_ns
        if prev is not None and close_ns - prev > MINUTE_NS:
            return prev + MINUTE_NS, close_ns - MINUTE_NS
        return None


def store_sink(history, venue: str, pairs: dict[str, str], log=print) -> Callable[[list[dict]], None]:
    """Hands closed bars to the history store's write path, HistoryStore.append_bars(venue, instrument, bars,
    source): bars as (open_time_ns, open, high, low, close, volume), source "live" or "refill", one call per
    instrument and source (a refill's bars in one call). Writes are idempotent, and a refill differing from a
    stored bar is recorded by the store, not written over it; the hub logs those."""

    def sink(msgs: list[dict]) -> None:
        batches: dict[tuple[str, str], list] = {}
        for m in msgs:
            pair = pairs.get(m["id"])
            if pair is not None:
                bar = (m["ts"] - MINUTE_NS, *(float(m[k]) for k in ("o", "h", "l", "c", "v")))
                batches.setdefault((pair, "refill" if m["refilled"] else "live"), []).append(bar)
        for (pair, source), bars in batches.items():
            out = history.append_bars(venue, pair, bars, source)
            if out.conflicts:
                log(f"hub {venue} {pair}: {out.conflicts} {source} bar(s) differ from the stored ones; "
                    "kept the stored, the store records the difference")
            if out.replaced:
                log(f"hub {venue} {pair}: {out.replaced} live bar(s) replaced by the venue's candle; "
                    "the store records both")

    return sink


def refill_bars(recent, pair: str, instrument_id: str, since_ns: int, until_ns: int, now_ns: int,
                precision: tuple[int, int] | None = None) -> list[dict]:
    """The venue's own closed 1-minute candles closing from since_ns to until_ns, as refilled bar messages.
    recent: the venue profile's ohlc_history ((pair, minutes) -> candles by open time, newest still forming).
    precision: the instrument's price and size decimals, so a refilled minute is written as a live one is
    (60000.10, not 60000.1)."""
    r = recent(pair, 1).iloc[:-1]
    closes = (r.index + pd.Timedelta(minutes=1)).as_unit("ns").asi8
    keep = (closes >= since_ns) & (closes <= until_ns)
    px, qty = (f".{precision[0]}f", f".{precision[1]}f") if precision else ("", "")
    return [protocol.bar(instrument_id, *(format(x, px) for x in (row.open, row.high, row.low, row.close)),
                         format(row.volume, qty), int(close), now_ns, refilled=True)
            for row, close in zip(r[keep].itertuples(), closes[keep])]


class HubRelayConfig(DataActorConfig):
    def __init__(self, *, instrument_ids: tuple[str, ...] = (), **kwargs) -> None:
        super().__init__(**kwargs)
        self.instrument_ids = tuple(instrument_ids)


class HubRelay(DataActor):
    """Subscribes to every instrument the hub serves and hands what arrives to the fan-out."""

    def __init__(self, config: HubRelayConfig) -> None:
        super().__init__(config)
        self.fanout = None  # attach() before the node runs
        self.sink: Callable[[list[dict]], None] = lambda bars: None
        self.pairs: dict[str, str] = {}  # instrument id -> pair, for the REST refill and the store
        self.recent = self.discover = None
        self.relayed: set[str] = set()
        self.definitions: dict[str, dict] = {}  # instrument id -> the instrument, as Instrument.to_dict()
        self.gaps = Gaps()
        self.minutes = Minutes()
        self._wanted: set[str] = set()
        self._lock = threading.Lock()
        self._last_tick = 0.0
        self._refill = ThreadPoolExecutor(1, thread_name_prefix="hub-refill")
        self._store = ThreadPoolExecutor(1, thread_name_prefix="hub-store")
        self._initial = tuple(config.instrument_ids)
        self._since: dict[str, int] = {}  # when the relay subscribed to each instrument
        # Per instrument, [trades that arrived after their minute's bar was built, all trades], written to
        # late_path every REFRESH_SECONDS as {pair: [late, total]} for the parity report (scripts/hub_parity.py).
        self.late: dict[str, list[int]] = {}
        self.late_path = None

    def attach(self, fanout, sink=None, pairs: dict[str, str] | None = None, recent=None,
               last_close: dict[str, int] | None = None, discover=None, late_path=None) -> "HubRelay":
        """pairs: {instrument id: pair}, kept up to date in place (the sink may hold the same dict). last_close:
        per instrument, the close of the last bar already stored, so the minutes the hub was down are refilled
        from its first bar. discover: () -> {instrument id: pair}, read every REFRESH_SECONDS, so a strategy
        added while the hub runs is relayed and stored without a restart."""
        self.fanout, self.recent, self.discover = fanout, recent, discover
        self.sink = sink or self.sink
        self.pairs = pairs if pairs is not None else {}
        self.gaps.last.update(last_close or {})
        self.late_path = late_path
        return self

    # --- the fan-out asks, from its own thread ------------------------------------------

    def known(self) -> set[str]:
        with self._lock:
            return set(self.relayed)

    def instruments(self, ids: set[str]) -> list[dict]:
        with self._lock:
            return [self.definitions[i] for i in sorted(ids) if i in self.definitions]

    def want(self, ids: set[str]) -> None:
        with self._lock:
            self._wanted |= ids

    def venue_up(self) -> bool:
        return time.time() - self._last_tick < VENUE_QUIET_SECONDS

    # --- the node's thread ------------------------------------------------------------

    def on_start(self) -> None:
        for iid in self._initial:
            self._relay(iid)
        self.clock.set_timer("hub-refresh", pd.Timedelta(seconds=REFRESH_SECONDS).to_pytimedelta(),
                             callback=self._refresh)
        # Every minute, BAR_GRACE_SECONDS after it ends: the only wait the venue-time bars add (P1-1-DELAY).
        first = (self.clock.timestamp_ns() // MINUTE_NS + 1) * MINUTE_NS + BAR_GRACE_SECONDS * 1_000_000_000
        self.clock.set_timer_ns("hub-bars", MINUTE_NS, start_time_ns=first, callback=self._build)

    def _refresh(self, _event=None) -> None:
        self._write_late()
        if self.discover is not None:
            try:
                found = self.discover()
            except Exception as exc:  # noqa: BLE001 - the database away for a moment: try again next pass
                self.log.warning(f"hub: can't list instruments: {exc}")
                found = {}
            self.pairs.update(found)
            self.want(set(found))
        with self._lock:
            wanted, self._wanted = self._wanted, set()
        for iid in sorted(wanted - self.relayed):
            try:
                self._relay(iid)
            except Exception as exc:  # noqa: BLE001 - an id the venue doesn't list: the client stays pending
                self.log.warning(f"hub: can't relay {iid}: {exc}")

    def _relay(self, iid: str) -> None:
        inst = InstrumentId.from_str(iid)
        self.subscribe_trades(inst)
        self.subscribe_quotes(inst)
        self._since[iid] = time.time_ns()
        inst = self.cache.instrument(inst)
        with self._lock:
            self.relayed.add(iid)
            if inst is not None:  # sent to each client on connecting, so paper needs no venue connection of its own
                self.definitions[iid] = inst.to_dict()

    def _write_late(self) -> None:
        if self.late_path is None:
            return
        counts = {self.pairs.get(iid, iid): n for iid, n in self.late.items()}
        try:
            tmp = self.late_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(counts))
            tmp.replace(self.late_path)
        except OSError as exc:
            self.log.warning(f"hub: can't write {self.late_path}: {exc}")

    def on_trade(self, tick) -> None:
        self._last_tick = time.time()
        iid = str(tick.instrument_id)
        n = self.late.setdefault(iid, [0, 0])
        late = tick.ts_event < self.gaps.last.get(iid, 0)  # its minute's bar already built: never put in a later one
        if not late:
            late = not self.minutes.add(iid, tick.ts_event, tick.price, tick.size)
        n[0] += late
        n[1] += 1
        self.fanout.publish(protocol.trade(tick, time.time_ns()))

    def on_quote(self, tick) -> None:
        self._last_tick = time.time()
        self.fanout.publish(protocol.quote(tick, time.time_ns()))

    def _build(self, event=None, now_ns: int | None = None) -> None:
        """The minutes over by BAR_GRACE_SECONDS before now: the timer's own time, so a timer firing a moment late
        builds the same minutes."""
        if now_ns is None:
            now_ns = getattr(event, "ts_event", None) or self.clock.timestamp_ns()
        for iid, close, o, h, l, c, v in self.minutes.close(now_ns, BAR_GRACE_SECONDS * 1_000_000_000):  # noqa: E741
            self._closed(protocol.bar(iid, o, h, l, c, v, close, time.time_ns()))

    def on_bar(self, bar) -> None:
        """A closed bar built elsewhere (a Nautilus bar), handled as the hub's own."""
        self._closed(protocol.bar_from_nautilus(bar, time.time_ns()))

    def _closed(self, msg: dict) -> None:
        iid, close = msg["id"], msg["ts"]
        missing = self.gaps.see(iid, close)
        # The first bar after subscribing opened before the hub saw any of its trades: a part bar. A bar with no
        # volume was built while no trades arrived (Minutes builds one at the last price), a dropped connection
        # as like as a quiet minute. Either way the venue's own candle stands in for it.
        stand_in = close - MINUTE_NS < self._since.get(iid, 0) or float(msg["v"]) == 0
        if missing is not None or stand_in:
            since, until = missing[0] if missing else close, close if stand_in else missing[1]
            self.fanout.publish({"t": "gap", "id": iid, "since": since, "until": until})
            self._refill.submit(self._fill, iid, since, until)
        if not stand_in:
            self.fanout.publish(msg)
            self._store.submit(self._keep, [msg])

    def _fill(self, iid: str, since_ns: int, until_ns: int) -> None:
        pair = self.pairs.get(iid)
        bars = []
        try:
            if pair is not None and self.recent is not None:  # else nothing to refill from: still say filled
                with self._lock:
                    d = self.definitions.get(iid)
                precision = (d["price_precision"], d["size_precision"]) if d else None
                bars = refill_bars(self.recent, pair, iid, since_ns, until_ns, time.time_ns(), precision)
        except Exception as exc:  # noqa: BLE001 - the venue's REST down too: the gap stays announced
            print(f"hub: refill of {iid} failed: {exc!r}")
        for b in bars:
            self.fanout.publish(b)
        if bars:
            self._store.submit(self._keep, bars)
        # Done, found or not: a client holding the live bar for this refill (QA P1-L2) stops waiting.
        self.fanout.publish({"t": "filled", "id": iid, "since": since_ns, "until": until_ns})

    def _keep(self, msgs: list[dict]) -> None:
        try:
            self.sink(msgs)
        except Exception as exc:  # noqa: BLE001 - a failed write mustn't stop the relay; the store's gap shows it
            print(f"hub: storing {len(msgs)} bar(s) of {msgs[0]['id']} from {msgs[0]['ts']} failed: {exc!r}")
