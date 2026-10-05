"""The hub's venue side (v2 P1-1): one Nautilus node per venue, holding the venue's one public market-data
connection, relaying every trade and quote and building each 1-minute bar once (Nautilus's own time-bar
aggregator, stamped at the bar's close, as the paper strategies build theirs today). Everything goes to the
fan-out (sleeve_fund.hub.server); closed bars also go to `sink`, the storage side's hook (the history store),
which this module leaves to its owner.

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
from nautilus_trader.model import BarType, InstrumentId

from sleeve_fund.hub import protocol

MINUTE_NS = 60_000_000_000
REFRESH_SECONDS = 60  # how often the relay picks up instruments asked for since it started
VENUE_QUIET_SECONDS = 60  # no trade or quote for this long: the heartbeat says the venue is down


def bar_type_for(instrument_id: str) -> BarType:
    return BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-INTERNAL")


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

    return sink


def refill_bars(recent, pair: str, instrument_id: str, since_ns: int, until_ns: int, now_ns: int) -> list[dict]:
    """The venue's own closed 1-minute candles closing from since_ns to until_ns, as refilled bar messages.
    recent: the venue profile's ohlc_history ((pair, minutes) -> candles by open time, newest still forming)."""
    r = recent(pair, 1).iloc[:-1]
    closes = (r.index + pd.Timedelta(minutes=1)).as_unit("ns").asi8
    keep = (closes >= since_ns) & (closes <= until_ns)
    return [protocol.bar(instrument_id, row.open, row.high, row.low, row.close, row.volume, int(close), now_ns,
                         refilled=True) for row, close in zip(r[keep].itertuples(), closes[keep])]


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
        self.subscribe_bars(bar_type_for(iid))
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
        n = self.late.setdefault(str(tick.instrument_id), [0, 0])
        n[0] += tick.ts_event < self.gaps.last.get(str(tick.instrument_id), 0)  # its minute's bar already built
        n[1] += 1
        self.fanout.publish(protocol.trade(tick, time.time_ns()))

    def on_quote(self, tick) -> None:
        self._last_tick = time.time()
        self.fanout.publish(protocol.quote(tick, time.time_ns()))

    def on_bar(self, bar) -> None:
        msg = protocol.bar_from_nautilus(bar, time.time_ns())
        iid, close = msg["id"], msg["ts"]
        missing = self.gaps.see(iid, close)
        # The first bar after subscribing opened before the hub saw any of its trades: a part bar. A bar with no
        # volume was built while no trades arrived (Nautilus builds one at the last price), a dropped connection
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
        if pair is None or self.recent is None:
            return
        try:
            bars = refill_bars(self.recent, pair, iid, since_ns, until_ns, time.time_ns())
        except Exception as exc:  # noqa: BLE001 - the venue's REST down too: the gap stays announced
            print(f"hub: refill of {iid} failed: {exc!r}")
            return
        for b in bars:
            self.fanout.publish(b)
        if bars:
            self._store.submit(self._keep, bars)

    def _keep(self, msgs: list[dict]) -> None:
        try:
            self.sink(msgs)
        except Exception as exc:  # noqa: BLE001 - a failed write mustn't stop the relay; the store's gap shows it
            print(f"hub: storing {len(msgs)} bar(s) of {msgs[0]['id']} from {msgs[0]['ts']} failed: {exc!r}")
