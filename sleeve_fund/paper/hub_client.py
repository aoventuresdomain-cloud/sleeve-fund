"""A paper node's market data from its venue's hub (v2 P1-1) instead of a venue connection of its own: a
Nautilus data client that reads the hub's stream (sleeve_fund.hub.protocol) and hands the node the same trades,
quotes and closed 1-minute bars every other strategy on the instrument gets, and the instrument's definition.

Bars reach the strategy as `<instrument>-<its bar>-LAST-EXTERNAL`. A strategy on longer bars gets them built here
from the hub's minutes, as a backtest builds them from the store's: a 15-minute bar is the minutes closing in it,
sent as the one closing at its end arrives and stamped at that close. Building them here rather than on the
node's clock means a bar never closes before its last minute has arrived. A bar at or before the last one
delivered (a refill landing after the live bars moved on) is not delivered again. When the hub goes away the
client reconnects; until it does, nothing arrives, so no bar closes and the strategy decides nothing. Minutes the
client missed while the hub kept them (a blip on the client's side), and those of the bar already under way when
the node starts, come from the history store the hub writes (QA P1-C2, P1-C5)."""

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

from sleeve_fund import bars as bar_rule
from sleeve_fund.hub import protocol

RECONNECT_SECONDS = (1, 2, 5)  # the hub is on the same host: back within seconds of it
CONNECT_TIMEOUT = 90  # the hub picks up an instrument it doesn't relay yet within a minute
LATE_BAR_SECONDS = 90  # a bar complete this long after its close is late: the strategy only exits on it
HEALTHY_SECONDS = 30  # a stream back this long is reconnected: one that breaks sooner keeps the backoff growing
# A live minute that shows a gap the history store can't fill yet waits this long for the hub's refill (QA P1-L2).
HOLD_SECONDS = 20
HUB_ALIVE_SECONDS = 30  # a hub heartbeat this recent means the hub itself is up (heartbeats come every 5 s)


MINUTE_NS = 60_000_000_000
SKIPPED_KEEP_NS = 6 * 60 * MINUTE_NS  # how long a refill of a minute passed over is still sent late


def hub_bar_spec(spec: str) -> str:
    """A strategy's bar spec as fed by the hub: 15-MINUTE-LAST-INTERNAL -> 15-MINUTE-LAST-EXTERNAL."""
    if not spec.endswith("-LAST-INTERNAL"):
        raise ValueError(f"bar spec {spec!r} is the venue's own candles, which the hub doesn't relay: a hub-fed "
                         "strategy decides on 1, 5 or 15-minute or 1-hour bars built from its minutes")
    return spec.replace("-INTERNAL", "-EXTERNAL")


def instrument_from(defn: dict):
    from nautilus_trader import model

    return getattr(model, defn["type"]).from_dict(defn)


class HubStatus:
    """What the hub last said about itself, shared with the strategy's price watchdog: while the hub is up but
    has lost its venue, restarting the node can't bring prices back (QA P1-C9)."""

    def __init__(self) -> None:
        self.heartbeat_ns = 0  # node clock, when the last heartbeat arrived
        self.venue_up = False
        # Bars sent with too many minutes missing (sleeve_fund.bars), by close ns -> minutes missing: the strategy
        # takes them before deciding on the bar, so no entry is opened on one, as in a backtest (board 5a, QA P1-D2).
        self.degraded: dict[int, int] = {}
        # Every bar sent with minutes missing, degraded or not (close ns -> minutes missing), for the strategy's slower
        # candles to count (v2 P1-4, Advisor 4.2).
        self.missing: dict[int, int] = {}
        # instrument -> closes of 1-minute bars lost to the hub being away (in a gap it announced, or before one it
        # announced without them: a hub restarted without its gap state, QA P1-L16) and not refilled yet. The
        # strategy opens nothing while any are (the L16 condition); minutes no trade happened in aren't here.
        self.lost: dict[str, set[int]] = {}

    def unfilled(self, iid: str) -> list[int]:
        return sorted(self.lost.get(iid, ()))

    def venue_down(self, now_ns: int) -> bool:
        """The hub is alive (a recent heartbeat) and says its venue connection is down."""
        return now_ns - self.heartbeat_ns <= HUB_ALIVE_SECONDS * 1_000_000_000 and not self.venue_up


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


def _span(closes: list[int]) -> str:
    n = len(closes)
    return (f"{n} minute{'s' * (n > 1)} closing {_hhmm(closes[0])}"
            f"{f' to {_hhmm(closes[-1])}' if n > 1 else ''}")


def _decimals(text: str) -> int:
    return len(text.partition(".")[2])


def _as_list(got) -> list:
    return got if isinstance(got, list) else [] if got is None else [got]


def _one_or_all(bars: list):
    """None, the one bar, or (minutes recovered across several bars) all of them, oldest first."""
    return None if not bars else bars[0] if len(bars) == 1 else bars


class Decoder:
    """Hub messages -> Nautilus data, delivering each instrument's bars in order and once.

    bar_spec: the strategy's bars, e.g. 15-MINUTE-LAST-EXTERNAL, built from the hub's minutes as a backtest builds
    them from the store's. Every minute goes into the bar it closes in, including one the hub refilled late or out
    of order while that bar is still being built, so a bar holds the same minutes the store's does. A minute for a
    bar already sent is history only. A bar is sent when its last minute arrives, or with the first minute after it
    if that never comes. A bar complete only more than LATE_BAR_SECONDS after its close (the hub refilling
    minutes it missed) is still sent, so the strategy's indicators see every bar a backtest's do and a stop or
    target breached inside it still runs (QA P1-C1); its receive stamp shows it late, and the strategy's late
    rule decides on it: no entries or additions, exits always (LongFlatStrategy._late_entry). The bar under way
    when the node starts is completed from the store (recover) and sent; one the store can't complete is a part
    bar, not sent, and told.

    recover(instrument_id, after_ns, before_ns): the stored minutes closing strictly between the two, as (close
    ns, open, high, low, close, volume) in time order: the history store the hub writes. Asked when minutes are
    missing between two the client got (the client lost them while the hub kept them, QA P1-C2), and at start
    for the minutes of the longer bar already under way (QA P1-C5). A recovered minute joins its bar as a live
    one would; on 1-minute bars each is sent as a bar, under the same late rule, and the gap is told.

    report(level, kind, message): told of late bars (when the first of a run is sent, then the run's extent
    when the feed is current again), of bars sent with minutes missing, of minutes missed on 1-minute bars, and
    of a bar under way at start that the store couldn't complete."""

    def __init__(self, bar_spec: str = "1-MINUTE-LAST-EXTERNAL", report=None, recover=None,
                 status: HubStatus | None = None) -> None:
        self.bar_spec = bar_spec
        self.status = status
        self.period = int(BarType.from_str(f"X.Y-{bar_spec}").spec.timedelta.total_seconds()) * 1_000_000_000
        self.report = report or (lambda level, kind, message: None)
        self.recover = recover
        self.last_bar: dict[str, int] = {}
        self.types: dict[str, BarType] = {}
        self.building: dict[str, _Building] = {}
        self.sent: set[str] = set()  # instruments with a bar sent: a part bar after that is a gap, still sent
        self.late = 0  # bars sent late
        self.held: dict[str, tuple[dict, int]] = {}  # instrument -> a live minute held for its gap's refill, since
        self.refilling: dict[str, set] = {}  # instrument -> the hub's announced gaps whose refill isn't done yet
        # 1-minute bars: instrument -> minutes a live one went past without (QA P1-L14, L15). A refill of one that
        # lands later is sent late, once, so its stop is still checked (the strategy replays it).
        self.skipped: dict[str, set[int]] = {}
        self.lost = status.lost if status is not None else {}  # shared with the strategy (HubStatus.lost)
        self._late_run: dict[str, list[int]] = {}  # instrument -> the closes of a run of late bars
        self._late_sent: dict[str, set[int]] = {}  # instrument -> minutes sent late (one report per run of them)
        # instrument -> its price and size decimals, from the hub's definitions. A refilled minute's numbers
        # can carry fewer ("60000.1" for 60000.10), so a bar is written at these, not at its minutes' own.
        self.precision: dict[str, tuple[int, int]] = {}

    def __call__(self, m: dict, now_ns: int):
        """The message as Nautilus data: a tick, a bar, a list of bars (a minute that brought back several from
        the store), or None."""
        t = m.get("t")
        if t == "trade":
            return TradeTick(InstrumentId.from_str(m["id"]), Price.from_str(m["px"]), Quantity.from_str(m["qty"]),
                             AggressorSide.from_str(m["side"]), TradeId(m["tid"]), m["ts"], now_ns)
        if t == "quote":
            return QuoteTick(InstrumentId.from_str(m["id"]), Price.from_str(m["bid"]), Price.from_str(m["ask"]),
                             Quantity.from_str(m["bid_qty"]), Quantity.from_str(m["ask_qty"]), m["ts"], now_ns)
        if t == "bar":
            return self._minute(m, now_ns)
        if t == "gap":  # a refill on its way: the live minute after it waits for it (QA P1-L2)
            iid, last = m["id"], self.last_bar.get(m["id"])
            spans = self.refilling.setdefault(iid, set())
            held = self.held.get(iid)
            # QA P1-L16: minutes before the gap the hub announced that never came either and aren't in a gap it is
            # refilling or held for one, so it won't refill them (a hub restarted without its last stored minute)
            before = [] if self.period != MINUTE_NS or last is None else [
                c for c in range(last + MINUTE_NS, m["since"], MINUTE_NS)
                if not any(a <= c <= b for a, b in spans) and (held is None or c != held[0]["ts"])]
            spans.add((m["since"], m["until"]))
            if before:
                self.lost.setdefault(iid, set()).update(before)
                self.report("warning", "hub_gap", f"{iid}: the hub announced a gap from {_hhmm(m['since'])} but not "
                                                  f"{_span(before)} before it, which never came either: no new "
                                                  "entries until they are refilled")
            return None
        if t == "filled":  # released first, so the minutes still missing are known as the gap's (lost)
            out = self._release(m["id"], now_ns, done=True)
            self.refilling.get(m["id"], set()).discard((m["since"], m["until"]))
            return _one_or_all(out)
        return None  # heartbeats and anything newer: read for their effect, not passed on

    def _minute(self, m: dict, now_ns: int, hold: bool = True):
        iid, close = m["id"], m["ts"]
        out = []
        held = self.held.get(iid)
        if held is not None and close >= held[0]["ts"]:  # a newer live minute: wait no longer for the gap's
            del self.held[iid]
            out += _as_list(self._minute(held[0], now_ns, hold=False))
            if close == held[0]["ts"]:
                return _one_or_all(out)
        end = -(-close // self.period) * self.period  # the close of the bar this minute is part of
        b = self.building.get(iid)
        last = self.last_bar.get(iid)
        if last is not None and close <= last:
            # Delivered before, or a refill landing after the live minutes moved on: it still joins the bar
            # being built if that bar lacks it. On 1-minute bars a refill of a minute passed over is sent late, as
            # its own bar (QA P1-L14, L15): the strategy replays it for its exits.
            if b is not None and b.end == end:
                b.minutes.setdefault(close, m)
            elif self.period == MINUTE_NS and m.get("refilled") and close in self.skipped.get(iid, ()):
                self.skipped[iid].discard(close)
                self.lost.get(iid, set()).discard(close)
                if close - MINUTE_NS not in self._late_sent.get(iid, ()):
                    self.report("warning", "hub_gap", f"{iid}: the hub's refill from {_hhmm(close)} came after "
                                                      "later minutes; sent late, for exits only")
                self._late_sent.setdefault(iid, set()).add(close)
                one = _Building(close, self.period)
                one.minutes[close] = m
                out += [x for x in [self._bar(iid, one, now_ns)] if x is not None]
            return _one_or_all(out)
        if last is not None and close - last > MINUTE_NS:  # minutes missing between the last one and this
            after = last
        elif last is None and self.period > MINUTE_NS and close != end - self.period + MINUTE_NS:
            after = end - self.period  # the bar under way at start: its minutes before this one
        else:
            return _one_or_all(out + self._add(iid, m, now_ns) + self._release(iid, now_ns))
        got = self._recovered(iid, after, close, m, now_ns, at_start=last is None)
        out += [b for r in got for b in self._add(iid, r, now_ns)]
        if (hold and last is not None and not m.get("refilled") and len(got) < (close - after) // MINUTE_NS - 1
                and any(a < close and b > after for a, b in self.refilling.get(iid, ()))):
            # QA P1-L2: the hub publishes a gap's live minute before its refill, which the store may not have yet.
            # Hold this minute until the refill is done ("filled"), a newer minute comes, or HOLD_SECONDS pass
            # (flush), so the refilled minutes reach the bars and the strategy in order rather than being dropped.
            self.held[iid] = (m, now_ns)
            return _one_or_all(out)
        if self.period == MINUTE_NS and last is not None:
            skipped = set(range(after + MINUTE_NS, close, MINUTE_NS)) - {r["ts"] for r in got}
            self._skip(iid, skipped, close)
            spans = self.refilling.get(iid, ())
            self.lost.setdefault(iid, set()).update(t for t in skipped if any(a <= t <= b for a, b in spans))
        return _one_or_all(out + self._add(iid, m, now_ns) + self._release(iid, now_ns))

    def _skip(self, iid: str, closes: set[int], now_close: int) -> None:
        """Minutes a live one went past without, kept SKIPPED_KEEP_NS for a refill that may still come."""
        keep = now_close - SKIPPED_KEEP_NS
        self.skipped[iid] = {t for t in self.skipped.get(iid, set()) | closes if t > keep}
        self._late_sent[iid] = {t for t in self._late_sent.get(iid, set()) if t > keep}
        if iid in self.lost:
            self.lost[iid] = {t for t in self.lost[iid] if t > keep}

    def _release(self, iid: str, now_ns: int, done: bool = False) -> list:
        """The minute held for a gap, once the refills have filled it or (done) the hub's refill is over."""
        held = self.held.get(iid)
        if held is None or (not done and held[0]["ts"] - self.last_bar.get(iid, 0) != MINUTE_NS):
            return []
        del self.held[iid]
        return _as_list(self._minute(held[0], now_ns, hold=False))

    def flush(self, now_ns: int) -> list:
        """The minutes held for a gap over HOLD_SECONDS, sent with the gap still in them (the hub's heartbeat
        calls this, so a refill that never comes holds nothing for long)."""
        out = []
        for iid, (m, since) in list(self.held.items()):
            if now_ns - since >= HOLD_SECONDS * 1_000_000_000:
                del self.held[iid]
                out += _as_list(self._minute(m, now_ns, hold=False))
        return out

    def _recovered(self, iid: str, after: int, before: int, like: dict, now_ns: int, at_start: bool) -> list[dict]:
        """The minutes closing between `after` and `before` from the store, as hub messages priced like `like`;
        on 1-minute bars none (they are history), with the gap told."""
        rows = []
        if self.recover is not None:
            try:
                rows = list(self.recover(iid, after, before))
            except Exception as exc:  # noqa: BLE001 - no store to hand: the minutes stay missing, as before
                self.report("warning", "hub_gap", f"{iid}: couldn't read the history store for the minutes the "
                                                  f"feed missed ({type(exc).__name__}: {exc})")
        px, qty = self._digits(iid, [like])
        got = [{"t": "bar", "id": iid, "o": f"{o:.{px}f}", "h": f"{h:.{px}f}", "l": f"{lo:.{px}f}", "c": f"{c:.{px}f}",
                "v": f"{v:.{qty}f}", "ts": int(ts), "recv": now_ns, "refilled": True, "recovered": True}
               for ts, o, h, lo, c, v in rows if after < ts < before]
        if self.period == MINUTE_NS:
            missed = list(range(after + MINUTE_NS, before, MINUTE_NS))
            have = {r["ts"] for r in got}
            lost = [ts for ts in missed if ts not in have]
            self.report("warning", "hub_gap",
                        f"{iid}: the feed missed {_span(missed)}; "
                        + (f"{len(missed) - len(lost)} recovered from the history store and sent late (exits only "
                           f"on any over {LATE_BAR_SECONDS} s)" + (f", {len(lost)} not there either" if lost else "")
                           if self.recover is not None else "the history store wasn't consulted"))
            return got
        if at_start and not any(r["ts"] == after + MINUTE_NS for r in got):  # its first minute makes it whole
            self.report("warning", "warmup",
                        f"{iid}: the bar closing {_hhmm(after + self.period)}, under way when this node started, "
                        f"lacks {(before - after) // MINUTE_NS - 1 - len(got)} of its minutes before the first "
                        "one received, even from the history store: not sent, so the indicators skip it")
        return got

    def _digits(self, iid: str, minutes) -> tuple[int, int]:
        """The instrument's price and size decimals; without its definition (a bare Decoder), the most any of
        these minutes is written with, so no number is rounded."""
        if (known := self.precision.get(iid)) is not None:
            return known
        minutes = list(minutes)
        return (max(_decimals(m[k]) for m in minutes for k in "ohlc"), max(_decimals(m["v"]) for m in minutes))

    def _add(self, iid: str, m: dict, now_ns: int):
        """A minute newer than every one before it: into its bar; the bars it completes, oldest first (the last
        one, whose closing minute never came, and this one's, if this closes it)."""
        close = m["ts"]
        end = -(-close // self.period) * self.period
        b = self.building.get(iid)
        self.last_bar[iid] = close
        out = []
        if b is not None and b.end != end:  # the last bar's closing minute never came (the hub was away)
            del self.building[iid]
            out.append(self._bar(iid, b, now_ns))
            b = None
        if b is None:
            b = self.building[iid] = _Building(end, self.period)
        b.minutes[close] = m
        if close == end:
            del self.building[iid]
            out.append(self._bar(iid, b, now_ns))
        return [x for x in out if x is not None]

    def _bar(self, iid: str, b: _Building, now_ns: int):
        if not b.whole and iid not in self.sent:
            return None
        self.sent.add(iid)
        if now_ns - b.end > LATE_BAR_SECONDS * 1_000_000_000:
            self.late += 1
            run = self._late_run.setdefault(iid, [])
            run.append(b.end)
            if len(run) == 1:  # told as it happens, so it is on record even if no bar follows (QA P1-C3)
                self.report("warning", "bar_late",
                            f"{iid}: the bar closing {_hhmm(b.end)} was complete only "
                            f"{(now_ns - b.end) / 1e9:.0f} s after its close, over the {LATE_BAR_SECONDS} s limit "
                            "(refilled after the feed was away): sent for exits only, no new entries")
        elif (late := self._late_run.pop(iid, None)) and len(late) > 1:
            self.report("warning", "bar_late",
                        f"{iid}: {len(late)} bars closing {_hhmm(late[0])} to {_hhmm(late[-1])} came over "
                        f"{LATE_BAR_SECONDS} s after their close, exits only; the feed is current again")
        if (missing := self.period // MINUTE_NS - len(b.minutes)) > 0:
            self.report("warning", "bar_incomplete",
                        f"{iid}: the bar closing {_hhmm(b.end)} was sent missing {missing} of its "
                        f"{self.period // MINUTE_NS} minutes")
            if self.status is not None:
                self.status.missing[b.end] = missing
                if bar_rule.degraded(missing, self.period // MINUTE_NS):
                    self.status.degraded[b.end] = missing
        bt = self.types.get(iid) or self.types.setdefault(iid, BarType.from_str(f"{iid}-{self.bar_spec}"))
        px, qty = self._digits(iid, b.minutes.values())
        *prices, v = b.ohlcv()
        return Bar(bt, *(Price.from_str(f"{n:.{px}f}") for n in prices), Quantity.from_str(f"{v:.{qty}f}"),
                   b.end, now_ns)


class HubDataClientConfig(DataClientConfig):
    def __init__(self, *, venue: str, instrument_ids: tuple[str, ...], host: str = "hub", port: int = 7700,
                 bar_spec: str = "1-MINUTE-LAST-EXTERNAL", report=None, recover=None,
                 status: HubStatus | None = None, **kwargs) -> None:
        """report(level, kind, message): where skipped and incomplete bars and connection losses are told,
        e.g. the strategy's journal events; by default the node's log. recover: the stored minutes, as
        Decoder takes them. status: kept up to date with the hub's heartbeats, for the strategy's watchdog."""
        super().__init__(**kwargs)
        self.venue, self.instrument_ids, self.host, self.port = venue, tuple(instrument_ids), host, port
        self.bar_spec, self.report, self.recover, self.status = bar_spec, report, recover, status


class HubDataClient(MarketDataClient):
    def __init__(self, *, name: str, config: HubDataClientConfig, cache, clock) -> None:
        super().__init__(name=name, config=config, cache=cache, clock=clock, venue=Venue(config.venue))
        self.cfg = config
        self._report = config.report or self._log_report
        self.decode = Decoder(config.bar_spec, self.report, config.recover, getattr(config, "status", None))
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
        if self._writer is not None:  # the last connection's socket, broken or not, is done with
            self._writer.close()
            self._writer = None
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
        in the cache when it starts. A hub not listening yet (starting, or restarting) is retried with the
        reconnect backoff for as long (QA P1-C9), not left to a restart of the node."""
        deadline = time.monotonic() + CONNECT_TIMEOUT
        attempt = 0
        while True:
            try:
                reader, hello = await self._open()
            except OSError:
                if time.monotonic() > deadline:
                    raise
                await asyncio.sleep(RECONNECT_SECONDS[min(attempt, len(RECONNECT_SECONDS) - 1)])
                attempt += 1
                continue
            defs = {d["id"]: d for d in hello.get("instruments", [])}
            if set(self.cfg.instrument_ids) <= set(defs):
                break
            self._writer.close()
            if time.monotonic() > deadline:
                missing = sorted(set(self.cfg.instrument_ids) - set(defs))
                raise ConnectionError(f"the hub doesn't serve {', '.join(missing)}")
            await asyncio.sleep(5)
        for d in defs.values():
            inst = instrument_from(d)
            self.decode.precision[d["id"]] = (inst.price_precision, inst.size_precision)
            self._handle_instrument(inst)
        self._reader_task = self.create_task(self._read(reader), name="hub-read")

    async def _read(self, reader: asyncio.StreamReader) -> None:
        """Until the node stops: hands the stream's data to the node, and on losing it for any reason (the hub
        restarting, a message it can't read) reconnects, saying once when it went and when it came back. It is
        back once the stream has flowed for HEALTHY_SECONDS: until then the backoff keeps growing, so a stream
        that breaks after every message or two is retried every 5 s and told once, not every second (QA
        P1-C6)."""
        attempt, lost, opened = 0, None, time.monotonic()
        while not self._closing:
            try:
                while line := await reader.readline():
                    m = json.loads(line)
                    now = self.clock.timestamp_ns()
                    if m.get("t") == "hb":
                        self.last_heartbeat_ns, self.venue_up = now, bool(m.get("venue_up"))
                        if (status := getattr(self.cfg, "status", None)) is not None:
                            status.heartbeat_ns, status.venue_up = now, self.venue_up
                        for d in self.decode.flush(now):
                            self._handle_data(d)
                    elif (data := self.decode(m, now)) is not None:
                        for d in data if isinstance(data, list) else (data,):
                            self._handle_data(d)
                    if attempt and time.monotonic() - opened >= HEALTHY_SECONDS:
                        attempt = 0
                    if lost is not None and not attempt:
                        self.report("info", "hub", f"Reconnected to the market data hub after "
                                                   f"{opened - lost:.0f} s")
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
            opened = time.monotonic()

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
