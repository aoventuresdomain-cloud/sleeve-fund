"""Run a venue's market data hub (v2 P1-1): python -m sleeve_fund.hub run --venue <venue> [--port 7700].

The hub is its venue's one history-store writer: besides the closed bars it relays, it runs the store's REST
backfill and funding refresh (python -m sleeve_fund.history run) on a thread of its own, so no second process
writes the venue's folder. Public market data only: it refuses to start with any venue credential in its
environment, as paper does."""

from __future__ import annotations

import argparse
import sys
import threading

import pandas as pd

from sleeve_fund.hub.relay import HubRelay, HubRelayConfig, store_sink
from sleeve_fund.hub.server import Fanout
from sleeve_fund.paper.safety import assert_keyless
from sleeve_fund.venues import venue as venue_profile

DEFAULT_PORT = 7700


def instruments(profile, store=None) -> dict[str, str]:
    """{instrument id: pair} for every instrument the venue's history store keeps (its core list, every
    strategy's instrument and those Research asked for: sleeve_fund.history._pairs_in_use)."""
    from sleeve_fund.history import _pairs_in_use

    return {f"{profile.symbol_of(pair)}.{profile.name}": pair for pair, _ in _pairs_in_use(profile.name, store)}


def last_closes(profile, pairs: dict[str, str], history=None, alert=None) -> dict[str, int]:
    """Per instrument, the close of the last minute the history store holds, so the hub refills what it missed
    while it was down. Read only: the store's writer is the storage side's. An instrument whose coverage can't be read
    (a torn coverage.json after a crash) is left out and alerted, `alert(key, message)`: the hub starts anyway, and its
    first live bar is relayed unseeded (F-HRF-1: no hub would cost every Strategy on the venue its data)."""
    from sleeve_fund.history import HistoryStore

    hs = history or HistoryStore()
    out = {}
    for iid, pair in pairs.items():
        try:
            cov = hs.coverage(profile.name, pair)
        except Exception as exc:  # noqa: BLE001 - an unreadable store never stops the hub; the alert says which
            message = (f"hub {profile.name} {pair}: can't read the stored history's coverage ({exc!r}); the hub "
                       f"started without it, so minutes missed while it was down are not refilled at start")
            print(message)
            if alert is not None:
                alert(f"hub-coverage-unreadable:{profile.name}:{pair}", message)
            continue
        if cov is not None:  # the last stored minute may still have been forming: count from the one before
            out[iid] = int(pd.Timestamp(cov.last).value)
    return out


REFILL_LOOKBACK = pd.Timedelta(hours=24)  # holes this recent are refilled at start: about as far as venues' candles go


def start_holes(profile, pairs: dict[str, str], history=None, now=None) -> dict[str, list[tuple[int, int]]]:
    """Per instrument, the stored history's holes from the last REFILL_LOOKBACK, as (first, last) bar closes in ns,
    for the relay to refill at start (HUB-RESTART-FILL). The stretch since the last stored minute isn't one: the first
    live bar finds it (last_closes). Read only; a store it can't read is skipped, the hub starts anyway."""
    from sleeve_fund.history import HistoryStore

    hs = history or HistoryStore()
    since = (now or pd.Timestamp.now(tz="UTC")) - REFILL_LOOKBACK
    minute = pd.Timedelta(minutes=1)
    out = {}
    for iid, pair in pairs.items():
        try:
            runs = hs.gaps(profile.name, pair)
        except Exception as exc:  # noqa: BLE001 - an unreadable month: the store's own checks report it
            print(f"hub {profile.name} {pair}: can't read the history's gaps: {exc!r}")
            continue
        found = [(int((max(a, since) + minute).value), int((b + minute).value)) for a, b in runs if b >= since]
        if found:
            out[iid] = found
    return out


def seeder(profile, pairs: dict[str, str], history):
    """relay.attach's seed: an instrument's last stored close and holes, read when it is first relayed."""

    def seed(iid: str) -> tuple[int | None, list[tuple[int, int]]]:
        pair = pairs.get(iid)
        if pair is None:
            return None, []
        one = {iid: pair}
        holes = start_holes(profile, one, history, now=history.clock())  # the store's clock: tests pass their own
        return last_closes(profile, one, history).get(iid), holes.get(iid, [])

    return seed


def _journal_event(sleeve, level, kind, message, ts=None) -> None:
    """A15: a hub staleness event in the journal, where the forwarder sends warnings on. The journal is opened on
    first use, so a hub with no database (locally) still runs and says so in its log."""
    global _journal
    try:
        if _journal is None:
            from sleeve_fund.store import Store

            _journal = Store(portfolio_gate=True)  # the paper journal runs under the gate; the hub only writes events
        _journal.event(sleeve, level, kind, message, ts=ts)
    except Exception as exc:  # noqa: BLE001 - the log line still says it
        print(f"hub: could not journal {kind}: {exc!r} ({message})")


_journal = None


def build(profile, port: int, sink=None):
    from nautilus_trader.common import Environment, LoggerConfig, LogLevel
    from nautilus_trader.live import LiveNode
    from nautilus_trader.model import TraderId

    if profile.data_client is None:
        raise SystemExit(f"{profile.label} has no live market data client")
    pairs = instruments(profile)
    relay = HubRelay(HubRelayConfig(instrument_ids=tuple(sorted(pairs))))
    fanout = Fanout(profile.name, known=relay.known, want=relay.want, venue_up=relay.venue_up,
                    instruments=relay.instruments)
    from sleeve_fund.history import HistoryStore, _alert
    from sleeve_fund.ops_health import HubStaleness

    store = HistoryStore()
    if sink is None:
        sink = store_sink(store, profile.name, pairs)  # the same dict the relay keeps up to date
    relay.attach(fanout, sink=sink, pairs=pairs, recent=profile.ohlc_history,
                 last_close=last_closes(profile, pairs, store, alert=lambda key, message: _alert(
                     key, "warning", "hub_history_unreadable", message)),
                 holes=start_holes(profile, pairs, store),
                 alert=lambda key, message: _alert(key, "warning", "hub_refill_failed", message),
                 seed=seeder(profile, pairs, store),
                 discover=lambda: instruments(profile), late_path=store.root / f"hub-late-{profile.name}.json",
                 stale=HubStaleness(profile.name), event=_journal_event)
    data_factory, data_config = profile.data_client()
    node = (
        LiveNode.builder(f"HUB-{profile.name}", TraderId.from_str(f"HUB-{profile.name}"), Environment.SANDBOX)
        .with_logging(LoggerConfig(stdout_level=LogLevel.INFO))
        .with_reconciliation(reconciliation=False)
        .add_data_client(None, data_factory, data_config)
        .build()
    )
    node.add_actor(relay)
    fanout.start(port=port)
    return node, fanout, relay


def collector(profile) -> threading.Thread:
    """The store's REST backfill and funding refresh (sleeve_fund.history's run loop), in this process: the store
    takes one writer per venue, and its lock is per process, shared by the backfill and the hub's bars."""
    from sleeve_fund import history

    def run() -> None:
        try:
            code = history.main(["run", "--venue", profile.name.lower()])
            print(f"hub {profile.name}: the history backfill stopped ({code}); live bars are still stored")
        except Exception as exc:  # noqa: BLE001 - the relay keeps going; the store's gaps show what's missing
            print(f"hub {profile.name}: the history backfill failed: {exc!r}; live bars are still stored")

    t = threading.Thread(target=run, name="hub-backfill", daemon=True)
    t.start()
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.hub", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="hold the venue connection and serve its trades, quotes and 1-minute bars")
    run.add_argument("--venue", default=None)
    run.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)
    assert_keyless()
    profile = venue_profile(args.venue)
    from sleeve_fund.history import HistoryStore, unwritable

    if problem := unwritable(HistoryStore().root / profile.name.upper()):
        print(f"hub {profile.name}: can't start: {problem}")
        return 2
    node, fanout, _ = build(profile, args.port)
    collector(profile)
    print(f"hub {profile.name}: serving on port {fanout.port}")
    try:
        node.run()
    finally:
        fanout.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
