"""Run a venue's market data hub (v2 P1-1): python -m sleeve_fund.hub run --venue binance [--port 7700].

Public market data only: it refuses to start with any venue credential in its environment, as paper does."""

from __future__ import annotations

import argparse
import sys

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


def last_closes(profile, pairs: dict[str, str], history=None) -> dict[str, int]:
    """Per instrument, the close of the last minute the history store holds, so the hub refills what it missed
    while it was down. Read only: the store's writer is the storage side's."""
    from sleeve_fund.history import HistoryStore

    hs = history or HistoryStore()
    out = {}
    for iid, pair in pairs.items():
        cov = hs.coverage(profile.name, pair)
        if cov is not None:  # the last stored minute may still have been forming: count from the one before
            out[iid] = int(pd.Timestamp(cov.last).value)
    return out


def build(profile, port: int, sink=None):
    from nautilus_trader.common import Environment, LoggerConfig, LogLevel
    from nautilus_trader.live import LiveNode
    from nautilus_trader.model import TraderId

    if profile.data_client is None:
        raise SystemExit(f"{profile.label} has no live market data client")
    pairs = instruments(profile)
    relay = HubRelay(HubRelayConfig(instrument_ids=tuple(sorted(pairs))))
    fanout = Fanout(profile.name, known=relay.known, want=relay.want, venue_up=relay.venue_up)
    if sink is None:
        from sleeve_fund.history import HistoryStore

        sink = store_sink(HistoryStore(), profile.name, pairs)  # the same dict the relay keeps up to date
    relay.attach(fanout, sink=sink, pairs=pairs, recent=profile.ohlc_history, last_close=last_closes(profile, pairs),
                 discover=lambda: instruments(profile))
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.hub", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="hold the venue connection and serve its trades, quotes and 1-minute bars")
    run.add_argument("--venue", default=None)
    run.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)
    assert_keyless()
    profile = venue_profile(args.venue)
    node, fanout, _ = build(profile, args.port)
    print(f"hub {profile.name}: serving on port {fanout.port}")
    try:
        node.run()
    finally:
        fanout.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
