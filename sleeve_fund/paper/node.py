"""Build and run one paper sleeve.

Data: the sleeve's venue's public market data (no credentials), from its venue profile.
Execution: Nautilus sandbox, a local matching engine fed by that live data.
There is deliberately no code path here that adds a venue execution client.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import threading
from pathlib import Path

import pandas as pd

from nautilus_trader.adapters.sandbox import SandboxExecutionClientConfig, SandboxExecutionClientFactory
from nautilus_trader.common import Environment, LoggerConfig, LogLevel
from nautilus_trader.live import LiveNode
from nautilus_trader.model import (
    AccountId,
    AccountType,
    BarType,
    Currency,
    InstrumentId,
    Money,
    OmsType,
    StrategyId,
    TraderId,
)

from sleeve_fund import markets, spreads
from sleeve_fund.instruments import ScheduleFeeModel, fill_model
from sleeve_fund.paper.config import SleeveConfig, from_store, load_sleeve
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.paper.safety import assert_keyless
from sleeve_fund.strategies import REGISTRY, check_perp_sizing
from sleeve_fund.venues import venue as venue_profile


def _tag(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-")


def strategy_id(cls_name: str, name: str) -> StrategyId:
    """The strategy's id in its node. Client order ids carry only the last hyphen-separated part of it (and of the
    trader id), so that part is a hash of the full name: two strategies whose names end alike (ping-pong-test and
    rsi-bands-test) got the same order ids, and the second to send one in a second lost its journal row."""
    digest = hashlib.sha1(name.encode()).hexdigest()[:8].upper()
    return StrategyId.from_str(f"{cls_name}-{_tag(name)[:20]}-{digest}")


# Warm-up bars from a store that stopped updating would leave a hole before the first live bar.
HISTORY_MAX_LAG = pd.Timedelta(hours=6)


def history_loader(venue: str, pair: str, store=None, recent=None):
    """Loads a sleeve's warm-up bars from the venue history store (the latest complete bars at the
    sleeve's interval), for bars the venue can't serve because they are built from live trades.

    recent: (pair, minutes) -> the venue's latest candles by open time, the newest still forming (a
    venue profile's ohlc_history). The store's loader can be hours behind; these fill the bars between
    its end and now, so the indicators run right up to the first live bar."""
    from sleeve_fund.data import bar_minutes, to_bars
    from sleeve_fund.history import HistoryStore

    def load(instrument, bar_type, limit: int):
        hs = store or HistoryStore()
        minutes = bar_minutes(bar_type)
        cov = hs.coverage(venue, pair)
        why = None
        if cov is None:
            why = f"no stored history for {pair}"
        elif (lag := pd.Timestamp.now(tz="UTC") - cov.last) > HISTORY_MAX_LAG:
            why = f"the stored history for {pair} is {lag.total_seconds() / 3600:.0f} hours old"
        if why is not None:
            # Without usable stored bars, warm up on the venue's own recent candles (one request serves 720 to
            # 1,500 candles), so after a restart the model's indicators are ready and settled on its first live bar rather
            # than waiting out their look-back with no exits (PM, 5 Oct 2026: a long that should have exited).
            df = _recent_closed(recent, pair, minutes, why) if recent is not None else None
            if df is None or not len(df):
                raise LookupError(why)
            load.source = "the venue's recent candles"
            return to_bars(df.tail(limit), instrument, bar_type)
        df = hs.read(venue, pair, minutes, start=cov.last - pd.Timedelta(minutes=minutes * (limit + 2)))
        if recent is not None and len(df):
            df = _top_up(df, recent, pair, minutes)
        load.source = "the history store"
        return to_bars(df.tail(limit), instrument, bar_type)

    load.source = "the history store"
    return load


def _recent_closed(recent, pair: str, minutes: int, why: str) -> pd.DataFrame:
    """The venue's complete recent candles, stamped at their close as the store's bars are. Raises LookupError
    with both reasons when the venue can't serve them either."""
    try:
        r = recent(pair, minutes)
    except Exception as exc:  # noqa: BLE001 - an unreachable venue or interval: nothing to warm up on
        raise LookupError(f"{why}, and the venue's recent candles failed: {exc}") from exc
    r = r.iloc[:-1]  # the newest candle is still forming
    r = r.set_axis(r.index + pd.Timedelta(minutes=minutes))
    return r[["open", "high", "low", "close", "volume"]]


def gap_loader(pair: str, recent):
    """The venue's own closed candles stamped from since_ns to until_ns (at their close, as live bars are), for
    candles built while no trades reached the process (LongFlatStrategy._fill_gap). recent: the venue profile's
    ohlc_history; the newest candle it returns is still forming and is left out."""
    from sleeve_fund.data import bar_minutes, to_bars

    def load(instrument, bar_type, since_ns: int, until_ns: int):
        minutes = bar_minutes(bar_type)
        r = recent(pair, minutes).iloc[:-1]
        r = r.set_axis(r.index + pd.Timedelta(minutes=minutes))
        stamps = r.index.as_unit("ns").asi8
        r = r[(stamps >= since_ns) & (stamps <= until_ns)]
        return to_bars(r[["open", "high", "low", "close", "volume"]], instrument, bar_type)

    return load


def _top_up(df: pd.DataFrame, recent, pair: str, minutes: int) -> pd.DataFrame:
    """The stored bars plus the venue's complete candles after them, stamped at their close as the
    store's are. A venue that can't serve this interval leaves the stored bars as they are; the
    warm-up event then says how many bars are missing."""
    try:
        r = recent(pair, minutes)
    except Exception:  # noqa: BLE001 - an unreachable venue or interval: warm up on what is stored
        return df
    r = r.iloc[:-1]  # the newest candle is still forming
    r = r.set_axis(r.index + pd.Timedelta(minutes=minutes))
    newer = r[r.index > df.index[-1]]
    return pd.concat([df, newer[list(df.columns)]]) if len(newer) else df


def hub_address(venue: str, environ=None) -> tuple[str, int] | None:
    """Where the venue's market data hub (v2 P1-1) serves, from HUB_<VENUE>=host:port (e.g.
    hub-<venue>:7700 in the stack), or None: then the node keeps a venue connection of its own, as before the hub."""
    raw = (os.environ if environ is None else environ).get(f"HUB_{venue.upper()}", "").strip()
    if not raw:
        return None
    host, _, port = raw.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"HUB_{venue.upper()} must be host:port, not {raw!r}")
    return host, int(port)


def build_node(sleeve: SleeveConfig, log_level: str = "INFO", runtime: SleeveRuntime | None = None,
               asset_fetch=None, recorder=None, history=None, hub: tuple[str, int] | None = None) -> LiveNode:
    """hub: (host, port) of the venue's market data hub; by default HUB_<VENUE> (hub_address)."""
    assert_keyless()
    tag = _tag(sleeve.name)
    profile = venue_profile(sleeve.venue)
    hub = hub or hub_address(profile.name)
    if profile.data_client is None and hub is None:
        raise ValueError(f"{profile.label} has no live market data client yet, so it can't run paper sleeves")
    venue = profile.venue
    base_code, quote_code = profile.asset_codes(sleeve.instrument, fetch=asset_fetch)
    if hub is not None:
        # The hub's trades, quotes and 1-minute bars, the same for every strategy on the instrument; longer bars
        # are built by the hub client from those minutes, as a backtest builds them from the store's.
        from sleeve_fund.paper.hub_client import HubDataClientConfig, HubDataClientFactory, hub_bar_spec

        spec = hub_bar_spec(sleeve.bar_spec)
        bar_type = BarType.from_str(f"{sleeve.instrument_id}-{spec}")
        data_factory = HubDataClientFactory()
        data_config = HubDataClientConfig(venue=profile.name, instrument_ids=(sleeve.instrument_id,), host=hub[0],
                                          port=hub[1], bar_spec=spec)
    else:
        bar_type = BarType.from_str(sleeve.bar_type)
        data_factory, data_config = profile.data_client()
    fee_model = ScheduleFeeModel(sleeve.fees)
    perp = markets.is_perp(sleeve.params)
    balances = [Money(sleeve.starting_balance, Currency.from_str(quote_code))]
    if runtime is not None:  # rebuild the paper book from the journal so a restart carries positions over
        book = runtime.book
        if perp:
            # A margin account holds the quote currency only. The position is put back by a restore order
            # at start (LongFlatStrategy._send_restore); the account opens with the cash it had when the
            # position was opened, so after that restore its cash equals the journal's.
            balances = [Money(book["cash"] + book["qty"] * (book["entry_px"] or 0.0), Currency.from_str(quote_code))]
        else:
            balances = [Money(book["cash"], Currency.from_str(quote_code))]
            if book["qty"] > 0:
                balances.append(Money(book["qty"], Currency.from_str(base_code)))
    node = (
        LiveNode.builder(f"PAPER-{tag}", TraderId.from_str(f"PAPER-{tag[:20]}"), Environment.SANDBOX)
        .with_logging(LoggerConfig(stdout_level=getattr(LogLevel, log_level)))
        .with_reconciliation(reconciliation=False)
        # The hub may take a minute to pick up an instrument it doesn't relay yet.
        .with_timeout_connection(120 if hub is not None else 60)
        .add_data_client("HUB" if hub is not None else None, data_factory, data_config)
        .add_simulated_exec_client(
            profile.name,
            SandboxExecutionClientFactory(),
            SandboxExecutionClientConfig(
                venue=venue,
                starting_balances=balances,
                account_id=AccountId.from_str(f"{profile.name}-PAPER-{tag[:20]}"),
                oms_type=OmsType.NETTING,
                # A perpetual trades on margin so it can go short; our own guards set its leverage.
                account_type=AccountType.MARGIN if perp else AccountType.CASH,
                default_leverage=markets.VENUE_LEVERAGE if perp else None,
                fee_model=fee_model,
                fill_model=fill_model(),
                # Fills come from trades and quotes only. The hub's 1-minute bars are EXTERNAL, which the
                # matching engine would otherwise replay through the book a minute late, at their receive time.
                bar_execution=False,
            ),
        )
        .build()
    )
    strategy_cls, config_cls = REGISTRY[sleeve.strategy]
    if recorder is not None:  # what a replay needs to rebuild this run exactly (sleeve_fund.research.replay)
        recorder.meta = {
            "balances": [str(b) for b in balances],
            "sleeve": {"name": sleeve.name, "strategy": sleeve.strategy, "instrument": sleeve.instrument,
                       "bar_spec": sleeve.bar_spec, "starting_balance": sleeve.starting_balance,
                       "risk_profile": sleeve.risk_profile, "params": sleeve.params,
                       "max_notional": sleeve.max_notional, "maker_fee": str(sleeve.fees.maker),
                       "taker_fee": str(sleeve.fees.taker),
                       "tick_seconds": runtime.tick_seconds if runtime is not None else None},
        }
    strategy = (
        strategy_cls(
            config_cls(
                instrument_id=InstrumentId.from_str(sleeve.instrument_id),
                bar_type=bar_type,
                max_notional=sleeve.max_notional,
                assumed_taker_fee=float(sleeve.fees.taker),
                # Sizing to a loss at the stop uses live quotes; this covers the moments before the first.
                assumed_half_spread=spreads.resolve(profile.name, sleeve.instrument,
                                                    runtime.store if runtime is not None else None).half_spread,
                warmup_bars=sleeve.warmup_bars,
                strategy_id=strategy_id(strategy_cls.__name__, sleeve.name),
                **sleeve.params,
            )
        ).attach_runtime(runtime).attach_recorder(recorder)
        .attach_history(history or history_loader(profile.name, sleeve.instrument, recent=profile.ohlc_history))
    )
    if profile.ohlc_history is not None:
        strategy.attach_gap_loader(gap_loader(sleeve.instrument, profile.ohlc_history))
    strategy.hub_fed = hub is not None
    # Post-only orders fill in slices as the tape earns them, as a backtest fills them (review round 9, M9-3).
    strategy.simulated_venue = True
    strategy.fee_model = fee_model
    node.add_strategy(strategy)
    return node


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.paper", description="Run one paper sleeve")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("sleeve", nargs="?", type=Path, help="sleeve TOML, e.g. configs/examples/btc_trend_smoke.toml")
    src.add_argument("--db-sleeve", help="run the named sleeve from the database (journal, controls, risk guard)")
    ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = run until Ctrl+C)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--record", type=Path, help="save every quote and trade received to this .jsonl.gz, for replay")
    args = ap.parse_args(argv)

    runtime = None
    if args.db_sleeve:
        from sleeve_fund.store import Store

        from sleeve_fund.fees import resolve

        store = Store()
        row = store.sleeve(args.db_sleeve)
        quote = resolve(getattr(row, "venue", None), store)
        sleeve = from_store(row, fee_schedule=quote.fees)
        runtime = SleeveRuntime(store, sleeve.name)
        perp = markets.terms(sleeve.params, sleeve.venue)
        store.event(sleeve.name, "info", "fees", f"Charging {quote.text}" if perp is None or perp.fees is None else
                    f"Charging {perp.label}: {perp.fees.maker:.2%} maker, {perp.fees.taker:.2%} taker, on the venue's "
                    "live prices; funding every 8 hours from our own ledger")
    else:
        sleeve = load_sleeve(args.sleeve)
        check_perp_sizing(sleeve.strategy, sleeve.params)  # a strategy in the store is refused by the supervisor
    recorder = None
    if args.record:
        if runtime is None:
            ap.error("--record needs --db-sleeve: only a sleeve with its runtime receives quotes and trades")
        from sleeve_fund.paper.recorder import Recorder

        recorder = Recorder(args.record)
    node = build_node(sleeve, log_level=args.log_level, runtime=runtime, recorder=recorder)
    if args.minutes > 0:
        handle = node.handle()
        timer = threading.Timer(args.minutes * 60, handle.stop)
        timer.daemon = True
        timer.start()
    try:
        node.run()
    finally:
        if recorder is not None:
            recorder.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
