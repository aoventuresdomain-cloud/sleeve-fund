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
from functools import partial
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
from sleeve_fund.paper.safety import assert_keyless, assert_portfolio_gate
from sleeve_fund.strategies import REGISTRY, check_perp_sizing, check_perp_stop
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


def stored_minutes(venue: str, pair: str, store=None):
    """The hub client's recover (sleeve_fund.paper.hub_client.Decoder): the history store's 1-minute bars
    closing strictly between two times (UNIX ns), as (close ns, open, high, low, close, volume), oldest first.
    The hub writes every minute it relays there as it relays it, so minutes the client missed are there."""
    from sleeve_fund.history import HistoryStore

    def recover(_instrument_id: str, after_ns: int, before_ns: int) -> list[tuple]:
        # Read synchronously inside the hub client's read loop: only on a gap or at start, a few minutes from
        # one instrument, in a process running one strategy, so the stall is short and nothing else waits on it.
        hs = store or HistoryStore()
        df = hs.read(venue, pair, 1, start=pd.Timestamp(after_ns, tz="UTC"), end=pd.Timestamp(before_ns, tz="UTC"))
        df = df[df.index < pd.Timestamp(before_ns, tz="UTC")]
        return [(int(ts.value), r.open, r.high, r.low, r.close, r.volume) for ts, r in df.iterrows()]

    return recover


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
    whole = {"missing": 0, "degraded": False}  # the venue's own candles: none of their minutes absent (board 5a)
    newer = newer.assign(**{c: v for c, v in whole.items() if c in df.columns})
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
        from sleeve_fund.paper.hub_client import HubDataClientConfig, HubDataClientFactory, HubStatus, hub_bar_spec

        spec = hub_bar_spec(sleeve.bar_spec)
        bar_type = BarType.from_str(f"{sleeve.instrument_id}-{spec}")
        data_factory = HubDataClientFactory()
        hub_status = HubStatus()
        data_config = HubDataClientConfig(venue=profile.name, instrument_ids=(sleeve.instrument_id,), host=hub[0],
                                          port=hub[1], bar_spec=spec, status=hub_status,
                                          recover=stored_minutes(profile.name, sleeve.instrument),
                                          report=None if runtime is None else
                                          lambda level, kind, message: runtime.store.event(sleeve.name, level, kind,
                                                                                           message))
    else:
        bar_type = BarType.from_str(sleeve.bar_type)
        data_factory, data_config = profile.data_client()
    fee_model = ScheduleFeeModel(sleeve.fees)
    perp = markets.is_perp(sleeve.params)
    balances = _balances(sleeve, runtime, base_code, quote_code)
    node = (
        LiveNode.builder(f"PAPER-{tag}", TraderId.from_str(f"PAPER-{tag[:20]}"), Environment.SANDBOX)
        .with_logging(LoggerConfig(stdout_level=getattr(LogLevel, log_level)))
        .with_reconciliation(reconciliation=False)
        # The hub may take a minute to pick up an instrument it doesn't relay yet.
        .with_timeout_connection(120 if hub is not None else 60)
        .add_data_client("HUB" if hub is not None else None, data_factory, data_config)
        .add_simulated_exec_client(profile.name, SandboxExecutionClientFactory(),
                                   _sandbox(venue, balances, f"{profile.name}-PAPER-{tag[:20]}", perp, fee_model, hub))
        .build()
    )
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
    node.add_strategy(_strategy(sleeve, profile, REGISTRY[sleeve.strategy], sleeve.instrument_id, bar_type, fee_model,
                                runtime=runtime, recorder=recorder, history=history, hub=hub,
                                hub_status=hub_status if hub is not None else None))
    return node



# P2-8 (HoE Option A, 14 Oct): a two-leg Strategy's second leg trades on this clone of its venue, an account of its own
# beside the first leg's in the same node. Only the simulated matching engine ever sees the clone's name.
LEG_CLONE = 2


def build_two_leg_node(lead: SleeveConfig, follow: SleeveConfig, log_level: str = "INFO",
                       runtimes: tuple[SleeveRuntime | None, SleeveRuntime | None] = (None, None), asset_fetch=None,
                       history=None, hub: tuple[str, int] | None = None):
    """One paper node for a two-leg Strategy (P2-8): the first leg (`lead`, the parent row) on its venue's own
    simulated account and the second (`follow`, its "~2" row) on the venue's clone, each with its account type,
    fees and terms from the real venue's profile and its own journal (runtimes), both fed by the one hub
    connection: the second leg's instrument handed on again as the clone's, in the same step. Returns (node, pair).

    Refused: a node without the venue's hub, legs on two venues or bar sizes, a leg on a venue clone already."""
    from sleeve_fund.paper.hub_client import HubDataClientConfig, HubDataClientFactory, HubStatus, hub_bar_spec
    from sleeve_fund.research.portfolio import clone_venue
    from sleeve_fund.strategies.base import LongFlatConfig
    from sleeve_fund.strategies.pair import LegFollower, Pair

    assert_keyless()
    profile = venue_profile(lead.venue)
    if venue_profile(follow.venue).name != profile.name:
        raise ValueError(f"a two-leg Strategy's legs trade on one venue, not {lead.venue} and {follow.venue}")
    if lead.bar_spec != follow.bar_spec:
        raise ValueError(f"a two-leg Strategy's legs share one bar size, not {lead.bar_spec} and {follow.bar_spec}")
    if lead.params.get("leg") is None or follow.params.get("leg") is None:
        raise ValueError("both legs need their side, ratio and leverage (two_legs.leg_params)")
    hub = hub or hub_address(profile.name)
    if hub is None:
        raise ValueError(f"a two-leg Strategy is fed by its venue's market data hub; HUB_{profile.name} isn't set")
    clone = clone_venue(profile.venue, LEG_CLONE)
    follow_iid = f"{InstrumentId.from_str(follow.instrument_id).symbol}.{clone}"
    tag = _tag(lead.name)
    spec = hub_bar_spec(lead.bar_spec)
    lead_runtime, follow_runtime = runtimes
    sources = tuple(dict.fromkeys((lead.instrument_id, follow.instrument_id)))  # one feed when both legs share it
    minutes = {lead.instrument_id: stored_minutes(profile.name, lead.instrument),
               follow.instrument_id: stored_minutes(profile.name, follow.instrument)}
    hub_status = HubStatus()
    data_config = HubDataClientConfig(
        venue=profile.name, instrument_ids=sources, host=hub[0], port=hub[1], bar_spec=spec, status=hub_status,
        recover=lambda iid, after, before: minutes[iid](iid, after, before),
        report=None if lead_runtime is None else
        lambda level, kind, message: lead_runtime.store.event(lead.name, level, kind, message),
        aliases=((follow.instrument_id, follow_iid),))
    accounts = []
    for sleeve, runtime, venue, name in ((lead, lead_runtime, profile.venue, profile.name),
                                         (follow, follow_runtime, clone, str(clone))):
        base_code, quote_code = profile.asset_codes(sleeve.instrument, fetch=asset_fetch)
        fee_model = ScheduleFeeModel(sleeve.fees)  # the real venue's schedule for the leg's own market
        perp = markets.is_perp(sleeve.params)
        balances = _balances(sleeve, runtime, base_code, quote_code)
        accounts.append((name, fee_model, _sandbox(venue, balances, f"{name}-PAPER-{_tag(sleeve.name)[:20]}", perp,
                                                   fee_model, hub)))
    builder = (
        LiveNode.builder(f"PAPER-{tag}", TraderId.from_str(f"PAPER-{tag[:20]}"), Environment.SANDBOX)
        .with_logging(LoggerConfig(stdout_level=getattr(LogLevel, log_level)))
        .with_reconciliation(reconciliation=False)
        .with_timeout_connection(120)
        .add_data_client("HUB", HubDataClientFactory(), data_config)
    )
    for name, _, config in accounts:
        builder = builder.add_simulated_exec_client(name, SandboxExecutionClientFactory(), config)
    node = builder.build()
    legs = []
    for sleeve, runtime, iid, classes, status, (_, fee_model, _) in (
            (lead, lead_runtime, lead.instrument_id, REGISTRY[lead.strategy], hub_status, accounts[0]),
            (follow, follow_runtime, follow_iid, (LegFollower, LongFlatConfig),
             hub_status.view({follow_iid: follow.instrument_id}), accounts[1])):
        legs.append(_strategy(sleeve, profile, classes, iid, BarType.from_str(f"{iid}-{spec}"), fee_model,
                              runtime=runtime, recorder=None, history=history, hub=hub, hub_status=status))
    pair = Pair(legs[0], legs[1], lead_side=lead.params["leg"]["side"], follow_side=follow.params["leg"]["side"],
                ratio=follow.params["leg"]["ratio"], name=lead.name)
    for strategy in legs:
        node.add_strategy(strategy)
    if lead_runtime is not None:  # its state kept in the journal; a trade left open by a restart is recovered (M3)
        pair.journal(lead_runtime.store, lead.name)  # after add_strategy: it reads the clock the node gives the legs
    return node, pair

def _balances(sleeve: SleeveConfig, runtime: SleeveRuntime | None, base_code: str, quote_code: str) -> list[Money]:
    """The paper account's opening balances: the sleeve's, or rebuilt from the journal so a restart carries its
    position over."""
    balances = [Money(sleeve.starting_balance, Currency.from_str(quote_code))]
    if runtime is not None:  # rebuild the paper book from the journal so a restart carries positions over
        book = runtime.book
        if markets.is_perp(sleeve.params):
            # A margin account holds the quote currency only. The position is put back by a restore order
            # at start (LongFlatStrategy._send_restore); the account opens with the cash it had when the
            # position was opened, so after that restore its cash equals the journal's.
            balances = [Money(book["cash"] + book["qty"] * (book["entry_px"] or 0.0), Currency.from_str(quote_code))]
        else:
            balances = [Money(book["cash"], Currency.from_str(quote_code))]
            if book["qty"] > 0:
                balances.append(Money(book["qty"], Currency.from_str(base_code)))
    return balances


def _sandbox(venue, balances: list[Money], account_id: str, perp: bool, fee_model,
             hub) -> SandboxExecutionClientConfig:
    """The simulated venue's account: the only kind of execution client a paper node has."""
    return SandboxExecutionClientConfig(
        venue=venue,
        starting_balances=balances,
        account_id=AccountId.from_str(account_id),
        oms_type=OmsType.NETTING,
        # A perpetual trades on margin so it can go short; our own guards set its leverage.
        account_type=AccountType.MARGIN if perp else AccountType.CASH,
        default_leverage=markets.VENUE_LEVERAGE if perp else None,
        fee_model=fee_model,
        fill_model=fill_model(),
        # A hub-fed node fills from trades and quotes only: its bars are EXTERNAL, which the matching
        # engine would otherwise replay through the book a minute late, at their receive time. A node on
        # its own venue connection keeps the engine's default until it moves to the hub.
        bar_execution=hub is None,
    )


def _strategy(sleeve: SleeveConfig, profile, classes, instrument_id: str, bar_type: BarType, fee_model, *, runtime,
              recorder, history, hub, hub_status):
    """The sleeve's strategy, on instrument_id (its own, or a two-leg Strategy's second leg's on the venue clone),
    with what a paper strategy is given. Everything read by venue (history, spreads) is read at the real venue."""
    strategy_cls, config_cls = classes
    strategy = (
        strategy_cls(
            config_cls(
                instrument_id=InstrumentId.from_str(instrument_id),
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
        # A hub-fed node's history is the hub's own, in the store it writes: never the venue's candles, which
        # could differ from the minutes it relayed (one hub, same data; QA P1-C8).
        .attach_history(history or history_loader(profile.name, sleeve.instrument,
                                                  recent=profile.ohlc_history if hub is None else None))
    )
    if profile.ohlc_history is not None and hub is None:
        strategy.attach_gap_loader(gap_loader(sleeve.instrument, profile.ohlc_history))
    if runtime is not None:
        # The measured spread in force at each time, read where there are no quotes yet and by a replay after an
        # outage; refreshed hourly as new measurements land (SPREAD-PIT).
        runtime.spread_loader = partial(spreads.series, profile.name, sleeve.instrument, runtime.store, strict=True)
        strategy.spread_series = spreads.series(profile.name, sleeve.instrument, runtime.store)
    if hub is None:  # its first bar after a start is built from the stored minutes, as the hub client's is (R1)
        strategy.attach_minutes(stored_minutes(profile.name, sleeve.instrument))
    strategy.hub_fed = hub is not None
    strategy.hub_status = hub_status
    if hub is not None and hasattr(strategy, "minute_source"):
        # first_touch (R2): the minutes the hub client built each bar from, as a backtest judges the store's
        iid = instrument_id
        strategy.minute_source = lambda after, until: hub_status.minutes_of(iid, after, until)
    # Post-only orders fill in slices as the tape earns them, as a backtest fills them (review round 9, M9-3).
    strategy.simulated_venue = True
    strategy.fee_model = fee_model
    return strategy


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
    leg = None  # a two-leg Strategy's second leg (P2-8): (its config, its runtime), run in this process
    if args.db_sleeve:
        from sleeve_fund.store import Store

        from sleeve_fund.fees import resolve

        store = Store(portfolio_gate=True)
        from sleeve_fund.paper.single_lock import INSTANCE_LOCK, duplicate_refused, instance_lock

        # One process per strategy (SINGLE-LOCK, Advisor 13 Oct 18:10 UK): a second one would double its orders and
        # positions, so it stops here, before it connects to market data or the venue. Held for the process's life.
        lock = instance_lock(args.db_sleeve)
        if lock is None:
            return duplicate_refused(store, args.db_sleeve)
        store.event(args.db_sleeve, "info", INSTANCE_LOCK, f"this process (pid {os.getpid()}) runs the strategy: it "
                    "holds its single-instance lock")
        assert_portfolio_gate(store)
        row = store.sleeve(args.db_sleeve)
        if row.parent is not None:  # the supervisor never starts one: its first leg's process trades it
            raise SystemExit(f"{row.name} is the second leg of {row.parent}: it runs in {row.parent}'s process")
        from sleeve_fund.paper.queued import QueuedStore
        from sleeve_fund.store import LEG

        journal = QueuedStore(store)  # journal writes off the decision path; one writer keeps both legs' in order
        try:
            second = store.sleeve(row.name + LEG)
        except KeyError:
            second = None
        configs = []
        for r in (row,) if second is None else (row, second):
            quote = resolve(getattr(r, "venue", None), store)
            configs.append((from_store(r, fee_schedule=quote.fees), SleeveRuntime(journal, r.name)))
            perp = markets.terms(configs[-1][0].params, configs[-1][0].venue)
            store.event(r.name, "info", "fees", f"Charging {quote.text}" if perp is None or perp.fees is None else
                        f"Charging {perp.label}: {perp.fees.maker:.2%} maker, {perp.fees.taker:.2%} taker, on the "
                        "venue's live prices; funding every 8 hours from our own ledger")
        (sleeve, runtime), leg = configs[0], (configs[1] if second is not None else None)
    else:
        sleeve = load_sleeve(args.sleeve)
        check_perp_sizing(sleeve.strategy, sleeve.params)  # a strategy in the store is refused by the supervisor
        check_perp_stop(sleeve.strategy, sleeve.params, sleeve.risk_profile)  # QA P1-S6
    recorder = None
    if args.record:
        if runtime is None:
            ap.error("--record needs --db-sleeve: only a sleeve with its runtime receives quotes and trades")
        from sleeve_fund.paper.recorder import Recorder

        recorder = Recorder(args.record)
    if leg is not None:
        if recorder is not None:
            ap.error("--record isn't built for a two-leg Strategy yet")
        node, _ = build_two_leg_node(sleeve, leg[0], log_level=args.log_level, runtimes=(runtime, leg[1]))
    else:
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
        if runtime is not None:
            # What the strategy journaled as it stopped, an exit's row still being retried included (QueuedStore),
            # inside the 45 s the supervisor gives a stopping process.
            if hasattr(runtime.store, "settle"):
                runtime.store.settle(timeout=30)
            else:
                runtime.store.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
