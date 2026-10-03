"""Build and run one paper sleeve.

Data: the sleeve's venue's public market data (no credentials), from its venue profile.
Execution: Nautilus sandbox, a local matching engine fed by that live data.
There is deliberately no code path here that adds a venue execution client.
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
from pathlib import Path

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
    Venue,
)

from sleeve_fund.instruments import ScheduleFeeModel, fill_model
from sleeve_fund.paper.config import SleeveConfig, from_store, load_sleeve
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.paper.safety import assert_keyless
from sleeve_fund.strategies import REGISTRY
from sleeve_fund.venues import venue as venue_profile


def _tag(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-")


def build_node(sleeve: SleeveConfig, log_level: str = "INFO", runtime: SleeveRuntime | None = None,
               asset_fetch=None) -> LiveNode:
    assert_keyless()
    tag = _tag(sleeve.name)
    profile = venue_profile(sleeve.venue)
    if profile.data_client is None:
        raise ValueError(f"{profile.label} has no live market data client yet, so it can't run paper sleeves")
    venue = profile.venue
    base_code, quote_code = profile.asset_codes(sleeve.instrument, fetch=asset_fetch)
    data_factory, data_config = profile.data_client()
    balances = [Money(sleeve.starting_balance, Currency.from_str(quote_code))]
    if runtime is not None:  # rebuild the paper book from the journal so a restart carries positions over
        balances = [Money(runtime.book["cash"], Currency.from_str(quote_code))]
        if runtime.book["qty"] > 0:
            balances.append(Money(runtime.book["qty"], Currency.from_str(base_code)))
    node = (
        LiveNode.builder(f"PAPER-{tag}", TraderId.from_str(f"PAPER-{tag[:20]}"), Environment.SANDBOX)
        .with_logging(LoggerConfig(stdout_level=getattr(LogLevel, log_level)))
        .with_reconciliation(reconciliation=False)
        .add_data_client(None, data_factory, data_config)
        .add_simulated_exec_client(
            profile.name,
            SandboxExecutionClientFactory(),
            SandboxExecutionClientConfig(
                venue=venue,
                starting_balances=balances,
                account_id=AccountId.from_str(f"{profile.name}-PAPER-{tag[:20]}"),
                oms_type=OmsType.NETTING,
                account_type=AccountType.CASH,
                fee_model=ScheduleFeeModel(sleeve.fees),
                fill_model=fill_model(),
            ),
        )
        .build()
    )
    strategy_cls, config_cls = REGISTRY[sleeve.strategy]
    node.add_strategy(
        strategy_cls(
            config_cls(
                instrument_id=InstrumentId.from_str(sleeve.instrument_id),
                bar_type=BarType.from_str(sleeve.bar_type),
                max_notional=sleeve.max_notional,
                assumed_taker_fee=float(sleeve.fees.taker),
                warmup_bars=sleeve.warmup_bars,
                strategy_id=StrategyId.from_str(f"{strategy_cls.__name__}-{tag[:20]}"),
                **sleeve.params,
            )
        ).attach_runtime(runtime)
    )
    return node


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.paper", description="Run one paper sleeve")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("sleeve", nargs="?", type=Path, help="sleeve TOML, e.g. configs/sleeves/btc_trend_smoke.toml")
    src.add_argument("--db-sleeve", help="run the named sleeve from the database (journal, controls, risk guard)")
    ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = run until Ctrl+C)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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
        store.event(sleeve.name, "info", "fees", f"Charging {quote.text}")
    else:
        sleeve = load_sleeve(args.sleeve)
    node = build_node(sleeve, log_level=args.log_level, runtime=runtime)
    if args.minutes > 0:
        handle = node.handle()
        timer = threading.Timer(args.minutes * 60, handle.stop)
        timer.daemon = True
        timer.start()
    node.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
