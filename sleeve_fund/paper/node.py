"""Build and run one paper sleeve.

Data: Kraken Spot public market data (no credentials).
Execution: Nautilus sandbox, a local matching engine fed by that live data.
There is deliberately no code path here that adds a venue execution client.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import urllib.request
from pathlib import Path

from nautilus_trader.adapters.kraken import (
    KrakenDataClientConfig,
    KrakenDataClientFactory,
    KrakenEnvironment,
    KrakenProductType,
)
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

from sleeve_fund.instruments import ScheduleFeeModel
from sleeve_fund.paper.config import SleeveConfig, from_store, load_sleeve
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.paper.safety import assert_keyless
from sleeve_fund.strategies import REGISTRY

KRAKEN = "KRAKEN"


def _tag(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-")


# Kraken names some assets differently in its instrument data than in the pair (USD is ZUSD,
# BTC is XXBT). The sandbox account must hold cash in the instrument's own quote currency
# or every buy is rejected, so look the codes up from Kraken's public pair list.
ASSET_PAIRS_URL = "https://api.kraken.com/0/public/AssetPairs"
_ALIASES = {"XBT": "BTC", "XDG": "DOGE"}


def _norm(code: str) -> str:
    return _ALIASES.get(code, code)


def kraken_asset_codes(pair: str, fetch=None) -> tuple[str, str]:
    """(base, quote) as Kraken's instrument data names them, e.g. SUI/USD -> (SUI, ZUSD).

    Falls back to the pair's own codes if Kraken can't be reached or doesn't list it.
    """
    base, quote = pair.split("/")
    try:
        if fetch is None:
            with urllib.request.urlopen(ASSET_PAIRS_URL, timeout=15) as r:  # public endpoint, no key
                data = json.load(r)
        else:
            data = fetch()
        for info in data.get("result", {}).values():
            ws = info.get("wsname", "")
            if "/" in ws and tuple(_norm(x) for x in ws.split("/")) == (_norm(base), _norm(quote)):
                return info["base"], info["quote"]
    except Exception as exc:  # noqa: BLE001 - fall back, the sleeve still runs and logs a mark warning
        print(f"asset code lookup failed for {pair}: {exc!r}", file=sys.stderr)
    return base, quote


def build_node(sleeve: SleeveConfig, log_level: str = "INFO", runtime: SleeveRuntime | None = None,
               asset_fetch=None) -> LiveNode:
    assert_keyless()
    tag = _tag(sleeve.name)
    venue = Venue.from_str(KRAKEN)
    _, quote_code = kraken_asset_codes(sleeve.instrument, fetch=asset_fetch)
    node = (
        LiveNode.builder(f"PAPER-{tag}", TraderId.from_str(f"PAPER-{tag[:20]}"), Environment.SANDBOX)
        .with_logging(LoggerConfig(stdout_level=getattr(LogLevel, log_level)))
        .with_reconciliation(reconciliation=False)
        .add_data_client(
            None,
            KrakenDataClientFactory(),
            # Real Kraken prices (LIVE is Kraken's production feed; DEMO is futures-only).
            # No api_key/api_secret: public market data only.
            KrakenDataClientConfig(product_type=KrakenProductType.SPOT, environment=KrakenEnvironment.LIVE),
        )
        .add_simulated_exec_client(
            KRAKEN,
            SandboxExecutionClientFactory(),
            SandboxExecutionClientConfig(
                venue=venue,
                starting_balances=[Money(sleeve.starting_balance, Currency.from_str(quote_code))],
                account_id=AccountId.from_str(f"{KRAKEN}-PAPER-{tag[:20]}"),
                oms_type=OmsType.NETTING,
                account_type=AccountType.CASH,
                fee_model=ScheduleFeeModel(sleeve.fees),
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

        store = Store()
        sleeve = from_store(store.sleeve(args.db_sleeve))
        runtime = SleeveRuntime(store, sleeve.name)
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
