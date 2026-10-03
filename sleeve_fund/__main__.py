"""Command line entry point.

    python -m sleeve_fund study trend_filter --store --base BTC --quote USD
    python -m sleeve_fund study trend_filter --data data/XBTUSD_1440.csv --base BTC --quote USD
    python -m sleeve_fund study trend_filter --synthetic
    python -m sleeve_fund counter
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from sleeve_fund.data import load_kraken_ohlcvt, synthetic_ohlcv
from sleeve_fund.risk import profile as risk_profile
from sleeve_fund.venues import venue
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.study import run_study
from sleeve_fund.research.tearsheet import render

ROOT = Path(__file__).resolve().parent.parent


def _spec(name: str):
    try:
        return importlib.import_module(f"sleeve_fund.strategies.{name}").SPEC
    except (ModuleNotFoundError, AttributeError) as exc:
        raise SystemExit(f"no strategy module with a SPEC called {name!r}") from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sleeve_fund")
    ap.add_argument("--ledger", default=str(ROOT / "research" / "idea_ledger.jsonl"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("study", help="run a full G1 study and write a tear sheet")
    st.add_argument("strategy")
    src = st.add_mutually_exclusive_group(required=True)
    src.add_argument("--store", action="store_true", help="daily bars from the venue history store (the server's)")
    src.add_argument("--data", help="Kraken OHLCVT daily CSV")
    src.add_argument("--synthetic", action="store_true", help="random-walk data, pipeline check only")
    st.add_argument("--base", default="BTC")
    st.add_argument("--quote", default="USD")
    st.add_argument("--venue", default=None, help="venue profile for fees (default: the default venue)")
    st.add_argument("--holdout-days", type=int, default=365)
    st.add_argument("--use-holdout", action="store_true", help="open the holdout (logged; do this once)")
    st.add_argument("--stop-loss", type=float, help="exit if price falls this fraction below entry, e.g. 0.08")
    st.add_argument("--take-profit", type=float, help="exit if price rises this fraction above entry, e.g. 0.2")
    st.add_argument("--risk-per-trade", type=float, help="size so a stop-out loses this fraction of equity")
    st.add_argument("--risk-profile", default="balanced",
                    help="cap positions (and the benchmark) as this paper risk profile does; 'none' for uncapped")
    st.add_argument("--out", help="tear sheet path (default research/tearsheets/<strategy>_<dataset>.md)")

    sub.add_parser("counter", help="print the idea counter")
    args = ap.parse_args(argv)

    if args.synthetic:
        # Synthetic runs get their own ledger so they never inflate the real idea counter.
        args.ledger = str(Path(args.ledger).with_name("idea_ledger.synthetic.jsonl"))
    ledger = IdeaLedger(args.ledger)

    if args.cmd == "counter":
        print(ledger.counts())
        return 0

    spec = _spec(args.strategy)
    if args.synthetic:
        prices, dataset = synthetic_ohlcv(), "synthetic"
    elif args.store:
        from sleeve_fund.history import HistoryStore

        name = venue(args.venue).name
        prices = HistoryStore().read(name, f"{args.base}/{args.quote}", 1440)
        dataset = f"{name.lower()}-{args.base}{args.quote}-store".lower()
    else:
        prices = load_kraken_ohlcvt(args.data)
        dataset = Path(args.data).stem
    import os

    from sleeve_fund.fees import resolve

    store = None
    if os.environ.get("DATABASE_URL"):  # on the server: use the connected account's fetched rates
        from sleeve_fund.store import Store

        store = Store()
    quote = resolve(args.venue, store)
    print(f"fees: {quote.text}")
    instrument = venue(args.venue).instrument(args.base, args.quote, fees=quote.fees)
    result = run_study(
        spec,
        prices,
        instrument,
        dataset=dataset,
        ledger=ledger,
        synthetic=args.synthetic,
        holdout_days=args.holdout_days,
        use_holdout=args.use_holdout,
        exits={"stop_loss": args.stop_loss, "take_profit": args.take_profit, "risk_per_trade": args.risk_per_trade},
        position_cap=None if args.risk_profile == "none" else risk_profile(args.risk_profile).max_position_pct,
    )
    out = Path(args.out) if args.out else ROOT / "research" / "tearsheets" / f"{spec.name}_{dataset}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(result, ledger), encoding="utf-8")
    print(f"tear sheet: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
