"""Command line entry point.

    python -m sleeve_fund study trend_filter --store --base BTC --quote USD
    python -m sleeve_fund study trend_filter --store --minutes 60 --train-days 365 --test-days 90 --holdout-days 90
    (the Research page runs the same --store study from the browser)
    python -m sleeve_fund study trend_filter --data data/XBTUSD_1440.csv --base BTC --quote USD
    python -m sleeve_fund study trend_filter --synthetic
    python -m sleeve_fund counter
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sleeve_fund.data import load_ohlcvt_csv, synthetic_ohlcv
from sleeve_fund.instruments import history_price_decimals
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.run import LEDGER, STUDY_MINUTES, TEARSHEETS, StudyRequest, run_store_study, spec_of
from sleeve_fund.research.study import run_study
from sleeve_fund.research.tearsheet import render
from sleeve_fund.venues import venue


def _spec(name: str):
    try:
        return spec_of(name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _journal():
    """On the server: the journal, for the connected account's fees and the measured spread."""
    if not os.environ.get("DATABASE_URL"):
        return None
    from sleeve_fund.store import Store

    return Store()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sleeve_fund")
    ap.add_argument("--ledger", default=str(LEDGER), help="idea counter (default: $IDEA_LEDGER, else the repository's)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("study", help="run a full G1 study and write a tear sheet")
    st.add_argument("strategy")
    src = st.add_mutually_exclusive_group(required=True)
    src.add_argument("--store", action="store_true", help="bars from the venue history store (the server's)")
    src.add_argument("--data", help="daily OHLCVT CSV (time, open, high, low, close, volume, trades)")
    src.add_argument("--synthetic", action="store_true", help="random-walk data, pipeline check only")
    st.add_argument("--base", default="BTC")
    st.add_argument("--quote", default="USD")
    st.add_argument("--venue", default=None, help="venue profile for fees (default: the default venue)")
    st.add_argument("--minutes", type=int, default=1440, choices=STUDY_MINUTES,
                    help="bar length with --store: 1440 daily, 240, 60 or 15; judged on daily returns either way")
    st.add_argument("--holdout-days", type=int, default=365)
    st.add_argument("--train-days", type=int, default=3 * 365, help="walk-forward training window")
    st.add_argument("--test-days", type=int, default=365, help="walk-forward test window")
    st.add_argument("--use-holdout", action="store_true", help="open the holdout (logged; do this once)")
    st.add_argument("--stop-loss", type=float, help="exit if price falls this fraction below entry, e.g. 0.08")
    st.add_argument("--take-profit", type=float, help="exit if price rises this fraction above entry, e.g. 0.2")
    st.add_argument("--risk-per-trade", type=float, help="size so a stop-out loses this fraction of equity")
    st.add_argument("--stop-atr", type=float, help="instead of --stop-loss: a stop this many average true ranges below entry")
    st.add_argument("--atr-bars", type=int, help="bars the average true range is taken over (default 14)")
    st.add_argument("--stop-swing-bars", type=int, help="instead of --stop-loss: a stop at the lowest low of this many bars")
    st.add_argument("--take-profit-r", type=float, help="instead of --take-profit: a target this many stop distances up")
    st.add_argument("--risk-profile", default="balanced",
                    help="trade under this paper risk profile (cap, drawdown halt, daily-loss pause); 'none' for "
                         "uncapped and unguarded")
    st.add_argument("--out", help="tear sheet path (default: $TEARSHEET_DIR, else research/tearsheets, "
                                  "<strategy>_<dataset>.md)")

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
    profile_name = None if args.risk_profile == "none" else args.risk_profile
    if args.store:
        req = StudyRequest(strategy=args.strategy, pair=f"{args.base}/{args.quote}", venue=args.venue,
                           minutes=args.minutes, risk_profile=profile_name, train_days=args.train_days,
                           test_days=args.test_days, holdout_days=args.holdout_days, use_holdout=args.use_holdout,
                           stop_loss=args.stop_loss, take_profit=args.take_profit, risk_per_trade=args.risk_per_trade,
                           stop_atr=args.stop_atr, stop_swing_bars=args.stop_swing_bars, atr_bars=args.atr_bars,
                           take_profit_r=args.take_profit_r)
        try:
            out = run_store_study(req, store=_journal(), ledger_path=Path(args.ledger),
                                  out_dir=Path(args.out).parent if args.out else None)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.out and out != Path(args.out):
            out = out.rename(args.out)
        print(f"tear sheet: {out}")
        return 0
    if args.synthetic:
        prices, dataset = synthetic_ohlcv(), "synthetic"
    else:
        prices = load_ohlcvt_csv(args.data)
        dataset = Path(args.data).stem
    from sleeve_fund.fees import resolve

    store = _journal()
    quote = resolve(args.venue, store)
    print(f"fees: {quote.text}")
    # A real data file's study counts in the trials register like any other run, is judged on its idea family's
    # N and passes the holdout lock (QA P1-T3). Synthetic runs prove plumbing and stay out of it.
    from sleeve_fund.research.trials import TrialsRegister

    register = TrialsRegister(store) if store is not None and not args.synthetic else None
    instrument = venue(args.venue).instrument(args.base, args.quote, fees=quote.fees,
                                              price_precision=history_price_decimals(prices["close"]))
    result = run_study(
        spec,
        prices,
        instrument,
        dataset=dataset,
        ledger=ledger,
        synthetic=args.synthetic,
        holdout_days=args.holdout_days,
        train_days=args.train_days,
        test_days=args.test_days,
        use_holdout=args.use_holdout,
        exits={"stop_loss": args.stop_loss, "take_profit": args.take_profit, "risk_per_trade": args.risk_per_trade,
               "stop_atr": args.stop_atr, "stop_swing_bars": args.stop_swing_bars, "take_profit_r": args.take_profit_r,
               **({"atr_bars": args.atr_bars} if args.stop_atr and args.atr_bars else {})},
        risk_profile=profile_name,
        register=register,
    )
    out = Path(args.out) if args.out else TEARSHEETS / f"{spec.name}_{dataset}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(result, ledger, register), encoding="utf-8")
    print(f"tear sheet: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
