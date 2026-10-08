"""Hub bar parity report (v2 P1-1): the history store's 1-minute bars against the venue's own candles.

    python scripts/hub_parity.py --venue binance --hours 24
    python scripts/hub_parity.py --venue binance --pair BTC/USDT --late late.json --out /path/report.md

Without --pair, every instrument the store holds for the venue. --late is the hub's late-trade counts,
{"BTC/USDT": [late, total], ...}; by default the file the hub keeps beside the store. Reads the store and the venue's public candles; writes only the report.
--sample N also prints N of the hub bars the CANON backfill replaced, per instrument, with both values and the bar
stored now (a read-only spot-check of the provenance log); exit 1 if any no longer holds the venue's bar.
Exit 0 when every instrument matches, 1 when any differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sleeve_fund.history import HistoryStore
from sleeve_fund.parity import canon_sample, late_counts, markdown, markdown_sample, run, window
from sleeve_fund.venues import venue


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--venue", required=True)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--pair", action="append", help="repeat for several; default: all the store holds")
    ap.add_argument("--tolerance", type=float, default=0.0, help="OHLC difference allowed, in price units")
    ap.add_argument("--late", type=Path, help="the hub's late-trade counts (JSON; default: the hub's own file)")
    ap.add_argument("--history", type=Path, help="the store's root (default: HISTORY_DIR)")
    ap.add_argument("--out", type=Path, help="write the report here as well as printing it")
    ap.add_argument("--sample", type=int, default=0, help="also print this many CANON replacements per instrument")
    args = ap.parse_args(argv)
    profile = venue(args.venue)
    store = HistoryStore(args.history)
    pairs = args.pair or [p for v, p in store.series() if v == profile.name]
    if not pairs:
        print(f"no stored history for {profile.label}", file=sys.stderr)
        return 2
    late = late_counts(store, profile.name, args.late)
    start, end = window(args.hours)
    results = run(store, profile, pairs, start, end, args.tolerance, late)
    report = markdown(profile.label, results)
    print(report)
    kept = True
    for pair in pairs if args.sample > 0 else []:
        rows = canon_sample(store, profile.name, pair, args.sample)
        total = len({r["minute"] for r in store.provenance(profile.name, pair)
                     if r["kind"] == "replaced" and r["source"] == "canon"})
        sample = markdown_sample(pair, rows, total)
        print(sample)
        report += "\n" + sample
        kept = kept and all(r["now_is_offered"] for r in rows)
    if args.out:
        args.out.write_text(report)
    return 0 if all(r.ok for r in results) and kept else 1


if __name__ == "__main__":
    sys.exit(main())
