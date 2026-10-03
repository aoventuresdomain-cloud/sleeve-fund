"""Strategy lab runner. Research only: reads public price history, places no orders.

    python scripts/lab.py data                      # fetch or extend minute history
    python scripts/lab.py classifier [--quick]      # test 1: does the day-type classifier work?

Results go to research/results/ as Markdown and JSON, and every evaluation is appended to the
idea ledger so later scores are judged against the number of tries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeve_fund.lab import classifier_test as ct  # noqa: E402
from sleeve_fund.lab import data  # noqa: E402
from sleeve_fund.research.ledger import IdeaLedger  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "minute"
RESULTS = ROOT / "research" / "results"
LEDGER = ROOT / "research" / "idea_ledger.jsonl"
BASKET = list(data.SYMBOLS)


def cmd_data(args) -> None:
    for inst in args.instruments:
        t = time.time()
        p = data.update(inst, DATA, log=lambda m: None)
        df = data.load(inst, DATA)
        print(f"{inst}: {data.span(df)} ({time.time() - t:.0f}s) -> {p.name}")


def _fmt(x, pct=False, d=2):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.0%}" if pct else f"{x:.{d}f}"


def cmd_classifier(args) -> None:
    bars = [5] if args.quick else [1, 5, 15]
    anchors = ["utc", "us_open"]
    decide = [60] if args.quick else [30, 60, 90]
    rows, spans = [], {}
    for inst in args.instruments:
        m = data.load(inst, DATA)
        spans[inst] = data.span(m)
        for bar in bars:
            b = data.resample(m, bar)
            for anchor in anchors:
                for d in decide:
                    if d % bar:
                        continue
                    so = ct.session_outcomes(b, anchor=anchor, decide_at=d)
                    for horizon in ("4h", "session"):
                        s = ct.summarise(so, horizon)
                        rows.append({"instrument": inst, "bar": bar, "anchor": anchor, "decide_at": d,
                                     "horizon": horizon, **s})
                    so["instrument"] = inst
                    so.to_csv(RESULTS / f"classifier_sessions_{inst.replace('/', '-')}_{bar}m_{anchor}_{d}.csv.gz",
                              index=False)
                    print(f"{inst} {bar}m {anchor} @{d}: {rows[-1]['verdict']}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (RESULTS / "classifier.json").write_text(json.dumps({"run": stamp, "spans": spans, "rows": rows}, default=str,
                                                        indent=1))
    lines = [f"# Classifier test ({stamp})", "",
             "Does the day-type classifier tell trending sessions from balanced ones, using only bars closed by the "
             "decision time? Continuation is the move after the decision in the trend's direction, in daily ATR; "
             "the 95% range is a bootstrap. Move is the absolute move. t compares trend and balance moves.", "",
             "## History used", "", "| Instrument | From | To | Years | Missing minutes |", "|---|---|---|---|---|"]
    for k, v in spans.items():
        lines.append(f"| {k} | {v.get('start')} | {v.get('end')} | {v.get('years')} | {v.get('missing_pct')}% |")
    lines += ["", "## Results", "",
              "| Instrument | Bar | Session | Decide | Horizon | Sessions | Trend share | Balance share | Continuation (95% range) "
              "| Hit | Move trend | Move balance | t | Verdict |", "|" + "---|" * 14]
    for r in rows:
        c = r.get("trend_continuation_atr") or (float("nan"),) * 3
        sh = r.get("share", {})
        lines.append(
            f"| {r['instrument']} | {r['bar']}m | {r['anchor']} | {r['decide_at']}m | {r['horizon']} | {r['sessions']} "
            f"| {_fmt(sh.get('trend_up', 0) + sh.get('trend_down', 0), True)} | {_fmt(sh.get('balance'), True)} "
            f"| {_fmt(c[0])} ({_fmt(c[1])} to {_fmt(c[2])}) | {_fmt(r.get('trend_continuation_hit'), True)} "
            f"| {_fmt(r.get('move_trend_atr'))} | {_fmt(r.get('move_balance_atr'))} | {_fmt(r.get('move_t_trend_vs_balance'), d=1)} "
            f"| {r['verdict']} |")
    (RESULTS / "classifier.md").write_text("\n".join(lines) + "\n")
    ledger = IdeaLedger(LEDGER)
    for r in rows:
        c = r.get("trend_continuation_atr") or (0,)
        ledger.record(idea="day_type_classifier", family="classifier",
                      params={"bar": r["bar"], "anchor": r["anchor"], "decide_at": r["decide_at"], "horizon": r["horizon"]},
                      dataset=f"binance_1m:{r['instrument']}", stage="diagnostic",
                      sharpe=float(c[0]) if c[0] == c[0] else 0.0)
    print(f"wrote {RESULTS / 'classifier.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["data", "classifier"])
    ap.add_argument("--instruments", nargs="*", default=BASKET)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"data": cmd_data, "classifier": cmd_classifier}[args.command](args)


if __name__ == "__main__":
    main()
