"""PAPER-TIERS-3: print a pair's maintenance tiers as the margin.Tier tuple venues.py keeps, from the paper venue's
own risk-limit reply (tests/data/venue_tables/risk-limit-<SYMBOL>.json, read by the venue-tables workflow). The kept
tuples are this script's output, never typed by hand; tests/test_paper_tiers_3.py re-reads the replies and asserts
every kept tuple equals them.

    python scripts/venue_tiers.py ETHUSDT SOLUSDT XRPUSDT
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "tests" / "data" / "venue_tables"


def _number(x: str) -> str:
    d = Decimal(x or "0")
    return f"{int(d):_}" if d == d.to_integral_value() else str(d.normalize())


def tuple_source(symbol: str) -> str:
    """The Python source of `<SYMBOL>_TIERS`: one Tier(ceiling, rate, deduction) per band, rising; the last ceiling
    unbounded (math.inf), its published value kept as maintenance_limits. A blank deduction is 0."""
    reply = json.loads((DATA / f"risk-limit-{symbol}.json").read_text())
    if reply.get("retCode") != 0 or reply["result"].get("nextPageCursor"):
        raise SystemExit(f"{symbol}: not a complete single-page reply")
    rows = sorted(reply["result"]["list"], key=lambda r: Decimal(r["riskLimitValue"]))
    lines = [f"{symbol}_TIERS = ("]
    for i, r in enumerate(rows):
        ceiling = "math.inf" if i == len(rows) - 1 else _number(r["riskLimitValue"])
        lines.append(f"    Tier({ceiling}, {_number(r['maintenanceMargin'])}, {_number(r['mmDeduction'])}),")
    return "\n".join([*lines, ")"])


if __name__ == "__main__":
    print("\n".join(tuple_source(s) for s in sys.argv[1:]))
