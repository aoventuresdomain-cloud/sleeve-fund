"""One ranked table from the per-definition results, judged by the Edge found definition's first two tests."""
import json
import sys
from pathlib import Path

rows = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("*.json"))]


def edge(r):
    re = r.get("random_entry") or {}
    weak = (r.get("in_market") or 0) > 0.6
    venue = r["net_return_ladder"].get("0.0550%", 0)
    double = r["net_return_ladder"].get("0.1100%", 0)
    pct = re.get("return_percentile")
    t1 = "weak" if weak else ("pass" if pct is not None and pct >= 95 else "fail")
    if weak and r["definition"].endswith("_long"):
        t1 += ", beats B&H" if venue > (r.get("buy_and_hold_same_window") or 0) else ", below B&H"
    return t1, "pass" if double > 0 else "fail", venue, pct


lines = ["| Strategy | Candle | Trades/yr | Avg gross bp | Break-even fee/side | Net at 0.055% | Net at 2x | "
         "Random-entry pct | In market | B&H | Max DD | Test 1 | Test 2 |", "|" + "---|" * 13]
rows.sort(key=lambda r: -(r["net_return_ladder"].get("0.0550%") or -9))
for r in rows:
    t1, t2, venue, pct = edge(r)
    be = r.get("break_even_fee_per_side")
    lines.append("| {} | {}m | {:.0f} | {} | {} | {:+.0%} | {:+.0%} | {} | {:.0%} | {:+.0%} | {} | {} | {} |".format(
        r["definition"], r["bar_minutes"], r.get("trades_per_year") or 0,
        f"{r['avg_gross_bp']:.0f}" if r.get("avg_gross_bp") is not None else "-",
        f"{be:.3%}" if be is not None else "none", venue, r["net_return_ladder"].get("0.1100%", 0),
        f"{pct:.0f}" if pct is not None else "-", r.get("in_market") or 0, r.get("buy_and_hold_same_window") or 0,
        f"{r['max_drawdown_venue']:.0%}" if r.get("max_drawdown_venue") is not None else "-", t1, t2))
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
