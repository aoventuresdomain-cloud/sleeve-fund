"""R-20 2a (HoE, 21 Oct): one page over a set of tear sheets, for the Researcher's report and the PM.

A row per strategy with its sheet's own printed figures (read with the dashboard's parsers, so the page and the
Results tab say the same). The ranking is the Advisor's (21 Oct):
- sections, never ranked against each other: "Pooled (one try)" first, then one per single asset ("BTC only", ...),
  since pooling lifts Sharpe through diversification;
- within a section, G1 PASS, then NOT JUDGED (muted, with its reason, so it never reads as nearly passed), then FAIL;
- within PASS, a clean holdout above a spent or leaked one, whatever the Sharpe;
- then by deflated Sharpe (it corrects the luck of the best of many tries), out-of-sample Sharpe breaking ties; a row
  without a deflated Sharpe, or whose tries the trials register doesn't track (so its deflation has no honest count),
  last in its group, its G1 cell saying "tries not tracked".
Every row shows its idea family's tries against the budget and the 2x-fee result.

Under the tables, the correlation of the strategies' daily returns (each sheet's .returns.csv, run.write_returns) over
the days they all share. A sheet without its returns file, or a returns file without its sheet, is refused: the page
never mixes a figure with a series from another run.

python -m sleeve_fund.research.fund_summary SHEET.md ... --out summary.md [--holdout-status FILE.json]
FILE.json, kept under research/quant-researcher/: {"strategy name": {"status": "spent" | "leaked", "date": "YYYY-MM-DD",
"reason": "..."}}, the Researcher's record of holdouts seen outside a sheet until the holdout lock records it (Advisor,
21 Oct). It can only mark a holdout worse than its sheet says, never clean, and the page names it as the source.
shortcut: 2a only. Pooled sheets, the end value from 1,000, the random-entry percentile and each asset's contribution
come in 2b, after R-10, R-14 and R-15; until then a sheet's asset is its tested instrument."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from sleeve_fund.research.run import returns_path

_TOTAL = re.compile(r"^Total return over the window: ([+-][\d.]+%) \(", re.M)
_DSR = re.compile(r"^- Deflated Sharpe: (\d+)% probability", re.M)
_BUDGET = re.compile(r"^- Budget: (\d+) of (\d+) configurations used", re.M)
_OPENED = re.compile(r"^Dataset .*· holdout: last \d+ days \(opened\)", re.M)
_ORDER = {"PASS": 0, "NOT JUDGED": 1, "FAIL": 2}
_WORSE = {"clean": 0, "spent": 1, "leaked": 2}
POOLED = "Pooled (one try)"
SHORT_DAYS = 90  # fewer shared days than this: the matrix is shown greyed, "too short to read" (Advisor, 21 Oct)
UNSTATED = "Asset not stated"


class Unmatched(ValueError):
    """A sheet without its returns file, or a returns file without its sheet."""


def _check(sheets: list[Path]) -> None:
    bad = [f"{s.name}: no returns file beside it" for s in sheets if not returns_path(s).exists()]
    for d in sorted({s.parent for s in sheets}):
        bad += [f"{f.name}: no tear sheet beside it" for f in sorted(d.glob("*.returns.csv"))
                if not f.with_name(f.name.removesuffix(".returns.csv") + ".md").exists()]
    if bad:
        raise Unmatched("; ".join(bad) + ". Re-run the study: it writes both in one step.")


def _num(s: str | None) -> float | None:
    try:
        return float(str(s).rstrip("%"))
    except (TypeError, ValueError):
        return None


def _double(d: dict | None) -> str:
    if d is None:
        return "not run"
    return "not tested" if d["verdict"] == "NOT TESTED" else f"{d['verdict']}, {d['net']}"


def _section(instrument: str | None) -> str:
    if not instrument:
        return UNSTATED
    if "+" in instrument or "," in instrument:
        return POOLED
    return f"{instrument.split('/')[0]} only"


def _key(r: dict) -> tuple:
    dsr, oos = _num(r["dsr"]), _num(r["oos_sharpe"])
    seen = r["g1"] == "PASS" and r["holdout"] != "clean"
    unranked = dsr is None or r["tries"] == "n/a"
    # an unranked row's deflated Sharpe isn't honest (or isn't there): among those, out-of-sample Sharpe alone
    return (_ORDER.get(r["g1"], 1), seen, unranked, 0.0 if unranked else -dsr, oos is None, -(oos or 0.0))


def _marks(status: dict) -> dict[str, str]:
    """{strategy: "spent" | "leaked"} from the Researcher's file, each entry dated and with its reason."""
    out = {}
    for name, e in status.items():
        if not isinstance(e, dict) or e.get("status") not in ("spent", "leaked"):
            raise ValueError(f"{name}: a holdout status is spent or leaked, never {e!r}: clean is read from the sheet")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(e.get("date", ""))) or not str(e.get("reason", "")).strip():
            raise ValueError(f"{name}: a holdout status needs its date (YYYY-MM-DD) and its reason")
        out[name] = e["status"]
    return out


def rows(sheets: list[Path], holdout_status: dict | None = None) -> dict[str, list[dict]]:
    """Each sheet's printed figures, ranked within its section; sections in page order."""
    from sleeve_fund.dashboard.development import read_sheet

    marks = _marks(holdout_status or {})
    out: dict[str, list[dict]] = {}
    seen = []
    for path in sheets:
        text = path.read_text(encoding="utf-8")
        s = read_sheet(path)
        seen.append(s)
        total, dsr, budget = _TOTAL.search(text), _DSR.search(text), _BUDGET.search(text)
        own = "spent" if _OPENED.search(text) else "clean"
        g1 = s["g1"] or "NOT JUDGED"
        row = {"name": path.stem, "file": path.name, "instrument": s["instrument"] or "unknown", "g1": g1,
               "why": s["evidence"] if g1 == "NOT JUDGED" else "",
               "holdout": max(own, marks.get(s["strategy"] or "", own), key=_WORSE.__getitem__),
               "oos_sharpe": s.get("oos_sharpe", "n/a"), "bench_sharpe": s.get("bench_sharpe", "n/a"),
               "oos_cagr": s.get("oos_cagr", "n/a"), "total": total.group(1) if total else "n/a",
               "dsr": dsr.group(1) + "%" if dsr else "n/a", "double": _double(s["double_fees"]),
               "tries": f"{budget.group(1)} of {budget.group(2)}" if budget else "n/a",
               "trades": str(s["oos_trades"]) if s["oos_trades"] is not None else "n/a"}
        out.setdefault(_section(s["instrument"]), []).append(row)
    # CR F3: a name matching no sheet's strategy (a typo) would leave a spent holdout showing clean: refused
    unmatched = sorted(set(marks) - {s["strategy"] for s in seen})
    if unmatched:
        raise ValueError(f"the holdout status names {', '.join(unmatched)}, which no sheet here is a study of: "
                         "fix the name, or drop the entry")
    order = sorted(out, key=lambda k: (k != POOLED, k == UNSTATED, k))
    return {k: sorted(out[k], key=_key) for k in order}


def correlation(sheets: list[Path]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """The correlation of the daily returns over the days every file has, and those days."""
    series = {p.stem: pd.read_csv(returns_path(p), index_col="date", parse_dates=["date"])["return"] for p in sheets}
    both = pd.concat(series, axis=1, join="inner")
    return both.corr(), both.index


def _cells(r: dict) -> list[str]:
    figures = [r["oos_sharpe"], r["bench_sharpe"], r["oos_cagr"], r["total"], r["dsr"], r["double"], r["tries"],
               r["trades"]]
    untracked = "; tries not tracked" if r["tries"] == "n/a" else ""
    if r["g1"] != "NOT JUDGED":
        g1 = f"{r['g1']} ({untracked[2:]})" if untracked else r["g1"]
        return [f"[{r['name']}]({r['file']})", r["instrument"], g1, r["holdout"], *figures]
    # muted, with its reason: a figure beside "not judged" must never read as nearly passed
    return [f"[{r['name']}]({r['file']})", r["instrument"], f"*NOT JUDGED: {r['why']}{untracked}*", r["holdout"],
            *(f"*{c}*" for c in figures)]


def page(sheets: list[Path], holdout_status: dict | None = None, status_source: str | None = None) -> str:
    _check(sheets)
    ranked = rows(sheets, holdout_status)
    out = ["# Fund summary", "",
           "Each row is its tear sheet's own figures, out of sample after fees unless it says otherwise. Sections are "
           "never ranked against each other: pooling lifts Sharpe through diversification. Within each, G1 PASS, then "
           "NOT JUDGED (in italics: neither a pass nor a fail), then FAIL; among passes a clean holdout above a spent "
           "or leaked one; then by deflated Sharpe, out-of-sample Sharpe breaking ties. A spent or leaked holdout "
           "means the out-of-sample figures were seen before; paper trading is the check. The total return over the "
           "window includes in-sample days.", "",
           f"Holdout status: from each sheet, and spent or leaked as recorded in `{status_source}`." if status_source
           else "Holdout status: from each sheet only.", ""]
    for name, section in ranked.items():
        out += [f"## {name}", "",
                "| Rank | Strategy | Asset | G1 | Holdout | Out-of-sample Sharpe | Benchmark Sharpe "
                "| Out-of-sample CAGR | Total return over the window (includes in-sample) | Deflated Sharpe | Double fees "
                "| Tries of the family's budget | Out-of-sample trades |",
                "| --- " * 13 + "|"]
        out += [f"| {i} | " + " | ".join(_cells(r)) + " |" for i, r in enumerate(section, 1)]
        out.append("")
    corr, common = correlation(sheets)
    out += ["## Correlation of daily returns", ""]
    if len(common) < 2:
        out.append("Not shown: the returns files share fewer than 2 days.")
        return "\n".join(out) + "\n"
    short = len(common) < SHORT_DAYS
    out += [f"Over the {len(common)} common days, {common[0]:%Y-%m-%d} to {common[-1]:%Y-%m-%d}: the days every "
            "file has, so a short history narrows it for all."
            + (f" **Too short to read:** under {SHORT_DAYS} days, shown greyed." if short else ""), ""]
    fmt = (lambda x: f"*{x:.2f}*") if short else (lambda x: f"{x:.2f}")  # noqa: E731
    names = [r["name"] for section in ranked.values() for r in section]
    out.append("| | " + " | ".join(str(i) for i in range(1, len(names) + 1)) + " |")
    out.append("| --- |" + " --- |" * len(names))
    for i, a in enumerate(names, 1):
        out.append(f"| {i} {a} | " + " | ".join(fmt(corr.loc[a, b]) for b in names) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.research.fund_summary")
    ap.add_argument("sheets", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--holdout-status", type=Path, help="JSON under research/quant-researcher/: {strategy: {status, date, reason}}")
    args = ap.parse_args(argv)
    marks = json.loads(args.holdout_status.read_text(encoding="utf-8")) if args.holdout_status else None
    text = page(args.sheets, marks, str(args.holdout_status) if args.holdout_status else None)
    args.out.write_text(text, encoding="utf-8")
    _, common = correlation(args.sheets)
    print(f"fund summary: {len(args.sheets)} strategies, {len(common)} common days: {args.out}")  # counts only
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
