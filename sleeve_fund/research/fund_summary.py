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
never mixes a figure with a series from another run. Then (d) FINALIST-CORR (Advisor, 21 Oct): the out-of-sample
daily returns (.oos-returns.csv, run.write_oos_returns) pair by pair over the days both have, each with its count,
blank under MIN_PAIR_DAYS, and each strategy against the plain BTC price return.

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

from sleeve_fund.research.run import oos_returns_path, returns_path

_TOTAL = re.compile(r"^Total return over the window: ([+-][\d.]+%) \(", re.M)
_DSR = re.compile(r"^- Deflated Sharpe: (\d+)% probability", re.M)
_BUDGET = re.compile(r"^- Budget: (\d+) of (\d+) configurations used", re.M)
_OPENED = re.compile(r"^Dataset .*· holdout: last \d+ days \(opened\)", re.M)
_ORDER = {"PASS": 0, "NOT JUDGED": 1, "FAIL": 2}
_WORSE = {"clean": 0, "spent": 1, "leaked": 2}
POOLED = "Pooled (one try)"
SHORT_DAYS = 90  # fewer shared days than this: the matrix is shown greyed, "too short to read" (Advisor, 21 Oct)
UNSTATED = "Asset not stated"
MIN_PAIR_DAYS = 60  # (d): a pair sharing fewer out-of-sample days is left blank, its count shown (Advisor, 21 Oct)


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


def _paired(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    """The correlation of two daily series over the calendar days both have, and that count; NaN under
    MIN_PAIR_DAYS."""
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    return (float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) >= MIN_PAIR_DAYS else float("nan")), len(both)


def pairwise(series: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(d) FINALIST-CORR: each pair's correlation over the days both have, and those days' count."""
    names = list(series)
    corr = pd.DataFrame(float("nan"), index=names, columns=names)
    days = pd.DataFrame(0, index=names, columns=names)
    for a in names:
        for b in names:
            corr.loc[a, b], days.loc[a, b] = _paired(series[a], series[b])
    return corr, days


def to_price(series: dict[str, pd.Series], price: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Each series' correlation to a price's daily return over the days both have, and the count."""
    out = {name: _paired(s, price) for name, s in series.items()}
    return pd.Series({k: v[0] for k, v in out.items()}), pd.Series({k: v[1] for k, v in out.items()})


def oos_section(sheets: list[Path], assets: dict[str, str]) -> list[str]:
    """(d) The finalists' out-of-sample daily returns correlated pair by pair, with BTC's price return beside them.
    `assets`: each sheet's stem to its instrument (a BTC one's price column is BTC's price)."""
    out = ["## Correlation of out-of-sample daily returns, pair by pair", ""]
    missing = [s.name for s in sheets if not oos_returns_path(s).exists()]
    if missing:
        return out + [f"Not shown: {', '.join(missing)} {'has' if len(missing) == 1 else 'have'} no out-of-sample "
                      "returns file. Re-run the study: it writes one beside the sheet.", ""]
    read = {s.stem: pd.read_csv(oos_returns_path(s), index_col="date", parse_dates=["date"]) for s in sheets}
    series = {k: f["return"] for k, f in read.items()}
    btc = [f["price"] for k, f in read.items() if str(assets.get(k, "")).split("/")[0] == "BTC"]
    price = pd.concat(btc).groupby(level=0).first() if btc else None
    corr, days = pairwise(series)
    to_btc, btc_days = to_price(series, price) if price is not None else (None, None)
    out += ["Out of sample only (the walk-forward test windows), net of costs; a day out of the market counts as 0. "
            "Each pair is over the calendar days both have, the count beside it, blank under "
            f"{MIN_PAIR_DAYS} days. The last column is each strategy against the plain BTC price return: a set all "
            "close to it is mostly long BTC."
            + ("" if price is not None else " No BTC sheet in the set, so that column reads n/a."), ""]
    cell = lambda c, n: f"({n} d)" if pd.isna(c) else f"{c:.2f} ({n} d)"  # noqa: E731
    names = list(series)
    out.append("| | " + " | ".join(str(i) for i in range(1, len(names) + 1)) + " | BTC price |")
    out.append("| --- |" + " --- |" * (len(names) + 1))
    for i, a in enumerate(names, 1):
        last = "n/a" if to_btc is None else cell(to_btc[a], int(btc_days[a]))
        out.append(f"| {i} {a} | " + " | ".join(cell(corr.loc[a, b], int(days.loc[a, b])) for b in names)
                   + f" | {last} |")
    return out + [""]


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
    names = [r["name"] for section in ranked.values() for r in section]
    finalists = oos_section(sorted(sheets, key=lambda s: names.index(s.stem)),
                            {r["name"]: r["instrument"] for section in ranked.values() for r in section})
    corr, common = correlation(sheets)
    out += ["## Correlation of daily returns", ""]
    if len(common) < 2:
        out.append("Not shown: the returns files share fewer than 2 days.")
        return "\n".join(out + [""] + finalists) + "\n"
    short = len(common) < SHORT_DAYS
    out += [f"Over the {len(common)} common days, {common[0]:%Y-%m-%d} to {common[-1]:%Y-%m-%d}: the days every "
            "file has, so a short history narrows it for all."
            + (f" **Too short to read:** under {SHORT_DAYS} days, shown greyed." if short else ""), ""]
    fmt = (lambda x: f"*{x:.2f}*") if short else (lambda x: f"{x:.2f}")  # noqa: E731
    out.append("| | " + " | ".join(str(i) for i in range(1, len(names) + 1)) + " |")
    out.append("| --- |" + " --- |" * len(names))
    for i, a in enumerate(names, 1):
        out.append(f"| {i} {a} | " + " | ".join(fmt(corr.loc[a, b]) for b in names) + " |")
    return "\n".join(out + [""] + finalists) + "\n"


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
