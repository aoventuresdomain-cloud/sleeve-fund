"""Accuracy check 2 (M1 item 9, PM 30 Oct): rebuild a tear sheet's figures from the files beside it, outside the engine.

    python scripts/recompute_sheet.py research/tearsheets/<sheet>.md [more sheets...]

Reads only the sheet's own files: <stem>-trades.csv, <stem>-oos-trades.csv and <stem>.oos-returns.csv beside a single
study's sheet, or <stem>-basket-trades.csv and <stem>-pooled.csv beside a basket's, and the figures the study stated
in <stem>.results.json ("check"). It imports nothing from sleeve_fund, numpy or pandas: plain Python arithmetic, so
an engine error can't be repeated here. Each line printed is one comparison, OK or MISMATCH; the exit status is 1 if
any mismatched, 2 if a file is missing.

What it rebuilds:
- single study: the full period's P&L against its start and end equity and its closed trade count; each test
  window's return from its own trades, its counted trades, and their compounded return and total against the
  sheet's out-of-sample figures; the out-of-sample return, worst drop and Sharpe from its daily returns.
- basket: each day's pooled return from the assets' P&L over the fund's equity the day before; the pooled net, worst
  drop and Sharpe; each asset's P&L and its trades opened on test days."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

RETURN_TOL = 1e-6  # returns and Sharpe: rounding only
MONEY_TOL = 0.01  # money: a cent


class Check:
    def __init__(self) -> None:
        self.failed = 0

    def same(self, what: str, got: float | int | None, stated: float | int | None, tol: float = RETURN_TOL) -> None:
        if got is None or stated is None:
            ok = got is None and stated is None
        elif isinstance(got, int) and isinstance(stated, int):
            ok = got == stated
        else:
            ok = math.isclose(float(got), float(stated), rel_tol=tol, abs_tol=tol)
        self.failed += not ok
        print(f"{'OK      ' if ok else 'MISMATCH'} {what}: rebuilt {_show(got)}, sheet {_show(stated)}")


def _show(x) -> str:
    return "none" if x is None else str(x) if isinstance(x, int) else f"{float(x):.10g}"


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(x: str) -> float:
    return float(x) if x not in ("", None) else 0.0


def compounded(returns: list[float]) -> float:
    return math.prod(1 + r for r in returns) - 1


def worst_drop(returns: list[float]) -> float:
    """The deepest fall from a high on these daily returns compounded from 1, as a negative fraction (0 when none)."""
    equity, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        equity *= 1 + r
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1)
    return worst


def sharpe(returns: list[float], days: int = 365) -> float | None:
    """Mean over sample standard deviation of daily returns, annualised over `days` (None when it can't be read)."""
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
    return mean / sd * math.sqrt(days) if sd > 0 else None


def check_study(sheet: Path, stated: dict, c: Check) -> None:
    trades = _rows(sheet.with_name(f"{sheet.stem}-trades.csv"))
    full = stated["full_period"]
    c.same("full period: trades' P&L against end less start equity", sum(_num(t["pnl"]) for t in trades),
           full["end_equity"] - full["start_equity"], MONEY_TOL)
    c.same("full period: closed trades", sum(1 for t in trades if t["exit_time"]), full["trades"])

    oos = stated["oos"]
    rows = _rows(sheet.with_name(f"{sheet.stem}-oos-trades.csv"))
    by_window: dict[int, list[dict]] = {}
    for r in rows:
        by_window.setdefault(int(r["window"]), []).append(r)
    window_returns = []
    for w in oos["windows"]:
        mine = by_window.get(w["window"], [])
        pnl = sum(_num(t["pnl"]) for t in mine)
        start = _num(mine[-1]["equity_after"]) - pnl if mine else None
        ret = pnl / start if start else 0.0
        window_returns.append(ret)
        c.same(f"test window {w['window']} ({w['from'][:10]} to {w['to'][:10]}): return from its trades", ret,
               w["total_return"] if w["total_return"] is not None else 0.0)
        c.same(f"test window {w['window']}: counted trades", sum(1 for t in mine if t["counted"] == "yes"), w["trades"])
    c.same("out of sample: counted trades", sum(1 for t in rows if t["counted"] == "yes"), oos["trades"])
    c.same("out of sample: return compounded from the windows' trades", compounded(window_returns),
           oos["total_return"] if oos["total_return"] is not None else 0.0)

    days = [float(r["return"]) for r in _rows(sheet.with_suffix(".oos-returns.csv")) if r["return"] not in ("", "nan")]
    c.same("out of sample: days", len(days), oos["days"])
    c.same("out of sample: return from its daily returns", compounded(days) if days else None, oos["total_return"])
    c.same("out of sample: worst drop", worst_drop(days) if days else None, oos["max_drawdown"])
    c.same("out of sample: Sharpe", sharpe(days), oos["sharpe"])


def check_basket(sheet: Path, stated: dict, c: Check) -> None:
    days = _rows(sheet.with_name(f"{sheet.stem}-pooled.csv"))
    assets = list(stated["assets"])
    pooled, worst_row = [], 0.0
    for d in days:
        r = sum(_num(d[f"pnl_{a}"]) for a in assets) / _num(d["equity_before"])
        worst_row = max(worst_row, abs(r - _num(d["pooled_return"])))
        pooled.append(r)
    c.same("pooled: each day's return against the assets' P&L over the equity before (largest gap)", worst_row, 0.0)
    c.same("pooled: test days", len(pooled), stated["days"])
    c.same("pooled: net", compounded(pooled), stated["net"])
    c.same("pooled: worst drop", worst_drop(pooled), stated["max_drawdown"])
    c.same("pooled: Sharpe", sharpe(pooled, stated["annualise_days"]), stated["sharpe"])
    trades = _rows(sheet.with_name(f"{sheet.stem}-basket-trades.csv"))
    for a in assets:
        c.same(f"{a}: P&L on the test days", sum(_num(d[f"pnl_{a}"]) for d in days), stated["assets"][a]["pnl"],
               MONEY_TOL)
        c.same(f"{a}: closed trades opened on a test day",
               sum(1 for t in trades if t["asset"] == a and t["test_day"] == "yes" and t["exit_time"]),
               stated["assets"][a]["trades"])


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    c = Check()
    for name in argv:
        sheet = Path(name)
        print(f"== {sheet.name}")
        try:
            stated = json.loads(sheet.with_suffix(".results.json").read_text(encoding="utf-8"))["check"]
            (check_basket if "basket" in stated else check_study)(sheet, stated.get("basket", stated), c)
        except (FileNotFoundError, KeyError) as exc:
            print(f"MISSING  {exc}: this sheet has no files to recompute from (written before 30 Oct?)")
            return 2
    print(f"{c.failed} mismatch(es)")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
