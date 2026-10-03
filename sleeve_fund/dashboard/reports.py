"""Monthly performance and CSV exports for reviews, the auditor and the tax tool."""

from __future__ import annotations

import csv
import io

import pandas as pd

from sleeve_fund.dashboard import book as bookm


def monthly(summaries: list[dict], frames: dict[str, pd.DataFrame], fills_by_sleeve: dict[str, list[dict]]) -> dict:
    """Book and per-sleeve returns by calendar month, plus fees and trading activity."""
    curve = bookm.book_curve(summaries, frames)
    if not len(curve):
        return {"rows": [], "sleeves": [], "grid": {}}
    month_end = curve.resample("ME").last()
    start = sum(x["sleeve"].starting_balance for x in summaries)
    prev_eq = month_end["equity"].shift(1).fillna(start)
    prev_b = month_end["benchmark"].shift(1).fillna(start)
    fills = pd.DataFrame([f for fs in fills_by_sleeve.values() for f in fs])
    if len(fills):
        fills["month"] = pd.to_datetime(fills["ts"], utc=True).dt.tz_localize(None).dt.to_period("M")
        fills["notional"] = fills["qty"] * fills["price"]
    rows = []
    for ts, r in month_end.iloc[::-1].iterrows():
        period = ts.tz_localize(None).to_period("M") if ts.tzinfo else ts.to_period("M")
        mf = fills[fills["month"] == period] if len(fills) else fills
        rows.append({
            "month": ts.strftime("%b %Y"),
            "equity": r["equity"],
            "pnl": r["equity"] - prev_eq[ts],
            "ret": r["equity"] / prev_eq[ts] - 1 if prev_eq[ts] else 0.0,
            "bench_ret": r["benchmark"] / prev_b[ts] - 1 if prev_b[ts] else 0.0,
            "worst_dd": float(curve.loc[_months(curve.index) == period, "drawdown"].max()),
            "fees": float(mf["fee"].sum()) if len(mf) else 0.0,
            "fills": int(len(mf)),
            "turnover": float(mf["notional"].sum()) if len(mf) else 0.0,
        })
    grid, months = {}, [r["month"] for r in rows]
    for x in summaries:
        f = frames[x["sleeve"].name]
        if not len(f):
            continue
        me = f["equity"].resample("ME").last()
        prev = me.shift(1).fillna(x["sleeve"].starting_balance)
        grid[x["sleeve"].name] = {ts.strftime("%b %Y"): float(v / prev[ts] - 1) for ts, v in me.items()}
    return {"rows": rows, "sleeves": list(grid), "grid": grid, "months": months}


def _months(idx):
    return (idx.tz_localize(None) if idx.tz is not None else idx).to_period("M")


def _cell(v):
    # Spreadsheet formula injection: a reason typed as "=HYPERLINK(...)" must stay text.
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    if isinstance(v, float):
        return float(f"{v:.12g}")  # drop binary noise such as 57651.420000000006
    return v.isoformat() if hasattr(v, "isoformat") else v


def to_csv(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(columns)
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in columns])
    return buf.getvalue()
