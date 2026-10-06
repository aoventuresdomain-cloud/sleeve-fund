"""Monthly performance and CSV exports for reviews, the auditor and the tax tool."""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timedelta, timezone

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
            "key": ts.strftime("%Y-%m"),  # the Records page's month chips filter by it
            "short": ts.strftime("%b"),
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


# --- Records page: one period and strategy filter over the performance figures and the decision log ---------

PERIODS = {"1m": ("1M", 30), "6m": ("6M", 182), "1y": ("1Y", 365), "all": ("All", None)}
DEFAULT_PERIOD = "all"  # what the Reports page showed: every month
MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")

# Decision log kinds, for the icon beside each line and the filter chips above the log.
DECISION_KINDS = {
    "started": ("Started", "▶"), "stopped": ("Stopped", "■"), "changed": ("Changed", "✎"),
    "risk": ("Risk", "⏸"), "approval": ("Approval", "✓"),
}
KIND_FILTERS = [("all", "All"), ("startstop", "Started / stopped"), ("risk", "Risk"), ("changed", "Settings changed"),
                ("approval", "Approvals")]
_KIND_OF = {"create": "started", "start": "started", "resume": "started", "restore": "started",
            "stop": "stopped", "archive": "stopped", "pause": "risk", "flatten": "risk", "flatten everything": "risk"}


def decision_kind(action: str) -> str:
    """started, stopped, changed, risk or approval: how the log draws a decision."""
    a = (action or "").lower()
    if "approv" in a or a.startswith("g2"):
        return "approval"
    return _KIND_OF.get(a, "changed")


def window(period: str, month: str, now: datetime) -> dict:
    """The dates the Records page covers: a month chip's month, else the period back from now.
    start/end are UTC datetimes or None (open). `period` is always a key of PERIODS."""
    period = period if period in PERIODS else DEFAULT_PERIOD
    m = MONTH_RE.match(month or "")
    days = PERIODS[period][1]
    pstart = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0) if days else None
    out = {"period": period, "month": None, "start": pstart, "end": None, "period_start": pstart}
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        start = datetime(y, mo, 1, tzinfo=timezone.utc)
        out.update(month=month, start=start, end=datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=timezone.utc),
                   month_label=start.strftime("%b %Y"))
    return out


def _utc(t):
    return None if t is None else pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC")


def _within(idx, start, end):
    keep = pd.Series(True, index=idx)
    if start is not None:
        keep &= idx >= _utc(start)
    if end is not None:
        keep &= idx < _utc(end)
    return keep.values


def performance(summaries: list[dict], frames: dict[str, pd.DataFrame], fills: list[dict], start, end) -> dict:
    """The four figures and the book vs benchmark curve over [start, end): return after fees, the gap to
    buy-and-hold in points, the worst drawdown with its date, and fees and fills. Returns are measured
    from the equity at the window's start (the starting capital when the window opens before the first mark)."""
    curve = bookm.book_curve(summaries, frames)
    capital = sum(x["sleeve"].starting_balance for x in summaries)
    f = pd.DataFrame(fills)
    if len(f):
        f["ts"] = pd.to_datetime(f["ts"], utc=True)
        f = f[_within(pd.DatetimeIndex(f["ts"]), start, end)]
    fees, n_fills = (float(f["fee"].sum()), int(len(f))) if len(f) else (0.0, 0)
    if not len(curve):
        return {"empty": True, "fees": fees, "fills": n_fills}
    before = curve[curve.index < _utc(start)] if start is not None else curve.iloc[0:0]
    base_eq = float(before["equity"].iloc[-1]) if len(before) else capital
    base_b = float(before["benchmark"].iloc[-1]) if len(before) else capital
    w = curve[_within(curve.index, start, end)]
    if not len(w) or not base_eq or not base_b:
        return {"empty": True, "fees": fees, "fills": n_fills}
    ret = float(w["equity"].iloc[-1]) / base_eq - 1
    bench = float(w["benchmark"].iloc[-1]) / base_b - 1
    peak = w["equity"].cummax().clip(lower=base_eq)
    dd = 1 - w["equity"] / peak
    worst = float(dd.max())
    book = [0.0] + [float(v) / base_eq - 1 for v in w["equity"]]
    bmk = [0.0] + [float(v) / base_b - 1 for v in w["benchmark"]]
    first = (w.index[0] - pd.Timedelta(days=1)) if not len(before) else before.index[-1]
    days = [first] + list(w.index)
    return {"empty": False, "ret": ret, "bench": bench, "vs": ret - bench, "worst_dd": worst,
            "worst_on": dd.idxmax() if worst > 0 else None, "fees": fees, "fills": n_fills,
            "pnl": float(w["equity"].iloc[-1]) - base_eq, "chart": curve_chart(days, book, bmk)}


def curve_chart(days: list, book: list[float], bench: list[float]) -> dict:
    """Book and benchmark returns for the Lightweight Charts line (QA U10): a UTC timestamp per day and each
    return in percent, rounded for the page."""
    return {"t": [int(pd.Timestamp(d).timestamp()) for d in days],
            "book": [round(v * 100, 4) for v in book], "bench": [round(v * 100, 4) for v in bench]}


def in_window(month_key: str, w: dict, use_month: bool = False) -> bool:
    """Whether a 'YYYY-MM' month falls in the window (the period's, unless use_month)."""
    start = w["start"] if use_month else w["period_start"]
    end = w["end"] if use_month else None
    if start is not None and month_key < start.strftime("%Y-%m"):
        return False
    return not (end is not None and month_key >= end.strftime("%Y-%m"))


def timeline(decisions: list[dict], now: datetime) -> list[dict]:
    """Decisions grouped by UTC day, newest first: [{"label": "Today" | "Yesterday" | "3 Oct", "items": [...]}]."""
    days: list[dict] = []
    today = now.date()
    for d in decisions:
        ts = d["ts"]
        day = ts.date()
        label = ("Today" if day == today else "Yesterday" if day == today - timedelta(days=1)
                 else ts.strftime("%-d %b" if ts.year == now.year else "%-d %b %Y"))
        if not days or days[-1]["label"] != label:
            days.append({"label": label, "items": []})
        days[-1]["items"].append({**d, "kind": decision_kind(d["action"])})
    return days
