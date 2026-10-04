"""Book-level numbers for the portfolio screen: daily curves, P&L windows, risk and allocation.

Everything is derived from the journal (equity marks and fills), so paper and live
sleeves are measured the same way.
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from sleeve_fund.store import Store, utcnow

DAYS_A_YEAR = 365  # the venue trades every day
MIN_DAYS_FOR_RATIOS = 30
MIN_DAYS_FOR_CORRELATION = 20


def daily(store: Store, sleeve: str) -> pd.DataFrame:
    """Last mark of each UTC day: equity, benchmark, cash, qty, price."""
    rows = store.equity_series(sleeve, limit=500_000)
    if not rows:
        return pd.DataFrame(columns=["equity", "benchmark", "cash", "qty", "price"])
    df = pd.DataFrame(rows).set_index("ts")[["equity", "benchmark", "cash", "qty", "price"]]
    df.index = pd.to_datetime(df.index, utc=True)
    return df.resample("1D").last().dropna(how="all")


def _start_of(now: datetime, unit: str) -> datetime:
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return day.replace(day=1) if unit == "month" else day


def _equity_at(store: Store, sleeve, ts: datetime) -> float:
    row = store.equity_at_or_before(sleeve.name, ts)
    return row["equity"] if row else sleeve.starting_balance


def sleeve_extras(store: Store, x: dict, frame: pd.DataFrame) -> dict:
    """Adds day/MTD P&L, open position, sparkline and last activity to a sleeve_summary dict."""
    s, now = x["sleeve"], utcnow()
    day_open = _equity_at(store, s, _start_of(now, "day"))
    month_open = _equity_at(store, s, _start_of(now, "month"))
    book = store.journal_book(s.name, s.starting_balance)
    last = store.last_equity(s.name)
    price = last["price"] if last else 0.0
    qty = last["qty"] if last else 0.0
    unreal = qty * (price - book["entry_px"]) if book["entry_px"] and qty > 0 else 0.0
    fill = store.fills(s.name, limit=1)
    rec = store.last_event(s.name, ("reconcile", "reconcile_mismatch"))
    tail = frame["equity"].tail(60).tolist() if len(frame) else []
    x.update(
        mode="paper",
        day_pnl=x["equity"] - day_open,
        day_ret=x["equity"] / day_open - 1 if day_open else 0.0,
        mtd_pnl=x["equity"] - month_open,
        qty=qty,
        price=price,
        position_value=qty * price,
        cash=last["cash"] if last else s.starting_balance,
        entry_px=book["entry_px"],
        unrealised=unreal,
        last_fill=fill[0] if fill else None,
        reconcile=rec,
        spark=spark_path(tail),
        spark_up=len(tail) < 2 or tail[-1] >= tail[0],
    )
    return x


def spark_path(values: list[float], w: int = 96, h: int = 24) -> str:
    """SVG polyline points for a small trend line, or '' when there is nothing to draw."""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = w / (len(values) - 1)
    return " ".join(f"{i * step:.1f},{h - 1 - (v - lo) / span * (h - 2):.1f}" for i, v in enumerate(values))


def book_view(store: Store, summaries: list[dict], frames: dict[str, pd.DataFrame]) -> dict:
    # Every strategy in the current book counts, stopped and archived ones included: their cash is still the
    # fund's, and stopping one must never rewrite the book's past. A clean slate starts a new book; the
    # strategies it put away are an earlier book's, so the caller leaves them out (Store.previous_book).
    active = summaries
    start = sum(x["sleeve"].starting_balance for x in active)
    equity = sum(x["equity"] for x in active)
    curve = book_curve(active, frames)
    rets = curve["equity"].pct_change().dropna() if len(curve) else pd.Series(dtype=float)
    enough = len(rets) >= MIN_DAYS_FOR_RATIOS
    vol = float(rets.std() * math.sqrt(DAYS_A_YEAR)) if enough else float("nan")
    sharpe = float(rets.mean() / rets.std() * math.sqrt(DAYS_A_YEAR)) if enough and rets.std() > 0 else float("nan")
    brets = curve["benchmark"].pct_change().dropna() if len(curve) else pd.Series(dtype=float)
    bench_sharpe = (float(brets.mean() / brets.std() * math.sqrt(DAYS_A_YEAR))
                    if len(brets) >= MIN_DAYS_FOR_RATIOS and brets.std() > 0 else float("nan"))
    peak = curve["equity"].cummax() if len(curve) else pd.Series(dtype=float)
    dd = (1 - curve["equity"] / peak) if len(curve) else pd.Series(dtype=float)
    exposure = sum(x["position_value"] for x in active)
    return {
        "sleeves": len(summaries),
        "running": sum(1 for x in summaries if x["sleeve"].status == "running"),
        "attention": sum(1 for x in summaries if x["sleeve"].status in ("halted", "error")),
        "unhealthy": sum(1 for x in active if not x["healthy"] and x["sleeve"].desired_state == "running"),
        "start": start,
        "equity": equity,
        "pnl": equity - start,
        "ret": equity / start - 1 if start else 0.0,
        "bench_ret": (sum(x["benchmark"] for x in active) / start - 1) if start else 0.0,
        "day_pnl": sum(x["day_pnl"] for x in active),
        "mtd_pnl": sum(x["mtd_pnl"] for x in active),
        "exposure": exposure,
        "exposure_pct": exposure / equity if equity else 0.0,
        "cash": sum(x["cash"] for x in active),
        "unrealised": sum(x["unrealised"] for x in active),
        "fees": sum(x["fees"] for x in active),
        "drawdown": float(dd.iloc[-1]) if len(dd) else 0.0,
        "max_drawdown": float(dd.max()) if len(dd) else 0.0,
        "vol": vol,
        "sharpe": sharpe,
        "bench_sharpe": bench_sharpe,
        "days": len(curve),
        "allocation": allocation(active, equity),
        "correlation": correlation(active, frames),
    }


def book_curve(summaries: list[dict], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily book equity and benchmark. A sleeve counts as idle cash before its first mark."""
    parts = {x["sleeve"].name: frames[x["sleeve"].name] for x in summaries if len(frames[x["sleeve"].name])}
    if not parts:
        return pd.DataFrame(columns=["equity", "benchmark"])
    idx = sorted(set().union(*(f.index for f in parts.values())))
    out = pd.DataFrame(index=pd.DatetimeIndex(idx), data={"equity": 0.0, "benchmark": 0.0})
    for x in summaries:
        f = parts.get(x["sleeve"].name)
        base = x["sleeve"].starting_balance
        for col in ("equity", "benchmark"):
            out[col] += f[col].reindex(out.index).ffill().fillna(base) if f is not None else base
    peak = out["equity"].cummax()
    out["drawdown"] = 1 - out["equity"] / peak
    return out


# Resolution of the short ranges: five-minute points for a day, half-hourly for a week.
RECENT_STEP = {1: "5min", 7: "30min"}


def recent_curve(store: Store, sleeves: list, days: int, prior_peak: float | None = None) -> pd.DataFrame:
    """Equity and benchmark summed over the given strategies at fine resolution for the last `days`.
    A strategy without a mark in a step carries its last value; one with no mark yet counts as its
    starting balance. prior_peak is the highest equity before the window, so drawdown is measured from
    the real peak rather than from the start of the window."""
    now = utcnow()
    step = RECENT_STEP.get(days, "30min")
    since = pd.Timestamp(now - pd.Timedelta(days=days)).floor(step)
    idx = pd.date_range(since, pd.Timestamp(now).floor(step), freq=step)
    out = pd.DataFrame(index=idx, data={"equity": 0.0, "benchmark": 0.0})
    for s in sleeves:
        before = store.equity_at_or_before(s.name, since.to_pydatetime())
        rows = store.equity_since(s.name, since.to_pydatetime())
        for col in ("equity", "benchmark"):
            start = before[col] if before else s.starting_balance
            if rows:
                ser = pd.Series([r[col] for r in rows], index=pd.to_datetime([r["ts"] for r in rows], utc=True))
                ser = ser.resample(step).last().reindex(idx).ffill().fillna(start)
            else:
                ser = pd.Series(start, index=idx)
            out[col] += ser
    peak = out["equity"].cummax()
    if prior_peak:
        peak = peak.clip(lower=prior_peak)
    out["drawdown"] = 1 - out["equity"] / peak
    return out


def allocation(summaries: list[dict], equity: float) -> list[dict]:
    """Capital by instrument (open positions) plus cash, as shares of book equity."""
    by_asset: dict[str, float] = {}
    for x in summaries:
        base = x["sleeve"].instrument.split("/")[0]
        if x["position_value"] > 0:
            by_asset[base] = by_asset.get(base, 0.0) + x["position_value"]
    rows = [{"name": k, "value": v} for k, v in sorted(by_asset.items(), key=lambda kv: -kv[1])]
    rows.append({"name": "Cash", "value": sum(x["cash"] for x in summaries)})
    for r in rows:
        r["share"] = r["value"] / equity if equity else 0.0
    return rows


def correlation(summaries: list[dict], frames: dict[str, pd.DataFrame]) -> dict:
    """Pairwise correlation of daily sleeve returns, where enough overlapping days exist."""
    names = [x["sleeve"].name for x in summaries if len(frames[x["sleeve"].name]) > MIN_DAYS_FOR_CORRELATION]
    if len(names) < 2:
        return {"names": names, "matrix": []}
    rets = pd.DataFrame({n: frames[n]["equity"] for n in names}).pct_change()
    corr = rets.corr(min_periods=MIN_DAYS_FOR_CORRELATION)
    return {"names": names, "matrix": [[(None if pd.isna(corr.loc[a, b]) else round(float(corr.loc[a, b]), 2))
                                         for b in names] for a in names]}
