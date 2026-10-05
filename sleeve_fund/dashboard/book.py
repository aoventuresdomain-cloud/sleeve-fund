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
    if unit == "week":  # Monday 00:00 UTC
        return day - pd.Timedelta(days=day.weekday())
    return day.replace(day=1) if unit == "month" else day


def _equity_at(store: Store, sleeve, ts: datetime) -> float:
    row = store.equity_at_or_before(sleeve.name, ts)
    return row["equity"] if row else sleeve.starting_balance


def sleeve_extras(store: Store, x: dict, frame: pd.DataFrame) -> dict:
    """Adds day/MTD P&L, open position, sparkline and last activity to a sleeve_summary dict."""
    s, now = x["sleeve"], utcnow()
    day_open = _equity_at(store, s, _start_of(now, "day"))
    week_open = _equity_at(store, s, _start_of(now, "week"))
    month_open = _equity_at(store, s, _start_of(now, "month"))
    book = store.journal_book(s.name, s.starting_balance)
    last = store.last_equity(s.name)
    price = last["price"] if last else 0.0
    qty = last["qty"] if last else 0.0
    unreal = qty * (price - book["entry_px"]) if book["entry_px"] and qty else 0.0  # a short's qty is negative
    fill = store.fills(s.name, limit=1)
    rec = store.last_event(s.name, ("reconcile", "reconcile_mismatch"))
    tail = frame["equity"].tail(60).tolist() if len(frame) else []
    x.update(
        mode="paper",
        day_pnl=x["equity"] - day_open,
        day_ret=x["equity"] / day_open - 1 if day_open else 0.0,
        week_pnl=x["equity"] - week_open,
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
    dd = curve["drawdown"] if len(curve) else pd.Series(dtype=float)  # from the starting capital too
    # Gross counts a short as exposure too; net lets a short offset a long. Signed values: + long, - short.
    exposure = sum(abs(x["position_value"]) for x in active)
    net = sum(x["position_value"] for x in active)
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
        "week_pnl": sum(x["week_pnl"] for x in active),
        "mtd_pnl": sum(x["mtd_pnl"] for x in active),
        "exposure": exposure,
        "exposure_pct": exposure / equity if equity else 0.0,
        "net_exposure": net,
        "net_pct": net / equity if equity else 0.0,
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
    # Measured from the capital the book started with too: its starting balances are never a daily close, so a
    # book whose first close was already a loss read 0% drawdown (round 12, M12-F1).
    start = sum(x["sleeve"].starting_balance for x in summaries)
    peak = out["equity"].cummax().clip(lower=start)
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
    """Capital by instrument (open positions) plus cash, as shares of book equity. A short is its own row
    with a negative value (its sale proceeds sit in cash, so cash can exceed equity). `bar` is the row's
    width in the allocation bar: positions by gross value, cash filling what is left."""
    by_asset: dict[str, float] = {}
    for x in summaries:
        base = x["sleeve"].instrument.split("/")[0]
        if x["position_value"]:
            key = base if x["position_value"] > 0 else f"{base} short"
            by_asset[key] = by_asset.get(key, 0.0) + x["position_value"]
    rows = [{"name": k, "value": v} for k, v in sorted(by_asset.items(), key=lambda kv: -abs(kv[1]))]
    gross = sum(abs(r["value"]) for r in rows)
    scale = max(equity, gross) if equity > 0 else gross
    for r in rows:
        r["share"] = r["value"] / equity if equity else 0.0
        r["bar"] = abs(r["value"]) / scale if scale else 0.0
    cash = sum(x["cash"] for x in summaries)
    rows.append({"name": "Cash", "value": cash, "share": cash / equity if equity else 0.0,
                 "bar": max(0.0, 1 - sum(r["bar"] for r in rows))})
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


HOLDINGS_SHOWN = 10


def holdings(rows: list[dict], equity: float, limit: int = HOLDINGS_SHOWN) -> dict:
    """The book's open positions netted by instrument across strategies, largest notional first, for the
    allocation bar and the top holdings table under it. rows are trading.book_positions rows. Notional is
    the net position's value at the mark; weight is notional over book value; share is the instrument's part
    of the summed notional (the bar's segment). Margin adds up each position's isolated margin when the rows
    carry it, else None. crossing: one strategy long and another short the same instrument."""
    by: dict[tuple[str, bool], dict] = {}
    for r in rows:
        perp = r.get("perp") is not None
        h = by.setdefault((r["pair"], perp), {"instrument": r["pair"], "perp": perp, "qty": 0.0, "value": 0.0,
                                               "unrealised": 0.0, "margin": 0.0, "held": [], "sides": set()})
        h["qty"] += r["qty"]
        h["value"] += r["value"]
        h["unrealised"] += r["unrealised"]
        m = r.get("margin")
        h["margin"] = None if m is None or h["margin"] is None else h["margin"] + m
        h["held"].append({"sleeve": r["sleeve"], "side": r["side"]})
        h["sides"].add(r["side"])
    out = []
    for h in by.values():
        sides = h.pop("sides")
        notional = abs(h["value"])
        out.append({**h, "notional": notional, "side": (1 if h["value"] > 0 else -1 if h["value"] < 0 else 0),
                    "weight": notional / equity if equity > 0 else 0.0, "crossing": len(sides) > 1})
    out.sort(key=lambda h: -h["notional"])
    total = sum(h["notional"] for h in out)
    for h in out:
        h["share"] = h["notional"] / total if total else 0.0
    return {"rows": out[:limit], "count": len(out), "notional": total}
