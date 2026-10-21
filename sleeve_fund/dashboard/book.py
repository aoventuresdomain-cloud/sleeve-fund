"""Book-level numbers for the portfolio screen: daily curves, P&L windows, risk and allocation.

Everything is derived from the journal (equity marks and fills), so paper and live
sleeves are measured the same way.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import pandas as pd

from sleeve_fund.dashboard.metrics import costs, shown_status
from sleeve_fund.store import Store, utcnow

# The book's curve and its benchmark inside the fund (A-20, B34-BENCHMARK): its one-line label, so it's never read as an
# account statement (Advisor, 20 Oct).
HISTORY_NOTE = "History shown at today's capital; deposits and withdrawals are not shown as gains or losses"
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


def is_test_account(store: Store, sleeve) -> bool:
    """Whether the Strategy is outside the fund's book: its quote currency held under 8 places, on the test account its
    venue names (CCY-8DP, F229-6). Its results are never Strategy evidence and the gate leaves it out, so the PM's
    figures list it apart (CCY-8DP-UI). The supervisor's one predicate decides, never a copy of it."""
    from sleeve_fund.supervisor import outside_fund_book

    return outside_fund_book(store, sleeve)


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
        test_account=is_test_account(store, s),
        day_pnl=x["equity"] - day_open,
        # Inside the fund its equity is its P&L from 0: today's move as a share of the fund's equity (B-20)
        day_ret=(_fund_share(store, x["equity"] - day_open) if x.get("in_fund")
                 else x["equity"] / day_open - 1 if day_open else 0.0),
        week_pnl=x["equity"] - week_open,
        mtd_pnl=x["equity"] - month_open,
        qty=qty,
        price=price,
        position_value=qty * price,
        cash=last["cash"] if last else s.starting_balance,
        entry_px=book["entry_px"],
        entry_fees=float(book.get("entry_fees") or 0.0),  # off the isolated margin (LIQ-FEE): the engine's liquidation
        unrealised=unreal,
        last_fill=fill[0] if fill else None,
        reconcile=rec,
        spark=spark_path(tail),
        spark_up=len(tail) < 2 or tail[-1] >= tail[0],
    )
    return x


def _fund_share(store: Store, money: float) -> float:
    """`money` as a share of the fund's marked equity; 0 while it can't be marked (B-20)."""
    from sleeve_fund.dashboard.metrics import fund_equity_or_none

    fund = fund_equity_or_none(store)
    return money / fund if fund else 0.0


def spark_path(values: list[float], w: int = 96, h: int = 24) -> str:
    """SVG polyline points for a small trend line, or '' when there is nothing to draw."""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = w / (len(values) - 1)
    return " ".join(f"{i * step:.1f},{h - 1 - (v - lo) / span * (h - 2):.1f}" for i, v in enumerate(values))


def fund_cash(store: Store) -> float:
    """A-20: the cash the fund holds that no Strategy carries. Once the fund holds the cash (the re-base deposit
    applied, supervisor.bridge_start) every Strategy is made with 0 (CUT-A (a)) and the fund's book is its own cash put
    in plus what the Strategies made (supervisor.fund_book), so the book's figures and curves add this to the
    Strategies' own; before the bridge it is 0 and the book is theirs alone."""
    from sleeve_fund.supervisor import bridge_start, fund_book

    return float(fund_book(store).capital) if bridge_start(store) is not None else 0.0


def fund_line(store: Store, start: float):
    """B34-BENCHMARK (Advisor, 20 Oct): inside the fund the book's benchmark is one hold of the fund's capital from the
    bridge, on the curve's own base `start` (its capital, A-20), counted once (fund_bench.fund_hold), never each
    Strategy's row added up. None before the bridge (each Strategy carries its own capital and benchmark then) or with
    no stored closes: the line then stays the book's cash, and the page says buy-and-hold isn't available."""
    from sleeve_fund.dashboard.fund_bench import fund_hold
    from sleeve_fund.supervisor import bridge_start

    bridge = bridge_start(store)
    if bridge is None:
        return None
    try:
        return fund_hold(store, bridge, start)
    except Exception as exc:  # noqa: BLE001 - the book's figures stand without it; said in the log and on the page
        logging.getLogger(__name__).warning(f"the fund's benchmark couldn't be read: {exc!r}")
        return None


def on_days(values: pd.Series, index: pd.DatetimeIndex, before: float) -> pd.Series:
    """The hold's last value in each UTC day of `index` (the daily curves' labels), carried forward; `before` until its
    first point."""
    days = values.groupby(values.index.floor("1D")).last()
    return days.reindex(days.index.union(index)).ffill().reindex(index).fillna(before)


def book_equity(store: Store, summaries: list[dict]) -> float:
    """The book's equity as book_view reads it, without the rest of the view: inside the fund the fund's marked equity
    (its own cash and every Strategy's P&L), else the Strategies' sum (B-20)."""
    base = fund_cash(store)
    own = sum(x["equity"] for x in summaries) + base
    if not base:
        return own
    from sleeve_fund.dashboard.metrics import fund_equity_or_none

    marked = fund_equity_or_none(store)
    return marked if marked is not None else own


def book_view(store: Store, summaries: list[dict], frames: dict[str, pd.DataFrame]) -> dict:
    # Every strategy in the current book counts, stopped and archived ones included: their cash is still the
    # fund's, and stopping one must never rewrite the book's past. A clean slate starts a new book; the
    # strategies it put away are an earlier book's, so the caller leaves them out (Store.previous_book). One put
    # away whose equity the poll has taken out (Store.withdrawn, F-C3B-A) is out of the book, as fund_equity
    # leaves it out (T-FIX4); it still lists as archived.
    out = store.withdrawn()
    active = [x for x in summaries if x["sleeve"].name not in out]
    put_away = sum(x["equity"] for x in summaries if x["sleeve"].name in out)
    base = fund_cash(store)
    start = sum(x["sleeve"].starting_balance for x in active) + base
    equity = sum(x["equity"] for x in active) + base
    if base:  # the fund's book as the gate and sizing read it: its cash, the journal's P&L, positions at their marks
        from sleeve_fund.dashboard.metrics import fund_equity_or_none

        # fund_equity raises while a position has no mark yet: the page then shows the marks' own sum, never a 500
        marked = fund_equity_or_none(store)
        equity = marked if marked is not None else equity
    line = fund_line(store, start) if base else None
    held = line is not None and len(line.values) > 0
    curve = book_curve(active, frames, base=base, bench=line.values if held else None)
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
    # Counted as Strategies: a two-leg Strategy's second leg row (P2-8) is part of its first leg's, not one more.
    ones = [x for x in summaries if not getattr(x["sleeve"], "parent", None)]
    archived = store.archived()
    return {
        "sleeves": len(ones),
        "running": sum(1 for x in ones if x["sleeve"].status == "running"),
        # One put away waits for nobody: its last halt or error is history (the top bar's count, B-29 add-on).
        "attention": sum(1 for x in ones if shown_status(x["sleeve"], store) in ("halted", "error")
                         and x["sleeve"].name not in archived),
        "unhealthy": len({getattr(x["sleeve"], "parent", None) or x["sleeve"].name for x in active
                          if not x["healthy"] and x["sleeve"].desired_state == "running"}),
        "start": start,
        "equity": equity,
        "fund_cash": base,  # the fund's own cash no Strategy carries, once the fund holds it (A-20)
        # Put away, not trading (Advisor on T-FIX4): what the withdrawn put-aways hold, apart from the book; the two
        # add up to the capital of every Strategy in the current book.
        "put_away": put_away,
        "capital": equity + put_away,
        "pnl": equity - start,
        "ret": equity / start - 1 if start else 0.0,
        # inside the fund its hold, to the latest stored close (never a day behind); else the Strategies' own rows
        # inside the fund with no hold to read (no stored closes): n/a, never the book's cash read as 0.00% (TEC-B34-NA)
        "bench_ret": (float(line.values.iloc[-1]) / start - 1 if held else None if base
                      else ((sum(x["benchmark"] for x in active) + base) / start - 1) if start else 0.0),
        "bench_as_of": line.as_of if held else None,
        "bench_source": line.source if held else None,
        # Advisor condition (20 Oct): the curve states the fund's history at today's capital (A-20), never a statement
        "history_note": HISTORY_NOTE if base else None,
        "day_pnl": sum(x["day_pnl"] for x in active),
        "week_pnl": sum(x["week_pnl"] for x in active),
        "mtd_pnl": sum(x["mtd_pnl"] for x in active),
        "exposure": exposure,
        "exposure_pct": exposure / equity if equity else 0.0,
        "net_exposure": net,
        "net_pct": net / equity if equity else 0.0,
        "cash": sum(x["cash"] for x in active) + base,
        "unrealised": sum(x["unrealised"] for x in active),
        "fees": sum(x["fees"] for x in active),
        "funding": sum(x["funding"] for x in active),
        **costs(equity - start, sum(x["fees"] for x in active), sum(x["funding"] for x in active)),
        "drawdown": float(dd.iloc[-1]) if len(dd) else 0.0,
        "max_drawdown": float(dd.max()) if len(dd) else 0.0,
        "vol": vol,
        "sharpe": sharpe,
        "bench_sharpe": bench_sharpe,
        "days": len(curve),
        "allocation": allocation(active, equity, base),
        "correlation": correlation(active, frames),
    }


def book_curve(summaries: list[dict], frames: dict[str, pd.DataFrame], base: float = 0.0,
               bench: pd.Series | None = None) -> pd.DataFrame:
    """Daily book equity and benchmark. A sleeve counts as idle cash before its first mark; `base` is the fund's own
    cash no Strategy carries (fund_cash). `bench`: inside the fund, the fund's hold (fund_line), the benchmark in place
    of the rows' sum."""
    parts = {x["sleeve"].name: frames[x["sleeve"].name] for x in summaries if len(frames[x["sleeve"].name])}
    if not parts:
        return pd.DataFrame(columns=["equity", "benchmark"])
    idx = sorted(set().union(*(f.index for f in parts.values())))
    out = pd.DataFrame(index=pd.DatetimeIndex(idx), data={"equity": base, "benchmark": base})
    for x in summaries:
        f = parts.get(x["sleeve"].name)
        own = x["sleeve"].starting_balance  # not `base`: that is the fund's cash, read again for `start` below
        for col in ("equity", "benchmark"):
            out[col] += f[col].reindex(out.index).ffill().fillna(own) if f is not None else own
    # Measured from the capital the book started with too: its starting balances are never a daily close, so a
    # book whose first close was already a loss read 0% drawdown (round 12, M12-F1).
    start = sum(x["sleeve"].starting_balance for x in summaries) + base
    if bench is not None:
        out["benchmark"] = on_days(bench, out.index, start)
    out["equity"] = out["equity"].astype(float)  # the marks are Money: 0/0 there raises before `where` (B-41)
    peak = out["equity"].cummax().clip(lower=float(start))
    out["drawdown"] = _drawdown(out["equity"], peak)
    return out


def _drawdown(equity: pd.Series, peak: pd.Series) -> pd.Series:
    """1 - equity / peak where the peak is above 0; 0 where it isn't (no capital, no drawdown: CUT-A), divided only
    where it is, in floats (B-41: a fund-held book whose peak is 0 raised on Money 0/0)."""
    out = pd.Series(0.0, index=equity.index)
    up = peak > 0
    out[up] = 1 - equity[up].astype(float) / peak[up].astype(float)
    return out


# Resolution of the short ranges: five-minute points for a day, half-hourly for a week.
RECENT_STEP = {1: "5min", 7: "30min"}


def recent_curve(store: Store, sleeves: list, days: int, prior_peak: float | None = None,
                 base: float = 0.0, bench: pd.Series | None = None) -> pd.DataFrame:
    """Equity and benchmark summed over the given strategies at fine resolution for the last `days`.
    A strategy without a mark in a step carries its last value; one with no mark yet counts as its
    starting balance. prior_peak is the highest equity before the window, so drawdown is measured from
    the real peak rather than from the start of the window."""
    now = utcnow()
    step = RECENT_STEP.get(days, "30min")
    since = pd.Timestamp(now - pd.Timedelta(days=days)).floor(step)
    idx = pd.date_range(since, pd.Timestamp(now).floor(step), freq=step)
    out = pd.DataFrame(index=idx, data={"equity": base, "benchmark": base})  # the fund's own cash (fund_cash)
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
    if bench is not None:  # the fund's hold (fund_line), in place of the rows' sum
        out["benchmark"] = bench.resample(step).last().reindex(bench.resample(step).last().index.union(idx)).ffill(
        ).reindex(idx).fillna(base)
    out["equity"] = out["equity"].astype(float)
    peak = out["equity"].cummax()
    if prior_peak:
        peak = peak.clip(lower=float(prior_peak))
    out["drawdown"] = _drawdown(out["equity"], peak)
    return out


def allocation(summaries: list[dict], equity: float, fund_held: float = 0.0) -> list[dict]:
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
    cash = sum(x["cash"] for x in summaries) + fund_held  # with the fund's own cash (fund_cash)
    rows.append({"name": "Cash", "value": cash, "share": cash / equity if equity else 0.0,
                 "bar": max(0.0, 1 - sum(r["bar"] for r in rows))})
    return rows


def correlation(summaries: list[dict], frames: dict[str, pd.DataFrame]) -> dict:
    """Pairwise correlation of daily sleeve returns, where enough overlapping days exist."""
    names = [x["sleeve"].name for x in summaries if len(frames[x["sleeve"].name]) > MIN_DAYS_FOR_CORRELATION]
    if len(names) < 2:
        return {"names": names, "matrix": []}
    marks = pd.DataFrame({n: frames[n]["equity"] for n in names})
    # Inside the fund a Strategy's marks are its P&L from 0, so a percent change is blank or meaningless: its daily
    # P&L, each day's share of the same fund equity, correlates as its return would (B-20).
    rets = marks.diff() if any(x.get("in_fund") for x in summaries) else marks.pct_change()
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
