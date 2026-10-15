"""Per-Strategy attribution (Bridge Drop A): what each Strategy made in money, its R per closed trade, its contribution
to the fund's return, and the ideas the compliance tests refused or trimmed, each with the test's name. Display only:
P&L and R are the Strategy page's own (trading.trips), the fund's base is its book at the period's start (book_marks),
the refusals are the gate's own rows."""

from __future__ import annotations

from datetime import datetime, timedelta

from sleeve_fund import compliance_rules as rules
from sleeve_fund.risk import trading_day
from sleeve_fund.store import Store

PERIODS = {"today": "Today", "7d": "7 days", "mtd": "Month to date", "all": "Since start"}


def since(period: str, now: datetime) -> datetime | None:
    day = datetime.combine(trading_day(now), datetime.min.time(), tzinfo=now.tzinfo)
    return {"today": day, "7d": now - timedelta(days=7), "mtd": day.replace(day=1)}.get(period)


def strategy(store: Store, name: str, trips: list[dict], now: datetime, period: str = "all") -> dict:
    """{period, label, pnl, fees, funding, trades, r_mean, r_trades, fund_start, contribution, refused: [...]}.

    pnl: the P&L of the trades closed in the period, after fees and funding (trading.trips); an open position's
    unrealised P&L is the Strategy page's own figure and isn't counted here, so a period's trades add up to it.
    r_mean: the mean R of the closed trades that carry one (sized from a stop). contribution: pnl over the fund's book
    at the period's start (or its first mark), None before the book's first mark."""
    start = since(period, now)
    closed = [t for t in trips if t.get("closed") and (start is None or t["closed"] >= start)]
    pnl = sum(float(t["pnl"]) for t in closed)
    rs = [float(t["r"]) for t in closed if t.get("r") is not None]
    if start is None:
        first = store.first_book_mark()
        base = first["equity"] if first else None
    else:
        base = store.book_equity_at_or_before(start)
        if base is None and (first := store.first_book_mark()) is not None:
            base = first["equity"]
    base = float(base) if base is not None else None
    refused = [{**r, "test": rules.refused_by(r.get("test_hit") or r.get("limit_hit"))}
               for r in store.refused_ideas(name, start)]
    return {"period": period, "label": PERIODS[period], "pnl": pnl,
            "fees": sum(float(t.get("fees", 0.0)) for t in closed),
            "funding": sum(float(t.get("funding", 0.0) or 0.0) for t in closed),
            "trades": len(closed), "r_mean": sum(rs) / len(rs) if rs else None, "r_trades": len(rs),
            "fund_start": base, "contribution": pnl / base if base else None, "refused": refused,
            "periods": PERIODS}
