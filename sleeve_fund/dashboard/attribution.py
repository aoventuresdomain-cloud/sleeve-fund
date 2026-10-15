"""Per-Strategy attribution (Bridge Drop A): what each Strategy made in money, its R per closed trade, its contribution
to the fund's return, and the ideas the compliance tests refused or trimmed, each with the test's name. Display only:
P&L and R are the Strategy page's own (trading.trips), the fund's base is its book at the period's start (book_marks),
the refusals are the gate's own rows."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from sleeve_fund import markets
from sleeve_fund.dashboard import compliance, trading
from sleeve_fund.risk import trading_day
from sleeve_fund.store import Store

PERIODS = {"today": "Today", "7d": "7 days", "mtd": "Month to date", "all": "Since start"}


def since(period: str, now: datetime) -> datetime | None:
    day = datetime.combine(trading_day(now), datetime.min.time(), tzinfo=now.tzinfo)
    return {"today": day, "7d": now - timedelta(days=7), "mtd": day.replace(day=1)}.get(period)


CENT8 = Decimal("1e-8")  # the journal's money precision


def _d(x) -> Decimal:
    """A money figure as an exact Decimal (F-A1-DECIMAL): a trip's float P&L to the journal's 8 dp, so 765.45 - 1,000
    is -234.55, not -234.54999999999995, and the rows add up to the cent."""
    return x if isinstance(x, Decimal) else Decimal(str(x or 0)).quantize(CENT8)


def _base(store: Store, start: datetime | None) -> Decimal | None:
    """The fund's book at the period's start, or its first mark when the period began before it; None before any."""
    base = None if start is None else store.book_equity_at_or_before(start)
    if base is None and (first := store.first_book_mark()) is not None:
        base = first["equity"]
    return None if base is None else _d(base)


def trips_of(store: Store, s) -> list[dict]:
    """A Strategy's round trips, read as its page reads them (app.sleeve_detail)."""
    perp = markets.is_perp(s.params)
    return trading.trips(store.fills(s.name, limit=100_000), store.events(s.name, limit=5000),
                         trading.orders_by_id(store, s.name), store.exit_plans(s.name), perp,
                         store.funding(s.name, limit=None) if perp else None,
                         store.insurance(s.name, limit=None) if perp else None)


def strategy(store: Store, name: str, trips: list[dict], now: datetime, period: str = "all") -> dict:
    """{period, label, pnl, fees, funding, trades, r_mean, r_trades, fund_start, contribution, refused: [...]}.

    pnl: the P&L of the trades closed in the period, after fees and funding (trading.trips); an open position's
    unrealised P&L is the Strategy page's own figure and isn't counted here, so a period's trades add up to it.
    r_mean: the mean R of the closed trades that carry one (sized from a stop). contribution: pnl over the fund's book
    at the period's start (or its first mark), None before the book's first mark."""
    start = since(period, now)
    closed = [t for t in trips if t.get("closed") and (start is None or t["closed"] >= start)]
    pnl = sum((_d(t["pnl"]) for t in closed), Decimal(0))  # F-A1-DECIMAL: money adds up to the cent
    rs = [float(t["r"]) for t in closed if t.get("r") is not None]
    base = _base(store, start)
    refused = compliance.refused(store, name, start)  # C6: one reader for this tab and Portfolio Settings
    return {"period": period, "label": PERIODS[period], "pnl": pnl,
            "fees": sum((_d(t.get("fees")) for t in closed), Decimal(0)),
            "funding": sum((_d(t.get("funding")) for t in closed), Decimal(0)),
            "trades": len(closed), "r_mean": sum(rs) / len(rs) if rs else None, "r_trades": len(rs),
            "fund_start": base, "contribution": pnl / base if base else None, "refused": refused,
            "periods": PERIODS}


def _book_now(store: Store) -> Decimal | None:
    """The fund's book at its last mark, as the gate holds it (portfolio_state), or None before the first mark or
    when it can't be read: then there is no total to show, never a made-up one."""
    from sleeve_fund.gate_ledger import DbLedger

    try:
        return DbLedger(store, lambda: ()).state().equity
    except Exception as exc:  # noqa: BLE001 - an unread book shows no total, said so on the page
        print(f"attribution: can't read the fund's book: {exc!r}")
        return None


def fund(store: Store, sleeves, now: datetime, period: str = "all") -> dict:
    """F-A1-FUNDLINE: the fund's attribution on Portfolio. {rows: [{name, pnl, contribution, trades}], fund: {pnl,
    contribution}, total: {pnl, contribution}, fund_start, label}. total is the fund's book at its last mark less
    its book at the period's start; the fund line is what no Strategy's closed trades carry (open positions, costs
    and cash outside a Strategy), so the rows plus the fund line are the total exactly. Contributions are None, and
    so is everything but the rows' P&L, before the book's first mark. The total runs to the last mark, not to `now`,
    because the rows have no upper bound either (every trade closed since the start): a period that ends before the
    last mark must bound both the rows and the total (HoE 15 Oct)."""
    rows: dict[str, dict] = {}
    names = {s.name for s in sleeves}
    # A two-leg Strategy is one row under its first leg's name: the second leg ("<name>~2", parent set) adds into it.
    for s in sorted(sleeves, key=lambda s: bool(getattr(s, "parent", None))):
        a = strategy(store, s.name, trips_of(store, s), now, period)
        parent = getattr(s, "parent", None)
        if parent in names and parent in rows:
            r = rows[parent]
            r["pnl"] += a["pnl"]
            r["trades"] += a["trades"]
            if r["contribution"] is not None and a["contribution"] is not None:
                r["contribution"] += a["contribution"]
            continue
        rows[s.name] = {"name": s.name, "pnl": a["pnl"], "contribution": a["contribution"], "trades": a["trades"]}
    rows = list(rows.values())
    base = _base(store, since(period, now))
    book = _book_now(store)
    out = {"rows": rows, "fund_start": base, "label": PERIODS[period], "fund": None, "total": None}
    if base and book is not None:
        total_pnl = _d(book) - base
        total = {"pnl": total_pnl, "contribution": total_pnl / base}
        out["total"] = total
        out["fund"] = {"pnl": total_pnl - sum((r["pnl"] for r in rows), Decimal(0)),
                       "contribution": total["contribution"] - sum((r["contribution"] for r in rows), Decimal(0))}
    return out
