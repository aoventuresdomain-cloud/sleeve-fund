"""Numbers the dashboard shows, computed from the journal."""

from __future__ import annotations

from datetime import timedelta

from sleeve_fund import markets
from sleeve_fund.research.metrics import trade_stats, trades
from sleeve_fund.risk import position_cap
from sleeve_fund.risk import profile as risk_profile
from sleeve_fund.store import Sleeve, Store, utcnow

STALE = timedelta(minutes=3)


def held_by_fund(store: Store, s: Sleeve) -> bool:
    """B-20: whether the fund holds this Strategy's cash (the PM's re-base deposit applied, supervisor.bridge_start, and
    the Strategy in the fund's book). Then it has no capital of its own (CUT-A (a)), its own drawdown halt and day pause
    don't act (the fund's brakes do, CUT-A (d)), and its marks are its P&L from 0, so a drawdown, a room to halt or a
    return on its own equity mean nothing. A test-account Strategy or a backtest is outside the fund's book."""
    from sleeve_fund.store import is_backtest
    from sleeve_fund.supervisor import bridge_start, outside_fund_book

    return bridge_start(store) is not None and not is_backtest(s.name) and not outside_fund_book(store, s)


def sleeve_summary(store: Store, s: Sleeve) -> dict:
    series = store.equity_series(s.name, limit=100_000)
    in_fund = held_by_fund(store, s)
    prof = risk_profile(s.risk_profile)
    fills = store.fills(s.name, limit=10_000)  # the trades below; the counts and fees are over every fill (CASH-2)
    n_fills, fees = store.fill_totals(s.name)
    out = {
        "sleeve": s,
        "profile": prof,
        "cap": position_cap(prof, s.params),  # on a perpetual, the margin cap times the leverage cap
        "equity": s.starting_balance,
        "benchmark": s.starting_balance,
        "ret": 0.0,
        "bench_ret": 0.0,
        "drawdown": 0.0,
        "max_drawdown": 0.0,
        "dd_used": 0.0,
        "room": s.starting_balance * prof.max_drawdown,  # what it can lose before the drawdown halt
        "in_fund": in_fund,  # B-20: the fund holds its cash; its own halt, room and drawdown are not shown
        "exposure": 0.0,
        "points": len(series),
        "fills": n_fills,
        "fees": fees,
        "pnl": 0.0,
        # Closed trips after fees (and a perpetual's funding), paired as the Trades tab pairs them: on a perpetual a sell from flat opens a
        # short, so the header, the Trades tab and the G2 checklist count the same trips (round 11, M11-4).
        "trades": trade_stats(trades(list(reversed(fills)), markets.is_perp(s.params),
                                     store.funding(s.name, limit=None) if markets.is_perp(s.params) else None,
                                     store.insurance(s.name, limit=None) if markets.is_perp(s.params) else None)),
        "healthy": bool(s.heartbeat_at and utcnow() - s.heartbeat_at < STALE),
        # Funding settled on a perpetual: + received, - paid. Already in its equity and P&L.
        "funding": store.funding_total(s.name) if markets.is_perp(s.params) else 0.0,
    }
    if series:
        last = series[-1]
        # From the starting balance too, which a first mark that is already a loss never reaches (M12-F1).
        peak = max(store.peak_equity(s.name) or max(p["equity"] for p in series), s.starting_balance)
        mdd = store.max_drawdown(s.name, s.starting_balance)  # over every mark, not only the latest ones read here
        out.update(
            equity=last["equity"],
            benchmark=last["benchmark"],
            # A Strategy with no capital of its own (the fund holds the cash, CUT-A) has no return on it: 0, never inf.
            ret=last["equity"] / s.starting_balance - 1 if s.starting_balance > 0 else 0.0,
            pnl=last["equity"] - s.starting_balance,
            bench_ret=last["benchmark"] / s.starting_balance - 1 if s.starting_balance > 0 else 0.0,
            drawdown=1 - last["equity"] / peak if peak > 0 else 0.0,
            max_drawdown=mdd,
            exposure=(last["qty"] * last["price"]) / last["equity"] if last["equity"] else 0.0,
        )
        out["dd_used"] = min(out["drawdown"] / prof.max_drawdown, 1.0)
        # In money from the peak, as the halt and the stop check measure it, not the drawdown gap times
        # today's equity, which understates it by equity / peak (U13-3).
        out["room"] = max(last["equity"] - peak * (1 - prof.max_drawdown), 0.0)
    if in_fund:
        # B-20: no drawdown, room or exposure on a P&L that starts at 0; exposure is over the fund's equity, the base
        # its entries are sized from (_sizing_base) and T4 measures against.
        out.update(drawdown=0.0, max_drawdown=0.0, dd_used=0.0, room=None, exposure=0.0)
        if series and series[-1]["qty"]:
            fund = fund_equity_or_none(store)
            out["exposure"] = (series[-1]["qty"] * series[-1]["price"]) / fund if fund else 0.0
    out.update(costs(out["pnl"], out["fees"], out["funding"]))
    return out


def fund_equity_or_none(store: Store) -> float | None:
    """supervisor.fund_equity as a float, or None when it can't be marked (a position with no mark yet raises there,
    which a page must never do)."""
    from sleeve_fund.supervisor import fund_equity

    try:
        return float(fund_equity(store))
    except Exception:  # noqa: BLE001 - a page shows the figure as not known; the gate refuses on its own reading
        return None


def costs(pnl: float, fees: float, funding: float) -> dict:
    """Fees and funding paid, the P&L before them, and the share of that gross P&L they took (UI v2, item 7).
    pnl is after both; funding is + received, - paid, so funding received lowers the cost. The share is of the
    gross P&L's size, so a loss's costs read as a share too; None while there is no gross P&L."""
    paid = float(fees) - float(funding)  # fees and funding are exact Decimals from the journal; this is display
    gross = pnl + paid
    return {"costs": paid, "gross_pnl": gross, "cost_share": paid / abs(gross) if gross else None}
