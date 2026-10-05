"""Numbers the dashboard shows, computed from the journal."""

from __future__ import annotations

from datetime import timedelta

from sleeve_fund import markets
from sleeve_fund.research.metrics import trade_stats, trades
from sleeve_fund.risk import position_cap
from sleeve_fund.risk import profile as risk_profile
from sleeve_fund.store import Sleeve, Store, utcnow

STALE = timedelta(minutes=3)


def sleeve_summary(store: Store, s: Sleeve) -> dict:
    series = store.equity_series(s.name, limit=100_000)
    prof = risk_profile(s.risk_profile)
    fills = store.fills(s.name, limit=10_000)
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
        "exposure": 0.0,
        "points": len(series),
        "fills": len(fills),
        "fees": sum(f["fee"] for f in fills),
        "pnl": 0.0,
        # Closed trips after fees (and a perpetual's funding), paired as the Trades tab pairs them: on a perpetual a sell from flat opens a
        # short, so the header, the Trades tab and the G2 checklist count the same trips (round 11, M11-4).
        "trades": trade_stats(trades(list(reversed(fills)), markets.is_perp(s.params),
                                     store.funding(s.name, limit=100_000) if markets.is_perp(s.params) else None,
                                     store.insurance(s.name) if markets.is_perp(s.params) else None)),
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
            ret=last["equity"] / s.starting_balance - 1,
            pnl=last["equity"] - s.starting_balance,
            bench_ret=last["benchmark"] / s.starting_balance - 1,
            drawdown=1 - last["equity"] / peak,
            max_drawdown=mdd,
            exposure=(last["qty"] * last["price"]) / last["equity"] if last["equity"] else 0.0,
        )
        out["dd_used"] = min(out["drawdown"] / prof.max_drawdown, 1.0)
        # In money from the peak, as the halt and the stop check measure it, not the drawdown gap times
        # today's equity, which understates it by equity / peak (U13-3).
        out["room"] = max(last["equity"] - peak * (1 - prof.max_drawdown), 0.0)
    out.update(costs(out["pnl"], out["fees"], out["funding"]))
    return out


def costs(pnl: float, fees: float, funding: float) -> dict:
    """Fees and funding paid, the P&L before them, and the share of that gross P&L they took (UI v2, item 7).
    pnl is after both; funding is + received, - paid, so funding received lowers the cost. The share is of the
    gross P&L's size, so a loss's costs read as a share too; None while there is no gross P&L."""
    paid = fees - funding
    gross = pnl + paid
    return {"costs": paid, "gross_pnl": gross, "cost_share": paid / abs(gross) if gross else None}
