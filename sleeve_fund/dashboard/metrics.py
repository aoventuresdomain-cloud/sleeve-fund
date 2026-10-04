"""Numbers the dashboard shows, computed from the journal."""

from __future__ import annotations

from datetime import timedelta

from sleeve_fund.research.metrics import trade_stats, trades
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
        "equity": s.starting_balance,
        "benchmark": s.starting_balance,
        "ret": 0.0,
        "bench_ret": 0.0,
        "drawdown": 0.0,
        "max_drawdown": 0.0,
        "dd_used": 0.0,
        "exposure": 0.0,
        "points": len(series),
        "fills": len(fills),
        "fees": sum(f["fee"] for f in fills),
        "pnl": 0.0,
        "trades": trade_stats(trades(list(reversed(fills)))),  # closed trips, after fees
        "healthy": bool(s.heartbeat_at and utcnow() - s.heartbeat_at < STALE),
    }
    if series:
        last = series[-1]
        peak = store.peak_equity(s.name) or max(p["equity"] for p in series)
        mdd = store.max_drawdown(s.name)  # over every mark, not only the latest ones read here
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
    return out
