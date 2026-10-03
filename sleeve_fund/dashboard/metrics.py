"""Numbers the dashboard shows, computed from the journal."""

from __future__ import annotations

from datetime import timedelta

from sleeve_fund.risk import profile as risk_profile
from sleeve_fund.store import Sleeve, Store, utcnow

STALE = timedelta(minutes=3)


def sleeve_summary(store: Store, s: Sleeve) -> dict:
    series = store.equity_series(s.name, limit=100_000)
    prof = risk_profile(s.risk_profile)
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
        "fills": len(store.fills(s.name, limit=10_000)),
        "fees": sum(f["fee"] for f in store.fills(s.name, limit=10_000)),
        "healthy": bool(s.heartbeat_at and utcnow() - s.heartbeat_at < STALE),
    }
    if series:
        last = series[-1]
        peak, mdd = 0.0, 0.0
        for p in series:
            peak = max(peak, p["equity"])
            mdd = max(mdd, 1 - p["equity"] / peak)
        out.update(
            equity=last["equity"],
            benchmark=last["benchmark"],
            ret=last["equity"] / s.starting_balance - 1,
            bench_ret=last["benchmark"] / s.starting_balance - 1,
            drawdown=1 - last["equity"] / peak,
            max_drawdown=mdd,
            exposure=(last["qty"] * last["price"]) / last["equity"] if last["equity"] else 0.0,
        )
        out["dd_used"] = min(out["drawdown"] / prof.max_drawdown, 1.0)
    return out


def portfolio_summary(summaries: list[dict]) -> dict:
    active = [x for x in summaries if x["sleeve"].desired_state == "running"]
    start = sum(x["sleeve"].starting_balance for x in active)
    equity = sum(x["equity"] for x in active)
    bench = sum(x["benchmark"] for x in active)
    return {
        "sleeves": len(summaries),
        "running": sum(1 for x in summaries if x["sleeve"].status == "running"),
        "attention": sum(1 for x in summaries if x["sleeve"].status in ("halted", "error")),
        "equity": equity,
        "start": start,
        "ret": equity / start - 1 if start else 0.0,
        "bench_ret": bench / start - 1 if start else 0.0,
        "unhealthy": sum(1 for x in active if not x["healthy"]),
    }
