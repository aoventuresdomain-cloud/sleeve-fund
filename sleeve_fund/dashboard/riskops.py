"""Risk centre and operations views: limits across the book, stress, health and housekeeping."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from sleeve_fund import backups
from sleeve_fund.store import Store, utcnow

# Market moves the stress table applies to every position at once: falls and rallies, so a short book's risk
# (a rally) shows as plainly as a long book's.
SHOCKS = (-0.50, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.50)
BREACH_KINDS = ("risk_halt", "risk_pause", "reconcile_mismatch", "instrument_not_found", "tick_failed", "liquidation",
                "liquidation_cut", "insurance_fund")


def risk_view(store: Store, summaries: list[dict], book: dict) -> dict:
    rows = []
    for x in summaries:
        s, p = x["sleeve"], x["profile"]
        peak = store.peak_equity(s.name) or s.starting_balance
        shocks = []
        for shock in SHOCKS:
            # Signed: a long loses in a fall, a short in a rally. An isolated-margin strategy is liquidated
            # before it loses more than its equity, so that is the most it can lose.
            loss = min(x["position_value"] * -shock, max(x["equity"], 0.0))
            after = x["equity"] - loss
            dd_after = 1 - after / max(peak, x["equity"]) if peak else 0.0
            shocks.append({"loss": loss, "breach": dd_after >= p.max_drawdown})
        rows.append({
            "x": x,
            "dd_used": x["dd_used"],
            "day_used": min(max(-x["day_ret"], 0.0) / p.daily_loss, 1.0) if p.daily_loss else 0.0,
            "cap_used": min(abs(x["exposure"]) / cap, 1.0) if (cap := x.get("cap", p.max_position_pct)) else 0.0,
            "headroom": max(p.max_drawdown - x["drawdown"], 0.0) * x["equity"],
            "has_stop": bool(s.params.get("stop_loss") or s.params.get("stop_atr") or s.params.get("stop_swing_bars")),
            "shocks": shocks,
        })
    equity = book["equity"] or 1.0
    scenarios = []
    for i, shock in enumerate(SHOCKS):
        loss = sum(r["shocks"][i]["loss"] for r in rows if r["x"]["sleeve"].desired_state == "running")
        scenarios.append({"shock": shock, "loss": loss, "pnl": -loss, "loss_pct": loss / equity,
                          "breaches": [r["x"]["sleeve"].name for r in rows if r["shocks"][i]["breach"]]})
    largest = largest_asset(book["allocation"], book["equity"])
    return {"rows": rows, "scenarios": scenarios, "largest": largest,
            "down20": next(sc for sc in scenarios if sc["shock"] == -0.20),
            "up20": next(sc for sc in scenarios if sc["shock"] == 0.20),
            "history": store.events_of(BREACH_KINDS, limit=50)}


def largest_asset(allocation: list[dict], equity: float) -> dict | None:
    """The book's biggest concentration in one asset, by gross exposure: its long rows and its short rows
    ("BTC short") together, both counted, with the net beside it. Taking the largest signed share read a
    short's negative share as nothing (round 12, M12-U2)."""
    assets: dict[str, dict] = {}
    for a in allocation:
        if a["name"] == "Cash":
            continue
        base = a["name"].removesuffix(" short")
        row = assets.setdefault(base, {"name": base, "gross": 0.0, "net": 0.0})
        row["gross"] += abs(a["value"])
        row["net"] += a["value"]
    if not assets:
        return None
    top = max(assets.values(), key=lambda r: r["gross"])
    return {**top, "share": top["gross"] / equity if equity else 0.0,
            "net_share": top["net"] / equity if equity else 0.0}


def ops_view(store: Store, summaries: list[dict]) -> dict:
    now = utcnow()
    procs = []
    for x in summaries:
        s = x["sleeve"]
        last = store.last_equity(s.name)
        procs.append({
            "x": x,
            "mark_age": (now - last["ts"]) if last else None,
            "restart": store.last_event(s.name, ("start", "restart", "restore")),
            "pending": len(store.pending_commands(s.name)),
            "errors": len(store.events(s.name, limit=200, min_level="error")),
        })
    disk = shutil.disk_usage("/")
    load = os.getloadavg() if hasattr(os, "getloadavg") else (float("nan"),) * 3
    return {
        "procs": procs,
        "tables": store.table_sizes(),
        "db_bytes": store.database_bytes(),
        "disk": {"used": disk.used, "total": disk.total, "share": disk.used / disk.total if disk.total else 0.0},
        "memory": _meminfo(),
        "load": load,
        "cpus": os.cpu_count() or 1,
        "backup": backups.latest(Path(os.environ.get("BACKUP_DIR", "/data/backups")), now),
        "alerts": store.last_event(None, ("alerts_config",)),
    }


def _meminfo() -> dict | None:
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
        total, avail = info["MemTotal"], info["MemAvailable"]
        return {"total": total, "used": total - avail, "share": (total - avail) / total}
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None
