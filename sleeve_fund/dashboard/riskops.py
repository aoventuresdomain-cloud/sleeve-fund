"""Risk centre and operations views: limits across the book, stress, health and housekeeping."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sleeve_fund.store import Store, utcnow

BACKUP_STALE = timedelta(hours=26)  # nightly, with room for a slow dump

SHOCKS = (-0.10, -0.20, -0.35, -0.50)
BREACH_KINDS = ("risk_halt", "risk_pause", "reconcile_mismatch", "instrument_not_found", "tick_failed")


def risk_view(store: Store, summaries: list[dict], book: dict) -> dict:
    rows = []
    for x in summaries:
        s, p = x["sleeve"], x["profile"]
        peak = store.peak_equity(s.name) or s.starting_balance
        shocks = []
        for shock in SHOCKS:
            loss = x["position_value"] * -shock
            after = x["equity"] - loss
            dd_after = 1 - after / max(peak, x["equity"]) if peak else 0.0
            shocks.append({"loss": loss, "breach": dd_after >= p.max_drawdown})
        rows.append({
            "x": x,
            "dd_used": x["dd_used"],
            "day_used": min(max(-x["day_ret"], 0.0) / p.daily_loss, 1.0) if p.daily_loss else 0.0,
            "cap_used": min(x["exposure"] / p.max_position_pct, 1.0) if p.max_position_pct else 0.0,
            "headroom": max(p.max_drawdown - x["drawdown"], 0.0) * x["equity"],
            "has_stop": bool(s.params.get("stop_loss")),
            "shocks": shocks,
        })
    equity = book["equity"] or 1.0
    scenarios = []
    for i, shock in enumerate(SHOCKS):
        loss = sum(r["shocks"][i]["loss"] for r in rows if r["x"]["sleeve"].desired_state == "running")
        scenarios.append({"shock": shock, "loss": loss, "loss_pct": loss / equity,
                          "breaches": [r["x"]["sleeve"].name for r in rows if r["shocks"][i]["breach"]]})
    largest = max((a for a in book["allocation"] if a["name"] != "Cash"), key=lambda a: a["share"], default=None)
    return {"rows": rows, "scenarios": scenarios, "largest": largest,
            "history": store.events_of(BREACH_KINDS, limit=50)}


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
        "backup": latest_backup(Path(os.environ.get("BACKUP_DIR", "/data/backups")), now),
        "alerts": store.last_event(None, ("alerts_config",)),
    }


def latest_backup(folder: Path, now: datetime) -> dict | None:
    """The newest nightly database dump the backup service wrote, or None if there is none here."""
    try:
        newest = max(folder.glob("*.dump"), key=lambda f: f.stat().st_mtime, default=None)
    except OSError:
        return None
    if newest is None:
        return None
    st = newest.stat()
    age = now - datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    return {"name": newest.name, "bytes": st.st_size, "age": age, "stale": age > BACKUP_STALE,
            "count": len(list(folder.glob("*.dump")))}


def _meminfo() -> dict | None:
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
        total, avail = info["MemTotal"], info["MemAvailable"]
        return {"total": total, "used": total - avail, "share": (total - avail) / total}
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None
