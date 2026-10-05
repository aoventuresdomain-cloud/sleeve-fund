"""Risk centre and operations views: limits across the book, stress, health and housekeeping."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from sleeve_fund import backups
from sleeve_fund.dashboard import trading
from sleeve_fund.store import Store, utcnow

FEED_FRESH_SECONDS = 60  # past this a strategy's price feed reads as stale, here and on its page

# Market moves the stress table applies to every position at once: falls and rallies, so a short book's risk
# (a rally) shows as plainly as a long book's.
SHOCKS = (-0.50, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.50)
BREACH_KINDS = ("risk_halt", "risk_pause", "reconcile_mismatch", "instrument_not_found", "tick_failed", "liquidation",
                "liquidation_cut", "insurance_fund")


def risk_view(store: Store, summaries: list[dict], book: dict) -> dict:
    rows = []
    # Each open position with the same margin, liquidation and Risk to stop as Portfolio and its strategy page.
    held = trading.book_positions(store, summaries)
    positions = {r["sleeve"]: r for r in held["rows"]}
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
            # A strategy with nothing left (wiped out) can't breach again: it is already halted (m12 fix re-check, mF-3).
            shocks.append({"loss": loss, "breach": x["equity"] > 0 and dd_after >= p.max_drawdown})
        rows.append({
            "x": x,
            "dd_used": x["dd_used"],
            "day_used": min(max(-x["day_ret"], 0.0) / p.daily_loss, 1.0) if p.daily_loss else 0.0,
            "cap_used": min(abs(x["exposure"]) / cap, 1.0) if (cap := x.get("cap", p.max_position_pct)) else 0.0,
            "headroom": x["room"],
            "has_stop": bool(s.params.get("stop_loss") or s.params.get("stop_atr") or s.params.get("stop_swing_bars")),
            "position": positions.get(s.name),
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
            "margin": held["margin"], "open_risk": held["open_risk"], "unbounded": held["unbounded"],
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


# --- Risk & health overview: one word for the whole desk, the health pills under it ---------------------------

# The price watchdog's events, with the events that start a process afresh (a restart clears what it noted).
FEED_KINDS = ("stale_price", "feed_dead", "price_feed_back", "start", "restart", "restore")
STALE_FEED = ("stale_price", "feed_dead")
NEAR_LIMIT = 0.8  # share of a limit used past which the desk needs a look
STRESS_SHOWN = (-0.20, -0.10, 0.10, 0.20)  # the overview's chart; the Limits tab keeps every scenario


def feed_fresh(store: Store, x: dict) -> bool:
    """Whether a strategy's price feed is fresh: trades or quotes from its venue are arriving.

    The process is reporting, and its venue's latest trade or quote (Store.last_feed) is at most a minute
    old, the same line as the strategy page's feed badge. Before its first trade arrives, the price
    watchdog's latest word since it started decides: not a stale-price warning."""
    if not x["healthy"]:
        return False
    seen = store.last_feed(x["sleeve"].name)
    if seen is not None:
        return (utcnow() - seen).total_seconds() <= FEED_FRESH_SECONDS
    last = store.last_event(x["sleeve"].name, FEED_KINDS)
    return not (last and last["kind"] in STALE_FEED)


def health_view(store: Store, summaries: list[dict], backup_dir: Path | None = None) -> dict:
    """The health pills: processes and feeds of the strategies meant to be running, balance checks, the
    latest backup, outside alerts and open alerts."""
    now = utcnow()
    wanted = [x for x in summaries if x["sleeve"].desired_state == "running"]
    down = [x["sleeve"].name for x in wanted if not x["healthy"]]
    stale = [x["sleeve"].name for x in wanted if x["healthy"] and not feed_fresh(store, x)]
    checked = [x for x in summaries if x.get("reconcile")]
    mismatched = [x["sleeve"].name for x in checked if x["reconcile"]["kind"] == "reconcile_mismatch"]
    folder = backup_dir or Path(os.environ.get("BACKUP_DIR", "/data/backups"))
    backup = backups.latest(folder, now)
    issue = backups.problem(folder, now)
    alerts_cfg = store.last_event(None, ("alerts_config",))
    return {
        "wanted": len(wanted),
        "processes_ok": len(wanted) - len(down),
        "down": down,
        "feeds_ok": len(wanted) - len(down) - len(stale),
        "stale_feeds": stale,
        "checked": len(checked),
        "mismatched": mismatched,
        "backup_age": backup["age"] if backup else None,
        "backup_issue": issue[1] if issue else None,
        "outside_alerts": None if alerts_cfg is None else "not set up" not in alerts_cfg["message"],
        "alerts_open": store.open_alert_count(),
    }


def status_items(rows: list[dict], health: dict) -> list[dict]:
    """What keeps the desk from All clear, worst first: each item's level ("bad" or "warn") and a line
    that names it. Bad: a strategy halted or in error, a process not reporting, a balance mismatch.
    Warn: a strategy paused, a limit more than 80% used, a stale price feed, a backup problem."""
    bad, warn = [], []
    for r in rows:
        s = r["x"]["sleeve"]
        if s.status in ("halted", "error"):
            bad.append(f"{s.name} is {s.status}" + (f": {s.status_reason}" if s.status_reason else ""))
        elif s.status == "paused":
            warn.append(f"{s.name} is paused" + (f": {s.status_reason}" if s.status_reason else ""))
    bad += [f"{name} is not reporting" for name in health["down"]]
    bad += [f"{name}'s balance doesn't match the venue's" for name in health["mismatched"]]
    for r in rows:
        s = r["x"]["sleeve"]
        if s.status in ("halted", "error"):
            continue  # already named, worse
        used = [(r["dd_used"], "drawdown"), (r["day_used"], "daily loss"), (r["cap_used"], "position cap")]
        share, which = max(used, key=lambda u: u[0])
        if share > NEAR_LIMIT:
            warn.append(f"{s.name} has used {share:.0%} of its {which} limit")
    warn += [f"{name} has had no trade or quote from the venue lately" for name in health["stale_feeds"]]
    if health["backup_issue"]:
        warn.append(health["backup_issue"])
    return [{"level": "bad", "text": t} for t in bad] + [{"level": "warn", "text": t} for t in warn]


def status_word(items: list[dict], running: int) -> dict:
    """The one word at the top of Risk & health and its line: All clear (green) when nothing is in
    `items`, otherwise Action needed (red) when any item is bad, or Needs a look (amber), naming the
    first item and counting the rest."""
    if not items:
        who = f"{running} strateg{'y' if running == 1 else 'ies'} running" if running else "No strategy running"
        return {"word": "All clear", "tone": "ok", "issues": items,
                "line": f"{who}. Every limit has room. Nothing needs you."}
    first = items[0]["text"]
    more = len(items) - 1
    line = f"{first}." + (f" And {more} more below." if more else "")
    if any(i["level"] == "bad" for i in items):
        return {"word": "Action needed", "tone": "bad", "issues": items, "line": line}
    return {"word": "Needs a look", "tone": "warn", "issues": items, "line": line}


def drawdown_chart(curve, days: int = 30, halt: float | None = None, w: float = 420, h: float = 110) -> dict:
    """The book's drawdown over the last `days` daily closes as SVG geometry: the line and its area, the
    current value and the worst in the window, and the y axis (0% at the top, deeper further down)."""
    dd = curve["drawdown"] if len(curve) else []
    if len(dd):
        dd = dd[dd.index >= dd.index[-1] - pd.Timedelta(days=days - 1)]
    values = [max(float(v), 0.0) for v in dd]
    worst = max(values) if values else 0.0
    current = values[-1] if values else 0.0
    left, top, bottom = 34.0, 10.0, h - 14
    scale = max(worst * 1.25, halt or 0.0, 0.01)
    y = lambda v: top + v / scale * (bottom - top)  # noqa: E731
    step = (w - 6 - left) / (len(values) - 1) if len(values) > 1 else 0.0
    pts = [(left + i * step, y(v)) for i, v in enumerate(values)]
    line = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    area = (f"M{left:.1f},{top:.1f} " + " ".join(f"L{px:.1f},{py:.1f}" for px, py in pts)
            + f" L{pts[-1][0]:.1f},{top:.1f} Z") if len(pts) > 1 else ""
    ticks = [{"y": y(f * scale), "label": f"{f * scale:.0%}" if scale >= 0.05 else f"{f * scale:.1%}"} for f in (0.0, 0.5, 1.0)]
    return {"line": line, "area": area, "dot": pts[-1] if pts else None, "ticks": ticks, "w": w, "h": h, "left": left,
            "halt": halt, "halt_y": y(halt) if halt else None, "current": current, "worst": worst, "points": len(values)}


def stress_bars(scenarios: list[dict]) -> dict:
    """The overview's "If the market moved now": the ±10% and ±20% scenarios as diverging bars, each bar's
    half-width scaled to the largest move shown, and which strategies would halt in each."""
    shown = [sc for sc in scenarios if any(abs(sc["shock"] - s) < 1e-9 for s in STRESS_SHOWN)]
    biggest = max((abs(sc["pnl"]) for sc in shown), default=0.0)
    bars = [{**sc, "width": (abs(sc["pnl"]) / biggest * 46 if biggest else 0.0)} for sc in shown]
    return {"bars": bars, "halts": [sc for sc in shown if sc["breaches"]]}
