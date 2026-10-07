"""Risk centre and operations views: limits across the book, stress, health and housekeeping."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from sleeve_fund import backups, markets
from sleeve_fund.dashboard import trading
from sleeve_fund.store import Store, utcnow
from sleeve_fund.venues import venue as venue_profile

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
        shocks, at_stake = [], most_it_can_lose(x)
        for shock in SHOCKS:
            # Signed: a long loses in a fall, a short in a rally, at most what it has at stake (most_it_can_lose).
            loss = min(x["position_value"] * -shock, at_stake)
            after = x["equity"] - loss
            dd_after = 1 - after / max(peak, x["equity"]) if peak else 0.0
            # A strategy with nothing left (wiped out) can't breach again: it is already halted (m12 fix re-check, mF-3).
            shocks.append({"loss": loss, "breach": x["equity"] > 0 and dd_after >= p.max_drawdown})
        rows.append({
            "x": x,
            "dd_used": x["dd_used"],
            "day_used": min(max(-x["day_ret"], 0.0) / p.daily_loss, 1.0) if p.daily_loss else 0.0,
            # Not capped at 100%: drift past the entry cap shows its true share, in amber (P1-U13).
            "cap_used": abs(x["exposure"]) / cap if (cap := x.get("cap", p.max_position_pct)) else 0.0,
            "headroom": x["room"],
            # An open position's own stop (an exit-plan edit moves it), else the model's setting (P1-U14).
            "has_stop": bool(pos["stop_px"]) if (pos := positions.get(s.name)) else bool(
                s.params.get("stop_loss") or s.params.get("stop_atr") or s.params.get("stop_swing_bars")),
            "stop_dist": abs(pos["stop_px"] / pos["entry_px"] - 1) if pos and pos["stop_px"] and pos["entry_px"] else None,
            "position": pos,
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
            **{k: held[k] for k in ("margin", "open_risk", "estimated", "through", "left_out", "trailing", "hint")},
            "down20": next(sc for sc in scenarios if sc["shock"] == -0.20),
            "up20": next(sc for sc in scenarios if sc["shock"] == 0.20),
            "history": store.events_of(BREACH_KINDS, limit=50)}


def most_it_can_lose(x: dict) -> float:
    """The most a strategy's open position can lose from the mark, however far the market moves. A perpetual
    on isolated margin is liquidated before it loses more than its margin (notional at entry over the profile's
    leverage cap, markets.isolated_margin) and what it has made or lost since entry, plus the fee on the
    liquidation; the rest of its equity is not at risk (m13-U6). Anything else can lose at most its equity."""
    equity, s, qty = max(x["equity"], 0.0), x["sleeve"], x.get("qty") or 0.0
    t = markets.terms(s.params, s.venue) if qty else None
    if t is None:
        return equity
    entry, lev = x.get("entry_px") or x["price"], x["profile"].max_leverage
    liq = markets.isolated_liquidation(x["cash"], qty, entry, lev, t.maintenance_margin)
    taker = float(markets.fees_for(s.params, venue_profile(s.venue).fees, s.venue).taker)
    cap = markets.gap_loss_cap(qty, entry, lev, x["cash"] + qty * entry, taker, liq)  # what the engine books (P1-D3)
    return min(max(cap + x.get("unrealised", 0.0), 0.0), equity)


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
        # The limits that halt or pause are judged on their own; drift past the entry cap gets its own line, so
        # it never hides a drawdown or daily-loss warning (P1-U13, Code Reviewer on #152).
        used = [(r["dd_used"], "drawdown"), (r["day_used"], "daily loss")]
        share, which = max(used, key=lambda u: u[0])
        if share > NEAR_LIMIT:
            warn.append(f"{s.name} has used {share:.0%} of its {which} limit")
        if r["cap_used"] > 1:
            # The cap limits new entries only: past it through price drift is information, not a call to act.
            warn.append(f"{s.name}'s exposure is {r['cap_used']:.0%} of its entry cap because the price moved; no action")
        elif r["cap_used"] > NEAR_LIMIT:
            warn.append(f"{s.name} has used {r['cap_used']:.0%} of its position cap")
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


def drawdown_chart(curve, days: int = 30, halt: float | None = None) -> dict:
    """The book's drawdown over the last `days` daily closes for the Lightweight Charts area (QA U10): a UTC
    timestamp per close and the drawdown in percent below the peak (negative), the current value and the worst
    in the window as shares, and the halt line's level when the book shares one."""
    dd = curve["drawdown"] if len(curve) else []
    if len(dd):
        dd = dd[dd.index >= dd.index[-1] - pd.Timedelta(days=days - 1)]
    values = [max(float(v), 0.0) for v in dd]
    return {"t": [int(pd.Timestamp(t).timestamp()) for t in (dd.index if len(dd) else [])],
            "dd": [round(-v * 100, 4) for v in values], "halt": halt,
            "current": values[-1] if values else 0.0, "worst": max(values) if values else 0.0, "points": len(values)}


def stress_bars(scenarios: list[dict]) -> dict:
    """The overview's "If the market moved now": the ±10% and ±20% scenarios as diverging bars, each bar's
    half-width scaled to the largest move shown, and which strategies would halt in each."""
    shown = [sc for sc in scenarios if any(abs(sc["shock"] - s) < 1e-9 for s in STRESS_SHOWN)]
    biggest = max((abs(sc["pnl"]) for sc in shown), default=0.0)
    bars = [{**sc, "width": (abs(sc["pnl"]) / biggest * 46 if biggest else 0.0)} for sc in shown]
    return {"bars": bars, "halts": [sc for sc in shown if sc["breaches"]]}
