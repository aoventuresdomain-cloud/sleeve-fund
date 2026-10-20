"""Risk centre and operations views: limits across the book, stress, health and housekeeping."""

from __future__ import annotations

import os
import shutil
from decimal import Decimal
from pathlib import Path

import pandas as pd

from sleeve_fund import backups, markets, risk
from sleeve_fund.dashboard import legs, trading
from sleeve_fund.dashboard.legs import label
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
        in_fund = bool(x.get("in_fund"))
        # Inside the fund its own equity is its P&L from 0 and its own halt doesn't act (B-20): what it has at stake
        # is bounded by the fund's equity, and only the fund's brakes halt.
        shocks, at_stake = [], most_it_can_lose(x, book["equity"]) if in_fund else most_it_can_lose(x)
        perp_held = bool(x.get("qty")) and markets.is_perp(s.params)
        for shock in SHOCKS:
            # Signed: a long loses in a fall, a short in a rally, at most what it has at stake (most_it_can_lose).
            raw = x["position_value"] * -shock
            loss = min(raw, at_stake)
            after = x["equity"] - loss
            dd_after = 1 - after / max(peak, x["equity"]) if peak else 0.0
            # A strategy with nothing left (wiped out) can't breach again: it is already halted (m12 fix re-check, mF-3).
            shocks.append({"loss": loss, "breach": not in_fund and x["equity"] > 0 and dd_after >= p.max_drawdown,
                           # A perp whose move reaches what it has at stake is liquidated in the gap (its margin
                           # and the liquidation fee lost): a two-leg Strategy's other leg is then left unhedged.
                           "liquidated": perp_held and at_stake > 0 and raw >= at_stake})
        rows.append({
            "x": x,
            "dd_used": x["dd_used"],
            # Inside the fund its own day pause doesn't act (CUT-A (d)): nothing used (B-20)
            "day_used": 0.0 if in_fund else min(max(-x["day_ret"], 0.0) / p.daily_loss, 1.0) if p.daily_loss else 0.0,
            "headroom": x["room"],
            # An open position's own stop (an exit-plan edit moves it), else the model's setting (P1-U14).
            "has_stop": bool(pos["stop_px"]) if (pos := positions.get(s.name)) else bool(
                s.params.get("stop_loss") or s.params.get("stop_atr") or s.params.get("stop_swing_bars")),
            "stop_dist": abs(pos["stop_px"] / pos["entry_px"] - 1) if pos and pos["stop_px"] and pos["entry_px"] else None,
            "position": pos,
            "shocks": shocks,
        })
    rows = _fold_pairs(store, rows)
    equity = book["equity"] or 1.0
    # Inside the fund only the fund's drawdown brake halts (T5b, B-20): a move that takes the book's drawdown to it
    fund_limit = fund_halt_limit(store) if book.get("fund_cash") else None
    # measured as the gate measures it, from PortfolioState.reference() (the high-water mark from every mark, or the
    # PM's Resume re-base), never from the daily curve's drawdown, which misses an intraday peak (CR MUST 1). None
    # when it can't be read: the fund's halt is then unknown, never "would not halt".
    reference = fund_halt_reference(store) if fund_limit is not None else None
    scenarios = []
    for i, shock in enumerate(SHOCKS):
        # The book's loss is every leg's, a pair's row carrying its two legs' signed sum; a test-account Strategy is
        # outside the fund's book (it still halts on its own limit).
        loss = sum(r["shocks"][i]["loss"] for r in rows
                   if r["x"]["sleeve"].desired_state == "running" and not r["x"].get("test_account"))
        fund_halt = (False if fund_limit is None else None if reference is None
                     else 1 - (book["equity"] - loss) / reference >= fund_limit)
        scenarios.append({"shock": shock, "loss": loss, "pnl": -loss, "loss_pct": loss / equity, "fund_halt": fund_halt,
                          "breaches": [r["x"]["sleeve"].name for r in rows if r["shocks"][i]["breach"]],
                          "unhedged": [r["x"]["sleeve"].name for r in rows if r["shocks"][i].get("unhedged")]})
    largest = largest_asset(book["allocation"], book["equity"])
    return {"rows": rows, "scenarios": scenarios, "largest": largest,
            "fund_halt_unknown": fund_limit is not None and reference is None,
            **{k: held[k] for k in ("accounts", "book", "unallocated", "margin", "open_risk", "estimated", "through", "left_out", "trailing", "hint", "spot",
                                       "spot_risk", "spot_left_out", "spot_hint")},
            "down20": next(sc for sc in scenarios if sc["shock"] == -0.20),
            "up20": next(sc for sc in scenarios if sc["shock"] == 0.20),
            "history": store.events_of(BREACH_KINDS, limit=50)}


def limits_in_force(store: Store):
    """The compliance tests the entry gate applies now (portfolio.compliance.load, the PM's saved version), or None
    when they can't be read, which the gate refuses on. Never the module's constant PORTFOLIO, which is only the seeded
    first version (B-20)."""
    from sleeve_fund.portfolio import compliance

    try:
        return compliance.load(store)
    except Exception as exc:  # noqa: BLE001 - said as not readable, never a stand-in limit
        print(f"dashboard: can't read the compliance tests in force: {exc!r}")
        return None


def fund_halt_limit(store: Store) -> float | None:
    """T5b, the fund's drawdown halt, as the gate reads it; None when off or unreadable (B-20)."""
    p = limits_in_force(store)
    return float(p.drawdown) if p is not None and p.drawdown else None


def fund_halt_reference(store: Store) -> float | None:
    """What the fund's drawdown halt measures from, as the gate reads it (PortfolioState.reference()); None before the
    supervisor's first mark or when the gate's state can't be read (printed)."""
    from sleeve_fund.paper.portfolio_gate import PaperGate

    try:
        ref = PaperGate(store).ledger.state().reference()
    except Exception as exc:  # noqa: BLE001 - unknown, said so on the page, never a stand-in peak
        print(f"risk page: can't read the fund's halt reference: {exc!r}")
        return None
    return float(ref) if ref is not None and ref > 0 else None


def _fold_pairs(store: Store, rows: list[dict]) -> list[dict]:
    """A two-leg Strategy's legs as one row (F-P28-UI1; Independent Quant Advisor, 14 Oct 10:05 UK), at its first
    leg's place, the legs nested under it (`legs`). Its limits are the summed guard's (ruling E): drawdown from the
    pair's high-water mark, today's loss from the pair's day start, the profile's limits once on the summed equity.
    Its cap use is gross, both legs' notional against the Strategy's cap, with no hedge credit (ruling A); each leg
    keeps its own. A shock is an instant gap: each leg's loss is its own (a perp at most its margin and the
    liquidation fee), the pair's their signed sum, and the halt is judged on the summed equity. A leg liquidated in
    the gap leaves the other unhedged, which the row names (`unhedged`)."""
    by_name = {r["x"]["sleeve"].name: r for r in rows}
    pairs = {p: n for p, n in legs.legs_of([r["x"]["sleeve"] for r in rows]).items() if p in by_name}
    second = set(pairs.values())
    out = []
    for r in rows:
        name = r["x"]["sleeve"].name
        if name in second:
            continue
        out.append(_pair_row(store, r, by_name[pairs[name]]) if name in pairs else r)
    return out


def _pair_row(store: Store, first: dict, second: dict) -> dict:
    x = legs.fold_one(store, first["x"], second["x"])
    p, pair = x["profile"], x["pair"]
    in_fund = bool(x.get("in_fund"))  # its own halt and day pause don't act inside the fund (B-20)
    if not in_fund:  # inside the fund fold_one's day_ret is already over the fund's equity
        day_start = float(pair["day_start_equity"]) if pair.get("day_start_equity") else x["equity"] - x["day_pnl"]
        x["day_ret"] = x["equity"] / day_start - 1 if day_start else 0.0
    shocks = []
    for a, b in zip(first["shocks"], second["shocks"], strict=True):
        loss = a["loss"] + b["loss"]
        after = x["equity"] - loss
        dd_after = 1 - after / x["peak"] if x["peak"] else 0.0
        shocks.append({"loss": loss, "breach": not in_fund and x["equity"] > 0 and dd_after >= p.max_drawdown,
                       "liquidated": a["liquidated"] or b["liquidated"],
                       "unhedged": [n for n, leg in ((1, a), (2, b)) if leg["liquidated"]]})
    for leg in (first, second):
        leg["perp"] = markets.is_perp(leg["x"]["sleeve"].params)
    return {
        "x": x, "legs": [first, second],
        "dd_used": x["dd_used"],
        "day_used": 0.0 if in_fund else min(max(-x["day_ret"], 0.0) / p.daily_loss, 1.0) if p.daily_loss else 0.0,
        "headroom": x["room"],
        "has_stop": first["has_stop"] and second["has_stop"], "stop_dist": None, "position": None,
        "shocks": shocks,
    }


# Total size turns amber from this share of its limit: a display threshold only, as the gate has the one limit
# (Advisor critique 2, 13 Oct 23:40 UK).
NEAR_LIMIT = 0.8


def gate_size(store: Store, book: dict) -> dict:
    """The book's total size (long + short) and net, x book, as the portfolio's entry gate reads them (UP-1): the
    fund's last mark and every position and live reservation it counts (paper.portfolio_gate, decide()'s "gross").
    Before the supervisor has marked the fund the gate reads nothing (it refuses entries as stale), so the figures
    are the strategies' own marks, and `source` says so ("marks"; "unreadable" when the gate's book couldn't be
    read, which is printed). tone: "" under NEAR_LIMIT of the limit, "warn" from it,
    "loss" at the limit, where the gate takes nothing more that raises total size. Display only."""
    from sleeve_fund.paper.portfolio_gate import PaperGate

    tests = limits_in_force(store)  # the saved T4a, not the seeded constant (B-20); None: off or unreadable
    limit = float(tests.gross) if tests is not None and tests.gross is not None else None
    total = net = None
    source = "gate"
    try:
        gate = PaperGate(store)
        if gate.ledger.state().equity is not None:
            equity, held = gate.ledger.positions()
            held = (*held, *(r.holding for r in gate.ledger.reservations()))
            if equity > 0:
                total = sum((abs(h.notional) for h in held), Decimal(0)) / equity
                net = sum((h.notional for h in held), Decimal(0)) / equity
    except Exception as exc:  # noqa: BLE001 - a book the gate can't read is shown from the strategies' marks
        print(f"risk page: can't read the entry gate's book, showing the strategies' marks: {exc!r}")
        total = net = None
        source = "unreadable"
    if total is None:
        total, net = Decimal(repr(book["exposure_pct"])), Decimal(repr(book["net_pct"]))
        source = "marks" if source == "gate" else source
    tone = ""
    if limit is not None:
        cap = Decimal(repr(limit))  # compared exactly, as the gate compares (limits.decide)
        tone = "loss" if total >= cap else "warn" if total >= cap * Decimal(repr(NEAR_LIMIT)) else ""
    total, net = float(total), float(net)
    return {"total": total, "net": net, "limit": limit, "tone": tone, "source": source,
            "limit_off": limit is None and tests is not None}


def most_it_can_lose(x: dict, bound: float | None = None) -> float:
    """The most a strategy's open position can lose from the mark, however far the market moves. A perpetual
    on isolated margin is liquidated before it loses more than its margin (notional at entry over the profile's
    leverage cap, markets.isolated_margin) and what it has made or lost since entry, plus the fee on the
    liquidation; the rest of its equity is not at risk (m13-U6). Anything else can lose at most its equity, or, inside
    the fund, `bound`: the fund's equity, since its own is its P&L from 0 (B-20)."""
    equity, s, qty = max(x["equity"] if bound is None else bound, 0.0), x["sleeve"], x.get("qty") or 0.0
    t, tiered = markets.held_terms(s.params, s.venue, s.instrument) if qty else (None, False)
    if t is None or not tiered:  # unknown liquidation (no kept table): no more than its equity
        return equity
    entry, lev = x.get("entry_px") or x["price"], risk.leverage_cap(x["profile"], s.params)
    cash = trading.margin_cash(x)  # inside the fund, with its funding added back as the engine does (B-20)
    liq = markets.isolated_liquidation(cash, qty, entry, lev, t.maintenance_margin)
    taker = float(markets.fees_for(s.params, venue_profile(s.venue).fees, s.venue).taker)
    cap = markets.gap_loss_cap(qty, entry, lev, cash + qty * entry, taker, liq)  # what the engine books (P1-D3)
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
            bad.append(f"{label(s.name)} is {s.status}" + (f": {s.status_reason}" if s.status_reason else ""))
        elif s.status == "paused":
            warn.append(f"{label(s.name)} is paused" + (f": {s.status_reason}" if s.status_reason else ""))
    bad += [f"{label(name)} is not reporting" for name in health["down"]]
    bad += [f"{label(name)}'s balance doesn't match the venue's" for name in health["mismatched"]]
    for r in rows:
        s = r["x"]["sleeve"]
        if s.status in ("halted", "error"):
            continue  # already named, worse
        # The limits that halt or pause (the per-Strategy position cap is gone: TEC-FE-CAP, HoE 14 Oct 2026).
        used = [(r["dd_used"], "drawdown"), (r["day_used"], "daily loss")]
        share, which = max(used, key=lambda u: u[0])
        if share > NEAR_LIMIT:
            warn.append(f"{label(s.name)} has used {share:.0%} of its {which} limit")
    warn += [f"{label(name)} has had no trade or quote from the venue lately" for name in health["stale_feeds"]]
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
    return {"bars": bars, "halts": [sc for sc in shown if sc["breaches"] or sc.get("fund_halt")]}
