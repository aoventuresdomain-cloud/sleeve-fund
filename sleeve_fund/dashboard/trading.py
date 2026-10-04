"""Orders, trades and open positions for the blotter and trade history, each with the reason
the strategy gave when it acted (journaled in the orders table at decision time)."""

from __future__ import annotations

from sleeve_fund.research.metrics import trade_stats, trades
from sleeve_fund.store import OPEN_ORDER_STATUSES, Store, utcnow

INTENTS = {"entry": "Entry", "exit": "Signal exit", "stop_loss": "Stop-loss", "take_profit": "Take-profit",
           "risk_halt": "Risk halt", "risk_pause": "Daily-loss pause", "pm_flatten": "PM flatten",
           "rebalance": "Rebalance"}
# Exit events, for trades closed before orders were journaled with a reason.
EXIT_EVENTS = {k: INTENTS[k] for k in ("stop_loss", "take_profit", "risk_halt", "risk_pause", "pm_flatten")}

STATUS_TABS = {
    "open": ("Open", OPEN_ORDER_STATUSES),
    "filled": ("Filled", ("filled",)),
    "canceled": ("Cancelled", ("canceled", "expired")),
    "rejected": ("Rejected", ("rejected", "denied")),
    "all": ("All", None),
}
STATUS_LABELS = {"submitted": "Sent", "accepted": "Working", "partially_filled": "Part filled", "filled": "Filled",
                 "canceled": "Cancelled", "rejected": "Rejected", "denied": "Blocked", "expired": "Expired"}
STATUS_TONES = {"submitted": "paused", "accepted": "paused", "partially_filled": "paused", "filled": "running",
                "canceled": "stopped", "expired": "stopped", "rejected": "halted", "denied": "halted"}

_PX = ("close", "price", "entry_px", "peak", "trail_stop")
_PCT = ("gap", "move", "stop_loss", "take_profit")


def signal_items(signal: dict | None) -> list[tuple[str, str]]:
    """The values behind a decision as (label, text) pairs for display."""
    out = []
    for k, v in (signal or {}).items():
        if v is None or k == "stop_cfg":  # the stop settings, kept to tell whether a later edit changed the stop
            continue
        if k.startswith(("sma_", "ema_")):
            label, text = f"{k[:3].upper()} {k[4:]}", _px(v)
        elif k == "rsi":
            label, text = "RSI", f"{v:.1f}"
        elif k == "volume_x":
            label, text = "Volume vs normal", f"{v:.2f}x"
        elif k == "atr":
            label, text = "ATR", _px(v)
        elif k in _PX:
            label, text = {"close": "Bar close", "price": "Last price", "entry_px": "Entry", "peak": "Peak close",
                           "trail_stop": "Trailing stop"}[k], _px(v)
        elif k in _PCT:
            label = {"gap": "Average gap", "move": "Move from entry", "stop_loss": "Stop", "take_profit": "Target"}[k]
            text = f"{v:+.2%}" if k in ("gap", "move") else f"{v:.1%}"
        elif k == "order_type":
            label, text = "Order", "Maker first (post-only limit)" if v == "maker" else "Market"
        elif k == "limit_px":
            label, text = "Limit", _px(v)
        elif k == "maker_wait_minutes":
            label, text = "Market after", f"{v:g} min"
        elif k == "maker_order":
            label, text = "Rest of order", str(v)
        elif k == "sized_by":
            label, text = "Size set by", str(v)
        elif k in ("budget", "equity", "peak_equity"):
            label, text = {"budget": "Budget", "equity": "Equity", "peak_equity": "Peak equity"}[k], f"{v:,.2f}"
        elif k == "risk_amount":
            label, text = "Loss at stop (1R)", f"{v:,.2f}"
        elif k == "planned_r":
            label, text = "Target in R", f"{v:+.2f}R"
        else:
            label, text = k.replace("_", " ").capitalize(), (f"{v:,.6g}" if isinstance(v, float) else str(v))
        out.append((label, text))
    return out


def _px(v: float) -> str:
    return f"{v:,.2f}" if abs(v) >= 100 else f"{v:,.4f}" if abs(v) >= 1 else f"{v:,.6f}"


def order_view(o: dict) -> dict:
    o = dict(o)
    o["status_label"] = STATUS_LABELS.get(o["status"], o["status"])
    o["tone"] = STATUS_TONES.get(o["status"], "stopped")
    o["intent_label"] = INTENTS.get(o["intent"], o["intent"])
    o["notional"] = (o["avg_px"] or 0.0) * o["filled_qty"]
    o["sig"] = signal_items(o["signal"])
    o["is_open"] = o["status"] in OPEN_ORDER_STATUSES
    return o


def plan_items(plan: dict) -> list[tuple[str, str]]:
    """An exit plan set after entry (see Store.set_exit_plan), as (label, text) pairs for display."""
    out = [("Exits", "edited by the PM" if plan["kind"] == "edit" else "set again after a restart")]
    if plan.get("stop_frac") is not None:
        s = plan["stop_frac"]
        out.append(("Stop now", f"{s:.1%} below the entry" if s >= 0 else f"{-s:.1%} above the entry"))
    if plan.get("tp_frac"):
        out.append(("Target now", f"{plan['tp_frac']:.1%}"))
    if plan.get("risk_amount"):
        out.append(("1R now", f"{plan['risk_amount']:,.2f}"))
    if plan.get("planned_r") is not None:
        out.append(("Target in R now", f"{plan['planned_r']:+.2f}R"))
    return out


def trips(fills: list[dict], events: list[dict], orders: dict[str, dict],
          plans: dict[str, dict] | None = None) -> list[dict]:
    """Closed round trips, newest first, with holding time and the journaled reason at each end.
    fills: newest first, as the store returns them. orders: journal rows keyed by order id. plans: exit
    plans set after entry (Store.exit_plans), by entry order id."""
    exits = [e for e in events if e["kind"] in EXIT_EVENTS]
    out = []
    for t in reversed(trades(list(reversed(fills)))):
        entry, exit_ = orders.get(t["entry_order"] or ""), orders.get(t["exit_order"] or "")
        if exit_:
            kind = exit_["intent"]
        else:  # closed before reasons were journaled: fall back to the exit event, if any
            kind = "exit"
            if t["opened"] and t["closed"]:
                hit = [e for e in exits if t["opened"] <= e["ts"] <= t["closed"]]
                kind = hit[0]["kind"] if hit else "exit"
        t.update(
            exit_kind=kind,
            reason=INTENTS.get(kind, "Signal exit"),
            entry_why=entry["reason"] if entry else None,
            exit_why=exit_["reason"] if exit_ else None,
            entry_items=signal_items(entry["signal"]) if entry else [],
            exit_items=signal_items(exit_["signal"]) if exit_ else [],
            held=(t["closed"] - t["opened"]) if t["opened"] and t["closed"] else None,
        )
        # R: the trade's P&L over what it would have lost at its stop, as sized at entry, or over what it
        # risked after a looser stop was set (the larger of the two; see LongFlatStrategy._set_plan).
        sig = (entry or {}).get("signal", {})
        plan = (plans or {}).get(t["entry_order"] or "")
        risk = (plan or {}).get("risk_amount") or sig.get("risk_amount")
        t["r"] = t["pnl"] / risk if risk else None
        t["planned_r"] = plan["planned_r"] if plan else sig.get("planned_r")
        t["exits_edited"] = bool(plan and plan["kind"] == "edit")
        if plan:
            t["entry_items"] = t["entry_items"] + plan_items(plan)
        out.append(t)
    return out


def open_lot(fills: list[dict]) -> dict | None:
    """The current position's opening fill, walking the journal (newest-first input) forwards."""
    qty, opened = 0.0, None
    for f in reversed(fills):
        if f["side"] == "BUY":
            if qty <= 1e-12:
                opened = f
            qty += f["qty"]
        else:
            qty = max(qty - f["qty"], 0.0)
            if qty <= 1e-12:
                opened = None
    return opened


def exit_fracs(params: dict, signal: dict | None, plan: dict | None = None) -> tuple[float | None, float | None]:
    """The open position's stop and target as shares of its entry price: the plan set since entry, if
    any, else what its entry journaled (an ATR or swing-low stop is set at entry), else the strategy's
    fixed % settings."""
    if plan is not None:
        return plan["stop_frac"], plan["tp_frac"]
    sig = signal or {}
    if "stop_frac" in sig or "tp_frac" in sig:
        return sig.get("stop_frac"), sig.get("tp_frac")
    return params.get("stop_loss"), params.get("take_profit")


def open_position(x: dict, fills: list[dict], orders: dict[str, dict],
                  plans: dict[str, dict] | None = None) -> dict | None:
    """An open position from a sleeve summary (with book extras), or None when flat."""
    if x["qty"] <= 0 or not x["entry_px"]:
        return None
    lot = open_lot(fills)
    entry = orders.get(lot["order_id"]) if lot else None
    plan = (plans or {}).get(lot["order_id"]) if lot else None
    sl, tp = exit_fracs(x["sleeve"].params, entry["signal"] if entry else None, plan)
    cost = x["qty"] * x["entry_px"]
    return {
        "sleeve": x["sleeve"].name,
        "pair": x["sleeve"].instrument,
        "qty": x["qty"],
        "entry_px": x["entry_px"],
        "price": x["price"],
        "value": x["position_value"],
        "unrealised": x["unrealised"],
        "unrealised_ret": x["unrealised"] / cost if cost else 0.0,
        "opened": lot["ts"] if lot else None,
        "held": (utcnow() - lot["ts"]) if lot else None,
        "stop_px": x["entry_px"] * (1 - sl) if sl is not None else None,
        "target_px": x["entry_px"] * (1 + tp) if tp else None,
        "why": entry["reason"] if entry else None,
        "sig": (signal_items(entry["signal"]) if entry else []) + (plan_items(plan) if plan else []),
        "exits_edited": bool(plan and plan["kind"] == "edit"),
        "weight": x["position_value"] / x["equity"] if x["equity"] else 0.0,
    }


def orders_by_id(store: Store, sleeve: str | None = None) -> dict[str, dict]:
    return {o["order_id"]: o for o in store.orders(sleeve, limit=100_000)}


def history(store: Store, summaries: list[dict], sleeve: str | None = None) -> dict:
    """Open positions, closed trades and their stats for one sleeve or the book."""
    chosen = [x for x in summaries if not sleeve or x["sleeve"].name == sleeve]
    positions, closed = [], []
    for x in chosen:
        name = x["sleeve"].name
        fills = store.fills(name, limit=1_000_000)
        orders = orders_by_id(store, name)
        plans = store.exit_plans(name)
        pos = open_position(x, fills, orders, plans)
        if pos:
            positions.append(pos)
        for t in trips(fills, store.events(name, limit=5000), orders, plans):
            t["sleeve"], t["pair"] = name, x["sleeve"].instrument
            closed.append(t)
    closed.sort(key=lambda t: t["closed"] or utcnow(), reverse=True)
    stats = trade_stats(closed)
    stats["fees"] = sum(t["fees"] for t in closed)
    return {
        "positions": positions,
        "trades": closed,
        "stats": stats,
        "unrealised": sum(p["unrealised"] for p in positions),
        "exposure": sum(p["value"] for p in positions),
    }
