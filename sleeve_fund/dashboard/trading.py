"""Orders, trades and open positions for the blotter and trade history, each with the reason
the strategy gave when it acted (journaled in the orders table at decision time)."""

from __future__ import annotations

from sleeve_fund import markets
from sleeve_fund.research.metrics import ZERO, _dec, trade_stats, trades
from sleeve_fund.store import OPEN_ORDER_STATUSES, Store, utcnow

INTENTS = {"entry": "Entry", "exit": "Signal exit", "stop_loss": "Stop-loss", "take_profit": "Take-profit",
           "risk_halt": "Risk halt", "risk_pause": "Daily-loss pause", "pm_flatten": "PM flatten",
           "rebalance": "Rebalance", "liquidation": "Liquidated", "liquidation_cut": "Cut before liquidation"}
# Exit events, for trades closed before orders were journaled with a reason.
EXIT_EVENTS = {k: INTENTS[k] for k in ("stop_loss", "take_profit", "risk_halt", "risk_pause", "pm_flatten",
                                       "liquidation", "liquidation_cut")}

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


def exit_ways(params: dict | None, side: int | None = None) -> dict[str, str]:
    """Which way a stop and a target sit from the entry, in words: for the side held, or, with no
    position, for every side the strategy can take (a long/short perpetual: both)."""
    if side and side < 0:
        return {"stop": "above", "tp": "below", "swing": "highest high", "swing_short": "high"}
    try:
        perp = markets.is_perp(params)
    except ValueError:  # an unknown market typed into a form: the form refuses it on submit
        perp = False
    if not side and perp and (params or {}).get("allow_short"):
        return {"stop": "below (a short: above)", "tp": "above (a short: below)",
                "swing": "lowest low (a short: highest high)", "swing_short": "low (a short: high)"}
    return {"stop": "below", "tp": "above", "swing": "lowest low", "swing_short": "low"}


def plan_items(plan: dict, side: int = 1) -> list[tuple[str, str]]:
    """An exit plan set after entry (see Store.set_exit_plan), as (label, text) pairs for display.
    side: the position's, so a short's stop reads above its entry."""
    out = [("Exits", "edited by the PM" if plan["kind"] == "edit" else "set again after a restart")]
    away, toward = ("below", "above") if side > 0 else ("above", "below")
    if plan.get("stop_frac") is not None:
        s = plan["stop_frac"]
        out.append(("Stop now", f"{s:.1%} {away} the entry" if s >= 0 else f"{-s:.1%} {toward} the entry"))
    if plan.get("tp_frac"):
        out.append(("Target now", f"{plan['tp_frac']:.1%} {toward} the entry"))
    if plan.get("risk_amount"):
        out.append(("1R now", f"{plan['risk_amount']:,.2f}"))
    if plan.get("planned_r") is not None:
        out.append(("Target in R now", f"{plan['planned_r']:+.2f}R"))
    return out


def trips(fills: list[dict], events: list[dict], orders: dict[str, dict],
          plans: dict[str, dict] | None = None, shorts: bool = False, funding: list[dict] | None = None,
          insurance: list[dict] | None = None) -> list[dict]:
    """Closed round trips, newest first, with holding time and the journaled reason at each end.
    fills: newest first, as the store returns them. orders: journal rows keyed by order id. plans: exit
    plans set after entry (Store.exit_plans), by entry order id. shorts: a perpetual's journal, where a
    sell from flat opens a short (metrics.trades). funding: a perpetual's payments, booked to each trip;
    insurance: what the venue's insurance fund took past a trip's bankruptcy price."""
    exits = [e for e in events if e["kind"] in EXIT_EVENTS]
    out = []
    for t in reversed(trades(list(reversed(fills)), shorts, funding, insurance)):
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
            t["entry_items"] = t["entry_items"] + plan_items(plan, t["side"])
        out.append(t)
    return out


def open_lot(fills: list[dict], shorts: bool = False) -> dict | None:
    """The current position's opening fill, walking the journal (newest-first input) forwards. With
    shorts (a perpetual), a sell from flat opens a short; on spot it is a journal read from mid-trip."""
    qty, opened = ZERO, None  # summed in Decimal, as metrics.trades does (review round 10, M10-4)
    for f in reversed(fills):
        before = qty
        qty += _dec(f["qty"]) if f["side"] == "BUY" else -_dec(f["qty"])
        if not shorts:
            qty = max(qty, ZERO)
        if qty == ZERO:
            opened = None
        elif before == ZERO or (before > ZERO) != (qty > ZERO):
            opened = f  # opened from flat, or went through flat to the other side
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


def position_margin(x: dict) -> tuple[float, float | None]:
    """The margin x's open position puts up and the price that liquidates it. On a perpetual, isolated at
    the risk profile's leverage cap, as paper and the demo copy hold it: notional at entry over the cap
    (markets.isolated_margin, U13-2), liquidated from that margin, not from the strategy's whole equity. On
    spot, what it cost, with nothing to liquidate. The one place every page takes these from (UI v2, item 7)."""
    qty, cash = x["qty"], x["cash"]
    entry = x.get("entry_px") or x["price"]
    if not qty:
        return 0.0, None
    t = markets.terms(x["sleeve"].params, getattr(x["sleeve"], "venue", None))
    if t is None:
        return abs(qty) * entry, None
    lev = x["profile"].max_leverage
    return (markets.isolated_margin(qty, entry, lev, cash + qty * entry),
            markets.isolated_liquidation(cash, qty, entry, lev, t.maintenance_margin))


def risk_to_stop(qty: float, price: float, stop_px: float | None) -> float | None:
    """What the position loses from the mark if its stop is hit: size x the move to the stop, a fall for a
    long, a rise for a short (0 once the mark is already past it). None when it has no stop: unbounded."""
    if stop_px is None:
        return None
    side = 1 if qty > 0 else -1
    return abs(qty) * max(side * (price - stop_px), 0.0)


def open_risk(positions: list[dict]) -> dict:
    """Margin and open risk across positions (open_position's dicts): margin put up, the sum of their Risk to
    stop, and the strategies whose position has no stop, which leave open risk unbounded."""
    return {
        "margin": sum(p["margin"] for p in positions),
        "open_risk": sum(p["risk_to_stop"] for p in positions if p["risk_to_stop"] is not None),
        "unbounded": [p["sleeve"] for p in positions if p["risk_to_stop"] is None],
    }


def open_position(x: dict, fills: list[dict], orders: dict[str, dict],
                  plans: dict[str, dict] | None = None) -> dict | None:
    """An open position from a sleeve summary (with book extras), or None when flat."""
    if not x["qty"] or not x["entry_px"]:
        return None
    side = 1 if x["qty"] > 0 else -1
    lot = open_lot(fills, shorts=markets.is_perp(x["sleeve"].params))
    entry = orders.get(lot["order_id"]) if lot else None
    plan = (plans or {}).get(lot["order_id"]) if lot else None
    sl, tp = exit_fracs(x["sleeve"].params, entry["signal"] if entry else None, plan)
    cost = abs(x["qty"]) * x["entry_px"]
    stop_px = x["entry_px"] * (1 - side * sl) if sl is not None else None
    margin, liq = position_margin(x)
    return {
        "sleeve": x["sleeve"].name,
        "pair": x["sleeve"].instrument,
        "qty": x["qty"],
        "side": side,
        "entry_px": x["entry_px"],
        "price": x["price"],
        "value": x["position_value"],
        "unrealised": x["unrealised"],
        "unrealised_ret": x["unrealised"] / cost if cost else 0.0,
        "opened": lot["ts"] if lot else None,
        "held": (utcnow() - lot["ts"]) if lot else None,
        "stop_px": stop_px,
        "target_px": x["entry_px"] * (1 + side * tp) if tp else None,
        "notional": abs(x["qty"]) * x["price"],
        "margin": margin,
        "leverage": cost / margin if margin else None,
        "liq_px": liq,
        "to_liq": abs(liq / x["price"] - 1) if liq and x["price"] else None,
        "risk_to_stop": risk_to_stop(x["qty"], x["price"], stop_px),
        "why": entry["reason"] if entry else None,
        "sig": (signal_items(entry["signal"]) if entry else []) + (plan_items(plan, side) if plan else []),
        "exits_edited": bool(plan and plan["kind"] == "edit"),
        "weight": x["position_value"] / x["equity"] if x["equity"] else 0.0,
    }


def perp_view(x: dict, position: dict | None, funding: list[dict]) -> dict | None:
    """What a perpetual position adds to a spot one: leverage, isolated margin, how far it is from
    liquidation, and the funding it has paid or received. funding: newest first, as the store returns it."""
    t = markets.terms(x["sleeve"].params, getattr(x["sleeve"], "venue", None))
    if t is None:
        return None
    qty, price, equity = x["qty"], x["price"], x["equity"]
    notional = abs(qty * price)
    margin, liq = position_margin(x)
    opened = position["opened"] if position else None
    held = [f for f in funding if opened is not None and f["ts"] >= opened]
    return {
        "leverage": notional / equity if equity > 0 else None,
        "margin": margin,
        "maintenance": notional * t.maintenance_margin,
        "maintenance_rate": t.maintenance_margin,
        "liq_px": liq,
        "to_liq": abs(liq / price - 1) if liq and price else None,
        "funding_rate": t.funding_rate,
        "funding_last": held[0] if held else None,
        "funding_open": sum(f["amount"] for f in held),
        "funding_total": sum(f["amount"] for f in funding),
        "label": t.label,
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
        perp = markets.is_perp(x["sleeve"].params)
        for t in trips(fills, store.events(name, limit=5000), orders, plans, perp,
                       store.funding(name, limit=1_000_000) if perp else None,
                       store.insurance(name) if perp else None):
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
        **open_risk(positions),
    }


def book_positions(store: Store, summaries: list[dict]) -> dict:
    """Every open position across the book for the Portfolio page, each with its strategy's P&L split as the
    strategy page shows it: unrealised on the open position, realised since the strategy started (its P&L
    less the unrealised part, so closed trades and funding are in it) and the fees it has paid. Display only:
    the summaries' own figures, added up."""
    rows = []
    for x in summaries:
        if not x["qty"] or not x["entry_px"]:
            continue
        name = x["sleeve"].name
        pos = open_position(x, store.fills(name, limit=100_000), orders_by_id(store, name), store.exit_plans(name))
        if pos is None:
            continue
        perp = markets.is_perp(x["sleeve"].params)
        rows.append({
            **pos,
            "x": x,
            "perp": perp_view(x, pos, store.funding(name, limit=100_000)) if perp else None,
            "realised": x["pnl"] - x["unrealised"],
            "fees": x["fees"],
            "flattening": any(c["command"] == "flatten" for c in store.pending_commands(name)),
        })
    return {
        "rows": rows,
        "unrealised": sum(r["unrealised"] for r in rows),
        "realised": sum(r["realised"] for r in rows),
        "fees": sum(r["fees"] for r in rows),
        **open_risk(rows),
    }


def book_fills(store: Store, sleeves: list, limit: int = 100) -> list[dict]:
    """The book's latest fills across its strategies, newest first, each with the reason its order journaled."""
    out = []
    for s in sleeves:
        name = s.name
        fills = store.fills(name, limit=limit)
        if not fills:
            continue
        orders = {o["order_id"]: o for o in store.orders(name, limit=limit * 10)}
        for f in fills:
            o = orders.get(f["order_id"])
            out.append({**f, "pair": s.instrument, "reason": o["reason"] if o else None,
                        "intent": o["intent"] if o else None,
                        "intent_label": INTENTS.get(o["intent"], o["intent"]) if o else None})
    out.sort(key=lambda f: (f["ts"], f["id"]), reverse=True)
    return out[:limit]


def book_funding(store: Store, summaries: list[dict], limit: int = 50) -> dict:
    """Funding settled on the book's perpetual positions: the latest payments, newest first, and each
    strategy's total since it started (+ received, - paid)."""
    payments, totals = [], []
    for x in summaries:
        s = x["sleeve"]
        if not markets.is_perp(s.params):
            continue
        rows = store.funding(s.name, limit=limit)
        payments += [{**f, "sleeve": s.name, "pair": s.instrument} for f in rows]
        if rows:
            totals.append({"sleeve": s.name, "pair": s.instrument, "total": store.funding_total(s.name)})
    payments.sort(key=lambda f: (f["ts"], f["id"]), reverse=True)
    return {"payments": payments[:limit], "totals": totals, "total": sum(t["total"] for t in totals),
            "perps": sum(1 for x in summaries if markets.is_perp(x["sleeve"].params))}


AUDIT_COLUMNS = ["ts", "strategy", "venue", "instrument", "side", "qty", "price", "notional", "fee", "realised_pnl",
                 "position_after", "intent", "reason", "order_id", "trade_id"]


def audit_rows(sleeve, fills: list[dict], orders: dict[str, dict], venue: str) -> tuple[list[dict], list[str]]:
    """One row per fill, oldest first, for checking every trade outside the dashboard (PM, 5 Oct 2026): what was
    traded, at what price and fee, the P&L it realised (on the average entry, after this fill's fee) and the
    position after it, with the reason and the indicator values the strategy decided on, from its order's signal.
    Returns the rows and the signal's columns, in the order they first appear. fills: oldest first. Funding on a
    perpetual is booked per trade, not per fill: the trades export carries it."""
    rows, keys, qty, avg = [], [], 0.0, 0.0
    for f in fills:
        signed = f["qty"] if f["side"] == "BUY" else -f["qty"]
        px, realised = float(f["price"]), 0.0
        if qty and (qty > 0) != (signed > 0):  # reduces, closes or turns the position
            closed = min(abs(signed), abs(qty))
            realised = closed * (px - avg) * (1 if qty > 0 else -1)
            rest = qty + signed
            if abs(rest) < 1e-12:
                qty, avg = 0.0, 0.0
            elif (rest > 0) != (qty > 0):  # turned: what's left opens at this price
                qty, avg = rest, px
            else:
                qty = rest
        else:
            avg = (avg * abs(qty) + px * abs(signed)) / (abs(qty) + abs(signed))
            qty += signed
        order = orders.get(f.get("order_id") or "") or {}
        signal = {k: v for k, v in (order.get("signal") or {}).items() if k != "stop_cfg"}
        for k in signal:
            if k not in keys:
                keys.append(k)
        rows.append({"ts": f["ts"], "strategy": sleeve.name, "venue": venue, "instrument": sleeve.instrument,
                     "side": f["side"], "qty": f["qty"], "price": px, "notional": round(f["qty"] * px, 8),
                     "fee": f["fee"], "realised_pnl": round(realised - float(f["fee"]), 8),
                     "position_after": round(qty, 12), "intent": order.get("intent"), "reason": order.get("reason"),
                     "order_id": f.get("order_id"), "trade_id": f.get("trade_id"),
                     **{k: v if not isinstance(v, (dict, list)) else str(v) for k, v in signal.items()}})
    return rows, keys
