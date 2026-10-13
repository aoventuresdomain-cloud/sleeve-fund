"""Orders, trades and open positions for the blotter and trade history, each with the reason
the strategy gave when it acted (journaled in the orders table at decision time)."""

from __future__ import annotations

from sleeve_fund import markets
from sleeve_fund.research.metrics import ZERO, _dec, trade_stats, trades
from sleeve_fund.paper.runtime import liquidation_head
from sleeve_fund.store import LIQUIDATION_RESET, OPEN_ORDER_STATUSES, Store, utcnow  # noqa: F401  (QA's tests import it here)

INTENTS = {"entry": "Entry", "exit": "Signal exit", "stop_loss": "Stop-loss", "take_profit": "Take-profit",
           "risk_halt": "Risk halt", "risk_pause": "Daily-loss pause", "pm_flatten": "PM flatten",
           "rebalance": "Rebalance", "liquidation": "Liquidated", "liquidation_cut": "Cut before liquidation"}
# Exit events, for trades closed before orders were journaled with a reason.
EXIT_EVENTS = {k: INTENTS[k] for k in ("stop_loss", "take_profit", "risk_halt", "risk_pause", "pm_flatten",
                                       "liquidation", "liquidation_cut")}

STATUS_TABS = {
    "open": ("Open", OPEN_ORDER_STATUSES),
    "filled": ("Filled", ("filled", "triggered")),
    "canceled": ("Cancelled", ("canceled", "expired")),
    "rejected": ("Rejected", ("rejected", "denied")),
    "all": ("All", None),
}
STATUS_LABELS = {"submitted": "Sent", "accepted": "Working", "partially_filled": "Part filled", "filled": "Filled",
                 "canceled": "Cancelled", "rejected": "Rejected", "denied": "Blocked", "expired": "Expired",
                 "triggered": "Triggered"}
STATUS_TONES = {"submitted": "paused", "accepted": "paused", "partially_filled": "paused", "filled": "running",
                "canceled": "stopped", "expired": "stopped", "rejected": "halted", "denied": "halted",
                "triggered": "running"}

_PX = ("close", "price", "entry_px", "peak", "trail_stop")
_PCT = ("gap", "move", "stop_loss", "take_profit")


def signal_items(signal: dict | None) -> list[tuple[str, str]]:
    """The values behind a decision as (label, text) pairs for display."""
    out = []
    for k, v in (signal or {}).items():
        # the stop settings, kept to tell whether a later edit changed the stop; the decision's bar, an audit record
        # (P1-1-CANON) whose close is shown as "Bar close"
        if v is None or k in ("stop_cfg", "bar"):
            continue
        if k.startswith(("sma_", "ema_")):
            label, text = f"{k[:3].upper()} {k[4:]}", _px(v)
        elif k == "rsi":
            label, text = "RSI", f"{v:.1f}"
        elif k == "volume_x":
            label, text = "Volume vs normal", f"{v:.2f}x"
        elif k == "atr":
            label, text = "Simple ATR", _px(v)
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


def stop_past_liquidation(order: dict | None) -> str | None:
    """A liquidation the engine booked from a stop whose fill, after slippage, was at or past the liquidation price
    (GAP-LIQ: the venue would have liquidated first), in the Advisor's words (7 Oct 18:50 UK), or None. Read from the
    order's stop_relabelled flag (QD FU-STOP-RELABEL-FLAG), never from its reason text."""
    sig = (order or {}).get("signal") or {}
    if order is None or order.get("intent") != "liquidation" or not sig.get("stop_relabelled"):
        return None
    stop = sig.get("stop_px")
    return (f"Liquidated: stop at {stop:,.6g} would have filled past liquidation" if stop is not None
            else "Liquidated: the stop would have filled past liquidation")


def order_view(o: dict) -> dict:
    o = dict(o)
    o["status_label"] = STATUS_LABELS.get(o["status"], o["status"])
    o["tone"] = STATUS_TONES.get(o["status"], "stopped")
    o["intent_label"] = INTENTS.get(o["intent"], o["intent"])
    o["relabelled"] = stop_past_liquidation(o)
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
            relabelled=stop_past_liquidation(exit_),
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


def liquidated_since_reset(store: Store, sleeve: str) -> bool:
    """Whether the strategy's position margin was lost with no reset after liquidation since: the engine's own rule
    (runtime.liquidation_head), so the dashboard and the gate can't disagree (CR 7). Read from the journal, not the
    latest halt: a Stop/Start that halts it again on drawdown must not make a Resume restart it (QA P1-U22)."""
    return liquidation_head(store, sleeve) is not None


def stop_basis(params: dict, signal: dict | None, plan: dict | None = None, side: int = 1) -> str | None:
    """How the open position's stop was set, in a few words, when it came from the market at entry: an ATR
    stop is the simple ATR the models use, not the chart's Wilder ATR (Advisor, atr-149 A2), so it says so.
    None for a fixed % stop or none. Reads the stop settings the entry, or a later plan, journaled."""
    cfg = (plan or {}).get("stop_cfg") if plan is not None else (signal or {}).get("stop_cfg")
    if cfg is None and plan is None and not (signal or {}).get("stop_frac"):
        cfg = params  # an entry from before stop settings were journaled: the strategy's own
    cfg = cfg or {}
    if cfg.get("stop_atr"):
        return f"{float(cfg['stop_atr']):g} simple ATR ({int(cfg.get('atr_bars') or 14)} bars) at entry"
    if cfg.get("stop_swing_bars"):
        return f"swing {'high' if side < 0 else 'low'} of {int(cfg['stop_swing_bars'])} bars at entry"
    return None


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


def stop_to_liquidation(price: float, stop_px: float | None, liq_px: float | None) -> float | None:
    """How far a stop sits along the way from the mark to liquidation: 0 at the mark, 1 at the liquidation
    price, over 1 past it (the venue would liquidate first). None without both, or once the price has gone
    through the stop (the Risk cell says so). A stop must sit no more than half way (Independent Quant Advisor;
    readout ruled 7 Oct 18:50 UK)."""
    if stop_px is None or not liq_px or price == liq_px:
        return None
    share = (price - stop_px) / (price - liq_px)
    return share if share > 0 else None


# Strategies whose stop trails the market inside the strategy and isn't journaled (A2-T will journal it).
TRAILING_STOP = {"rsi_pullback"}


def _atr_pct(now):
    """The engine's daily ATR share for a strategy (open_risk.history_atr_pct, as the open-risk limit reads it), asked
    once per strategy per call."""
    from sleeve_fund import open_risk as limit
    from sleeve_fund.venues import venue as venue_profile

    seen: dict[str, float | None] = {}

    def atr(s):
        if s.name not in seen:
            try:
                seen[s.name] = limit.history_atr_pct(venue_profile(s.venue).name, s.instrument, now)
            except (KeyError, ValueError):
                seen[s.name] = None
        return seen[s.name]
    return atr


def risk_cell(p: dict, row: dict | None) -> dict:
    """What a position's Risk column reads and how it adds into Open risk. A perpetual reads what the open-risk limit
    counts for it (open_risk.book_open_risk), so the rows add up to the gate's figure: kind stop (measured to its
    stop), estimated (no stop, or a trail checked at the close: the stopless measure) or through (the price has gone
    through its stop, still open: the stopless measure); unknown (no daily ATR yet) and none (archived, the limit
    doesn't count it) are left out. Spot is outside the limit and reads its risk to stop as before (P1-U24); left
    out: trailing (level not journaled), unbounded (no stop) and spot_through (past its stop, still open)."""
    if p.get("perp_limit"):
        if row is None:
            return {"kind": "none", "amount": None}
        if row["risk"] is None:
            return {"kind": "unknown", "amount": None, "through": row["basis"] == "gapped"}
        kind = {"stop": "stop", "stopless": "estimated", "gapped": "through"}[row["basis"]]
        return {"kind": kind, "amount": row["risk"], "trailing": bool(p.get("trailing_model"))}
    if p.get("trailing"):
        return {"kind": "trailing", "amount": None}
    if p["stop_px"] is None:
        return {"kind": "unbounded", "amount": None}
    if (p["price"] - p["stop_px"]) * p["qty"] <= 0:
        return {"kind": "spot_through", "amount": None}
    return {"kind": "stop", "amount": p["risk_to_stop"]}


# Why a part is left out of Open risk, by risk_cell kind, as the hover says it.
LEFT_OUT = {"unknown": "no stop to measure and its daily ATR isn't known yet, so not counted",
            "none": "not counted by the open-risk limit (archived)",
            "trailing": "trailing stop, level not shown, not counted",
            "unbounded": "no stop, so unbounded",
            "spot_through": "the price has gone through its stop and the position is still open, not counted"}
STOPLESS_WORDS = "counted at the larger of 10% and 3 daily ATRs"


def _tally(cells: list[tuple[str, dict]]) -> tuple[float, int, list[str], list[str], dict[str, list[str]]]:
    """(total, parts counted, estimated, through, left out by kind) over (strategy, risk cell) pairs."""
    total, counted, estimated, through, left_out = 0.0, 0, [], [], {}
    for name, cell in cells:
        if cell["amount"] is None:
            left_out.setdefault(cell["kind"], []).append(name)
            continue
        total, counted = total + cell["amount"], counted + 1
        if cell["kind"] == "estimated":
            estimated.append(name)
        elif cell["kind"] == "through":
            through.append(name)
    return total, counted, estimated, through, left_out


def open_risk(positions: list[dict], store: Store | None = None) -> dict:
    """Margin and Open risk across positions (open_position's dicts), each position given its "risk" cell
    (risk_cell). Open risk is the 5% open-risk limit's figure and nothing else, so the tile equals the gate
    (open_risk.book_open_risk, one formula; Advisor 7 Oct 18:50 UK; QA R200-1): perpetuals only, a stop measured
    from the mark; no stop, a trail checked at the close or a stop the price has gone through at the stopless
    measure, named as estimated. Spot is outside the limit and has its own line, its risk to stop (spot_risk).
    Whatever can't be measured is left out and named, so the figure carries the "+" (P1-U24-1); with nothing
    measured at all it has no figure (None) rather than a 0."""
    from sleeve_fund import open_risk as limit

    rows = {}
    if store is not None and any(p.get("perp_limit") for p in positions):
        rows = {r["sleeve"]: r for r in limit.book_open_risk(store, _atr_pct(utcnow()))}
    for p in positions:
        p["risk"] = risk_cell(p, rows.get(p["sleeve"]))
    perps = [(p["sleeve"], p["risk"]) for p in positions if p.get("perp_limit")]
    spots = [(p["sleeve"], p["risk"]) for p in positions if not p.get("perp_limit")]
    total, counted, estimated, through, left_out = _tally(perps)
    spot_total, spot_counted, _, _, spot_left = _tally(spots)
    trailing = [p["sleeve"] for p in positions if p.get("trailing_model")]
    return {
        "accounts": _by_account(store, perps) if store is not None else [],
        "margin": sum(p["margin"] for p in positions),
        "open_risk": total if counted or not left_out else None,
        "estimated": estimated,
        "through": through,
        "left_out": [n for names in left_out.values() for n in names],
        "trailing": [p["sleeve"] for p in positions if p.get("trailing")],
        "hint": open_risk_hint(estimated, through, left_out, trailing) or (
            "Money lost if every stop is hit, as the 5% open-risk limit counts it" if perps else
            "No perpetual positions: the 5% open-risk limit counts perpetuals only"),
        "spot": [n for n, _ in spots],
        "spot_risk": (spot_total if spot_counted or not spot_left else None) if spots else None,
        "spot_left_out": [n for names in spot_left.values() for n in names],
        "spot_hint": open_risk_hint([], [], spot_left, trailing) or "Money lost if every spot stop is hit",
    }


def _by_account(store: Store, perps: list[tuple[str, dict]]) -> list[dict]:
    """The limit's figure for each account holding a perpetual strategy that isn't archived, against that account's
    own book (open_risk.account_equity), as the gate measures it per account (RR-1). A spot-only account has no 5%
    limit to show headroom to, so it gets no line (QA-F215-1). {name, open_risk, left_out, book}. Empty when every
    strategy is on one account: the tile then reads as it always has."""
    from sleeve_fund import markets
    from sleeve_fund import open_risk as limit

    archived = store.archived()
    live = [s for s in store.sleeves() if s.name not in archived]
    if len({store.account_of(s.name) for s in live}) < 2:
        return []
    perp_names = {s.name for s in live if markets.is_perp(s.params)}
    out = []
    for a in store.accounts():
        if not perp_names.intersection(a["sleeves"]):
            continue
        mine = [(n, c) for n, c in perps if store.account_of(n) == a["name"]]
        total, counted, _, _, left_out = _tally(mine)
        out.append({"name": a["name"], "open_risk": total if counted or not left_out else None,
                    "left_out": [n for names in left_out.values() for n in names],
                    "book": limit.account_equity(store, a["name"])})
    return out


def open_risk_hint(estimated: list[str], through: list[str], left_out: dict[str, list[str]],
                   trailing: list[str]) -> str:
    """The hover on an Open risk figure: each strategy whose part is estimated or left out, and why."""
    parts = []
    if through:
        parts.append(f"{', '.join(through)}: the price has gone through its stop and the position is still open, "
                     f"so it is {STOPLESS_WORDS}")
    if nostop := [n for n in estimated if n not in trailing]:
        parts.append(f"{', '.join(nostop)}: no stop, {STOPLESS_WORDS}")
    if trail := [n for n in estimated if n in trailing]:
        parts.append(f"{', '.join(trail)}: trailing stop checked at the close, {STOPLESS_WORDS}")
    for kind in ("unbounded", "trailing", "spot_through", "unknown", "none"):
        if left_out.get(kind):
            parts.append(f"{', '.join(left_out[kind])}: {LEFT_OUT[kind]}")
    return " · ".join(parts)


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
    # The position's own fees: its opening fill and every one after it (newest-first input), not the strategy's
    # since it started (QA U4). By place in the journal, as fills in one second share a timestamp.
    own = next((i for i, f in enumerate(fills) if f is lot), None)
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
        "stop_basis": stop_basis(x["sleeve"].params, entry["signal"] if entry else None, plan, side) if stop_px else None,
        "target_px": x["entry_px"] * (1 + side * tp) if tp else None,
        "notional": abs(x["qty"]) * x["price"],
        "margin": margin,
        "leverage": cost / margin if margin else None,
        "liq_px": liq,
        "to_liq": abs(liq / x["price"] - 1) if liq and x["price"] else None,
        "risk_to_stop": risk_to_stop(x["qty"], x["price"], stop_px),
        "stop_liq": stop_to_liquidation(x["price"], stop_px, liq),
        "perp_limit": markets.is_perp(x["sleeve"].params),
        "trailing_model": x["sleeve"].strategy in TRAILING_STOP,
        "trailing": stop_px is None and x["sleeve"].strategy in TRAILING_STOP,
        "why": entry["reason"] if entry else None,
        "sig": (signal_items(entry["signal"]) if entry else []) + (plan_items(plan, side) if plan else []),
        "exits_edited": bool(plan and plan["kind"] == "edit"),
        "weight": x["position_value"] / x["equity"] if x["equity"] else 0.0,
        "fees": sum(float(f["fee"] or 0.0) for f in fills[:own + 1]) if own is not None else None,
    }


def perp_view(x: dict, position: dict | None, funding: list[dict]) -> dict | None:
    """What a perpetual position adds to a spot one: leverage, exposure, isolated margin, how far it is from
    liquidation, and the funding it has paid or received. funding: newest first, as the store returns it.
    Leverage is the position's notional at entry over its isolated margin (the open position's own figure);
    exposure is its notional at the mark over the strategy's equity (QA U2, as the Advisor worded them)."""
    t = markets.terms(x["sleeve"].params, getattr(x["sleeve"], "venue", None))
    if t is None:
        return None
    qty, price, equity = x["qty"], x["price"], x["equity"]
    notional = abs(qty * price)
    margin, liq = position_margin(x)
    opened = position["opened"] if position else None
    held = [f for f in funding if opened is not None and f["ts"] >= opened]
    return {
        "leverage": position["leverage"] if position else None,
        "exposure": notional / equity if equity > 0 else None,
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
                       store.funding(name, limit=None) if perp else None,
                       store.insurance(name, limit=None) if perp else None):
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
        **open_risk(positions, store),
    }


def book_positions(store: Store, summaries: list[dict]) -> dict:
    """Every open position across the book for the Portfolio page, each with its strategy's P&L split as the
    strategy page shows it: unrealised on the open position, realised since the strategy started (its P&L
    less the unrealised part, so closed trades and funding are in it) and the fees the open position has paid.
    Display only: the summaries' own figures and the fills, added up."""
    rows = []
    for x in summaries:
        if not x["qty"] or not x["entry_px"]:
            continue
        name = x["sleeve"].name
        fills = store.fills(name, limit=100_000)
        pos = open_position(x, fills, orders_by_id(store, name), store.exit_plans(name))
        if pos is None:
            continue
        perp = markets.is_perp(x["sleeve"].params)
        rows.append({
            **pos,
            "x": x,
            "perp": perp_view(x, pos, store.funding(name, limit=None)) if perp else None,
            "realised": x["pnl"] - x["unrealised"],
            "fees": pos["fees"] if pos["fees"] is not None else x["fees"],
            "flattening": any(c["command"] == "flatten" for c in store.pending_commands(name)),
        })
    return {
        "rows": rows,
        "unrealised": sum(r["unrealised"] for r in rows),
        "notional": sum(abs(r["value"]) for r in rows),
        "realised": sum(r["realised"] for r in rows),
        "fees": sum(r["fees"] for r in rows),
        **open_risk(rows, store),
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
        signal = {k: v for k, v in (order.get("signal") or {}).items() if k not in ("stop_cfg", "bar")}
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


def timing_view(timings: list[dict]) -> dict | None:
    """Close to fill for the strategy's latest orders decided on a bar (Store.timings, v2 P1-2): median and 95th
    percentile in milliseconds, and the median of how much of it was ours (bar close to order sent). None
    before any such order has filled."""
    done = [t for t in timings if t["bar_close"] and t["first_fill"]]
    if not done:
        return None

    def ms(rows, a, b):
        return sorted((r[b] - r[a]).total_seconds() * 1000 for r in rows if r[a] and r[b])

    def pick(v, q):
        return v[min(len(v) - 1, int(q * len(v)))] if v else None

    fill, ours = ms(done, "bar_close", "first_fill"), ms(done, "bar_close", "sent")
    return {"n": len(fill), "median": pick(fill, 0.5), "p95": pick(fill, 0.95), "ours": pick(ours, 0.5)}
