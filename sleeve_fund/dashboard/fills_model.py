"""A8 fills vs model (HoE Done-when; Independent Quant Advisor's ruling, 15 Oct 2026 09:47 UK). Each paper fill against
the price the model decided at, so the PM sees what paper execution costs next to what the backtest assumes.

The model price is the order's signal "price", the last price when the Strategy decided (strategies/base.py _submit; a
bar's close on a bar decision): a "decision" fill. Without it, the stop or trigger the engine read (signal stop_px, else
trigger, as base.py reads a stop): a "trigger" fill. Neither: no model price, shown as such and in no figure. Slippage is
in basis points, adverse positive (a buy above the model, a sell below it). Aggregates are notional-weighted (quantity x
fill price). The delay is the decision to the first fill (order_timings); orders with no timings row (stops, restores,
risk sends) have none. Display only: every figure is read from the journal's orders, fills and timings."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from sleeve_fund import markets
from sleeve_fund.instruments import taker_slippage
from sleeve_fund.store import fills_t, is_backtest, order_timings_t, orders_t
from sleeve_fund.venues import venue as venue_profile

DECISION, TRIGGER = "bar", "trigger"  # model_source: the decision's price, or a stop or trigger level
MIN_FILLS = 20  # Advisor: under this a line says it is too few to judge
ROW_LIMIT = 500  # the page's newest fills; the aggregates cover every fill


def model_price(signal: dict | None) -> tuple[float | None, str | None]:
    """(price, source) the fill is judged against; (None, None) when the order carries neither."""
    sig = signal or {}
    for key, source in (("price", DECISION), ("stop_px", TRIGGER), ("trigger", TRIGGER)):
        v = sig.get(key)
        if v is not None and not isinstance(v, bool):
            try:
                px = float(v)
            except (TypeError, ValueError):
                continue
            if px > 0:
                return px, source
    return None, None


def slip_bp(side: str, model: float, fill: float) -> float:
    """Adverse positive: a buy above the model or a sell below it costs."""
    sign = 1.0 if side == "BUY" else -1.0
    return sign * (fill - model) / model * 1e4


def _assumed(params: dict, venue: str | None) -> dict[str, float]:
    """What the backtest assumes, in bp, by order type: a decision's market order fills at the half spread; a trigger
    fills at the taker's slippage (the half spread, at least its floor; instruments.taker_slippage)."""
    half = markets.half_spread_for(params, venue_profile(venue).assumed_half_spread, venue)
    return {DECISION: half * 1e4, TRIGGER: float(taker_slippage(half)) * 1e4}


class _Agg:
    def __init__(self):
        self.fills, self.notional, self.weighted = 0, 0.0, 0.0

    def add(self, notional: float, bp: float) -> None:
        self.fills += 1
        self.notional += notional
        self.weighted += bp * notional

    def out(self) -> dict:
        return {"fills": self.fills, "slip_bp": self.weighted / self.notional if self.notional else None}


def view(store, now: datetime, limit: int = ROW_LIMIT) -> dict:
    """Every paper fill up to `now`, newest first (the first `limit` as rows), and its aggregates:
    rows: [{order_id, sleeve, instrument, side, ts, qty, model_px, model_source, fill_px, fee, delay_s, slip_bp}];
    by_strategy: {name: {fills, slip_bp, decision: {fills, slip_bp, assumed_bp, gap_bp}, trigger: {...},
                         fees, slip_cost, cost, all_fills}};
    by_instrument: {instrument: {fills, slip_bp}}. fills counts those with a model price; all_fills every one."""
    q = (select(fills_t.c.order_id, fills_t.c.sleeve, fills_t.c.instrument, fills_t.c.side, fills_t.c.ts,
                fills_t.c.qty, fills_t.c.price, fills_t.c.fee, orders_t.c.signal, orders_t.c.venue,
                order_timings_t.c.decided, order_timings_t.c.first_fill)
         .select_from(fills_t.outerjoin(orders_t, orders_t.c.order_id == fills_t.c.order_id)
                      .outerjoin(order_timings_t, order_timings_t.c.order_id == fills_t.c.order_id))
         .where(fills_t.c.ts <= now).order_by(fills_t.c.ts.desc(), fills_t.c.id.desc()))
    with store.engine.connect() as c:
        found = [r for r in c.execute(q) if not is_backtest(r.sleeve)]
    sleeves = {s.name: s for s in store.sleeves()}
    rows, strat, inst, assumed = [], {}, {}, {}
    for r in found:
        s = sleeves.get(r.sleeve)
        instrument = r.instrument or (s.instrument if s else "")
        qty, fill, fee = float(r.qty), float(r.price), float(r.fee)
        model, source = model_price(r.signal)
        bp = None if model is None else slip_bp(r.side, model, fill)
        delay = (None if r.decided is None or r.first_fill is None
                 else (r.first_fill - r.decided).total_seconds())
        rows.append({"order_id": r.order_id, "sleeve": r.sleeve, "instrument": instrument, "side": r.side,
                     "ts": r.ts, "qty": qty, "model_px": model, "model_source": source, "fill_px": fill, "fee": fee,
                     "delay_s": delay, "slip_bp": bp})
        g = strat.setdefault(r.sleeve, {"all": _Agg(), DECISION: _Agg(), TRIGGER: _Agg(), "fees": 0.0,
                                        "slip_cost": 0.0, "all_fills": 0, "venue": r.venue})
        g["fees"] += fee
        g["all_fills"] += 1
        if bp is None:
            continue
        notional = qty * fill
        g["all"].add(notional, bp)
        g[source].add(notional, bp)
        g["slip_cost"] += (1.0 if r.side == "BUY" else -1.0) * (fill - model) * qty
        inst.setdefault(instrument, _Agg()).add(notional, bp)
    by_strategy = {}
    for name, g in strat.items():
        s = sleeves.get(name)
        key = (name, g["venue"])
        if key not in assumed:
            assumed[key] = _assumed(s.params if s else {}, g["venue"] or (s.venue if s else None))
        lines = {}
        for source, label in ((DECISION, "decision"), (TRIGGER, "trigger")):
            o = g[source].out()
            o["assumed_bp"] = assumed[key][source]
            o["gap_bp"] = None if o["slip_bp"] is None else o["slip_bp"] - o["assumed_bp"]
            o["judged"] = o["fills"] >= MIN_FILLS
            lines[label] = o
        by_strategy[name] = {**g["all"].out(), **lines, "fees": g["fees"], "slip_cost": g["slip_cost"],
                             "cost": g["fees"] + g["slip_cost"], "all_fills": g["all_fills"]}
    return {"rows": rows[:limit], "by_strategy": by_strategy,
            "by_instrument": {k: a.out() for k, a in inst.items()}, "min_fills": MIN_FILLS}
