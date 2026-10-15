"""Shadow outcomes: what a trimmed or refused idea would have made (multi-asset spec §5; Advisor MA-SPEC, settled
15 Oct ~00:10 UK), so the record shows what the compliance tests' capacity cost.

- A trim (method trade_scaled): the real trade's P&L after fees per unit it opened, times what the trim held back
  (asked - approved). No replay. Waits until that trade has closed.
- A refusal (method replay, hypothetical): the idea's own planned stop and target replayed on its asset's bars after its
  decision, at the asked size, net of a fee each way. A bar reaching both counts the stop, the worse of the two. Waits
  while neither has been reached, or forever when the idea had no stop and target to replay.

Each outcome is written once (Store.record_shadow_outcome); nothing here writes to a Strategy's book."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
from sqlalchemy import select

from sleeve_fund.money import money
from sleeve_fund.research.metrics import trades
from sleeve_fund.venues import PAPER_PERP_FEES
from sleeve_fund.store import _aware, gate_decisions_t, shadow_outcomes_t, sleeves_t

DEFAULT_FEE = PAPER_PERP_FEES.taker  # a taker fee each way, the paper venue's published VIP 0 (FEE-LIVE)


def _waiting(store) -> list:
    g, t, o = gate_decisions_t, sleeves_t, shadow_outcomes_t
    q = (select(g.c.id, g.c.outcome, g.c.order_id, g.c.requested_qty, g.c.approved_qty, g.c.price, g.c.side,
                g.c.planned_stop, g.c.planned_target, g.c.bar_ts, t.c.name, t.c.instrument)
         .select_from(g.join(t, t.c.id == g.c.sleeve_id).outerjoin(o, o.c.decision_id == g.c.id))
         .where(g.c.outcome.in_(("trimmed", "rejected")), o.c.decision_id.is_(None)).order_by(g.c.id))
    with store.engine.connect() as c:
        return c.execute(q).all()


def trim_outcome(store, d) -> Decimal | None:
    """The trimmed order's trade, once closed: its P&L per unit opened x (asked - approved)."""
    if d.order_id is None:
        return None
    fills = store.fills_after(d.name, 0, limit=100_000)
    first = next((f for f in fills if f["order_id"] == d.order_id), None)
    if first is None:
        return None
    at = _aware(first["ts"])
    for trip in trades(fills, shorts=True):
        if trip["closed"] is not None and _aware(trip["opened"]) <= at <= _aware(trip["closed"]):
            per_unit = Decimal(repr(trip["pnl"])) / Decimal(repr(trip["qty"]))
            return per_unit * (money(d.requested_qty) - money(d.approved_qty))
    return None


def replay_outcome(d, bars: pd.DataFrame | None, fee_rate: Decimal) -> Decimal | None:
    """The refused idea's stop and target on its bars after the decision, at the asked size, net of fees."""
    if bars is None or d.planned_stop is None or d.planned_target is None or d.side not in (1, -1):
        return None
    side, stop, target = d.side, money(d.planned_stop), money(d.planned_target)
    entry, qty = money(d.price), money(d.requested_qty)
    for _, bar in bars[bars.index > pd.Timestamp(_aware(d.bar_ts))].iterrows():
        high, low = money(Decimal(repr(float(bar["high"])))), money(Decimal(repr(float(bar["low"]))))
        stopped = low <= stop if side > 0 else high >= stop
        reached = high >= target if side > 0 else low <= target
        if stopped or reached:
            exit_px = stop if stopped else target
            return side * (exit_px - entry) * qty - fee_rate * (entry + exit_px) * qty
    return None


def fill_outcomes(store, bars: dict[str, pd.DataFrame], fee_rate: Decimal = DEFAULT_FEE) -> dict[str, int]:
    """Write every outcome that can be known now. bars: each instrument's bars (high, low; a UTC index). Returns how
    many were written by each method and how many still wait."""
    out = {"trade_scaled": 0, "replay": 0, "waiting": 0}
    for d in _waiting(store):
        if d.outcome == "trimmed":
            kind, method, pnl = "trim", "trade_scaled", trim_outcome(store, d)
        else:
            kind, method, pnl = "refusal", "replay", replay_outcome(d, bars.get(d.instrument), fee_rate)
        if pnl is None:
            out["waiting"] += 1
            continue
        store.record_shadow_outcome(d.id, kind=kind, pnl=pnl, method=method)
        out[method] += 1
    return out
