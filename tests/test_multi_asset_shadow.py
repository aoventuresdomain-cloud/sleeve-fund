"""Multi-asset spec §5 on DA 0028: a trimmed or refused idea is kept for its shadow outcome. The gate's decision row
carries the idea itself (its side, planned stop and planned target), so a refusal's outcome can be replayed later and a
trim's scaled from the trade that did go (Advisor MA-SPEC). Paper's gate and the portfolio backtest's build the same
tagged intent (research.portfolio.tagged_intent), so both write it."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal as D

from sleeve_fund.portfolio.gate import check_order
from sleeve_fund.portfolio.limits import Holding
from sleeve_fund.research.portfolio import tagged_intent
from sleeve_fund.risk import PORTFOLIO
from test_gate_ledger import _marked, tagged
from test_portfolio_gate import T0, _buy

NOW = datetime(2026, 10, 15, tzinfo=UTC)


def _idea(side, stop_frac=0.02, tp_frac=0.04):
    return {"side": side, "qty": D("0.5"), "close": D(100), "price": D(100), "stop_frac": stop_frac, "tp_frac": tp_frac,
            "what": "entry", "step": D("0.001"), "min_qty": D("0.001"), "instrument": "ETH/USDT", "opens": True,
            "leverage": None, "atr_pct": None, "taker": D(0)}


def test_the_tagged_intent_carries_the_planned_stop_and_target_on_the_ideas_side():
    long, short = tagged_intent(_idea(1), NOW), tagged_intent(_idea(-1), NOW)
    assert (long.planned_stop, long.planned_target) == (D(98), D(104))
    assert (short.planned_stop, short.planned_target) == (D(102), D(96))
    bare = tagged_intent(_idea(1, stop_frac=None, tp_frac=None) | {"atr_pct": 0.03}, NOW)
    assert (bare.planned_stop, bare.planned_target) == (None, None)  # a stopless idea with no target has neither


def test_a_refused_idea_is_journalled_with_its_side_stop_and_target_and_listed_for_its_shadow_outcome():
    ledger = _marked()
    ledger.held = [Holding("BTC", D(1_000_000), D(1_000), D(15))]  # far over the gross cap: the next entry is refused
    it = tagged(_buy("0.1", side=-1), T0, planned_stop=D(61_200), planned_target=D(57_600))
    c = check_order(ledger, "a", it, PORTFOLIO, T0)
    assert c.decision.outcome == "rejected"
    [row] = ledger.store.shadow_outcomes("a")
    assert (row["side"], row["planned_stop"], row["planned_target"]) == (-1, D(61_200), D(57_600))
    assert row["kind"] is None  # its outcome comes later (record_shadow_outcome)


def test_an_approved_idea_is_journalled_with_its_side_too():
    ledger = _marked()
    c = check_order(ledger, "a", tagged(_buy("0.1"), T0, planned_stop=D(58_800)), PORTFOLIO, T0)
    assert c.decision.outcome == "approved"
    from sqlalchemy import select

    from sleeve_fund.store import gate_decisions_t

    with ledger.engine.connect() as conn:
        row = conn.execute(select(gate_decisions_t)).one()
    assert (row.side, row.planned_stop, row.planned_target) == (1, D(58_800), None)
