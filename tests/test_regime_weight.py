"""P2-5 regime weighting: the weight per entry (pre-registered map, floor, unknown, no look-ahead, fixed at entry, never
an exit), the counted variant in the portfolio backtest, and the acceptance test's nulls and verdict (Advisor spec
quant-review/specs-p2-3-p2-8-p2-5.md section 4)."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest
from test_portfolio_run import _spec

from sleeve_fund.gate_ledger import TaggedIntent
from sleeve_fund.portfolio.regime_weight import FLOOR, FULL, MID, Weighting, weighed, weight
from sleeve_fund.research import regime_test as rt
from sleeve_fund.research.portfolio_run import run_portfolio


def _at(d: date, hour: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc)


@pytest.mark.parametrize("family,state,w", [
    ("trend", "high", FULL), ("trend", "mid", MID), ("trend", "low", FLOOR),
    ("breakout", "high", FULL), ("breakout", "low", FLOOR),
    ("mean-reversion", "low", FULL), ("mean-reversion", "mid", MID), ("mean-reversion", "high", FLOOR),
    ("trend", "unknown", FULL), ("trend", None, FULL), ("mean-reversion", "unknown", FULL),
    ("pullback-in-trend", "low", FULL), ("benchmark", "low", FULL),  # no pre-registered direction: not weighted
])
def test_the_pre_registered_map_with_its_floor_and_unknown_at_one(family, state, w):
    assert weight(family, state) == w
    assert D("0.25") <= weight(family, state) <= D("1")


def test_an_entry_uses_the_row_dated_its_day_which_reads_only_the_prior_close():
    # The row dated D is built from closes through D-1 (the pre-registered lag); the row dated D+1 is the future.
    rows = {date(2026, 3, 2): "low", date(2026, 3, 3): "high"}
    w = Weighting(rows.get, {"a": "trend"})
    assert w.for_order("a", True, _at(date(2026, 3, 2), 23)) == (FLOOR, "low")
    assert w.for_order("a", True, _at(date(2026, 3, 3))) == (FULL, "high")
    assert w.for_order("a", True, _at(date(2026, 3, 9))) == (FULL, "unknown")  # no row: unknown, weight 1.0


def test_the_weight_is_fixed_at_entry_and_an_add_reuses_it():
    rows = {date(2026, 3, 2): "low", date(2026, 3, 3): "high"}
    w = Weighting(rows.get, {"a": "trend", "b": "trend"})
    assert w.for_order("a", True, _at(date(2026, 3, 2))) == (FLOOR, "low")
    assert w.for_order("a", False, _at(date(2026, 3, 3))) == (FLOOR, "low")  # the add, a day later in "high"
    assert w.for_order("b", False, _at(date(2026, 3, 2))) == (FULL, "unknown")  # opened before weighting began


def test_a_weighed_order_rounds_down_to_the_step_and_keeps_its_weight_for_the_journal():
    order = TaggedIntent("BTC", 1, D("1.003"), D("100"), D("100"), D("2"), D("0.001"), D("0.001"))
    out = weighed(order, MID, "mid")
    assert (out.qty, out.regime_weight, out.regime_state) == (D("0.601"), MID, "mid")
    assert out.price == order.price and out.risk_per_unit == order.risk_per_unit


def test_an_exit_never_reaches_the_weight():
    from sleeve_fund.paper.portfolio_gate import NEVER_GATED, PaperGate

    gate = object.__new__(PaperGate)  # an exit returns before any field is read
    for what in NEVER_GATED:
        assert gate("a", {"what": what, "qty": D(1)}, 0) is None
    assert gate("a", {"what": "entry", "reduce_only": True, "qty": D(1)}, 0) is None


@pytest.fixture(scope="module")
def runs():
    specs = [_spec("a", 2, model="ping_pong"), _spec("b", 10, model="ping_pong")]
    days = pd.date_range(specs[0].prices.index[0].date() - timedelta(days=2), periods=160, freq="D").date
    rows = {d: ("low" if i % 3 == 0 else "high") for i, d in enumerate(days)}
    plain = run_portfolio(specs)
    weighted = run_portfolio(specs, regime=Weighting(rows.get, {"a": "trend", "b": "trend"}))
    return plain, weighted, rows


def test_unweighted_is_the_default_and_records_no_weight(runs):
    plain, _, _ = runs
    assert plain.decisions and all(d["regime_weight"] is None and d["regime_state"] is None for d in plain.decisions)


def test_the_variant_weighs_each_opening_entry_by_its_days_regime(runs):
    plain, weighted, rows = runs
    opened = [d for d in weighted.decisions if d["regime_state"] in ("low", "high")]
    assert {d["regime_state"] for d in opened} == {"low", "high"}
    for d in opened:
        assert d["regime_state"] == rows[d["bar_ts"].date()]
        assert d["regime_weight"] == (FLOOR if d["regime_state"] == "low" else FULL)
    low = [d for d in opened if d["regime_state"] == "low"]
    assert all(d["requested"] <= D("0.25") * max(p["requested"] for p in plain.decisions) + D("0.001") for d in low)


def test_the_shifted_null_keeps_the_states_and_their_runs_but_moves_their_timing():
    states = pd.Series(["low"] * 50 + ["mid"] * 70 + ["high"] * 80, index=pd.date_range("2025-01-01", periods=200))
    rng = np.random.default_rng(7)
    for _ in range(20):
        s = rt.shifted_states(states, rng)
        assert (s.value_counts() == states.value_counts()).all()
        assert (s != states).any() and (s.index == states.index).all()
        k = next(k for k in range(200) if (np.roll(states.to_numpy(), k) == s.to_numpy()).all())
        assert rt.MIN_SHIFT_DAYS <= k <= 200 - rt.MIN_SHIFT_DAYS
    with pytest.raises(ValueError):
        rt.shifted_states(states.iloc[:120], rng)


def test_the_iid_null_draws_with_the_regime_runs_frequencies():
    used = [D("0.25")] * 300 + [D("0.6")] * 200 + [D("1.0")] * 500
    got = rt.iid_weights(used, np.random.default_rng(3))
    shares = {w: got.count(w) / len(got) for w in (0.25, 0.6, 1.0)}
    assert len(got) == 1000 and shares[0.25] == pytest.approx(0.3, abs=0.05) and shares[1.0] == pytest.approx(0.5,
                                                                                                             abs=0.05)


def _returns(mu, sd=0.01, n=400, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n))


ENOUGH = {"low": 30, "mid": 30, "high": 30}


def test_a_pass_needs_every_condition():
    w, u = _returns(0.002, seed=1), _returns(0.001, seed=1)
    nulls = list(np.linspace(-1, 1, rt.DRAWS))
    v = rt.judge(w, u, nulls, nulls, True, ENOUGH)
    assert v.result == rt.PASS and not v.reasons
    assert rt.judge(w, u, nulls, nulls, False, ENOUGH).result == rt.FAIL  # double fees
    high = [x + 10 for x in nulls]
    assert rt.judge(w, u, high, nulls, True, ENOUGH).result == rt.FAIL  # primary null
    assert rt.judge(w, u, nulls, high, True, ENOUGH).result == rt.FAIL  # secondary null
    assert rt.judge(u, w, nulls, nulls, True, ENOUGH).result == rt.FAIL  # below unweighted


def test_a_drawdown_more_than_ten_percent_worse_fails():
    u = pd.Series([0.01] * 50 + [-0.10] + [0.01] * 50)
    w = pd.Series([0.012] * 50 + [-0.115] + [0.012] * 50)  # deeper by more than 10% relative
    nulls = [0.0] * rt.DRAWS
    v = rt.judge(w, u, nulls, nulls, True, ENOUGH)
    assert v.result == rt.FAIL and any("drawdown" in r for r in v.reasons)


def test_too_few_entries_in_a_state_or_a_short_null_is_not_judged_never_a_pass():
    w, u = _returns(0.002, seed=1), _returns(0.001, seed=1)
    nulls = [-1.0] * rt.DRAWS
    assert rt.judge(w, u, nulls, nulls, True, {**ENOUGH, "mid": 29}).result == rt.NOT_JUDGED
    assert rt.judge(w, u, nulls[:-1], nulls, True, ENOUGH).result == rt.NOT_JUDGED
