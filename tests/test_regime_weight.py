"""P2-5 regime weighting: the weight per entry (pre-registered map, floor, unknown, no look-ahead, fixed at entry, never
an exit), the counted variant in the portfolio backtest, and the acceptance test's nulls and verdict (Advisor spec
quant-review/specs-p2-3-p2-8-p2-5.md section 4)."""

import dataclasses
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
from sleeve_fund.risk import PortfolioProfile

ORDER = dict(side=1, qty=D(2), close=D(100), price=D("100.05"), stop_frac=0.02, step=D("0.001"),
             min_qty=D("0.001"), instrument="XBT", leverage=None, atr_pct=None, taker=D(0))


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
    with pytest.raises(ValueError, match="no weight to carry"):  # never 1.0, never a fresh read (Advisor 19:45)
        w.for_order("b", False, _at(date(2026, 3, 2)))


def test_an_entry_sent_while_a_position_is_held_is_journaled_as_an_add():
    from sleeve_fund.research.portfolio import tagged_intent

    now = _at(date(2026, 3, 2))
    assert tagged_intent({**ORDER, "opens": False}, now).kind == "add"
    assert tagged_intent({**ORDER, "opens": True}, now).kind == "entry"
    assert tagged_intent(ORDER, now).kind == "entry"  # an intent from before P2-5 carries no flag
    assert tagged_intent({**ORDER, "what": "rebalance", "opens": False}, now).kind == "rebalance"


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


def _rows(specs):
    days = pd.date_range(specs[0].prices.index[0].date() - timedelta(days=2), periods=160, freq="D").date
    return {d: ("low" if i % 3 == 0 else "high") for i, d in enumerate(days)}


@pytest.fixture(scope="module")
def runs():
    specs = [_spec("a", 2, model="ping_pong"), _spec("b", 10, model="ping_pong")]
    rows = _rows(specs)
    plain = run_portfolio(specs)
    weighted = run_portfolio(specs, regime=Weighting(rows.get, {"a": "trend", "b": "trend"}))
    return plain, weighted, rows


@pytest.fixture(scope="module")
def ones(runs):
    # The regime run with every weight forced to 1.0: a Weighting with no family registered.
    specs = [_spec("a", 2, model="ping_pong"), _spec("b", 10, model="ping_pong")]
    return run_portfolio(specs, regime=Weighting(runs[2].get, {}))


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


def _spells(length: int, days: int) -> pd.Series:
    cycle = ["low", "mid", "high"]
    return pd.Series([cycle[(i // length) % 3] for i in range(days)], index=pd.date_range("2025-01-01", periods=days))


def test_the_shifted_null_keeps_the_states_and_their_runs_but_moves_their_timing():
    states = pd.Series(["low"] * 10 + ["mid"] * 7 + ["high"] * 8, index=pd.date_range("2025-01-01", periods=25))
    states = pd.concat([states] * 16).set_axis(pd.date_range("2025-01-01", periods=400))
    offsets = rt.shifts(states, np.random.default_rng(7), 20)
    assert len(offsets) == 20 and rt.min_shift(states) == rt.MIN_SHIFT_DAYS
    for k in offsets:
        assert rt.MIN_SHIFT_DAYS <= k <= 400 - rt.MIN_SHIFT_DAYS  # never near N, a small shift the other way
        s = rt.shifted_states(states, k)
        assert (s.value_counts() == states.value_counts()).all() and (s.index == states.index).all()
    with pytest.raises(ValueError):
        rt.shifts(states.iloc[:120], np.random.default_rng(7), 20)


def test_the_minimum_shift_is_three_mean_spells_when_that_is_longer_than_sixty_days():
    states = _spells(40, 600)
    assert rt.mean_spell(states) == 40 and rt.min_shift(states) == 120
    assert all(120 <= k <= 480 for k in rt.shifts(states, np.random.default_rng(1), 50))
    assert rt.min_shift(_spells(5, 600)) == rt.MIN_SHIFT_DAYS


def test_with_fewer_admissible_shifts_than_draws_every_one_is_used_once_and_that_null_is_complete():
    states = _spells(5, 131)  # L = 60: shifts 60..71
    assert rt.shifts(states, np.random.default_rng(1), rt.DRAWS) == list(range(60, 72))
    w, u = _returns(0.002, seed=1), _returns(0.001, seed=1)
    v = rt.judge(w, u, [-1.0] * 12, [-1.0] * rt.DRAWS, True, ENOUGH, rt.daily_sharpe(w), primary_complete=12)
    assert v.result == rt.PASS
    assert rt.judge(w, u, [-1.0] * 11, [-1.0] * rt.DRAWS, True, ENOUGH, rt.daily_sharpe(w),
                    primary_complete=12).result == rt.NOT_JUDGED


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
    v = rt.judge(w, u, nulls, nulls, True, ENOUGH, rt.daily_sharpe(w))
    assert v.result == rt.PASS and not v.reasons
    assert rt.judge(w, u, nulls, nulls, False, ENOUGH, rt.daily_sharpe(w)).result == rt.FAIL  # double fees
    high = [x + 10 for x in nulls]
    assert rt.judge(w, u, high, nulls, True, ENOUGH, rt.daily_sharpe(w)).result == rt.FAIL  # primary null
    assert rt.judge(w, u, nulls, high, True, ENOUGH, rt.daily_sharpe(w)).result == rt.FAIL  # secondary null
    assert rt.judge(u, w, nulls, nulls, True, ENOUGH, rt.daily_sharpe(w)).result == rt.FAIL  # below unweighted
    # The nulls rank the rebuilt regime Sharpe, like for like, never the real run's (Advisor 19:45 UK).
    assert rt.judge(w, u, nulls, nulls, True, ENOUGH, -5.0).result == rt.FAIL


def test_an_unfaithful_rebuild_is_not_judged_whatever_else_holds():
    w, u = _returns(0.002, seed=1), _returns(0.001, seed=1)
    nulls = list(np.linspace(-1, 1, rt.DRAWS))
    v = rt.judge(w, u, nulls, nulls, True, ENOUGH, rt.daily_sharpe(w), ("3 of 40 entries differ",), seed=5)
    assert v.result == rt.NOT_JUDGED and v.reasons == ("3 of 40 entries differ",) and v.seed == 5


def test_a_drawdown_more_than_ten_percent_worse_fails():
    u = pd.Series([0.01] * 50 + [-0.10] + [0.01] * 50)
    w = pd.Series([0.012] * 50 + [-0.115] + [0.012] * 50)  # deeper by more than 10% relative
    nulls = [0.0] * rt.DRAWS
    v = rt.judge(w, u, nulls, nulls, True, ENOUGH, rt.daily_sharpe(w))
    assert v.result == rt.FAIL and any("drawdown" in r for r in v.reasons)


def test_too_few_entries_in_a_state_or_a_short_null_is_not_judged_never_a_pass():
    w, u = _returns(0.002, seed=1), _returns(0.001, seed=1)
    nulls = [-1.0] * rt.DRAWS
    assert rt.judge(w, u, nulls, nulls, True, {**ENOUGH, "mid": 29}, 9.0).result == rt.NOT_JUDGED
    assert rt.judge(w, u, nulls[:-1], nulls, True, ENOUGH, 9.0).result == rt.NOT_JUDGED


def test_the_rebuild_at_weight_one_is_the_unweighted_book(runs):
    plain, _, _ = runs
    t = rt.trades(plain)
    assert t.entries and all(e[0] in ("a", "b") for e in t.entries)
    got = rt.rebuilt_returns(t, [1.0] * len(t.entries))
    real = rt.metrics.daily_returns(plain.book)
    both = got.index.intersection(real.index)
    assert len(both) > 100
    assert np.allclose(got[both].to_numpy(), real[both].to_numpy(), atol=1e-9)


LOOSE = PortfolioProfile(gross=50, net_instrument=50, margin=1.0, open_risk=1.0, drawdown=0.99, daily_loss=0.99)


def _rebuilt_gap(plain, weighted, rows):
    t = rt.trades(plain)
    rebuilt = rt.daily_sharpe(rt.rebuilt_returns(t, rt.regime_weights(t, rows.get, {"a": "trend", "b": "trend"})))
    return abs(rebuilt - rt.daily_sharpe(rt.metrics.daily_returns(weighted.book)))


def test_with_no_limit_binding_the_rebuilt_regime_run_tracks_the_real_one():
    specs = [_spec("a", 2, model="ping_pong"), _spec("b", 10, model="ping_pong")]
    rows = _rows(specs)
    plain = run_portfolio(specs, LOOSE)
    weighted = run_portfolio(specs, LOOSE, regime=Weighting(rows.get, {"a": "trend", "b": "trend"}))
    assert {d["outcome"] for d in plain.decisions} == {"approved"}
    assert _rebuilt_gap(plain, weighted, rows) <= rt.FIDELITY_SHARPE


def test_a_rebuild_off_the_real_regime_run_is_not_judged(runs, ones):
    # Rebuild with a mapping the real regime run didn't use (every day "high", so weight 1.0): the rebuilt Sharpe is
    # the unweighted one, off the real regime run's by more than the fidelity limit (Advisor 19:45 UK).
    plain, weighted, rows = runs
    fams, folds = {"a": "trend", "b": "trend"}, [(plain.book.index[0], plain.book.index[-1])]
    every_high = pd.Series({d: "high" for d in rows})
    assert _rebuilt_gap(plain, weighted, every_high.to_dict()) > rt.FIDELITY_SHARPE
    v = rt.acceptance(plain, weighted, every_high, fams, folds, (plain, weighted), ones=ones, holdout_start=None,
                      draws=5)
    assert v.result == rt.NOT_JUDGED and any("rebuild is off" in r for r in v.reasons)
    assert any("shifted null needs more than" in r for r in v.reasons)  # one spell all along: no primary null


def test_entries_differing_by_more_than_two_percent_is_not_judged(runs, ones):
    plain, weighted, rows = runs
    opening = [d for d in weighted.decisions if d["kind"] == "entry" and d["approved"] > 0]
    fewer = dataclasses.replace(weighted, decisions=[d for d in weighted.decisions if d is not opening[-1]])
    fams, folds = {"a": "trend", "b": "trend"}, [(plain.book.index[0], plain.book.index[-1])]
    v = rt.acceptance(plain, fewer, pd.Series(rows), fams, folds, (plain, fewer), ones=ones, holdout_start=None,
                      draws=5)
    assert v.result == rt.NOT_JUDGED and any("entries differ" in r for r in v.reasons)
    assert v.fidelity["entries_differ"] == 1 and v.fidelity["entries"] == len(opening)  # the figures, not just a fail


def test_the_acceptance_run_is_seeded_and_never_passes_without_the_double_fee_pair(runs, ones):
    plain, weighted, rows = runs
    states = pd.Series(rows)
    fams, folds = {"a": "trend", "b": "trend"}, [(plain.book.index[0], plain.book.index[-1])]
    one = rt.acceptance(plain, weighted, states, fams, folds, None, ones=ones, holdout_start=None, seed=11,
                        draws=30)
    two = rt.acceptance(plain, weighted, states, fams, folds, None, ones=ones, holdout_start=None, seed=11,
                        draws=30)
    assert one == two and one.seed == 11 and one.result == rt.NOT_JUDGED



def test_with_every_weight_forced_to_one_the_run_is_the_unweighted_run_to_the_cent(runs, ones):
    plain, weighted, _ = runs
    assert rt.exact(plain, ones) == []
    assert rt.exact(plain, weighted)  # a real weighting moves the sizes and the book: the check bites
    assert rt.exact(plain, None) == ["the run with every weight forced to 1.0 hasn't run"]
    nudged = dataclasses.replace(ones, book=ones.book + 0.01)
    assert rt.exact(plain, nudged) == ["with every weight forced to 1.0 the book differs from the unweighted run's"]


def test_without_the_forced_one_run_or_with_it_off_the_verdict_is_not_judged(runs, ones):
    plain, weighted, rows = runs
    fams, folds = {"a": "trend", "b": "trend"}, [(plain.book.index[0], plain.book.index[-1])]
    for other in (None, weighted):
        v = rt.acceptance(plain, weighted, pd.Series(rows), fams, folds, (plain, weighted), ones=other,
                          holdout_start=None, draws=5)
        assert v.result == rt.NOT_JUDGED and any("forced to 1.0" in r for r in v.reasons)


def test_the_verdict_never_reads_the_holdout(runs, ones):
    plain, weighted, rows = runs
    fams, first, last = {"a": "trend", "b": "trend"}, plain.book.index[0], plain.book.index[-1]
    with pytest.raises(ValueError, match="held-back period: a fold"):
        rt.acceptance(plain, weighted, pd.Series(rows), fams, [(first, last)], None, ones=ones,
                      holdout_start=last - pd.Timedelta(days=30), draws=5)
    with pytest.raises(ValueError, match="held-back period: a run"):  # the folds stop short, the runs don't
        rt.acceptance(plain, weighted, pd.Series(rows), fams, [(first, last - pd.Timedelta(days=60))], None,
                      ones=ones, holdout_start=last - pd.Timedelta(days=30), draws=5)
    v = rt.acceptance(plain, weighted, pd.Series(rows), fams, [(first, last)], None, ones=ones,
                      holdout_start=last + pd.Timedelta(days=1), draws=5)
    assert v.result == rt.NOT_JUDGED


def test_per_family_figures_are_reported_and_never_decide(runs, ones):
    plain, weighted, rows = runs
    fams, folds = {"a": "trend", "b": "mean-reversion"}, [(plain.book.index[0], plain.book.index[-1])]
    v = rt.acceptance(plain, weighted, pd.Series(rows), fams, folds, None, ones=ones, holdout_start=None, draws=5)
    assert set(v.families) == {"trend", "mean-reversion"}
    assert v.fidelity["entries_differ"] == 0 and 0 <= v.fidelity["sharpe_gap"] < float("inf")  # printed on any result
    for f in v.families.values():
        assert set(f) == {"sharpe", "unweighted_sharpe", "drawdown", "unweighted_drawdown"}
    assert not any(f in r for r in v.reasons for f in ("trend", "mean-reversion"))
