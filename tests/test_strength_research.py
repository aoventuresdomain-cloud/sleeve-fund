"""G5 research side (Done-when 5; Advisor ruling G5-DW5, 15 Oct 2026 ~10:07 UK, pe4-plans/g5-dw5-done-when.md): the
strength-sized version of an idea is its own counted variant; random entry draws its sizes from the variant's own sizes;
its tear sheet puts it side by side with the flat version on risk-adjusted terms; it runs the edge chain on its own; and
each out-of-sample trade carries its size factor. Built OFF: a flat study is byte-identical.

Red first on 47c2e498 (head 2): Trade has no size, there is no strength_report, Fold has no trip sizes."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import random_entry as re_mod
from sleeve_fund.research.random_entry import Trade, random_entry

LINEAR = {"mapping": "linear", "floor": 0.3}


def _closes(seed=7, n=400):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(0, 0.01, n))


TRIPS = [(10, 15, 1), (30, 38, 1), (60, 62, -1), (120, 130, 1), (150, 160, 1), (210, 214, -1), (260, 270, 1),
         (300, 305, 1)]
WINDOWS = [(0, 199), (200, 399)]


# ---- DW1: a counted variant of its base idea ---------------------------------------------------------------------------
class _Rows:
    def __init__(self):
        self.rows = []

    def add_trials(self, rows):
        self.rows.extend({"status": "ok", "kind": "single", **r} for r in rows)
        return len(rows)

    def trials(self, idea_hash=None):
        return [r for r in self.rows if idea_hash is None or r["idea_hash"] == idea_hash]


def _count(register, params):
    from sleeve_fund.research.trials import (
        legacy_definition_hash,
        legacy_idea_hash,
        run_setup,
    )

    setup = run_setup(risk_profile="balanced", fee=0.0005, windows=(365, 90, 180))
    register.record(definition_hash=legacy_definition_hash("trend_filter", params, setup),
                    idea_hash=legacy_idea_hash("trend_filter"), name="trend_filter", family="trend", settings=params,
                    dataset="d", stage="in_sample", source="study", sharpe=0.4)


def test_dw1_the_strength_sized_version_is_its_own_counted_variant_of_the_same_idea():
    from sleeve_fund.research.trials import TrialsRegister, legacy_idea_hash

    reg = TrialsRegister(_Rows())
    base = {"fast": 20, "slow": 100}
    _count(reg, base)
    _count(reg, {**base, "signal_strength": LINEAR})
    idea = legacy_idea_hash("trend_filter")
    assert reg.counts(idea)["variants"] == 2 and reg.counts()["ideas"] == 1


def test_dw1_each_distinct_mapping_configuration_tried_counts_once_not_per_band():
    """Advisor Q3: one setting per distinct mapping configuration tried; a 3-band steps mapping is 1, a second set of
    bands makes 2; trying the same one again doesn't add."""
    from sleeve_fund.research.trials import TrialsRegister, legacy_idea_hash

    reg = TrialsRegister(_Rows())
    base = {"fast": 20, "slow": 100}
    _count(reg, base)
    _count(reg, {**base, "signal_strength": {"mapping": "steps", "bands": [0.33, 0.66, 1.0]}})
    assert reg.counts(legacy_idea_hash("trend_filter"))["variants"] == 2
    _count(reg, {**base, "signal_strength": {"mapping": "steps", "bands": [0.25, 0.5, 1.0]}})
    _count(reg, {**base, "signal_strength": {"mapping": "steps", "bands": [0.25, 0.5, 1.0]}})
    assert reg.counts(legacy_idea_hash("trend_filter"))["variants"] == 3


# ---- DW2: random entry draws its sizes from the variant's own ---------------------------------------------------------
def test_dw2_a_flat_benchmark_is_byte_identical_to_before():
    """Pinned on 47c2e498: trades without a size (f = 1) give exactly the same draws and percentiles."""
    trades = [Trade(a, b, s) for a, b, s in TRIPS]
    r = random_entry(_closes(), trades, WINDOWS, 0.0005, draws=200, seed=3)
    assert (r.return_percentile, r.sharpe_percentile) == (34.5, 42.0)
    assert r.strategy_return == -0.08670417367596761 and r.median_random_return == -0.06333866208038946
    ones = random_entry(_closes(), [Trade(a, b, s, size=1.0) for a, b, s in TRIPS], WINDOWS, 0.0005, draws=200, seed=3)
    assert ones == r and r.sizing == ""


SIZES = [0.4, 1.0, 0.6, 0.3, 0.9, 0.5, 0.7, 0.8]


def _reference(closes, sized, windows, cost, draws, seed, flat_draws=False):
    """The benchmark rebuilt from its parts: each draw's trades take a permutation of their own window's sizes."""
    rng = np.random.default_rng(seed)
    per_window, _, _ = re_mod._by_window(sized, windows)
    f = [np.array([t.size for t in w[5]]) for w in per_window]
    actual = np.concatenate([re_mod._trade_returns(closes, e, h, s, cost) * fw
                             for (_, _, e, h, s, _), fw in zip(per_window, f)])
    rets = np.empty(draws)
    for d in range(draws):
        parts = []
        for (start, end, _, holds, sides, _), fw in zip(per_window, f):
            order, entries = re_mod._random_entries(rng, start, end, holds)
            drawn = rng.permutation(fw)
            parts.append(re_mod._trade_returns(closes, entries, holds[order], sides[order], cost)
                         * (np.ones_like(fw) if flat_draws else drawn))
        rets[d] = re_mod._compound(np.concatenate(parts))
    return float((rets < re_mod._compound(actual)).mean() * 100), re_mod._compound(actual)


def test_dw2_each_draw_takes_a_permutation_of_its_windows_own_sizes():
    closes = _closes()
    sized = [Trade(a, b, s, size=f) for (a, b, s), f in zip(TRIPS, SIZES)]
    r = random_entry(closes, sized, WINDOWS, 0.0005, draws=300, seed=5)
    pct, ret = _reference(closes, sized, WINDOWS, 0.0005, 300, 5)
    assert r.return_percentile == pct and r.strategy_return == pytest.approx(ret, abs=1e-12)
    assert "permuted within each window" in r.sizing
    # mutant: draws at full size (f = 1) while the strategy is sized: a different reading
    assert _reference(closes, sized, WINDOWS, 0.0005, 300, 5, flat_draws=True)[0] != r.return_percentile


def test_dw2_the_strategys_return_is_its_trades_at_their_own_sizes():
    closes = _closes()
    sized = [Trade(a, b, s, size=f) for (a, b, s), f in zip(TRIPS, SIZES)]
    unit = [Trade(a, b, s) for a, b, s in TRIPS]
    r_sized = random_entry(closes, sized, WINDOWS, 0.0005, draws=10, seed=1)
    per_trade = np.concatenate([re_mod._trade_returns(closes, e, h, s, 0.0005)
                                for (_, _, e, h, s, _) in re_mod._by_window(unit, WINDOWS)[0]])
    assert r_sized.strategy_return == pytest.approx(float(np.prod(1 + per_trade * np.array(SIZES)) - 1), abs=1e-12)


def test_dw2_fallback_draws_with_replacement_when_the_count_differs_and_says_so():
    rng = np.random.default_rng(0)
    same, how = re_mod._draw_sizes(rng, np.array([0.2, 0.5, 0.9]), 3)
    assert sorted(same) == [0.2, 0.5, 0.9] and how == re_mod.PERMUTED
    other, how = re_mod._draw_sizes(rng, np.array([0.2, 0.5]), 4)
    assert len(other) == 4 and set(other) <= {0.2, 0.5} and how == re_mod.REPLACED


def test_dw2_a_skip_below_the_floor_is_not_a_zero_size_trade():
    """Only trades actually taken carry a size; a size of 0 can't be a trade."""
    with pytest.raises(ValueError, match="size"):
        random_entry(_closes(), [Trade(10, 15, 1, size=0.0)], WINDOWS, 0.0005, draws=5)


# ---- DW4: the edge chain on its own ------------------------------------------------------------------------------------
def test_dw4_the_variant_fails_random_entry_on_its_own_where_the_flat_version_passes():
    """Good timing (each trade bought before a rise) passes flat; sized small on its winners and full on its one loser,
    the variant's own reading fails: it never borrows the flat version's pass."""
    n = 400
    closes = np.full(n, 100.0)
    for a in range(10, 390, 40):
        closes[a + 1:a + 6] = closes[a] * np.linspace(1.01, 1.05, 5)  # a rise right after each entry
        closes[a + 6:] = closes[a + 5]
    closes[370:375] = closes[369] * np.linspace(0.99, 0.80, 5)
    closes[375:] = closes[374]
    trips = [(a, a + 5, 1) for a in range(10, 330, 40)] + [(369, 374, 1)]
    flat = random_entry(closes, [Trade(a, b, s) for a, b, s in trips], [(0, 399)], 0.0005, draws=300, seed=2)
    sized = random_entry(closes, [Trade(a, b, s, size=0.05 if b < 369 else 1.0) for a, b, s in trips], [(0, 399)],
                         0.0005, draws=300, seed=2)
    assert flat.verdict == "PASS" and sized.verdict == "FAIL", (flat.words, sized.words)


# ---- DW3: side by side, risk-adjusted ------------------------------------------------------------------------------------
def _study(returns, exposure_sum, bars, trades=40, params=None, breakeven="breaks even at 0.12% a side"):
    folds = [SimpleNamespace(gross_exposure=exposure_sum, window_bars=bars, test_trades=trades)]
    return SimpleNamespace(oos_returns=pd.Series(returns, index=pd.date_range("2025-01-01", periods=len(returns),
                                                                                freq="D", tz="UTC")),
                           folds=folds, breakeven=breakeven, chosen_params=params or {}, default_params=params or {})


def test_dw3_the_stats_are_risk_adjusted_with_exposure_time_weighted_and_flat_time_zero():
    from sleeve_fund.research import strength_report as sr

    s = sr.stats(_study([0.01, -0.005, 0.0, 0.002] * 10, exposure_sum=20.0, bars=100))
    assert s.exposure == pytest.approx(0.2)  # mean |notional / equity| over every test bar, flat bars 0
    assert s.per_exposure == pytest.approx(s.net_return / 0.2)
    assert s.trades == 40 and math.isfinite(s.sharpe)


def _pair(flat_ret, sized_ret, flat_exp=50.0, sized_exp=25.0):
    from sleeve_fund.research import strength_report as sr

    flat = _study(flat_ret, exposure_sum=flat_exp, bars=100)
    sized = _study(sized_ret, exposure_sum=sized_exp, bars=100, params={"signal_strength": LINEAR})
    return sr.compare(flat, sized, seed=0)


def test_dw3_kept_only_if_it_beats_the_flat_version_on_sharpe_and_on_return_per_unit_of_exposure():
    rng = np.random.default_rng(1)
    base = rng.normal(0.001, 0.01, 120)
    better = _pair(base, base * 0.5 + 0.0004)  # half the exposure, better Sharpe and better per unit
    assert better.verdict == "kept", better.lines
    worse_sharpe = _pair(base, base * 0.5 - 0.0004)
    assert worse_sharpe.verdict == "flat stays"


def test_dw3_a_tie_means_flat_stays():
    base = np.random.default_rng(2).normal(0.001, 0.01, 120)
    tie = _pair(base, base, flat_exp=50.0, sized_exp=50.0)
    assert tie.verdict == "flat stays"


def test_dw3_without_a_flat_version_to_compare_the_flat_version_stays():
    from sleeve_fund.research import strength_report as sr

    out = sr.compare(None, _study([0.01, 0.0] * 20, 10.0, 100, params={"signal_strength": LINEAR}))
    assert out.verdict == "flat stays" and "no flat version" in " ".join(out.lines)


def test_dw3_the_paired_difference_carries_a_block_bootstrap_90pc_interval():
    """Advisor FOLLOW-UP (information only): strength minus flat, day by day, with a block-bootstrap 90% interval."""
    base = np.random.default_rng(3).normal(0.001, 0.01, 120)
    out = _pair(base, base + 0.001)
    lo, hi = out.interval
    assert lo == pytest.approx(0.001, abs=1e-12) and hi == pytest.approx(0.001, abs=1e-12)  # a constant difference
    noisy = _pair(base, base + np.random.default_rng(4).normal(0.0, 0.002, 120))
    assert noisy.interval[0] < noisy.interval[1]
    assert any("90%" in line for line in noisy.lines)


def test_dw3_the_section_is_a_table_of_both_versions_with_the_verdict():
    from sleeve_fund.research import strength_report as sr

    base = np.random.default_rng(1).normal(0.001, 0.01, 120)
    flat = _study(base, 50.0, 100)
    sized = _study(base * 0.5 + 0.0004, 25.0, 100, params={"signal_strength": LINEAR})
    text = "\n".join(sr.section(sized, flat))
    for row in ("Sharpe", "Return per unit of exposure", "Net return", "Average gross exposure", "Trades",
                "Break-even fee", "Strength minus flat"):
        assert row in text, text
    assert "| Flat | Strength-sized |" in text and "Verdict: kept" in text


def test_dw3_a_flat_study_has_no_strength_section():
    from sleeve_fund.research import strength_report as sr

    assert sr.section(_study([0.01, 0.0] * 20, 10.0, 100), None) == []


# ---- DW5: each trip carries its size factor ------------------------------------------------------------------------------
def test_dw5_a_trips_size_factor_is_its_entry_orders():
    from sleeve_fund.research.study import _trip_sizes

    t0 = pd.Timestamp("2025-01-02", tz="UTC")
    fills = pd.DataFrame({"ts_last": [t0, t0 + pd.Timedelta(hours=5), t0 + pd.Timedelta(days=1)]},
                         index=["o1", "o2", "o3"])
    decisions = {"o1": {"signal": {"size_factor": 0.62}}, "o2": {"signal": {}},
                 "o3": {"signal": {"size_factor": 0.4}}}
    trips = [{"opened": t0, "closed": t0 + pd.Timedelta(hours=5), "side": 1},
             {"opened": t0 + pd.Timedelta(days=1), "closed": None, "side": 1}]
    assert _trip_sizes(trips, t0 - pd.Timedelta(minutes=1), fills, decisions) == [(t0, 0.62)]


# ---- end to end: a study of the strength-sized variant beside its flat version ----------------------------------------
def _strength_trend(monkeypatch):
    """trend_filter with a strength from its decision bar's close (a closed bar only), as a strength block would."""
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.strategies.trend_filter import TrendFilter, TrendFilterConfig

    class Sized(TrendFilter):
        def explain(self, bar, target):
            reason, values = super().explain(bar, target)
            return reason, {**values, "strength": 0.35 + (int(bar.close.as_double()) % 7) / 10}

    monkeypatch.setitem(REGISTRY, "trend_filter", (Sized, TrendFilterConfig))


def test_e2e_a_sized_study_carries_its_trip_sizes_exposure_and_the_side_by_side_section(tmp_path, instrument,
                                                                                         monkeypatch):
    import dataclasses

    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.tearsheet import render
    from sleeve_fund.strategies.trend_filter import SPEC

    _strength_trend(monkeypatch)
    prices = synthetic_ohlcv(days=900, seed=3)
    grid = {"fast": [5], "slow": [20]}
    flat_spec = dataclasses.replace(SPEC, param_grid=grid)
    sized_spec = dataclasses.replace(SPEC, param_grid={**grid, "signal_strength": [LINEAR]})
    common = {"dataset": "syn", "synthetic": True, "holdout_days": 0, "train_days": 365, "test_days": 180}
    flat = run_study(flat_spec, prices, instrument, ledger=IdeaLedger(tmp_path / "f.jsonl"), **common)
    sized = run_study(sized_spec, prices, instrument, ledger=IdeaLedger(tmp_path / "s.jsonl"), **common)

    assert all(not f.trip_sizes for f in flat.folds) and flat.random_entry.sizing == ""
    sizes = [f for fold in sized.folds for _, f in fold.trip_sizes]
    assert sizes and all(0.3 <= f <= 1 for f in sizes) and len(set(sizes)) > 1, sizes
    assert sum(f.test_trades for f in sized.folds) == len(sizes)  # every counted trip carries its own f
    assert "permuted within each window" in sized.random_entry.sizing
    flat_exp = sum(f.gross_exposure for f in flat.folds)
    assert 0 < sum(f.gross_exposure for f in sized.folds) < flat_exp  # strength only ever sizes down

    sheet = render(sized, IdeaLedger(tmp_path / "s.jsonl"), flat=flat)
    assert "## Sizing by signal strength, side by side with the flat version" in sheet and "Verdict: " in sheet
    assert "Sizing by signal strength" not in render(flat, IdeaLedger(tmp_path / "f.jsonl"))
    assert "Verdict: flat stays" in render(sized, IdeaLedger(tmp_path / "s.jsonl"))  # no flat version given
