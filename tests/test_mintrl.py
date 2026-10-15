"""R5: minimum track record length (Advisor 6 Oct 14:05): 95%, fat-tail adjusted, daily returns. Shown at G1, never a
G1 pass or fail; at G2 a strategy stays at minimum size until its own track is that long."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import mintrl
from sleeve_fund.research.mintrl import CHECK, MIN_DAYS, may_scale, min_trl


def _series(x):
    return pd.Series(np.asarray(x, dtype=float), index=pd.date_range("2025-01-01", periods=len(x), freq="D"))


def test_min_trl_matches_the_hand_formula_with_the_returns_own_skew_and_kurtosis():
    r = _series(np.random.default_rng(4).standard_t(4, 400) * 0.01 + 0.002)
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = r.skew(), r.kurt() + 3
    z = NormalDist().inv_cdf(0.95)
    want = 1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (z / sr) ** 2
    got = min_trl(r)
    assert got.days_needed == pytest.approx(want) and got.sharpe == pytest.approx(sr) and got.days_have == 400
    assert f"{math.ceil(want):,} days" in got.words and "95% confidence" in got.words


def test_fat_tails_and_negative_skew_need_a_longer_track_than_normal_returns_of_the_same_sharpe():
    normal = _series(np.random.default_rng(1).normal(0.001, 0.01, 2000))
    crashy = normal.copy()
    crashy.iloc[::50] -= 0.06  # occasional large losses: negative skew, fat tails
    crashy += (normal.mean() / normal.std() * crashy.std()) - crashy.mean()  # the same Sharpe again
    assert crashy.mean() / crashy.std() == pytest.approx(normal.mean() / normal.std())
    assert min_trl(crashy).days_needed > min_trl(normal).days_needed


def test_no_edge_is_never_and_too_few_days_is_not_computed():
    flat = min_trl(_series(np.random.default_rng(2).normal(-0.001, 0.01, 300)))
    assert flat.days_needed is None and not flat.met and flat.words.startswith("never")
    short = min_trl(_series(np.random.default_rng(2).normal(0.001, 0.01, MIN_DAYS - 1)))
    assert short.days_needed is None and short.words.startswith("not computed")


def test_the_g1_row_is_shown_only_never_a_pass_or_fail():
    from sleeve_fund.research.tearsheet import g1_verdict

    row = mintrl.g1_row(_series(np.random.default_rng(3).normal(0.001, 0.01, 300)))
    assert row[:2] == (CHECK, "INFO")
    assert g1_verdict([("Something else", "PASS", ""), row]) == ("PASS", [])
    assert g1_verdict([("Something else", "PASS", ""), mintrl.g1_row(_series([0.0] * 5))]) == ("PASS", [])


def test_g2_a_strategy_scales_only_once_its_own_track_is_at_least_its_minimum_track_record():
    rng = np.random.default_rng(5)
    strong = _series(rng.normal(0.004, 0.01, 400))  # a high Sharpe: a short track is enough
    weak = _series(np.tile([0.0105, -0.0095], 60))  # Sharpe 0.05 a day: about 1,100 days needed, 120 had
    ok, why = may_scale(strong)
    assert ok and why.startswith("may scale")
    ok, why = may_scale(weak)
    assert not ok and "stays at minimum size" in why and "of the" in why
    assert may_scale(_series([0.001] * 10))[0] is False  # too short to measure: never scales
