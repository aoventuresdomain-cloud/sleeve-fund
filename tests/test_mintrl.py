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


def test_r5_a_hand_computed_reference_case_matches_to_1e_6():
    """HoE Done-when (15 Oct): 40 days of 2%, -1%, 0.5%, 0% repeating, worked by hand from the sample moments (Sharpe
    0.342053, adjusted skew 0.336052, excess kurtosis -1.057575, z 1.644854): MinTRL 22.103668 days, so 40 is enough."""
    got = min_trl(_series([0.02, -0.01, 0.005, 0.0] * 10))
    assert got.days_needed == pytest.approx(22.103668, abs=1e-6)
    assert got.sharpe == pytest.approx(0.342053, abs=1e-6) and got.days_have == 40 and got.met


def test_r5_symmetric_fat_tails_alone_lengthen_the_track_at_the_same_sharpe_and_no_skew():
    """HoE Done-when: fat tails lengthen the track against normal-shaped returns of the same Sharpe; here with no skew
    on either, so the kurtosis term alone does it."""
    def scaled(z):
        z = np.asarray(z, dtype=float)
        return _series(0.001 + 0.01 * (z - z.mean()) / z.std(ddof=1))

    thin = scaled(np.tile([1.0, -1.0], 200))
    fat = scaled(np.tile([4.0] + [0.0] * 18 + [-4.0] + [0.0] * 18, 10)[:400])
    assert abs(thin.skew()) < 1e-9 and abs(fat.skew()) < 1e-9 and fat.kurt() > thin.kurt() + 5
    assert min_trl(fat).sharpe == pytest.approx(min_trl(thin).sharpe)
    # the kurtosis term alone: z^2 x (kurtosis fat - kurtosis thin) / 4, about 12 days here, not float noise
    z = NormalDist().inv_cdf(0.95)
    gap = min_trl(fat).days_needed - min_trl(thin).days_needed
    assert gap == pytest.approx(z**2 * (fat.kurt() - thin.kurt()) / 4, rel=1e-9) and gap > 10


def test_r5_too_few_returns_read_as_not_computed_on_the_g1_row():
    """HoE Done-when ("n/a"): under 30 days there is no figure, and the G1 row says so instead of a number."""
    row = mintrl.g1_row(_series(np.random.default_rng(6).normal(0.002, 0.01, MIN_DAYS - 1)))
    assert row[1] == "INFO" and row[2].startswith("not computed") and "days" in row[2] and not any(
        w.strip(",").isdigit() and int(w.strip(",")) > MIN_DAYS for w in row[2].split())


def test_r5_the_g1_row_names_its_formula_estimator_and_z_with_the_daily_sharpe():
    """Advisor (15 Oct): the row names what it computes, so the figure can be checked: the MinTRL formula (Bailey and
    Lopez de Prado), the bias-adjusted sample skew and kurtosis it uses, z at 95%, and the Sharpe in the same daily
    units as the zero it is tested against."""
    words = mintrl.g1_row(_series([0.02, -0.01, 0.005, 0.0] * 10))[2]
    assert "Bailey and Lopez de Prado" in words and "bias-adjusted sample skew and kurtosis" in words
    assert "z 1.645" in words and "daily Sharpe 0.342" in words
