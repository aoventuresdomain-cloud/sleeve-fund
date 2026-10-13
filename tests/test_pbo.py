"""R4: the probability of backtest overfitting (research.pbo), PE1's cells R4-1..R4-8."""

import itertools

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.pbo import (
    NA,
    PBO_CHECK,
    TOO_FEW,
    _ranks,
    g1_row,
    pbo,
    verdict_of,
)
from sleeve_fund.research.study import grid, run_study
from sleeve_fund.research.tearsheet import g1_verdict, render
from sleeve_fund.strategies.trend_filter import SPEC

DAYS = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")


def _reference(x: np.ndarray, blocks: int) -> float:
    """CSCV the long way: every split, Sharpes from scratch, ranks by counting."""
    parts = np.array_split(np.arange(len(x)), blocks)
    n, below, total = x.shape[1], 0, 0
    for half in itertools.combinations(range(blocks), blocks // 2):
        ins = np.concatenate([parts[b] for b in half])
        out = np.concatenate([parts[b] for b in range(blocks) if b not in half])

        def sharpes(rows):
            m, s = x[rows].mean(axis=0), x[rows].std(axis=0, ddof=1)
            return [m[j] / s[j] if s[j] > 1e-15 else 0.0 for j in range(n)]

        a, b = sharpes(ins), sharpes(out)
        best = max(range(n), key=lambda j: (a[j], -j))
        rank = sum(1 for v in b if v < b[best]) + (sum(1 for v in b if v == b[best]) + 1) / 2
        w = rank / (n + 1)
        below += np.log(w / (1 - w)) <= 0
        total += 1
    return below / total


def _frame(x):
    return pd.DataFrame(x, index=DAYS[: len(x)], columns=[f"v{i}" for i in range(x.shape[1])])


def test_r4_1_it_matches_cscv_computed_the_long_way():
    x = np.random.default_rng(7).normal(0.001, 0.01, size=(64, 10))
    x[:, 3] += np.linspace(-0.01, 0.01, 64)  # one configuration whose edge flips half-way
    got = pbo(_frame(x), blocks=8)
    assert got.value == pytest.approx(_reference(x, 8), abs=1e-9)
    assert got.n_variants == 10 and got.n_combos == 70


def test_r4_1_when_the_in_sample_best_is_the_worst_out_of_sample_it_is_overfit():
    """Configuration j earns about +j a day in the first half of the days and -j in the second: whichever half of the
    blocks holds more early days, its best is the other half's worst."""
    rng = np.random.default_rng(2)
    j = np.arange(10, dtype=float) * 0.001
    x = np.vstack([j + rng.normal(0, 0.002, size=(32, 10)), -j + rng.normal(0, 0.002, size=(32, 10))])
    got = pbo(_frame(x), blocks=8)
    assert got.value == pytest.approx(_reference(x, 8), abs=1e-9)
    assert got.value > 0.7 and got.verdict == "FAIL"  # the splits with two early blocks a side are a coin toss


def test_r4_2_pure_noise_is_overfit_about_half_the_time():
    values = [pbo(_frame(np.random.default_rng(seed).normal(0, 0.01, size=(320, 20)))).value for seed in range(8)]
    assert 0.4 < np.mean(values) < 0.6


def test_r4_3_one_dominant_configuration_is_not_overfit():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.01, size=(320, 12))
    x[:, 4] = rng.normal(0.01, 0.002, size=320)  # better than the rest on every block
    got = pbo(_frame(x))
    assert got.value == 0.0 and got.verdict == "PASS" and got.n_combos == 12_870


def test_r4_4_too_few_variants_or_days_give_no_number():
    x = np.random.default_rng(1).normal(0, 0.01, size=(320, 9))
    got = pbo(_frame(x))
    assert (got.value, got.verdict) == (None, TOO_FEW) and "fewer than the 10" in got.words
    short = pbo(_frame(np.random.default_rng(1).normal(0, 0.01, size=(31, 12))))
    assert (short.value, short.verdict) == (None, NA) and "31 days" in short.words
    assert g1_row(got)[1] == "N/A" and g1_row(None) == (PBO_CHECK, "N/A", "not computed for this study")


def test_r4_5_thresholds_at_their_boundaries():
    assert [verdict_of(v) for v in (0.0, 0.2499, 0.25, 0.5, 0.5001, 1.0)] == [
        "PASS", "PASS", "WARN", "WARN", "FAIL", "FAIL"]


def test_r4_6_a_fail_fails_g1_and_a_warn_does_not():
    ok = ("Something else", "PASS", "")
    assert g1_verdict([ok, (PBO_CHECK, "FAIL", "x")]) == ("FAIL", [PBO_CHECK])
    assert g1_verdict([ok, (PBO_CHECK, "WARN", "x")]) == ("PASS", [])
    assert g1_verdict([ok, (PBO_CHECK, "N/A", "x")]) == ("PASS", [])


def test_r4_8_ties_take_the_average_rank():
    assert _ranks(np.array([[0.5, 0.1, 0.5, 0.9]])).tolist() == [[2.5, 1.0, 2.5, 4.0]]
    flat = np.zeros((64, 10))  # every Sharpe 0: the pick ranks 5.5 of 10, the median, which counts as overfit
    assert pbo(_frame(flat), blocks=8).value == 1.0


def test_the_same_returns_give_the_same_value_and_gaps_are_dropped_together():
    x = np.random.default_rng(3).normal(0, 0.01, size=(320, 10))
    f = _frame(x)
    assert pbo(f) == pbo(f.copy())
    gappy = f.copy()
    gappy.iloc[0, 3] = np.nan
    assert pbo(gappy) == pbo(f.iloc[1:])


def test_r4_7_the_study_feeds_every_configuration_tried_and_shows_it(tmp_path, instrument):
    prices = synthetic_ohlcv(days=1900, seed=3)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, prices, instrument, dataset="syn", ledger=ledger, synthetic=True, holdout_days=200,
                  train_days=730, test_days=365)
    assert r.pbo.n_variants == len(grid(SPEC.param_grid))  # every grid point, the discarded ones too
    row = next(c for c in render(r, ledger).splitlines() if c.startswith(f"| {PBO_CHECK} |"))
    assert r.pbo.words in row or r.pbo.verdict in row
