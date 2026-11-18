"""N5 RETURN Z-SCORE (Advisor 17 Nov): z = r_t / sd(r_t-N ... r_t-1), simple close-to-close returns on the block's
timeframe, sample sd (ddof 1) over the N returns BEFORE the current one, no mean subtracted. No value until N earlier
returns exist (N + 2 closes) or while they don't vary. Studies only until the PM approves it for paper."""

from __future__ import annotations

import numpy as np
import pytest

from sleeve_fund.dashboard import strategy_words
from sleeve_fund.strategies import study_only_uses
from sleeve_fund.strategies.definitions import (
    CATALOGUE,
    ReturnZScore,
    check_definition,
    to_params,
)


def _feed(closes, period=20):
    b, out = ReturnZScore(period=period), []
    for c in closes:
        b.update_ohlcv(c, c, c, c, 1.0)
        out.append((b.values["z"], b.values["ret"]) if b.initialized else None)
    return b, out


def _hand(closes, period):
    r = np.diff(np.asarray(closes, float)) / np.asarray(closes[:-1], float)
    out = [None, *[None] * len(r)]
    for t in range(period, len(r)):
        out[t + 1] = (r[t] / np.std(r[t - period:t], ddof=1), r[t])
    return out


def test_it_is_the_hand_formula_with_its_first_value_after_period_plus_two_closes():
    rng = np.random.default_rng(9)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 80))))
    b, got = _feed(closes)
    want = _hand(closes, 20)
    assert [g is None for g in got] == [w is None for w in want]
    assert got.index(next(g for g in got if g is not None)) == 21 and b.warmup_bars == 22
    for g, w in zip(got, want, strict=True):
        if w is not None:
            assert g == pytest.approx(w, rel=1e-12)


def test_the_current_return_is_left_out_of_its_own_sd_and_enters_the_next_candles():
    """A 20% day after 20 quiet ones: its z is r over the quiet days' sd alone. Were it in its own sd, the z would be
    far smaller; the day after, it is in the window, and a quiet return reads small."""
    quiet = [100.0 * (1.001 if i % 2 else 0.999) ** 1 for i in range(22)]
    closes = [100.0]
    for q in quiet[1:]:
        closes.append(closes[-1] * q / 100.0)
    closes += [closes[-1] * 1.2, closes[-1] * 1.2 * 1.001]
    _, got = _feed(closes)
    r = np.diff(closes) / np.asarray(closes[:-1])
    excl = r[-2] / np.std(r[-22:-2], ddof=1)
    incl = r[-2] / np.std(r[-21:-1], ddof=1)
    assert got[-2][0] == pytest.approx(excl) and abs(got[-2][0] - incl) > 10 * abs(incl)
    assert got[-1][0] == pytest.approx(r[-1] / np.std(r[-21:-1], ddof=1)) and abs(got[-1][0]) < 1


def test_no_mean_is_subtracted():
    closes = [100.0]
    for i in range(30):
        closes.append(closes[-1] * (1.02 if i % 2 else 1.01))  # every return positive, mean 1.5%
    _, got = _feed(closes)
    r = np.diff(closes) / np.asarray(closes[:-1])
    assert got[-1][0] == pytest.approx(r[-1] / np.std(r[-21:-1], ddof=1))  # not (r - mean) / sd


def test_flat_prior_returns_give_no_value_and_are_counted():
    b, got = _feed([100.0] * 23 + [101.0])
    assert got[-1] is None and b.flat == 3  # returns 21 to 23 each read against 20 flat ones


def test_it_checks_reads_in_words_and_is_studies_only():
    d = {"version": 1, "reason": "A test case, written before it is run.",
         "blocks": {"rz": {"kind": "return_zscore", "timeframe": "1d"}},
         "long": {"entry": {"left": "rz.z", "op": "<=", "right": -1.5}}}
    checked = check_definition(d, bar_minutes=240)
    assert checked.warmup[1440] == 22
    assert strategy_words.rules(checked, 240)[0]["entry"] == (
        "Buy when the day's return in standard deviations of the 20 days before it is at or below -1.5")
    assert study_only_uses(to_params(d)) == ["the return z-score block"]
    with pytest.raises(ValueError, match="period"):
        check_definition({**d, "blocks": {"rz": {"kind": "return_zscore", "period": 1}}}, bar_minutes=1440)
    assert "return_zscore" in CATALOGUE
