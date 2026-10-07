"""R2-G1: G1 and a first_touch rule's ambiguous candles (Advisor 6 Oct ~22:07 and ~22:11). Over 5% of the
out-of-sample candles that reached either level, the study re-runs every test window with the opposite resolution
and G1 judges on the worse; the tear sheet shows both. A model without a first_touch rule is unaffected."""

from types import SimpleNamespace

import pandas as pd
import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research import study as study_mod
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.metrics import summary
from sleeve_fund.research.study import FIRST_TOUCH_FLIP_SHARE, first_touch_share, needs_flip, run_study
from sleeve_fund.research.tearsheet import FIRST_TOUCH_CHECK, g1_checks
from sleeve_fund.strategies.trend_filter import SPEC

RULE = "long.entry"


def _stats(ambiguous: int, reached: int = 1000, flip: bool = False) -> dict:
    """One rule's first_touch_stats (rules.first_touch_stats' keys), its ambiguous candles all same-minute ones."""
    return {RULE: {"judged": 2 * reached, "resolved_true": reached // 2, "reached": reached, "same_minute": ambiguous,
                   "unknown": 0, "inconsistent": 0, "resolved": ("true" if flip else "false") + " when ambiguous"}}


@pytest.mark.parametrize("ambiguous, flips", [(49, False), (50, False), (51, True)])
def test_g1_re_runs_only_when_ambiguous_candles_are_over_5_percent_of_those_reaching_a_level(ambiguous, flips):
    """4.9% and exactly 5% are judged as ruled; 5.1% is re-run the opposite way."""
    share = first_touch_share([SimpleNamespace(first_touch=_stats(ambiguous))])
    assert share == pytest.approx(ambiguous / 1000)
    assert needs_flip(share) is flips
    assert FIRST_TOUCH_FLIP_SHARE == 0.05


def test_the_share_counts_every_kind_of_ambiguous_candle_across_every_window_and_rule():
    a = {RULE: {**_stats(10)[RULE], "unknown": 10, "inconsistent": 5}}  # 20 of 1000: inconsistent is within unknown
    b = {"other": _stats(40, reached=500)[RULE]}  # 40 of 500
    assert first_touch_share([SimpleNamespace(first_touch=a), SimpleNamespace(first_touch=b)]) == pytest.approx(60 / 1500)
    assert first_touch_share([SimpleNamespace(first_touch={})]) is None  # no first_touch rule
    assert first_touch_share([SimpleNamespace(first_touch=_stats(0, reached=0))]) == 0.0  # no level reached


def _patched(monkeypatch, ambiguous: int | None):
    """study.run_backtest stood in by the real one plus first_touch stats of `ambiguous` per 1000 (None: none), the
    flipped runs losing 0.05% a day more. Returns the flip argument of every run."""
    real, flips = study_mod.run_backtest, []

    def fake(*a, first_touch_flip=False, **kw):
        flips.append(first_touch_flip)
        res = real(*a, **kw)
        if ambiguous is not None and a[0] != "buy_and_hold":
            res.first_touch = _stats(ambiguous, flip=first_touch_flip)
            if first_touch_flip and len(res.equity):
                res.equity = res.equity * pd.Series(0.9995, index=res.equity.index).cumprod()
        return res

    monkeypatch.setattr(study_mod, "run_backtest", fake)
    return flips


def _study(tmp_path, instrument, **kw):
    prices = synthetic_ohlcv(days=1200, seed=3)
    return run_study(SPEC, prices, instrument, dataset="syn", ledger=IdeaLedger(tmp_path / "l.jsonl"), synthetic=True,
                     holdout_days=0, train_days=365, test_days=365, **kw)


def test_over_5_percent_g1_shows_both_results_and_judges_the_worse(tmp_path, instrument, monkeypatch):
    flips = _patched(monkeypatch, 51)
    r = _study(tmp_path, instrument)
    ft = r.first_touch
    assert ft["flipped"] and ft["share"] == pytest.approx(0.051)
    assert ft["opposite"]["sharpe"] < ft["as_ruled"]["sharpe"]
    assert ft["judged_on"] == "opposite"
    assert summary(r.oos_returns)["sharpe"] == pytest.approx(ft["opposite"]["sharpe"])  # G1 reads the worse
    assert sum(flips) == len([f for f in r.folds if not f.unscored])  # each window once more, flipped
    row = next(c for c in g1_checks(r, IdeaLedger(tmp_path / "l.jsonl")) if c[0] == FIRST_TOUCH_CHECK)
    assert row[1] == "INFO"
    assert "As ruled (long.entry false when ambiguous): Sharpe" in row[2]
    assert "opposite (long.entry true when ambiguous): Sharpe" in row[2] and "Judged on the worse, the opposite" in row[2]


def test_at_4_9_percent_g1_judges_as_ruled_and_never_re_runs(tmp_path, instrument, monkeypatch):
    flips = _patched(monkeypatch, 49)
    r = _study(tmp_path, instrument)
    assert r.first_touch == {"share": pytest.approx(0.049), "ambiguous": 98, "reached": 2000,
                             "fold_shares": [pytest.approx(0.049)] * 2, "threshold": 0.05, "min_reached": 20,
                             "flipped": False, "judged_on": "as ruled", "resolved": "long.entry false when ambiguous"}
    assert not any(flips)
    row = next(c for c in g1_checks(r, IdeaLedger(tmp_path / "l.jsonl")) if c[0] == FIRST_TOUCH_CHECK)
    assert row[2].endswith("judged as ruled")


def test_a_model_without_a_first_touch_rule_is_unaffected(tmp_path, instrument, monkeypatch):
    flips = _patched(monkeypatch, None)
    r = _study(tmp_path, instrument)
    assert r.first_touch == {}
    assert not any(flips)
    assert all(c[0] != FIRST_TOUCH_CHECK for c in g1_checks(r, IdeaLedger(tmp_path / "l.jsonl")))


def test_over_5_percent_the_holdout_is_judged_on_the_worse_too(tmp_path, instrument, monkeypatch):
    _patched(monkeypatch, 51)
    prices = synthetic_ohlcv(days=1400, seed=3)
    r = run_study(SPEC, prices, instrument, dataset="syn", ledger=IdeaLedger(tmp_path / "l.jsonl"), synthetic=True,
                  holdout_days=200, train_days=365, test_days=365, use_holdout=True)
    assert r.holdout is not None
    assert r.first_touch["holdout_judged_on"] == "opposite"
    row = next(c for c in g1_checks(r, IdeaLedger(tmp_path / "l.jsonl")) if c[0] == FIRST_TOUCH_CHECK)
    assert row[2].endswith("the holdout on the opposite")


def test_a_runs_first_touch_report_counts_only_the_candles_from_the_test_window_on(instrument):
    """CR #171: the share G1 reads is out of sample. The ambiguous candle (the second, candle B) is counted when the
    count starts at its own close, and not when it starts at the next candle's."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies.definitions import to_params
    from test_first_touch import DEFN, FLAT, _frames

    candles, minutes = _frames(FLAT * 4 + [(101.5, 98.5)] + FLAT * 10)

    def stats(count_from):
        res = run_backtest("rules", candles, instrument, params=to_params(DEFN), bar_minutes=15, half_spread=0,
                           exec_prices=minutes, exec_minutes=1, first_touch_count_from=count_from)
        (st,) = res.first_touch.values()
        return st

    whole, from_it, after_it = stats(None), stats(candles.index[1]), stats(candles.index[2])
    assert whole["same_minute"] == from_it["same_minute"] == 1 and after_it["same_minute"] == 0
    assert after_it["judged"] < from_it["judged"] <= whole["judged"]


def test_each_windows_share_is_counted_from_its_first_test_candle(tmp_path, instrument, monkeypatch):
    starts = []
    real = study_mod.run_backtest

    def fake(*a, first_touch_count_from=None, **kw):
        if first_touch_count_from is not None:
            starts.append(first_touch_count_from)
        return real(*a, **kw)

    monkeypatch.setattr(study_mod, "run_backtest", fake)
    res = _study(tmp_path, instrument)
    assert starts and len(starts) == len(res.folds)
    assert all(f.train_end < s <= f.test_end for f, s in zip(res.folds, starts))  # each window's first test candle
