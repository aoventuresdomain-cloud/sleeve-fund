"""R2-G1 #171 Done-when (QA Tester 2; Advisor ruling of 7 Oct on R2-G1-F1, advisor-rulings.md): the first_touch
re-run on the worse resolution is judged on the OUT-OF-SAMPLE level-reaching candles only, never training days.
  - pooled share > 5% (strictly over: exactly 5.0% does not trigger), OR
  - any single fold's share > 5%, OR
  - fewer than 20 out-of-sample level-reaching candles in all (even at 0% ambiguous);
  - tear sheet: "X of Y out-of-sample level-reaching candles (Z%) were same-minute ambiguous."

Real end-to-end studies (a rules strategy with a first_touch entry through run_study; nothing stubbed). Flat price 100,
hourly candles over 1-minute bars; levels from the candle before's close: 100 x 1.002 (X) and 100 x 0.998 (Y).
 - "decided" candle: one minute touches Y only -> a level reached, not ambiguous, the entry stays false (flat);
 - "ambiguous" candle: one minute touches both X and Y -> same-minute, resolved false on entry (flat).
Both leave the strategy flat, so every candle is judged. Study: 120 days, train 30, test 30 = 3 folds; days 0-30 are
training only (500 decided candles there, to dilute any count that includes them); fold k's test days are
30+30k to 60+30k. Each test first checks, per fold, that the test window alone reaches (reached, ambiguous) as designed.

  PYTHONDONTWRITEBYTECODE=1 BACKTEST_ISOLATE=0 PYTHONPATH=. python -m pytest -p no:cacheprovider <this file>
Strict xfails (raises=AssertionError) until the fix lands; QD removes each mark as its cell passes.
"""
import dataclasses
import re
import tempfile

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.research.study import run_study
from sleeve_fund.research.tearsheet import g1_checks
from sleeve_fund.strategies.definitions import to_params
from sleeve_fund.strategies.rules import SPEC
from sleeve_fund.venues import venue

FT = {"first_touch": {"reach": {"mul": ["close", 1.002]}, "before": {"mul": ["close", 0.998]}}}
DEFN = {"version": 1, "reason": "QA probe, written before it is run.", "blocks": {},
        "long": {"entry": FT, "exit": {"left": "close", "op": "<", "right": 0}}}
DAYS, TRAIN, TEST = 120, 30, 30


def _data(folds):
    """folds: [(reached, ambiguous)] per out-of-sample fold; plus 500 decided candles in the training-only days."""
    n = DAYS * 1440
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    mn = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 10.0},
                      index=idx + pd.Timedelta(minutes=1))
    events = [(h, "dec") for h in range(500)]  # hours 0-499 of day 0..20: training only
    for k, (reached, amb) in enumerate(folds):
        base = (TRAIN + TEST * k) * 24 + 1
        events += [(base + i, "amb" if i < amb else "dec") for i in range(reached)]
    for h, kind in events:
        m = h * 60 + 30
        mn.iloc[m, mn.columns.get_loc("low")] = 99.75  # Y only
        if kind == "amb":
            mn.iloc[m, mn.columns.get_loc("high")] = 100.25  # and X in the same minute
    hr = mn.resample("60min", closed="right", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return idx, mn, hr


def _study(folds):
    inst = venue("KRAKEN").instrument("BTC", "USD")
    idx, mn, hr = _data(folds)
    params = to_params(DEFN)
    # set-up: each fold's test window alone reaches what was designed
    for k, (reached, amb) in enumerate(folds):
        lo, hi = idx[(TRAIN + TEST * k) * 1440], idx[(TRAIN + TEST * (k + 1)) * 1440 - 1]
        h_, m_ = hr[(hr.index > lo) & (hr.index <= hi)], mn[(mn.index > lo) & (mn.index <= hi)]
        st = run_backtest("rules", h_, inst, params, bar_minutes=60, half_spread=0, exec_prices=m_,
                          exec_minutes=1).first_touch["long.entry"]
        assert (st["reached"], st["same_minute"] + st["unknown"]) == (reached, amb), (k, st, reached, amb)
    spec = dataclasses.replace(SPEC, param_grid={k: [v] for k, v in params.items()})
    ledger = IdeaLedger(tempfile.mkdtemp() + "/l.jsonl")
    r = run_study(spec, hr, inst, dataset="qa", ledger=ledger, default_params=params, synthetic=True, holdout_days=0,
                  train_days=TRAIN, test_days=TEST, exec_prices=mn)
    assert len([f for f in r.folds if not f.unscored]) == len(folds), r.folds
    return r, ledger


def _words(r, ledger):
    rows = [c[2] for c in g1_checks(r, ledger) if "level-reaching" in str(c[2])]
    assert rows, "no tear sheet row says 'level-reaching candles'"
    return rows[0]


def test_pooled_over_5_percent_of_out_of_sample_candles_re_runs_and_the_row_reads_x_of_y():
    r, ledger = _study([(15, 15), (14, 14), (14, 14)])  # 43 of 43: the Advisor's pin
    assert r.first_touch["flipped"], r.first_touch
    w = _words(r, ledger)
    assert re.search(r"43 of 43 out-of-sample level-reaching candles \(100(\.0)?%\) were same-minute ambiguous", w), w


def test_one_fold_over_5_percent_re_runs_though_the_pooled_share_is_under():
    r, _ = _study([(50, 4), (400, 0), (400, 0)])  # fold 1 at 8%, pooled 4 of 850 = 0.5%
    assert r.first_touch["flipped"], r.first_touch


def test_fewer_than_20_out_of_sample_level_reaching_candles_re_runs_even_at_0_percent_ambiguous():
    r, _ = _study([(7, 0), (6, 0), (6, 0)])  # 19 candles, none ambiguous
    assert r.first_touch["flipped"], r.first_touch


def test_exactly_5_percent_pooled_and_in_every_fold_does_not_re_run_and_the_row_says_so():
    r, ledger = _study([(200, 10), (200, 10), (200, 10)])  # 30 of 600 = 5.0%, each fold 5.0%
    assert not r.first_touch["flipped"], r.first_touch
    w = _words(r, ledger)
    assert re.search(r"30 of 600 out-of-sample level-reaching candles \(5(\.0)?%\) were same-minute ambiguous", w), w


def test_just_over_5_percent_pooled_re_runs():
    r, _ = _study([(200, 11), (200, 10), (200, 10)])  # 31 of 600 = 5.17% pooled; fold 1 at 5.5%
    assert r.first_touch["flipped"], r.first_touch
