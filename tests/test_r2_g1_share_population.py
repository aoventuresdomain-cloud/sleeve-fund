"""R2-G1 #171 (9fb0962), QA Tester 2: the 5 % share is counted over every candle of each window's run (training days
included), but the code comment and the tear sheet call it the OUT-OF-SAMPLE share, and G1 judges out of sample.
A real rules strategy with a first_touch entry runs through run_study (no stubbed stats).

Data (synthetic, hourly candles over 1-minute bars, levels close x 1.002 / x 0.998 from the candle before's close):
- training days (0-150): ordinary moves, so levels are often reached one at a time (decided by the minutes);
- out-of-sample days (150-330): dead calm, except one minute in one hour in a hundred spikes through both levels:
  the only out-of-sample candles that reach a level are same-minute ones, so the OUT-OF-SAMPLE share is 100 %.
The study reports 1.4 %: the training days' decided reaches dilute it, G1 judges as ruled, and the tear sheet says
"1.4 % of the out-of-sample candles ... were ambiguous". Strict xfail until the share is the out-of-sample one (or the
Advisor rules the pooled share, and the tear sheet stops saying out-of-sample).

  PYTHONDONTWRITEBYTECODE=1 BACKTEST_ISOLATE=0 PYTHONPATH=. python -m pytest -p no:cacheprovider <this file>   (45 s)
"""
import dataclasses
import tempfile

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.research.study import run_study
from sleeve_fund.strategies.definitions import to_params
from sleeve_fund.strategies.rules import SPEC
from sleeve_fund.venues import venue

FT = {"first_touch": {"reach": {"mul": ["close", 1.002]}, "before": {"mul": ["close", 0.998]}}}
DEFN = {"version": 1, "reason": "QA probe, written before it is run.", "blocks": {},
        "long": {"entry": FT, "exit": {"left": "close", "op": "<", "right": 0}}}
OOS_FROM = 150  # day


def _data():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=330 * 1440, freq="min", tz="UTC")
    i = np.arange(len(idx))
    oos = i >= OOS_FROM * 1440
    spike = oos & ((i // 60) % 100 == 0) & (i % 60 == 30)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1, len(idx)) * np.where(oos, 0.00002, 0.0004)))
    mn = pd.DataFrame({"open": np.r_[close[0], close[:-1]], "close": close}, index=idx + pd.Timedelta(minutes=1))
    wick = np.where(spike, 0.004, np.where(oos, 0.00002, 0.00005))
    mn["high"] = mn[["open", "close"]].max(axis=1) * (1 + wick)
    mn["low"] = mn[["open", "close"]].min(axis=1) * (1 - wick)
    mn["volume"] = 10.0
    mn = mn[["open", "high", "low", "close", "volume"]]
    h = mn.resample("60min", closed="right", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return idx, mn, h


def test_the_ambiguous_share_is_the_out_of_sample_one():
    inst = venue("KRAKEN").instrument("BTC", "USD")
    idx, mn, h = _data()
    params = to_params(DEFN)
    spec = dataclasses.replace(SPEC, param_grid={k: [v] for k, v in params.items()})
    oos_h, oos_m = h[h.index > idx[OOS_FROM * 1440]], mn[mn.index > idx[OOS_FROM * 1440]]
    res = run_backtest("rules", oos_h, inst, params, bar_minutes=60, half_spread=0, exec_prices=oos_m, exec_minutes=1)
    st = res.first_touch["long.entry"]
    assert st["reached"] > 0 and st["same_minute"] == st["reached"], st  # set-up: every OOS reach is ambiguous
    r = run_study(spec, h, inst, dataset="qa", ledger=IdeaLedger(tempfile.mkdtemp() + "/l.jsonl"), default_params=params,
                  synthetic=True, holdout_days=0, train_days=150, test_days=90, exec_prices=mn)
    assert r.first_touch["share"] == pytest.approx(1.0), r.first_touch  # the out-of-sample candles' share
    assert r.first_touch["flipped"], r.first_touch  # so G1 re-runs the opposite resolution
