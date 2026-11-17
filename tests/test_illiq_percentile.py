"""N5 ILLIQ (HoE 17 Nov): the illiq_percentile block, Amihud's illiquidity |daily return| / (volume x close) ranked
against its own trailing `period` (180) days. pct is the share of the window, the latest included, strictly below the
latest (0 to 100); value the reading itself. Daily candles only; a candle with no dollar volume gives no reading and
never enters the window. Studies only until the PM approves it for paper."""

from __future__ import annotations

import numpy as np
import pytest

from sleeve_fund.dashboard import strategy_words
from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies import check_study_only, study_only_uses
from sleeve_fund.strategies.definitions import (
    CATALOGUE,
    IlliqPercentile,
    check_definition,
    illiq_of,
    to_params,
)


def _defn(**block):
    return {"version": 1, "reason": "A test case, written before it is run.",
            "blocks": {"il": {"kind": "illiq_percentile", **block}},
            "long": {"entry": {"left": "il.pct", "op": ">=", "right": 90}},
            "exits": {"time_stop": {"bars": 3}}}


def test_one_reading_is_the_absolute_return_per_unit_of_dollar_volume():
    assert illiq_of(100.0, 110.0, 5.0) == pytest.approx(0.1 / 550.0)
    assert illiq_of(100.0, 90.0, 5.0) == pytest.approx(0.1 / 450.0)
    assert illiq_of(100.0, 100.0, 5.0) == 0.0
    assert illiq_of(100.0, 110.0, 0.0) is None and illiq_of(0.0, 110.0, 5.0) is None


def _reference(close: np.ndarray, volume: np.ndarray, period: int) -> list:
    """pct at each candle, or None: by hand, from the definition."""
    window, out = [], [None]
    for t in range(1, len(close)):
        x = illiq_of(close[t - 1], close[t], volume[t])
        if x is None:
            out.append(None)
            continue
        window = (window + [x])[-period:]
        out.append(100.0 * sum(w < x for w in window) / period if len(window) == period else None)
    return out


def test_the_block_ranks_each_reading_in_its_trailing_window_and_skips_a_day_with_no_volume():
    rng = np.random.default_rng(3)
    n, period = 120, 30
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    volume = rng.uniform(50, 500, n)
    volume[[40, 41, 77]] = 0.0
    b = IlliqPercentile(period=period)
    got = []
    for c, v in zip(close, volume, strict=True):
        b.update_ohlcv(c, c, c, c, v)
        got.append(b.values["pct"] if b.initialized else None)
    want = _reference(close, volume, period)
    assert [g is None for g in got] == [w is None for w in want]
    assert [g for g in got if g is not None] == pytest.approx([w for w in want if w is not None])
    assert b.no_volume == 3 and sum(g is not None for g in got) > 50
    assert b.values["value"] == pytest.approx(illiq_of(close[-2], close[-1], volume[-1]))
    assert b.warmup_bars == period + 1


def test_it_reads_daily_candles_only_and_its_period_is_bounded():
    assert check_definition(_defn(timeframe="1d"), bar_minutes=60).warmup[1440] == 181
    assert check_definition(_defn(), bar_minutes=1440).warmup[None] == 181
    with pytest.raises(ValueError, match="reads daily candles"):
        check_definition(_defn(), bar_minutes=60)
    with pytest.raises(ValueError, match="reads daily candles"):
        check_definition(_defn(timeframe="4h"), bar_minutes=60)
    for bad in (19, 1001):
        with pytest.raises(ValueError, match="period"):
            check_definition(_defn(period=bad), bar_minutes=1440)
    assert "illiq_percentile" in CATALOGUE


def test_studies_only_until_the_pm_approves_it_for_paper():
    params = to_params(_defn(timeframe="1d"))
    assert study_only_uses(params) == ["the illiquidity percentile block"]
    with pytest.raises(ValueError, match="illiquidity percentile"):
        check_study_only("rules", params)


def test_it_reads_in_words():
    words = strategy_words.rules(check_definition(_defn(timeframe="1d"), bar_minutes=60), 60)
    assert words[0]["entry"] == ("Buy when the illiquidity percentile (|return| ÷ dollar volume, against its last 180 "
                                 "days) is at or above 90")


def test_a_daily_backtest_enters_on_the_rank_the_block_gives_by_hand(instrument):
    df = synthetic_ohlcv(days=600, seed=11)
    res = run_backtest("rules", df, instrument, params=to_params(_defn(period=60)), bar_minutes=1440, half_spread=0)
    want = _reference(df["close"].to_numpy(), df["volume"].to_numpy(), 60)
    at_close = {round(c, 2): w for c, w in zip(df["close"], want, strict=True)}
    entries = [d["signal"] for d in res.decisions.values() if d["intent"] == "entry"]
    assert len(entries) >= 5
    for sig in entries:
        assert sig["blocks"]["il.pct"] >= 90
        assert sig["blocks"]["il.pct"] == pytest.approx(at_close[round(sig["bar"]["c"], 2)])
