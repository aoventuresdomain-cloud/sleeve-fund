"""M1 item 1 (PM 30 Oct, next strategy round): the funding rate's z-score and percentile rank in its trailing window, two
more blocks read as the decile is read (R-06): only settlements strictly before the decision candle's close, no value
when one due is missing or the window is too sparse, and studies only (paper and live refuse them on Start)."""

import math

import numpy as np
import pytest

from sleeve_fund import funding
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.research.study import time_warmup_minutes
from sleeve_fund.strategies import check_study_only, study_only_uses
from sleeve_fund.strategies.definitions import (CATALOGUE, FundingPercentile, FundingZScore, check_definition,
                                                funding_percentile_of, funding_zscore_of, to_params)
from sleeve_fund.strategies.timeframes import DAY_MINUTES
from sleeve_fund.venues import venue
from test_research_variants import EIGHT, H, PERP, _entries, _hourly, _rates, _settlements

KINDS = (("funding_zscore", FundingZScore), ("funding_percentile", FundingPercentile))


def _block(cls, times, rates, window_days=7):
    b = cls(window_days=window_days)
    b.source = (np.asarray(times, dtype=np.int64), np.asarray(rates, dtype=float), EIGHT)
    return b


def test_the_z_score_is_the_distance_from_the_windows_mean_in_sample_deviations():
    w = [1.0, 2.0, 3.0, 4.0, 5.0]
    sd = math.sqrt(sum((x - 3.0) ** 2 for x in w) / 4)
    assert funding_zscore_of(w, 5.0) == pytest.approx(2.0 / sd)
    assert funding_zscore_of(w, 3.0) == 0.0
    assert funding_zscore_of([0.0001] * 9, 0.0001) is None  # a window that doesn't vary has no z-score


def test_the_percentile_is_the_share_strictly_below_and_ties_count_as_not_below():
    assert funding_percentile_of(list(range(10)), 9) == 90.0
    assert funding_percentile_of(list(range(10)), 0) == 0.0
    assert funding_percentile_of([5.0] * 20, 5.0) == 0.0


@pytest.mark.parametrize("kind,cls", KINDS)
def test_no_look_ahead_a_settlement_after_the_close_is_never_read(kind, cls):
    times = _settlements()
    rates = [0.0001 * (i % 5) for i in range(len(times))]
    close = times[-2] + 4 * H
    seen = _block(cls, times[:-1], rates[:-1])
    seen.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=close)
    future = _block(cls, times, rates[:-1] + [0.05])  # the next settlement, after the close, is the highest ever
    future.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=close)
    assert seen.value is not None and future.value == seen.value


@pytest.mark.parametrize("kind,cls", KINDS)
def test_a_settlement_stamped_at_the_close_is_not_read_on_it(kind, cls):
    times = _settlements()
    rates = [0.0001 * (i % 5) for i in range(len(times) - 1)] + [0.05]
    before = _block(cls, times[:-1], rates[:-1])
    before.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1])
    for stamp in (0, -30_000_000_000, 30_000_000_000):
        b = _block(cls, times[:-1] + [times[-1] + stamp], rates)
        b.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1])
        assert b.value == before.value, stamp
    later = _block(cls, times, rates)
    later.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1] + 4 * H)
    assert later.value > before.value  # once settled, the spike is read


@pytest.mark.parametrize("kind,cls", KINDS)
def test_with_no_stored_rate_or_too_few_it_gives_no_value_and_counts_the_candle(kind, cls):
    times = _settlements()
    rates = [0.0001 * (1 + i % 3) for i in range(len(times))]
    missing = _block(cls, times[:-2] + times[-1:], rates[:-1])
    missing.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1])
    assert missing.value is None and missing.no_rate == 1
    sparse = _block(cls, times[::2], rates[::2])
    sparse.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1] + 4 * H)
    assert sparse.value is None and sparse.no_rate == 1
    paper = cls(window_days=7)  # never handed a source: it never reads a value
    paper.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1])
    assert paper.value is None and paper.no_rate == 1


def test_a_flat_window_gives_the_z_score_no_value_never_an_assumed_zero():
    times = _settlements()
    b = _block(FundingZScore, times, [0.0001] * len(times))
    b.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1] + 4 * H)
    assert b.value is None and b.no_rate == 1 and b.top_decile == 0


def test_the_values_read_are_the_trailing_windows():
    times = _settlements(days=20)
    rates = [0.0001 * (1 + i % 4) for i in range(len(times))]
    rates[:10] = [0.05] * 10  # older than a 7-day window
    rates[-1] = 0.0004
    window = [r for t, r in zip(times, rates) if t > times[-1] + 4 * H - 7 * 24 * H]
    z = _block(FundingZScore, times, rates)
    p = _block(FundingPercentile, times, rates)
    for b in (z, p):
        b.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1] + 4 * H)
    assert z.value == pytest.approx(funding_zscore_of(window, 0.0004))
    assert p.value == pytest.approx(funding_percentile_of(window, 0.0004))


def test_only_a_percentile_at_or_above_90_is_counted_as_top():
    times = _settlements()[:-1]
    high = [0.0] * (len(times) - 1) + [0.5]
    low = [1.0] * (len(times) - 1) + [0.5]
    for rates, top in ((high, 1), (low, 0)):
        b = _block(FundingPercentile, times, rates)
        b.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1] + 4 * H)
        assert b.top_decile == top


@pytest.mark.parametrize("kind,cls", KINDS)
def test_they_are_offered_studies_only_take_no_timeframe_and_warm_up_over_their_days(kind, cls):
    assert "studies only" in CATALOGUE[kind]
    defn = {"version": 1, "reason": "a fixture", "blocks": {"f": {"kind": kind, "window_days": 30}},
            "long": {"entry": {"left": "f", "op": "<", "right": 0}, "exit": {"left": "f", "op": ">=", "right": 1}}}
    check_definition(defn)
    with pytest.raises(ValueError, match="takes no timeframe"):
        check_definition({**defn, "blocks": {"f": {"kind": kind, "timeframe": "1d"}}})
    params = {**to_params(defn), **PERP}
    assert study_only_uses(params)
    with pytest.raises(ValueError):
        check_study_only("rules", params)
    assert time_warmup_minutes(kind, {"window_days": 30}) == 30 * DAY_MINUTES


def test_a_backtest_reads_the_stored_rates_and_gates_its_entries_on_them(monkeypatch):
    gated = {"version": 1, "reason": "a fixture",
             "blocks": {"avg": {"kind": "sma", "period": 3}, "fund": {"kind": "funding_percentile", "window_days": 7}},
             "long": {"entry": {"all": [{"left": "close", "op": ">", "right": "avg"},
                                        {"left": "fund", "op": "<", "right": 90}]},
                      "exit": {"left": "close", "op": "<", "right": "avg"}}}

    def run(rates):
        monkeypatch.setattr(funding, "rates", lambda v, pair, root=None: rates)
        return run_backtest("rules", _hourly(), venue("BINANCE").instrument("BTC", "USDT", price_precision=1),
                            params={**to_params(gated), **PERP}, bar_minutes=60, half_spread=0, starting_capital=10_000)

    n = 4 * 3 * 14
    flat = run(_rates([0.0001] * n))  # every rate equal: the latest is at the 0th percentile
    rising = run(_rates([0.0001 * (1 + i / n) for i in range(n)]))  # the latest is always the highest: 99th+
    assert _entries(flat) and not _entries(rising)
    assert rising.rule_counts["funding_top_decile"] == len(_hourly())
