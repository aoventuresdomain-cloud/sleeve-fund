"""Random-entry benchmark (v2 P1-7, C3)."""

import numpy as np
import pytest

from sleeve_fund.research.random_entry import BAR, Trade, _random_entries, random_entry, random_side


def _walk(n=3000, seed=1):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))


def test_random_entries_never_overlap_and_stay_in_the_window():
    rng = np.random.default_rng(0)
    holds = np.array([5, 1, 9, 3])
    for _ in range(500):
        order, entries = _random_entries(rng, 100, 140, holds)
        exits = entries + holds[order]
        assert entries[0] >= 100 and exits[-1] <= 140
        assert (entries[1:] >= exits[:-1]).all()


def test_perfect_timing_passes():
    c = 100 + 10 * np.sin(np.arange(2000) / 20)  # buy each trough, sell the next peak
    troughs = [i for i in range(1, 1999) if c[i] < c[i - 1] and c[i] <= c[i + 1]]
    trades = [Trade(t, t + 63, 1) for t in troughs if t + 63 < 2000]
    r = random_entry(c, trades, [(0, 1999)], cost_per_side=0.0005, draws=300)
    assert r.verdict == "PASS" and r.return_percentile >= BAR and r.trades == len(trades)


def test_random_timing_passes_about_one_time_in_twenty():
    c = _walk()
    rng = np.random.default_rng(7)
    passes = 0
    for k in range(100):
        order, entries = _random_entries(rng, 0, 2999, np.full(20, 10))
        trades = [Trade(int(e), int(e) + 10, 1) for e in entries]
        passes += random_entry(c, trades, [(0, 2999)], 0.0005, draws=200, seed=k).verdict == "PASS"
    assert passes <= 12  # a 5% test: 5 expected in 100, 12 or more has under 1% chance


def test_mostly_in_the_market_is_weak_not_passed():
    c = np.linspace(100, 200, 1000)  # a straight rise: any long wins
    trades = [Trade(i, i + 70, 1) for i in range(0, 900, 100)]  # 70% of the bars held
    r = random_entry(c, trades, [(0, 999)], 0.0, draws=100)
    assert r.weak and r.verdict == "WEAK" and "a weak test" in r.words


def test_pooled_across_windows_with_costs_charged_per_side():
    c = _walk()
    trades = [Trade(10, 20, 1), Trade(1510, 1520, -1)]
    free = random_entry(c, trades, [(0, 1499), (1500, 2999)], 0.0, draws=50)
    costly = random_entry(c, trades, [(0, 1499), (1500, 2999)], 0.01, draws=50)
    gross = (c[20] / c[10]) * (2 - c[1520] / c[1510]) - 1
    assert free.trades == 2 and free.strategy_return == pytest.approx(gross)
    assert costly.strategy_return == pytest.approx((c[20] / c[10] - 0.02) * (2 - c[1520] / c[1510] - 0.02) - 1)


def test_no_trades_is_not_applicable_and_runs_repeat():
    c = _walk()
    assert random_entry(c, [], [(0, 2999)], 0.001).verdict == "N/A"
    trades = [Trade(100, 150, 1), Trade(900, 920, 1)]
    a, b = (random_entry(c, trades, [(0, 2999)], 0.001, draws=100, seed=3) for _ in range(2))
    assert a == b


def test_overlapping_trades_are_refused():
    with pytest.raises(ValueError):
        random_entry(_walk(), [Trade(0, 80, 1), Trade(10, 90, 1)], [(0, 999)], 0.0)


def test_picking_the_direction_passes_the_random_side_test():
    """C3b: the same entries and holding periods, long or short drawn at random."""
    c = 100 + 10 * np.sin(np.arange(2000) / 20)
    trades = [Trade(i, i + 20, 1 if c[i + 20] > c[i] else -1) for i in range(0, 1960, 40)]
    r = random_side(c, trades, [(0, 1999)], cost_per_side=0.0005, draws=300)
    assert r.verdict == "PASS" and r.trades == len(trades)
    wrong = [Trade(t.entry, t.exit, -t.side) for t in trades]
    assert random_side(c, wrong, [(0, 1999)], 0.0005, draws=300).verdict == "FAIL"


def test_random_side_with_no_trades_is_not_applicable():
    assert random_side(_walk(), [], [(0, 100)], 0.0005).verdict == "N/A"


def test_window_trips_leave_out_trades_at_the_edges():
    """Advisor, 19:19: carried in from training, or still open at the window's end, don't count."""
    import pandas as pd

    from sleeve_fund.research.study import _window_trips

    t = pd.Timestamp
    trips = [{"opened": t("2025-01-01", tz="UTC"), "closed": t("2025-02-02", tz="UTC"), "side": 1},  # carried in
             {"opened": t("2025-02-03", tz="UTC"), "closed": t("2025-02-10", tz="UTC"), "side": -1},
             {"opened": t("2025-02-20", tz="UTC"), "closed": None, "side": 1}]  # still open at the end
    out = _window_trips(trips, t("2025-02-01", tz="UTC"))
    assert (out["test_trades"], out["carried_in"], out["carried_out"]) == (1, 1, 1)
    assert out["trips"] == [(t("2025-02-03", tz="UTC"), t("2025-02-10", tz="UTC"), -1)]
