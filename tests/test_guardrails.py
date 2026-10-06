"""Research guardrails (v2 P1-6): the nearby-settings check and the 100 out-of-sample trade bar."""

import pandas as pd

from sleeve_fund.research.guardrails import MIN_OOS_TRADES, neighbours, nearby_settings

PARAMS = ["fast", "slow"]


def _grid(sharpe):
    rows = [{"fast": f, "slow": s, "sharpe": sharpe(f, s)} for f in (5, 10, 20, 40) for s in (50, 100, 200)]
    return pd.DataFrame(rows)


def test_neighbours_are_one_grid_step_along_one_setting():
    near = neighbours(_grid(lambda f, s: 1.0), {"fast": 10, "slow": 100}, PARAMS)
    assert sorted(zip(near["fast"], near["slow"])) == [(5, 100), (10, 50), (10, 200), (20, 100)]
    edge = neighbours(_grid(lambda f, s: 1.0), {"fast": 5, "slow": 50}, PARAMS)
    assert sorted(zip(edge["fast"], edge["slow"])) == [(5, 100), (10, 50)]


def test_a_broad_plateau_holds():
    verdict, words = nearby_settings(_grid(lambda f, s: 1.0 + 0.01 * f), {"fast": 10, "slow": 100}, PARAMS)
    assert verdict == "PASS" and words.startswith("4 of 4 nearby settings keep a positive Sharpe")


def test_a_lone_peak_fails_and_names_the_weakest_neighbour():
    peak = _grid(lambda f, s: 2.0 if (f, s) == (10, 100) else (-0.3 if (f, s) == (20, 100) else 1.5))
    verdict, words = nearby_settings(peak, {"fast": 10, "slow": 100}, PARAMS)
    assert verdict == "FAIL" and "3 of 4" in words and "fast 20, slow 100 at -0.30" in words


def test_one_weak_neighbour_passes_while_the_median_holds():
    # Advisor's rule: half the chosen Sharpe at the median neighbour, not every one; positive at all of them.
    one_weak = _grid(lambda f, s: 2.0 if (f, s) == (10, 100) else (0.3 if f == 5 else 1.8))
    assert nearby_settings(one_weak, {"fast": 10, "slow": 100}, PARAMS)[0] == "PASS"


def test_a_median_below_half_the_chosen_sharpe_fails():
    cliff = _grid(lambda f, s: 2.0 if (f, s) == (10, 100) else (0.9 if f in (5, 20) or s == 50 else 1.8))
    verdict, words = nearby_settings(cliff, {"fast": 10, "slow": 100}, PARAMS)
    assert verdict == "FAIL" and "the median one 0.90 against a bar of 1.00" in words


def test_a_losing_centre_or_nan_neighbour_fails():
    assert nearby_settings(_grid(lambda f, s: -0.1), {"fast": 10, "slow": 100}, PARAMS)[0] == "FAIL"
    nan = _grid(lambda f, s: float("nan") if (f, s) == (5, 100) else 1.0)
    assert nearby_settings(nan, {"fast": 10, "slow": 100}, PARAMS)[0] == "FAIL"


def test_nothing_tunable_passes_flagged_and_an_unchecked_choice_fails():
    # Independent Quant Advisor (6 Oct 2026, P1-G3): N/A let an untested choice through G1. Only a grid
    # that varies no setting passes, flagged; a choice off the grid, or with no value, fails with why.
    one = pd.DataFrame([{"fast": 10, "slow": 100, "sharpe": 1.0}])
    verdict, words = nearby_settings(one, {"fast": 10, "slow": 100}, PARAMS)
    assert verdict == "PASS" and words.startswith("flag:")
    assert nearby_settings(pd.DataFrame(), {}, [])[0] == "PASS"
    verdict, words = nearby_settings(_grid(lambda f, s: 1.0), {"fast": 7, "slow": 100}, PARAMS)
    assert verdict == "FAIL" and "not on the grid" in words
    verdict, words = nearby_settings(_grid(lambda f, s: 1.0), {"slow": 100}, PARAMS)
    assert verdict == "FAIL" and "no chosen value for fast" in words


def test_a_choice_at_the_grid_edge_fails():
    # P1-G4: an edge point has a neighbour on one side only, and the best value may lie past the grid.
    verdict, words = nearby_settings(_grid(lambda f, s: 1.0), {"fast": 5, "slow": 100}, PARAMS)
    assert verdict == "FAIL" and words == "chosen at grid edge (fast 5), extend the grid"
    verdict, words = nearby_settings(_grid(lambda f, s: 1.0), {"fast": 40, "slow": 200}, PARAMS)
    assert verdict == "FAIL" and "fast 40, slow 200" in words


def test_a_setting_the_grid_does_not_vary_has_no_edge():
    flat_slow = pd.DataFrame([{"fast": f, "slow": 100, "sharpe": 1.0} for f in (5, 10, 20)])
    assert nearby_settings(flat_slow, {"fast": 10, "slow": 100}, PARAMS)[0] == "PASS"


def test_a_grid_mixing_words_numbers_and_none_is_checked_not_crashed():
    # QA F5: sorted() raised on mixed values, and a None default never equalled itself in pandas.
    rows = [{"mode": m, "fast": f, "sharpe": 1.0} for m in (None, 3, "close") for f in (5, 10, 20)]
    grid = pd.DataFrame(rows, dtype=object)
    assert nearby_settings(grid, {"mode": 3, "fast": 10}, ["mode", "fast"])[0] == "PASS"
    verdict, words = nearby_settings(grid, {"mode": None, "fast": 10}, ["mode", "fast"])
    assert verdict == "FAIL" and "grid edge (mode None)" in words


def test_the_evidence_gives_the_neighbours_trades():
    grid = _grid(lambda f, s: 2.0 if (f, s) == (10, 100) else (-0.2 if f == 20 else 1.8))
    grid["round_trips"] = [0 if f == 20 else 30 for f in grid["fast"]]
    verdict, words = nearby_settings(grid, {"fast": 10, "slow": 100}, PARAMS)
    assert verdict == "FAIL" and "1 of them made no trades" in words and "at -0.20, 0 trades" in words


def test_the_trade_bar_is_the_specs_hundred():
    assert MIN_OOS_TRADES == 100
