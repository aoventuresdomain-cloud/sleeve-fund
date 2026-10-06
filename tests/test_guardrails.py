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
    assert verdict == "PASS" and words.startswith("4 of 4 nearby settings")


def test_a_lone_peak_fails_and_names_the_weakest_neighbour():
    peak = _grid(lambda f, s: 2.0 if (f, s) == (10, 100) else (-0.3 if (f, s) == (20, 100) else 1.5))
    verdict, words = nearby_settings(peak, {"fast": 10, "slow": 100}, PARAMS)
    assert verdict == "FAIL" and "3 of 4" in words and "fast 20, slow 100 at -0.30" in words


def test_a_cliff_below_half_the_chosen_sharpe_fails():
    cliff = _grid(lambda f, s: 2.0 if (f, s) == (10, 100) else (0.9 if f == 5 else 1.8))
    assert nearby_settings(cliff, {"fast": 10, "slow": 100}, PARAMS)[0] == "FAIL"


def test_a_losing_centre_or_nan_neighbour_fails():
    assert nearby_settings(_grid(lambda f, s: -0.1), {"fast": 10, "slow": 100}, PARAMS)[0] == "FAIL"
    nan = _grid(lambda f, s: float("nan") if (f, s) == (5, 100) else 1.0)
    assert nearby_settings(nan, {"fast": 10, "slow": 100}, PARAMS)[0] == "FAIL"


def test_nothing_nearby_is_not_applicable():
    one = pd.DataFrame([{"fast": 10, "slow": 100, "sharpe": 1.0}])
    assert nearby_settings(one, {"fast": 10, "slow": 100}, PARAMS)[0] == "N/A"
    assert nearby_settings(_grid(lambda f, s: 1.0), {"fast": 7, "slow": 100}, PARAMS)[0] == "N/A"
    assert nearby_settings(pd.DataFrame(), {}, [])[0] == "N/A"


def test_the_trade_bar_is_the_specs_hundred():
    assert MIN_OOS_TRADES == 100
