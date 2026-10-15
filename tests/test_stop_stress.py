"""The stop slippage stress line (D13-INTRAMINUTE-JUMP (3); GAP-DOWN, Advisor 15 Oct ~02:00 UK, point 3): G1 shows
out-of-sample again with 20 bp more on each stop exit, charged on the equity curve from the fill, with no re-sizing; a
Sharpe test that would fail under it is flagged to the PM, and it stays INFO. Required on the sheet when the entries'
median stop distance is under 0.5%."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from sleeve_fund.research import study, tearsheet
from sleeve_fund.research.study import STOP_STRESS, _stop_stress

DAY = pd.Timedelta("1D")


def _run():
    idx = pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC")
    equity = pd.Series([10_000.0 + 10 * k for k in range(10)], index=idx)
    fills = pd.DataFrame({"ts_last": [idx[3] - pd.Timedelta(hours=6), idx[6] - pd.Timedelta(hours=1), idx[1]],
                          "filled_qty": [0.1, 0.1, 0.1], "avg_px": [50_000.0, 49_000.0, 51_000.0]},
                         index=["e1", "s1", "s0"])
    decisions = {"e1": {"intent": "entry", "signal": {"stop_frac": 0.004}},
                 "s1": {"intent": "stop_loss", "signal": {}},
                 "s0": {"intent": "stop_loss", "signal": {}}}  # before the window: not this window's
    return SimpleNamespace(equity=equity, starting_capital=10_000.0, fills=fills, decisions=decisions), idx


def test_a_windows_stop_exit_costs_20_bp_of_its_notional_from_its_fill_on_and_nothing_before():
    run, idx = _run()
    test_idx = idx[2:]
    test_ret = study.whole_days(study.daily_returns(study._flat_until(run, test_idx[0] - DAY)), test_idx[0], DAY)
    charge, stops, dist = _stop_stress(run, test_idx, DAY, test_ret)
    assert stops == 1 and dist == [0.004]
    cost = 9.80  # 20 bp of the stop exit's 0.1 x 49,000 notional
    assert STOP_STRESS == 0.002
    hit = idx[6]  # the day the stop filled in: equity from there on is 9.80 lower
    assert charge[charge.index < hit].abs().max() == 0.0
    assert charge[hit] == pytest.approx(cost / float(run.equity[idx[5]]), rel=1e-9)
    stressed_end = (1 + test_ret - charge).prod()
    # From the window's flat start (P1-D13): the stressed curve ends exactly the cost lower, nothing re-sized
    assert stressed_end * run.starting_capital == pytest.approx(float(run.equity.iloc[-1]) - cost, rel=1e-9)


def test_a_window_without_stop_exits_charges_nothing():
    run, idx = _run()
    run.decisions["s1"]["intent"] = "exit"
    test_idx = idx[2:]
    test_ret = study.whole_days(study.daily_returns(study._flat_until(run, test_idx[0] - DAY)), test_idx[0], DAY)
    charge, stops, _ = _stop_stress(run, test_idx, DAY, test_ret)
    assert stops == 0 and (charge == 0).all() and charge.index.equals(test_ret.index)


def _result(median, stops=3):
    days = pd.date_range("2026-01-01", periods=60, freq="1D", tz="UTC")
    base = pd.Series([0.002, -0.001, 0.003] * 20, index=days)
    bench = pd.Series([0.0005, -0.0005] * 30, index=days)  # a Sharpe of about 0
    return SimpleNamespace(compare_returns=base, oos_benchmark_returns=bench,
                           stop_stress={"charge": pd.Series(0.0005, index=days), "stops": stops, "bp": 20.0,
                                        "median_stop": median, "required_below": 0.005})


@pytest.mark.parametrize("median, required", [(0.004, True), (0.02, False)])
def test_the_row_is_required_under_a_half_percent_median_stop_and_says_both_sharpes(median, required):
    name, verdict, words = tearsheet._stop_stress_row(_result(median), {"variants": 1}, None, insufficient=True)
    assert name == tearsheet.STOP_STRESS_CHECK and verdict == "INFO"
    assert words.startswith(f"required, median stop {median:.2%} is under 0.5%: ") is required
    assert "3 stop exits out of sample at +20 bp each: Sharpe " in words


@pytest.mark.parametrize("beats, flagged", [((0.99, 0.5), True), ((0.99, 0.99), False)])
def test_a_sharpe_test_that_fails_under_the_stress_is_flagged_to_the_pm_and_stays_info(monkeypatch, beats, flagged):
    seq = iter(beats)
    monkeypatch.setattr(tearsheet, "sharpe_beats_probability", lambda *a, **k: (next(seq), 0.0))
    monkeypatch.setattr(tearsheet, "_trial_spread", lambda r, register: None)
    _, verdict, words = tearsheet._stop_stress_row(_result(0.004), {"variants": 4}, None, insufficient=False)
    assert verdict == "INFO"
    assert ("the Sharpe test would FAIL under it: flagged to the PM" in words) is flagged
    assert ("still passes under it" in words) is not flagged


def test_no_stop_exits_says_so():
    _, verdict, words = tearsheet._stop_stress_row(_result(None, stops=0), {"variants": 1}, None, insufficient=True)
    assert verdict == "INFO" and words == "no stop exits out of sample, so nothing to stress"


def test_a_study_with_a_tight_stop_shows_the_required_line_on_its_sheet(tmp_path, instrument):
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC

    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, synthetic_ohlcv(days=1900, seed=3), instrument, dataset="syn", ledger=ledger, synthetic=True,
                  holdout_days=200, train_days=730, test_days=365, exits={"stop_loss": 0.004})
    assert r.stop_stress["stops"] > 0 and r.stop_stress["median_stop"] == pytest.approx(0.004)
    row = next(c for c in tearsheet.render(r, ledger).splitlines()
               if c.startswith(f"| {tearsheet.STOP_STRESS_CHECK} |"))
    assert "required, median stop 0.40% is under 0.5%" in row and "stop exits out of sample at +20 bp each" in row
