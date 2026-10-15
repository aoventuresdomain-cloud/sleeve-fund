"""ABSORB-GAP (INSURE-RULE point 5; Advisor 15 Oct ~02:05 UK, points 2-4): a liquidation is booked at its bankruptcy
price (GAP-LIQ-CAP), so a backtest that gapped past it never pays what the venue absorbed. Research charges it: the
run's equity less the absorbed gaps to date, stepped down at each liquidation, nothing re-sized ("gap charged, no
re-sizing"); every test that reads returns reads that curve, PBO and the walk-forward folds included. Each window
reports what it absorbed, with out-of-sample and the holdout uncharged beside, for information only. An idea that
passes uncharged but fails charged FAILs, saying it "passes only because the venue absorbs liquidation gaps"."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas as pd
import pytest
from test_long_short import PERP, _gapped

from sleeve_fund.research import study, tearsheet
from sleeve_fund.research.metrics import summary
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.research.study import _charge_absorbed, _liquidations


def _gapped_run(prices, instrument):
    closes = [100.0, 100.5, 100.8, 101.5, 101.5, 400.0, 400.0, 400.0]  # a short gapped far past its bankruptcy price
    return run_backtest("ping_pong", _gapped(prices, closes), instrument, PERP, half_spread=0,
                        risk_profile="aggressive")


def test_a_gapped_liquidation_charges_the_curve_from_its_bar_on_and_nothing_before(prices, instrument):
    res = _gapped_run(prices, instrument)
    engine = res.equity.copy()
    (gap,) = res.absorbed
    assert _charge_absorbed(res) is res
    pd.testing.assert_series_equal(res.uncharged_equity, engine)
    step = int(engine.index.searchsorted(pd.Timestamp(gap["ts"]), side="left"))
    assert 0 < step < len(engine)
    before, after = slice(None, step), slice(step, None)
    pd.testing.assert_series_equal(res.equity.iloc[before], engine.iloc[before])
    assert ((engine.iloc[after] - res.equity.iloc[after]).round(2) == round(gap["amount"], 2)).all()


def test_the_liquidation_row_names_the_gap_the_alert_named_to_the_cent(prices, instrument):
    res = _gapped_run(prices, instrument)
    (gap,) = res.absorbed
    (liq,) = _liquidations(res, res.equity.index[0], res.equity.index[-1], "holdout")
    assert liq["absorbed"] == round(gap["amount"], 2) > 0


def test_a_run_that_absorbed_nothing_is_left_as_it_was(prices, instrument):
    closes = [100.0, 100.5, 100.8, 101.5, 101.0, 100.0, 99.0, 98.0, 98.0]
    res = run_backtest("ping_pong", _gapped(prices, closes), instrument, {**PERP, "dip": 0.05}, half_spread=0)
    engine = res.equity
    assert res.absorbed == [] and _charge_absorbed(res).equity is engine and res.uncharged_equity is None


GAP = 400.0


def test_every_return_the_study_reads_is_charged_and_each_window_reports_its_own(monkeypatch, tmp_path, instrument):
    """A gap absorbed on every run's last bar: the sensitivity grid (what PBO is fed), the walk-forward folds and the
    holdout all read the charged curve; each window names its gap, and the uncharged days sit beside."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.strategies.trend_filter import SPEC

    real = study.run_backtest

    def gapped(*a, **k):
        res = real(*a, **k)
        if a[0] != "buy_and_hold" and len(res.equity):
            res.absorbed = [{"ts": res.equity.index[-1], "amount": GAP}]
        return res

    monkeypatch.setattr(study, "run_backtest", gapped)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = study.run_study(SPEC, synthetic_ohlcv(days=1900, seed=3), instrument, dataset="syn", ledger=ledger,
                        synthetic=True, holdout_days=200, train_days=730, test_days=365, use_holdout=True)
    assert r.folds and all(f.absorbed and f.absorbed[0]["amount"] == GAP for f in r.folds if not f.unscored)
    scored = [f for f in r.folds if not f.unscored]
    for f in scored:
        assert summary(f.uncharged_returns)["cagr"] > f.test["cagr"]
    assert r.oos_uncharged_returns is not None and not r.oos_uncharged_returns.equals(r.oos_returns)
    assert [a["amount"] for a in r.holdout_absorbed] == [GAP]
    assert r.holdout_uncharged["sharpe"] != pytest.approx(r.holdout["sharpe"])
    full = r.full_period
    assert full.uncharged_equity is not None
    assert full.equity.iloc[-1] == pytest.approx(full.uncharged_equity.iloc[-1] - GAP)
    row = next(c for c in tearsheet.render(r, ledger).splitlines() if c.startswith(f"| {tearsheet.ABSORB_CHECK} |"))
    assert f"{GAP * (len(scored) + 1):,.2f} over {len(scored) + 1} liquidations" in row
    assert "out-of-sample test window 1 400.00 over 1 liquidation" in row and "holdout 400.00" in row
    assert "gap charged, no re-sizing" in row and "Uncharged, for information only: out-of-sample Sharpe" in row


@dataclass
class _Fold:
    absorbed: list = field(default_factory=list)


@dataclass
class _Result:
    folds: list
    oos_returns: pd.Series
    oos_uncharged_returns: pd.Series | None
    oos_compare_returns: pd.Series | None = None
    holdout: dict | None = None
    holdout_absorbed: list = field(default_factory=list)
    holdout_uncharged: dict | None = None


DAYS = pd.date_range("2026-01-01", periods=60, freq="1D", tz="UTC")
GOOD = pd.Series([0.002, -0.001, 0.003] * 20, index=DAYS)
BAD = pd.Series([-0.002, 0.001, -0.003] * 20, index=DAYS)


def _judge(view):
    return [(tearsheet.SHARPE_CHECK, "PASS" if summary(view.oos_returns)["sharpe"] > 0 else "FAIL", "")]


@pytest.mark.parametrize("charged, uncharged, fail_row", [(BAD, GOOD, True), (GOOD, GOOD, False), (BAD, BAD, False)])
def test_an_idea_that_passes_only_uncharged_fails_saying_so_and_its_uncharged_pass_still_shows(charged, uncharged,
                                                                                                   fail_row):
    r = _Result(folds=[_Fold([{"ts": DAYS[5], "amount": 250.0}])], oos_returns=charged, oos_uncharged_returns=uncharged)
    checks = _judge(r)
    rows = tearsheet._absorb_rows(r, checks, _judge)
    (name, verdict, words), *rest = rows
    assert (name, verdict) == (tearsheet.ABSORB_CHECK, "INFO")
    assert words.startswith("250.00 over 1 liquidation (out-of-sample test window 1 250.00 over 1 liquidation)")
    assert f"out-of-sample Sharpe {tearsheet._num(summary(uncharged)['sharpe'])} " in words
    assert bool(rest) is fail_row
    if fail_row:
        ((fname, fverdict, fwords),) = rest
        assert (fname, fverdict) == (tearsheet.ABSORB_FAIL_CHECK, "FAIL")
        assert fwords == f"{tearsheet.ABSORB_FAIL}: it passes G1 uncharged, and charged it fails {tearsheet.SHARPE_CHECK}"
        assert tearsheet.g1_verdict(checks + rows) == ("FAIL", [tearsheet.SHARPE_CHECK, tearsheet.ABSORB_FAIL_CHECK])


def test_nothing_absorbed_adds_no_row():
    r = _Result(folds=[_Fold()], oos_returns=GOOD, oos_uncharged_returns=None)
    assert tearsheet._absorb_rows(r, _judge(r), _judge) == []


# Random entry (Advisor ~02:55 UK, option A): the strategy's liquidated trip pays its absorbed gap, over its entry
# notional, the unit and time of its other trips; the draws never do.

def test_random_entry_charges_the_strategys_liquidated_trip_and_never_a_draw():
    import numpy as np

    from sleeve_fund.research.random_entry import Trade, random_entry, random_side

    closes = np.linspace(100.0, 110.0, 40)
    trips = [Trade(2, 6, 1), Trade(10, 15, 1), Trade(20, 30, -1)]
    charged = [trips[0], Trade(10, 15, 1, charge=0.05), trips[2]]
    windows = [(0, 39)]
    free, paid = random_entry(closes, trips, windows, 0.001), random_entry(closes, charged, windows, 0.001)
    assert paid.median_random_return == free.median_random_return  # the draws are never charged
    assert (1 + paid.strategy_return) == pytest.approx((1 + free.strategy_return) * (1 - 0.05 / (1 + _ret(closes, 10, 15))))
    side_free, side_paid = random_side(closes, trips, windows, 0.001), random_side(closes, charged, windows, 0.001)
    assert side_paid.median_random_return == side_free.median_random_return
    assert side_paid.strategy_return < side_free.strategy_return


def _ret(closes, a, b):
    return closes[b] / closes[a] - 1 - 2 * 0.001


def test_a_gap_is_charged_to_the_counted_trip_it_closed_over_that_trips_entry_notional():
    t = lambda d: pd.Timestamp(d, tz="UTC")  # noqa: E731
    trips = [{"opened": t("2026-01-02"), "closed": t("2026-01-04"), "cost": 6_000.0},
             {"opened": t("2026-01-05"), "closed": t("2026-01-07"), "cost": 8_000.0},
             {"opened": t("2026-01-08"), "closed": None, "cost": 1_000.0}]  # still open: in no comparison
    gaps = [{"ts": t("2026-01-07"), "amount": 400.0}]
    assert study._trip_charges(trips, t("2026-01-01"), gaps) == [(t("2026-01-05"), 0.05)]


@dataclass
class _REResult(_Result):
    random_entry: object = None
    random_entry_uncharged: object = None


def _judge_random(view):
    return [(tearsheet.RANDOM_ENTRY_CHECK, "PASS" if view.random_entry.return_percentile >= 95 else "FAIL", "")]


def test_an_idea_whose_random_entry_passes_only_uncharged_fails_saying_so():
    r = _REResult(folds=[_Fold([{"ts": DAYS[5], "amount": 250.0}])], oos_returns=GOOD, oos_uncharged_returns=GOOD,
                  random_entry=SimpleNamespace(return_percentile=80.0),
                  random_entry_uncharged=SimpleNamespace(return_percentile=97.0))
    checks = _judge_random(r)
    info, fail = tearsheet._absorb_rows(r, checks, _judge_random)
    assert "random entry beats 97% uncharged (charged 80%)" in info[2]
    assert fail == (tearsheet.ABSORB_FAIL_CHECK, "FAIL",
                    f"{tearsheet.ABSORB_FAIL}: it passes G1 uncharged, and charged it fails {tearsheet.RANDOM_ENTRY_CHECK}")
