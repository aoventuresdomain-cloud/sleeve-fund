import pandas as pd
import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.metrics import expected_max_sharpe, max_drawdown, summary
from sleeve_fund.strategies.trend_filter import SPEC
from sleeve_fund.research.study import grid, run_study
from sleeve_fund.research.tearsheet import render


def test_grid_drops_invalid_combos():
    combos = grid({"fast": [50, 100], "slow": [100, 200]})
    assert {"fast": 100, "slow": 100} not in combos
    assert len(combos) == 3


def test_luck_hurdle_rises_with_trials():
    assert expected_max_sharpe(1, 0.5) == 0
    assert expected_max_sharpe(10, 0.5) < expected_max_sharpe(100, 0.5) < expected_max_sharpe(1000, 0.5)


def test_metrics_on_known_series():
    r = pd.Series([0.1, -0.5, 0.2])
    assert max_drawdown((1 + r).cumprod()) == pytest.approx(-0.5)
    assert summary(r)["max_drawdown"] == pytest.approx(-0.5)


def test_ledger_counts_variants_not_refits(tmp_path):
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    for stage in ("sensitivity", "wf_train", "wf_train"):
        ledger.record(idea="a", family="trend", params={"x": 1}, dataset="d", stage=stage, sharpe=0.1)
    ledger.record(idea="b", family="momentum", params={"x": 1}, dataset="d", stage="sensitivity", sharpe=0.2)
    ledger.record(idea="buy_and_hold", family="benchmark", params={}, dataset="d", stage="x", sharpe=0.3)
    c = ledger.counts()
    assert (c["ideas"], c["variants"], c["evaluations"]) == (2, 2, 4)
    assert not ledger.holdout_used("a", "d")


def test_study_end_to_end_and_holdout_flag(tmp_path, instrument):
    prices = synthetic_ohlcv(days=1900, seed=3)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    kw = dict(dataset="syn", ledger=ledger, synthetic=True, holdout_days=200, train_days=730, test_days=365)
    r = run_study(SPEC, prices, instrument, **kw)
    assert r.research_end < prices.index[-200]
    assert len(r.folds) == 2
    assert r.oos_returns.index.max() <= r.research_end
    assert r.holdout is None
    assert "Synthetic data" in render(r, ledger)

    first = run_study(SPEC, prices, instrument, use_holdout=True, **kw)
    assert first.holdout is not None and not first.holdout_reused
    second = run_study(SPEC, prices, instrument, use_holdout=True, **kw)
    assert second.holdout_reused
    assert "holdout opened more than once" in render(second, ledger)


def test_trade_stats_after_fees_with_partial_fills():
    from sleeve_fund.research.metrics import trade_stats, trades

    rows = [
        {"side": "BUY", "qty": 0.5, "price": 100, "fee": 0.4},
        {"side": "BUY", "qty": 0.5, "price": 100, "fee": 0.4},
        {"side": "SELL", "qty": 1.0, "price": 120, "fee": 0.96},  # +20 gross, +18.24 net
        {"side": "SELL", "qty": 1.0, "price": 999, "fee": 0},  # nothing open: ignored
        {"side": "BUY", "qty": 1.0, "price": 100, "fee": 0.8},
        {"side": "SELL", "qty": 1.0, "price": 95, "fee": 0.76},  # -5 gross, -6.56 net
        {"side": "BUY", "qty": 1.0, "price": 100, "fee": 0.8},  # still open: not counted
    ]
    t = trades(rows)
    assert [round(x["pnl"], 2) for x in t] == [18.24, -6.56]
    s = trade_stats(t)
    assert s["trades"] == 2 and s["win_rate"] == 0.5 and round(s["pnl"], 2) == 11.68
    assert round(s["profit_factor"], 2) == round(18.24 / 6.56, 2)
    assert trade_stats([])["trades"] == 0


def test_deflated_sharpe_with_the_no_skill_spread_gets_harder_with_more_tries():
    import numpy as np
    import pandas as pd

    from sleeve_fund.research.metrics import deflated_sharpe_probability

    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.02, 1500))
    p1, p20, p200 = (deflated_sharpe_probability(r, n) for n in (1, 20, 200))
    assert 0 < p200 < p20 < p1 <= 1
