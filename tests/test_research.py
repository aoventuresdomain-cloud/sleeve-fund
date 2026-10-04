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


def test_study_caps_strategy_and_benchmark_like_paper(tmp_path, instrument):
    """Review R2-M5: research is judged at the paper risk profile's exposure, benchmark included."""
    prices = synthetic_ohlcv(days=1500, seed=3)
    kw = dict(dataset="syn", ledger=IdeaLedger(tmp_path / "l.jsonl"), synthetic=True, holdout_days=100,
              train_days=730, test_days=300)
    r = run_study(SPEC, prices, instrument, position_cap=0.33, **kw)
    for res in (r.full_period, r.full_period_benchmark):  # sized at the cap on entry (it drifts with price after)
        first = res.exposure[res.exposure > 0].iloc[0]
        assert 0.3 < first <= 0.34
    assert any("capped at 33%" in n for n in r.notes)
    sheet = render(r, kw["ledger"])
    assert "capped at 33%" in sheet
    # Round 4, B5: the sheet says what it tested, and G1 needs more than a higher Sharpe by luck.
    assert "Tested on `BTC/USD` at 1440-minute bars" in sheet
    assert "likely to beat it by more than the best of" in sheet and "bar: 95%" in sheet
    with pytest.raises(ValueError):
        run_study(SPEC, prices, instrument, position_cap=1.5, **kw)


def test_cli_reads_the_history_store(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    from sleeve_fund import history
    from sleeve_fund.__main__ import main

    idx = pd.date_range("2020-01-01", periods=1700 * 1440 // 60, freq="60min", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.004, len(idx))))
    hourly = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    minutes = hourly.resample("1min").ffill()
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "BTC/USD", minutes, cursor="x")
    out = tmp_path / "sheet.md"
    assert main(["--ledger", str(tmp_path / "l.jsonl"), "study", "buy_and_hold", "--store", "--holdout-days", "100",
                 "--out", str(out)]) == 0
    assert "capped at 33%" in out.read_text()


def test_an_hourly_study_is_judged_on_daily_returns(tmp_path, instrument):
    """Review rounds 2 to 6: research was daily only, so hourly and minute strategies had no G1; and
    intraday returns annualised with sqrt(365) and bootstrapped in blocks of bars would mislead. The
    study now takes any bar that divides a day, with its windows in days, and judges daily returns."""
    import numpy as np

    from sleeve_fund.research.tearsheet import g1_checks

    idx = pd.date_range("2022-01-01 01:00", periods=420 * 24, freq="60min", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.006, len(idx))))
    o = np.r_[c[0], c[:-1]]
    hourly = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                           "volume": 1e6}, index=idx)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, hourly, instrument, dataset="syn-60m", ledger=ledger, synthetic=True, holdout_days=60,
                  train_days=180, test_days=60, default_params={"fast": 20, "slow": 100})
    assert r.bar_minutes == 60
    assert r.research_end == hourly.index[-60 * 24 - 1]  # the holdout is 60 days of hours
    assert len(r.folds) == 3  # 360 research days: train 180, then three 60-day tests
    days = pd.Series(r.oos_returns.index).diff().dropna()
    assert (days == pd.Timedelta("1D")).all() and len(r.oos_returns) == 3 * 60  # each test day, once
    assert r.oos_returns.index.max() <= r.research_end.ceil("1D")
    sheet = render(r, ledger)
    assert "Tested on `BTC/USD` at 60-minute bars" in sheet
    # 180 daily observations are too few to judge: not a pass, however the Sharpe reads.
    sharpe = next(c for c in g1_checks(r, ledger) if c[0].startswith("G1 test"))
    assert sharpe[1] == "FAIL" and "too few independent out-of-sample days" in sharpe[2]
    with pytest.raises(ValueError, match="don't divide a day"):
        run_study(SPEC, hourly.iloc[::7], instrument, dataset="x", ledger=ledger)


def test_slow_regimes_count_as_few_independent_days():
    """Review round 6: the independence check looked one day back, so 60-day regimes with a lag-1
    autocorrelation of 0.3 passed G1 by luck 10.7% of the time. It now sums every autocorrelation
    that matters."""
    import math

    import numpy as np

    from sleeve_fund.research.metrics import independent_days

    rng = np.random.default_rng(0)
    n = 1095
    iid = rng.normal(size=n)
    regimes = math.sqrt(0.3) * np.repeat(rng.choice([-1, 1], size=n // 60 + 1), 60)[:n] + math.sqrt(0.7) * rng.normal(size=n)
    assert independent_days(iid) > 0.9 * n
    assert independent_days(regimes) < 250  # below MIN_INDEPENDENT_DAYS: not judged
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.3 * ar[i - 1] + rng.normal()
    assert 0.35 * n < independent_days(ar) < 0.75 * n  # about (1 - 0.3) / (1 + 0.3) of the days


def test_cli_studies_hourly_bars_from_the_store(tmp_path, monkeypatch):
    import numpy as np

    from sleeve_fund import history
    from sleeve_fund.__main__ import main

    idx = pd.date_range("2024-01-01", periods=500 * 24, freq="60min", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.004, len(idx))))
    minutes = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx).resample("1min").ffill()
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "BTC/USD", minutes, cursor="x")
    assert main(["--ledger", str(tmp_path / "l.jsonl"), "study", "buy_and_hold", "--store", "--minutes", "60",
                 "--holdout-days", "60", "--train-days", "180", "--test-days", "90", "--out", str(tmp_path / "s.md")]) == 0
    sheet = (tmp_path / "s.md").read_text()
    assert "at 60-minute bars" in sheet and "kraken-btcusd-store-60m" in sheet
