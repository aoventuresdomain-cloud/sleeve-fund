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


@pytest.mark.parametrize("minutes,start", [(15, "2022-01-01 23:45"), (5, "2022-01-01 23:55")])
def test_a_study_on_bars_that_start_mid_day_counts_each_whole_day_once(tmp_path, instrument, minutes, start):
    """Review round 7, R7-M1: 15- and 5-minute studies crashed in the tear sheet ("duplicate labels"):
    a window starting at 23:45 put one day in two folds, and a part day at a window's end was passed
    off as a whole one. Each fold now counts only the days wholly inside its test window."""
    import numpy as np

    idx = pd.date_range(start, periods=150 * 1440 // minutes, freq=f"{minutes}min", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.003, len(idx))))
    o = np.r_[c[0], c[:-1]]
    bars = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1e6},
                        index=idx)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, bars, instrument, dataset=f"syn-{minutes}m", ledger=ledger, synthetic=True, holdout_days=30,
                  train_days=60, test_days=30, default_params={"fast": 20, "slow": 100}, use_holdout=True)
    assert len(r.folds) == 2
    days = pd.Series(r.oos_returns.index)
    assert days.is_unique and (days.diff().dropna() >= pd.Timedelta("1D")).all()
    assert 2 * 29 <= len(days) <= 2 * 30  # a test window starting at 23:45 loses the day it starts in
    assert (days.dt.floor("1D") == days).all() and days.max() <= r.research_end
    assert r.holdout is not None
    render(r, ledger)  # the tear sheet no longer crashes


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


def _stored_minutes(days, seed=6):
    """`days` of minute bars ending today, for the history store."""
    import numpy as np

    now = pd.Timestamp.now(tz="UTC").floor("1D")
    idx = pd.date_range(now - pd.Timedelta(days=days), periods=days * 1440, freq="1min", tz="UTC")
    c = 2_000 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.0008, len(idx))))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 5.0},
                        index=idx)


def test_a_store_study_runs_under_paper_rules(tmp_path):
    """Review round 7: research ran without the paper risk rules or execution bars, so a study could
    show a strategy trading on through a drawdown that halts it in paper. A store study now trades
    under the risk profile (halt and pause included) with resting orders matched on shorter bars."""
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.research.run import StudyRequest, run_store_study

    hist = HistoryStore(tmp_path / "hist")
    hist.append("KRAKEN", "ETH/USD", _stored_minutes(130), cursor="x")
    req = StudyRequest(strategy="buy_and_hold", pair="ETH/USD", minutes=240, risk_profile="conservative",
                       train_days=60, test_days=30, holdout_days=30)
    done = []
    sheet = run_store_study(req, progress=done.append, ledger_path=tmp_path / "l.jsonl", out_dir=tmp_path / "ts",
                            history=hist).read_text()
    assert "trades under the conservative risk profile" in sheet and "drawdown halt" in sheet
    assert "matched on 5-minute bars" in sheet  # 130 days fit the budget at 5 minutes
    assert "Tested on `ETH/USD` at 240-minute bars" in sheet
    assert done and done == sorted(done) and done[-1] <= 1
    with pytest.raises(ValueError, match="would take hours"):
        run_store_study(StudyRequest(strategy="buy_and_hold", pair="ETH/USD", minutes=1), history=hist)
    with pytest.raises(ValueError, match="no stored Kraken spot history for SOL/USD"):
        run_store_study(StudyRequest(strategy="buy_and_hold", pair="SOL/USD"), history=hist)


def test_a_halt_in_research_leaves_the_strategy_flat_as_paper_would(tmp_path, instrument):
    """The 60-minute study showed a 20% in-sample drawdown, past the balanced halt. Under a profile
    the strategy goes flat at the halt and stays flat; the benchmark is never halted."""
    import numpy as np

    idx = pd.date_range("2022-01-01", periods=200, freq="1D", tz="UTC")
    c = np.r_[np.linspace(100, 110, 60), np.linspace(110, 30, 40), np.linspace(30, 60, 100)]
    daily = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx)
    from sleeve_fund.strategies.buy_and_hold import SPEC as HOLD

    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(HOLD, daily, instrument, dataset="syn", ledger=ledger, synthetic=True, holdout_days=0,
                  train_days=60, test_days=60, risk_profile="conservative")
    eq = r.full_period.equity
    assert eq.iloc[-1] == pytest.approx(eq.iloc[-30], rel=1e-9)  # flat since the halt, though the price rose
    assert r.full_period_benchmark.equity.iloc[-1] > r.full_period_benchmark.equity.iloc[-30]


def test_the_sheet_says_when_out_of_sample_halted_and_counts_test_trades(tmp_path, instrument, monkeypatch):
    """Review round 8, R8-M4: a 15-minute study read +0.0% and Sharpe 0.00 in every test fold without
    saying the balanced halt had stopped it, passed "Enough trades" on in-sample trades, and printed
    "Deflated Sharpe: n/a probability"."""
    import numpy as np

    from sleeve_fund.research.tearsheet import g1_checks, oos_gaps
    from sleeve_fund.strategies.buy_and_hold import SPEC as HOLD

    idx = pd.date_range("2022-01-01", periods=200, freq="1D", tz="UTC")
    c = np.r_[np.linspace(100, 110, 60), np.linspace(110, 30, 40), np.linspace(30, 60, 100)]
    daily = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(HOLD, daily, instrument, dataset="syn", ledger=ledger, synthetic=True, holdout_days=0,
                  train_days=60, test_days=60, risk_profile="conservative")
    first, second = r.folds
    assert "in the test window" in first.halted and first.test_trades == 1  # the halt closed the trade
    assert "in the training stretch" in second.halted and second.test_trades == 0
    assert r.oos_trades == 1
    gaps = oos_gaps(r)
    assert "No trades out-of-sample in 1 of 2 test windows" in gaps
    assert "The conservative risk profile halted the strategy in 2 of 2 folds" in gaps
    enough = next(c for c in g1_checks(r, ledger) if c[0] == "Enough out-of-sample trades to judge")
    assert enough[1] == "FAIL" and enough[2].startswith("1 closed in the 2 walk-forward test windows")
    sheet = render(r, ledger)
    assert "> **No trades out-of-sample in 1 of 2 test windows.**" in sheet
    assert "| 0 (halted 29 Mar 2022) |" in sheet and "| 1 (halted 27 Mar 2022) |" in sheet  # halt dates per fold
    assert "**G1: FAIL**" in sheet  # one test window traded into its halt: a result, judged
    assert "kept in the repository" not in sheet
    import sleeve_fund.research.tearsheet as tearsheet

    monkeypatch.setattr(tearsheet, "deflated_sharpe_probability", lambda *a: float("nan"))
    sheet = render(r, ledger)
    assert "n/a probability" not in sheet
    assert "Deflated Sharpe: can't be computed here: out-of-sample needs at least 30 days whose returns vary" in sheet


def test_a_sheet_whose_test_windows_all_traded_has_no_gap_note():
    from types import SimpleNamespace

    from sleeve_fund.research.tearsheet import oos_gaps

    fold = SimpleNamespace(test_trades=3, halted="", test_end=pd.Timestamp("2024-01-01"))
    quiet = SimpleNamespace(test_trades=0, halted="", test_end=pd.Timestamp("2024-07-01"))
    assert oos_gaps(SimpleNamespace(folds=[fold, fold], risk_profile="balanced")) == ""
    words = oos_gaps(SimpleNamespace(folds=[fold, quiet], risk_profile=None))
    assert "No trades out-of-sample in 1 of 2" in words and "signal never closed a trade" in words
    assert "halted" not in words


def test_a_study_on_history_still_being_collected_says_where_it_ends(tmp_path):
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.research.run import StudyRequest, run_store_study

    hist = HistoryStore(tmp_path / "hist")
    m = _stored_minutes(130)
    hist.append("KRAKEN", "ETH/USD", m.set_axis(m.index - pd.Timedelta(days=10)), cursor="x")
    req = StudyRequest(strategy="buy_and_hold", pair="ETH/USD", minutes=240, train_days=60, test_days=30,
                       holdout_days=0)
    sheet = run_store_study(req, ledger_path=tmp_path / "l.jsonl", out_dir=tmp_path / "ts", history=hist).read_text()
    assert "The stored history ends" in sheet and "the collector is still catching up" in sheet


def _crashes(days=200, at=(20, 75)):
    """Daily closes that fall 70% over the 20 days after each of `at`, flat in between, then rise."""
    import numpy as np

    c = np.full(days, 100.0)
    for start in at:
        c[start:] *= np.r_[np.linspace(1, 0.3, 20), np.full(max(days - start - 20, 0), 0.3)][:days - start]
    c[at[-1] + 25:] *= np.linspace(1, 1.5, days - at[-1] - 25)
    idx = pd.date_range("2022-01-01", periods=days, freq="1D", tz="UTC")
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx)


def test_a_study_halted_before_every_test_window_is_not_judged_and_keeps_its_holdout(tmp_path, instrument):
    """Review round 8, M8-4: a study whose runs all halted in training read G1 FAIL on test windows that
    sat flat, and opened (spent) the holdout on it. It now reads "not judged", which is neither a pass
    nor a fail, and the holdout it asked for stays closed."""
    from sleeve_fund.dashboard.pipeline import sheet_facts
    from sleeve_fund.research.tearsheet import NOT_JUDGED, g1_checks, g1_verdict
    from sleeve_fund.strategies.buy_and_hold import SPEC as HOLD

    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(HOLD, _crashes(), instrument, dataset="syn-1440m", ledger=ledger, synthetic=True, holdout_days=20,
                  train_days=60, test_days=60, risk_profile="conservative", use_holdout=True)
    assert len(r.folds) == 2 and all(f.halted_before_test for f in r.folds)
    assert r.not_judged.startswith("the risk guard halted 2 of 2 folds before their test windows began")
    assert r.holdout is None and "left closed, though asked for" in r.holdout_withheld
    assert not ledger.holdout_used("buy_and_hold", "syn-1440m")
    assert g1_verdict(g1_checks(r, ledger))[0] == NOT_JUDGED
    sheet = tmp_path / "s.md"
    sheet.write_text(render(r, ledger))
    assert "**G1: NOT JUDGED**" in sheet.read_text() and sheet_facts(sheet)["g1"] == NOT_JUDGED


@pytest.mark.strategy_errors
def test_a_study_whose_strategy_raises_is_not_judged(tmp_path, instrument, monkeypatch):
    """Review round 8, M8-3: studies and tear sheets ignored the strategy's own errors, so a broken
    signal read as a result. The errors are counted, and G1 doesn't judge the study."""
    from sleeve_fund.research.tearsheet import NOT_JUDGED, g1_checks, g1_verdict
    from sleeve_fund.strategies import buy_and_hold

    def broken(self, bar):
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(buy_and_hold.BuyAndHold, "want_long", broken)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(buy_and_hold.SPEC, synthetic_ohlcv(days=200, seed=4), instrument, dataset="syn", ledger=ledger,
                  synthetic=True, holdout_days=20, train_days=60, test_days=60, use_holdout=True)
    assert r.error_count > 100 and r.errors[0][1:] == ("on_bar", "ZeroDivisionError('float division by zero')")
    assert "handling a bar: float division by zero (ZeroDivisionError)" in r.not_judged
    assert g1_verdict(g1_checks(r, ledger))[0] == NOT_JUDGED and r.holdout is None


def test_a_study_charges_the_spread_it_is_given(tmp_path, instrument):
    """Review round 8, m8-T: research ran with no spread and every test still passed."""
    from sleeve_fund.strategies.buy_and_hold import SPEC as HOLD

    r = run_study(HOLD, synthetic_ohlcv(days=200, seed=4), instrument, dataset="syn", ledger=IdeaLedger(tmp_path / "l"),
                  synthetic=True, holdout_days=0, train_days=60, test_days=60, half_spread=0.001)
    assert r.full_period.half_spread == 0.001 and r.full_period.spread_paid > 0
    assert "0.100% of the price as half the bid-ask spread" in r.fee_note


def test_whole_days_drops_the_day_a_window_starts_part_way_through():
    """Review round 8, m8-T: a test window whose first bar starts at noon must not count that day."""
    from sleeve_fund.research.metrics import whole_days

    days = pd.date_range("2022-01-01", periods=5, freq="1D", tz="UTC")  # each return stamped at the day's close
    returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=days)
    bar = pd.Timedelta(minutes=15)
    kept = whole_days(returns, pd.Timestamp("2022-01-01 12:15", tz="UTC"), bar)  # first bar 12:00 to 12:15
    assert list(kept.index) == list(days[2:])  # not 2 Jan's close: that day began before the window
    assert list(whole_days(returns, pd.Timestamp("2022-01-01 00:15", tz="UTC"), bar).index) == list(days[1:])


def test_a_holdout_opened_at_one_bar_length_is_spent_at_every_other(tmp_path):
    """Review round 8, m8-T: the same days seen once on hourly bars are not fresh on 15-minute ones."""
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    ledger.record(idea="a", family="trend", params={}, dataset="kraken-btcusd-store-60m", stage="holdout", sharpe=0.1)
    assert ledger.holdout_used("a", "kraken-btcusd-store-15m") and ledger.holdout_used("a", "kraken-btcusd-store-1440m")
    assert not ledger.holdout_used("a", "kraken-ethusd-store-60m") and not ledger.holdout_used("b", "kraken-btcusd-store-60m")
