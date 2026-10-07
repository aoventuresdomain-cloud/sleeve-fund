import math

import numpy as np
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
    # C3: the out-of-sample trades against random entry times, in G1 (v2 P1-7).
    assert r.random_entry is not None and r.random_entry.trades <= r.oos_trades
    assert "Beats random entry times" in render(r, ledger)

    first = run_study(SPEC, prices, instrument, use_holdout=True, **kw)
    assert first.holdout is not None
    # A second opening, at any bar length, leaves it closed and says when it was opened (review round 10, M10-1).
    for dataset in ("syn", "syn-60m"):
        second = run_study(SPEC, prices, instrument, use_holdout=True, **dict(kw, dataset=dataset))
        assert second.holdout is None
        assert "opened on " in second.holdout_withheld and "on daily bars" in second.holdout_withheld
        assert "a second look can't be fresh" in render(second, ledger)
    assert sum(e["stage"] == "holdout" for e in ledger.entries()) == 1


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
    with pytest.raises(ValueError, match="no stored history for SOL/USD on this venue"):
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
    # Each test window starts flat (P1-D13 [flat-rule]): the first enters on its own first bar and the halt closes
    # that trade inside the window, so it counts; nothing is carried in from training, where no run halts now.
    assert "in the test window" in first.halted and first.test_trades == 1 and first.carried_in == 0
    assert not second.halted and second.test_trades == 0
    assert not any(f.halted_before_test for f in r.folds)
    assert r.oos_trades == 1 and r.excluded_trades == 1
    gaps = oos_gaps(r)
    assert "No trades out-of-sample in 1 of 2 test windows" in gaps
    assert ("The conservative risk profile halted the strategy in 1 of 2 folds: 1 inside the test window, flat "
            "from then on (1 of them closed a trade first)") in gaps
    assert "Each test window starts flat" in gaps
    enough = next(c for c in g1_checks(r, ledger) if c[0] == "Enough out-of-sample trades to judge")
    # One of two windows blind, so half: not judged since round 10 (M10-1), and the trade count can't fail.
    assert enough[1] == "N/A" and enough[2].startswith("not judged: 1 closed in the 2 walk-forward test windows")
    assert "1 left out at the windows' edges" in enough[2]
    sheet = render(r, ledger)
    assert "> **No trades out-of-sample in 1 of 2 test windows.**" in sheet
    assert "| 1 (halted 29 Mar 2022) |" in sheet  # the halt date on its fold
    assert "**G1: NOT JUDGED**" in sheet  # the one window left traded into its halt
    assert "kept in the repository" not in sheet
    import sleeve_fund.research.tearsheet as tearsheet

    monkeypatch.setattr(tearsheet, "deflated_sharpe_probability", lambda *a: float("nan"))
    sheet = render(r, ledger)
    assert "n/a probability" not in sheet
    assert "Deflated Sharpe: can't be computed here: out-of-sample needs at least 30 days whose returns vary" in sheet


def test_a_sheet_whose_test_windows_all_traded_has_no_gap_note():
    from types import SimpleNamespace

    from sleeve_fund.research.tearsheet import oos_gaps

    fold = SimpleNamespace(test_trades=3, closed_in_window=3, halted="", test_end=pd.Timestamp("2024-01-01"))
    quiet = SimpleNamespace(test_trades=0, closed_in_window=0, halted="", test_end=pd.Timestamp("2024-07-01"))
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


def test_a_study_g1_cannot_judge_is_not_judged_and_keeps_its_holdout(tmp_path, instrument):
    """Review round 8, M8-4: a study whose runs all halted in training read G1 FAIL on test windows that
    sat flat, and opened (spent) the holdout on it. It now reads "not judged", which is neither a pass
    nor a fail, and the holdout it asked for stays closed. Since P1-D13 every test window starts flat, so no run
    halts before one; this study is not judged because the risk guard's resting stops need 1-minute bars."""
    from sleeve_fund.dashboard.pipeline import sheet_facts
    from sleeve_fund.research.tearsheet import NOT_JUDGED, g1_checks, g1_verdict
    from sleeve_fund.strategies.buy_and_hold import SPEC as HOLD

    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(HOLD, _crashes(), instrument, dataset="syn-1440m", ledger=ledger, synthetic=True, holdout_days=20,
                  train_days=60, test_days=60, risk_profile="conservative", use_holdout=True)
    assert len(r.folds) == 2 and not any(f.halted_before_test for f in r.folds)
    assert r.not_judged.startswith("no 1-minute execution data")
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


def _folds(*spec):
    """Folds as (halted, halted_before_test, test_trades)."""
    from types import SimpleNamespace

    return [SimpleNamespace(halted="12 Mar 2022 (drawdown)" if h else "", halted_before_test=b, test_trades=t, closed_in_window=t)
            for h, b, t in spec]


def test_a_study_whose_test_windows_are_half_or_more_blind_is_not_judged():
    """Review round 9, N7: a 5-minute study with 5 of 6 folds halted and 4 of 6 test windows empty read
    G1 FAIL on the 2 windows left. A window is blind when a halt kept it flat or left it without a trade;
    with half or more blind the study is not judged. A window the signal simply never traded in, or one
    halted after it traded, is still a result."""
    from types import SimpleNamespace

    from sleeve_fund.research.study import StudyResult

    def why(folds):
        return StudyResult.not_judged.fget(SimpleNamespace(error_count=0, folds=folds,
                                                           oos_trades=sum(f.test_trades for f in folds)))

    reviewer = _folds((1, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 0), (1, 0, 3), (0, 0, 4))
    assert why(reviewer).startswith("the risk guard halted the strategy in 5 of 6 folds, leaving 4 of 6 test "
                                    "windows flat or without a trade, and out-of-sample closed 7 trades")
    assert why(_folds((1, 1, 0), (1, 0, 2), (0, 0, 0), (0, 0, 0), (0, 0, 5), (0, 0, 1))) == ""  # 1 blind, 2 quiet
    # Review round 10, M10-1: exactly half blind is not judged either; it read FAIL on one live window.
    round10 = _folds((0, 0, 4), (1, 0, 1), (1, 1, 0), (1, 0, 1), (1, 1, 0), (1, 0, 0))
    assert why(round10).startswith("the risk guard halted the strategy in 5 of 6 folds, leaving 3 of 6 test windows")
    assert why(_folds((1, 1, 0), (0, 0, 2), (0, 0, 1), (0, 0, 4))) == ""  # a quarter blind is judged
    assert why(_folds((1, 1, 0), (0, 0, 0))) != ""  # no trade out-of-sample at all, with a halt behind it


def test_checks_resting_on_missing_out_of_sample_are_not_failed_on_a_not_judged_sheet(tmp_path, instrument):
    """Review round 9, N6: a NOT JUDGED sheet still showed red Fail chips for the Sharpe, robustness and
    trade-count rows: they measure the out-of-sample the study didn't get."""
    from sleeve_fund.research.tearsheet import NOT_APPLICABLE, NOT_JUDGED, OOS_CHECKS, g1_checks, g1_verdict
    from sleeve_fund.strategies.buy_and_hold import SPEC as HOLD

    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(HOLD, _crashes(), instrument, dataset="syn-1440m", ledger=ledger, synthetic=True, holdout_days=0,
                  train_days=60, test_days=60, risk_profile="conservative")
    checks = g1_checks(r, ledger)
    assert g1_verdict(checks)[0] == NOT_JUDGED
    rows = {name: (verdict, ev) for name, verdict, ev in checks}
    assert all(rows[name][0] in (NOT_APPLICABLE, "PASS") for name in OOS_CHECKS)
    assert rows["Enough out-of-sample trades to judge"][0] == NOT_APPLICABLE
    assert rows["Enough out-of-sample trades to judge"][1].startswith("not judged: 1 closed")  # windows start flat
    assert "| FAIL |" not in render(r, ledger)


def test_every_study_runs_the_cost_ladder_and_names_the_break_even_fee(tmp_path, instrument):
    """PM, 5 Oct 2026: each idea is tested at 0, 0.02, 0.05, 0.1 and 0.8% per side, and the tear sheet says the fee
    at which it stops making money. The rungs differ only in fees, so return falls as the fee rises, and the
    ladder leaves the idea counter alone."""
    from sleeve_fund.research.study import COST_LADDER
    from sleeve_fund.research.study import breakeven_fee as breakeven_fee_of
    from sleeve_fund.research.tearsheet import g1_checks as g1_checks_of

    prices = synthetic_ohlcv(days=1500, seed=3)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, prices, instrument, dataset="syn", ledger=ledger, synthetic=True, holdout_days=100,
                  train_days=730, test_days=365)
    assert [x.fee for x in r.cost_ladder] == list(COST_LADDER)
    assert r.cost_ladder[0].fees_paid == 0 and r.cost_ladder[-1].fees_paid > r.cost_ladder[1].fees_paid > 0
    returns = [x.total_return for x in r.cost_ladder]
    assert returns == sorted(returns, reverse=True) and returns[0] > returns[-1]
    assert len({x.round_trips for x in r.cost_ladder[:3]}) == 1  # same trades; only the fee moves
    sheet = render(r, ledger)
    assert "## Cost ladder" in sheet and "**Break-even fee:**" in sheet and "| 0.80% |" in sheet
    assert not any(e["stage"].startswith("ladder") for e in ledger.entries())
    assert r.ladder_slippage == 0.0002 and "0.02% slippage" in sheet  # BTC: 2 basis points; 5 elsewhere
    from sleeve_fund.research.study import ladder_slippage

    assert (ladder_slippage("ETH/USDT"), ladder_slippage("SUI/USD")) == (0.0002, 0.0005)
    # v2 P1-6: every grid point gets its own ladder and break-even fee. P1-G1/G2: the one above is the chosen
    # settings', re-run to verify it; the grid's others are interpolated.
    assert r.sensitivity["breakeven"].map(bool).all()
    chosen = r.sensitivity[(r.sensitivity["fast"] == r.chosen_params["fast"])
                           & (r.sensitivity["slow"] == r.chosen_params["slow"])].iloc[0]
    assert chosen["breakeven"] == r.breakeven and r.chosen_params == r.sensitivity.loc[r.sensitivity["sharpe"].idxmax(),
                                                                                       ["fast", "slow"]].to_dict()
    assert r.breakeven.split(" (")[0] == breakeven_fee_of(r.cost_ladder)[1].split(" (")[0]
    assert "| Break-even fee per side, on top of spread and slippage |" in sheet
    from sleeve_fund.dashboard import pipeline
    from sleeve_fund.research.guardrails import G1_RULES

    (tmp_path / "s.md").write_text(sheet)  # QA F3: a new sheet says its rules, so it is never an old bar
    assert f"G1 rules: {G1_RULES}" in sheet and not pipeline.sheet_facts(tmp_path / "s.md")["evidence"].startswith("Old")
    checks = dict((name, verdict) for name, verdict, _ in g1_checks_of(r, ledger))
    assert checks["Break-even fee (shown, not a test)"] == "INFO" and "Holds at nearby settings" in checks


def test_the_break_even_fee_is_read_between_the_rungs_either_side_of_zero():
    from sleeve_fund.research.study import LadderRung, breakeven_fee

    def ladder(*rets):
        return [LadderRung(fee=f, total_return=r, sharpe=0.0, round_trips=10, fees_paid=0.0)
                for f, r in zip((0.0, 0.0002, 0.0005, 0.001, 0.008), rets)]

    fee, words = breakeven_fee(ladder(0.10, 0.08, 0.05, -0.05, -0.9))
    # P1-G1: log(1 + return) is interpolated, since fees compound per trade; a straight line gave 0.075%.
    assert fee == pytest.approx(0.0005 + 0.0005 * math.log(1.05) / math.log(1.05 / 0.95))
    assert "about 0.074%" in words and "interpolated between 0.05% and 0.10%" in words
    assert breakeven_fee(ladder(-0.01, -0.02, -0.03, -0.04, -0.5)) == (None, "loses money even at 0.00% fees")
    assert breakeven_fee(ladder(0.5, 0.5, 0.4, 0.4, 0.1))[1] == "still makes money at 0.80% per side, the top of the ladder"


def _ar1_daily(phi: float, seed: int, days: int = 1500) -> pd.DataFrame:
    """Daily bars whose returns carry autocorrelation, so a fast trend filter earns a gross edge over many
    trades: QA's stress case for the break-even fee (quant-review/v2-p1/guardrails.md, g3b)."""
    from sleeve_fund.data import validate_ohlcv

    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.02, days)
    rets = np.zeros(days)
    for i in range(1, days):
        rets[i] = phi * rets[i - 1] + eps[i]
    close = 10_000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[10_000.0], close[:-1]])
    wig = np.abs(rng.normal(0, 0.005, days))
    idx = pd.date_range("2019-01-01", periods=days, freq="1D", tz="UTC") + pd.Timedelta("1D")
    return validate_ohlcv(pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + wig),
                                        "low": np.minimum(open_, close) * (1 - wig), "close": close,
                                        "volume": 1_000.0}, index=pd.DatetimeIndex(idx, name="timestamp")))


def test_the_break_even_fee_is_where_a_re_run_nets_zero_on_qas_fast_2_slow_3_case():
    """QA P1-G1: fast 2 / slow 3 on AR(1) daily bars (phi 0.1, seed 3) reported 0.291% per side between the
    0.10% and 0.80% rungs, against a true 0.163%, and lost 46% at the reported fee. The figure must now be
    one a re-run at that fee confirms nets about zero, from the old wide gap as well as the new ladder."""
    from decimal import Decimal

    from sleeve_fund.instruments import FeeSchedule
    from sleeve_fund.research.metrics import round_trips
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.research.study import (BREAKEVEN_TOLERANCE, COST_LADDER, LadderRung, breakeven_fee,
                                            ladder_slippage)
    from sleeve_fund.strategies.trend_filter import SPEC as TREND
    from sleeve_fund.venues import venue

    research = _ar1_daily(0.1, 3).iloc[:-100]
    inst = venue("KRAKEN").instrument("BTC", "USD")
    params = {**TREND.default_params, "fast": 2, "slow": 3}
    half = venue("KRAKEN").assumed_half_spread + ladder_slippage("BTC/USD")
    runs = []

    def run(fee: float):
        return run_backtest(TREND.name, research, inst, params, bar_minutes=1440, half_spread=half,
                            fees=FeeSchedule(maker=Decimal(str(fee)), taker=Decimal(str(fee))))

    def run_at(fee: float) -> float:
        runs.append(fee)
        return float(run(fee).equity.iloc[-1] / 10_000.0 - 1)

    def rung(fee: float) -> LadderRung:
        res = run(fee)
        return LadderRung(fee=fee, total_return=float(res.equity.iloc[-1] / 10_000.0 - 1), sharpe=0.0,
                          round_trips=len(round_trips(res.fills, res.shorts)), fees_paid=res.fees_paid)

    old = [rung(f) for f in (0.0, 0.0002, 0.0005, 0.001, 0.008)]
    lo, hi = old[3], old[4]
    chord = lo.fee + (hi.fee - lo.fee) * lo.total_return / (lo.total_return - hi.total_return)
    assert chord == pytest.approx(0.00291, abs=0.00005)  # the old straight line, as QA found it

    for ladder in (old, [rung(f) for f in COST_LADDER]):
        runs.clear()
        fee, words = breakeven_fee(ladder, run_at)
        assert fee == pytest.approx(0.00163, abs=0.00005) and "verified" in words
        assert abs(run_at(fee)) <= BREAKEVEN_TOLERANCE and 1 <= len(runs) - 1 <= 6
    assert "interpolated" in breakeven_fee(ladder)[1]


def test_a_variant_without_trades_is_not_said_to_lose_money():
    # QA F6: no trades read "loses money even at 0.00% fees" and "loses at no fee".
    from sleeve_fund.research.study import LadderRung, breakeven_fee
    from sleeve_fund.research.tearsheet import _breakeven_cell

    idle = [LadderRung(fee=f, total_return=0.0, sharpe=float("nan"), round_trips=0, fees_paid=0.0) for f in (0.0, 0.001)]
    fee, words = breakeven_fee(idle)
    assert fee is None and words.startswith("made no trades")
    assert _breakeven_cell({"breakeven_fee": float("nan"), "breakeven": words}) == "no trades"


def test_the_trade_bar_counts_out_of_sample_trades_99_fails_100_passes(tmp_path, instrument):
    """QA F8: the bar was only tested as a constant. Through g1_checks, 99 out-of-sample trades fail and 100
    pass, whatever the in-sample count."""
    import dataclasses

    from sleeve_fund.research.tearsheet import g1_checks

    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, synthetic_ohlcv(days=1500, seed=3), instrument, dataset="syn", ledger=ledger,
                  synthetic=True, holdout_days=100, train_days=730, test_days=365)
    in_sample = len(r.round_trips)

    def row(total: int):
        per = [total // len(r.folds)] * len(r.folds)
        per[0] += total - sum(per)
        folds = [dataclasses.replace(f, test_trades=n) for f, n in zip(r.folds, per)]
        checks = g1_checks(dataclasses.replace(r, folds=folds), ledger)
        return next((v, ev) for name, v, ev in checks if name == "Enough out-of-sample trades to judge")

    verdict, words = row(99)
    assert verdict == "FAIL" and words.startswith("99 closed") and f"{in_sample} over the full research period" in words
    assert row(100)[0] == "PASS"


def test_the_sensitivity_table_shows_grid_values_as_set():
    """Round 11 minor: int() of each grid value crashed the tear sheet on a word and showed 0.005 as 0."""
    from sleeve_fund.research.tearsheet import _param

    assert [_param(v) for v in (20, 20.0, 0.005, 1.5, "ema")] == ["20", "20", "0.005", "1.5", "ema"]


def test_a_fold_where_no_setting_scores_fails_with_a_reason_not_a_crash(tmp_path, instrument, monkeypatch):
    """Code Reviewer on #156: when every training Sharpe in a fold is NaN, nothing is chosen. The study used to
    raise; now the fold sits flat, the nearby-settings check fails it with the reason, and the holdout stays shut."""
    from sleeve_fund.research import study
    from sleeve_fund.research.tearsheet import NEARBY_CHECK, g1_checks

    real = study.summary
    monkeypatch.setattr(study, "summary", lambda r: {**real(r), "sharpe": float("nan")})
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, synthetic_ohlcv(days=1900, seed=3), instrument, dataset="syn", ledger=ledger, synthetic=True,
                  holdout_days=200, train_days=730, test_days=365, use_holdout=True)
    assert r.folds and all(f.unscored and f.chosen == {} for f in r.folds)
    assert len(r.oos_returns) and (r.oos_returns == 0).all()  # sat flat through every test window
    verdict, words = next(c[1:] for c in g1_checks(r, ledger) if c[0] == NEARBY_CHECK)
    assert verdict == "FAIL" and "no setting scored a Sharpe on this fold's training stretch" in words
    assert r.holdout is None and "no setting scored" in r.holdout_withheld
    assert render(r, ledger)  # the sheet renders
