"""Review B5: G1 passes only when the out-of-sample Sharpe beats the benchmark's by more than luck
across every variant tried, and a pass counts only for the instrument and bar length tested."""

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.dashboard import pipeline
from sleeve_fund.research.metrics import sharpe_beats_probability
from sleeve_fund.store import Sleeve


def _sheet(path, strategy, verdict, instrument="BTC/USD", minutes=1440, dataset="kraken-btcusd-store"):
    tested = f"Tested on `{instrument}` at {minutes}-minute bars\n\n" if instrument else ""
    path.write_text(f"# Tear sheet: {strategy}\n\n{tested}Dataset `{dataset}` · research period x\n\n"
                    f"| Check | Result | Evidence |\n| --- | --- | --- |\n"
                    f"| G1 test: out-of-sample Sharpe clearly beats benchmark after fees | {verdict} | Sharpe 1.2 vs 0.8 |\n")


def test_a_pass_counts_only_for_the_instrument_and_bars_tested(tmp_path):
    _sheet(tmp_path / "trend_filter_btc.md", "trend_filter", "PASS")
    assert pipeline.g1_for(tmp_path, "trend_filter", "btc/usd", 1440) == "PASS"
    assert pipeline.g1_for(tmp_path, "trend_filter", "ETH/USD", 1440) is None
    assert pipeline.g1_for(tmp_path, "trend_filter", "BTC/USD", 60) is None


def test_the_badge_follows_the_sheet_not_its_file_name(tmp_path):
    # The file name starts with trend_filter, but the sheet is rsi_pullback's.
    _sheet(tmp_path / "trend_filter_rsi_variant.md", "rsi_pullback", "PASS")
    rows = {r["name"]: r for r in pipeline.strategies(tmp_path, [])}
    assert rows["trend_filter"]["g1"] is None and rows["rsi_pullback"]["passed_where"] == ["BTC/USD daily"]


def test_synthetic_and_unplaced_passes_badge_nothing(tmp_path):
    _sheet(tmp_path / "trend_filter_synthetic.md", "trend_filter", "PASS", dataset="synthetic")
    _sheet(tmp_path / "trend_filter_old.md", "trend_filter", "PASS", instrument=None)
    row = next(r for r in pipeline.strategies(tmp_path, []) if r["name"] == "trend_filter")
    assert row["g1"] is None and row["passed_on"] == []


def test_a_strategy_on_another_instrument_is_an_observation(tmp_path):
    _sheet(tmp_path / "trend_filter_btc.md", "trend_filter", "PASS")

    def sleeve(instrument, bar_spec):
        return Sleeve(id=1, name="x", strategy="trend_filter", instrument=instrument, bar_spec=bar_spec, params={},
                      starting_balance=1000, risk_profile="balanced", warmup_bars=0, desired_state="running",
                      status="running", status_reason="", paused_until=None, heartbeat_at=None,
                      created_at=None, updated_at=None)

    row = lambda s: next(r for r in pipeline.strategies(tmp_path, [s]) if r["name"] == "trend_filter")  # noqa: E731
    assert not row(sleeve("BTC/USD", "1-DAY-LAST-EXTERNAL"))["observation"]
    assert row(sleeve("ETH/USD", "1-DAY-LAST-EXTERNAL"))["observation"]
    assert row(sleeve("BTC/USD", "1-HOUR-LAST-INTERNAL"))["observation"]


def test_a_higher_sharpe_by_luck_does_not_clear_the_bar():
    rng = np.random.default_rng(3)
    n = 4 * 365  # four years out of sample
    idx = pd.date_range("2021-01-01", periods=n)
    bench = pd.Series(rng.normal(0.0005, 0.03, n), idx)
    same = pd.Series(rng.normal(0.0005, 0.03, n), idx)  # no edge at all
    strong = pd.Series(rng.normal(0.006, 0.02, n), idx)  # a Sharpe near 5 against about 0.3
    p_same, _ = sharpe_beats_probability(same, bench, 20)
    p_strong, hurdle = sharpe_beats_probability(strong, bench, 20)
    assert p_same < 0.5 and p_strong >= 0.95 and hurdle > 0
    # More variants tried, higher bar; and the answer is the same every time.
    assert sharpe_beats_probability(strong, bench, 200)[1] > hurdle
    assert sharpe_beats_probability(strong, bench, 20) == (p_strong, hurdle)
    assert np.isnan(sharpe_beats_probability(strong[:30], bench[:30], 20)[0])  # too short to judge


def test_the_backtest_page_warns_unless_g1_passed_here(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.store import Store
    from sleeve_fund.venues import KRAKEN

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=200, seed=3))
    preview._history.clear()
    _sheet(tmp_path / "buy_and_hold_eth.md", "buy_and_hold", "PASS", instrument="ETH/USD")
    c = TestClient(app_mod.create_app(Store(f"sqlite:///{tmp_path}/t.db")))
    eth = c.get("/backtest?run=1&instrument=ETH/USD&strategy=buy_and_hold", auth=("pm", "test-pw")).text
    sol = c.get("/backtest?run=1&instrument=SOL/USD&strategy=buy_and_hold", auth=("pm", "test-pw")).text
    assert "Not G1 evidence" not in eth and "G1 comes from the research tests" in eth
    assert "Not G1 evidence for SOL/USD" in sol, sol[sol.find("banner"):][:300]
    form = c.get("/sleeves/new?strategy=buy_and_hold", auth=("pm", "test-pw")).text
    assert 'data-g1="ETH/USD@1440"' in form


@pytest.mark.parametrize("bars, minutes", [(pd.date_range("2024-01-01", periods=50, freq="1D"), 1440),
                                           (pd.date_range("2024-01-01", periods=50, freq="1h"), 60)])
def test_the_bar_length_tested_is_read_from_the_prices(bars, minutes):
    from sleeve_fund.research.study import bar_minutes_of

    assert bar_minutes_of(pd.DataFrame({"close": 1.0}, index=bars)) == minutes
