"""Review round 4, NEW-1: the engine's simple moving average (and every indicator built on it) aborts
the whole process once its period passes 1,024. Strategies use our own averages, which match the
engine's below that limit and have none above it."""

import ast
import random
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nautilus_trader.indicators import AverageTrueRange, SimpleMovingAverage

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.store import Store
from sleeve_fund.strategies.indicators import Atr, Sma
from sleeve_fund.venues import KRAKEN

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}
STRATEGIES = Path(__file__).parent.parent / "sleeve_fund" / "strategies"
# Engine indicators a strategy may use: checked below to survive a period far past 1,024.
SAFE_ENGINE_INDICATORS = {"ExponentialMovingAverage", "RelativeStrengthIndex"}


def _bars(n, seed=1):
    random.seed(seed)
    px = 100.0
    for _ in range(n):
        o = px
        px *= 1 + random.gauss(0, 0.01)
        yield max(o, px) * 1.002, min(o, px) * 0.998, px


@pytest.mark.parametrize("period", [2, 14, 200, 1024])
def test_our_averages_match_the_engines_where_it_works(period):
    ours, theirs, our_atr, their_atr = Sma(period), SimpleMovingAverage(period), Atr(period), AverageTrueRange(period)
    for h, low, c in _bars(3000):
        ours.update_raw(c)
        theirs.update_raw(c)
        our_atr.update_raw(h, low, c)
        their_atr.update_raw(h, low, c)
        assert ours.initialized == theirs.initialized and our_atr.initialized == their_atr.initialized
        assert ours.value == pytest.approx(theirs.value, rel=1e-12)
        assert our_atr.value == pytest.approx(their_atr.value, rel=1e-12)


def test_our_averages_have_no_period_limit():
    closes = [c for _, _, c in _bars(6000)]
    sma, atr = Sma(5000), Atr(5000)
    for i, c in enumerate(closes):
        sma.update_raw(c)
        atr.update_raw(c * 1.01, c * 0.99, c)
        assert sma.initialized == (i >= 4999)
    assert sma.value == pytest.approx(sum(closes[-5000:]) / 5000, rel=1e-12) and atr.value > 0


def test_strategies_only_use_engine_indicators_that_survive_long_periods():
    for path in STRATEGIES.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("nautilus_trader.indicators"):
                names = {a.name for a in node.names}
                assert names <= SAFE_ENGINE_INDICATORS, (
                    f"{path.name} imports {names - SAFE_ENGINE_INDICATORS} from the engine; most of its indicators "
                    "abort the process past 1,024 bars. Use sleeve_fund.strategies.indicators or check it here.")


def test_the_engine_indicators_allowed_survive_long_periods():
    # In a child process: if the engine panics it takes only the child down.
    code = ("import nautilus_trader.indicators as m\n"
            f"for name in {sorted(SAFE_ENGINE_INDICATORS)!r}:\n"
            "    i = getattr(m, name)(5000)\n"
            "    for k in range(6000): i.update_raw(100.0 + k % 7)\n"
            "    assert i.initialized, name\n"
            "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-500:]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.dashboard import preview

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=2600, seed=3, vol=0.03))
    preview._history.clear()
    return TestClient(app_mod.create_app(Store(f"sqlite:///{tmp_path}/t.db")))


@pytest.mark.parametrize("params", [
    {"strategy": "trend_filter", "p_trend_filter__fast": "50", "p_trend_filter__slow": "2000"},
    {"strategy": "rsi_pullback", "p_rsi_pullback__vol_period": "1500", "p_rsi_pullback__atr_period": "1500",
     "p_rsi_pullback__ema_period": "1500"},
])
def test_a_long_average_backtests_instead_of_killing_the_dashboard(client, params):
    r = client.get("/backtest", params={"run": "1", "instrument": "BTC/USD", "risk_profile": "balanced",
                                        "period": "all", **params}, auth=AUTH)
    assert r.status_code == 200 and "Couldn't run it" not in r.text and "Every trade" in r.text


def test_a_minute_strategy_with_a_2400_bar_average_can_be_created(client):
    form = {"name": "slow-tf", "strategy": "trend_filter", "instrument": "BTC/USD",
            "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": "1000", "risk_profile": "balanced",
            "reason": "t", "p_trend_filter__fast": "600", "p_trend_filter__slow": "2400"}
    r = client.post("/sleeves/new", data=form, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.headers["location"] == "/sleeves/slow-tf"
