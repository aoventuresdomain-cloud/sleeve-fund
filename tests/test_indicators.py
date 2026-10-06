"""Review round 4, NEW-1: the engine's simple moving average (and every indicator built on it) aborts
the whole process once its period passes 1,024. Strategies use our own averages, which match the
engine's below that limit and have none above it."""

import ast
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from nautilus_trader.indicators import AverageTrueRange, SimpleMovingAverage

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.store import Store
from sleeve_fund.strategies.indicators import AtrSma, Rsi, Sma
from sleeve_fund.venues import KRAKEN

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}
STRATEGIES = Path(__file__).parent.parent / "sleeve_fund" / "strategies"
# Engine indicators a strategy may use: checked below to survive a period far past 1,024.
# Not RelativeStrengthIndex: it smooths exponentially, not as the standard RSI the chart draws (strategies.indicators.Rsi).
SAFE_ENGINE_INDICATORS = {"ExponentialMovingAverage"}


def _bars(n, seed=1):
    random.seed(seed)
    px = 100.0
    for _ in range(n):
        o = px
        px *= 1 + random.gauss(0, 0.01)
        yield max(o, px) * 1.002, min(o, px) * 0.998, px


@pytest.mark.parametrize("period", [2, 14, 200, 1024])
def test_our_averages_match_the_engines_where_it_works(period):
    ours, theirs, our_atr, their_atr = Sma(period), SimpleMovingAverage(period), AtrSma(period), AverageTrueRange(period)
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
    sma, atr = Sma(5000), AtrSma(5000)
    for i, c in enumerate(closes):
        sma.update_raw(c)
        atr.update_raw(c * 1.01, c * 0.99, c)
        assert sma.initialized == (i >= 4999)
    assert sma.value == pytest.approx(sum(closes[-5000:]) / 5000, rel=1e-12) and atr.value > 0


def test_strategies_only_use_engine_indicators_that_survive_long_periods():
    for path in STRATEGIES.rglob("*.py"):  # the indicators package too
        for node in ast.walk(ast.parse(path.read_text())):
            # The module itself, however imported, would let any indicator in unchecked.
            whole = (isinstance(node, ast.Import) and any(a.name.startswith("nautilus_trader.indicators")
                                                          for a in node.names)) or (
                isinstance(node, ast.ImportFrom) and node.module == "nautilus_trader"
                and any(a.name == "indicators" for a in node.names))
            assert not whole, f"{path.name} imports the engine's indicators module; import the names it needs"
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


# --- RSI: the standard (Wilder) RSI in the strategies and on the chart (review round 11, M11-1) ---------

def _reference(closes, n):
    """Wilder's RSI written out from its definition: the mean of the first n gains and losses, then
    Wilder's smoothing."""
    d = np.diff(closes)
    gains, losses = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = [None] * len(closes)
    ag, al = gains[:n].mean(), losses[:n].mean()
    for i in range(n, len(d) + 1):
        if i > n:
            ag = (ag * (n - 1) + gains[i - 1]) / n
            al = (al * (n - 1) + losses[i - 1]) / n
        out[i] = 50.0 if ag == al == 0 else 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def _closes(seed=3, n=300):
    return list(100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n))))


def test_wilder_rsi_matches_its_definition():
    closes = _closes()
    rsi, got = Rsi(14), []
    for c in closes:
        rsi.update_raw(c)
        got.append(rsi.value if rsi.initialized else None)
    assert got[:14] == [None] * 14 and got[14] is not None  # 14 changes need 15 closes
    assert got == pytest.approx(_reference(closes, 14), abs=1e-9)


def test_textbook_values():
    # The 14 changes alternate +1 and -1 (7 each): RSI 50; a rise after it lifts it, a fall drops it.
    rsi = Rsi(14)
    for c in [100 + (i % 2) for i in range(15)]:
        rsi.update_raw(c)
    assert rsi.value == pytest.approx(50.0)
    flat = Rsi(5)
    for _ in range(10):
        flat.update_raw(100.0)
    assert flat.initialized and flat.value == 50.0  # a flat line is neutral, not overbought


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_chart_draws_the_same_rsi():
    js = (STRATEGIES.parent / "dashboard" / "static" / "console.js").read_text()
    parts = [re.search(rf"^  const {name} = .*?^  }};$", js, re.S | re.M).group(0) for name in ("wilder", "rsi")]
    nulls = re.search(r"^  const nulls = .*;$", js, re.M).group(0)
    closes = _closes(seed=5) + [100.0] * 30  # ends flat, so the neutral case is drawn too
    script = "\n".join([nulls, *parts, f"console.log(JSON.stringify(rsi({json.dumps(closes)}, 14)));"])
    drawn = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert drawn == pytest.approx(_reference(closes, 14), abs=1e-9)


def test_rsi_bands_trades_the_standard_rsi(prices, instrument):
    """The reasons rsi_bands journals quote the RSI the chart shows at that bar."""
    from sleeve_fund.research.runner import run_backtest
    from test_backtest import _path

    closes = [round(c, 1) for c in _closes(seed=9, n=200)]  # on the instrument's tick, as the venue prices them
    res = run_backtest("rsi_bands", _path(prices, closes), instrument, {"rsi_period": 5}, half_spread=0)
    ref = _reference(closes, 5)
    assert res.decisions
    for d in res.decisions.values():
        if d["intent"] != "entry":
            continue
        quoted = float(re.search(r"RSI ([\d.]+)", d["reason"]).group(1))
        assert any(r is not None and abs(r - quoted) < 0.051 for r in ref), d["reason"]
        assert min(abs(r - d["signal"]["rsi"]) for r in ref if r is not None) < 1e-6
