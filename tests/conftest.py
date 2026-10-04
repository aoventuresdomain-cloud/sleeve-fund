import os

import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.venues import venue

# Backtests run in-process under test, where the tests' stand-in price feeds apply; the server runs
# each in a process of its own (test_jobs.py covers that).
os.environ.setdefault("BACKTEST_ISOLATE", "0")


@pytest.fixture(scope="session")
def instrument():
    return venue("KRAKEN").instrument("BTC", "USD")


@pytest.fixture(scope="session")
def prices():
    return synthetic_ohlcv(days=900, seed=11)


@pytest.fixture(autouse=True)
def _no_swallowed_strategy_errors(capfd, request):
    """The engine catches exceptions raised in strategy callbacks: on_bar and timers it only logs,
    and order and market data handlers it drops without a word, so the strategy reports those itself
    (LongFlatStrategy._reporting). Either way a broken handler could pass a test that checks
    something else, so fail any test that logs one."""
    yield
    out, err = capfd.readouterr()
    if request.node.get_closest_marker("strategy_errors"):  # raised on purpose, and checked by the test
        return
    for line in (out + err).splitlines():
        if ("Python " in line and ("failed" in line or "raised exception" in line)) or "strategy handler " in line or "sleeve tick failed" in line:
            pytest.fail(f"a strategy callback raised inside the engine: {line}")


@pytest.fixture
def maker_on(monkeypatch):
    """Maker-first orders are switched off by default (strategies.base.maker_orders_enabled); the tests of
    the post-only path switch them on."""
    monkeypatch.setenv("SLEEVE_MAKER_ORDERS", "1")
