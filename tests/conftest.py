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


@pytest.fixture
def full_margin(monkeypatch):
    """Every risk profile puts the strategy's whole equity up as margin, so a perpetual sizes at its full
    leverage cap (2x on balanced) and its liquidation price is near enough to test. The default margin
    cap (PM, 5 Oct 2026: 33% on balanced, so 0.66x notional) leaves liquidation out of reach."""
    import dataclasses

    from sleeve_fund import risk

    for name, p in list(risk.PROFILES.items()):
        monkeypatch.setitem(risk.PROFILES, name, dataclasses.replace(p, max_position_pct=1.0))
