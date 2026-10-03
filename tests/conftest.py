import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.venues import venue


@pytest.fixture(scope="session")
def instrument():
    return venue("KRAKEN").instrument("BTC", "USD")


@pytest.fixture(scope="session")
def prices():
    return synthetic_ohlcv(days=900, seed=11)


@pytest.fixture(autouse=True)
def _no_swallowed_strategy_errors(capfd):
    """The engine catches exceptions raised in strategy callbacks and only logs them, so a broken
    on_bar or timer can pass a test that checks something else. Fail any test that logs one."""
    yield
    out, err = capfd.readouterr()
    for line in (out + err).splitlines():
        if "Python " in line and ("failed" in line or "raised exception" in line):
            pytest.fail(f"a strategy callback raised inside the engine: {line}")
