import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.venues import venue


@pytest.fixture(scope="session")
def instrument():
    return venue("KRAKEN").instrument("BTC", "USD")


@pytest.fixture(scope="session")
def prices():
    return synthetic_ohlcv(days=900, seed=11)
