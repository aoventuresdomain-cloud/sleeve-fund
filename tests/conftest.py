import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.instruments import spot_pair


@pytest.fixture(scope="session")
def instrument():
    return spot_pair("BTC", "USD")


@pytest.fixture(scope="session")
def prices():
    return synthetic_ohlcv(days=900, seed=11)
