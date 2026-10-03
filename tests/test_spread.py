"""Backtests charge half the bid-ask spread on orders that take liquidity, as paper's quotes would."""

import pytest

from sleeve_fund import spreads
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store
from test_backtest import _path
from test_maker import _daily, _minutes


def test_a_market_buy_pays_half_the_spread_in_its_price(prices, instrument):
    df = _path(prices, [100.0] * 20)
    flat = run_backtest("buy_and_hold", df, instrument, half_spread=0)
    wide = run_backtest("buy_and_hold", df, instrument, half_spread=0.001)
    assert float(flat.fills["avg_px"].iloc[0]) == pytest.approx(100.0)
    assert float(wide.fills["avg_px"].iloc[0]) == pytest.approx(100.1)
    assert wide.fees_paid == pytest.approx(flat.fees_paid, abs=0.01)  # the venue's fee alone
    qty = float(wide.fills["filled_qty"].iloc[0])
    assert wide.spread_paid == pytest.approx(qty * 100 * 0.001, abs=0.01)
    assert wide.equity.iloc[-1] == pytest.approx(flat.equity.iloc[-1] - wide.spread_paid, abs=0.02)


def test_a_sell_takes_the_bid(prices, instrument):
    closes = [100.0] * 10 + [100.0 * 0.99**i for i in range(1, 30)]
    res = run_backtest("buy_and_hold", _path(prices, closes), instrument, {"stop_loss": 0.05}, half_spread=0.001)
    assert float(res.fills["avg_px"].iloc[1]) == pytest.approx(95.0 * 0.999, rel=1e-4)


def test_a_maker_fill_pays_no_spread(instrument):
    day = [10_000.0] * 1440
    dip = [10_000.0 - (i % 3) for i in range(1440)]  # trades through one tick under the last price
    m = _minutes(day + dip)
    res = run_backtest("buy_and_hold", _daily(m), instrument, {"maker_wait_minutes": 15}, exec_prices=m,
                       half_spread=0.001)
    fill = res.fills.iloc[0]
    assert fill["liquidity_side"] == "MAKER" and res.spread_paid == 0
    assert float(fill["avg_px"]) == pytest.approx(9_999.99)


def test_without_a_measurement_the_venue_assumption_is_used(prices, instrument):
    res = run_backtest("buy_and_hold", _path(prices, [100.0] * 20), instrument)
    assert res.half_spread == 0.0005
    q = spreads.resolve("KRAKEN", "BTC/USD", Store.in_memory())
    assert q.source == "assumed" and q.half_spread == 0.0005 and "assumed" in q.text


def test_a_measured_spread_wins(instrument):
    store = Store.in_memory()
    store.record_spread("KRAKEN", "BTC/USD", 0.00002, samples=1200)
    q = spreads.resolve("KRAKEN", "BTC/USD", store)
    assert q.source == "measured" and q.half_spread == 0.00002
    assert "0.004% bid-ask spread, the median of 1,200 live quotes" in q.text
    assert spreads.resolve("KRAKEN", "SUI/USD", store).source == "assumed"
    with pytest.raises(ValueError):
        store.record_spread("KRAKEN", "BTC/USD", 0.2, samples=1)
