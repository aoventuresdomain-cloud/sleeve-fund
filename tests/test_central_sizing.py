"""P2-1 central sizing, opt-in (Head of Engineering and Independent Quant Advisor, 6 Oct 2026): a strategy made from
now on, or any with sizing="central", is sized by sizing.size_entry; every existing model keeps the sizing it was
tested with, to the byte."""

from decimal import Decimal

import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.portfolio.sizing import ATR_STOP_MULTIPLE, loss_at_stop
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies import base
from sleeve_fund.venues import venue


@pytest.fixture(scope="module")
def bars():
    return synthetic_ohlcv(days=900, seed=3, vol=0.03)


@pytest.fixture(scope="module")
def btc():
    return venue("KRAKEN").instrument("BTC", "USD")


def _fills(r):
    return [(str(f.side), str(f.quantity), f"{float(f.avg_px):.6f}", f.ts_last.strftime("%Y-%m-%d"))
            for f in r.fills.itertuples()]


# Recorded on main before P2-1 (0ceac6f). The same 21 runs over every model, with and without a risk profile and
# with each kind of stop, were checked identical to the byte; these three are pinned here.
LEGACY = [
    ({"stop_loss": 0.05, "risk_per_trade": 0.02}, None, 9800.0,
     [("BUY", "0.20752534", "14483.207985", "2018-07-20"), ("SELL", "0.20752534", "13745.293915", "2018-08-09")]),
    ({"market": "perp", "allow_short": True, "stop_loss": 0.04}, "balanced", 9579.701757,
     [("BUY", "0.45592799", "14477.417597", "2018-07-20"), ("SELL", "0.45592799", "14175.242334", "2018-07-25"),
      ("BUY", "0.43499491", "14942.424093", "2018-07-26"), ("SELL", "0.43499491", "14341.855671", "2018-07-27")]),
    ({}, "balanced", 15440.0,
     [("BUY", "0.22796399", "14483.207985", "2018-07-20"), ("SELL", "0.22796399", "33492.735255", "2019-07-19"),
      ("BUY", "0.14467298", "32511.417585", "2019-07-20"), ("SELL", "0.14467298", "41355.571870", "2020-02-11")]),
]


@pytest.mark.parametrize("params, profile, equity, fills", LEGACY)
def test_a_legacy_strategy_trades_exactly_as_before(bars, btc, params, profile, equity, fills, monkeypatch):
    called = []
    monkeypatch.setattr(base, "size_entry", lambda i: called.append(i))
    for p in (params, {**params, "sizing": "legacy"}):
        r = run_backtest("trend_filter", bars, btc, dict(p), risk_profile=profile)
        assert _fills(r) == fills and round(float(r.equity.iloc[-1]), 6) == equity
    assert not called  # central sizing never ran


def test_an_unknown_sizing_is_refused(bars, btc):
    with pytest.raises(ValueError, match="sizing is one of legacy, central"):
        run_backtest("trend_filter", bars, btc, {"sizing": "kelly"})


def test_a_central_strategy_with_no_stop_is_sized_to_a_placed_wilder_fallback_stop(bars, btc, monkeypatch):
    calls = []
    real = base.size_entry
    monkeypatch.setattr(base, "size_entry", lambda i: calls.append((i, real(i))) or calls[-1][1])
    r = run_backtest("trend_filter", bars, btc, {"market": "perp", "allow_short": True, "sizing": "central"},
                     risk_profile="balanced")
    assert calls
    i, s = calls[0]
    entry = next(d for d in r.decisions.values() if d["intent"] == "entry")["signal"]
    # The fallback: 2.5 x Wilder's ATR(14), declared as the default, and placed as a real stop order.
    assert entry["stop_cfg"] == {"stop_atr": ATR_STOP_MULTIPLE, "atr_bars": 14}
    assert "Wilder average true range" in entry["stop_basis"] and "none declared" in entry["stop_fallback"]
    assert i.stop_frac == pytest.approx(entry["stop_frac"], abs=1e-6)
    assert "STOP_MARKET" in set(r.fills["type"].astype(str))
    # 1% of equity at risk to that stop, the stop filling one half spread past its price (the stop-slippage term).
    assert i.risk_per_trade == base.DEFAULT_RISK_PER_TRADE and i.stop_slippage is None and i.half_spread > 0
    per_unit = loss_at_stop(i.stop_frac, i.leg_cost, i.side) + (1 - i.side * i.stop_frac) * i.half_spread
    assert s.sized_by == "risk per trade" and s.risk_budget == pytest.approx(0.01 * i.allocated_equity)
    assert s.risk_amount == pytest.approx(float(s.qty) * i.price * per_unit) and s.risk_amount <= s.risk_budget
    # Rounded down to the venue's step, and the order is that quantity.
    assert s.qty == s.qty.quantize(i.lot) and Decimal(str(r.fills.iloc[0]["quantity"])) == s.qty
    assert float(s.qty) * i.price * per_unit > s.risk_budget - float(i.lot) * i.price * per_unit
    # On a perpetual the PM's rule is in force: the stop within half the distance to liquidation.
    assert i.perp and i.stop_to_liquidation == 0.5 and i.maintenance_margin > 0
    assert any(k.startswith("liquidation rule") for k in s.limits)


def test_central_sizing_takes_the_smaller_of_the_stop_and_volatility_sizes():
    """The wiring passes the overlay through; which size wins is size_entry's (test_sizing covers the cases)."""
    from sleeve_fund.portfolio.sizing import SizingInputs, size_entry

    common = dict(allocated_equity=10_000, price=100.0, side=1, leg_cost=0.001, half_spread=0.0005,
                  risk_per_trade=0.01, position_cap_pct=1.0, lot=Decimal("0.001"), min_qty=Decimal("0.001"),
                  stop_frac=0.05)
    by_stop = size_entry(SizingInputs(**common))
    both = size_entry(SizingInputs(**common, overlay="vol_target", vol_target=0.002, instrument_vol=0.04))
    assert both.qty <= by_stop.qty and both.qty < by_stop.qty


def test_central_sizing_reaches_new_strategies_only():
    from sleeve_fund.dashboard import app

    made = app._form_params({"market": "spot"}, "trend_filter")
    assert made["sizing"] == "central"
    assert "sizing" not in app._form_params({}, "buy_and_hold")  # the benchmark sizes its own way
    assert base.central_sizing({"sizing": "central"}) and not base.central_sizing({})
    assert not base.central_sizing({"sizing": "central", "rebalance_band": 0.05})
