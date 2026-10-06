"""P2-1 central sizing, opt-in (Head of Engineering and Independent Quant Advisor, 6 Oct 2026): a strategy made from
now on, or any with sizing="central", is sized by sizing.size_entry; every existing model keeps the sizing it was
tested with, to the byte."""

from decimal import Decimal

import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.portfolio.sizing import ATR_STOP_MULTIPLE, DEFAULT_STOP_SLIPPAGE, loss_at_stop
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


# Recorded on main (5df446a, re-recorded after main moved the fee rounding's cent into the price); the branch matches
# it exactly. The same 21 runs over every model, with and without a risk profile and with each kind of stop, were
# checked identical to the byte; these three are pinned here.
LEGACY = [
    ({"stop_loss": 0.05, "risk_per_trade": 0.02}, None, 9800.0,
     [("BUY", "0.20752534", "14483.246220", "2018-07-20"), ("SELL", "0.20752534", "13745.327462", "2018-08-09")]),
    ({"market": "perp", "allow_short": True, "stop_loss": 0.04}, "balanced", 9579.70196,
     [("BUY", "0.45592799", "14477.417597", "2018-07-20"), ("SELL", "0.45592799", "14175.234336", "2018-07-25"),
      ("BUY", "0.43499491", "14942.424270", "2018-07-26"), ("SELL", "0.43499491", "14341.864696", "2018-07-27")]),
    ({}, "balanced", 15440.0,
     [("BUY", "0.22796399", "14483.207985", "2018-07-20"), ("SELL", "0.22796399", "33492.732967", "2019-07-19"),
      ("BUY", "0.14467298", "32511.413531", "2019-07-20"), ("SELL", "0.14467298", "41355.592699", "2020-02-11")]),
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
    # 1% of equity at risk to that stop, the stop filling past its price by the venue's default stop slippage: the
    # larger of half the spread and 0.05% (Advisor, 16:45).
    assert i.risk_per_trade == base.DEFAULT_RISK_PER_TRADE and i.stop_slippage is None and i.half_spread > 0
    slip = max(i.half_spread, DEFAULT_STOP_SLIPPAGE)
    per_unit = loss_at_stop(i.stop_frac, i.leg_cost, i.side) + (1 - i.side * i.stop_frac) * slip
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
    both = size_entry(SizingInputs(**common, overlay="vol_target", vol_target=0.002, instrument_vol=0.04,
                                   vol_floor=0.01))
    assert both.qty <= by_stop.qty and both.qty < by_stop.qty


def test_central_sizing_reaches_new_strategies_only():
    from sleeve_fund.dashboard import app

    made = app._form_params({"market": "spot"}, "trend_filter")
    assert made["sizing"] == "central"
    assert "sizing" not in app._form_params({}, "buy_and_hold")  # the benchmark sizes its own way
    assert base.central_sizing({"sizing": "central"}) and not base.central_sizing({})
    assert not base.central_sizing({"sizing": "central", "rebalance_band": 0.05})


@pytest.mark.parametrize("stop", [{"stop_loss": 0.0}, {"stop_atr": 0.0}])
def test_a_declared_stop_of_zero_is_refused_up_front(bars, btc, stop):
    """Advisor, 16:45: a declared stop of 0 is refused at validation, never sized or replaced by the default."""
    with pytest.raises(ValueError):
        run_backtest("trend_filter", bars, btc, {"sizing": "central", **stop})


def test_steps_between_fractions_send_only_the_difference_and_skip_below_the_minimum():
    from sleeve_fund.portfolio.sizing import step_order

    held, sent = Decimal(0), []
    for f in (1 / 3, 2 / 3, 1.0, 0.0):
        o = step_order(Decimal("0.900"), held, f, Decimal("0.001"), Decimal("0.001"))
        sent.append(o.qty)
        held += o.qty
    assert sent == [Decimal("0.300"), Decimal("0.300"), Decimal("0.300"), Decimal("-0.900")]
    small = step_order(Decimal("0.900"), Decimal(0), 1 / 3, Decimal("0.001"), Decimal("0.350"))
    assert small.qty == 0 and "below the venue's smallest order" in small.skipped
    # Closing to 0 always goes in full, however small.
    assert step_order(Decimal("0.900"), Decimal("0.010"), 0.0, Decimal("0.001"), Decimal("0.350")).qty == Decimal("-0.010")


@pytest.mark.parametrize("after, due", [("2026-10-06T16:00", "2026-11-02"), ("2026-05-31T23:59", "2026-06-01"),
                                        ("2026-09-30T00:00", "2026-10-05"), ("2026-12-08T00:00", "2027-01-04"),
                                        ("2026-11-02T00:00", "2026-12-07")])
def test_the_monthly_rebalance_is_the_first_monday_at_midnight_utc(after, due):
    from datetime import datetime, timezone

    from sleeve_fund.portfolio.allocation import next_rebalance

    t = datetime.fromisoformat(after).replace(tzinfo=timezone.utc)
    assert next_rebalance(t) == datetime.fromisoformat(due).replace(tzinfo=timezone.utc)
