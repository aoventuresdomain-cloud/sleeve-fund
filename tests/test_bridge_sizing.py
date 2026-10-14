"""BRIDGE-3 (Advisor 14 Oct ~21:55 UK; bridge Drop A, orders lane): a Strategy sizes its own entries, from the fund's
equity instead of its own, and never from a gain not yet taken: base = fund equity + min(0, unrealised). The
leverage cap and free cash stay on its own margin (order validity). Unset, the Strategy is the whole fund, so a
single Strategy sizes exactly as before (the golden pin, GOLD-STOPMARK byte-identical)."""

from types import SimpleNamespace

import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.base import LongFlatStrategy
from tests.test_backtest import _path

RISK = {"stop_loss": 0.05, "risk_per_trade": 0.01}


def _first_notional(prices, instrument, fund=None, monkeypatch=None):
    if fund is not None:
        real = LongFlatStrategy.on_start

        def on_start(self):
            self.fund_equity = lambda: fund
            real(self)

        monkeypatch.setattr(LongFlatStrategy, "on_start", on_start)
    res = run_backtest("buy_and_hold", _path(prices, [10_000.0] * 30), instrument, RISK, starting_capital=10_000)
    return float(res.fills["filled_qty"].iloc[0]) * float(res.fills["avg_px"].iloc[0])


def test_an_entry_is_sized_from_the_funds_equity(prices, instrument, monkeypatch):
    own = _first_notional(prices, instrument)
    doubled = _first_notional(prices, instrument, fund=20_000.0, monkeypatch=monkeypatch)
    assert own == pytest.approx(1_500, rel=0.05)
    assert doubled == pytest.approx(2 * own, rel=0.01)  # risk x fund equity / loss at the stop


def test_a_fund_equal_to_the_strategys_balance_sizes_as_before(prices, instrument, monkeypatch):
    own = _first_notional(prices, instrument)
    assert _first_notional(prices, instrument, fund=10_000.0, monkeypatch=monkeypatch) == pytest.approx(own, abs=1e-6)


@pytest.mark.parametrize("price,base", [(110.0, 1_000.0), (90.0, 990.0), (100.0, 1_000.0)])
def test_a_gain_not_yet_taken_never_sizes_up_and_a_loss_sizes_down(price, base):
    """Equity 1,000 marked with 1 unit held from 100: a 10 gain is left out, a 10 loss stays in (realised + min(0, unrealised))."""
    s = SimpleNamespace(fund_equity=None, _fund_held=lambda: False, _entry_px=100.0, _entry_qty=1.0, _entry_side=1, _price=lambda: price)
    equity = 1_000.0 + (price - 100.0)
    assert LongFlatStrategy._sizing_base(s, equity) == pytest.approx(base)


def test_a_short_gain_is_left_out_too():
    s = SimpleNamespace(fund_equity=None, _fund_held=lambda: False, _entry_px=100.0, _entry_qty=2.0, _entry_side=-1, _price=lambda: 95.0)
    assert LongFlatStrategy._sizing_base(s, 1_010.0) == pytest.approx(1_000.0)


def test_the_funds_figure_is_taken_as_the_fund_gives_it():
    """Paper's book (PE2) gives the fund's equity with its own min(0, unrealised) already applied."""
    s = SimpleNamespace(fund_equity=lambda: 100_000.0, _entry_px=None, _entry_qty=0.0, _entry_side=0,
                        _price=lambda: 1.0)
    assert LongFlatStrategy._sizing_base(s, 10_000.0) == 100_000.0
