"""Advisor ruling 7 Oct (via the HoE), #178: a rule-builder breakout entry pays the half spread and its breakout
slippage added together, each charged exactly once; the commission the report shows is the venue's fee alone."""

from decimal import Decimal

import pandas as pd
import pytest
from nautilus_trader.model import OrderSide, Price, Quantity

from sleeve_fund.instruments import FeeSchedule, ScheduleFeeModel
from sleeve_fund.research.runner import _spread_into_prices
from sleeve_fund.venues import venue

TAKER = Decimal("0.0005")


def _order(coid):
    return type("O", (), {"client_order_id": coid, "is_post_only": False, "side": OrderSide.BUY})()


def _fill(fm, coid, inst):
    """A 1-unit buy the venue fills at 100 (bars carry no quotes), as the report then shows it."""
    charged = fm.get_commission(_order(coid), Quantity(1, 8), Price(100, 2), inst)
    report = pd.DataFrame({"filled_qty": ["1"], "avg_px": ["100"], "side": ["OrderSide.BUY"],
                           "commissions": [[str(charged)]]}, index=[coid])
    out = _spread_into_prices(report, fm.spread_paid, fm.fee_paid)
    return charged.as_double(), float(out.at[coid, "avg_px"]), float(str(out.at[coid, "commissions"][0]).split()[0])


@pytest.mark.parametrize(("breakout", "booked"), [(Decimal("0.0005"), 100.06), (None, 100.01)])
def test_a_breakout_entry_books_spread_plus_slippage_once_and_the_commission_is_the_fee_alone(breakout, booked):
    """Long entry at 100, half spread 1 bp, breakout slippage 5 bp: booked at 100.06, the fee on 100.06 and
    nothing else. A market order with no breakout slippage pays the half spread alone (100.01)."""
    inst = venue("KRAKEN").instrument("BTC", "USD")
    fm = ScheduleFeeModel(FeeSchedule(Decimal("0.0002"), TAKER), half_spread=0.0001)
    if breakout is not None:
        fm.slippage["E"] = breakout
    charged, px, commission = _fill(fm, "E", inst)
    fee = booked * float(TAKER)
    assert [p for p, _ in fm.booked_fills["E"]] == [pytest.approx(booked)]
    assert px == pytest.approx(booked) and commission == pytest.approx(fee, abs=0.005)
    # Cash: the venue's 100 plus the one charge is exactly 100.06 (or 100.01) plus its fee; nothing is paid twice.
    assert 100 + charged == pytest.approx(booked + fee, abs=0.005)
