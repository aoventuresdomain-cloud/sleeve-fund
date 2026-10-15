"""FEE-LIVE (5) (Advisor, HoE 15 Oct): the loss if stops are hit, T1 per asset, T2 per group and T3 the whole fund,
includes the taker fee on the stop's notional, |qty| x stop x taker, from the same fee source the fills are charged
at. A stopless position counts the fee on its stopless exit too. The rate is required: a caller with no rate is refused
(fail closed), never charged nothing."""
from __future__ import annotations

from decimal import Decimal

import pytest

from sleeve_fund import open_risk
from sleeve_fund.portfolio.holding import Position, holding_for
from sleeve_fund.portfolio.limits import Book, decide
from sleeve_fund.research.portfolio import gate_intent
from sleeve_fund.risk import PortfolioProfile

TAKER = Decimal("0.00055")  # the paper venue's published VIP 0 perp taker rate, 15 Oct 2026


@pytest.mark.parametrize(("qty", "stop", "want"), [
    # long 2 at 100, stop 95: 2 x 5 = 10, plus 2 x 95 x 0.055% = 0.1045
    (Decimal(2), Decimal(95), Decimal("10.1045")),
    # short 2 at 100, stop 105: 2 x 5 = 10, plus 2 x 105 x 0.055% = 0.1155
    (Decimal(-2), Decimal(105), Decimal("10.1155")),
])
def test_the_loss_at_the_stop_includes_the_taker_fee_on_the_stops_notional_exactly(qty, stop, want):
    assert open_risk.position_risk(qty, Decimal(100), stop, taker=TAKER) == want
    assert open_risk.position_risk(float(qty), 100.0, float(stop), taker=float(TAKER)) == pytest.approx(float(want))


def test_a_stopless_position_counts_the_fee_on_its_stopless_exit():
    """10% floor: a long of 1 at 100 exits at 90, 10 + 90 x 0.055% = 10.0495; a short at 110, 10 + 0.0605."""
    assert open_risk.position_risk(Decimal(1), Decimal(100), None, 0.01, taker=TAKER) == Decimal("10.0495")
    assert open_risk.position_risk(Decimal(-1), Decimal(100), None, 0.01, taker=TAKER) == Decimal("10.0605")


def test_the_rate_is_required_and_a_rate_that_cant_be_used_is_refused():
    with pytest.raises(TypeError):
        open_risk.position_risk(Decimal(1), Decimal(100), Decimal(95))  # noqa - no rate: never charged nothing
    for bad in (None, Decimal(-1), Decimal("NaN"), Decimal("0.05")):
        with pytest.raises(ValueError, match="taker fee"):
            open_risk.position_risk(Decimal(1), Decimal(100), Decimal(95), taker=bad)


def test_the_books_holding_carries_the_rate_and_one_with_none_fails_closed():
    held = holding_for(Position("s", "BTC", Decimal(2), Decimal(95), 3.0, taker=TAKER), Decimal(100), None)
    assert held.risk == Decimal("10.1045")
    with pytest.raises(ValueError, match="taker fee"):
        holding_for(Position("s", "BTC", Decimal(2), Decimal(95), 3.0), Decimal(100), None)


def _intent(qty: str):
    return gate_intent({"side": 1, "price": Decimal(60_000), "stop_frac": 0.01, "instrument": "BTC/USDT",
                        "leverage": 3.0, "qty": Decimal(qty), "step": Decimal("0.001"), "min_qty": Decimal("0.001"),
                        "taker": TAKER})


def test_t1_at_its_limit_passes_and_one_lot_more_is_trimmed_by_the_fee():
    """100,000 fund, T1 4% = 4,000 at the stops in BTC. One unit at 60,000 with a 1% stop risks 600 plus
    59,400 x 0.055% = 32.67, so 632.67: 6.322 fits (3,999.74), 6.323 doesn't (4,000.37). Without the fee 6.666 would."""
    prof = PortfolioProfile(gross=10, net_instrument=10, margin=10, open_risk=1, asset_open_risk=0.04)
    book = Book(Decimal(100_000))
    assert _intent("6.322").risk_per_unit == Decimal("632.67")
    at = decide(book, _intent("6.322"), prof)
    assert at.approved_qty == Decimal("6.322") and at.limit_hit is None
    over = decide(book, _intent("6.323"), prof)
    assert over.approved_qty == Decimal("6.322") and over.limit_hit == "asset_open_risk"


def test_an_intent_with_no_rate_fails_closed():
    d = {"side": 1, "price": Decimal(60_000), "stop_frac": 0.01, "instrument": "BTC/USDT", "leverage": 3.0,
         "qty": Decimal(1), "step": Decimal("0.001"), "min_qty": Decimal("0.001")}
    with pytest.raises(ValueError, match="taker fee"):
        gate_intent(d)
