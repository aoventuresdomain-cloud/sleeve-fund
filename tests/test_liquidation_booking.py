"""GAP-LIQ-CAP (Independent Quant Advisor, 6 Oct 23:42 and 7 Oct 00:19): on isolated margin a liquidation books its
fill at the bankruptcy price, so it loses exactly X, the posted margin plus the entry and liquidation fees, gapped or
not. How far the market went past bankruptcy is the venue's insurance fund's, and margin left when it closed short of
bankruptcy is kept by the venue: diagnostics only, never P&L. The pure booking; the engine wiring stacks on the
stop-safety work (GAP-LIQ)."""

import pytest

from sleeve_fund import markets
from sleeve_fund.markets import liquidation_booking

FEES = {"entry_fee": 0.06, "liquidation_fee": 0.04}


@pytest.mark.parametrize("qty, lev, exit_px", [
    (1.0, 3.0, 50.0),  # a 3x long gapped from 100 to 50: bankruptcy is about 66.7
    (1.0, 3.0, 66.0),
    (1.0, 3.0, 67.5),  # inside the margin: short of bankruptcy
    (-1.0, 2.0, 170.0),  # a 2x short gapped to 170: bankruptcy is 150
    (-1.0, 2.0, 140.0),
    (0.3, 3.0, 1.0),  # almost everything gone
], ids=["long-gap", "long-just-past", "long-short-of-bankruptcy", "short-gap", "short-short-of", "long-near-zero"])
def test_a_liquidation_books_exactly_the_posted_margin_and_both_fees_gapped_or_not(qty, lev, exit_px):
    b = liquidation_booking(qty, 100.0, exit_px, lev, **FEES)
    x = markets.isolated_margin(qty, 100.0, lev) + 0.10  # the RAL halt's X (#155 _margin_lost)
    assert b.loss == pytest.approx(x)
    assert -qty * (b.fill_px - 100.0) + 0.10 == pytest.approx(x)  # the fill at bankruptcy books X with the fees
    assert b.market_px == exit_px
    assert b.insurance >= 0 and b.forfeited >= 0 and not (b.insurance and b.forfeited)


def test_the_loss_past_bankruptcy_is_the_insurance_funds_and_the_margin_left_short_of_it_is_forfeited():
    gap = liquidation_booking(1.0, 100.0, 50.0, 3.0, **FEES)
    assert gap.fill_px == pytest.approx(200 / 3) and gap.insurance == pytest.approx(200 / 3 - 50.0)
    assert gap.forfeited == 0
    short = liquidation_booking(-1.0, 100.0, 170.0, 2.0, **FEES)
    assert short.fill_px == pytest.approx(150.0) and short.insurance == pytest.approx(20.0)
    assert short.loss == pytest.approx(50.10)
    at_liq = markets.isolated_liquidation(10_000 - 100.0, 1.0, 100.0, 3.0, 0.005)  # the venue's price for it
    near = liquidation_booking(1.0, 100.0, at_liq, 3.0, **FEES)
    assert near.insurance == 0 and near.forfeited == pytest.approx(at_liq - 200 / 3)  # the maintenance margin left
    assert near.fill_px == pytest.approx(200 / 3) and near.loss == pytest.approx(100 / 3 + 0.10)


def test_qas_set_ups_excess_past_bankruptcy():
    """QA's GAP-LIQ-CAP cases: 3x short 0.05 at 60,000 gapped to 96,000; 2x long 0.0333 at 60,000 gapped to 24,000."""
    short = liquidation_booking(-0.05, 60_000.0, 96_000.0, 3.0, entry_fee=0, liquidation_fee=0)
    assert (short.fill_px, short.insurance, short.loss) == (pytest.approx(80_000.0), pytest.approx(800.0),
                                                             pytest.approx(1_000.0))
    long = liquidation_booking(1 / 30, 60_000.0, 24_000.0, 2.0, entry_fee=0, liquidation_fee=0)
    assert (long.fill_px, long.insurance) == (pytest.approx(30_000.0), pytest.approx(200.0))


def test_at_exactly_the_bankruptcy_price_there_is_no_cover_and_nothing_forfeited():
    b = liquidation_booking(-1.0, 100.0, 150.0, 2.0, **FEES)
    assert b.insurance == pytest.approx(0, abs=1e-12) and b.forfeited == pytest.approx(0, abs=1e-12)


def test_with_adds_x_is_the_whole_liquidated_quantity_at_its_average_entry():
    """RAL 7/8: X over the whole liquidated qty. 1 at 100 then 1 at 110: 2 at 105, margin 70 at 3x."""
    b = liquidation_booking(2.0, 105.0, 60.0, 3.0, entry_fee=0.13, liquidation_fee=0.07)
    assert b.loss == pytest.approx(70.0 + 0.20) and b.fill_px == pytest.approx(70.0)
    assert b.insurance == pytest.approx(2 * (70.0 - 60.0))


def test_the_margin_is_never_more_than_there_was_to_put_up():
    b = liquidation_booking(1.0, 100.0, 50.0, 3.0, balance=20.0, **FEES)
    assert b.loss == pytest.approx(20.10) and b.fill_px == pytest.approx(80.0) and b.insurance == pytest.approx(30.0)


@pytest.mark.parametrize("args", [
    (0.0, 100.0, 50.0, 3.0, 0.1, 0.1), (1.0, 0.0, 50.0, 3.0, 0.1, 0.1), (1.0, 100.0, 0.0, 3.0, 0.1, 0.1),
    (1.0, 100.0, 50.0, 0.0, 0.1, 0.1), (1.0, 100.0, 50.0, 3.0, -0.1, 0.1), (1.0, 100.0, 50.0, 3.0, 0.1, -0.1),
])
def test_a_booking_without_a_position_prices_leverage_or_with_a_negative_fee_is_refused(args):
    with pytest.raises(ValueError):
        liquidation_booking(*args)
