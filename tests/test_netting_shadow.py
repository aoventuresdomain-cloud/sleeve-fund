"""P2-3 netting, shadow core (Independent Quant Advisor 13 Oct 19:10 UK, quant-review/specs-p2-3-p2-8-p2-5.md §2): one net
order per instrument per bar, each Strategy's sub-ledger its own step A lines, the saving at book level, the
reconcile residual and its block, and the opposed exposure per pair. Mark 100, half spread 0.1%, taker 0.05%."""

from decimal import Decimal as D

import pytest

from sleeve_fund.portfolio.netting import (
    RESIDUAL_BP,
    Fill,
    Funding,
    net_orders,
    opposed_exposure,
    reconcile,
    shadow,
)

TAKER, HALF = D("0.0005"), D("0.001")
T = 1_000


def mark(instrument, ts):
    return D(100)


def half(instrument, ts):
    return HALF


def alone(strategy, qty, ts=T, instrument="BTC"):
    """A step A market fill trading alone: the mark moved by the slippage against its side, the taker fee on it."""
    qty = D(qty)
    price = D(100) * (1 + HALF) if qty > 0 else D(100) * (1 - HALF)
    return Fill(strategy, instrument, ts, qty, price, abs(qty) * price * TAKER)


def test_opposite_signals_go_out_as_one_net_order_and_the_crossed_volume_pays_no_fee():
    (o,) = net_orders([alone("A", 2), alone("B", -1)], mark, half, TAKER)
    assert (o.qty, o.crossed, o.price, o.strategies) == (D(1), D(1), D("100.1"), ("A", "B"))
    assert o.fee == D("0.05005")  # the venue's fee on the one net unit only


def test_intents_that_fully_cross_send_nothing_and_book_at_the_mark():
    (o,) = net_orders([alone("A", 1), alone("B", -1)], mark, half, TAKER)
    assert (o.qty, o.crossed, o.price, o.fee) == (D(0), D(1), D(100), D(0))


def test_each_strategys_sub_ledger_is_its_step_a_pnl_and_the_saving_is_the_books():
    fills = [alone("A", 2), alone("B", -1)]
    funding = [Funding("A", "BTC", T + 1, D("0.0001"), D(100), D("-0.02")),
               Funding("B", "BTC", T + 1, D("0.0001"), D(100), D("0.01"))]
    s = shadow(fills, funding, mark, half, TAKER)
    assert s.strategy_cash() == {"A": fills[0].cash - D("0.02"), "B": fills[1].cash + D("0.01")}  # step A, line for line
    assert s.saving() == {"crossed": D("0.2"), "fee": D("0.1")}  # funding is linear in the position: nothing saved
    assert all(r["kind"] != "residual" for r in s.book)
    rec = reconcile(sum(s.strategy_cash().values()), sum(s.saving().values()), s.account, D(10_000))
    assert rec["amount"] == 0 and not rec["block"]


def test_same_side_intents_net_with_nothing_crossed_and_nothing_saved():
    s = shadow([alone("A", 1), alone("B", 2)], [], mark, half, TAKER)
    (o,) = s.orders
    assert (o.qty, o.crossed) == (D(3), D(0)) and s.saving() == {}
    assert s.account == sum(s.strategy_cash().values())


def test_bars_and_instruments_are_netted_apart():
    fills = [alone("A", 1), alone("B", -1, ts=T + 60), alone("C", -1, instrument="ETH")]
    orders = net_orders(fills, mark, half, TAKER)
    assert [(o.ts, o.instrument, o.qty) for o in orders] == [(T, "BTC", D(1)), (T, "ETH", D(-1)),
                                                             (T + 60, "BTC", D(-1))]


def test_the_net_positions_funding_differs_only_by_each_accounts_rounding():
    """0.3 x 0.0001 x 100 per Strategy rounds to the cent apart; the account rounds the net once: the difference is a
    book line, never a Strategy's."""
    fills = [alone("A", "0.333"), alone("B", "0.333")]
    funding = [Funding(n, "BTC", T + 1, D("0.0001"), D(100), D("-0.00")) for n in ("A", "B")]  # each rounded to 0.00
    s = shadow(fills, funding, mark, half, TAKER, funding_quantum=D("0.01"))
    assert s.saving().get("funding") == D("-0.01")  # 0.666 x 0.01 = 0.00666 rounds to 0.01 paid once
    assert s.strategy_cash()["A"] == fills[0].cash


@pytest.mark.parametrize(("residual", "block"), [(D("4.99"), False), (D("5.01"), True)])
def test_a_residual_above_five_bp_of_equity_alerts_and_blocks_entries(residual, block):
    rec = reconcile(D(-100), D(0), D(-100) + residual, D(10_000))
    assert rec["kind"] == "residual" and rec["amount"] == residual and rec["block"] is block
    assert RESIDUAL_BP == 5


def test_money_is_exact_a_float_is_refused():
    with pytest.raises(TypeError):
        Fill("A", "BTC", T, 1.0, D(100), D(0))
    with pytest.raises(TypeError):
        reconcile(D(0), D(0), D(0), 10_000.0)


def test_opposed_exposure_is_the_share_of_the_pairs_gross_that_cancels():
    fills = [alone("A", 2), alone("B", -1), alone("C", 1)]
    got = opposed_exposure(fills, mark, [T])
    assert got[("A", "B")] == pytest.approx(2 * 100 / 300)
    assert got[("A", "C")] == 0.0 and got[("B", "C")] == pytest.approx(1.0)
