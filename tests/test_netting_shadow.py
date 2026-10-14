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
    assert s.saving() == {"crossed": D("0.2"), "fee": D("0.1")}  # funding has no saving line (Advisor 20:25 UK)
    assert s.account_funding == D("-0.01")  # the net unit's: what the two sub-ledgers' funding sums to
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


def test_the_net_positions_funding_differs_only_by_rounding_and_that_goes_to_the_residual():
    """0.333 x 0.0001 x 100 per Strategy rounds to the cent apart; the account rounds the net once. Advisor 20:25 UK:
    no funding saving line; the sub-ledgers' funding reconciles to the account's and the difference is residual."""
    fills = [alone("A", "0.333"), alone("B", "0.333")]
    funding = [Funding(n, "BTC", T + 1, D("0.0001"), D(100), D("-0.00")) for n in ("A", "B")]  # each rounded to 0.00
    s = shadow(fills, funding, mark, half, TAKER, funding_quantum=D("0.01"))
    assert "funding" not in s.saving() and s.account_funding == D("-0.01")  # 0.00666 rounds to 0.01 paid once
    assert s.strategy_cash()["A"] == fills[0].cash
    rec = reconcile(sum(s.strategy_cash().values()), sum(s.saving().values()), s.account, D(10_000))
    assert rec["amount"] == D("-0.01") and not rec["block"]  # 0.01 bp of the account, inside the 5 bp


def test_sub_ledgers_plus_saving_plus_residual_are_the_account_to_the_cent():
    """Advisor 13 Oct 20:25 UK: the identity the book's page rests on."""
    fills = [alone("A", "2.5"), alone("B", "-1.25"), alone("C", "0.4"), alone("A", "-1", ts=T + 60),
             alone("B", "1", ts=T + 60)]
    funding = [Funding(n, "BTC", T + 30, D("0.0001"), D(100), a) for n, a in (("A", D("-0.03")), ("B", D("0.01")),
                                                                              ("C", D("-0.00")))]
    s = shadow(fills, funding, mark, half, TAKER, funding_quantum=D("0.01"))
    sub, saving = sum(s.strategy_cash().values()), sum(s.saving().values())
    rec = reconcile(sub, saving, s.account, D(10_000))
    cent = D("0.01")
    assert (sub + saving + rec["amount"]).quantize(cent) == s.account.quantize(cent)


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


def test_the_conservative_funding_baseline_is_its_own_line_by_formula_and_nil_with_every_rate_present():
    """Advisor 13 Oct 20:55 UK: at a baseline settlement (a missing rate, every side pays) opposed Strategies pay more
    than the account on its net; that difference is a named line by formula, never a plug, and the residual is
    measured without it."""
    fills = [alone("A", 2), alone("B", -1)]
    rows = [Funding("A", "BTC", T + 1, D("0.0001"), D(100), D("-0.02"), both_pay=True),
            Funding("B", "BTC", T + 1, D("0.0001"), D(100), D("-0.01"), both_pay=True)]  # B short pays too
    s = shadow(fills, rows, mark, half, TAKER)
    (line,) = s.baseline
    assert line["kind"] == "conservative_funding_baseline" and line["amount"] == D("0.02")  # 0.03 alone, 0.01 net
    assert s.account_funding == D("-0.01")
    rec = reconcile(sum(s.strategy_cash().values()), sum(s.saving().values()) + s.baseline_total(), s.account,
                    D(10_000))
    assert rec["amount"] == 0 and not rec["block"]
    signed = [Funding("A", "BTC", T + 1, D("0.0001"), D(100), D("-0.02")),
              Funding("B", "BTC", T + 1, D("0.0001"), D(100), D("0.01"))]
    assert shadow(fills, signed, mark, half, TAKER).baseline == ()  # every rate present: nil
