"""P2-3 netting, step 2: the shadow beside a portfolio backtest (Independent Quant Advisor 13 Oct 19:10 UK,
quant-review/specs-p2-3-p2-8-p2-5.md §2). Off by default; on, it reads what step A recorded and changes none of it.
A real run: two ping-pong Strategies on one perpetual flipping long and short on their own rules, opposed at times
(their intents seldom land on one bar, so crossing is pinned on legs built by hand below)."""

from decimal import Decimal as D
from types import SimpleNamespace

import pandas as pd
import pytest
from test_portfolio_run import BUSY, LOOSE, PERP, _same_fills, _spec

from sleeve_fund.instruments import FeeSchedule
from sleeve_fund.research.portfolio_run import netting_shadow, run_portfolio


def _specs():
    return [_spec("a", 2, PERP, **BUSY), _spec("b", 2, PERP | {"rise": 0.02, "dip": 0.01}, **BUSY)]


@pytest.fixture(scope="module")
def runs():
    return run_portfolio(_specs(), LOOSE), run_portfolio(_specs(), LOOSE, netting=True)


def test_off_by_default_and_on_it_changes_nothing_step_a_reports(runs):
    off, on = runs
    assert off.netting is None and on.netting is not None
    assert on.book.equals(off.book) and on.decisions == off.decisions and on.halt == off.halt
    assert all(_same_fills(on.legs[n], off.legs[n]) for n in off.legs)


def test_every_strategys_sub_ledger_is_its_step_a_pnl(runs):
    """MUST (§2.3): the sub-ledger, cash plus the open position at the last close, is the leg's own equity change."""
    gaps = runs[1].netting["sub_ledger_vs_step_a"]
    assert set(gaps) == {"a", "b"} and all(abs(g) < 1e-6 for g in gaps.values())


def test_sub_ledgers_saving_baseline_and_residual_are_the_account_to_the_cent(runs):
    """Advisor 13 Oct 20:25 and 20:55 UK: no funding saving line; this window's funding is the baseline for a missing
    rate, which both sides pay, so while the two are opposed the account pays less than its Strategies did. That is the
    conservative funding baseline line, by formula; the residual, measured without it, is rounding only."""
    n = runs[1].netting
    line = n["conservative_funding_baseline"]
    total = n["sub_ledger_total"] + sum(n["saving"].values()) + line + n["residual"]["amount"]
    assert round(total, 2) == round(n["account"], 2)
    assert "funding" not in n["saving"] and line > 0
    assert line == pytest.approx(n["funding"]["account"] - n["funding"]["sub_ledgers"], abs=0.01)
    assert abs(n["residual"]["amount"]) < 0.01 and n["residual"]["block"] is False


# --- crossing intents, on legs built by hand: two Strategies deciding opposite ways on one bar ---------------------

BAR = pd.Timestamp("2026-10-01", tz="UTC")
TAKER = D("0.0005")


def _leg(name, rows, close=100.0):
    """A leg's record as run_portfolio keeps it: market fills at the bar's close + 1 ns, at the close moved by the half
    spread (0.1%) against their side, the taker fee on that."""
    fills, decisions = [], {}
    for n, qty in enumerate(rows):
        px = close * (1.001 if qty > 0 else 0.999)
        oid = f"{name}-{n}"
        fills.append({"client_order_id": oid, "side": "BUY" if qty > 0 else "SELL", "filled_qty": abs(qty),
                      "avg_px": px, "commissions": [f"{abs(qty) * px * float(TAKER):.8f} USD"],
                      "ts_last": BAR + pd.Timedelta(1, "ns")})
        decisions[oid] = {"intent": "entry", "signal": {"close": close}}
    held = sum(rows)
    cash = sum(-q * close * (1.001 if q > 0 else 0.999) * (1 + (float(TAKER) if q > 0 else -float(TAKER)))
               for q in rows)
    equity = pd.Series([10_000.0, 10_000.0 + cash + held * close], index=[BAR - pd.Timedelta("1D"), BAR])
    return SimpleNamespace(fills=pd.DataFrame(fills).set_index("client_order_id"), decisions=decisions, funding=[],
                           equity=equity, starting_capital=10_000.0, insurance=[], half_spread=0.001)


def _hand(a, b):
    inst = SimpleNamespace(id="BTC-PERP.VENUE", taker_fee=TAKER)
    prices = pd.DataFrame({"close": [100.0, 100.0]}, index=[BAR - pd.Timedelta("1D"), BAR])
    specs = [SimpleNamespace(name=n, instrument=inst, fees=FeeSchedule(D("0.0002"), TAKER), half_spread=0.001,
                             capital=10_000.0, prices=prices) for n in ("a", "b")]
    hours = {n: prices for n in ("a", "b")}
    return netting_shadow({"a": _leg("a", a), "b": _leg("b", b)}, specs, hours)


def test_opposite_intents_on_one_bar_go_out_as_one_order_and_save_the_crossed_spread_and_fee():
    n = _hand([2.0], [-1.0])  # a buys 2, b sells 1: one net buy of 1, 1 crossed inside the account
    assert (n["orders_sent"], n["orders_alone"]) == (1, 2)
    assert n["crossed_notional"] == pytest.approx(100.0)  # one unit at the 100 mark
    # crossed: the spread both sides of that unit would have paid alone (0.1 + 0.1); fee: 0.05 on each side of it
    assert n["saving"]["crossed"] == pytest.approx(0.2) and n["saving"]["fee"] == pytest.approx(0.1)
    assert n["residual"]["amount"] == pytest.approx(0, abs=1e-6) and n["residual"]["block"] is False
    assert n["conservative_funding_baseline"] == 0  # no baseline settlement
    assert all(abs(g) < 1e-6 for g in n["sub_ledger_vs_step_a"].values())  # the sub-ledgers are step A's


def test_intents_that_fully_cross_send_nothing_and_same_side_intents_save_nothing():
    full = _hand([1.0], [-1.0])
    assert full["orders_sent"] == 0 and full["saving"]["fee"] == pytest.approx(0.1)
    same = _hand([1.0], [2.0])
    assert (same["orders_sent"], same["crossed_notional"], same["saving"]) == (1, 0.0, {})


def test_the_page_figures_opposed_exposure_and_margin_on_net_against_the_sum(runs):
    n = runs[1].netting
    assert 0 < n["opposed_exposure"]["a/b"] <= 1
    assert 0 < n["notional_peak"]["net"] <= n["notional_peak"]["sum"]  # margin on net: information, never P&L
