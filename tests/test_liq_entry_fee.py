"""LIQ-FEE (HoE 14:06 UK; Advisor ruling (a), 15 Oct): a perp position's isolated margin is its notional at entry over
the leverage cap LESS the entry fees it still holds, so its liquidation price is nearer the entry by fee / qty: the
entry fee rate x the entry price (0.055 % of the entry at any leverage), the entry fee only, not a round trip. Relative
to the liquidation price that is the rate x entry / liq: about 2 x the rate at 2x (0.109 % at 0.055 %), 1.5 x at 3x,
which is the gap the QA master's hand derivation (test_hub_146_qa, C-NA3-LIQ, also the entry fee only) showed against
the engine, which kept the fee in and was hidden by the 0.1 % tolerance at 0.05 % taker.

Only the liquidation price moves: the fee is booked once, to cash, as before (Advisor MUST 1). The fee taken off is the
money actually charged on the position's own fills, as the journal books it (replay_book entry_fees), and the
decision's estimate uses the rate the venue's resolver gave the Strategy (MUST 2). One liquidation function
(margin.isolated_liquidation) at the fill, the estimate, a restart's recompute and the dashboard (MUST 5).
CI also runs the restore cell against Postgres via TEST_DATABASE_URL."""

from decimal import Decimal

import pytest
from test_resume_leg_restore import (
    FUNDED,
    PERP_OPEN,
    _perp_enter,
    _perp_legs,
    _perp_restart,
)
from test_sleeve_runtime import store as _store_fixture

from sleeve_fund import margin
from sleeve_fund.store import replay_book

store = _store_fixture  # the same journal, on Postgres when TEST_DATABASE_URL is set

MM = 0.005


def _by_hand(qty, entry, lev, fees, mm=MM):
    """Independently of margin.py: margin = notional / lev - fees; formula A (maintenance on the value at the price)
    and B (on the entry value), the nearer kept (VS-1-TABLE)."""
    side = 1 if qty > 0 else -1
    im = abs(qty) * entry / lev - fees
    a = (entry - side * im / abs(qty)) / (1 - side * mm)
    b = entry - side * (im - mm * abs(qty) * entry) / abs(qty)
    return max(a, b) if side > 0 else min(a, b)


@pytest.mark.parametrize("qty, lev", [(0.11, 2.0), (-0.11, 2.0), (0.25, 3.0), (-0.25, 3.0)])
def test_the_isolated_margin_is_the_notional_over_the_leverage_less_the_entry_fees(qty, lev):
    entry, rate = 60_007.8, 0.00055
    fees = abs(qty) * entry * rate
    cash = 50_000.0 - qty * entry - fees  # spot-style: the balance after the fee, less qty x entry
    got = margin.isolated_liquidation(cash, qty, entry, lev, MM, fees=fees)
    assert got == pytest.approx(_by_hand(qty, entry, lev, fees), rel=1e-12)
    without = margin.isolated_liquidation(cash, qty, entry, lev, MM)
    moved = (got - without) * (1 if qty > 0 else -1)  # nearer by fee / qty (formula B), or that over 1 -/+ mm (A)
    assert fees / abs(qty) / (1 + MM) - 1e-9 <= moved <= fees / abs(qty) / (1 - MM) + 1e-9


def test_the_gap_is_the_entry_fee_alone_not_a_round_trip():
    """Advisor MUST 3: at 2x long (formula B binds) the price moves by exactly the entry fee rate x the entry, 0.055 %
    of it, at any size; against the liquidation price that reads about 0.109 %, PE1's figure. Not 2 x the rate."""
    entry, rate, mm = 60_007.8, 0.00055, MM
    for lev, qty in ((2.0, 0.11), (2.0, 1.7)):
        fees = qty * entry * rate
        cash = 500_000.0 - qty * entry - fees  # a balance the margin fits in, so the cap at it never binds
        got = margin.isolated_liquidation(cash, qty, entry, lev, mm, fees=fees)
        without = margin.isolated_liquidation(cash, qty, entry, lev, mm)
        assert (got - without) / entry == pytest.approx(rate, rel=1e-9)  # the entry fee alone
        assert (got - without) / got == pytest.approx(0.00109, abs=0.00001)  # as the master's tolerance reads it


def test_the_decisions_estimate_takes_its_fee_off_the_margin_as_the_fill_will():
    qty, close, fee = 0.11, 60_007.8, 0.00055
    for side in (1, -1):
        liq, _ = margin.entry_liquidation(50_000.0, qty, close, side, fee, MM, 2.0)
        assert liq == pytest.approx(_by_hand(side * qty, close, 2.0, qty * close * fee), rel=1e-12), side


SEQUENCES = {
    "entry, add, reduce": [(1.0, 6.0), (0.5, 3.0), (-0.6, 2.0)],
    "then flat": [(1.0, 6.0), (0.5, 3.0), (-0.6, 2.0), (-0.9, 1.0)],
    "through flat": [(1.0, 6.0), (-1.5, 4.5)],
    "a short adds": [(-2.0, 8.0), (-1.0, 2.0)],
    "flat then again": [(1.0, 6.0), (-1.0, 3.0), (0.4, 1.2)],
}


@pytest.mark.parametrize("fills", SEQUENCES.values(), ids=SEQUENCES.keys())
def test_the_engines_entry_fees_held_fill_by_fill_are_the_journals(fills):
    """The engine keeps the fees fill by fill (margin.entry_fees_after); a restart takes the journal's (replay_book
    entry_fees). The two agree, so a restart's recompute is the fill's own price."""
    held = pos = 0.0
    for q, fee in fills:
        held, pos = margin.entry_fees_after(held, pos, q, fee), pos + q
    journal = [{"side": "BUY" if q > 0 else "SELL", "qty": abs(q), "price": 100.0, "fee": fee} for q, fee in fills]
    assert held == pytest.approx(float(replay_book(journal, 0.0)["entry_fees"]), abs=1e-12)
    assert SEQUENCES["entry, add, reduce"] and margin.entry_fees_after(9.0, 1.5, -0.6, 2.0) == pytest.approx(9.0 * 0.6)


def test_a_restart_counts_the_entry_fee_off_the_margin_as_the_fill_did(store, instrument):
    """The fill journaled its price with the fee off the margin: the restart's recompute agrees within a tick, so it
    raises no liquidation mismatch and keeps the journaled price."""
    _perp_legs(store)
    e = dict(PERP_OPEN["perp-x"])
    _perp_enter(store, "perp-x", minutes=0, **{**e, "liq": 1.0})
    strat = _perp_restart(store, instrument, "perp-x")
    fee = round(e["px"] * e["qty"] * 0.0026, 2)  # what _perp_enter journaled
    book = strat.runtime.book
    assert float(book["entry_fees"]) == pytest.approx(fee)
    with_fee = margin.isolated_liquidation(book["cash"] + FUNDED, book["qty"], e["px"],
                                           strat.runtime.profile.max_leverage, strat._cfg.perp.maintenance_margin,
                                           fees=fee)
    store.merge_order_signal("E-perp-x", {"position_liquidation_px": with_fee})
    strat = _perp_restart(store, instrument, "perp-x")
    assert strat._liq_px == pytest.approx(with_fee)
    assert len([x for x in store.events("perp-x", limit=200) if x["kind"] == "liquidation_mismatch"]) == 1  # the 1.0


def test_the_fee_is_booked_once_to_cash_and_only_the_liquidation_price_moves(monkeypatch):
    """Advisor MUST 1: the same 2x long entered through the hub-fed paper node, once as now and once with the fee kept
    in the margin (as before LIQ-FEE). Every fill, its fee and the journal's cash, and every equity mark's cash, are the
    same to the cent; only the liquidation price journaled at the fill moves, nearer by the fee charged / qty."""
    from test_hub_146_qa import _probe_classes, filled_orders, flat_prices, paper

    from sleeve_fund import markets
    from sleeve_fund.strategies import REGISTRY

    monkeypatch.setitem(REGISTRY, "probe", _probe_classes())  # the QA harness's probe Strategy

    def run():
        r = paper(flat_prices(30), side=1, perp=True, profile="balanced", stop=0.01, leave=25)
        (entry,) = filled_orders(r, "entry")
        return r, entry

    now, entry_now = run()
    real = margin.isolated_liquidation
    before = lambda *a, fees=0.0, **kw: real(*a, **kw)  # the fee kept in the margin
    monkeypatch.setattr(margin, "isolated_liquidation", before)
    monkeypatch.setattr(markets, "isolated_liquidation", before)
    was, entry_was = run()

    def cents(x):
        return Decimal(str(x)).quantize(Decimal("0.01"))

    assert [(f["side"], f["qty"], f["price"], cents(f["fee"])) for f in now.fills] == \
           [(f["side"], f["qty"], f["price"], cents(f["fee"])) for f in was.fills]
    assert cents(now.store.journal_book("q146", 0)["cash"]) == cents(was.store.journal_book("q146", 0)["cash"])
    assert [cents(m["cash"]) for m in now.store.equity_series("q146")] == \
           [cents(m["cash"]) for m in was.store.equity_series("q146")]
    (fill,) = [f for f in now.fills if f["order_id"] == entry_now["order_id"]]
    liq_now = entry_now["signal"]["position_liquidation_px"]
    liq_was = entry_was["signal"]["position_liquidation_px"]
    assert liq_now - liq_was == pytest.approx(float(fill["fee"]) / float(fill["qty"]), rel=1e-6)
