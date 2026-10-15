"""LIQ-FEE (HoE 14:06 UK, 15 Oct; Advisor ruling pending): a perp position's isolated margin is its notional at entry
over the leverage cap LESS the entry fees it paid, so its liquidation price is nearer the entry by about fee / qty than
one worked from the whole notional / leverage. The QA master's hand derivation (test_hub_146_qa, C-NA3-LIQ) takes the fee
off; the engine kept it in, which the 0.1% tolerance hid at 0.05% taker and the 0.055% published rate showed.

One margin figure everywhere it is worked: the fill (_set_liq), the decision's estimate (entry_liquidation), a restart's
recompute (_restored_liq) and its safety stop. A partly reduced position keeps the share of its entry fees still held.
CI also runs the restore cell against Postgres via TEST_DATABASE_URL."""

import pytest

from sleeve_fund import margin

from test_resume_leg_restore import FUNDED, PERP_OPEN, _perp_enter, _perp_legs, _perp_restart  # noqa: F401
from test_resume_leg_restore import store  # noqa: F401  (the journal fixture, on Postgres when configured)

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
    entry, fees = 60_007.8, abs(0.11) * 60_007.8 * 0.00055
    cash = 50_000.0 - qty * entry - fees  # spot-style: the balance after the fee, less qty x entry
    got = margin.isolated_liquidation(cash, qty, entry, lev, MM, fees=fees)
    assert got == pytest.approx(_by_hand(qty, entry, lev, fees), rel=1e-12)
    without = margin.isolated_liquidation(cash, qty, entry, lev, MM)
    assert (got - without) * (1 if qty > 0 else -1) == pytest.approx(fees / abs(qty), rel=1e-9)  # nearer by fee/qty


def test_the_decisions_estimate_takes_its_fee_off_the_margin_as_the_fill_will():
    qty, close, fee = 0.11, 60_007.8, 0.00055
    for side in (1, -1):
        liq, _ = margin.entry_liquidation(50_000.0, qty, close, side, fee, MM, 2.0)
        assert liq == pytest.approx(_by_hand(side * qty, close, 2.0, qty * close * fee), rel=1e-12), side


def test_a_reduced_position_keeps_the_share_of_its_entry_fees_it_still_holds():
    # entry 1.0 (fee 6), add 0.5 (fee 3), reduce 0.6: 0.9 of 1.5 held -> 9 x 0.6; a reduction's own fee never counts
    fills = [(1.0, 6.0), (0.5, 3.0), (-0.6, 2.0)]
    assert margin.entry_fees_held(fills) == pytest.approx(9.0 * 0.9 / 1.5)
    assert margin.entry_fees_held(fills + [(-0.9, 1.0)]) == 0.0  # flat: nothing held
    # through flat: the reversing fill's fee, in the share of it that opened the other side
    assert margin.entry_fees_held([(1.0, 6.0), (-1.5, 4.5)]) == pytest.approx(4.5 / 3)
    assert margin.entry_fees_held([(-2.0, 8.0), (-1.0, 2.0)]) == pytest.approx(10.0)  # a short adds the same way


def test_a_restart_counts_the_entry_fee_off_the_margin_as_the_fill_did(store, instrument):
    """The fill journaled its price with the fee off the margin: the restart's recompute agrees within a tick, so it
    raises no liquidation mismatch and keeps the journaled price."""
    _perp_legs(store)
    e = dict(PERP_OPEN["perp-x"])
    _perp_enter(store, "perp-x", minutes=0, **{**e, "liq": 1.0})
    strat = _perp_restart(store, instrument, "perp-x")
    fee = round(e["px"] * e["qty"] * 0.0026, 2)  # what _perp_enter journaled
    book = strat.runtime.book
    with_fee = margin.isolated_liquidation(book["cash"] + FUNDED, book["qty"], e["px"],
                                           strat.runtime.profile.max_leverage, strat._cfg.perp.maintenance_margin,
                                           fees=fee)
    store.merge_order_signal("E-perp-x", {"position_liquidation_px": with_fee})
    strat = _perp_restart(store, instrument, "perp-x")
    assert strat._liq_px == pytest.approx(with_fee)
    assert len([x for x in store.events("perp-x", limit=200) if x["kind"] == "liquidation_mismatch"]) == 1  # the 1.0
