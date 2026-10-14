"""P2-8 Advisor M1 on paper: a two-leg entry is checked for both legs at once under the gate's one lock
(PaperGate.pair over portfolio.gate.check_pair), one share for both or neither, each leg reserved and journaled under
its own row; neither leg is ever reserved alone."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

from sqlalchemy import select

from sleeve_fund.paper.portfolio_gate import PaperGate
from sleeve_fund.store import Store, gate_decisions_t, gate_reservations_t, sleeves_t
from test_p21b_wiring_qa import _entry_intent, _ns, portfolio_pass

T0 = datetime(2026, 10, 14, 9, 0, 3, tzinfo=timezone.utc)
SPOT = {"instrument": "BTC/USDT", "params": {}, "venue": "KRAKEN", "side": 1}
PERP = {"instrument": "BTC/USDT", "params": {"market": "perp"}, "venue": "KRAKEN", "side": -1}


def _carry():
    store = Store.in_memory(portfolio_gate=True)
    store.create_two_legs(name="carry", strategy="buy_and_hold", bar_spec="1-DAY-LAST-EXTERNAL",
                          starting_balance=10_000, first=SPOT, second=PERP)
    return store


def _legs(qty="0.05"):
    spot = _entry_intent(qty)
    perp = {**_entry_intent(qty), "side": -1, "leverage": 1.0}
    return spot, perp


def _rows(store):
    with store.engine.connect() as c:
        names = {i: n for i, n in c.execute(select(sleeves_t.c.id, sleeves_t.c.name))}
        decided = [(names[r.sleeve_id], r.outcome, r.approved_qty) for r in
                   c.execute(select(gate_decisions_t).order_by(gate_decisions_t.c.seq))]
        held = sorted((names[r.sleeve_id], r.remaining_qty) for r in
                      c.execute(select(gate_reservations_t).where(gate_reservations_t.c.released_at.is_(None))))
    return decided, held


def test_both_legs_are_checked_together_and_each_is_reserved_under_its_own_row():
    store = _carry()
    portfolio_pass(store, T0, D(20_000))
    a, b = PaperGate(store).pair(("carry", "carry~2"), _legs(), _ns(T0 + timedelta(seconds=1)))
    assert a.approved_qty == b.approved_qty == D("0.05")
    decided, held = _rows(store)
    assert [d[0] for d in decided] == ["carry", "carry~2"]
    assert held == [("carry", D("0.05")), ("carry~2", D("0.05"))]


def test_a_pair_too_big_for_the_book_is_cut_by_one_share_on_both_legs():
    store = _carry()
    portfolio_pass(store, T0, D(20_000))
    a, b = PaperGate(store).pair(("carry", "carry~2"), _legs("5"), _ns(T0 + timedelta(seconds=1)))
    assert 0 < a.approved_qty < D(5) and a.approved_qty == b.approved_qty


def test_a_book_never_marked_refuses_both_legs_and_reserves_neither():
    store = _carry()
    a, b = PaperGate(store).pair(("carry", "carry~2"), _legs(), _ns(T0))
    assert a.approved_qty == b.approved_qty == 0
    decided, held = _rows(store)
    assert held == [] and {d[0] for d in decided} == {"carry", "carry~2"}


def test_an_intent_the_gate_cant_read_refuses_both_legs():
    store = _carry()
    portfolio_pass(store, T0, D(20_000))
    spot, perp = _legs()
    a, b = PaperGate(store).pair(("carry", "carry~2"), (spot, {**perp, "price": 60_000.0}),
                                 _ns(T0 + timedelta(seconds=1)))
    assert a.approved_qty == b.approved_qty == 0 and _rows(store)[1] == []


# --- the second leg's retry draws on the pair's reservation (Advisor 14 Oct 05:12 UK, leg-2 retry MUST) ---------------


def _sent(store, name, coid, gated, qty="0.05"):
    store.record_order(name, order_id=coid, side="BUY" if name == "carry" else "SELL", qty=float(qty), intent="entry",
                       reason="test", reservation=gated.checked.reservation)


def _paired(store):
    """Both legs checked and sent; the first filled, the second still out and held for its retry."""
    gate = PaperGate(store)
    a, b = gate.pair(("carry", "carry~2"), _legs(), _ns(T0 + timedelta(seconds=1)))
    _sent(store, "carry", "lead-1", a)
    store.update_order("lead-1", status="filled", fill_qty=0.05, fill_px=60_000.0)
    _sent(store, "carry~2", "follow-1", b)
    store.hold_for_retry("follow-1")
    return gate


def test_another_strategy_using_up_the_book_between_the_legs_cant_take_the_second_legs_share():
    """MUST 4's cell: the second leg's order is refused at the venue, another Strategy then takes all the room left in
    the book (margin binds first at 1x), and the retry still fills in full from the pair's reservation. A fresh check
    for the second leg would have refused it and forced the unwind the reservation exists to avoid."""
    store = _carry()
    store.create_sleeve(name="solo", strategy="trend_filter", instrument="ETH/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=1000)
    portfolio_pass(store, T0, D(20_000))
    gate = _paired(store)
    store.update_order("follow-1", status="rejected", message="venue refused")
    held = {n: q for n, q in _rows(store)[1]}
    assert held["carry~2"] == D("0.05")  # the reject kept it
    took = gate("solo", {**_entry_intent("1"), "instrument": "ETH"}, _ns(T0 + timedelta(seconds=2)))
    assert 0 < took.approved_qty < 1  # it took what was left of the book, and no more
    assert gate("carry~2", _legs()[1], _ns(T0 + timedelta(seconds=3))).approved_qty == 0  # a fresh check: refused
    rid, left = store.hand_on("follow-1")
    assert left == D("0.05")
    store.record_order("carry~2", order_id="follow-2", side="SELL", qty=0.05, intent="entry", reason="retry",
                       reservation=rid)
    store.update_order("follow-2", status="filled", fill_qty=0.05, fill_px=60_000.0)
    held = {n: q for n, q in _rows(store)[1]}
    assert "carry~2" not in held  # filled: released as a fill, the whole share used and never more
    with store.engine.connect() as c:
        why = c.execute(select(gate_reservations_t.c.release_reason)
                        .where(gate_reservations_t.c.decision_id == rid)).scalar()
    assert why == "fill"


def test_the_retry_takes_only_what_is_left_after_a_part_fill():
    """MUST 2: a part fill reduces the reservation, so the retry is cut to what is left."""
    store = _carry()
    portfolio_pass(store, T0, D(20_000))
    _paired(store)
    store.update_order("follow-1", fill_qty=0.02, fill_px=60_000.0)
    store.update_order("follow-1", status="canceled")
    assert dict(_rows(store)[1])["carry~2"] == D("0.03")
    assert store.hand_on("follow-1")[1] == D("0.03")


def test_an_unwind_or_deadline_releases_what_the_second_leg_still_held():
    """MUST 1: once the pair unwinds or its retry deadline passes, the share goes back to the book, and a later
    entry is checked against it again."""
    store = _carry()
    portfolio_pass(store, T0, D(20_000))
    _paired(store)
    store.update_order("follow-1", status="rejected")
    store.release_held("follow-1")
    assert "carry~2" not in dict(_rows(store)[1])
    assert store.hand_on("follow-1") is None  # nothing left to hand on


def test_a_second_leg_not_held_for_a_retry_releases_on_its_reject_as_any_order():
    store = _carry()
    portfolio_pass(store, T0, D(20_000))
    a, b = PaperGate(store).pair(("carry", "carry~2"), _legs(), _ns(T0 + timedelta(seconds=1)))
    _sent(store, "carry~2", "follow-1", b)
    store.update_order("follow-1", status="rejected")
    assert "carry~2" not in dict(_rows(store)[1]) and store.hand_on("follow-1") is None


def test_the_perp_legs_margin_in_the_book_is_at_its_own_leverage_not_the_profiles():
    """HoQA P2-8 F3 (Advisor 05:35 UK): the fund's holdings margin a carry's perp leg at its effective leverage,
    min(the leg's 1x, 2x, balanced's 2x) = 1x, as its reservation did; a single perp Strategy keeps the profile's cap."""
    from sleeve_fund import risk
    from sleeve_fund.paper.portfolio_gate import fund_holdings

    store = _carry()
    store.create_sleeve(name="solo", strategy="trend_filter", instrument="ETH/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=5000, params={"market": "perp"})
    for name, side, inst, px in (("carry", "BUY", "BTC", 60_000.0), ("carry~2", "SELL", "BTC", 60_000.0),
                                 ("solo", "BUY", "ETH", 3_000.0)):
        store.record_fill(name, side=side, qty=0.02 if inst == "BTC" else 0.4, price=px, fee=0.0, order_id=f"o-{name}",
                          trade_id=f"t-{name}", ts=T0)
    held = {(h.underlying, h.notional > 0): h for h in fund_holdings(store, T0, lambda s: 0.05)}
    assert held[("BTC", False)].margin == D(1200)  # the perp leg, 0.02 x 60,000 at 1x: not 600 at the profile's 2x
    assert held[("BTC", True)].margin == D(1200)  # the spot leg: its notional
    assert held[("ETH", True)].margin == D(600)  # a single perp Strategy: 0.4 x 3,000 at balanced's 2x, as before
    assert risk.leverage_cap(risk.profile("balanced"), store.sleeve("carry~2").params) == 1.0
    assert risk.leverage_cap(risk.profile("conservative"), {"leg": {"leverage": "2"}}) == 1.0  # never above the profile
