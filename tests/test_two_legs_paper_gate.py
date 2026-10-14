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
