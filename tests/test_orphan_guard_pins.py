"""A2 orphan guard, three Done-when lines the CR's mutants showed unpinned (HoE, 15 Oct). Each cell goes red when its
guard is taken out: the close sent from the late fill itself (base.py's on_order_filled), the safety stop armed once the
close's grace has run (_late_safety, MUST 3) and the fund-wide hold kept while the owner is still down
(unprotected_pass, line 6)."""
from datetime import timedelta
from types import SimpleNamespace

import pytest
from test_orphan_late_fill import DOWN, _closes, _down, _fill, _late

from sleeve_fund.paper import orphan_guard as og


@pytest.fixture
def jstore():
    from test_gate_ledger import _store, _strategies

    s = _store()
    _strategies(s)
    return s


def test_a_gated_entrys_fill_sends_its_late_close_from_the_fill_itself(monkeypatch):
    """A2-DW4: a gated entry that fills after its cancel is closed as its fill arrives, not at the next bar. The fill
    handler calls the one sender (_send_late_closes) itself, once, while guarding."""
    from test_hub_146_qa import _probe_classes, flat_prices, paper

    from sleeve_fund.strategies import REGISTRY

    probe, config = _probe_classes()
    calls = []

    class Gated(probe):
        filling = False

        def on_order_filled(self, event) -> None:
            coid = str(event.client_order_id)
            gated = self.decisions.get(coid, {}).get("intent") == "entry"
            if gated:  # as the portfolio gate's answer holds it (_reserve); the fill hands its quantity back
                self._reserved[coid] = SimpleNamespace(filled=lambda qty: None, release=lambda *a, **k: None)
            self.filling = gated
            try:
                super().on_order_filled(event)
            finally:
                self.filling = False
                self._reserved.pop(coid, None)

        def _guarding(self) -> bool:  # paper with the gate, for the gated entry's fill only
            return self.filling

        def _send_late_closes(self) -> None:
            calls.append(self.filling)

    monkeypatch.setitem(REGISTRY, "probe", (Gated, config))
    run = paper(flat_prices(40), leave=30)
    entries = [o for o in run.orders if o["intent"] == "entry"]
    assert entries and {f["order_id"] for f in run.fills} >= {entries[0]["order_id"]}, "set-up: the entry never filled"
    assert calls == [True] * len(entries), f"the gated entry's fill did not send its late close itself: {calls}"


def test_the_safety_stop_is_armed_once_the_late_close_is_unfilled_for_backstop_grace_ms():
    """MUST 3: inside the grace nothing is armed; once backstop_grace_ms has run with the close unfilled, _late_safety
    arms the safety stop for the late fill's position, with why."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    strat, t = _late()
    strat._safety_stop, strat._safety_pending, strat._safety_why = False, None, None
    strat._late_safety_due = lambda price: LongFlatStrategy._late_safety_due(strat, price)

    t[0] += 999_000_000
    LongFlatStrategy._late_safety(strat, 59_900.0)
    assert strat._safety_stop is False and strat._safety_pending is None  # 999 ms: inside the grace

    t[0] += 1_000_000
    LongFlatStrategy._late_safety(strat, 59_900.0)
    assert strat._safety_stop is True, "the grace ran out and the safety stop was never armed"
    assert strat._safety_pending == (1_000.0, 0.1)  # the late fill's position, as _mark holds it
    assert "1,000 ms" in strat._safety_why


def test_the_fund_wide_hold_stays_while_the_owner_is_down_even_once_the_late_fill_is_flat(jstore):
    """Line 6, Advisor HEAD3-DW (b): the hold lifts only when the owner is back AND the late fill is flat. Flat with the
    owner still down keeps the CRITICAL episode open and every Strategy in the fund held."""
    _down(jstore)
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1))
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))
    assert "fund" in og.resting_block(jstore, "b", DOWN + timedelta(seconds=6))

    (close,) = _closes(jstore)
    _fill(jstore, "c1", 0.02, DOWN + timedelta(seconds=7), order_id=close["order_id"], side="SELL")  # flat, owner down
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=10))
    assert jstore.last_event("a", (og.UNPROTECTED, og.PROTECTED))["kind"] == og.UNPROTECTED
    assert "fund" in (og.resting_block(jstore, "b", DOWN + timedelta(seconds=11)) or ""), "released while still down"

    og.owner_up(jstore, "a", DOWN + timedelta(minutes=1))  # the control: back and flat, the hold lifts
    og.supervisor_pass(jstore, DOWN + timedelta(minutes=1, seconds=5))
    assert jstore.last_event("a", (og.UNPROTECTED, og.PROTECTED))["kind"] == og.PROTECTED
    assert og.resting_block(jstore, "b", DOWN + timedelta(minutes=1, seconds=6)) is None
