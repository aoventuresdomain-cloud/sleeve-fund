"""Guards-on twins (Head of QA, 6 Oct 21:25) of #155's three restart X/Y cases in test_degraded_155_qa.py, which lift
the open-risk limit and the restart's safety stop to count what a liquidation loses. On their exact set-up, with the
guard on, each guard does its job: the stopless 2x short is never opened, and a short carried over a restart is
closed by the safety stop, not liquidated."""
import dataclasses

import pytest

import test_replay
from sleeve_fund import risk
from sleeve_fund.research.replay import replay
from sleeve_fund.store import Store
from test_degraded_155_qa import PERP, _record_session, _session_meta

NAME = "ping-pong-test"
PARAMS = {"rise": 0.01, "dip": 0.005, **PERP}
OPENS_SHORT = [(5, 0.0), (20, 0.015), (2, 0.0)]  # the three cases' first session: a 1.5% rise, and ping_pong shorts


@pytest.fixture
def whole_equity(monkeypatch):
    """As the three cases: every profile puts the whole equity up as margin (2x on balanced)."""
    for nm, p in list(risk.PROFILES.items()):
        monkeypatch.setitem(risk.PROFILES, nm, dataclasses.replace(p, max_position_pct=1.0))


def _kinds(store):
    return {e["kind"] for e in store.events(NAME, limit=1000)}


def test_with_the_open_risk_limit_on_their_stopless_2x_short_is_never_opened(tmp_path, whole_equity):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    path = tmp_path / "s1.jsonl.gz"
    _record_session(path, _session_meta(10_000, PARAMS), OPENS_SHORT)
    assert replay(path, store=store) == []  # nothing sent
    assert store.journal_book(NAME, 10_000)["qty"] == 0 and not store.fills(NAME)
    kinds = _kinds(store)
    assert "entry_refused_open_risk" in kinds and not {"liquidation", "incident"} & kinds


def _carried_then(tmp_path, legs):
    """The three cases' restart: the short opened in one process (open-risk limit lifted, so there is one to carry),
    then a restarted process on a later session moving by `legs`, every other guard on. Returns the store and the
    restart's orders."""
    store = Store(f"sqlite:///{tmp_path}/t.db")
    start0 = test_replay.START
    try:
        s1 = tmp_path / "s1.jsonl.gz"
        _record_session(s1, _session_meta(10_000, PARAMS), OPENS_SHORT)
        before = len(replay(s1, store=store))
        book = store.journal_book(NAME, 10_000)
        assert book["qty"] < 0  # carried into the restart
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
        test_replay.START = start0 + 2 * 3600 * 10**9
        s2 = tmp_path / "s2.jsonl.gz"
        _record_session(s2, _session_meta(book["cash"] + book["qty"] * book["entry_px"], PARAMS), legs,
                        px=60_000.0 * 1.015)
        orders = replay(s2, store=store)[before:]
    finally:
        test_replay.START = start0
    incidents = [e["message"] for e in store.events(NAME, limit=1000) if e["kind"] == "incident"]
    assert len([m for m in incidents if "safety stop" in m]) == 1, incidents  # the restart set it, with an incident
    return store, orders


LIFT = pytest.mark.no_open_risk_limit(reason="guards off: liquidation mechanics only (the short must open to be carried)")


@LIFT
def test_with_the_safety_stop_on_the_cases_gap_is_closed_by_the_safety_stop_rebooked_as_the_liquidation(tmp_path, whole_equity):
    """The cases' own +60% gap, safety stop on: the order that closes the short is the safety stop, not the engine's
    own liquidation of a stopless position; filled past the liquidation price, it is re-booked as the liquidation
    (GAP-LIQ)."""
    store, orders = _carried_then(tmp_path, [(5, 0.0), (0, 0.6), (5, 0.0)])
    (closing,) = [o for o in orders if o["side"] == "BUY"]
    # A stopless model has no stop of its own: the only stop there is to hit is the restart's safety stop. It filled
    # on the gap past the liquidation price, so GAP-LIQ re-books it as the liquidation.
    rebooked = [e for e in store.events(NAME, limit=1000) if e["kind"] == "order_rebooked"]
    assert closing["intent"] == "liquidation" and closing["reason"].startswith("Liquidated: the stop filled at"), (
        closing["intent"], closing["reason"])
    assert [e["message"].split(" re-booked")[0] for e in rebooked] == [f"Order {closing['order_id']}"]
    liquidation_incidents = [e for e in store.events(NAME, limit=1000)
                             if e["kind"] == "incident" and "safety stop" not in e["message"]]
    assert len(liquidation_incidents) == 1, liquidation_incidents  # one incident per liquidation
    assert store.journal_book(NAME, 10_000)["qty"] == 0


@LIFT
def test_with_every_guard_but_the_entry_limit_on_a_steady_rise_closes_the_carried_short_long_before_liquidation(tmp_path, whole_equity):
    """No gap: a steady 30% rise. The daily-loss pause (3% of equity, 1.5% of price at 2x) flattens it first, well
    inside the safety stop and the liquidation price: nothing is liquidated."""
    store, orders = _carried_then(tmp_path, [(5, 0.0), (30, 0.3), (5, 0.0)])
    closing = [o for o in orders if o["side"] == "BUY"]
    assert [o["intent"] for o in closing] == ["risk_pause"], orders
    assert store.journal_book(NAME, 10_000)["qty"] == 0
    assert "liquidation" not in _kinds(store) and not store.insurance(NAME)
    assert not (store.sleeve(NAME).status_reason or "").startswith("Position margin lost")
