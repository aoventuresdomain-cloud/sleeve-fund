"""Own signed call for protective stops (Advisor OWN-STOP-SPEC, STOP-PATH 2a-g, STOP-CLAIM; spec
/mnt/project-files/sleeve-fund/qd/own-stop-path-spec.md). Drill prep, built OFF. paper.venue_stops only SENDS what the
planner (paper.stop_mover) decided: reduce-only conditional market stops for this Strategy's own quantity, with the
last-price trigger explicit, signed with the account's trade-only key, through the shared limiter. Who owns an order
the framework client hands back is decided by the fund's journals: this Strategy's (book), a sibling's (ignore), ours
with no owner (orphan), or not the fund's at all (foreign: CRITICAL once fund-wide, entries on the instrument held)."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

import pytest

from exec_stub import VenueStub
from sleeve_fund.paper.exec_client import Limiter
from sleeve_fund.paper.stop_mover import StopMover, new_order_id
from sleeve_fund.paper.venue_stops import ForeignWatch, OwnershipGate, VenueStops, classify

T0 = datetime(2026, 10, 15, 13, tzinfo=UTC)


@pytest.fixture
def venue():
    v = VenueStub().start()
    yield v
    v.stop()


@pytest.fixture
def stops(venue):
    calls = []

    class Spy(Limiter):
        def take(self, endpoint, *, opening, since=None):
            calls.append((endpoint, opening))
            return super().take(endpoint, opening=opening, since=since)

    s = VenueStops(venue.http_url, symbol="BTCUSDT", api_key="stub-key", api_secret="stub-secret", limiter=Spy(),
                   now_ms=lambda: 1792066007495)
    s.limiter_calls = calls
    return s


# sending -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("side, direction", [("SELL", 2), ("BUY", 1)])
def test_a_a_stop_is_reduce_only_for_its_own_quantity_with_the_last_price_trigger_sent_explicitly(
        venue, stops, side, direction):
    oid = new_order_id()
    assert stops.place(oid, side=side, qty="0.010", level="60000.0") == ("accepted", venue.orders[oid]["orderId"])
    [body] = venue.bodies("order/create")
    assert body == {"category": "linear", "symbol": "BTCUSDT", "side": side.title(), "orderType": "Market",
                    "qty": "0.010", "triggerPrice": "60000.0", "triggerBy": "LastPrice",
                    "triggerDirection": direction, "reduceOnly": True, "positionIdx": 0, "orderLinkId": oid}
    assert "closeOnTrigger" not in body and "tpslMode" not in body  # never a whole-position stop (STOP-PATH 2c)


def test_b_every_call_is_signed_with_the_venue_s_scheme_and_never_carries_the_secret(venue, stops):
    oid = new_order_id()
    stops.place(oid, side="SELL", qty="0.010", level="60000.0")
    method, path, body = venue.seen[-1]
    h = venue.headers[-1]
    assert h["X-BAPI-API-KEY"] == "stub-key" and h["X-BAPI-TIMESTAMP"] == "1792066007495"
    want = hmac.new(b"stub-secret", f"1792066007495stub-key{h['X-BAPI-RECV-WINDOW']}{body}".encode(),
                    hashlib.sha256).hexdigest()
    assert h["X-BAPI-SIGN"] == want
    assert "stub-secret" not in body and "stub-secret" not in json.dumps(h)


def test_c_every_call_goes_through_the_limiter_as_a_protective_order(venue, stops):
    oid = new_order_id()
    stops.place(oid, side="SELL", qty="0.010", level="60000.0")
    stops.lookup(oid)
    stops.cancel(oid)
    assert stops.limiter_calls == [("order/create", False), ("order/realtime", False), ("order/cancel", False)]


def test_d_only_the_fund_s_own_ids_are_ever_sent(stops):
    with pytest.raises(ValueError):
        stops.place("by-hand-1", side="SELL", qty="0.010", level="60000.0")
    with pytest.raises(ValueError):
        stops.place("sf-" + "0" * 40, side="SELL", qty="0.010", level="60000.0")


def test_e_the_read_back_returns_the_venue_s_row_and_cancel_reaches_it(venue, stops):
    oid = new_order_id()
    stops.place(oid, side="SELL", qty="0.010", level="60000.0")
    row = stops.lookup(oid)
    assert row["orderLinkId"] == oid and row["triggerBy"] == "LastPrice" and row["triggerPrice"] == "60000.0"
    assert stops.cancel(oid) == ("accepted", venue.orders[oid]["orderId"])
    assert venue.orders[oid]["orderStatus"] == "Cancelled"


def test_f_a_refusal_comes_back_with_its_code_for_the_reject_class(venue, stops):
    assert stops.cancel(new_order_id())[0:2] == ("rejected", 110001)


def test_g_no_answer_is_unknown_never_assumed(stops):
    dead = VenueStops("http://127.0.0.1:9", symbol="BTCUSDT", api_key="k", api_secret="s", limiter=Limiter(),
                      timeout_s=0.5)
    assert dead.place(new_order_id(), side="SELL", qty="0.010", level="60000.0")[0] == "unknown"


# whose order is it (OWN-STOP-SPEC MUST 1, STOP-CLAIM) ------------------------------------------------------------
@pytest.mark.parametrize("oid, want", [
    ("sf-mine", "book"), ("sf-sibling", "sibling"), ("sf-nobody", "orphan"), ("by-hand-1", "foreign")])
def test_h_whose_order_by_the_fund_s_journals(oid, want):
    assert classify(oid, own={"sf-mine"}, fund={"sf-mine", "sf-sibling"}) == want


def test_i_two_strategies_on_one_instrument_each_book_only_their_own_a_sibling_is_silent():
    fund = {"sf-a1", "sf-b1"}
    a, b = OwnershipGate(own={"sf-a1"}), OwnershipGate(own={"sf-b1"})
    assert [a.on_claimed("sf-a1", fund, T0), a.on_claimed("sf-b1", fund, T0)] == ["book", "ignore"]
    assert [b.on_claimed("sf-a1", fund, T0), b.on_claimed("sf-b1", fund, T0)] == ["ignore", "book"]


def test_j_a_fill_whose_journal_row_is_not_visible_yet_is_held_then_rechecked_never_called_foreign_at_once():
    g = OwnershipGate(own=set(), grace=timedelta(seconds=2))
    assert g.on_claimed("sf-late", fund=set(), now=T0) == "hold"
    g.own.add("sf-late")  # the journal row lands
    assert g.recheck(fund={"sf-late"}, now=T0 + timedelta(seconds=1)) == [("sf-late", "book")]
    g2 = OwnershipGate(own=set(), grace=timedelta(seconds=2))
    g2.on_claimed("sf-ghost", fund=set(), now=T0)
    assert g2.recheck(fund=set(), now=T0 + timedelta(seconds=1)) == []  # still within the grace
    assert g2.recheck(fund=set(), now=T0 + timedelta(seconds=3)) == [("sf-ghost", "orphan")]


def test_k_a_foreign_order_is_critical_once_fund_wide_and_holds_entries_on_its_instrument_until_gone_and_tied():
    alerted = set()
    w = ForeignWatch(already_alerted=lambda oid: oid in alerted, mark_alerted=alerted.add)
    acts = w.on_foreign("by-hand-1", "BTCUSDT", status="Untriggered")
    assert ("hold", "BTCUSDT") in acts
    [alert] = [a for a in acts if a[0] == "alert"]
    assert alert[1] == "critical" and "foreign order on the account" in alert[2] and "BTCUSDT" in alert[2]
    other = ForeignWatch(already_alerted=lambda oid: oid in alerted, mark_alerted=alerted.add)  # a second process
    assert [a for a in other.on_foreign("by-hand-1", "BTCUSDT", status="Untriggered") if a[0] == "alert"] == []
    assert w.held() == {"BTCUSDT"}
    assert w.on_gone("by-hand-1", reconcile_ties=False) == []  # gone, but the account doesn't tie yet: still held
    assert w.on_gone("by-hand-1", reconcile_ties=True) == [("release", "BTCUSDT")]
    assert w.held() == set()


def test_l_a_filled_foreign_order_is_never_booked_and_holds_until_the_reconcile_ties():
    w = ForeignWatch(already_alerted=lambda oid: False, mark_alerted=lambda oid: None)
    acts = w.on_foreign("by-hand-2", "BTCUSDT", status="Filled")
    assert not [a for a in acts if a[0] in ("book", "cancel")]
    assert ("hold", "BTCUSDT") in acts
    assert w.on_gone("by-hand-2", reconcile_ties=True) == [("release", "BTCUSDT")]


# restart, both directions (OWN-STOP-SPEC MUST 3) -------------------------------------------------------------------
def test_m_a_restart_places_a_missing_stop_at_once_at_the_safety_level_and_alerts():
    acts = StopMover.reconcile([], journal_current=None, holding="0.010", side="SELL", safety_level="57000.0")
    [place] = [a for a in acts if a[0] == "place"]
    assert place[2:] == ("57000.0", "0.010") and place[1].startswith("sf-")
    assert any(a[0] == "alert" for a in acts)


def test_n_a_restart_with_its_stop_resting_places_nothing():
    assert StopMover.reconcile(["sf-cur"], journal_current="sf-cur", holding="0.010", side="SELL",
                               safety_level="57000.0") == []
