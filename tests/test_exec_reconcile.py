"""EXEC-CLIENT restart reconcile (PE2, head 3; ships OFF): at start the venue's position and open orders for one account
and instrument are set against the journal's, per Strategy tag. Equal: each tag keeps its journal quantity, exactly. Any
gap holds entries on every Strategy on that account and instrument, alerted, and nothing is ever split or reassigned by
guess (DRILL-VENUE MUST 3). The hold lifts only on a clean re-run, never on a timer.

Recorded shapes: GET /v5/position/list (https://bybit-exchange.github.io/docs/v5/position) and GET /v5/order/realtime
(https://bybit-exchange.github.io/docs/v5/order/open-order), the same fields sleeve_fund.mirror reads on the demo account.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pytest

from sleeve_fund.paper import exec_reconcile as er
from sleeve_fund.store import Store

T0 = datetime(2026, 10, 15, 14, 0, tzinfo=UTC)


def pos(side: str, size: str, idx: int = 0) -> dict:
    return {"symbol": "BTCUSDT", "side": side, "size": size, "positionIdx": idx}


def order(link: str, side: str = "Sell", leaves: str = "0.01", reduce_only: bool = True) -> dict:
    return {"symbol": "BTCUSDT", "orderId": f"v-{link}", "orderLinkId": link, "side": side, "qty": leaves,
            "leavesQty": leaves, "reduceOnly": reduce_only, "orderStatus": "New"}


# --- the venue's figures ---------------------------------------------------------------------------------------------

def test_venue_net_is_signed_by_side_and_flat_reads_zero():
    assert er.venue_net([pos("Buy", "0.015")]) == D("0.015")
    assert er.venue_net([pos("Sell", "0.015")]) == D("-0.015")
    assert er.venue_net([pos("", "0")]) == 0
    assert er.venue_net([]) == 0


def test_a_hedge_mode_position_is_refused_never_netted():
    with pytest.raises(er.Unreadable, match="one-way"):
        er.venue_net([pos("Buy", "0.01", idx=1), pos("Sell", "0.01", idx=2)])


@pytest.mark.parametrize("bad", [[{"symbol": "BTCUSDT", "side": "Buy"}], [pos("Buy", "x")], [pos("Long", "0.01")],
                                 [pos("Buy", "-0.01")]])
def test_a_position_row_that_cannot_be_read_is_refused(bad):
    with pytest.raises(er.Unreadable):
        er.venue_net(bad)


def test_open_orders_key_on_our_order_id():
    assert er.venue_open([order("a1"), order("b2", leaves="0.002")]) == {"a1": D("0.01"), "b2": D("0.002")}


def test_an_open_order_without_our_id_is_refused():
    with pytest.raises(er.Unreadable, match="orderLinkId"):
        er.venue_open([{**order("x"), "orderLinkId": ""}])


# --- Done-when 1: equal means each tag keeps its own quantity ------------------------------------------------------

def test_two_strategies_on_one_account_keep_their_journal_quantities_when_the_venue_net_ties():
    r = er.reconcile("BTC/USDT", {"trend": D("0.02"), "carry": D("-0.005")}, {"t-stop": D("0.02")},
                     [pos("Buy", "0.015")], [order("t-stop", leaves="0.02")])
    assert r.clean
    assert r.gaps == ()
    assert r.per_tag == {"trend": D("0.02"), "carry": D("-0.005")}


def test_flat_everywhere_is_clean():
    r = er.reconcile("BTC/USDT", {"trend": D(0)}, {}, [pos("", "0")], [])
    assert r.clean and r.per_tag == {"trend": D(0)}


# --- Done-when 2: any gap holds, named, and never splits ------------------------------------------------------------

def test_a_net_gap_is_named_and_no_quantity_is_reassigned():
    r = er.reconcile("BTC/USDT", {"trend": D("0.02"), "carry": D("-0.005")}, {}, [pos("Buy", "0.016")], [])
    assert not r.clean
    assert r.per_tag == {}  # never a guessed split
    assert any("0.016" in g and "0.015" in g for g in r.gaps)


def test_a_flipped_side_is_a_gap():
    r = er.reconcile("BTC/USDT", {"trend": D("0.01")}, {}, [pos("Sell", "0.01")], [])
    assert not r.clean


def test_a_venue_order_not_in_the_journal_is_a_gap():
    r = er.reconcile("BTC/USDT", {"trend": D("0.01")}, {}, [pos("Buy", "0.01")], [order("stranger")])
    assert not r.clean
    assert any("stranger" in g for g in r.gaps)


def test_a_journaled_open_order_missing_at_the_venue_is_a_gap():
    r = er.reconcile("BTC/USDT", {"trend": D("0.01")}, {"t-stop": D("0.01")}, [pos("Buy", "0.01")], [])
    assert not r.clean
    assert any("t-stop" in g for g in r.gaps)


def test_an_open_order_with_a_different_remaining_qty_is_a_gap():
    r = er.reconcile("BTC/USDT", {"trend": D("0.01")}, {"t-stop": D("0.01")}, [pos("Buy", "0.01")],
                     [order("t-stop", leaves="0.004")])
    assert not r.clean


def test_unreadable_venue_figures_hold_rather_than_error():
    r = er.reconcile("BTC/USDT", {"trend": D("0.01")}, {}, [pos("Buy", "x")], [])
    assert not r.clean
    assert r.per_tag == {}


# --- the hold in the journal: every tag held, alerted, lifted only by a clean re-run --------------------------------

@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'j.db'}")


def test_a_gap_holds_every_tag_with_an_error_event_and_a_clean_rerun_lifts_it(store):
    bad = er.reconcile("BTC/USDT", {"trend": D("0.01"), "carry": D(0)}, {}, [pos("Buy", "0.02")], [])
    er.record(store, "acct-1", "BTC/USDT", bad, now=T0)
    for tag in ("trend", "carry"):
        why = er.hold(store, tag)
        assert why is not None and "BTC/USDT" in why
        ev = store.last_event(tag, (er.HOLD, er.CLEARED))
        assert ev["kind"] == er.HOLD and ev["level"] == "error"

    # time alone never lifts it
    assert er.hold(store, "trend") is not None

    good = er.reconcile("BTC/USDT", {"trend": D("0.01"), "carry": D(0)}, {}, [pos("Buy", "0.01")], [])
    er.record(store, "acct-1", "BTC/USDT", good, now=T0 + timedelta(hours=6))
    assert er.hold(store, "trend") is None and er.hold(store, "carry") is None


def test_a_clean_run_on_a_tag_never_held_journals_nothing(store):
    good = er.reconcile("BTC/USDT", {"trend": D("0.01")}, {}, [pos("Buy", "0.01")], [])
    er.record(store, "acct-1", "BTC/USDT", good, now=T0)
    assert store.last_event("trend", (er.HOLD, er.CLEARED)) is None


def test_a_journal_that_cannot_be_read_holds(store):
    class Broken:
        def last_event(self, *a, **k):
            raise RuntimeError("down")

    assert er.hold(Broken(), "trend") is not None


def test_the_alert_names_no_venue(store):
    bad = er.reconcile("BTC/USDT", {"trend": D("0.01")}, {}, [pos("Buy", "0.02")], [])
    er.record(store, "acct-1", "BTC/USDT", bad, now=T0)
    msg = store.last_event("trend", (er.HOLD,))["message"].lower()
    assert "bybit" not in msg and "binance" not in msg
