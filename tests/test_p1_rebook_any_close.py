"""T-FIX7, QT2 pin _qa_p1_rebook.py (353ed642), F283-P1 follow-on (found by P1 mutant 4 on T-s5): 38d0f5e4 has _gap_liquidation relabel ANY closing fill at or
past the liquidation price as a liquidation and queue its journal re-book (_rebook), but the journals' _check_rebook
still accepts only a filled stop-loss. Pinned here at the journal, for each close the engine now relabels: the
re-book of a filled close of that intent goes through, the order reads 'liquidation', and an order_rebooked event
keeps the lineage. Red on T means a gapped halt, PM, exit, wind-down or unwind close raises inside on_order_filled
and the liquidation is never journaled as one."""

from __future__ import annotations

import pytest

from sleeve_fund.store import Store

CLOSES = ["stop_loss", "risk_halt", "risk_pause", "pm_flatten", "exit", "leg_unwind", "wind_down"]


@pytest.fixture
def store(tmp_path):
    import os

    url = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db"
    return Store(url)  # Postgres: one database for the module, so each cell has its own Strategy


@pytest.mark.parametrize("intent", CLOSES)
def test_p1_rebook_a_filled_close_of_any_relabelled_intent_is_rebooked_as_a_liquidation(store, intent):
    oid, name = f"qa-{intent}", f"perp_{intent}"
    store.create_sleeve(name=name, strategy="trend_filter", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        params={"market": "perp"}, starting_balance=1000, venue="KRAKEN")
    store.record_order(name, order_id=oid, side="SELL", qty=0.02, intent=intent, reason="QA close",
                       signal={"liquidation_px": 30226.13})
    store.update_order(oid, status="filled", fill_qty=0.02, fill_px=29000.0)
    store.rebook_liquidation(oid, "Liquidated: QA", {"liquidation_px": 30226.13, "relabelled_from": intent})
    (o,) = [o for o in store.orders(name) if o["order_id"] == oid]
    assert o["intent"] == "liquidation"
    assert any(e["kind"] == "order_rebooked" for e in store.events(name, limit=50))
