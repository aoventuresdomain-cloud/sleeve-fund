"""Per-Strategy attribution (Bridge Drop A): money P&L of the closed trades, R per trade, contribution to the fund's
return over the book at the period's start, and each refused or cut idea with the name of the test that did it."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

from sqlalchemy import insert
from test_dashboard import (  # noqa: F401 - the fixtures
    AUTH,
    _new,
    _strategy_accounts,
    client,
)

from sleeve_fund.store import book_marks_t, gate_decisions_t, sleeves_t, utcnow


def _traded(c, store):
    from sleeve_fund.paper.runtime import SleeveRuntime

    _new(c, stop_loss_pct="3")
    rt = SleeveRuntime(store, "btc-test")
    rt.on_order(order_id="E-1", side="BUY", qty=0.1, intent="entry", reason="Signal to be long",
                signal={"stop_frac": 0.02, "risk_amount": 100.0, "stop_cfg": {"stop_loss": 0.02}})
    rt.on_fill(side="BUY", qty=0.1, price=50_000.0, fee=1.0, order_id="E-1", trade_id="T-1")
    rt.on_order(order_id="X-1", side="SELL", qty=0.1, intent="exit", reason="Signal to be flat", signal={})
    rt.on_fill(side="SELL", qty=0.1, price=51_000.0, fee=1.0, order_id="X-1", trade_id="T-2")
    now = utcnow()
    with store.engine.begin() as conn:
        conn.execute(insert(book_marks_t), [{
            "ts": now - timedelta(days=40), "equity": D(10000), "gross": D(0), "net_max": D(0), "margin_used": D(0),
            "open_risk": D(0), "hwm": D(10000), "reference_equity": D(10000), "day_start_equity": D(10000),
            "status": "ok", "profile_version": 1}])
        sid = conn.execute(sleeves_t.select().where(sleeves_t.c.name == "btc-test")).first().id
        base = {"sleeve_id": sid, "bar_ts": now, "intent": "entry", "underlying": "BTC", "profile_version": 1,
                "price": D(50000), "stage": "submit"}
        conn.execute(insert(gate_decisions_t), [
            {**base, "seq": 1, "intent_id": "i1", "outcome": "rejected", "limit_hit": "net_instrument",
             "requested_qty": D("0.4"), "approved_qty": D(0), "decided_at": now - timedelta(minutes=3)},
            {**base, "seq": 2, "intent_id": "i2", "outcome": "trimmed", "limit_hit": "open_risk",
             "requested_qty": D("0.6"), "approved_qty": D("0.45"), "decided_at": now - timedelta(minutes=2)},
            {**base, "seq": 3, "intent_id": "i3", "outcome": "approved", "limit_hit": None,
             "requested_qty": D("0.1"), "approved_qty": D("0.1"), "decided_at": now - timedelta(minutes=1)},
        ])


def test_the_strategy_shows_its_pnl_r_contribution_and_refused_ideas_by_test(client):  # noqa: F811
    c, store = client
    _traded(c, store)
    from sleeve_fund.dashboard import attribution, trading

    trips = trading.trips(store.fills("btc-test", limit=1000), store.events("btc-test", limit=1000),
                          trading.orders_by_id(store, "btc-test"), store.exit_plans("btc-test"))
    a = attribution.strategy(store, "btc-test", trips, utcnow())
    assert a["trades"] == 1 and round(a["pnl"], 2) == 98.0  # 0.1 x 1,000 less 2 of fees
    assert round(a["r_mean"], 4) == 0.98  # 98 over the 100 it risked at its stop
    assert a["fund_start"] == 10_000 and round(a["contribution"], 6) == 0.0098
    assert [(r["outcome"], r["test"]) for r in a["refused"]] == [("trimmed", "T3 Whole fund"),
                                                                ("rejected", "T4 Net per asset")]
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    panel = page.split('id="attr-h"')[1].split('id="tab-activity"')[0]  # the summary on Trades, the list on Refused ideas
    assert "+0.98R" in panel and "+0.98%" in panel and "98.00" in panel
    assert "Refused" in panel and "Trimmed to 0.45"  # TEC-FE-C6 in panel and "T4 Net per asset" in panel and "T3 Whole fund" in panel


def test_a_period_counts_only_its_own_trades_and_refusals(client):  # noqa: F811
    c, store = client
    _traded(c, store)
    from sleeve_fund.dashboard import attribution

    later = utcnow() + timedelta(days=2)
    a = attribution.strategy(store, "btc-test", [], later, "today")
    assert a["trades"] == 0 and a["refused"] == [] and a["r_mean"] is None and a["contribution"] == 0.0


def test_before_the_books_first_mark_the_contribution_reads_as_a_dash(client):  # noqa: F811
    c, store = client
    _new(c)
    from sleeve_fund.dashboard import attribution

    a = attribution.strategy(store, "btc-test", [], utcnow())
    assert a["fund_start"] is None and a["contribution"] is None
    panel = c.get("/sleeves/btc-test", auth=AUTH).text.split('id="tab-refused"')[1]
    assert "No idea was refused or cut in this period." in panel
