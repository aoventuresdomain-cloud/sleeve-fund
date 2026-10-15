"""F-RECON-DUST (HoE 15 Oct 09:20 UK; HoQA master v2-p2/drop-3a/test_f_recon_dust_qa.py): the first read-only nightly
recon on prod failed falsely on an 18-place residue (-2e-18) in a stopped row's orders' filled_qty, written before the
journal kept quantities exact (DA-9, #207). The two sides are rounded to 8 dp and then compared exactly (DA MUST 3: no
tolerance band), so the residue is equal and a gap of one 8-dp step still breaks. Today's writer sums exact Decimals,
so new rows carry no residue (pinned). SQLite rounds an 18-place value away; Postgres (TEST_DATABASE_URL) proves it."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

from test_halt_ref import T0, store  # noqa: F401

from sleeve_fund import reconcile
from sleeve_fund.store import orders_t


def _row(store, name="dust"):  # noqa: F811
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", desired_state="stopped")


def _trade(store, side, qty, oid, at=T0):  # noqa: F811
    store.record_order("dust", order_id=oid, side=side, qty=qty, intent="entry", reason="r", ts=at)
    store.record_fill("dust", side=side, qty=qty, price=100.0, fee=0.0, order_id=oid, trade_id=f"t-{oid}", ts=at)
    store.update_order(oid, status="filled")


def _filled(store, oid, qty):  # noqa: F811
    with store.engine.begin() as c:
        c.execute(orders_t.update().where(orders_t.c.order_id == oid).values(filled_qty=D(qty)))


def _breaks(store):  # noqa: F811
    return [(f.row, f.kind) for f in reconcile.nightly(store)]


def test_a_the_prod_residue_on_a_flat_row_is_no_break(store):  # noqa: F811
    _row(store)
    _trade(store, "BUY", 0.1, "b1")
    _trade(store, "SELL", 0.1, "s1", at=T0 + timedelta(minutes=1))
    _filled(store, "s1", "0.100000000000000002")
    assert _breaks(store) == []


def test_b_the_residue_on_a_held_row_is_no_break(store):  # noqa: F811
    _row(store)
    _trade(store, "BUY", 0.1, "b1")
    _filled(store, "b1", "0.100000000000000002")
    assert _breaks(store) == []


def test_c_one_8dp_step_and_one_lot_still_break_with_the_exact_figures_said(store):  # noqa: F811
    _row(store)
    _trade(store, "BUY", 0.1, "b1")
    _filled(store, "b1", "0.10000001")
    [f] = reconcile.nightly(store)
    assert (f.row, f.kind) == ("dust", "position") and "0.10000001" in f.what
    _filled(store, "b1", "0.101")
    assert _breaks(store) == [("dust", "position")]


def test_d_new_partial_fills_carry_no_residue(store):  # noqa: F811
    _row(store)
    store.record_order("dust", order_id="o", side="BUY", qty=0.3, intent="entry", reason="r", ts=T0)
    for i, q in enumerate((0.1, 0.2)):  # 0.1 + 0.2 is 0.30000000000000004 as floats
        store.record_fill("dust", side="BUY", qty=q, price=100.0, fee=0.0, order_id="o", trade_id=f"t{i}", ts=T0)
        store.update_order("o", fill_qty=q, fill_px=100.0)
    with store.engine.connect() as c:
        assert D(str(c.execute(orders_t.select()).first().filled_qty)) == D("0.3")
    assert reconcile.rebuild_positions(store) == reconcile.stored_positions(store)
