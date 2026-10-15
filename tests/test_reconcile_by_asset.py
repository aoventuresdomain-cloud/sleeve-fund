"""NIGHTLY-RECON by (Strategy, asset) (Drop B, PE2; DA 0028 MUSTs; HoQA b4/b4b): every journal row's position rebuilt
from its fills against what its orders record, exactly, keyed (COALESCE(parent, name), the row's instrument), a
carry's second leg by its contract too. SQLite, and Postgres with TEST_DATABASE_URL."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

from sqlalchemy import text
from test_halt_ref import T0, store  # noqa: F401

from sleeve_fund import nightly_recon, reconcile
from sleeve_fund.store import fills_t, sleeve_archive_t

MIN = timedelta(minutes=1)
BAR = "1-MINUTE-LAST-INTERNAL"
PERP = {"market": "perp", "allow_short": True, "stop_loss": 0.02}


def _fund(store):
    if not store.fund_flows():
        at = T0 - timedelta(hours=1)
        store.settle_flow(store.fund_flow("deposit", D(100_000), actor="pm", reason="QA", rebase=True, now=at), at)


def _multi(store, name="m", assets=("BTC/USDT", "ETH/USDT", "SOL/USDT")):
    _fund(store)
    if len(assets) == 1:
        store.create_sleeve(name=name, strategy="ping_pong", instrument=assets[0], bar_spec=BAR, starting_balance=0,
                            venue="BINANCE", desired_state="running", params=dict(PERP))
        return
    store.create_legs(name=name, strategy="ping_pong", bar_spec=BAR, assets=list(assets), params=dict(PERP),
                      starting_balance=0, universe_date="2026-10-15", venue="BINANCE", desired_state="running")


def _carry(store):
    store.create_two_legs(name="carry", strategy="ping_pong", bar_spec=BAR, starting_balance=10_000,
                          first={"instrument": "BTC/USDT", "params": {}, "venue": "BINANCE", "side": 1},
                          second={"instrument": "BTC/USDT", "params": {"market": "perp"}, "venue": "BINANCE",
                                  "side": -1}, desired_state="stopped")


def _fill(store, row, side, qty, oid, *, order=True, at=T0):
    if order:
        store.record_order(row, order_id=oid, side=side, qty=qty, intent="entry", reason="r", ts=at)
    store.record_fill(row, side=side, qty=qty, price=100.0, fee=0.0, order_id=oid, trade_id=f"t-{oid}", ts=at)
    if order:
        store.update_order(oid, status="filled")


def _tamper(store, oid, qty):
    """A fill row's quantity changed behind the journal's back, at its exact scale."""
    with store.engine.begin() as c:
        c.execute(fills_t.update().where(fills_t.c.order_id == oid).values(qty=D(str(qty))))


def _rows(found):
    return [(f.strategy, f.asset, f.row) for f in found]


# --- DA MUST 1: a carry's legs trade one symbol, so the second leg's key carries its contract -------------------------


def test_must1_a_carrys_two_legs_are_two_keys_and_never_net_to_0(store):  # noqa: F811
    _carry(store)
    _fill(store, "carry", "BUY", 1.0, "c1")
    _fill(store, "carry~2", "SELL", 1.0, "c2")
    want = {("carry", "BTC/USDT"): D(1), ("carry", "BTC/USDT (perp)"): D(-1)}
    assert reconcile.rebuild_positions(store) == want == store.positions_by_asset()
    assert reconcile.nightly(store) == []


def test_must1_a_broken_carry_leg_is_named_by_its_own_row(store):  # noqa: F811
    _carry(store)
    _fill(store, "carry", "BUY", 1.0, "c1")
    _fill(store, "carry~2", "SELL", 1.0, "c2")
    _tamper(store, "c2", 0.5)
    assert _rows(reconcile.nightly(store)) == [("carry", "BTC/USDT (perp)", "carry~2")]


# --- DA MUST 2: a multi-asset parent is its first asset, never the sum of its legs ---------------------------------------


def test_must2_the_parent_is_keyed_by_its_first_asset_and_its_legs_are_never_summed_into_it(store):  # noqa: F811
    _multi(store)
    _fill(store, "m", "BUY", 0.1, "m1")
    _fill(store, "m~ETH", "SELL", 2.0, "m2")
    _fill(store, "m~SOL", "BUY", 10.0, "m3")
    got = reconcile.rebuild_positions(store)
    assert got == {("m", "BTC/USDT"): D("0.1"), ("m", "ETH/USDT"): D(-2), ("m", "SOL/USDT"): D(10)}
    assert got == store.positions_by_asset()


def test_must2_a_tampered_leg_is_named_on_that_asset_and_the_parents_key_is_clean(store):  # noqa: F811
    _multi(store)
    _fill(store, "m", "BUY", 0.1, "m1")
    _fill(store, "m~SOL", "BUY", 10.0, "m3")
    _tamper(store, "m3", 9.0)
    [f] = reconcile.nightly(store)
    assert (f.strategy, f.asset, f.row, f.rebuilt, f.stored) == ("m", "SOL/USDT", "m~SOL", D(9), D(10))
    assert "m, SOL/USDT (row m~SOL): its fills come to 9 but its orders record 10, -1 apart" == str(f)


# --- DA MUST 3: exact, to the last place, no tolerance ------------------------------------------------------------------


def test_must3_a_gap_of_one_unit_in_the_last_place_the_journal_keeps_is_a_break(store):  # noqa: F811
    """The eighteenth place on Postgres (NUMERIC(38, 18)); SQLite keeps a REAL (DA-9), so a ninth-place gap there."""
    gap = D("1e-18") if store.engine.dialect.name == "postgresql" else D("1e-9")
    _multi(store, "e", ("BTC/USDT",))
    _fill(store, "e", "BUY", 0.1, "e1")
    _tamper(store, "e1", D("0.1") + gap)
    [f] = reconcile.nightly(store)
    assert f.rebuilt - f.stored == gap and isinstance(f.rebuilt, D) and isinstance(f.stored, D)


def test_must3_many_fills_sum_exactly_with_no_float_noise(store):  # noqa: F811
    """Ten fills of 0.1 against one order of 1: exactly equal, though ten floats of 0.1 sum to 0.9999999999999999."""
    _multi(store, "f", ("BTC/USDT",))
    store.record_order("f", order_id="f1", side="BUY", qty=1.0, intent="entry", reason="r", ts=T0)
    for i in range(10):
        store.book_fill("f", side="BUY", qty=0.1, price=100.0, fee=0.0, order_id="f1", trade_id=f"t{i}", ts=T0)
    assert reconcile.rebuild_positions(store)[("f", "BTC/USDT")] == D(1) == store.positions_by_asset()[("f", "BTC/USDT")]
    assert reconcile.nightly(store) == []


# --- the rows covered, the stored side, the DA's reader -----------------------------------------------------------------


def test_a_run_put_away_counts_and_a_saved_backtest_does_not(store):  # noqa: F811
    from sleeve_fund.store import BACKTEST_PREFIX

    _multi(store, "r", ("BTC/USDT",))
    _fill(store, "r", "BUY", 0.1, "r1")
    with store.engine.begin() as c:  # put away, not flat: exactly what the night is for
        c.execute(sleeve_archive_t.insert().values(sleeve="r", archived_at=T0 + MIN, withdrawn=0))
    bt = BACKTEST_PREFIX + "x"
    store.create_sleeve(name=bt, strategy="ping_pong", instrument="BTC/USDT", bar_spec=BAR, starting_balance=1_000,
                        venue="BINANCE")
    _fill(store, bt, "BUY", 1.0, "bt1")
    _tamper(store, "bt1", 0.5)
    _tamper(store, "r1", 0.2)
    assert ("r", "BTC/USDT") in reconcile.rebuild_positions(store)
    assert not any(k[0] == bt for k in reconcile.rebuild_positions(store))
    assert _rows(reconcile.nightly(store)) == [("r", "BTC/USDT", "r")]


def test_a_fill_with_no_order_row_counts_on_both_sides_and_is_listed_not_a_gap(store):  # noqa: F811
    _multi(store, "u", ("BTC/USDT",))
    _fill(store, "u", "BUY", 0.1, "u1", order=False)
    assert reconcile.nightly(store) == [] and reconcile.unanchored(store) == {"u": 1}
    assert store.positions_by_asset()[("u", "BTC/USDT")] == D("0.1")


def test_an_order_booked_by_the_runtime_is_stored_by_its_filled_qty(store):  # noqa: F811
    _multi(store, "b", ("BTC/USDT",))
    store.record_order("b", order_id="b1", side="BUY", qty=0.3, intent="entry", reason="r", ts=T0)
    store.book_fill("b", side="BUY", qty=0.1, price=100.0, fee=0.0, order_id="b1", trade_id="t1", ts=T0)
    store.update_order("b1", status="canceled")  # part filled, the rest cancelled
    assert store.positions_by_asset()[("b", "BTC/USDT")] == D("0.1") and reconcile.nightly(store) == []


def test_a_row_written_under_another_asset_is_a_finding(store):  # noqa: F811
    _multi(store)
    _fill(store, "m~ETH", "BUY", 1.0, "x1")
    with store.engine.begin() as c:
        c.execute(text("UPDATE orders SET instrument = 'SOL/USDT' WHERE order_id = 'x1'"))
    found = reconcile.nightly(store)
    assert [(f.kind, f.row) for f in found] == [("instrument", "m~ETH")]
    assert "orders row" in found[0].what and "written under SOL/USDT" in found[0].what


def test_the_night_fails_on_a_tampered_position_and_names_it(store, tmp_path):  # noqa: F811
    _multi(store)
    _fill(store, "m", "BUY", 0.1, "m1")
    _fill(store, "m~ETH", "BUY", 2.0, "m2")
    _tamper(store, "m2", 1.0)
    url = store.engine.url.render_as_string(hide_password=False)
    store.engine.dispose()
    ok, path, fails = nightly_recon.run(tmp_path / "recon", url, T0 + timedelta(days=1))
    assert not ok and "m, ETH/USDT (row m~ETH): its fills come to 1 but its orders record 2, -1 apart" in fails
    assert "Positions by Strategy and asset:" in path.read_text()


def test_a_clean_night_passes_the_positions_section(store, tmp_path):  # noqa: F811
    _multi(store)
    _fill(store, "m", "BUY", 0.1, "m1")
    url = store.engine.url.render_as_string(hide_password=False)
    store.engine.dispose()
    _, path, fails = nightly_recon.run(tmp_path / "recon", url, T0 + timedelta(days=1))
    assert not [f for f in fails if "orders record" in f]
    assert "every position its fills come to is what its orders record" in path.read_text()



def test_two_rows_on_one_key_are_reported_never_one_hidden_behind_the_other(store, monkeypatch):  # noqa: F811
    """0028's asset index rules this out; if it ever happens, the night names both rows and the keyed reader refuses."""
    import pytest

    _multi(store, "k", ("BTC/USDT",))
    _multi(store, "j", ("BTC/USDT",))
    real = reconcile._rows
    monkeypatch.setattr(reconcile, "_rows", lambda c: {r: ("k", "BTC/USDT") for r in real(c)})
    found = [f for f in reconcile.nightly(store) if f.kind == "key"]
    assert [f.what for f in found] == ["k, BTC/USDT: rows j and k share one key, so neither may stand for it"]
    with pytest.raises(ValueError, match="share a"):
        store.positions_by_asset()


def test_more_mismatches_than_a_night_lists_says_so(store, monkeypatch):  # noqa: F811
    _multi(store)
    row = {"table": "fills", "id": 1, "sleeve": "m~ETH", "instrument": "SOL/USDT", "expected": "ETH/USDT"}
    monkeypatch.setattr(reconcile, "MISMATCHES", 2)
    monkeypatch.setattr(type(store), "asset_mismatches", lambda self, limit=50: [row] * limit)
    found = reconcile.nightly(store)
    assert len(found) == 3 and "more than 2 journal rows are written under another asset" in found[0].what
