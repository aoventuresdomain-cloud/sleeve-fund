"""F-WN-1 (HoE 15 Oct, PE2): once the fund holds the cash (Bridge Drop A), a Strategy has no capital of its own, so a
put-away moves nothing. Settled flat, it leaves the book with withdrawn = 0 and an info line; never the
WITHDRAWAL_NONE warning (a false alert to the PM on every archive), and it is in withdrawn(), so the book's checks
(HOLD-STALE-1's positions_breaks) know it is out. Not flat, it still waits for the poll. Before the bridge nothing
changes. SQLite, and Postgres with TEST_DATABASE_URL."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

from test_drop_a_fund_book import MIN, _rebased
from test_halt_ref import T0, _make, _mark, _settle, store  # noqa: F401

from sleeve_fund import supervisor
from sleeve_fund.store import WITHDRAWAL_DEFERRED, WITHDRAWAL_NONE


def _kinds(store, *kinds):  # noqa: F811
    return [e for e in store.events_of(kinds, limit=1_000)]


def _round_trip_marked_mid_way(store, name, at):  # noqa: F811
    """A round trip whose last mark was taken while it held: the mark's 1 is stale once the sell lands."""
    store.record_fill(name, side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id="o1", trade_id="t1", ts=at)
    store.record_equity(name, equity=1_000.0, cash=0.0, qty=1.0, price=1_000.0, benchmark=0.0, ts=at)
    store.record_fill(name, side="SELL", qty=1.0, price=1_050.0, fee=0.0, order_id="o2", trade_id="t2",
                      ts=at + timedelta(seconds=30))


def test_a_a_flat_put_away_after_the_bridge_is_out_at_0_with_no_warning_and_its_stale_mark_never_breaks(store):  # noqa: F811
    _rebased(store)
    _make(store, "y", 0)
    _round_trip_marked_mid_way(store, "y", T0 + MIN)
    _mark(store, T0 + MIN)
    assert supervisor.positions_breaks(store) == []  # the mark's fills, up to its time, are the 1 it carries
    before = supervisor.fund_equity(store)
    store.archive("y")  # out at once: nothing of its own moves, so no mark to wait for (DA condition 1)
    assert store.withdrawn() == {"y": D(0)}
    assert _kinds(store, WITHDRAWAL_NONE) == [] and _kinds(store, "portfolio_flow") == []
    assert supervisor.fund_equity(store) == before
    _mark(store, T0 + 3 * MIN)
    assert store.withdrawn() == {"y": D(0)}
    assert not [e for e in store.events_after(0, limit=1_000) if e["sleeve"] == "y"]  # nothing at warning or above
    [line] = _kinds(store, "put_away_fund_held")
    assert line["level"] == "info" and "y" in line["message"] and "fund holds the cash" in line["message"]
    assert store.withdrawal_pending() == [] and store.nothing_withdrawn() == {}
    # Its P&L stays in the fund's book: nothing moved.
    assert supervisor.fund_book(store).total == D(100_050)
    # HOLD-STALE-1: the stale mark (1 held) of a Strategy that left the book flat never breaks it.
    assert supervisor.positions_breaks(store) == []


def test_b_a_put_away_still_holding_after_the_bridge_waits_for_the_poll_and_goes_once_flat(store):  # noqa: F811
    _rebased(store)
    _make(store, "y", 0)
    store.record_fill("y", side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id="o1", trade_id="t1", ts=T0 + MIN)
    store.record_equity("y", equity=0.0, cash=-1_000.0, qty=1.0, price=1_000.0, benchmark=0.0, ts=T0 + MIN)
    _mark(store, T0 + MIN)
    store.archive("y")
    _mark(store, T0 + 2 * MIN)
    assert "y" not in store.withdrawn() and store.withdrawal_pending() == ["y"]
    assert len(_kinds(store, WITHDRAWAL_DEFERRED)) == 1 and _kinds(store, WITHDRAWAL_NONE) == []
    store.record_fill("y", side="SELL", qty=1.0, price=1_000.0, fee=0.0, order_id="o2", trade_id="t2",
                      ts=T0 + 3 * MIN)
    _settle(store, "y", T0 + 3 * MIN)
    _mark(store, T0 + 4 * MIN)
    assert store.withdrawn() == {"y": D(0)} and _kinds(store, WITHDRAWAL_NONE) == []


def test_c_before_the_bridge_a_put_away_still_takes_its_equity_out_and_0_is_still_none(store):  # noqa: F811
    _make(store, "a", 10_000)
    _settle(store, "a", T0)
    _make(store, "z", 10_000)
    store.record_fill("z", side="BUY", qty=1.0, price=10_000.0, fee=0.0, order_id="o1", trade_id="t1", ts=T0)
    store.record_fill("z", side="SELL", qty=1.0, price=0.0, fee=0.0, order_id="o2", trade_id="t2", ts=T0)
    _settle(store, "z", T0)  # lost everything: 0
    _mark(store, T0)
    store.archive("a")
    store.archive("z")
    _mark(store, T0 + MIN)
    assert store.withdrawn() == {"a": D(10_000)}
    [none] = _kinds(store, WITHDRAWAL_NONE)
    assert none["sleeve"] == "z" and none["level"] == "warning"
    assert _kinds(store, "put_away_fund_held") == []


def test_d_unarchive_after_the_bridge_deletes_the_row_with_no_flow(store):  # noqa: F811
    _rebased(store)
    _make(store, "y", 0)
    _mark(store, T0 + MIN)
    store.archive("y")
    before = supervisor.fund_equity(store)
    store.unarchive("y")
    assert "y" not in store.withdrawn() and "y" not in store.archived()
    assert _kinds(store, "portfolio_flow") == [] and supervisor.fund_equity(store) == before


def test_e_a_non_zero_book_part_under_the_bridge_is_a_broken_invariant_and_still_warns(store, monkeypatch):  # noqa: F811
    _rebased(store)
    _make(store, "y", 0)
    _mark(store, T0 + MIN)
    monkeypatch.setattr(supervisor, "book_part", lambda st, s: D("12.34"))
    store.archive("y")
    assert "y" not in store.withdrawn()
    [w] = _kinds(store, WITHDRAWAL_NONE)
    assert w["level"] == "warning" and "12.34" in w["message"]
    assert _kinds(store, "put_away_fund_held") == []
