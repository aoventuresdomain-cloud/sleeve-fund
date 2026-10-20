"""CR FOLLOW-UP (head 3, HIGH): two ways the A2 late-fill hold never lifted.

- A late fill counted as closed only through its own late close's fills. When the Strategy's book went flat another way
  (its stop-loss fired first, and the reduce-only late close then ended unfilled), the late fill stayed "unflat": the
  Strategy was blocked for good, a fresh close was owed on a flat book on every pass, and a fund-wide hold it opened
  never closed. A flat book now closes it (orphan_guard.late_fills: flat_book).
- A Strategy put away while its process was down stayed "down" for good, so the fund-wide hold its late fill opened
  (unprotected_pass) never lifted. Archiving now journals owner_up: no process runs for it, so none is down."""
from datetime import timedelta

import pytest
from test_orphan_late_fill import DOWN, _closes, _down, _fill

from sleeve_fund.paper import orphan_guard as og


@pytest.fixture
def jstore():
    from test_gate_ledger import _store, _strategies

    s = _store()
    _strategies(s)
    return s


def _stop_fires(store, at, qty=0.02):
    store.record_order("a", order_id="o-stop", side="SELL", qty=qty, intent="stop_loss", reason="stop", ts=at)
    _fill(store, "s1", qty, at, order_id="o-stop", side="SELL")


def test_a_late_fill_the_stop_already_closed_is_closed_and_owes_no_fresh_close(jstore):
    _down(jstore)
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1))
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))
    (close,) = _closes(jstore)
    og.owner_up(jstore, "a", DOWN + timedelta(seconds=20))
    _stop_fires(jstore, DOWN + timedelta(seconds=21))  # the book is flat: the stop got there first
    jstore.update_order(close["order_id"], status="rejected", message="reduce-only: nothing to reduce")
    assert jstore.lifecycle_flat("a")[1]
    assert og.unflat(jstore, "a") == []
    [x] = og.late_fills(jstore, "a")
    assert x["flat_book"] and x["closed"] == x["late"]
    assert og.close_orphans(jstore, DOWN + timedelta(seconds=30)) == []  # no close owed on a flat book
    og.cancel_confirmed(jstore, "o-rest")  # the venue confirms the resting entry's cancel the pass asked for
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=35))  # the fund-wide episode closes on the next pass
    assert og.resting_block(jstore, "a", DOWN + timedelta(seconds=36)) is None
    assert og.resting_block(jstore, "b", DOWN + timedelta(seconds=36)) is None  # nothing held fund-wide


def test_a_book_still_holding_the_late_fill_is_not_closed_by_this(jstore):
    """The control: flat_book only applies when the journal's position is flat."""
    _down(jstore)
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1))
    [x] = og.late_fills(jstore, "a")
    assert not x["flat_book"] and og.unflat(jstore, "a") and og.close_orphans(jstore, DOWN + timedelta(seconds=5))


def test_putting_away_a_strategy_whose_process_is_down_lifts_the_fund_wide_hold_once_flat(jstore):
    _down(jstore)
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1))
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))
    og.unprotected_pass(jstore, DOWN + timedelta(seconds=5))
    assert "fund" in (og.resting_block(jstore, "b", DOWN + timedelta(seconds=6)) or "")
    _stop_fires(jstore, DOWN + timedelta(seconds=8))  # the venue-resting stop closes it while the process is down
    jstore.set_desired_state("a", "stopped")
    jstore.archive("a")
    assert og.down_since(jstore, "a") is None
    og.unprotected_pass(jstore, DOWN + timedelta(minutes=1))
    assert jstore.last_event("a", (og.UNPROTECTED, og.PROTECTED))["kind"] == og.PROTECTED
    assert og.resting_block(jstore, "b", DOWN + timedelta(minutes=1, seconds=1)) is None


def test_putting_away_a_strategy_whose_late_fill_is_still_held_keeps_the_hold(jstore):
    """The control: archived but not flat, nothing guards the late fill, so every Strategy stays held."""
    _down(jstore)
    _fill(jstore, "t1", 0.02, DOWN + timedelta(seconds=1))
    og.supervisor_pass(jstore, DOWN + timedelta(seconds=5))
    og.unprotected_pass(jstore, DOWN + timedelta(seconds=5))
    jstore.set_desired_state("a", "stopped")
    jstore.archive("a")
    og.unprotected_pass(jstore, DOWN + timedelta(minutes=1))
    assert "fund" in (og.resting_block(jstore, "b", DOWN + timedelta(minutes=1, seconds=1)) or "")
