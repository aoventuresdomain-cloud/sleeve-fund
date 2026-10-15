"""A15-F2 (HoQA triage, A15 Done-when 3): a live stop is a journaled stop for the open quantity. A Strategy whose
working stop covers less than the position it holds (an add that never resized it, a partial resize) counts as having
no live stop: one position_without_stop error at 60 s, and position_stop_back once a stop covers it all again."""

from datetime import timedelta

import pytest
from test_ops_health import T0, _holder, _kinds

from sleeve_fund.ops_health import NoStopWatch


def _stop_row(store, oid, qty, minute=0, status="accepted"):
    """The working stop the journal shows for the position (paper's watched stop, or a resting one)."""
    store.record_order("s1", order_id=oid, side="SELL", qty=qty, intent="stop_loss", reason="Stop-loss",
                       signal={"watched": True, "trigger": 59_400.0}, order_type="STOP",
                       ts=T0 + timedelta(seconds=minute * 60 + 1))
    store.update_order(oid, status=status)


def _run(store, watch, *offsets):
    for s in offsets:
        watch.check(store, T0 + timedelta(seconds=s))


@pytest.mark.parametrize("stop_qty", [0.05, 0.099])
def test_f2a_a_stop_for_less_than_the_position_counts_as_missing(tmp_path, stop_qty):
    store = _holder(tmp_path, params={"stop_loss": 0.01}, qty=0.1)
    _stop_row(store, "st1", stop_qty)
    watch = NoStopWatch()
    _run(store, watch, 60, 119)
    assert _kinds(store, "position_without_stop") == []  # 60 s, absolute
    _run(store, watch, 120, 150, 210)
    [e] = _kinds(store, "position_without_stop")
    assert e["level"] == "error" and e["sleeve"] == "s1"


def test_f2b_a_stop_for_the_whole_position_never_alerts(tmp_path):
    store = _holder(tmp_path, params={"stop_loss": 0.01}, qty=0.1)
    _stop_row(store, "st1", 0.1)
    watch = NoStopWatch()
    _run(store, watch, *range(60, 600, 30))
    assert _kinds(store, "position_without_stop") == []


def test_f2c_resized_to_the_whole_position_it_is_back(tmp_path):
    store = _holder(tmp_path, params={"stop_loss": 0.01}, qty=0.1)
    _stop_row(store, "st1", 0.05)
    watch = NoStopWatch()
    _run(store, watch, 60, 130)
    assert len(_kinds(store, "position_without_stop")) == 1
    store.update_order("st1", qty=0.1)  # resized at the venue to the held quantity
    _run(store, watch, 190)
    assert len(_kinds(store, "position_stop_back")) == 1
    _run(store, watch, 250, 400)
    assert len(_kinds(store, "position_without_stop")) == 1


def test_f2d_a_finished_undersized_stop_is_not_the_working_stop(tmp_path):
    """Only a working stop is judged: an old one cancelled when the stop was replaced doesn't make it undersized."""
    store = _holder(tmp_path, params={"stop_loss": 0.01}, qty=0.1)
    _stop_row(store, "old", 0.05, status="canceled")
    _stop_row(store, "st1", 0.1, minute=0)
    watch = NoStopWatch()
    _run(store, watch, *range(60, 400, 30))
    assert _kinds(store, "position_without_stop") == []
