"""A reset after liquidation on a strategy the PM has stopped (CR on #193): no process runs for a stopped, flat
strategy, so the store applies it in the command's own transaction, as the engine would; it is then no longer
liquidated and can be started, and a restart keeps the new high-water mark. Uses QA's RAL set-up
(test_ral_xfails)."""

import pytest

from sleeve_fund.dashboard import trading
from sleeve_fund.paper.runtime import liquidation_head
from tests.test_ral_xfails import (NAME, NEXT_DAY, _liquidate, _noted, _post, _ral, _ral_events, _runtime,  # noqa: F401
                                   client, store)


def test_a_stopped_liquidated_strategy_given_a_ral_is_no_longer_liquidated_and_can_be_started(store, client, tmp_path):
    f = _liquidate(tmp_path, store)
    store.set_desired_state(NAME, "stopped")
    iid = _noted(store, f.liq)
    _ral(store, incident=iid)
    assert not store.pending_commands(NAME), store.pending_commands(NAME)  # applied here, not left pending
    assert liquidation_head(store, NAME) is None and not trading.liquidated_since_reset(store, NAME)
    assert len(_ral_events(store)) == 1 and f"{f.rem:,.2f}" in _ral_events(store)[0]["message"]
    s = store.sleeve(NAME)
    assert (s.status, s.desired_state) == ("stopped", "stopped"), (s.status, s.status_reason)
    with pytest.raises(ValueError, match="already reset for this liquidation|isn't halted after a liquidation"):
        _ral(store, incident=iid)  # answered once
    r = _post(client, {"command": "start", "reason": "testing: start it again after the reset after liquidation"})
    assert r.status_code == 303 and "error" not in r.headers["location"].lower(), r.headers["location"]
    assert store.sleeve(NAME).desired_state == "running"
    rt = _runtime(store, NEXT_DAY, f.rem)  # the process the supervisor starts: the new mark, not halted
    assert rt.peak == pytest.approx(f.rem, abs=0.01) and rt.liquidated is None
    assert rt.status != "halted", store.sleeve(NAME).status_reason


def test_a_ral_for_a_running_strategy_is_left_to_its_process(store, tmp_path):
    """The process applies it, as before: the store only queues it."""
    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    assert [c["command"] for c in store.pending_commands(NAME)] == ["reset_after_liquidation"]
    assert not _ral_events(store) and liquidation_head(store, NAME) is not None


def test_a_stop_after_the_ral_applies_it_rather_than_dropping_it(store, client, tmp_path):
    """QA RAL-F2, the other order: the RAL is sent while it runs, then the PM stops it before its process applied it.
    Stop drops waiting commands, but not this one (its incident can't be answered twice): it is applied, so it can
    be started."""
    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    r = _post(client, {"command": "stop", "reason": "testing: stopped before its process applied the reset"})
    assert r.status_code == 303 and "error" not in r.headers["location"].lower(), r.headers["location"]
    assert not store.pending_commands(NAME) and liquidation_head(store, NAME) is None
    assert len(_ral_events(store)) == 1
    r = _post(client, {"command": "start", "reason": "testing: start it again after the reset after liquidation"})
    assert r.status_code == 303 and "error" not in r.headers["location"].lower(), r.headers["location"]
    rt = _runtime(store, NEXT_DAY, f.rem)
    assert rt.peak == pytest.approx(f.rem, abs=0.01) and rt.liquidated is None and rt.status != "halted"
