"""QA adversarial probes for P1-RAL (reset after liquidation): the process's own step, ported from the Head of QA round
on #193.

Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/ral-193-scripts/test_qa_ral_193_probes.py (p5b, p6b)
Findings: QA-193-F4 (the process carried out a RAL row with no incident or note: only Store.command checked them) and
QA-193-F3 (the command was taken before liquidation_reset was journaled, so a failure between them used the incident
up and left no way to reset). Both are fixed, so these cells carry no xfail mark.

Built on test_ral_xfails' helpers (a real liquidation replayed into the store, a paper runtime restarted on the
journal).
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import insert

from tests.test_ral_xfails import (  # noqa: F401  (store is a fixture)
    NAME,
    NEXT_DAY,
    RAL,
    _liquidate,
    _noted,
    _ral,
    _ral_events,
    _runtime,
    store,
)


def _raw(store, command, name=NAME, incident=None):
    """A command row written without Store.command's checks: a stale page, a script, a row replayed from a backup."""
    from sleeve_fund.store import commands_t, utcnow

    with store.engine.begin() as c:
        c.execute(insert(commands_t).values(sleeve=name, command=command, reason="QA raw row", created_at=utcnow(),
                                            incident=incident))


def _liquidated(store, name=NAME):
    from sleeve_fund.paper.runtime import liquidation_head

    return liquidation_head(store, name) is not None


def test_p5b_the_process_applies_a_ral_row_only_with_its_noted_incident(store, tmp_path):
    """Defence in depth: Store.command checks the incident and its note, the process doesn't. A RAL row with no
    incident (a restored backup, a script) is carried out, journaled as 'incident #None (note by nobody)'."""
    f = _liquidate(tmp_path, store)
    _raw(store, RAL)
    rt = _runtime(store, NEXT_DAY, f.rem)
    assert rt.status == "halted" and _liquidated(store), [e["message"] for e in _ral_events(store)]


def test_p6b_a_failure_inside_the_ral_step_leaves_a_way_to_reset(store, tmp_path, monkeypatch):
    """The process fails between taking the command (mark_applied) and journaling liquidation_reset (a DB blip or a
    crash): after the restart either the reset is carried out, or the PM can send it again."""
    from sleeve_fund.store import LIQUIDATION_RESET, Store

    f = _liquidate(tmp_path, store)
    iid = _noted(store, f.liq)
    _ral(store, incident=iid)
    real = Store.event

    def _boom(self, sleeve, level, kind, *a, **kw):
        if kind == LIQUIDATION_RESET:
            raise RuntimeError("QA: the process dies here")
        return real(self, sleeve, level, kind, *a, **kw)

    monkeypatch.setattr(Store, "event", _boom)
    with pytest.raises(RuntimeError):
        _runtime(store, NEXT_DAY, f.rem)
    monkeypatch.setattr(Store, "event", real)
    rt = _runtime(store, NEXT_DAY + timedelta(minutes=1), f.rem)
    if rt.status != "running":
        _ral(store, incident=iid)  # the PM sends it again: must be taken
        rt = _runtime(store, NEXT_DAY + timedelta(minutes=2), f.rem)
    assert rt.status == "running" and len(_ral_events(store)) == 1


# --- The fix's own pins (not QA cells) -----------------------------------------------------------------------------

def test_a_ral_row_with_no_noted_incident_is_refused_in_words_and_a_proper_one_still_resets(store, tmp_path):
    """QA-193-F4: the process refuses the raw row with Store.command's own reason, and it doesn't use anything up."""
    f = _liquidate(tmp_path, store)
    _raw(store, RAL)
    _runtime(store, NEXT_DAY, f.rem)
    refused = [e for e in store.events(NAME, limit=50) if e["kind"] == "ral_refused"]
    assert len(refused) == 1 and "name the liquidation's incident" in refused[0]["message"]
    assert not store.pending_commands(NAME) and not _ral_events(store)
    _ral(store, incident=_noted(store, f.liq))
    rt = _runtime(store, NEXT_DAY + timedelta(minutes=1), f.rem)
    assert rt.status == "running" and not _liquidated(store) and len(_ral_events(store)) == 1


def test_a_failure_after_the_reset_is_journaled_does_not_reset_twice(store, tmp_path, monkeypatch):
    """QA-193-F3, the other side: journaled but not yet marked applied, the restart takes the command again, finds
    its reset journaled and marks it, so the journal keeps one liquidation_reset and never calls it ignored (CR on
    #217)."""
    from sleeve_fund.store import Store

    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    real = Store.mark_applied

    def _boom(self, command_id):
        raise RuntimeError("QA: the process dies here")

    monkeypatch.setattr(Store, "mark_applied", _boom)
    with pytest.raises(RuntimeError):
        _runtime(store, NEXT_DAY, f.rem)
    monkeypatch.setattr(Store, "mark_applied", real)
    assert len(store.pending_commands(NAME)) == 1 and len(_ral_events(store)) == 1
    rt = _runtime(store, NEXT_DAY + timedelta(minutes=1), f.rem)
    assert rt.status == "running" and not _liquidated(store)
    assert not store.pending_commands(NAME) and len(_ral_events(store)) == 1
    assert not [e for e in store.events(NAME, limit=50) if e["kind"] in ("ral_ignored", "ral_finished")]


def test_a_failure_after_the_journal_but_before_the_halt_lifts_finishes_the_reset_on_restart(store, tmp_path,
                                                                                               monkeypatch):
    """QA-193-F3, the window PE2 and the HoE named: liquidation_reset journaled, the process dies before the halt is
    lifted. The journal has answered the liquidation, so a further reset would be refused; the restart finishes this
    one instead of leaving the strategy halted for a liquidation nothing can clear."""
    from sleeve_fund.paper.runtime import SleeveRuntime

    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    real = SleeveRuntime._set

    def _boom(self, status, *a, **kw):
        if status == "running":
            raise RuntimeError("QA: the process dies here")
        return real(self, status, *a, **kw)

    monkeypatch.setattr(SleeveRuntime, "_set", _boom)
    with pytest.raises(RuntimeError):
        _runtime(store, NEXT_DAY, f.rem)
    monkeypatch.setattr(SleeveRuntime, "_set", real)
    assert store.sleeve(NAME).status == "halted" and not _liquidated(store) and len(_ral_events(store)) == 1
    rt = _runtime(store, NEXT_DAY + timedelta(minutes=1), f.rem)
    assert rt.status == "running" and store.sleeve(NAME).status == "running"
    assert not store.pending_commands(NAME) and len(_ral_events(store)) == 1


# --- QA F217-1 (Head of QA on #217): both crash windows still lapse a reset asked before the liquidation -------------
# Source: /mnt/project-files/sleeve-fund/quant-review/v2-p2/pr217/test_217_ral_window_probes.py (W0, W1)

def _stale_reset(store):
    """A reset asked for before the liquidation (a page left open, ADV-9): it must lapse when the RAL answers."""
    from sleeve_fund.store import resets_t, utcnow

    with store.engine.begin() as c:
        c.execute(insert(resets_t).values(sleeve=NAME, reason="QA: asked before the liquidation", actor="PM",
                                          restart=1, created_at=utcnow() - timedelta(minutes=5)))


def _open_resets(store):
    return [r for r in store.pending_resets() if r["sleeve"] == NAME]


def test_w0_a_normal_ral_lapses_the_earlier_reset(store, tmp_path):
    f = _liquidate(tmp_path, store)
    _stale_reset(store)
    _ral(store, incident=_noted(store, f.liq))
    rt = _runtime(store, NEXT_DAY, f.rem)
    assert rt.status == "running" and not _open_resets(store)


@pytest.mark.parametrize("window", ["before_halt_lifts", "before_marked_applied"])
def test_w1_a_crash_inside_the_ral_step_still_lapses_the_earlier_reset(store, tmp_path, monkeypatch, window):
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import Store

    f = _liquidate(tmp_path, store)
    _stale_reset(store)
    _ral(store, incident=_noted(store, f.liq))
    if window == "before_halt_lifts":
        real = SleeveRuntime._set
        monkeypatch.setattr(SleeveRuntime, "_set", lambda self, status, *a, **kw: (_ for _ in ()).throw(
            RuntimeError("QA: dies")) if status == "running" else real(self, status, *a, **kw))
        target, attr = SleeveRuntime, "_set"
    else:
        real = Store.mark_applied
        monkeypatch.setattr(Store, "mark_applied", lambda self, cid: (_ for _ in ()).throw(RuntimeError("QA: dies")))
        target, attr = Store, "mark_applied"
    with pytest.raises(RuntimeError):
        _runtime(store, NEXT_DAY, f.rem)
    monkeypatch.setattr(target, attr, real)
    rt = _runtime(store, NEXT_DAY + timedelta(minutes=1), f.rem)
    assert rt.status == "running" and len(_ral_events(store)) == 1
    assert not _open_resets(store), "a reset asked before the liquidation is still pending after the RAL finished"


def test_a_stray_ral_with_nothing_to_reset_lapses_nothing(store, tmp_path):
    """F217-1's limit: only a RAL whose incident the journal has answered lapses earlier resets. One sent again after
    its reset (a stale page) is ignored, as before, and a reset asked since is left to run."""
    f = _liquidate(tmp_path, store)
    _ral(store, incident=(iid := _noted(store, f.liq)))
    _runtime(store, NEXT_DAY, f.rem)
    _stale_reset(store)
    _raw(store, RAL, incident=None)
    _runtime(store, NEXT_DAY + timedelta(minutes=1), f.rem)
    assert len(_open_resets(store)) == 1 and len(_ral_events(store)) == 1
    assert any(e["kind"] == "ral_ignored" for e in store.events(NAME, limit=50)), iid


def test_a_crash_before_the_halt_lifts_is_finished_from_the_journal_whatever_the_halt_now_says(store, tmp_path,
                                                                                                 monkeypatch):
    """PE2 on #217: the recovery is keyed on the liquidation_reset journaled for this command, not on the halt's
    reason, so a later halt that rewrote the reason (here a drawdown halt) can't leave the strategy halted with its
    reset spent. The journal says what happened: one liquidation_reset, then ral_finished, never ral_ignored."""
    from sleeve_fund.paper.runtime import SleeveRuntime

    f = _liquidate(tmp_path, store)
    _stale_reset(store)
    _ral(store, incident=_noted(store, f.liq))
    real = SleeveRuntime._set
    monkeypatch.setattr(SleeveRuntime, "_set", lambda self, status, *a, **kw: (_ for _ in ()).throw(
        RuntimeError("QA: dies")) if status == "running" else real(self, status, *a, **kw))
    with pytest.raises(RuntimeError):
        _runtime(store, NEXT_DAY, f.rem)
    monkeypatch.setattr(SleeveRuntime, "_set", real)
    store.set_status(NAME, "halted", "drawdown 33.1% hit the 25.0% limit")
    rt = _runtime(store, NEXT_DAY + timedelta(minutes=1), f.rem)
    assert rt.status == "running" and store.sleeve(NAME).status == "running"
    assert not store.pending_commands(NAME) and not _open_resets(store) and len(_ral_events(store)) == 1
    kinds = [e["kind"] for e in store.events(NAME, limit=50)]
    assert "ral_finished" in kinds and "ral_ignored" not in kinds
