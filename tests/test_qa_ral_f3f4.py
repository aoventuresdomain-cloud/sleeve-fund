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
    nothing left to reset and marks it, so the journal keeps one liquidation_reset."""
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
