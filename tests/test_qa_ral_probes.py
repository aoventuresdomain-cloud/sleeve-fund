"""QA adversarial probes for P1-RAL (reset after liquidation), ported from the Head of QA round on #193.

Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/ral-193-scripts/test_qa_ral_193_probes.py
Findings: QA-193-F1 (events.kind too short for 'pm_reset_after_liquidation_ignored', Postgres only) and QA-193-F2
(a stopped, liquidated strategy locked: the RAL was sent but never applied, and could not be sent again). Both are
fixed on main, so these cells carry no xfail mark.

Built on test_ral_xfails' helpers (a real liquidation replayed into the store, a paper runtime restarted on the
journal). With TEST_DATABASE_URL set (Postgres) the unique index and the foreign key are enforced too.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert, select

from tests.test_ral_xfails import (  # noqa: F401  (store and client are fixtures)
    NAME,
    NEXT_DAY,
    PARAMS,
    RAL,
    _liquidate,
    _noted,
    _ordinary,
    _post,
    _ral,
    _ral_events,
    _runtime,
    _tick,
    client,
    store,
)

T = datetime(2025, 10, 3, 10, 0, tzinfo=timezone.utc)  # _ordinary's time


def _raw(store, command, name=NAME, incident=None):
    """A command row written without Store.command's checks: a stale page, a script, a row replayed from a backup."""
    from sleeve_fund.store import commands_t, utcnow

    with store.engine.begin() as c:
        c.execute(insert(commands_t).values(sleeve=name, command=command, reason="QA raw row", created_at=utcnow(),
                                            incident=incident))


def _ral_rows(store, name=NAME):
    from sleeve_fund.store import commands_t

    with store.engine.connect() as c:
        return c.execute(select(commands_t).where(commands_t.c.sleeve == name,
                                                  commands_t.c.command == RAL)).all()


def _liquidated(store, name=NAME):
    from sleeve_fund.paper.runtime import liquidation_head

    return liquidation_head(store, name) is not None


# --- 1. a RAL on a strategy that was never liquidated: refused at the call, and a raw row is a no-op -------------

@pytest.mark.parametrize("kind", ["running", "drawdown_halt", "daily_pause"])
def test_p1_ral_on_a_strategy_never_liquidated_is_refused_and_a_raw_row_changes_nothing(store, kind):
    """Source: quant-review/v2-p1/ral-193-scripts/test_qa_ral_193_probes.py; finding QA-193-F1 (fixed)."""
    if kind == "running":
        store.create_sleeve(name=NAME, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, params=PARAMS)
        rt = _runtime(store, T - timedelta(minutes=5), 10_000.0, price=60_000.0)
        equity = 10_000.0
    else:
        rt = _ordinary(store, kind)
        equity = 7_900.0 if kind == "drawdown_halt" else 9_400.0
    s0 = store.sleeve(NAME)
    peak0, day0 = rt.peak, rt._day_open
    iid = _noted(store, at=T + timedelta(minutes=1))
    with pytest.raises(ValueError):
        _ral(store, incident=iid)
    assert not _ral_rows(store)
    _raw(store, RAL, incident=iid)  # past the call's checks
    _tick(rt, T + timedelta(minutes=2), equity, price=60_000.0)
    s = store.sleeve(NAME)
    assert (s.status, s.status_reason, s.paused_until) == (s0.status, s0.status_reason, s0.paused_until)
    assert (rt.peak, rt._day_open) == (peak0, day0)
    assert not _ral_events(store) and not store.pending_commands(NAME)
    rt2 = _runtime(store, T + timedelta(minutes=3), equity, price=60_000.0)  # and through a restart
    assert rt2.status == s0.status


# --- 7. the PM stopped the liquidated strategy: the RAL must not be accepted into a dead end -----------------------

@pytest.mark.parametrize("order", ["stop_then_ral", "ral_then_stop"])
def test_p7_a_stopped_liquidated_strategy_can_still_be_reset_and_started(store, tmp_path, client, order):
    """Stop is the PM's normal move after a liquidation (the RAL master's own retire guard stops it). Whichever came
    first, the PM must not be left with a reset that is 'sent' but never applied and can't be sent again: either the
    RAL is refused in words, or Start then runs it.

    Source: quant-review/v2-p1/ral-193-scripts/test_qa_ral_193_probes.py; finding QA-193-F2 (fixed).
    """
    from sleeve_fund import supervisor

    f = _liquidate(tmp_path, store)
    iid = _noted(store, f.liq)
    stop = {"command": "stop", "reason": "QA: stop it after the liquidation"}
    ral = {"command": RAL, "incident": str(iid), "reason": "incident note written"}
    first, then = (stop, ral) if order == "stop_then_ral" else (ral, stop)
    r1, r2 = _post(client, first), _post(client, then)
    assert "command_error" not in r1.headers["location"], r1.headers["location"]
    sent = "command_error" not in r2.headers["location"] if then is ral else True
    if not sent:
        return  # refused in words: no dead end
    start = _post(client, {"command": "start", "reason": "QA: trade again after the reset"})
    started = []
    sup = supervisor.Supervisor(store)
    sup._start = lambda name, proc: started.append(name)
    sup.step()
    if NAME in started:
        _runtime(store, NEXT_DAY, f.rem)
    assert not _liquidated(store), (start.headers["location"], [c["command"] for c in store.pending_commands(NAME)],
                                    started)


def test_p7c_stopped_and_liquidated_then_the_note_then_ral_then_start_trades_again(store, tmp_path, client):
    """The Code Reviewer's case, all through the page: liquidated, the PM stops it, writes the note, sends Reset after
    liquidation, then presses Start. The reset must be carried out and the strategy run again.

    Source: quant-review/v2-p1/ral-193-scripts/test_qa_ral_193_probes.py; finding QA-193-F2 (fixed).
    """
    from sleeve_fund import supervisor
    from tests.test_ral_xfails import AUTH, AUTHOR, SAME, WHY, _incident

    f = _liquidate(tmp_path, store)
    r = _post(client, {"command": "stop", "reason": "QA: stop it while the incident is looked at"})
    assert "command_error" not in r.headers["location"], r.headers["location"]
    iid = _incident(store, f.liq)
    r = client.post(f"/sleeves/{NAME}/incident-note", data={"incident": str(iid), "author": AUTHOR,
                    "why_stop_did_not_protect": WHY}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert "command_error" not in r.headers["location"], r.headers["location"]
    r = _post(client, {"command": RAL, "incident": str(iid), "reason": "incident note written"})
    assert "command_error" not in r.headers["location"], r.headers["location"]  # "Reset after liquidation sent"
    start = _post(client, {"command": "start", "reason": "QA: trade again after the reset"})
    started = []
    sup = supervisor.Supervisor(store)
    sup._start = lambda name, proc: started.append(name)
    sup.step()
    if NAME in started:
        _runtime(store, NEXT_DAY, f.rem)
    assert not _liquidated(store) and store.sleeve(NAME).status == "running", (
        start.headers["location"], [c["command"] for c in store.pending_commands(NAME)], started)
