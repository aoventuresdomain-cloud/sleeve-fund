"""F-REFRESH-RES end to end (HoE, Drop A SHOULD; PE1's call site on the DA's Store.close_dead_orders). The PM's refresh
puts away a Strategy left with an entry the journal still shows 'accepted'. While its process may still be alive,
nothing is closed and the refresh waits (exit 2). Once the process is confirmed gone (none of the supervisor's alive,
and its single-instance lock free), the supervisor's poll closes the order and releases its room, and the refresh
reaches its deposit."""

from datetime import timedelta
from decimal import Decimal as D

import pytest
from test_halt_ref import T0, _make, _mark, store  # noqa: F401
from test_refresh_res import _reserve

from sleeve_fund import reset_book
from sleeve_fund.paper.single_lock import instance_lock
from sleeve_fund.supervisor import Proc, Supervisor

SLATE = "2026-10-15-reset"


class _Alive:
    def poll(self):
        return None


def _run(store, minute):
    return reset_book.run(store, "operator", "abc1234", SLATE, D(100_000), now=T0 + timedelta(minutes=minute))


@pytest.fixture(autouse=True)
def _locks(tmp_path, monkeypatch):
    monkeypatch.setenv("SLEEVE_LOCK_DIR", str(tmp_path / "locks"))


def test_a_put_away_strategys_accepted_order_is_closed_only_once_its_process_is_gone_then_the_refresh_deposits(store):
    _make(store, "a", 10_000)
    _reserve(store, "a", 1, order_id="o1")  # an entry left resting when its process stopped
    _mark(store, T0)
    code, _ = _run(store, 1)
    assert code == reset_book.WAIT and "a" in store.archived()
    assert store.withdrawal_blocker("a") == "an order of it is still open"

    sup = Supervisor(store, python="true")
    sup.procs["a"] = Proc(popen=_Alive())  # its process may still be running: nothing is closed
    sup.close_dead_orders()
    held = instance_lock("a")  # nor while a process started elsewhere holds its lock
    sup.procs["a"] = Proc()
    sup.close_dead_orders()
    held.close()
    assert store.withdrawal_blocker("a") == "an order of it is still open"
    _mark(store, T0 + timedelta(minutes=2))
    assert _run(store, 3)[0] == reset_book.WAIT

    sup.step()  # the poll, with the process gone: closed, its room released
    assert store.withdrawal_blocker("a") is None
    assert [e["kind"] for e in store.events("a")].count("orders_closed_stopped") == 1
    for minute in range(4, 10):
        _mark(store, T0 + timedelta(minutes=minute, seconds=30))
        code, lines = _run(store, minute + 1)
        if code != reset_book.WAIT:
            break
    assert code == reset_book.DONE, lines
    assert [(f["kind"], D(str(f["amount"]))) for f in store.fund_flows() if f["rebase"]] == [("deposit", D(100_000))]
