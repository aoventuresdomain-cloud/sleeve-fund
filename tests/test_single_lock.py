"""SINGLE-LOCK (Independent Quant Advisor, 13 Oct 2026 18:10 UK): one paper process per strategy. A second start stops
before it connects to market data or the venue, journals "duplicate refused" once and exits 75; the supervisor never
reads that exit as the strategy's own process going down."""
from __future__ import annotations

import pytest

from sleeve_fund.paper import single_lock as sl


def test_the_lock_is_held_by_one_process_at_a_time_and_freed_when_its_holder_lets_go(tmp_path):
    first = sl.instance_lock("s", tmp_path)
    assert first is not None
    assert sl.instance_lock("s", tmp_path) is None  # flock is per open file: a second holder is refused
    assert sl.instance_lock("other", tmp_path) is not None  # one lock per strategy
    first.close()  # as the OS does when the holder dies
    again = sl.instance_lock("s", tmp_path)
    assert again is not None
    again.close()


def test_a_second_process_for_a_strategy_is_refused_before_it_connects_and_alerts_once(tmp_path, monkeypatch):
    from sleeve_fund.paper import node
    from sleeve_fund.store import Store

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/dup.db")
    monkeypatch.setenv("SLEEVE_LOCK_DIR", str(tmp_path / "locks"))
    store = Store()
    store.create_sleeve(name="dup", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=1000)
    built = []
    monkeypatch.setattr(node, "build_node", lambda *a, **k: built.append(1) or (_ for _ in ()).throw(AssertionError))
    first = sl.instance_lock("dup")  # the strategy's own process, running
    assert first is not None
    assert node.main(["--db-sleeve", "dup"]) == sl.EXIT_DUPLICATE
    assert node.main(["--db-sleeve", "dup"]) == sl.EXIT_DUPLICATE
    assert built == [] and store.orders("dup") == []  # never connected, nothing placed
    told = [e for e in store.events("dup", limit=100) if e["kind"] == sl.DUPLICATE_REFUSED]
    assert len(told) == 1 and "Duplicate refused" in told[0]["message"]
    first.close()


@pytest.mark.parametrize("code", [sl.EXIT_DUPLICATE, 1])
def test_a_refused_duplicates_exit_is_not_read_as_the_strategy_crashing(tmp_path, code):
    from sleeve_fund import supervisor as sup
    from sleeve_fund.store import Store

    class Exited:
        pid, returncode = 4243, code
        poll = lambda self: code

    store = Store(f"sqlite:///{tmp_path}/sv.db", portfolio_gate=True)
    store.create_sleeve(name="s", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=1000)
    store.set_desired_state("s", "running")
    sv = sup.Supervisor(store)
    sv.procs["s"] = sup.Proc(popen=Exited())
    sv.step()
    crashed = [e for e in store.events("s", limit=100) if e["kind"] == "process_crash"]
    if code == sl.EXIT_DUPLICATE:
        assert not crashed and store.sleeve("s").status != "error"
    else:  # the control: a real crash still reads as one
        assert crashed and store.sleeve("s").status == "error"
    assert sv.procs["s"].next_start is not None  # tried again after the backoff either way
