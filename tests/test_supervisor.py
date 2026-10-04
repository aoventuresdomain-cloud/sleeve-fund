from datetime import timedelta

import pytest

from sleeve_fund.store import Store, utcnow
from sleeve_fund.supervisor import Proc, clear, decide, seed


class FakePopen:
    pid = 1

    def __init__(self, alive=True, code=None):
        self._alive, self.returncode = alive, code

    def poll(self):
        return None if self._alive else self.returncode


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path}/t.db")


@pytest.fixture
def sleeve(store):
    return store.create_sleeve(name="s", strategy="trend_filter", instrument="BTC/USD",
                               bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=1000)


def test_start_stop_and_backoff(sleeve):
    now = utcnow()
    assert decide(sleeve, Proc(), now) == "start"
    assert decide(sleeve, Proc(next_start=now + timedelta(seconds=30)), now) == "wait"
    assert decide(sleeve, Proc(popen=FakePopen(alive=False, code=1)), now) == "crashed"
    sleeve.desired_state = "stopped"
    assert decide(sleeve, Proc(popen=FakePopen(), started_at=now), now) == "stop"
    assert decide(sleeve, Proc(), now) == "none"


def test_stale_heartbeat_restarts_only_after_grace(sleeve):
    now = utcnow()
    young = Proc(popen=FakePopen(), started_at=now - timedelta(minutes=1))
    old = Proc(popen=FakePopen(), started_at=now - timedelta(minutes=10))
    assert decide(sleeve, young, now) == "none"
    assert decide(sleeve, old, now) == "restart_stale"
    sleeve.heartbeat_at = now - timedelta(seconds=20)
    assert decide(sleeve, old, now) == "none"


def test_seed_is_idempotent(store):
    paths = ["configs/examples/btc_trend_smoke.toml", "configs/examples/btc_trend_daily.toml"]
    assert seed(store, paths) == ["btc-trend-smoke", "btc-trend-daily"]
    assert seed(store, paths) == []
    s = store.sleeve("btc-trend-smoke")
    assert s.params == {"fast": 5, "slow": 20, "max_notional": 1000}
    assert s.risk_profile == "aggressive"


def test_the_seeded_strategies_are_the_two_test_strategies(store):
    import glob

    assert sorted(seed(store, sorted(glob.glob("configs/sleeves/*.toml")))) == [
        "ping-pong-ls-test", "ping-pong-test", "rsi-bands-ls-test", "rsi-bands-test"]
    assert store.sleeve("rsi-bands-test").bar_spec == "1-MINUTE-LAST-INTERNAL"
    # The long/short pair trade a perpetual and are added stopped, for the PM to start.
    ls = store.sleeve("ping-pong-ls-test")
    assert (ls.params["market"], ls.params["allow_short"], ls.desired_state, ls.status) == ("perp", True, "stopped", "stopped")
    assert store.sleeve("ping-pong-test").desired_state == "running"


def test_clear_puts_every_strategy_away_once_and_keeps_its_journal(store, sleeve, tmp_path):
    """The PM's clean slate (4 Oct 2026): every strategy on the book is stopped and archived, its journal
    untouched; the entry applies once, so restarts and strategies added afterwards are left alone."""
    store.set_desired_state("s", "running")
    store.record_fill("s", side="BUY", qty=0.01, price=100.0, fee=0.008, order_id="o1", trade_id="t1")
    store.create_sleeve(name="old", strategy="buy_and_hold", instrument="ETH/USD",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=500)
    store.set_desired_state("old", "stopped")
    store.archive("old")
    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "2026-10-04"\nreason = "PM asked for a clean slate"\n')
    assert clear(store, str(path)) == ["s"]
    assert store.sleeve("s").desired_state == "stopped" and set(store.archived()) == {"s", "old"}
    assert len(store.fills("s")) == 1
    assert [d["action"] for d in store.decisions("s")][:2] in (["archive", "stop"], ["stop", "archive"])
    seed(store, ["configs/sleeves/ping_pong_test.toml"])
    assert clear(store, str(path)) == []  # applied already: the new strategy stays
    assert "ping-pong-test" not in store.archived()
    assert any(e["kind"] == "book_cleared" for e in store.events())


def test_the_shipped_clear_file_reads(store):
    assert clear(store, "configs/clear.toml") == []
    assert store.decisions(action="clear")[0]["reason"].startswith("2026-10-04: ")


def test_changed_settings_restart_a_running_strategy_once(store, sleeve, monkeypatch):
    """Saving new settings restarts the running process so it trades under them; the strategy itself
    never acts on the reload, and a stopped strategy just picks them up when it starts."""
    import subprocess

    from sleeve_fund.supervisor import Supervisor

    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, env=None: started.append(args) or FakePopen())
    sup = Supervisor(store)
    sup.step()
    assert len(started) == 1
    assert store.change_settings("s", risk_profile="conservative", params={}, warmup_bars=0)
    assert store.change_settings("s", risk_profile="conservative", params={"stop_loss": 0.05}, warmup_bars=0)
    assert len(store.pending_commands("s")) == 1  # two saves, one restart
    monkeypatch.setattr(sup, "_stop", lambda name, proc, why: setattr(proc, "popen", None))
    sup.step()
    assert len(started) == 2 and store.pending_reload("s") is None
    assert any(e["kind"] == "settings_applied" for e in store.events("s"))
    sup.step()
    assert len(started) == 2
    store.set_desired_state("s", "stopped")
    assert store.change_settings("s", risk_profile="balanced", params={}, warmup_bars=0) is False
    assert store.pending_reload("s") is None and store.sleeve("s").risk_profile == "balanced"


def test_the_strategy_leaves_a_reload_to_the_supervisor(store, sleeve):
    from sleeve_fund.paper.runtime import SleeveRuntime

    store.change_settings("s", risk_profile="balanced", params={}, warmup_bars=0)
    rt = SleeveRuntime(store, "s")
    rt.on_start(0.008)
    rt.tick(equity=1000, cash=1000, qty=0, price=1)
    assert store.pending_reload("s") is not None
