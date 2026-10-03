from datetime import timedelta

import pytest

from sleeve_fund.store import Store, utcnow
from sleeve_fund.supervisor import Proc, decide, seed


class FakePopen:
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
    paths = ["configs/sleeves/btc_trend_smoke.toml", "configs/sleeves/btc_trend_daily.toml"]
    assert seed(store, paths) == ["btc-trend-smoke", "btc-trend-daily"]
    assert seed(store, paths) == []
    s = store.sleeve("btc-trend-smoke")
    assert s.params == {"fast": 5, "slow": 20, "max_notional": 1000}
    assert s.risk_profile == "aggressive"
