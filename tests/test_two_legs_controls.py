"""P2-8: the PM acts on a two-leg Strategy through its first leg's row, and what reaches one leg reaches both.
Pause, resume and flatten are queued for both legs' runtimes in one transaction (each leg keeps its own status, so a
leg left halted would refuse its half of the next pair: Advisor M2); start and stop set both rows. Anything not built
for two legs yet (resets, account moves, archiving, lifecycle, settings) is refused rather than applied to one leg,
and a second leg's row takes no action of its own."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.store import RAL, Store

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}
FIRST = {"instrument": "BTC/USD", "params": {}, "venue": "KRAKEN", "side": 1}
SECOND = {"instrument": "BTC/USD", "params": {"market": "perp"}, "venue": "KRAKEN", "side": -1}


def _carry(store: Store) -> None:
    store.create_two_legs(name="carry", strategy="trend_filter", bar_spec="1-HOUR-LAST-INTERNAL",
                          starting_balance=10_000, first=FIRST, second=SECOND)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    _carry(store)
    return TestClient(app_mod.create_app(store)), store


def _pending(store, name):
    return [c["command"] for c in store.pending_commands(name)]


@pytest.mark.parametrize("command", ["pause", "resume", "flatten"])
def test_a_command_on_the_first_leg_reaches_both_legs_once(tmp_path, command):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    _carry(store)
    store.command("carry", command, "testing")
    assert _pending(store, "carry") == [command] and _pending(store, "carry~2") == [command]
    store.command("carry", command, "again")  # the leg already has it waiting: not queued twice for it
    assert _pending(store, "carry") == [command, command] and _pending(store, "carry~2") == [command]


def test_start_and_stop_set_both_rows(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    _carry(store)
    store.set_desired_state("carry", "stopped")
    assert store.sleeve("carry").desired_state == store.sleeve("carry~2").desired_state == "stopped"
    store.set_desired_state("carry", "running")
    assert store.sleeve("carry~2").desired_state == "running"


def test_a_reset_after_liquidation_is_refused_on_either_leg(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    _carry(store)
    for name in ("carry", "carry~2"):
        with pytest.raises(ValueError, match="isn't built for a two-leg Strategy"):
            store.command(name, RAL, "after the liquidation", incident=1)
    assert _pending(store, "carry") == _pending(store, "carry~2") == []


def test_the_dashboard_refuses_any_action_on_the_second_legs_row(client):
    c, store = client
    for path, data in (("command", {"command": "flatten", "reason": "x"}), ("reset", {"reason": "x"}),
                       ("settings", {"risk_profile": "balanced"})):
        r = c.post(f"/sleeves/carry~2/{path}", data=data, auth=AUTH, headers=SAME, follow_redirects=False)
        assert r.status_code == 303 and "second+leg+of+carry" in r.headers["location"]
    assert _pending(store, "carry~2") == [] and store.pending_reset("carry~2") is None


@pytest.mark.parametrize("path, words", [("reset", "a+reset"), ("account", "a+move+to+another+account"),
                                         ("archive", "archiving"), ("lifecycle", "winding+down+or+retiring"),
                                         ("settings", "a+change+of+settings")])
def test_what_isnt_built_for_two_legs_is_refused_on_the_first_leg_and_changes_nothing(client, path, words):
    c, store = client
    before = store.sleeve("carry")
    r = c.post(f"/sleeves/carry/{path}", data={"reason": "x", "action": "archive", "account": "other"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and f"{words}+isn%27t+built+for+a+two-leg+Strategy" in r.headers["location"]
    assert store.sleeve("carry") == before and store.pending_reset("carry") is None


def test_a_pause_on_the_dashboard_reaches_both_legs(client):
    c, store = client
    r = c.post("/sleeves/carry/command", data={"command": "pause", "reason": "testing"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303 and "command_error" not in r.headers["location"]
    assert _pending(store, "carry") == ["pause"] and _pending(store, "carry~2") == ["pause"]


def test_a_single_strategy_is_untouched_by_the_two_leg_rules(client):
    c, store = client
    store.create_sleeve(name="solo", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=1000)
    store.command("solo", "pause", "testing")
    assert _pending(store, "solo") == ["pause"]
    from sleeve_fund.dashboard.app import two_leg_refusal

    assert all(two_leg_refusal(store, "solo", a) is None for a in ("reset", "account", "settings", "command"))
