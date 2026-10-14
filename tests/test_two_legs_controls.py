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


# --- CR #283 first pass: F283-1 to F283-3 -----------------------------------------------------------------------------

def _leg_holds(store, qty=0.05):
    """The second leg still short `qty` while the first is flat: its close sent by the Pair, not yet filled."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    store.record_fill("carry~2", side="SELL", qty=qty, price=60_000.0, fee=0.0, order_id="l2", trade_id="l2", ts=now)
    cash = float(store.journal_book("carry~2", store.sleeve("carry~2").starting_balance)["cash"])
    store.record_equity("carry~2", equity=cash - qty * 60_000.0, cash=cash, qty=-qty, price=60_000.0,
                        benchmark=cash, ts=now)


def test_f283_1_a_reset_is_refused_on_either_legs_row_in_the_store_itself(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    _carry(store)
    for name in ("carry", "carry~2"):
        for book in (False, True):
            with pytest.raises(ValueError, match="isn't built for a two-leg Strategy"):
                store.request_reset(name, "testing", book=book)
    assert store.pending_reset("carry") is None and store.pending_reset("carry~2") is None


def test_f283_1_a_book_reset_leaves_a_two_leg_strategy_out_and_resets_the_rest(client):
    c, store = client
    store.create_sleeve(name="solo", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=1000)
    r = c.post("/book/reset", data={"reason": "testing"}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and "reset_error" not in r.headers["location"]
    assert store.pending_reset("solo") is not None
    assert store.pending_reset("carry") is None and store.pending_reset("carry~2") is None


def test_f283_2_a_second_leg_still_holding_keeps_the_strategy_holding_for_the_supervisor(tmp_path):
    from sleeve_fund.supervisor import Proc, Supervisor

    store = Store(f"sqlite:///{tmp_path}/t.db")
    _carry(store)
    store.set_desired_state("carry", "stopped")
    assert Supervisor(store)._holds(store.sleeve("carry"), Proc()) is False
    _leg_holds(store)
    assert store.holds_any("carry") and store.holds_any("carry~2")
    assert Supervisor(store)._holds(store.sleeve("carry"), Proc()) is True  # so it runs for its exits only


def test_f283_2_a_portfolio_halt_flattens_a_holding_second_leg_through_its_first_legs_row(tmp_path):
    from datetime import datetime, timezone

    from sleeve_fund.supervisor import Supervisor

    store = Store(f"sqlite:///{tmp_path}/t.db")
    _carry(store)
    _leg_holds(store)
    sup = Supervisor(store)
    sup._halt_every_strategy("testing", datetime.now(timezone.utc))
    assert _pending(store, "carry") == ["flatten"] and _pending(store, "carry~2") == ["flatten"]
    assert set(sup._flatten_told) == {"carry"}  # told once, as the Strategy, never the second leg's row alone


def test_f283_2_a_clean_slate_flattens_and_starts_the_strategy_whole_and_archives_both_legs_once_flat(tmp_path):
    from sleeve_fund import supervisor

    store = Store(f"sqlite:///{tmp_path}/t.db")
    _carry(store)
    store.set_desired_state("carry", "stopped")
    _leg_holds(store)
    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "s1"\nreason = "fresh book"\n')
    assert supervisor.clear(store, str(path)) == []
    assert _pending(store, "carry") == ["flatten"] and _pending(store, "carry~2") == ["flatten"]
    assert store.sleeve("carry").desired_state == store.sleeve("carry~2").desired_state == "running"
    assert store.archived() == {}
    store.record_fill("carry~2", side="BUY", qty=0.05, price=60_000.0, fee=0.0, order_id="l3", trade_id="l3")
    store.drop_pending("carry", "flat now")
    assert supervisor.clear(store, str(path)) == ["carry"]
    assert set(store.archived()) == {"carry", "carry~2"}


def test_f283_2_archive_puts_both_legs_away_from_the_first_and_never_the_second_alone(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    _carry(store)
    store.set_desired_state("carry", "stopped")
    with pytest.raises(ValueError, match="second leg of carry"):
        store.archive("carry~2")
    store.archive("carry")
    assert set(store.archived()) == {"carry", "carry~2"}
    store.unarchive("carry")
    assert store.archived() == {}


def test_f283_2_the_kill_switch_tells_a_strategy_whose_second_leg_holds_through_its_first_legs_row(client):
    c, store = client
    store.set_desired_state("carry", "stopped")
    _leg_holds(store)
    r = c.post("/book/flatten", data={"reason": "testing"}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and "killed=1" in r.headers["location"]
    assert _pending(store, "carry") == ["flatten"] and _pending(store, "carry~2") == ["flatten"]
    assert store.sleeve("carry").desired_state == store.sleeve("carry~2").desired_state == "running"


def test_f283_3_the_second_legs_queue_follows_the_first_legs_in_order(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    _carry(store)
    for command in ("resume", "pause", "resume"):
        store.command("carry", command, "testing")
    assert _pending(store, "carry") == _pending(store, "carry~2") == ["resume", "pause", "resume"]


def test_f283_3_a_stop_drops_both_legs_waiting_commands(client):
    c, store = client
    store.command("carry", "pause", "testing")
    r = c.post("/sleeves/carry/command", data={"command": "stop", "reason": "testing"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303 and "command_error" not in r.headers["location"]
    assert _pending(store, "carry") == _pending(store, "carry~2") == []
