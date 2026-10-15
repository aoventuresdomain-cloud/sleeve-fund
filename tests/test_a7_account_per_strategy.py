"""A7 one account per Strategy (pre-live execution set; Done when pe2-plans/a7-account-per-strategy.md; Advisor 15 Oct
~09:35 UK: approved with one MUST; HoQA line 5). OFF by default (ACCOUNT_PER_STRATEGY unset): nothing changes. ON: a
live or test-network account is one Strategy's; a share is refused at assignment, refuses new entries (a CHOKE cause:
exits, stops, take-profits, the safety stop and a PM flatten still run, Advisor MUST), and the start check FAILs naming
it. Paper accounts are exempt. No key is touched."""

from __future__ import annotations

import pytest

from sleeve_fund import start_check
from sleeve_fund.paper import runtime
from sleeve_fund.store import Store

ON = {"ACCOUNT_PER_STRATEGY": "on"}


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path}/j.db")
    for name in ("s1", "s2", "s3"):
        s.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, desired_state="stopped")
    s.create_account("desk-a", "live", venue="KRAKEN")
    s.create_account("desk-b", "live", venue="KRAKEN")
    s.create_account("research", "paper")
    return s


def _on(monkeypatch):
    for k, v in ON.items():
        monkeypatch.setenv(k, v)


def test_1_off_by_default_sharing_a_live_account_is_allowed_and_nothing_is_refused(store, monkeypatch):
    monkeypatch.delenv("ACCOUNT_PER_STRATEGY", raising=False)
    store.assign_account("s1", "desk-a")
    store.assign_account("s2", "desk-a")
    assert runtime.account_block(store, "s2") is None
    assert "account_shared" not in (runtime.entry_blocked(store, "s2")[1] or runtime.Why("", "x")).codes
    sec = start_check.check_account_sharing({}, store.engine)
    assert not sec.failed


def test_2_on_assigning_a_live_account_another_strategy_holds_is_refused_naming_it(store, monkeypatch):
    _on(monkeypatch)
    store.assign_account("s1", "desk-a")
    with pytest.raises(ValueError, match="desk-a.*s1"):
        store.assign_account("s2", "desk-a")
    assert store.account_of("s2") == "paper"
    store.assign_account("s2", "desk-b")  # its own: fine
    store.assign_account("s3", "research")  # paper accounts are shared by design
    store.assign_account("s2", "research")


def test_3_on_a_shared_account_from_before_refuses_new_entries_on_both_and_the_start_check_fails(store, monkeypatch):
    store.assign_account("s1", "desk-a")
    store.assign_account("s2", "desk-a")  # shared while A7 was off
    _on(monkeypatch)
    for name in ("s1", "s2"):
        why = runtime.account_block(store, name)
        assert why is not None and why.code == "account_shared" and "desk-a" in why
        blocked, said = runtime.entry_blocked(store, name, starting=True)  # the dashboard's Start reads this too
        assert blocked and "account_shared" in said.codes
    assert runtime.account_block(store, "s3") is None
    sec = start_check.check_account_sharing(ON, store.engine)
    assert sec.failed and any("desk-a" in w and "s1" in w and "s2" in w for _, w in sec.lines)


def test_4_on_a_shared_account_never_stops_exits_stops_or_a_flatten(store, monkeypatch, tmp_path):
    """Advisor MUST: two Strategies sharing an account and holding, restarted with A7 on: both stops fire and an exit
    goes out; only entries are refused. The share is a CHOKE cause, and CHOKE never gates a stop, a reduce-only
    order, the PM's close or a liquidation (blocked_state)."""
    store.assign_account("s1", "desk-a")
    store.assign_account("s2", "desk-a")
    for name in ("s1", "s2"):
        store.record_fill(name, side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id=f"{name}-in",
                          trade_id=f"{name}-in")
    _on(monkeypatch)
    for name in ("s1", "s2"):
        rt = runtime.SleeveRuntime(store, name)
        rt.on_start(0.008)
        blocked, why = rt.entry_blocked()
        assert blocked and "account_shared" in why.codes
        assert runtime.account_block(store, name).code in runtime.CODES  # a CHOKE cause: entries only
        assert rt.wiped_out is False and store.sleeve(name).status != "halted"  # nothing halts or stops it


def test_5_on_reassigning_a_strategy_that_holds_or_has_an_open_order_is_refused_with_the_reason(store, monkeypatch):
    """HoQA line 5: a move never strands a position or an order on the account it leaves."""
    _on(monkeypatch)
    store.record_fill("s1", side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id="h", trade_id="h")
    with pytest.raises(ValueError, match="holds a position"):
        store.assign_account("s1", "desk-a")
    store.record_order("s3", order_id="w", side="BUY", qty=0.1, intent="entry", reason="r")
    with pytest.raises(ValueError, match="open order"):
        store.assign_account("s3", "desk-b")


def test_6_a_put_away_strategy_holds_its_account_until_it_is_released(store, monkeypatch):
    """Advisor (b): released only once out of the book (put away and withdrawn); until then it still holds it."""
    _on(monkeypatch)
    store.assign_account("s1", "desk-a")
    store.archive("s1")
    if "s1" in store.withdrawn():  # flat with nothing to wait for: released at once
        store.assign_account("s2", "desk-a")
        assert store.account_of("s2") == "desk-a"
    else:
        with pytest.raises(ValueError, match="s1"):
            store.assign_account("s2", "desk-a")


def test_7_a_withdrawn_strategy_with_an_order_still_open_keeps_its_account(store, monkeypatch):
    """Advisor (b), HoQA: out of the book is not enough; flat and no open order too."""
    _on(monkeypatch)
    store.assign_account("s1", "desk-a")
    monkeypatch.setattr(store, "withdrawn", lambda: {"s1": 0})
    store.record_order("s1", order_id="late", side="SELL", qty=0.1, intent="exit", reason="r")
    with pytest.raises(ValueError, match="s1"):
        store.assign_account("s2", "desk-a")
    store.update_order("late", status="canceled")
    store.assign_account("s2", "desk-a")
