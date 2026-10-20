"""B-05 M4: the order client's connect check and runtime holds (drill-prep MUST 3, 8, 9 clock, 10, 12, 13, 14). On the
venue's own answers (recorded shapes, tests/exec_stub.py), the check names every way the key, the account or an
instrument isn't as the fund trades: a trade-only key, isolated margin, one-way, the venue's leverage at the engine's
cap, the clocks in step. It never writes to the account. A FAIL is journaled and refuses new entries at runtime
through the one entry gate (runtime.exec_target_block); exits and stops still run; a later PASS clears it. A key
refused mid-session is the same FAIL, CRITICAL "cannot protect" while we hold a position or order."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from exec_stub import VenueStub

from sleeve_fund.paper import exec_client
from sleeve_fund.paper.exec_client import (
    CONNECT_FAIL,
    connect_check,
    key_permissions,
    record_connect,
)

DEMO_ENV = {"EXEC_MODE": "demo", "EXEC_HOST": "api-demo.bybit.com", "BYBIT_DEMO_API_KEY": "stub",
            "BYBIT_DEMO_API_SECRET": "stub"}
NOW = datetime(2026, 10, 20, 18, 0, tzinfo=UTC)
# user/query-api (venue docs: user/apikey-info; to be confirmed with MUST 17's check: the docs host is unreachable)
KEY_OK = {"retCode": 0, "result": {"readOnly": 0, "permissions": {"ContractTrade": ["Order", "Position"],
                                                                  "Wallet": ["AccountTransfer"]}}}
ACCOUNT_OK = {"retCode": 0, "result": {"marginMode": "ISOLATED_MARGIN", "unifiedMarginStatus": 5}}
POSITION_OK = {"symbol": "BTCUSDT", "positionIdx": 0, "leverage": "2", "size": "0", "side": ""}


def _check(**kw):
    args = {"drill_go": True, "key": KEY_OK, "account": ACCOUNT_OK, "positions": {"BTCUSDT": POSITION_OK},
            "caps": {"BTCUSDT": 2.0}, "server_ms": 1_000_000, "local_ms": 1_000_200}
    args.update(kw)
    return connect_check(exec_client.exec_settings(DEMO_ENV), **args)


def _fails(lines):
    return [t for k, t in lines if k == "FAIL"]


def test_a_sound_account_passes_in_one_line():
    [(kind, text)] = _check()
    assert kind == "PASS" and "trade-only" in text and "isolated" in text


@pytest.mark.parametrize("change, words", [
    ({"key": {"retCode": 0, "result": {"readOnly": 0, "permissions": {"ContractTrade": ["Order"],
                                                                      "Wallet": ["Withdraw"]}}}}, "withdraw"),
    ({"key": {"retCode": 0, "result": {"readOnly": 1, "permissions": {"ContractTrade": ["Order"]}}}}, "can't trade"),
    ({"key": None}, "permissions couldn't be read"),
    ({"key": {"retCode": 10003, "retMsg": "API key is invalid."}}, "permissions couldn't be read"),
    ({"account": {"retCode": 0, "result": {"marginMode": "REGULAR_MARGIN"}}}, "not isolated"),
    ({"account": None}, "unreadable, not isolated"),
    ({"positions": {"BTCUSDT": {**POSITION_OK, "positionIdx": 1}}}, "hedge mode"),
    ({"positions": {"BTCUSDT": {**POSITION_OK, "leverage": "10"}}}, "the venue's leverage is 10"),
    ({"positions": {"BTCUSDT": None}}, "couldn't be read"),
    ({"caps": {"BTCUSDT": None}}, "cap unknown"),
    ({"server_ms": None}, "clock couldn't be read"),
    ({"local_ms": 1_002_500}, "2500 ms off"),
])
def test_every_way_the_key_account_or_instrument_is_wrong_is_a_fail_named(change, words):
    fails = _fails(_check(**change))
    assert fails and any(words in f for f in fails), fails


def test_the_pms_gates_still_apply_at_connect():
    assert any("drill-go" in f for f in _fails(_check(drill_go=False)))


def test_permissions_are_read_from_the_venues_answer():
    assert key_permissions(KEY_OK) == {"trade": True, "withdraw": False, "read_only": False}
    assert key_permissions(None) is None and key_permissions({"retCode": 10003}) is None


# the journal and the entry gate -------------------------------------------------------------------------------------
@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    import os

    from sleeve_fund.store import Store, make_engine, metadata

    if request.param == "sqlite":
        s = Store(f"sqlite:///{tmp_path}/j.db")
    else:
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("CI runs these against Postgres too (TEST_DATABASE_URL)")
        engine = make_engine(url)
        metadata.drop_all(engine)
        s = Store(engine=engine)
    s.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                    starting_balance=10_000, desired_state="running")
    return s


def _gate(store, monkeypatch, env):
    from sleeve_fund.paper import runtime

    for k in list(DEMO_ENV) + ["EXEC_MODE"]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return runtime.exec_target_block(store=store, name="s1")


def test_a_failed_check_refuses_entries_at_runtime_until_a_check_passes(store, monkeypatch):
    assert _gate(store, monkeypatch, DEMO_ENV) is None
    record_connect(store, "s1", _check(account={"retCode": 0, "result": {"marginMode": "REGULAR_MARGIN"}}), NOW)
    why = _gate(store, monkeypatch, DEMO_ENV)
    assert why is not None and why.code == "exec_target" and "not isolated" in str(why)
    record_connect(store, "s1", _check(), NOW + timedelta(minutes=1))
    assert _gate(store, monkeypatch, DEMO_ENV) is None
    kinds = [e["kind"] for e in store.events("s1")]
    assert kinds.count(CONNECT_FAIL) == 1 and kinds.count(exec_client.CONNECT_PASS) == 1


def test_a_pass_with_nothing_held_writes_nothing(store):
    record_connect(store, "s1", _check(), NOW)
    assert not [e for e in store.events("s1") if e["kind"].startswith("exec_connect")]


def test_off_the_journals_check_is_not_read(store, monkeypatch):
    record_connect(store, "s1", _check(key=None), NOW)
    assert _gate(store, monkeypatch, {}) is None  # OFF: the simulated client, nothing from the order client applies


def test_a_restart_check_that_didnt_tie_holds_entries_on_through_the_same_gate(store, monkeypatch):
    from sleeve_fund.paper import exec_reconcile

    result = exec_reconcile.Result("BTCUSDT", ("s1",), ("the account holds 0.010, the journal 0",))
    exec_reconcile.record(store, "BYBIT-PAPER-S1", "BTCUSDT", result, NOW)
    why = _gate(store, monkeypatch, DEMO_ENV)
    assert why is not None and "Restart check" in str(why)


def test_an_unreadable_journal_holds(monkeypatch):
    class Broken:
        def last_event(self, *a, **k):
            raise RuntimeError("down")

    assert exec_client.holds(Broken(), "s1") == ["the order client's last check couldn't be read from the journal"]


def test_the_journals_gate_reads_the_same_answer(store, monkeypatch):
    """The supervisor's and dashboard's gate (runtime.entry_blocked) sees the hold too."""
    from sleeve_fund.paper import runtime

    for k, v in DEMO_ENV.items():
        monkeypatch.setenv(k, v)
    record_connect(store, "s1", _check(server_ms=None), NOW)
    blocked, why = runtime.entry_blocked(store, "s1", NOW)
    assert blocked and "clock couldn't be read" in str(why)


# a key lost mid-session ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("code, holds, critical", [(10003, True, True), (10005, False, False), (110007, True, None)])
def test_a_key_refused_mid_session_is_critical_while_we_hold_and_a_fail_either_way(code, holds, critical):
    why = exec_client.lost_permission(code, venue_holds=holds)
    if critical is None:
        assert why is None  # not an auth refusal: the reject path handles it
    else:
        assert why is not None and why.startswith("CRITICAL: cannot protect") is critical


def test_the_model_journals_a_key_refusal_as_a_failed_check(store, monkeypatch):
    from sleeve_fund.strategies.base import LongFlatStrategy

    for k, v in DEMO_ENV.items():
        monkeypatch.setenv(k, v)
    me = SimpleNamespace(cache=SimpleNamespace(orders_open=lambda strategy_id=None: []), strategy_id=None,
                         _signed_qty=lambda: Decimal("0.01"),
                         runtime=SimpleNamespace(store=store, name="s1", now=lambda: NOW))
    LongFlatStrategy._check_lost_key(me, SimpleNamespace(reason="retCode=10003 API key is invalid."))
    [e] = [e for e in store.events("s1") if e["kind"] == CONNECT_FAIL]
    assert "CRITICAL: cannot protect" in e["message"] and e["level"] == "error"
    from sleeve_fund.paper import runtime

    assert runtime.exec_target_block(store=store, name="s1") is not None
    LongFlatStrategy._check_lost_key(me, SimpleNamespace(reason="110007 ab not enough for new order"))
    assert len([e for e in store.events("s1") if e["kind"] == CONNECT_FAIL]) == 1


# the limiter --------------------------------------------------------------------------------------------------------
def test_an_entry_over_the_limit_must_carry_its_first_asked_time():
    lim = exec_client.Limiter({"order/create": (1, 1.0, "")}, now=lambda: NOW)
    assert lim.take("order/create", opening=True, since=NOW) == "send"
    with pytest.raises(ValueError, match="first asked"):
        lim.take("order/create", opening=True)
    assert lim.take("order/create", opening=False) == "send"  # an exit never needs one


# the reads, signed, against the stand-in ------------------------------------------------------------------------------
class _Answers(VenueStub):
    def __init__(self, server_ms):
        super().__init__()
        self.server_ms = server_ms

    def route(self, method, path, body):
        if path.startswith("/v5/user/query-api"):
            return KEY_OK
        if path.startswith("/v5/account/info"):
            return ACCOUNT_OK
        if path.startswith("/v5/position/list"):
            return {"retCode": 0, "result": {"category": "linear", "list": [POSITION_OK]}}
        if path.startswith("/v5/market/time"):
            return {"retCode": 0, "result": {"timeSecond": str(self.server_ms // 1000),
                                             "timeNano": str(self.server_ms * 1_000_000)}}
        return super().route(method, path, body)


def test_the_reads_go_signed_as_gets_and_feed_the_check():
    from sleeve_fund.paper.venue_reads import VenueReads

    venue = _Answers(int(time.time() * 1000)).start()
    try:
        reads = VenueReads(venue.http_url, api_key="k", api_secret="s", limiter=exec_client.Limiter())
        lines = reads.check(exec_client.exec_settings(DEMO_ENV), drill_go=True, caps={"BTCUSDT": 2.0},
                            local_ms=lambda: int(time.time() * 1000))
    finally:
        venue.stop()
    assert [k for k, _ in lines] == ["PASS"], lines
    assert {m for m, _, _ in venue.seen} == {"GET"}  # reads only: nothing written to the account
    assert all({k.lower(): v for k, v in h.items()}.get("x-bapi-sign") for h in venue.headers)
    assert "k" not in repr(reads) and "s" not in repr(reads).replace("VenueReads", "")


def test_no_answer_is_never_a_pass():
    from sleeve_fund.paper.venue_reads import VenueReads

    reads = VenueReads("http://127.0.0.1:9", api_key="k", api_secret="s", limiter=exec_client.Limiter(),
                       timeout_s=0.5)
    lines = reads.check(exec_client.exec_settings(DEMO_ENV), drill_go=True, caps={"BTCUSDT": 2.0},
                        local_ms=lambda: 0)
    assert "PASS" not in [k for k, _ in lines] and len(_fails(lines)) >= 4
    json.dumps(lines)  # plain words, journal-ready


# the limiter in front of the model's orders -------------------------------------------------------------------------
def _model(limiter, *, on=True, adds=True):
    from sleeve_fund.strategies.base import _rate_refused

    me = SimpleNamespace(runtime=SimpleNamespace(exec_limiter=limiter), _backtest=False,
                         _cfg=SimpleNamespace(exec_on=on), _adds=lambda side: adds,
                         clock=SimpleNamespace(utc_now=lambda: NOW))
    return lambda intent: _rate_refused(me, "BUY", intent)


def test_an_entry_over_the_limit_is_refused_now_and_an_exit_always_goes():
    lim = exec_client.Limiter({"order/create": (1, 1.0, "")}, now=lambda: NOW)
    ask = _model(lim)
    assert ask("entry") is None  # the one token
    assert "rate limit" in ask("entry")  # over: refused now, never queued
    for intent in ("exit", "stop_loss", "take_profit", "flatten"):
        assert ask(intent) is None  # never held back, over the limit or not
    assert "rate limit" in ask("rebalance")  # a rebalance that adds is an entry
    assert _model(lim, adds=False)("rebalance") is None  # one that reduces is not


def test_off_or_with_no_limiter_nothing_changes():
    lim = exec_client.Limiter({"order/create": (0, 1.0, "")}, now=lambda: NOW)
    assert _model(lim, on=False)("entry") is None
    assert _model(None)("entry") is None
