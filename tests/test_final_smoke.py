"""Final-head smoke (scripts/final_smoke.py, read-only) and the B3 round trip (scripts/round_trip.py): each check's
PASS and FAIL on a journal in memory, no server and no network."""

import importlib.util
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace

import pytest

from sleeve_fund.store import Store

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 14, 9, 0, 3, tzinfo=UTC)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fs = _load("final_smoke")
rt = _load("round_trip")


def _perp(store, name="ping-pong-ls-binance", desired="stopped"):
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, params={"market": "perp", "rise": 0.01, "dip": 0.005},
                        venue="BINANCE", desired_state=desired)


def _fill(store, name, side, oid, ts, qty=0.01, price=60_000.0, intent="entry"):
    store.record_order(name, order_id=oid, side=side, qty=qty, intent=intent, reason="smoke", ts=ts)
    store.book_fill(name, side=side, qty=qty, price=price, fee=round(qty * price * 0.00055, 8), order_id=oid,
                    trade_id=f"t-{oid}", ts=ts)


# --- B6 safety -------------------------------------------------------------------------------------------------------

def test_safety_passes_without_credentials_and_names_a_credential_without_its_value(capfd):
    assert fs.safety({"PATH": "/bin", "BINANCE_API_KEY": ""}) == 0  # set but empty: nothing usable
    assert fs.safety({"BINANCE_API_KEY": "sekrit-value", "KRAKEN_API_SECRET__DESK": "x"}) == 1
    out = capfd.readouterr().out
    assert "B6 safety: PASS" in out and "B6 safety: FAIL" in out
    assert "BINANCE_API_KEY" in out and "sekrit-value" not in out


def test_safety_in_the_mirror_allows_only_its_demo_account_names(capfd):
    demo = {"BYBIT_DEMO_API_KEY": "k", "BYBIT_DEMO_API_SECRET": "s", "DERIBIT_TESTNET_API_KEY": "k"}
    assert fs.safety(demo, mirror_container=True) == 0
    assert fs.safety(demo) == 1  # anywhere else, even a demo key fails
    assert fs.safety({**demo, "BYBIT_API_KEY": "live-value"}, mirror_container=True) == 1
    assert fs.safety({"BINANCE_API_KEY__DESK": "x"}, mirror_container=True) == 1
    out = capfd.readouterr().out
    assert "credential set: BYBIT_API_KEY" in out and "live-value" not in out


# --- B2 hub ----------------------------------------------------------------------------------------------------------

class _Hub:
    def __init__(self, lines):
        self.lines, self.sent = lines, []

    def __call__(self, address, timeout):
        self.address = address
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def sendall(self, b):
        self.sent.append(b)

    def makefile(self, mode):
        from sleeve_fund.hub import protocol
        return [protocol.encode(m) for m in self.lines]


def test_hub_seen_stops_once_every_instrument_has_a_trade_and_a_quote():
    ns = int(T0.timestamp() * 1e9)
    hub = _Hub([{"t": "hello", "v": 1}, {"t": "trade", "id": "X.BINANCE", "ts": ns}, {"t": "hb", "ts": ns},
                {"t": "quote", "id": "X.BINANCE", "ts": ns + 1}, {"t": "trade", "id": "X.BINANCE", "ts": ns + 9}])
    seen = fs.hub_seen("hub-binance:7700", ["X.BINANCE"], 5, connect=hub)
    assert seen["X.BINANCE"] == {"trade": ns, "quote": ns + 1}  # stopped before the last line
    assert hub.address == ("hub-binance", 7700) and b'"sub"' in hub.sent[0]


def _history(closed):
    return SimpleNamespace(coverage=lambda venue, pair: SimpleNamespace(closed=closed))


@pytest.mark.parametrize("trade_age,closed_behind,ok", [(5, 1, True), (90, 1, False), (5, 3, False)],
                         ids=["fresh", "stale-trade", "history-behind"])
def test_hub_is_fresh_only_with_recent_prints_and_the_last_closed_minute_stored(monkeypatch, capfd, trade_age,
                                                                               closed_behind, ok):
    import pandas as pd

    monkeypatch.chdir(ROOT)
    now = T0
    ids = fs._hub_ids(fs.configured())
    assert ids, "a configured Strategy is hub-fed"
    stamp = {i: {"trade": int((now.timestamp() - trade_age) * 1e9), "quote": int((now.timestamp() - 2) * 1e9)}
             for i in ids}
    monkeypatch.setattr(fs, "hub_seen", lambda *a, **k: stamp)
    closed = pd.Timestamp(now.replace(second=0, microsecond=0) - timedelta(minutes=closed_behind))
    assert fs.hub(address="hub-binance:7700", history=_history(closed), now=now) == (0 if ok else 1)
    assert ("B2 hub fresh: PASS" in capfd.readouterr().out) is ok


# --- B4 journal ------------------------------------------------------------------------------------------------------

def test_journal_passes_a_round_trip_and_prints_the_fee_by_hand(capfd):
    store = Store.in_memory()
    _perp(store)
    _fill(store, "ping-pong-ls-binance", "BUY", "o1", T0 + timedelta(minutes=1))
    _fill(store, "ping-pong-ls-binance", "SELL", "o2", T0 + timedelta(minutes=2), price=60_100.0, intent="exit")
    assert fs.journal("ping-pong-ls-binance", T0, store=store) == 0
    out = capfd.readouterr().out
    assert "qty x price x taker = 0.33000000" in out and "B4 journal: PASS" in out


def test_journal_fails_an_open_order_a_funding_conflict_and_a_held_position(capfd):
    store = Store.in_memory()
    _perp(store)
    _fill(store, "ping-pong-ls-binance", "BUY", "o1", T0 + timedelta(minutes=1))
    store.record_order("ping-pong-ls-binance", order_id="o3", side="BUY", qty=0.01, intent="entry", reason="r",
                       order_type="LIMIT", ts=T0 + timedelta(minutes=3))
    store.event("ping-pong-ls-binance", "error", "funding_conflict", "two rates", ts=T0 + timedelta(minutes=4))
    assert fs.journal("ping-pong-ls-binance", T0, store=store) == 1
    line = next(x for x in capfd.readouterr().out.splitlines() if x.startswith("B4 journal"))
    assert "o3 still" in line and "funding_conflict" in line and "no round trip" in line and "still holding" in line


# --- B5 gate ---------------------------------------------------------------------------------------------------------

def test_gate_fails_a_book_never_marked():
    store = Store.in_memory(portfolio_gate=True)
    _perp(store)
    assert fs.gate(store=store, now=T0) == 1


def test_gate_passes_a_marked_book_where_the_tile_equals_the_gate_and_a_dry_overlimit_entry_bites(capfd):
    from test_p21b_wiring_qa import portfolio_pass

    store = Store.in_memory(portfolio_gate=True)
    _perp(store)
    portfolio_pass(store, T0, D(20_000))
    assert fs.gate(store=store, now=T0 + timedelta(seconds=1), atr_pct=lambda s: 0.03) == 0
    out = capfd.readouterr().out
    assert "open risk: gate 0.00, tile 0.00" in out and "B5 gate: PASS" in out
    dry = next(x for x in out.splitlines() if x.startswith("dry entry"))
    assert ": rejected" in dry or ": trimmed" in dry, dry


# --- B3 round trip ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,desired,held,why", [
    ("rsi-bands-ls-binance", "stopped", False, "allowlist"),
    ("ping-pong-ls-binance", "running", False, "starts only a stopped"),
    ("ping-pong-ls-binance", "stopped", True, "only from flat"),
], ids=["not-allowlisted", "running", "holding"])
def test_round_trip_refuses_and_changes_nothing(capfd, name, desired, held, why):
    store = Store.in_memory(portfolio_gate=True)
    _perp(store, name, desired)
    if held:
        _fill(store, name, "BUY", "o1", T0)
    assert rt.run(store, name, 60, sleep=lambda s: pytest.fail("waited")) == 1
    assert why in capfd.readouterr().out
    assert store.sleeve(name).desired_state == desired and store.decisions(limit=10) == []


def _supervisor(store, name):
    """Stands in for the supervisor and the strategy process: the entry fills once the Strategy is running, and the
    exit once a flatten is waiting."""
    def sleep(_):
        rows = store.fills(name)
        if store.sleeve(name).desired_state == "running" and store.sleeve(name).status != "running":
            store.set_status(name, "running", "")  # the supervisor has started it (B3-1: the round trip checks)
        if store.sleeve(name).desired_state == "running" and not rows:
            _fill(store, name, "BUY", "e1", datetime.now(UTC))
        elif any(c["command"] == "flatten" for c in store.pending_commands(name)) and len(rows) == 1:
            _fill(store, name, "SELL", "x1", datetime.now(UTC), intent="pm_flatten")
    return sleep


def test_round_trip_starts_flattens_after_the_entry_and_stops_again(monkeypatch, capfd):
    from sleeve_fund.paper import runtime

    monkeypatch.setattr(runtime, "entry_blocked", lambda *a, **k: (False, None))
    store = Store.in_memory(portfolio_gate=True)
    _perp(store)
    store._update_sleeve("ping-pong-ls-binance", risk_profile="conservative")  # as after its first round trip (B3-1)
    assert rt.run(store, "ping-pong-ls-binance", 600, sleep=_supervisor(store, "ping-pong-ls-binance")) == 0
    out = capfd.readouterr().out
    assert "flatten sent" in out and "B3 round trip: PASS" in out
    assert store.sleeve("ping-pong-ls-binance").desired_state == "stopped"
    actions = [d["action"] for d in reversed(store.decisions(limit=10))]
    assert actions[0] == "start" and actions[-1] == "stop" and "flatten" in actions  # the stub never takes the flatten
    assert {d["actor"] for d in store.decisions(limit=10) if d["action"] in ("start", "flatten", "stop")} == {rt.ACTOR}


def test_round_trip_stops_the_strategy_on_an_error_and_on_a_timeout(monkeypatch, capfd):
    from sleeve_fund.paper import runtime

    monkeypatch.setattr(runtime, "entry_blocked", lambda *a, **k: (False, None))
    store = Store.in_memory(portfolio_gate=True)
    _perp(store)

    def boom(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        rt.run(store, "ping-pong-ls-binance", 600, sleep=boom)
    assert store.sleeve("ping-pong-ls-binance").desired_state == "stopped"
    ticks = iter(range(0, 10_000, 100))
    assert rt.run(store, "ping-pong-ls-binance", 250, sleep=lambda s: None, clock=lambda: next(ticks)) == 1
    assert store.sleeve("ping-pong-ls-binance").desired_state == "stopped"
    assert "B3 round trip: FAIL (fills on no side; it was never seen running)" in capfd.readouterr().out


# --- B3-1 (14 Oct, run 38582237679): a start the supervisor refuses fails at once, never after the timeout -----------

def test_round_trip_fails_before_the_start_when_the_supervisor_would_refuse_the_model(capfd):
    """Ping pong places no stop: on the balanced profile (2x) the supervisor refuses it at start (check_perp_stop).
    The round trip says so before starting it, and waits for nothing."""
    store = Store.in_memory(portfolio_gate=True)
    _perp(store)
    assert rt.run(store, "ping-pong-ls-binance", 1800, sleep=lambda s: pytest.fail("waited"), conservative=False) == 1
    out = capfd.readouterr().out
    assert "B3 round trip: FAIL (the supervisor would refuse it:" in out and "places no stop" in out
    assert store.sleeve("ping-pong-ls-binance").desired_state == "stopped" and store.decisions(limit=10) == []


def test_round_trip_fails_at_once_on_the_supervisors_start_refused_event(monkeypatch, capfd):
    from sleeve_fund.paper import runtime

    monkeypatch.setattr(runtime, "entry_blocked", lambda *a, **k: (False, None))
    store = Store.in_memory(portfolio_gate=True)
    _perp(store)
    waits = []

    def supervisor_refuses(_):
        waits.append(1)
        if len(waits) > 1:
            pytest.fail("waited on after the refusal")
        store.event("ping-pong-ls-binance", "error", "start_refused", "Not started: the hub is down")

    assert rt.run(store, "ping-pong-ls-binance", 1800, sleep=supervisor_refuses) == 1
    out = capfd.readouterr().out
    assert "B3 round trip: FAIL (the supervisor refused the start: Not started: the hub is down" in out
    assert store.sleeve("ping-pong-ls-binance").desired_state == "stopped"


def test_round_trip_puts_the_smoke_strategy_on_conservative_journaled_before_it_starts(monkeypatch, capfd):
    from sleeve_fund.paper import runtime

    monkeypatch.setattr(runtime, "entry_blocked", lambda *a, **k: (False, None))
    store = Store.in_memory(portfolio_gate=True)
    _perp(store)
    assert store.sleeve("ping-pong-ls-binance").risk_profile == "balanced"
    assert rt.run(store, "ping-pong-ls-binance", 600, sleep=_supervisor(store, "ping-pong-ls-binance")) == 0
    assert store.sleeve("ping-pong-ls-binance").risk_profile == "conservative"
    out = capfd.readouterr().out
    assert "Risk profile: balanced (2x) to conservative (1x) (journaled)" in out
    assert out.index("(journaled)") < out.index("start sent") < out.index("started:")
    actions = [(d["action"], d["actor"]) for d in reversed(store.decisions(limit=10))]
    assert actions[:2] == [("change_settings", rt.ACTOR), ("start", rt.ACTOR)]
    said = next(d["reason"] for d in store.decisions(limit=10) if d["action"] == "change_settings")
    assert "balanced (2x) to conservative (1x)" in said  # the old and the new value, journaled
    assert store.pending_reload("ping-pong-ls-binance") is None  # stopped when changed: nothing restarts


def test_round_trip_never_says_started_when_the_supervisor_refuses(monkeypatch, capfd):
    from sleeve_fund.paper import runtime

    monkeypatch.setattr(runtime, "entry_blocked", lambda *a, **k: (False, None))
    store = Store.in_memory(portfolio_gate=True)
    _perp(store)
    store.event("ping-pong-ls-binance", "error", "start_refused", "Not started: an older refusal")  # before the start

    def refuse(_):
        store.event("ping-pong-ls-binance", "error", "start_refused", "Not started: the hub is down")

    assert rt.run(store, "ping-pong-ls-binance", 1800, sleep=refuse) == 1
    out = capfd.readouterr().out
    assert "started: ping-pong-ls-binance running" not in out and "an older refusal" not in out
    assert "FAIL (the supervisor refused the start: Not started: the hub is down" in out


@pytest.mark.parametrize("case, why", [
    ("other", "allowlist"), ("running", "starts only a stopped"), ("holding", "only from flat"),
    ("off-paper", "not paper")])
def test_the_conservative_step_refuses_any_other_strategy_a_running_one_one_holding_or_one_off_paper(case, why):
    store = Store.in_memory(portfolio_gate=True)
    name = "rsi-bands-ls-binance" if case == "other" else "ping-pong-ls-binance"
    _perp(store, name, "running" if case == "running" else "stopped")
    if case == "holding":
        _fill(store, name, "BUY", "o1", T0)
    if case == "off-paper":
        store.create_account("smoke-other", "paper")
        store.assign_account(name, "smoke-other")
    with pytest.raises(ValueError, match=why):
        rt.to_conservative(store, name)
    assert store.sleeve(name).risk_profile == "balanced"
    assert [d for d in store.decisions(limit=10) if d["action"] == "change_settings"] == []
    assert [e for e in store.events(name, limit=50) if e["kind"] == "settings_change"] == []


def test_the_conservative_step_on_a_conservative_strategy_writes_nothing():
    from dataclasses import asdict

    store = Store.in_memory(portfolio_gate=True)
    _perp(store)
    store._update_sleeve("ping-pong-ls-binance", risk_profile="conservative")
    before = repr(sorted(asdict(store.sleeve("ping-pong-ls-binance")).items()))
    rt.to_conservative(store, "ping-pong-ls-binance")
    assert repr(sorted(asdict(store.sleeve("ping-pong-ls-binance")).items())) == before
    assert store.decisions(limit=10) == []
    assert [e for e in store.events("ping-pong-ls-binance", limit=50) if e["kind"] == "settings_change"] == []


def test_the_conservative_step_leaves_every_other_strategy_byte_identical(monkeypatch):
    from dataclasses import asdict

    from sleeve_fund.paper import runtime

    monkeypatch.setattr(runtime, "entry_blocked", lambda *a, **k: (False, None))
    store = Store.in_memory(portfolio_gate=True)
    _perp(store)
    for other, desired in (("rsi-bands-ls-binance", "stopped"), ("trend-perp", "running")):
        _perp(store, other, desired)
    others = {s.name: repr(sorted(asdict(s).items())) for s in store.sleeves(include_backtests=True)
              if s.name != "ping-pong-ls-binance"}
    assert rt.run(store, "ping-pong-ls-binance", 600, sleep=_supervisor(store, "ping-pong-ls-binance")) == 0
    assert {s.name: repr(sorted(asdict(s).items())) for s in store.sleeves(include_backtests=True)
            if s.name != "ping-pong-ls-binance"} == others
    assert {d["sleeve"] for d in store.decisions(limit=50)} == {"ping-pong-ls-binance"}
