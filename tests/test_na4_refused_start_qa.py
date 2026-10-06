"""NA-4 (Advisor 6 Oct 16:53, "S6 applies to every refused start that holds a position"): strict xfails for the
stop-safety PR (Platform Engineer 2), written by QA from the #146 round (late-146.md) before any build.

The ruling: a strategy that the supervisor refuses to start while it still holds a position runs EXITS ONLY, gets
a SAFETY STOP from the mark (half the remaining distance to liquidation; half the distance to zero at 1x long or on
spot), raises an alert, and its position is never left unwatched. On #146 (ebe39c5) it is stopped, said to need an
engineer, never flattened (pinned in v2-p1/hub-146-scripts/test_hub_146_qa.py), but nothing watches its stop.

Two refusals exist today, both perp-only:
- the hub's venue-candle refusal (QA P1-C10, Supervisor._refused / check_hub_bar_spec);
- the weight-sized-perp refusal (check_perp_sizing) when no flatten is waiting.
Spot can't reach either (Binance lists perpetuals only; the weight refusal is perp-only), so the spot and 1x-long
"half way to zero" level is pinned by test_stop_safety_xfails.py's S4/S5 tests, which this ruling generalises; the
level itself (S3/S4) is likewise pinned there (`_safety_level`) and applies unchanged to an exits-only start.

ASSUMED INTERFACES (adapt the names, never the assertions): the alert is a warning or error event (Store.alerts)
whose message mentions the "safety stop"; an exits-only start is a supervisor start (a process is launched) and the
strategy's desired state stays "running"; build_node builds a node for such a strategy rather than raising.

Run from a worktree root (no Postgres needed):
  PYTHONDONTWRITEBYTECODE=1 BACKTEST_ISOLATE=0 PYTHONPATH=<worktree> /tmp/claude-0/v/bin/python -m pytest -q \
    -p no:cacheprovider -rxXs /mnt/project-files/sleeve-fund/quant-review/stop-safety-xfails/test_na4_refused_start_xfails.py
On #146 ebe39c5: 4 passed (guards), 12 xfailed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

REASON = "stop-safety PR (NA-4, generalises S6)"


def xf(what: str):
    return pytest.mark.xfail(strict=True, reason=f"{REASON}: {what}")


REFUSALS = {  # (venue, bar_spec, strategy, params): refused at start on #146 when stored
    "venue-candles": ("binance", "1-DAY-LAST-EXTERNAL", "buy_and_hold", {"market": "perp"}),
    "perp-weight": ("binance", "1-HOUR-LAST-INTERNAL", "donchian", {"market": "perp"}),
}
PROFILES = {"1x": "conservative", "3x": "aggressive"}


def _supervised(tmp_path, monkeypatch, refusal, profile, side, held=True):
    from sleeve_fund import supervisor
    from sleeve_fund.store import Store

    venue, spec, strategy, params = REFUSALS[refusal]
    store = Store(f"sqlite:///{tmp_path}/na4.db")
    for name in ("held", "flat"):
        store.create_sleeve(name=name, strategy=strategy, instrument="BTC/USDT", bar_spec=spec, starting_balance=10_000,
                            venue=venue, params=params, risk_profile=profile)
    if held:
        word = "BUY" if side > 0 else "SELL"
        store.record_order("held", order_id="o1", side=word, qty=0.05, intent="entry", reason="carried",
                           signal={"stop_frac": 0.02})
        store.record_fill("held", side=word, qty=0.05, price=60_000.0, fee=1.5, order_id="o1", trade_id="t1")
    started = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: started.append(a[0][-1]) or
                        SimpleNamespace(pid=1, poll=lambda: None, send_signal=lambda *a: None, wait=lambda **k: 0))
    sup = supervisor.Supervisor(store)
    for _ in range(3):
        sup.step()
    return store, started


C10 = xf("the hub's venue-candle refusal (QA P1-C10) comes with #146; a 1x stopless model is otherwise allowed, so "
         "until then nothing refuses it and the flat one starts too (Platform 1 routes C10 through "
         "Supervisor._exits_only)")
CASES = [pytest.param(r, lev, side, marks=C10) if (r, lev) == ("venue-candles", "1x") else (r, lev, side)
         for r in REFUSALS for lev in PROFILES for side in (1, -1)]
CASE_IDS = [f"{r}-{lev}-{'long' if s > 0 else 'short'}" for r in REFUSALS for lev in PROFILES for s in (1, -1)]


@pytest.mark.parametrize("refusal, lev, side", CASES, ids=CASE_IDS)
def test_na4_a_refused_start_holding_a_position_runs_exits_only(tmp_path, monkeypatch, refusal, lev, side):
    store, started = _supervised(tmp_path, monkeypatch, refusal, PROFILES[lev], side)
    assert "flat" not in started and store.sleeve("flat").desired_state == "stopped"
    assert started.count("held") == 1, started
    assert store.sleeve("held").desired_state == "running"


@pytest.mark.parametrize("refusal", list(REFUSALS))
def test_na4_the_alert_names_the_safety_stop(tmp_path, monkeypatch, refusal):
    store, _ = _supervised(tmp_path, monkeypatch, refusal, "aggressive", 1)
    alerts = [a for a in store.alerts(limit=200) if a["sleeve"] == "held"]
    assert any("safety stop" in a["message"].lower() for a in alerts), [a["message"] for a in alerts]


@xf("a node is built for a venue-candle strategy that still holds a position (exits only, on the hub's minutes), "
    "rather than raising at build as it does for a flat one (QA P1-C10: #146 and Platform 1)")
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_na4_build_node_builds_an_exits_only_node_for_a_held_venue_candle_strategy(tmp_path, side):
    from sleeve_fund.paper import node as node_mod
    from sleeve_fund.paper.config import SleeveConfig
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/n.db")
    store.create_sleeve(name="held", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, venue="binance", params={"market": "perp"}, risk_profile="aggressive")
    word = "BUY" if side > 0 else "SELL"
    store.record_fill("held", side=word, qty=0.05, price=60_000.0, fee=1.5, order_id="o1", trade_id="t1")
    cfg = SleeveConfig(name="held", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                       starting_balance=10_000.0, params={"market": "perp"}, venue="BINANCE")
    node_mod.build_node(cfg, log_level="ERROR", runtime=SleeveRuntime(store, "held"), asset_fetch=dict,
                        hub=("127.0.0.1", 1))


# --- guards: what #146 already does and the build must keep ---------------------------------------------------


@pytest.mark.parametrize("refusal", list(REFUSALS))
def test_guard_na4_a_refused_start_never_flattens_or_drops_the_position(tmp_path, monkeypatch, refusal):
    """Passes on #146: refused, the journal's position is untouched and no order is sent."""
    store, started = _supervised(tmp_path, monkeypatch, refusal, "aggressive", 1)
    assert store.journal_book("held", 10_000)["qty"] == pytest.approx(0.05)
    assert [o["order_id"] for o in store.orders("held", limit=10)] == ["o1"]


@pytest.mark.parametrize("refusal", list(REFUSALS))
def test_guard_na4_a_refused_start_raises_an_alert(tmp_path, monkeypatch, refusal):
    """Passes on #146: the refusal is an error event, so it reaches the alerts inbox."""
    store, _ = _supervised(tmp_path, monkeypatch, refusal, "aggressive", 1)
    assert [a for a in store.alerts(limit=200) if a["sleeve"] == "held" and a["kind"] == "start_refused"]
