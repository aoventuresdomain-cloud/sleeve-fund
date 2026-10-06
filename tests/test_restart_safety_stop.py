"""The restart edge for a position whose stop can't be restored (HoE, 6 Oct; Independent Quant Advisor agreed; QA P1-S1,
S7): its entry journaled no stop, and an ATR or swing stop needs more bars than a restart has. Until the model's own
stop is set again it works to a safety stop half the REMAINING distance from the first mark after the restart to its
isolated liquidation price (half way to zero where there is none), so it always sits strictly between the two; no new
entry opens; an incident is raised; the position is never closed for it. The model's stop, once set, replaces the
safety stop only if it is tighter, and entries open again. A refused start still holding a position runs the same
way for its exits only, and never opens again."""

from types import SimpleNamespace

import pytest
from sqlalchemy import update

from sleeve_fund import markets, risk
from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store, orders_t

PERP = {"market": "perp"}
SIX_HOURS = 6 * 3600


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path}/t.db")


def _held(store, instrument, journaled, stop_atr=2.0):
    """A balanced (2x) perp buy-and-hold with an ATR stop, holding after its first run; `journaled` False strips the
    stop its entry journaled, as an entry from before stops were recorded."""
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, params={**PERP, "stop_atr": stop_atr}, risk_profile="balanced")
    prices = synthetic_ohlcv(days=60, seed=3)
    run_backtest("buy_and_hold", prices.iloc[:25], instrument, {**PERP, "stop_atr": stop_atr},
                 runtime=SleeveRuntime(store, "s1", tick_seconds=SIX_HOURS))
    (entry,) = [o for o in store.orders("s1") if o["intent"] == "entry"]
    if not journaled:
        sig = {k: v for k, v in entry["signal"].items() if k not in ("stop_frac", "stop_basis")}
        with store.engine.begin() as c:
            c.execute(update(orders_t).where(orders_t.c.order_id == entry["order_id"]).values(signal=sig))
    return prices, entry


def _restart(store, instrument, prices, stop_atr):
    run_backtest("buy_and_hold", prices, instrument, {**PERP, "stop_atr": stop_atr},
                 runtime=SleeveRuntime(store, "s1", tick_seconds=SIX_HOURS))


def _kinds(store):
    return [e["kind"] for e in reversed(store.events("s1", limit=500))]


def _incident(store):
    (e,) = [e for e in store.events("s1", limit=500) if e["kind"] == "incident"]
    return e


def _level(message):
    return float(message.split("safety stop at ")[1].split(" ")[0].replace(",", ""))  # "36,123.4," -> 36123.4


def test_a_stop_that_cant_be_restored_gets_a_safety_stop_half_way_from_the_mark_to_liquidation(store, instrument):
    prices, entry = _held(store, instrument, journaled=False)
    book = store.journal_book("s1", 10_000)
    _restart(store, instrument, prices.iloc[25:27], 2.0)  # two bars: too few for the 14-bar ATR
    liq = markets.isolated_liquidation(book["cash"], book["qty"], book["entry_px"], risk.profile("balanced").max_leverage,
                                       markets.terms(PERP).maintenance_margin)
    mark = float(prices["close"].iloc[25])  # the first price after the restart
    assert liq is not None and liq < mark
    alert = _incident(store)
    assert alert["level"] == "error" and "Incident, s1:" in alert["message"]
    assert _level(alert["message"]) == pytest.approx(mark - 0.5 * (mark - liq), rel=1e-5)
    assert f"from the {mark:,.6g} mark" in alert["message"] and "not closed" in alert["message"]
    assert [f["side"] for f in store.fills("s1")] == ["BUY"]  # never flattened for it
    assert "stop_restored" not in _kinds(store)


def test_a_stop_restored_normally_sets_no_safety_stop(store, instrument):
    prices, _ = _held(store, instrument, journaled=True)
    _restart(store, instrument, prices.iloc[25:27], 2.0)
    assert "incident" not in _kinds(store)


@pytest.mark.parametrize("stop_atr, tighter", [(1.0, True), (10.0, False)])
def test_the_models_stop_replaces_the_safety_stop_only_if_tighter_then_entries_open_again(store, instrument, stop_atr,
                                                                                          tighter):
    prices, entry = _held(store, instrument, journaled=False)
    _restart(store, instrument, prices.iloc[25:], stop_atr)  # the restart's own ATR stop, set once there are bars
    kinds = _kinds(store)
    assert kinds.index("incident") < kinds.index("stop_reset") < kinds.index("stop_restored")
    plan = store.exit_plan("s1", entry["order_id"])
    pct = 1 - _level(_incident(store)["message"]) / store.journal_book("s1", 10_000)["entry_px"]
    if tighter:
        assert plan["stop_frac"] < pct and "kept" not in plan["basis"]
    else:
        assert plan["stop_frac"] == pytest.approx(pct, abs=1e-4) and plan["basis"].startswith("kept")


def test_no_entry_opens_while_the_safety_stop_works(store, instrument):
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies import REGISTRY

    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, params={**PERP, "stop_atr": 2.0})
    cls, config_cls = REGISTRY["buy_and_hold"]
    strat = cls(config_cls(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL"),
                           assumed_taker_fee=0.008, stop_atr=2.0))
    strat.runtime = SleeveRuntime(store, "s1")
    strat.runtime.status = "running"
    sent = []
    strat._submit = lambda *a, **k: sent.append(a)
    strat._safety_stop, strat._entry_px, strat._entry_side = True, 60_000.0, 1
    strat._open(-1, SimpleNamespace(ts_event=0), "flip", {})
    assert sent == [] and "entry_held_safety_stop" in _kinds(store)


def _strategy(store, instrument, profile, params, status=("running", "")):
    """A buy_and_hold with an ATR stop, attached to a strategy in the store but not run: the restart's safety stop
    is placed by hand (_safety_stop_on_restore, then _place_safety_stop at the first price)."""
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies import REGISTRY

    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, params=params, risk_profile=profile)
    store.set_status("s1", *status)
    cls, config_cls = REGISTRY["buy_and_hold"]
    strat = cls(config_cls(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL"),
                           assumed_taker_fee=0.0005, **params))
    strat.runtime = SleeveRuntime(store, "s1")
    strat.runtime.now = lambda: None
    return strat


def _restart_at(strat, entry, qty, mark, leverage):
    """Holding qty from entry (a perp's spot-style cash: balance less qty x entry), restarted with no stop restored,
    first priced at mark. The safety stop's price."""
    strat._entry_px, strat._entry_side, strat._entry_qty = entry, 1 if qty > 0 else -1, abs(qty)
    book = {"cash": 10_000 - qty * entry, "qty": qty}
    strat._safety_stop_on_restore(book)
    strat._place_safety_stop(mark)
    liq = (markets.isolated_liquidation(book["cash"], qty, entry, leverage, markets.terms(PERP).maintenance_margin)
           if strat._margin else None)
    return entry * (1 - strat._entry_side * strat._stop_frac), liq


@pytest.mark.parametrize("profile, qty, mark", [
    ("balanced", 2 * 10_000 / 60_000, 42_000.0),  # QA P1-S1: entered at 60,000, restarted at 42,000
    ("conservative", 10_000 / 60_000, 42_000.0),  # 1x long: no liquidation price, half way to zero
    ("conservative", -10_000 / 60_000, 70_000.0),  # 1x short: half way to its liquidation price
    ("balanced", -2 * 10_000 / 60_000, 75_000.0),
])
def test_the_safety_stop_is_half_the_remaining_distance_from_the_mark_to_liquidation(store, instrument, profile, qty,
                                                                                     mark):
    strat = _strategy(store, instrument, profile, {**PERP, "allow_short": True, "stop_atr": 2.0})
    stop, liq = _restart_at(strat, 60_000.0, qty, mark, risk.profile(profile).max_leverage)
    target = liq if liq is not None else 0.0
    assert (liq is None) == (profile == "conservative" and qty > 0)
    assert stop == pytest.approx(mark - 0.5 * (mark - target))
    assert min(mark, target) < stop < max(mark, target)  # always between the mark and liquidation
    message = _incident(store)["message"]
    assert f"safety stop at {stop:,.6g}" in message and "not closed" in message and strat._safety_stop


def test_an_entry_based_stop_would_have_fired_at_once_the_mark_based_one_does_not(store, instrument):
    """Half way from the 60,000 entry to a 2x long's liquidation is about 45,000: above the 42,000 mark, so the old
    safety stop sold the position on the first price after the restart."""
    strat = _strategy(store, instrument, "balanced", {**PERP, "stop_atr": 2.0})
    stop, liq = _restart_at(strat, 60_000.0, 2 * 10_000 / 60_000, 42_000.0, 2.0)
    assert 60_000 - 0.5 * (60_000 - liq) > 42_000 > stop > liq


def test_a_restored_stop_tighter_than_the_safety_stop_is_kept(store, instrument):
    strat = _strategy(store, instrument, "balanced", {**PERP, "stop_loss": 0.02})
    strat._stop_frac = 0.02  # restored from the journal; a fixed stop needs no replan
    strat._entry_px, strat._entry_side, strat._entry_qty = 60_000.0, 1, 0.3
    strat._exits_only = False
    store.set_status("s1", "paused", "exits only: places no stop")
    strat._safety_stop_on_restore({"cash": 10_000 - 0.3 * 60_000, "qty": 0.3})
    strat._place_safety_stop(59_000.0)
    assert strat._stop_frac == 0.02 and "keeps its restored stop at 58,800" in _incident(store)["message"]


def test_spot_gets_a_safety_stop_half_way_from_the_mark_to_zero(store, instrument):
    strat = _strategy(store, instrument, "aggressive", {"stop_atr": 2.0})
    assert not strat._margin
    stop, liq = _restart_at(strat, 60_000.0, 0.1, 50_000.0, 1.0)
    assert liq is None and stop == pytest.approx(25_000.0) and strat._safety_stop


def test_a_start_for_its_exits_only_sets_a_safety_stop_and_never_opens_again(store, instrument):
    strat = _strategy(store, instrument, "balanced", PERP,
                      status=("paused", "exits only: places no stop, so it is capped at 1x"))
    stop, liq = _restart_at(strat, 60_000.0, 0.2, 58_000.0, 2.0)
    assert strat._exits_only and stop == pytest.approx(58_000 - 0.5 * (58_000 - liq))
    message = _incident(store)["message"]
    assert "started for its exits only (places no stop, so it is capped at 1x)" in message
    assert "until the model's own stop" not in message
    sent = []
    strat._submit = lambda *a, **k: sent.append(a)
    strat._entry_px = None  # even once the position has closed: only the PM starts it again
    strat._open(1, SimpleNamespace(ts_event=0), "entry", {})
    assert sent == []


def test_the_supervisor_starts_a_refused_strategy_still_holding_for_its_exits_only(tmp_path, monkeypatch):
    """QA P1-S7, Advisor NA-4: a refused start that still holds a position is never left unwatched."""
    from sleeve_fund import supervisor
    from test_supervisor import FakePopen

    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="pp", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"rise": 0.01, "dip": 0.005, "market": "perp"}, venue="BINANCE",
                        risk_profile="balanced", desired_state="running")
    store.record_fill("pp", side="BUY", qty=0.2, price=60_000.0, fee=6.0, order_id="o1", trade_id="t1")
    started = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: started.append(a) or FakePopen())
    sup = supervisor.Supervisor(store)
    sup.step()
    sup.step()
    s = store.sleeve("pp")
    assert started and s.desired_state == "running" and s.status == "paused" and s.status_reason.startswith("exits only")
    incidents = [e for e in store.events("pp", limit=50) if e["kind"] == "incident"]
    assert len(incidents) == 1 and incidents[0]["level"] == "error" and "places no stop" in incidents[0]["message"]
    assert s.risk_profile == "balanced"  # refused, not silently changed


def _watched_rows(store, open_only=True):
    from sleeve_fund.store import OPEN_ORDER_STATUSES

    return [o for o in store.orders("s1", statuses=OPEN_ORDER_STATUSES if open_only else None, limit=100)
            if (o.get("signal") or {}).get("watched")]


def test_the_watched_stop_row_shows_the_safety_stop_once_with_its_incident_and_is_never_a_venue_order(store,
                                                                                                    instrument):
    """P1-U35 (HoE, QA): paper's stop is watched in the process, so the journal shows it as one open stop_loss row at
    its level, linked to its incident. Asked again with nothing changed it is not placed twice; a restart cancels it
    with the rest of the old process's orders and the new process places one, so one is open after a restart; replay
    and a stop that fires closes it. (Replay leaves it out of the orders it reports sent.)"""
    strat = _strategy(store, instrument, "balanced", {**PERP, "allow_short": True, "stop_atr": 2.0})
    stop, _ = _restart_at(strat, 60_000.0, 0.2, 58_000.0, 2.0)
    (row,) = _watched_rows(store)
    assert (row["intent"], row["order_type"], row["side"], row["qty"]) == ("stop_loss", "STOP (watched)", "SELL", 0.2)
    assert row["signal"]["stop_px"] == pytest.approx(stop) and row["signal"]["incident"] == _incident(store)["id"]
    strat._sync_watched_stop()
    strat._sync_watched_stop()
    assert [r["order_id"] for r in _watched_rows(store)] == [row["order_id"]]  # never double-placed

    SleeveRuntime(store, "s1").on_start(0.008)  # a restart: the old process's row goes with it
    assert _watched_rows(store) == []
    strat._watched = None  # the new process knows nothing of the old row, and places its own
    strat._sync_watched_stop()
    (row,) = _watched_rows(store)

    sent = []
    strat._busy, strat._mark = (lambda: False), (lambda: (0.0, 0.0, 0.0, 0.0))  # no venue in this harness
    strat._sell_all = lambda *a, **k: sent.append(a[0])
    assert strat._check_exits(stop * 0.99)  # through the stop: paper sends its market stop-loss
    assert sent == ["stop_loss"] and _watched_rows(store) == []
    (done,) = [r for r in _watched_rows(store, open_only=False) if r["order_id"] == row["order_id"]]
    assert done["status"] == "canceled" and done["message"].startswith("stop fired at ")


def test_a_second_restart_further_against_the_position_never_loosens_the_safety_stop(store, instrument):
    """QA SG4: a 2x short restarted at 65,000 gets its safety stop half way to liquidation; a deploy after the price
    has moved further against it, with no fill since, keeps that stop rather than measuring a looser one from the new
    mark, and writes no second incident."""
    strat = _strategy(store, instrument, "balanced", {**PERP, "allow_short": True, "stop_atr": 2.0})
    qty = -2 * 10_000 / 60_000
    first, liq = _restart_at(strat, 60_000.0, qty, 65_000.0, 2.0)
    assert first < liq
    SleeveRuntime(store, "s1").on_start(0.008)  # the deploy: the old process's orders go with it
    strat._watched, strat._stop_frac = None, None  # the new process restores no stop of its own
    second, _ = _restart_at(strat, 60_000.0, qty, 70_000.0, 2.0)
    assert second == pytest.approx(first)  # not 70,000 + half the way to liquidation
    assert "never loosens" in strat._stop_basis or "before the last restart" in strat._stop_basis
    _incident(store)  # still exactly one
