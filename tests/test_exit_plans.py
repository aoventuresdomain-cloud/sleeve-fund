"""Stops and targets an open position works to after the PM edits them (review round 8, B8-1, M8-1).

The settings tab restarts a running strategy to apply an edit; these drive that restart's plan directly,
on a journaled entry at 100, since a backtest rests exits only for entries it made itself."""

from datetime import datetime, timedelta, timezone

import pytest

from sleeve_fund.dashboard import trading
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.strategies.base import r_target

from test_sleeve_runtime import _sleeve
from test_sleeve_runtime import store as _store_fixture

store = _store_fixture  # the same journal, on Postgres when TEST_DATABASE_URL is set

T0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
COST = 0.008  # the taker fee; no spread assumed and no live quotes
ATR_ENTRY = {"stop_frac": 0.0143, "stop_basis": "2 x the 14-bar average true range (0.715)",
             "tp_frac": round(r_target(2, 0.0143, COST), 6), "risk_amount": 30.19, "planned_r": 2.0,
             "stop_cfg": {"stop_atr": 2.0, "atr_bars": 14}}


def _enter(store, signal):
    rt = SleeveRuntime(store, "s1", now=lambda: T0)
    rt.on_order(order_id="E-1", side="BUY", qty=10.0, intent="entry", reason="Signal to be long", signal=signal)
    rt.on_fill(side="BUY", qty=10.0, price=100.0, fee=8.0, order_id="E-1", trade_id="T-1")


def _edit(store, text, minutes=5):
    store.event("s1", "info", "exits_change", f"Settings changed by PM: {text}. Restarting to apply them.",
                ts=T0 + timedelta(minutes=minutes))


def _restart(store, instrument, minutes=10, **exits):
    """A strategy with these exit settings restarted over the open position, as the settings edit does."""
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies import REGISTRY

    cls, config_cls = REGISTRY["buy_and_hold"]
    strat = cls(config_cls(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL"),
                           assumed_taker_fee=COST, **exits))
    strat.runtime = SleeveRuntime(store, "s1", now=lambda: T0 + timedelta(minutes=minutes))
    strat._entry_px, strat._entry_qty = 100.0, 10.0
    strat._restore_plan()
    return strat


def _applied(store):
    return [e for e in store.events("s1", limit=500) if e["kind"] in ("exits_applied", "stop_reset")]


@pytest.mark.parametrize("journaled_cfg", [True, False])
def test_a_target_only_edit_keeps_the_stop_to_the_tick(store, instrument, journaled_cfg):
    """Editing only the target used to drop the stop the entry set from the market and measure a new one
    from today's close: no stop for a while, then one at the wrong distance (B8-1). The stop now stays
    exactly where it was, and the target is set again around it. An entry journaled before its stop
    settings were is told apart by what the change says."""
    _sleeve(store)
    sig = ATR_ENTRY if journaled_cfg else {k: v for k, v in ATR_ENTRY.items() if k != "stop_cfg"}
    _enter(store, sig)
    _edit(store, "Take-profit 2R after costs to 3R after costs")
    strat = _restart(store, instrument, stop_atr=2.0, take_profit_r=3.0)
    assert strat._stop_frac == 0.0143 and strat._replan_pending is None
    assert strat._tp_frac == pytest.approx(r_target(3, 0.0143, COST))
    plan = store.exit_plan("s1", "E-1")
    assert plan["kind"] == "edit" and plan["stop_frac"] == 0.0143 and plan["planned_r"] == pytest.approx(3.0, abs=0.02)
    (event,) = _applied(store)
    assert "stop at 98.57, 1.4% below the entry" in event["message"]
    # The next restart reads the plan back as it is: nothing is set again and nothing is said.
    again = _restart(store, instrument, minutes=20, stop_atr=2.0, take_profit_r=3.0)
    assert (again._stop_frac, again._tp_frac) == (strat._stop_frac, round(strat._tp_frac, 6))
    assert len(_applied(store)) == 1


def test_a_new_market_stop_keeps_the_old_one_working_and_sits_at_the_swing_low(store, instrument):
    """A new ATR or swing-low stop needs bars to be set. Until then the old stop keeps working; when
    it is set, it sits at the level its words name (the swing low itself), as a share of the entry."""
    _sleeve(store)
    _enter(store, ATR_ENTRY)
    _edit(store, "Stop-loss 2 average true ranges (14 bars) below the entry to at the lowest low of 3 bars")
    strat = _restart(store, instrument, stop_swing_bars=3, take_profit_r=2.0)
    assert strat._stop_frac == 0.0143 and strat._replan_pending == ("edit", store.last_event("s1", ("exits_change",))["id"])
    assert _applied(store) == []
    strat._lows.extend([99.4, 99.0, 100.2])
    strat._replan(101.0)
    assert 100.0 * (1 - strat._stop_frac) == pytest.approx(99.0)  # at the low, not 1 - 99/101 below the entry
    assert strat._tp_frac == pytest.approx(r_target(2, 0.01, COST)) and strat._replan_pending is None
    assert "at the lowest low of the last 3 bars (99)" in strat._stop_basis
    assert store.exit_plan("s1", "E-1")["stop_frac"] == pytest.approx(0.01)


def test_a_market_stop_set_after_entry_only_tightens(store, instrument):
    """A stop set from the market on a position already open can't loosen the one working: widening a
    stop is a % edit, which the settings tab checks against the position's size (M8-1)."""
    _sleeve(store)
    _enter(store, ATR_ENTRY)
    _edit(store, "Stop-loss 2 average true ranges (14 bars) below the entry to at the lowest low of 3 bars")
    strat = _restart(store, instrument, stop_swing_bars=3)
    strat._lows.extend([97.0, 99.0, 100.2])
    strat._replan(101.0)
    assert strat._stop_frac == 0.0143 and strat._stop_basis.startswith("kept: the new setting")


def test_a_new_percent_stop_applies_at_once(store, instrument):
    _sleeve(store)
    _enter(store, ATR_ENTRY)
    _edit(store, "Stop-loss 2 average true ranges (14 bars) below the entry to 2% below the entry")
    strat = _restart(store, instrument, stop_loss=0.02, take_profit_r=2.0)
    assert strat._stop_frac == 0.02 and strat._tp_frac == pytest.approx(r_target(2, 0.02, COST))
    assert strat._replan_pending is None


def test_trades_measure_r_on_the_risk_after_an_edit_and_mark_it(store):
    """A looser stop risks more than the entry did. R is measured on the larger of the two, and the
    trade says its exits were edited, with the plan after the edit (M8-1)."""
    _sleeve(store)
    _enter(store, ATR_ENTRY)
    store.set_exit_plan("s1", "E-1", kind="edit", stop_frac=0.03, tp_frac=0.08, risk_amount=45.6, planned_r=1.4,
                        ts=T0 + timedelta(minutes=10))
    rt = SleeveRuntime(store, "s1", now=lambda: T0 + timedelta(hours=2))
    rt.on_order(order_id="X-1", side="SELL", qty=10.0, intent="stop_loss", reason="Stop-loss", signal={})
    rt.on_fill(side="SELL", qty=10.0, price=97.0, fee=7.76, order_id="X-1", trade_id="T-2")
    (t,) = trading.trips(store.fills("s1"), [], trading.orders_by_id(store, "s1"), store.exit_plans("s1"))
    assert t["r"] == pytest.approx(t["pnl"] / 45.6) and t["planned_r"] == 1.4 and t["exits_edited"]
    assert ("1R now", "45.60") in t["entry_items"] and ("Stop now", "3.0% below the entry") in t["entry_items"]
    assert not any(label == "Stop cfg" for label, _ in t["entry_items"])
