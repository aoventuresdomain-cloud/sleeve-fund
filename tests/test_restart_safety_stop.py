"""The restart edge for a perp position whose stop can't be restored (HoE, 6 Oct; Independent Quant Advisor agreed): its
entry journaled no stop, and an ATR or swing stop needs more bars than a restart has. Until the model's own stop is set
again it works to a safety stop half way from its entry to its isolated liquidation price, set at once with no bars
needed; no new entry opens; an error event alerts; the position is never closed for it. The model's stop, once set,
replaces the safety stop only if it is tighter, and entries open again."""

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


def test_a_stop_that_cant_be_restored_gets_a_safety_stop_half_way_to_liquidation_and_an_alert(store, instrument):
    prices, entry = _held(store, instrument, journaled=False)
    book = store.journal_book("s1", 10_000)
    _restart(store, instrument, prices.iloc[25:27], 2.0)  # two bars: too few for the 14-bar ATR
    liq = markets.isolated_liquidation(book["cash"], book["qty"], book["entry_px"], risk.profile("balanced").max_leverage,
                                       markets.terms(PERP).maintenance_margin)
    assert liq is not None and liq < book["entry_px"]
    stop = book["entry_px"] - 0.5 * (book["entry_px"] - liq)
    (alert,) = [e for e in store.alerts() if e["kind"] == "stop_not_restored"]
    assert alert["level"] == "error"
    assert f"safety stop at {stop:,.6g}" in alert["message"] and "not closed" in alert["message"]
    assert [f["side"] for f in store.fills("s1")] == ["BUY"]  # never flattened for it
    assert "stop_restored" not in _kinds(store)


def test_a_stop_restored_normally_sets_no_safety_stop(store, instrument):
    prices, _ = _held(store, instrument, journaled=True)
    _restart(store, instrument, prices.iloc[25:27], 2.0)
    assert "stop_not_restored" not in _kinds(store)


@pytest.mark.parametrize("stop_atr, tighter", [(1.0, True), (10.0, False)])
def test_the_models_stop_replaces_the_safety_stop_only_if_tighter_then_entries_open_again(store, instrument, stop_atr,
                                                                                          tighter):
    prices, entry = _held(store, instrument, journaled=False)
    _restart(store, instrument, prices.iloc[25:], stop_atr)  # the restart's own ATR stop, set once there are bars
    kinds = _kinds(store)
    assert kinds.index("stop_not_restored") < kinds.index("stop_reset") < kinds.index("stop_restored")
    plan = store.exit_plan("s1", entry["order_id"])
    safety = next(e for e in store.events("s1", limit=500) if e["kind"] == "stop_not_restored")["message"]
    pct = float(safety.split("safety stop at ")[1].split("(")[1].split("%")[0]) / 100
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
