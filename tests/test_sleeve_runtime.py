"""The sleeve runtime (journal, PM controls, risk guard) driven through a real backtest."""

import os

import numpy as np
import pytest

from sleeve_fund import risk
from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store

SIX_HOURS = 6 * 3600


@pytest.fixture
def store(tmp_path):
    # CI also runs these against Postgres (the production engine) via TEST_DATABASE_URL.
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine)
    return Store(f"sqlite:///{tmp_path}/t.db")


def crash_prices(days=60, crash_day=30, drop=0.7):
    df = synthetic_ohlcv(days=days, seed=5, vol=0.002, drift=0.0)
    factor = np.where(np.arange(days) >= crash_day, 1 - drop, 1.0)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].mul(factor, axis=0)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def _sleeve(store, name="s1", profile="balanced"):
    store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile=profile)


def _run(store, instrument, prices, name="s1"):
    rt = SleeveRuntime(store, name, tick_seconds=SIX_HOURS)
    run_backtest("buy_and_hold", prices, instrument, runtime=rt)
    return store.sleeve(name)


def test_drawdown_halts_flattens_and_journals(store, instrument):
    _sleeve(store)
    s = _run(store, instrument, crash_prices())
    assert s.status == "halted" and "drawdown" in s.status_reason
    fills = store.fills("s1")
    assert [f["side"] for f in reversed(fills)] == ["BUY", "SELL"]  # bought, then flattened
    assert all(f["fee"] > 0 for f in fills)
    assert any(e["kind"] == "risk_halt" for e in store.events("s1"))
    eq = store.equity_series("s1")
    assert len(eq) > 50 and eq[0]["benchmark"] < 10_000  # benchmark pays the entry fee
    # The balanced profile caps the position at 33% of equity, so a 70% crash costs about 23%.
    assert eq[-1]["equity"] == pytest.approx(10_000 * (1 - 0.33 * 0.7), rel=0.05)


def test_pm_pause_blocks_entries(store, instrument):
    _sleeve(store)
    store.command("s1", "pause", "testing pause")
    s = _run(store, instrument, synthetic_ohlcv(days=20, seed=1))
    # The first tick applies the pause before the first bar can buy... or flattens what it bought.
    assert s.status == "paused"
    assert store.pending_commands("s1") == []
    assert any(d["action"] == "pause" for d in store.decisions("s1"))


def test_commands_need_a_reason(store):
    _sleeve(store)
    with pytest.raises(ValueError):
        store.command("s1", "flatten", "  ")
    with pytest.raises(ValueError):
        store.command("s1", "launch", "x")


@pytest.mark.parametrize(
    "equity,peak,day_open,expected",
    [
        (100, 100, 100, None),
        (79, 100, 79.5, "halt"),  # 21% drawdown
        (94, 100, 100, "pause_day"),  # 6% daily loss, 6% drawdown
        (96, 100, 100, None),
        (float("nan"), 100, 100, "halt"),  # bad data fails closed
        (0, 100, 100, "halt"),
    ],
)
def test_guard_balanced(equity, peak, day_open, expected):
    b = risk.check(risk.profile("balanced"), equity, peak, day_open)
    assert (b.action if b else None) == expected


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        risk.profile("yolo")


def test_journal_book_replays_fills_with_fees(store):
    _sleeve(store)
    store.record_fill("s1", side="BUY", qty=1.0, price=100.0, fee=0.8, order_id="o1", trade_id="t1")
    store.record_fill("s1", side="BUY", qty=1.0, price=200.0, fee=1.6, order_id="o2", trade_id="t2")
    store.record_fill("s1", side="SELL", qty=0.5, price=300.0, fee=1.2, order_id="o3", trade_id="t3")
    book = store.journal_book("s1", 10_000)
    assert book["fills"] == 3
    assert book["qty"] == pytest.approx(1.5)
    assert book["entry_px"] == pytest.approx(150.0)  # a partial sell keeps the average entry
    assert book["cash"] == pytest.approx(10_000 - 100.8 - 201.6 + 148.8)
    store.record_fill("s1", side="SELL", qty=1.5, price=300.0, fee=3.6, order_id="o4", trade_id="t4")
    assert store.journal_book("s1", 10_000)["entry_px"] is None


def test_restart_restores_the_book_and_reconciles(store, instrument):
    _sleeve(store)
    prices = synthetic_ohlcv(days=40, seed=3)
    _run(store, instrument, prices.iloc[:20])  # buys and holds
    before = store.journal_book("s1", 10_000)
    assert before["qty"] > 0

    rt = SleeveRuntime(store, "s1", tick_seconds=SIX_HOURS)  # the "restart"
    run_backtest("buy_and_hold", prices.iloc[20:], instrument, runtime=rt)
    kinds = [e["kind"] for e in store.events("s1")]
    assert "restore" in kinds and "reconcile" in kinds and "reconcile_mismatch" not in kinds
    # Still long from before, so no second buy (and no second entry fee).
    assert [f["side"] for f in store.fills("s1")] == ["BUY"]
    eq = store.last_equity("s1")
    assert eq["qty"] == pytest.approx(before["qty"]) and eq["cash"] == pytest.approx(before["cash"], abs=0.01)
    assert store.sleeve("s1").status == "stopped"  # the backtest ended cleanly, not halted


def test_flatten_sells_a_restored_position(store, instrument):
    _sleeve(store)
    prices = synthetic_ohlcv(days=40, seed=3)
    _run(store, instrument, prices.iloc[:20])
    store.command("s1", "flatten", "test carry-over exit")
    rt = SleeveRuntime(store, "s1", tick_seconds=SIX_HOURS)
    run_backtest("buy_and_hold", prices.iloc[20:], instrument, runtime=rt)
    assert [f["side"] for f in reversed(store.fills("s1"))] == ["BUY", "SELL"]
    assert store.journal_book("s1", 10_000)["qty"] == pytest.approx(0.0, abs=1e-8)


def test_reconcile_mismatch_halts_without_trading(store, instrument):
    _sleeve(store)
    rt = SleeveRuntime(store, "s1", tick_seconds=SIX_HOURS)
    # A fill the engine never saw: the journal and the engine now disagree.
    store.record_fill("s1", side="BUY", qty=0.1, price=100.0, fee=0.08, order_id="x", trade_id="x")
    run_backtest("buy_and_hold", synthetic_ohlcv(days=20, seed=1), instrument, runtime=rt)
    s = store.sleeve("s1")
    assert s.status == "halted" and "reconciliation" in s.status_reason
    assert any(e["kind"] == "reconcile_mismatch" for e in store.events("s1", min_level="error"))
    assert len(store.fills("s1")) == 1  # only the bogus row: nothing traded, nothing corrected


def test_reconcile_runs_every_24_hours(store):
    from datetime import timedelta

    _sleeve(store)
    clock = [utcnow_fixed()]
    rt = SleeveRuntime(store, "s1", now=lambda: clock[0])
    assert rt.reconcile_due()
    assert rt.reconcile(cash=10_000, qty=0)
    clock[0] += timedelta(hours=23)
    assert not rt.reconcile_due()
    clock[0] += timedelta(hours=1)
    assert rt.reconcile_due()


def utcnow_fixed():
    from datetime import datetime, timezone

    return datetime(2026, 10, 3, tzinfo=timezone.utc)


def test_every_order_is_journaled_with_its_reason_before_it_fills(store, instrument):
    _sleeve(store)
    _run(store, instrument, crash_prices())
    orders = list(reversed(store.orders("s1")))
    assert [(o["side"], o["intent"], o["status"]) for o in orders] == [("BUY", "entry", "filled"),
                                                                          ("SELL", "risk_halt", "filled")]
    entry, halt = orders
    assert entry["reason"].startswith("Buy and hold") and entry["signal"]["sized_by"] == "balanced risk profile cap"
    assert halt["reason"].startswith("Risk halt: drawdown")
    fills = {f["order_id"]: f for f in store.fills("s1")}
    for o in orders:  # the order rows agree with the fills journal
        f = fills[o["order_id"]]
        assert o["filled_qty"] == pytest.approx(f["qty"]) and o["avg_px"] == pytest.approx(f["price"])
        assert o["fee"] == pytest.approx(f["fee"])


def test_signal_values_recorded_at_entry_and_exit(store, instrument):
    store.create_sleeve(name="tf", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, params={"fast": 5, "slow": 20}, risk_profile="aggressive")
    rt = SleeveRuntime(store, "tf", tick_seconds=SIX_HOURS)
    run_backtest("trend_filter", synthetic_ohlcv(days=200, seed=3, vol=0.03), instrument,
                 params={"fast": 5, "slow": 20}, runtime=rt)
    orders = store.orders("tf")
    entries = [o for o in orders if o["intent"] == "entry"]
    exits = [o for o in orders if o["intent"] == "exit"]
    assert entries and exits
    assert "5-bar average" in entries[0]["reason"] and "above the 20-bar average" in entries[0]["reason"]
    assert "below the 20-bar average" in exits[0]["reason"]
    sig = entries[0]["signal"]
    assert sig["sma_5"] > sig["sma_20"] and sig["close"] > 0 and sig["price"] > 0


def test_order_status_moves_forward_only(store):
    _sleeve(store)
    store.record_order("s1", order_id="O-1", side="BUY", qty=2.0, intent="entry", reason="test")
    store.update_order("O-1", status="accepted")
    store.update_order("O-1", fill_qty=0.5, fill_px=100.0, fee=0.4)
    assert store.orders("s1")[0]["status"] == "partially_filled"
    store.update_order("O-1", fill_qty=1.5, fill_px=104.0, fee=1.2)
    store.update_order("O-1", status="accepted")  # a late event never reopens a filled order
    o = store.orders("s1")[0]
    assert o["status"] == "filled" and o["avg_px"] == pytest.approx(103.0) and o["fee"] == pytest.approx(1.6)
    store.record_order("s1", order_id="O-2", side="SELL", qty=1.0, intent="exit", reason="test")
    store.update_order("O-2", status="rejected", message="insufficient balance")
    assert store.orders("s1", statuses=("rejected",))[0]["message"] == "insufficient balance"
    store.update_order("unknown", status="canceled")  # orders from before the journal: ignored
    assert store.order_counts("s1") == {"filled": 1, "rejected": 1}
    with pytest.raises(ValueError):
        store.record_order("s1", order_id="O-3", side="BUY", qty=1, intent="yolo", reason="x")


def test_maker_first_orders_are_journaled_with_the_market_fallback(store, instrument):
    from test_maker import _daily, _minutes

    _sleeve(store)
    day = [10_000.0] * 1440
    m = _minutes(day + [10_000.0 + i for i in range(1, 2 * 1440 + 1)])  # runs away from the limit
    rt = SleeveRuntime(store, "s1", tick_seconds=SIX_HOURS)
    run_backtest("buy_and_hold", _daily(m), instrument, {"maker_wait_minutes": 15}, runtime=rt, exec_prices=m)
    first, second = reversed(store.orders("s1"))
    assert (first["order_type"], first["status"]) == ("POST-ONLY LIMIT", "canceled")
    assert (second["order_type"], second["status"]) == ("MARKET", "filled")
    assert "not filled within 15 minutes" in second["reason"] and second["signal"]["maker_order"] == first["order_id"]
    fills = store.fills("s1")  # the market order may walk a level or two of the thin test book
    assert fills and {(f["side"], f["order_id"]) for f in fills} == {("BUY", second["order_id"])}


def test_live_quotes_record_the_typical_spread_hourly(store):
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2026, 10, 3, tzinfo=timezone.utc)]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    for i in range(150):  # mostly a 2-cent spread on 100, with a few wide outliers the median ignores
        t[0] += timedelta(seconds=20)
        rt.on_quote(99.99, 100.01 if i % 10 else 100.50, venue="KRAKEN")
    assert store.latest_spread("KRAKEN", "BTC/USD") is None  # not an hour yet
    for _ in range(40):
        t[0] += timedelta(seconds=20)
        rt.on_quote(99.99, 100.01, venue="KRAKEN")
    row = store.latest_spread("KRAKEN", "BTC/USD")
    assert row["half_spread"] == pytest.approx(0.0001) and row["samples"] >= 100

    t[0] += timedelta(hours=2)
    rt.on_quote(100.0, 100.02, venue="KRAKEN")  # one quote in a quiet hour: too few to record
    assert store.latest_spread("KRAKEN", "BTC/USD")["samples"] == row["samples"]


def test_max_drawdown_counts_every_mark_not_just_the_latest(store):
    """The screens read the latest marks only; a paper strategy marks every few seconds, so after a
    month the deepest drawdown could fall out of view. It is measured in the database, over every mark."""
    from datetime import datetime, timedelta, timezone

    from sleeve_fund.dashboard.metrics import sleeve_summary

    _sleeve(store)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i, eq in enumerate([10_000, 12_000, 9_000, 13_000, 11_700, 14_000]):
        store.record_equity("s1", equity=eq, cash=eq, qty=0, price=1, benchmark=10_000, ts=t0 + timedelta(days=i))
    assert store.max_drawdown("s1") == pytest.approx(0.25) and store.max_drawdown("nobody") == 0.0
    assert sleeve_summary(store, store.sleeve("s1"))["max_drawdown"] == pytest.approx(0.25)


@pytest.mark.parametrize("journaled", [True, False])
def test_a_restart_keeps_the_stop_its_entry_set(store, instrument, journaled):
    """An ATR stop is set from the market at entry, so a restart reads it back from the entry's journal
    rather than setting a different one from the bars since. An entry journaled before stops were
    recorded gets its stop set again once there are bars enough, and says so."""
    from sqlalchemy import update

    from sleeve_fund.store import orders_t

    _sleeve(store)
    prices = synthetic_ohlcv(days=60, seed=3)
    exits = {"stop_atr": 10.0}  # wide: these runs are about the restart, not a stop-out
    rt = SleeveRuntime(store, "s1", tick_seconds=SIX_HOURS)
    run_backtest("buy_and_hold", prices.iloc[:25], instrument, exits, runtime=rt)
    (entry,) = [o for o in store.orders("s1") if o["intent"] == "entry"]
    assert entry["signal"]["stop_frac"] > 0 and "average true range" in entry["signal"]["stop_basis"]
    if not journaled:
        sig = {k: v for k, v in entry["signal"].items() if k not in ("stop_frac", "stop_basis")}
        with store.engine.begin() as c:
            c.execute(update(orders_t).where(orders_t.c.order_id == entry["order_id"]).values(signal=sig))
    rt = SleeveRuntime(store, "s1", tick_seconds=SIX_HOURS)
    run_backtest("buy_and_hold", prices.iloc[25:], instrument, exits, runtime=rt)
    resets = [e for e in store.events("s1", limit=500) if e["kind"] == "stop_reset"]
    assert len(resets) == (0 if journaled else 1)
    assert [f["side"] for f in store.fills("s1")] == ["BUY"]


@pytest.mark.parametrize("changed", [False, True])
def test_a_settings_change_moves_the_open_positions_stop(store, instrument, changed):
    """A stop or target the PM changes on the Settings tab applies to the position already open: after
    the restart that applies it, the new plan is set, not the one the entry journaled. (Checked on the
    plan itself: a backtest rests exits at its venue only for entries it made, so it can't show this.)"""
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies import REGISTRY

    _sleeve(store)
    prices = synthetic_ohlcv(days=60, seed=3)
    rt = SleeveRuntime(store, "s1", tick_seconds=SIX_HOURS)
    run_backtest("buy_and_hold", prices.iloc[:25], instrument, {"stop_atr": 10.0}, runtime=rt)
    (entry,) = [o for o in store.orders("s1") if o["intent"] == "entry"]
    assert entry["signal"]["stop_frac"] == 0.5 and not entry["signal"].get("tp_frac")
    if changed:
        store.event("s1", "info", "exits_change", "Settings changed by PM: Stop-loss ... to 4% below the entry")
    cls, config_cls = REGISTRY["buy_and_hold"]
    strat = cls(config_cls(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL"),
                           assumed_taker_fee=0.008, stop_loss=0.04, take_profit=0.05))
    strat.runtime = SleeveRuntime(store, "s1")
    strat._entry_px = entry["avg_px"]
    strat._restore_plan()
    assert (strat._stop_frac, strat._tp_frac) == ((0.04, 0.05) if changed else (0.5, None))
