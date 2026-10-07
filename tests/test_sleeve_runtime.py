"""The sleeve runtime (journal, PM controls, risk guard) driven through a real backtest."""

import os
from datetime import datetime, timedelta, timezone

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


def test_a_mismatch_line_shows_the_gap(store):
    """Review round 10, m1: "engine position 9075.15 vs journal 9075.15" hid a 1e-7 gap."""
    _sleeve(store)
    store.record_fill("s1", side="BUY", qty=9075.15, price=0.5, fee=0.0, order_id="x", trade_id="x")
    rt = SleeveRuntime(store, "s1", now=utcnow_fixed)
    assert not rt.reconcile(cash=10_000 - 9075.15 * 0.5, qty=9075.1500001, qty_tolerance=1e-8)
    msg = next(e["message"] for e in store.events("s1") if e["kind"] == "reconcile_mismatch")
    assert "engine position 9075.1500001 vs journal 9075.15, +1e-07 apart (1 fills)" in msg


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


@pytest.mark.usefixtures("maker_on")
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


@pytest.mark.parametrize("reload", [False, True])
def test_a_reload_mid_day_keeps_the_days_opening_equity(store, reload):
    """A settings edit reloads the strategy. The first tick after it must measure the day's loss from the
    equity the day opened at, not from the equity at the reload, or the daily-loss pause never fires
    (review round 8, B8-2: down 4%, an edit, then down 6.5% was never paused)."""
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2024, 3, 1, 23, 0, tzinfo=timezone.utc)]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.008)
    mark = {"cash": 0.0, "qty": 1.0}
    assert rt.tick(equity=10_000, price=10_000, **mark) is None
    t[0] += timedelta(hours=10)  # next day
    assert rt.tick(equity=9_600, price=9_600, **mark) is None  # down 4%: under the 5% limit
    if reload:
        rt = SleeveRuntime(store, "s1", now=lambda: t[0])
        rt.on_start(0.008)
    t[0] += timedelta(hours=1)
    assert rt.tick(equity=9_350, price=9_350, **mark) == "flatten"  # down 6.5% on the day
    assert store.sleeve("s1").status == "paused"


@pytest.mark.parametrize("reload", [False, True])
def test_a_pm_resume_after_a_daily_pause_survives_a_reload(store, reload):
    """The PM's resume resets the day's baseline. A reload the same day (a settings edit) must keep it,
    not restore the midnight open and re-pause and flatten at once (review round 9, M9-2 case f)."""
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2024, 3, 1, 23, 0, tzinfo=timezone.utc)]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.008)
    mark = {"cash": 0.0, "qty": 1.0}
    assert rt.tick(equity=10_000, price=10_000, **mark) is None
    t[0] += timedelta(hours=10)
    assert rt.tick(equity=9_450, price=9_450, **mark) == "flatten"  # down 5.5%: paused
    store.command("s1", "resume", "checked, carry on")
    t[0] += timedelta(minutes=1)
    assert rt.tick(equity=9_450, price=9_450, **mark) is None
    assert store.sleeve("s1").status == "running"
    if reload:
        rt = SleeveRuntime(store, "s1", now=lambda: t[0])
        rt.on_start(0.008)
    t[0] += timedelta(minutes=1)
    assert rt.tick(equity=9_440, price=9_440, **mark) is None  # 5.6% off midnight, 0.1% off the resume
    assert store.sleeve("s1").status == "running"
    t[0] += timedelta(minutes=1)
    assert rt.tick(equity=8_950, price=8_950, **mark) == "flatten"  # 5.3% off the resume: paused again
    assert store.sleeve("s1").status == "paused"


@pytest.mark.parametrize("reload", [False, True])
def test_a_pm_resume_after_a_halt_keeps_the_new_drawdown_reference_through_a_restart(store, reload):
    """A resume after a drawdown halt measures drawdown from the equity at the resume. A restart must
    keep that, not restore the all-time peak and re-halt and flatten at once (review round 9, M9-2 case g)."""
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.008)
    mark = {"cash": 0.0, "qty": 1.0}
    equity = 10_000.0
    for _ in range(7):  # 4% a day: never a daily pause, past the 20% drawdown halt on day seven
        rt.tick(equity=equity, price=equity, **mark)
        t[0] += timedelta(days=1)
        equity *= 0.96
    assert store.sleeve("s1").status == "halted"
    store.command("s1", "resume", "reviewed the halt")
    clock, calls = rt.now, iter(range(10**6))
    rt.now = lambda: clock() + timedelta(milliseconds=next(calls))  # paper's events land just after the mark
    assert rt.tick(equity=equity, price=equity, **mark) is None
    rt.now = clock
    assert store.sleeve("s1").status == "running"
    if reload:
        rt = SleeveRuntime(store, "s1", now=lambda: t[0] + timedelta(hours=1))
        rt.on_start(0.008)
    assert rt.peak == pytest.approx(equity)
    t[0] += timedelta(days=1)
    assert rt.tick(equity=equity * 0.99, price=equity, **mark) is None
    assert store.sleeve("s1").status == "running"


def test_a_restart_closes_the_orders_the_last_process_left_working(store):
    """Review round 8, m8-9: a reload while a post-only order was working left it "Working" for ever.
    Paper's venue is simulated in the process, so its orders end with it."""
    _sleeve(store)
    rt = SleeveRuntime(store, "s1")
    rt.on_order(order_id="O-1", side="BUY", qty=0.1, intent="entry", reason="Signal to be long", signal={},
                order_type="POST-ONLY LIMIT")
    rt.on_order_status("O-1", "accepted")
    SleeveRuntime(store, "s1").on_start(0.008)
    (o,) = store.orders("s1")
    assert o["status"] == "canceled" and "restarted" in o["message"]


@pytest.mark.parametrize("command", ["pause", "flatten"])
def test_a_pm_pause_or_flatten_survives_a_restart(store, command):
    """The PM's pause, flatten (which pauses) and the book kill switch (a flatten per strategy) have no end
    time: only a resume lifts them, not a settings reload, a stale-heartbeat or a crash restart (round 10, B10-3)."""
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.008)
    mark = {"cash": 10_000.0, "qty": 0.0}
    store.command("s1", command, "PM wants it stopped")
    rt.tick(equity=10_000, price=100, **mark)
    assert store.sleeve("s1").status == "paused"
    t[0] += timedelta(minutes=5)
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])  # the restart
    rt.on_start(0.008)
    assert store.sleeve("s1").status == "paused" and not rt.can_open()
    store.command("s1", "resume", "carry on")
    rt.tick(equity=10_000, price=100, **mark)
    assert store.sleeve("s1").status == "running" and rt.can_open()


def test_a_crash_keeps_a_pause_and_its_end_time(store):
    """A crash used to overwrite any status with "error", losing a daily-loss pause's end time, so the
    restart traded again hours early. Now the status stays and the crash is an event (round 10, B10-3)."""
    from datetime import timedelta

    from sleeve_fund.store import utcnow
    from sleeve_fund.supervisor import Proc, Supervisor

    class Dead:
        pid, returncode = 1, 1

        def poll(self):
            return 1

    _sleeve(store)
    store.set_desired_state("s1", "running")
    until = utcnow() + timedelta(hours=20)
    store.set_status("s1", "paused", "daily loss", until)
    sup = Supervisor(store)
    sup.procs["s1"] = Proc(popen=Dead(), started_at=utcnow())
    sup.step()
    s = store.sleeve("s1")
    assert s.status == "paused" and abs((s.paused_until - until).total_seconds()) < 1
    assert any(e["kind"] == "process_crash" and "still paused" in e["message"] for e in store.events("s1"))
    rt = SleeveRuntime(store, "s1")
    rt.on_start(0.008)
    assert store.sleeve("s1").status == "paused" and not rt.can_open()
def _restart(store, t):
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.008)
    return rt


@pytest.mark.parametrize("command,reason,sells", [
    ("flatten", "PM flatten", True),
    ("flatten", "Book kill switch: stop everything", True),
    ("pause", "PM pause", False),  # a pause keeps the position, before and after a restart
])
def test_a_flatten_cut_short_by_a_restart_sells_again(store, command, reason, sells):
    """Sanity S-3: a flatten is marked done when its sell is sent. If the process stops before the fill,
    the restart still holds the position, so the first tick sells again, once, with the reason kept."""
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2025, 10, 3, 12, tzinfo=timezone.utc)]
    rt = _restart(store, t)
    store.command("s1", command, reason)
    held = {"equity": 10_000, "cash": 7_000, "qty": 0.05, "price": 60_000}
    assert rt.tick(**held) == ("flatten" if command == "flatten" else None)
    t[0] += timedelta(minutes=1)
    rt = _restart(store, t)  # the sell never filled
    assert rt.tick(**held) == ("flatten" if sells else None)
    if sells:
        assert rt.flatten_why[0] == "pm_flatten" and reason in rt.flatten_why[1]
        assert store.last_event("s1", ("flatten_retry",)) is not None
    t[0] += timedelta(seconds=5)
    assert rt.tick(**held, busy=True) is None  # the strategy's own sell is in flight: wait for it
    assert store.sleeve("s1").status == "paused"
    rt = _restart(store, t)
    assert rt.tick(equity=10_000, cash=10_000, qty=0.0, price=60_000) is None  # sold: nothing owed


def test_a_flatten_that_does_not_close_is_sent_again_then_handed_to_the_pm(store):
    """Sanity, 4 Oct: a flatten whose order is rejected (or cut short) is owed until the position closes:
    sent again while no order is working, up to FLATTEN_RETRIES times, then an error asks the PM."""
    from datetime import datetime, timedelta, timezone

    from sleeve_fund.paper.runtime import FLATTEN_RETRIES

    _sleeve(store)
    t = [datetime(2025, 10, 3, 12, tzinfo=timezone.utc)]
    rt = _restart(store, t)
    store.command("s1", "flatten", "PM flatten")
    short = {"equity": 10_000, "cash": 13_000, "qty": -0.05, "price": 60_000}  # a short closes by buying
    assert rt.tick(**short) == "flatten"
    assert rt.tick(**short, busy=True) is None
    for _ in range(FLATTEN_RETRIES):
        t[0] += timedelta(seconds=30)
        assert rt.tick(**short) == "flatten" and rt.flatten_why[0] == "pm_flatten"
    t[0] += timedelta(seconds=30)
    assert rt.tick(**short) is None
    assert store.last_event("s1", ("flatten_failed",))["level"] == "error"
    assert rt.tick(**short) is None and len([e for e in store.events("s1") if e["kind"] == "flatten_failed"]) == 1


def test_dust_owes_no_flatten(store):
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2025, 10, 3, 12, tzinfo=timezone.utc)]
    rt = _restart(store, t)
    store.command("s1", "flatten", "PM flatten")
    assert rt.tick(equity=10_000, cash=7_000, qty=0.05, price=60_000) == "flatten"
    t[0] += timedelta(minutes=1)
    rt = _restart(store, t)
    rt.close_floor = 1e-8
    assert rt.tick(equity=10_000, cash=10_000, qty=3e-9, price=60_000) is None  # below one lot: can't be sold
    assert store.last_event("s1", ("flatten_retry",)) is None


def test_a_risk_halt_cut_short_sells_again_but_a_resume_or_a_reconcile_halt_does_not(store):
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2025, 10, 3, 12, tzinfo=timezone.utc)]
    rt = _restart(store, t)
    rt.tick(equity=10_000, cash=0, qty=1.0, price=10_000)
    t[0] += timedelta(hours=1)
    assert rt.tick(equity=7_000, cash=0, qty=1.0, price=7_000) == "flatten"  # 30% down: halt
    assert store.sleeve("s1").status == "halted"
    rt = _restart(store, t)
    assert rt.tick(equity=7_000, cash=0, qty=1.0, price=7_000) == "flatten"
    assert rt.flatten_why[0] == "risk_halt"
    store.command("s1", "resume", "checked")
    rt.tick(equity=7_000, cash=0, qty=1.0, price=7_000)
    assert store.sleeve("s1").status == "running"
    # A reconcile halt never trades or corrects, before a restart or after one.
    assert not rt.reconcile(cash=123.0, qty=1.0)
    rt = _restart(store, t)
    assert store.sleeve("s1").status == "halted"
    assert rt.tick(equity=7_000, cash=0, qty=1.0, price=7_000) is None


def test_an_expired_daily_loss_pause_owes_nothing_after_a_restart(store):
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2025, 10, 3, 1, tzinfo=timezone.utc)]
    rt = _restart(store, t)
    rt.tick(equity=10_000, cash=0, qty=1.0, price=10_000)
    t[0] += timedelta(hours=1)
    assert rt.tick(equity=9_400, cash=0, qty=1.0, price=9_400) == "flatten"  # 6% today: daily pause
    assert store.sleeve("s1").status == "paused"
    rt = _restart(store, t)
    assert rt.tick(equity=9_400, cash=0, qty=1.0, price=9_400) == "flatten"  # still paused: sell again
    t[0] += timedelta(hours=25)
    rt = _restart(store, t)
    assert rt.tick(equity=9_400, cash=0, qty=1.0, price=9_400) is None


def test_a_resume_on_a_running_strategy_is_ignored_and_keeps_the_days_baseline(store):
    """Review round 10, m10-3: a resume on a strategy already running reset the day's loss baseline."""
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2025, 10, 3, 12, tzinfo=timezone.utc)]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.008)
    rt.tick(equity=10_000, cash=10_000, qty=0.0, price=100)
    t[0] += timedelta(minutes=1)
    rt.tick(equity=9_700, cash=9_700, qty=0.0, price=100)
    store.command("s1", "resume", "carry on")
    t[0] += timedelta(minutes=1)
    assert rt.tick(equity=9_700, cash=9_700, qty=0.0, price=100) is None
    assert rt._day_open == 10_000
    assert any(e["kind"] == "pm_resume_ignored" for e in store.events("s1"))
    t[0] += timedelta(minutes=1)
    assert rt.tick(equity=9_450, cash=9_450, qty=0.0, price=100) == "flatten"  # 5.5% on the day: paused


def test_reconcile_allows_the_float_step_at_a_large_position(store):
    """Review round 12, M12-E1: at 1.6e8 units (a sub-cent instrument) one float step is 3e-8, wider than two
    lots of 8 decimals, so a gap no float could avoid halted a backtest. A real gap still halts."""
    import math

    _sleeve(store)
    held = 159660965.631
    store.record_fill("s1", side="BUY", qty=held, price=0.00002, fee=0.0, order_id="x", trade_id="x")
    rt = SleeveRuntime(store, "s1", now=utcnow_fixed)
    cash = 10_000 - held * 0.00002
    drift = held + math.ulp(held)  # one float step: about 3e-8 here
    assert drift - held > 2e-8  # more than two 8-decimal lots
    assert rt.reconcile(cash=cash, qty=drift, qty_tolerance=2e-8)
    assert not rt.reconcile(cash=cash, qty=held + 0.001, qty_tolerance=2e-8)


def _liquidated(store, t):
    """A strategy liquidated and halted in the ruled words, as the engine journals it: the liquidation, a drawdown
    halt on the same tick, then the liquidation's own halt."""
    _sleeve(store)
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.0005)
    rt.tick(equity=10_000, cash=10_000, qty=0.0, price=60_000)
    store.event("s1", "error", "liquidation", "Liquidated: the price 97,440 reached the liquidation price 90,490.4")
    rt.tick(equity=5_900, cash=5_900, qty=0.0, price=97_440,
            ruined="Position margin lost (liquidated): 4,100.00, 41% of strategy equity")
    assert store.sleeve("s1").status == "halted"
    return rt


def _after(store, t, minutes=1):
    t[0] += timedelta(minutes=minutes)
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.0005)
    rt.tick(equity=5_900, cash=5_900, qty=0.0, price=97_440)
    return rt


def test_a_pm_resume_never_runs_a_liquidated_strategy_even_for_a_tick(store):
    """QA on #164, HoE 6 Oct 19:11: after a liquidation the PM's resume leaves it halted in the liquidation's words,
    with no drawdown reset, and nothing may open on the tick that takes the resume or after."""
    t = [datetime(2025, 10, 3, 10, 0, tzinfo=timezone.utc)]
    rt = _liquidated(store, t)
    store.command("s1", "resume", "carry on")
    t[0] += timedelta(minutes=1)
    rt.tick(equity=5_900, cash=5_900, qty=0.0, price=97_440)
    s = store.sleeve("s1")
    assert s.status == "halted" and s.status_reason.startswith("Position margin lost (liquidated): 4,100.00")
    assert not rt.can_open() and not _after(store, t).can_open()
    assert store.last_event("s1", ("drawdown_reset",)) is None
    assert not [c for c in store.pending_commands("s1") if c["command"] == "resume"]  # taken up, not left waiting


def test_a_liquidation_survives_a_stop_start_a_restart_while_halted_and_a_resume(store):
    """QA on #164, HoE 6 Oct 19:11: the journal decides, not the latest reason. Liquidated, then Stop/Start (which
    wrote "stopped" over the halt before HC), then a restart while halted that halts again on drawdown, then a
    resume: still liquidated, halted in its words, in this process and in a fresh one."""
    t = [datetime(2025, 10, 3, 10, 0, tzinfo=timezone.utc)]
    _liquidated(store, t)
    store.set_status("s1", "stopped", "stopped by PM")
    store.event("s1", "error", "risk_halt", "drawdown 41.0% hit the 20% limit; flattened, PM must resume")
    store.set_status("s1", "halted", "drawdown 41.0% hit the 20% limit")
    rt = _after(store, t)
    assert rt.liquidated is not None and not rt.can_open()
    store.command("s1", "resume", "carry on")
    t[0] += timedelta(minutes=1)
    rt.tick(equity=5_900, cash=5_900, qty=0.0, price=97_440)
    s = store.sleeve("s1")
    assert s.status == "halted" and s.status_reason.startswith("Position margin lost (liquidated): 4,100.00"), s
    fresh = _after(store, t)
    assert not fresh.can_open() and store.sleeve("s1").status == "halted"
    assert store.last_event("s1", ("drawdown_reset",)) is None


def test_a_reset_after_liquidation_in_the_journal_ends_the_liquidated_state(store):
    """The journal rule's other half: a liquidation before the last reset after liquidation no longer counts."""
    from sleeve_fund.paper.runtime import RESET_AFTER_LIQUIDATION

    t = [datetime(2025, 10, 3, 10, 0, tzinfo=timezone.utc)]
    _liquidated(store, t)
    store.event("s1", "info", RESET_AFTER_LIQUIDATION, "reset after the liquidation")
    assert SleeveRuntime(store, "s1", now=lambda: t[0]).liquidated is None


def test_the_pms_commands_wait_while_a_liquidation_order_is_working(store):
    """QA P1-U34: a resume queued while paused is never applied on the tick that liquidates (the guard has sent the
    liquidation, which hasn't filled): it waits, so the strategy is never running with the position still held; the
    next tick has the liquidation's halt, which a resume doesn't clear."""
    from datetime import datetime, timedelta, timezone

    _sleeve(store)
    t = [datetime(2024, 3, 2, 12, tzinfo=timezone.utc)]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.008)
    mark = {"cash": 0.0, "qty": 1.0}
    rt.tick(equity=10_000, price=10_000, **mark)
    store.command("s1", "pause", "holding")
    t[0] += timedelta(minutes=1)
    rt.tick(equity=10_000, price=10_000, **mark)
    assert store.sleeve("s1").status == "paused"
    store.command("s1", "resume", "carry on")
    t[0] += timedelta(minutes=1)
    rt.tick(equity=9_900, price=9_900, liquidating=True, **mark)
    assert store.sleeve("s1").status == "paused" and not rt.can_open()
    assert [c["command"] for c in store.pending_commands("s1")] == ["resume"]  # still waiting


@pytest.mark.parametrize("kind", ["store", "memory"])
def test_orders_by_intent_and_funding_and_insurance_before_a_time_are_read_in_the_query(store, kind):
    """Code Reviewer on 81e6d6f (minor 2): the liquidation figures read only the liquidation orders, and only the
    funding and insurance booked before the position opened, not every row the strategy ever had."""
    from datetime import datetime, timedelta, timezone

    from sleeve_fund.paper.journal import MemoryJournal

    j = store if kind == "store" else MemoryJournal()
    _sleeve(j)
    t0 = datetime(2024, 3, 2, tzinfo=timezone.utc)
    for i, intent in enumerate(("entry", "liquidation", "exit", "liquidation")):
        j.record_order("s1", order_id=f"o{i}", side="BUY", qty=1.0, intent=intent, reason="", ts=t0 + timedelta(hours=i))
    assert {o["order_id"] for o in j.orders("s1", intents=("liquidation",))} == {"o1", "o3"}
    for h, amount in ((0, 1.5), (8, -0.5), (16, 2.0)):
        j.record_funding("s1", qty=1.0, price=100.0, rate=0.0001, amount=amount, ts=t0 + timedelta(hours=h))
        j.record_insurance("s1", price=100.0, amount=amount * 10, ts=t0 + timedelta(hours=h))
    cut = t0 + timedelta(hours=16)
    assert j.funding_total("s1", before=cut) == pytest.approx(1.0) and j.funding_total("s1") == pytest.approx(3.0)
    assert j.insurance_total("s1", before=cut) == pytest.approx(10.0) and j.insurance_total("s1") == pytest.approx(30.0)
