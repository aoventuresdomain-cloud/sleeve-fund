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
