"""QUIET-TICKS (HoE 17 Nov): a profiled backtest fed 1-minute bars ran paper's full tick (mark, guard, funding,
reconcile) on every minute of every leg, 7-11x slower than without a profile. A long run now ticks every minute
only while its book is live (a position, an order working or owed) and on the minute after; flat with nothing
working, only where a mark is kept or a clock acts (LongFlatStrategy._exec_tick_due). Done when the same run gives
the same fills, orders, journal, funding, risk events and kept equity marks either way, on runs that trade, stop out,
halt and are sized at their profile's cap, on spot and on a 2x perp, alone and as a guarded basket."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.paper.journal import KEEP_ALL_MARKS
from sleeve_fund.research import runner
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.base import LongFlatStrategy
from sleeve_fund.venues import venue

PARAMS = {"fast": 2, "slow": 3, "stop_loss": 0.04, "take_profit": 0.05}


def _minutes(seed: int, days: int = 90, start: str = "2024-01-01 00:01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.06 / np.sqrt(1440), days * 1440)
    jumps = rng.random(days * 1440) < 2 / 1440
    r[jumps] += rng.normal(0, 0.06, jumps.sum())
    n = (days - 3) * 1440
    c = 1000 * np.exp(np.cumsum(r))[:n]
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995, "close": c,
                         "volume": 50.0}, index=idx)


def _daily(m: pd.DataFrame) -> pd.DataFrame:
    return m.resample("1D", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


@pytest.fixture
def ticks(monkeypatch):
    """Counts the full ticks a run takes."""
    seen = {"n": 0}
    real = LongFlatStrategy._on_tick

    def counted(self, *a, **k):
        seen["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(LongFlatStrategy, "_on_tick", counted)
    return seen


def _both(monkeypatch, ticks, run):
    """run() with quiet ticks (the default) and with every minute ticked, and how many full ticks each took."""
    out = []
    for every in (False, True):
        monkeypatch.setattr(runner, "QUIET_TICKS_FROM", 10**15 if every else 2 * KEEP_ALL_MARKS)
        ticks["n"] = 0
        out.append((run(), ticks["n"]))
    return out


def _events(j) -> list:
    return [(e["kind"], e["ts"], e["level"], e["message"]) for e in j.events_]


def _assert_same(quiet, every) -> None:
    drop = ["init_id"]  # a random id the engine gives each order's init event
    assert quiet.fills.drop(columns=drop).equals(every.fills.drop(columns=drop))
    assert quiet.equity.equals(every.equity) and quiet.exposure.equals(every.exposure)
    assert quiet.decisions == every.decisions and quiet.risk_events == every.risk_events
    assert quiet.funding == every.funding and quiet.funding_marks == every.funding_marks
    jq, je = quiet.journal, every.journal
    assert list(jq.orders_.values()) == list(je.orders_.values())
    assert jq.fills_ == je.fills_ and jq.funding_ == je.funding_ and jq.insurance_ == je.insurance_
    assert _events(jq) == _events(je)
    kept = jq.marks_to_keep()
    assert kept == je.marks_to_keep() and len(kept) > 24
    assert jq.max_drawdown() == je.max_drawdown()


ETH = venue("kraken").instrument("ETH", "USD")


@pytest.mark.parametrize("seed, profile, perp", [(2, "conservative", False), (5, "balanced", True)])
def test_a_long_profiled_run_comes_out_the_same_with_its_quiet_minutes_left_unticked(monkeypatch, ticks, seed,
                                                                                     profile, perp):
    m = _minutes(seed)
    params = {**PARAMS, **({"market": "perp", "allow_short": True, "max_leverage": 2} if perp else {})}

    def run():
        return run_backtest("trend_filter", _daily(m), ETH, starting_capital=10_000, exec_prices=m, exec_minutes=1,
                            risk_profile=profile, bar_minutes=1440, half_spread=0.0, params=params)

    (quiet, n_quiet), (every, n_every) = _both(monkeypatch, ticks, run)
    _assert_same(quiet, every)
    # Not vacuous: it traded, stopped out, took profit, halted, was sized at its profile's cap, and paid funding.
    intents = {o["intent"] for o in quiet.journal.orders_.values()}
    assert {"entry", "stop_loss", "take_profit", "risk_halt"} <= intents
    assert {o["signal"].get("sized_by") for o in quiet.journal.orders_.values() if o["intent"] == "entry"} == {
        f"{profile} risk profile cap"}
    assert not perp or (quiet.funding and quiet.funding_marks)
    assert n_every >= len(m) and n_quiet < n_every / 2  # the quiet minutes did go unticked


@pytest.mark.parametrize("days, drop", [(90, ("10:59", "23:59")), (100, ("23:59",))])
def test_gaps_at_an_hours_last_minute_keep_the_same_marks(monkeypatch, ticks, days, drop):
    """CR #361 M1: where the data lacks an hour's :59 the hour's last mark is :58, and past 90 days a day's is the
    last bar before a missing 23:59; the quiet run ticks the data's own last bar of each hour, so its kept marks are
    the same."""
    m = _minutes(2, days=days)
    m = m[~m.index.strftime("%H:%M").isin(drop)]

    def run():
        return run_backtest("trend_filter", _daily(m), ETH, starting_capital=10_000, exec_prices=m, exec_minutes=1,
                            risk_profile="conservative", bar_minutes=1440, half_spread=0.0, params=dict(PARAMS))

    (quiet, n_quiet), (every, n_every) = _both(monkeypatch, ticks, run)
    _assert_same(quiet, every)
    kept = {k["ts"].strftime("%H:%M") for k in quiet.journal.marks_to_keep()}
    assert ("10:58" in kept) is (days <= 90) and "23:58" in kept and not kept & set(drop)
    assert (quiet.equity.index[-1] - quiet.equity.index[0]).days > 90 or days <= 90
    assert n_quiet < n_every / 2


def test_an_order_sent_on_a_quiet_minute_ticks_the_next(monkeypatch):
    """A quiet minute is flat with nothing working, so nothing can fill on it; were this strategy to send an order
    there, the next minute ticks (LongFlatStrategy._on_exec_bar). Pinned by sending a stand-in one."""
    m = _minutes(2, days=12)
    seen, real_tick = [], LongFlatStrategy._on_tick

    def tick(self, *a, **k):
        seen.append(self.clock.timestamp_ns())
        return real_tick(self, *a, **k)

    def run():
        seen.clear()
        return run_backtest("trend_filter", _daily(m), ETH, starting_capital=10_000, exec_prices=m, exec_minutes=1,
                            risk_profile="conservative", bar_minutes=1440, half_spread=0.0, params=dict(PARAMS))

    monkeypatch.setattr(LongFlatStrategy, "_on_tick", tick)
    run()
    ticked, minute = set(seen), 60_000_000_000
    t = next(int(x) for x in m.index.as_unit("ns").asi8[3000:] if {x, x + minute, x - minute}.isdisjoint(ticked))
    real_target = LongFlatStrategy._bar_target
    from nautilus_trader.model import ClientOrderId

    stand_in = ClientOrderId("O-STAND-IN")  # no such order: cache.order() finds none, so nothing else reads it

    def target(self, bar):
        now = self.clock.timestamp_ns()
        if now == t:
            self._sent.append(stand_in)
        elif now == t + minute and stand_in in self._sent:
            self._sent.remove(stand_in)
        return real_target(self, bar)

    monkeypatch.setattr(LongFlatStrategy, "_bar_target", target)
    run()
    assert t + minute in seen and t not in seen


@pytest.mark.parametrize("brakes", [False, True])
def test_a_guarded_basket_comes_out_the_same(monkeypatch, ticks, brakes):
    """Two profiled legs in one engine under the fund's limits, one trimmed by its gross cap, and (brakes) the fund's
    drawdown and daily-loss brakes tight enough to fire: the gate's decisions, the fund's book and each leg's record
    are the same either way."""
    from sleeve_fund.research.portfolio_run import LegSpec, run_portfolio
    from sleeve_fund.risk import PortfolioProfile

    loss = 0.02 if brakes else 0.99
    tight = PortfolioProfile(gross=0.3, net_instrument=100, margin=100, open_risk=100, drawdown=loss, daily_loss=loss)
    legs = []
    for name, seed in (("a", 2), ("b", 7)):
        m = _minutes(seed, days=60)
        d = _daily(m)
        warm, window = d.iloc[:16], d.iloc[16:]
        legs.append(LegSpec(name, "trend_filter", ETH, window, m[m.index > warm.index[-1]], warm, exec_minutes=1,
                            params=dict(PARAMS), risk_profile="aggressive", half_spread=0.0))

    (quiet, n_quiet), (every, n_every) = _both(monkeypatch, ticks, lambda: run_portfolio(legs, tight))
    assert quiet.book.equals(every.book)
    strip = [{k: v for k, v in d.items() if k != "order_id"} for d in quiet.decisions]
    assert strip == [{k: v for k, v in d.items() if k != "order_id"} for d in every.decisions]
    assert quiet.refusals == every.refusals and quiet.halt == every.halt and quiet.pauses == every.pauses
    assert brakes or any(d["outcome"] == "trimmed" for d in quiet.decisions)  # the gross cap bit
    assert bool(quiet.halt or quiet.pauses) is brakes  # tight, a fund brake fired (and stopped the trading early)
    for name in ("a", "b"):
        _assert_same(quiet.legs[name], every.legs[name])
    assert n_quiet < n_every / 2


def test_a_short_run_or_one_on_its_own_bars_ticks_every_bar(monkeypatch):
    """Below QUIET_TICKS_FROM execution bars a run keeps every mark (MemoryJournal.KEEP_ALL_MARKS), so every minute
    ticks; a run fed only its decision bars has no quiet minutes to skip."""
    assert runner.QUIET_TICKS_FROM == 2 * KEEP_ALL_MARKS
    m = _minutes(2, days=8)
    short = runner.plan_run("trend_filter", _daily(m), ETH, dict(PARAMS), exec_prices=m, exec_minutes=1,
                            risk_profile="conservative", bar_minutes=1440)
    assert short.runtime.quiet_marks is None
    long_ = _minutes(2, days=12)
    plan = runner.plan_run("trend_filter", _daily(long_), ETH, dict(PARAMS), exec_prices=long_, exec_minutes=1,
                           risk_profile="conservative", bar_minutes=1440)
    assert plan.runtime.quiet_marks and max(plan.runtime.quiet_marks) == plan.end_ns and plan.runtime.store._thinned
    bars_only = runner.plan_run("trend_filter", _daily(long_), ETH, dict(PARAMS), risk_profile="conservative",
                                bar_minutes=1440)
    assert bars_only.runtime.quiet_marks is None
