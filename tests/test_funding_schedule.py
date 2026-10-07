"""An instrument the venue moves from 8-hour to 1-hour settlements mid-run (Advisor, 6 Oct 2026): the interim guard
says so on the backtest, and every settlement is charged at its stored time (QA P1-O1, #155)."""

from __future__ import annotations

import pandas as pd
import pytest

from o17_harness import _no_swallowed_strategy_errors, _o17win, backtest, binance, hourly_bars, utc, win, write_rates  # noqa: F401


def _switch():
    eight = pd.date_range("2025-10-03 00:00", "2025-10-04 08:00", freq="8h", tz="UTC")
    hourly = pd.date_range("2025-10-04 09:00", "2025-10-05 00:00", freq="1h", tz="UTC")
    write_rates({t: 0.0003 for t in eight.append(hourly)})
    return list(eight) + list(hourly)


def _run(b):
    return backtest(b, hourly_bars("2025-10-03 00:00", "2025-10-05 00:00"), win(("2025-10-03 01:00", "2026-01-01", 1)))


def test_the_backtest_says_the_settlement_schedule_does_not_fit(binance):
    _switch()
    assert _run(binance).funding_schedule.startswith("funding schedule mismatch: BTC/USDT settled 1 hours apart")


def test_an_unchanged_schedule_is_no_mismatch(binance):
    write_rates({t: 0.0003 for t in pd.date_range("2025-10-03 00:00", "2025-10-05 00:00", freq="8h", tz="UTC")})
    assert _run(binance).funding_schedule == ""


def test_every_settlement_is_charged_after_the_switch(binance):
    every = _switch()
    held = [t for t in every if utc("2025-10-03 01:00") < t <= utc("2025-10-05 00:00")]
    assert sorted(utc(f["ts"]) for f in _run(binance).funding) == held


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_single_missing_settlement_on_the_8_hour_schedule_is_charged_once_at_the_baseline(binance, side):
    """Advisor (7 Oct, gap inference): one settlement missing between stored 8-hourly records is filled at the 8-hour
    step and charged once, at the baseline, whichever side is held; every other settlement once at its own rate."""
    every = [t for t in pd.date_range("2025-10-03 00:00", "2025-10-05 00:00", freq="8h", tz="UTC")]
    gone = every[3]
    write_rates({t: 0.0003 for t in every if t != gone})
    r = backtest(binance, hourly_bars("2025-10-03 00:00", "2025-10-05 00:00"),
                 win(("2025-10-03 01:00", "2026-01-01", side)))
    charged = sorted(utc(f["ts"]) for f in r.funding)
    assert charged == [t for t in every if t > utc("2025-10-03 01:00")]  # each once, the missing one included
    rows = {utc(f["ts"]): f for f in r.funding}
    assert rows[gone]["kind"] == "baseline" and rows[gone]["amount"] < 0
    assert all(rows[t]["kind"] == "settled" for t in rows if t != gone)
    assert r.funding_at_baseline == 1
