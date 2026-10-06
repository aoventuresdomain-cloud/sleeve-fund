"""An instrument the venue moves from 8-hour to 1-hour settlements mid-run (Advisor, 6 Oct 2026): the interim guard
says so on the backtest, and once funding is charged at each stored rate's own time (DA-11) every settlement is."""

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


@pytest.mark.xfail(strict=True, reason="DA-11: funding charged at each stored rate's own time, not the fixed schedule")
def test_every_settlement_is_charged_after_the_switch(binance):
    every = _switch()
    held = [t for t in every if utc("2025-10-03 01:00") < t <= utc("2025-10-05 00:00")]
    assert sorted(utc(f["ts"]) for f in _run(binance).funding) == held
