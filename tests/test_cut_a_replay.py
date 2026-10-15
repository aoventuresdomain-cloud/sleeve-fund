"""CUT-A on the replay of missed minutes (RESUME-LEG line 1, found by the decision-path scan): inside the fund no
decision reads a Strategy's own equity, and the replay after a restart or a feed gap is a decision (it closes the
position as an outage exit). Its drawdown-halt and daily-loss-pause levels are the Strategy's own brakes, which the
fund's T5 replaces there (Advisor SIZE-FUND Q4), as SleeveRuntime.tick and _guard_breached already skip them. Inside
the fund the Strategy's own equity starts at 0, so its own "drawdown level" sits at about its entry price."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from test_exposure_gate_xfails import _offline  # noqa: F401 - the paper harness's offline venue (autouse)
from test_timing import NS, _booked, _late_strategy, _minutes

from sleeve_fund import risk


def _restarted_long(fund_held: bool, equity: float, cash: float, peak: float, minutes):
    """A 1-unit long from 100 with a 2% stop (98) and a 5% target, restarted over the stored `minutes` (high, low)."""
    s, bar, events, sold, opened = _late_strategy()
    _booked(s, sold)
    s._entry_px, s._entry_side, s._stop_frac, s._tp_frac = 100.0, 1, 0.02, 0.05
    s._last_alive = s._last_seen = datetime.fromtimestamp(NS / 1e9, tz=timezone.utc)
    s.runtime.store.fills = lambda name, limit: [{"ts": datetime.fromtimestamp((NS - 600e9) / 1e9, tz=timezone.utc)}]
    s.runtime.profile, s.runtime.halts, s.runtime.fund_held = risk.profile("balanced"), True, fund_held
    s.runtime.peak, s.runtime._day_open = peak, peak
    s._mark = lambda: (equity, cash, 1.0, 101.0)
    s._liq_now = lambda cash, qty: None
    s._guard_book = lambda equity, cash, qty, price: (equity, cash, qty)
    ticks = []
    s._on_tick = lambda *a: ticks.append(s._guard_equity)
    s.history_loader = lambda instrument, bar_type, n: _minutes(*minutes)
    s.instrument = None
    return s, sold, events, ticks


# Own equity from 0 inside the fund: the buy's 100 and its fee came out of the Strategy's cash; marked at 101 it is up
# 0.92, so its own "20% drawdown level" is (0.92 x 0.8 + 100.08) / 1 = 100.816, above its entry.
FUND = {"equity": 0.92, "cash": -100.08, "peak": 0.92}


def test_inside_the_fund_a_replayed_dip_says_nothing_and_sends_nothing():
    s, sold, events, ticks = _restarted_long(True, **FUND, minutes=[(100.5, 99.5), (101.0, 100.0)])
    assert s._check_outage_exits() is False
    assert sold == [] and ticks == [] and not [e for e in events if e[2] == "outage_exit"], events


def test_inside_the_fund_the_stop_a_later_missed_minute_passed_is_still_booked():
    """The money path: the replay stops at its first hit, so a phantom drawdown hit in minute 1 would leave the stop
    the venue filled in minute 2 unbooked, and the minutes marked as seen."""
    s, sold, events, ticks = _restarted_long(True, **FUND, minutes=[(100.5, 99.5), (99.0, 97.0)])
    assert s._check_outage_exits() is True
    (intent, reason, book), = sold
    assert intent == "stop_loss" and "Risk-halt" not in " ".join(e[3] for e in events), events
    assert ticks == []


def test_outside_the_fund_the_replay_still_judges_the_strategys_own_drawdown_level():
    # Its own equity, 1,000.92, against a peak of 1,250: the 20% halt is at 1,000 of equity, a price of
    # (1,250 x 0.8 - 899.92) / 1 = 100.08, which the replayed minute's 99.5 passes; the tick then judges it.
    s, sold, events, ticks = _restarted_long(False, equity=1_000.92, cash=899.92, peak=1_250.0,
                                             minutes=[(100.5, 99.5), (101.0, 100.0)])
    assert s._check_outage_exits() is True
    assert len(ticks) == 1 and ticks[0] is not None
    (_, _, kind, said), = [e for e in events if e[2] == "outage_exit"]
    assert said.startswith("Risk-halt reached") and "drawdown halt's level" in said, said
