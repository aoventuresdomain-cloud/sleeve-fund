"""PM 23 Oct (BTC on 1-minute bars, second fix): a pinned list's study sizes its execution bars on its pinned window,
not on the whole store, and runs on 1-minute bars or is refused; it never falls back to coarser fills quietly. The
22 Oct 23:52 BTC run sized them on the store, which reaches back before the list's data start, went over the budget
and ran on 5-minute bars. rescore's dry run reports the bars a row would get, so this is proved on the real store
before a full run."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from qa155_lib import synth_1m

from sleeve_fund import history
from sleeve_fund.research import pinned as pinning
from sleeve_fund.research import rescore
from sleeve_fund.research import run as research_run
from sleeve_fund.venues import venue

STORE_START = "2025-01-01"
STORE_DAYS = 40


def _store():
    m1 = synth_1m(days=STORE_DAYS, seed=5, start=STORE_START)
    hs = history.HistoryStore(Path(tempfile.mkdtemp(prefix="pinned-minutes-")) / "hist")
    hs.append("KRAKEN", "BTC/USD", m1.set_axis(m1.index - pd.Timedelta(minutes=1)), cursor="x")
    return hs


def _pins(start=date(2025, 1, 21)):
    """19 days of research and holdout from `start`, well inside the 40-day store."""
    fold = pinning.Fold(start, date(2025, 1, 28), date(2025, 2, 3))
    return pinning.Pinned(start, date(2025, 2, 3), date(2025, 2, 4), date(2025, 2, 8), (fold,))


def _req(pins, minutes=15):
    return research_run.StudyRequest(strategy="buy_and_hold", pair="BTC/USD", venue="KRAKEN", minutes=minutes,
                                     train_days=7, test_days=7, holdout_days=pins.holdout_days, pinned=pins)


def _bars(req, hs):
    profile = venue(req.venue)
    return research_run.execution_bars(req, hs.read(profile.name, req.pair, req.minutes), profile, hs)


def test_the_pinned_window_fits_on_1_minute_bars_though_the_store_does_not(monkeypatch):
    """The store's 40 days are over a 25-day budget; the pinned 19 days are not: 1-minute bars over the window."""
    monkeypatch.setattr(research_run, "OOS_EXEC_BUDGET", 25 * 1440)
    hs, pins = _store(), _pins()
    _, oos_exec, oos_step = _bars(_req(pins), hs)
    assert oos_step == 1
    assert oos_exec.index[0] <= pd.Timestamp(pins.data_start, tz="UTC") + pd.Timedelta(minutes=1)
    assert oos_exec.index[-1] <= pd.Timestamp(pins.end, tz="UTC") + pd.Timedelta(days=1)
    assert (oos_exec.index.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all()


def test_a_pinned_window_over_the_budget_is_refused_not_run_on_5_minute_bars(monkeypatch):
    monkeypatch.setattr(research_run, "OOS_EXEC_BUDGET", 10 * 1440)
    with pytest.raises(ValueError, match="no 1-minute run"):
        _bars(_req(_pins()), _store())


def test_minutes_missing_at_the_window_start_are_refused():
    """The 15-minute bars cover the window but the 1-minute ones start later: refused, not a short 1-minute run."""
    hs, pins = _store(), _pins()
    real = hs.read

    def late(venue_, pair, minutes=1440, start=None, end=None):
        df = real(venue_, pair, minutes, start=start, end=end)
        return df[df.index >= pd.Timestamp("2025-01-25", tz="UTC")] if minutes == 1 else df

    hs.read = late
    with pytest.raises(ValueError, match="don't cover the pinned window"):
        _bars(_req(pins), hs)


def test_the_round_1_list_fits_on_1_minute_bars_but_the_whole_store_would_not():
    """The real dates: the round 1 list's pinned window (21 Oct 2020 to the holdout end) fits the budget at 1 minute;
    a store from the BTC/USDT perpetual's first days (Sep 2019) would not, which is how the 22 Oct run fell back."""
    text = (Path(__file__).parent.parent / "configs/studies/qr-round1-list.md").read_text(encoding="utf-8")
    pins = pinning.parse(text)
    window = pd.Timestamp(pins.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timestamp(pins.data_start, tz="UTC")
    assert window / pd.Timedelta(minutes=1) <= research_run.OOS_EXEC_BUDGET
    store = pd.Timestamp("2026-10-23", tz="UTC") - pd.Timestamp("2019-09-08", tz="UTC")
    assert store / pd.Timedelta(minutes=1) > research_run.OOS_EXEC_BUDGET


def test_the_dry_run_says_which_bars_a_row_gets(monkeypatch):
    monkeypatch.setattr(research_run, "OOS_EXEC_BUDGET", 25 * 1440)
    hs, pins = _store(), _pins()
    said = rescore._minute_check(_req(pins), history=hs)
    assert said.startswith("; execution bars: 1-minute, 21 Jan 2025 to "), said
    monkeypatch.setattr(research_run, "OOS_EXEC_BUDGET", 10 * 1440)
    assert "REFUSED" in rescore._minute_check(_req(pins), history=hs)
    monkeypatch.setattr(research_run, "OOS_EXEC_BUDGET", 25 * 1440)
    assert rescore._minute_check(_req(pins, minutes=1440), history=hs).startswith("; execution bars: 1-minute")
