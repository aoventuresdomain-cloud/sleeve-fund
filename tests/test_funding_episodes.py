"""The funding staleness episode and the backtest's baseline warning (CR on #163): one episode per instrument,
whether the collector or a paper strategy notices first; it stays open while any settlement is still missing; and a
backtest's one funding_fallback warning names a settlement a position was held through, never a flat one."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from o17_harness import (  # noqa: F401  (fixtures are used by name)
    NEVER,
    PAIR,
    _no_swallowed_strategy_errors,
    _o17win,
    backtest,
    binance,
    hourly_bars,
    kinds,
    paper,
    settlements,
    utc,
    win,
    write_rates,
)


@pytest.fixture(autouse=True)
def _caps_known(monkeypatch):
    """The staleness alerts only: the cap's own alert is pinned in test_funding_caps, and the venue isn't asked."""
    from sleeve_fund import history

    monkeypatch.setattr(history, "_keep_funding_caps", lambda *a, **k: None)
    monkeypatch.setattr(history, "_alert_missing_cap", lambda *a, **k: None)


def test_a_second_missing_settlement_keeps_the_episode_open_when_the_first_arrives(tmp_path, monkeypatch, binance):
    """08:00 never arrives, 16:00 arrives at 16:30: one funding_stale for the whole stretch and no
    funding_stale_cleared, so the inbox never says the instrument is fine while a charge is still on the baseline."""
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 16:40", 1)),
                "2025-10-03 07:50", 530, step=30, rates={"2025-10-03 16:00": 0.0002},
                published={"2025-10-03 08:00": NEVER, "2025-10-03 16:00": "2025-10-03 16:30"})
    assert len(kinds(out["events"], "funding_stale")) == 1
    assert not kinds(out["events"], "funding_stale_cleared")


def test_a_backtest_says_the_baseline_only_for_a_held_settlement(binance):
    """08:00 is missing while flat, 16:00 missing while long: the one warning is about 16:00."""
    bars = hourly_bars("2025-10-02 00:00", "2025-10-04 00:00")
    missing = {utc("2025-10-03 08:00"), utc("2025-10-03 16:00")}
    write_rates({t: 0.0001 for t in settlements(bars.index[0] - pd.Timedelta(days=1), bars.index[-1])
                 if t not in missing})
    r = backtest(binance, bars, win(("2025-10-03 12:00", "2025-10-03 20:00", 1)))
    (fallback,) = [e for e in r.journal.events_ if e["kind"] == "funding_fallback"]
    assert "03 Oct 2025 16:00" in fallback["message"], fallback["message"]


class _Journal:
    """The alerts inbox with the journal's last staleness events, as Store.events_of returns them (newest first)."""

    def __init__(self, *, events=(), engine=None):
        self.kept = list(events)
        self.sent = []

    def events_of(self, kinds, limit=100):
        return [e for e in self.kept if e["kind"] in kinds][:limit]

    def event(self, sleeve, level, kind, message, ts=None):
        self.sent.append((level, kind, message))
        self.kept.insert(0, {"kind": kind, "message": message, "ts": ts or pd.Timestamp.now(tz="UTC")})


def test_the_collector_does_not_alert_an_episode_a_strategy_already_opened(tmp_path, monkeypatch):
    """A paper strategy raised funding_stale for the instrument first: the collector's own staleness check stays
    quiet, and closes the same episode when the rates catch up."""
    from sleeve_fund import funding, history
    from sleeve_fund.venues import venue

    H8 = pd.Timedelta(hours=8)
    now = pd.Timestamp.now(tz="UTC").floor("8h")
    path = funding._path("BINANCE", "BTC/USDT", tmp_path)
    path.parent.mkdir(parents=True)
    kept = [now - (6 - i) * H8 for i in range(3)]
    path.write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), 0.0001] for t in kept]}))
    tag = funding.stale_tag("BINANCE", "BTC/USDT")
    inbox = _Journal(events=[{"kind": "funding_stale", "message": f"{tag} No settled funding rate from the venue",
                              "ts": now}])
    monkeypatch.setattr("sleeve_fund.store.Store", lambda *a, **k: inbox)
    profile = venue("BINANCE")
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [])
    monkeypatch.setattr(profile, "stats_loaders", {})
    history._warned.clear()
    monkeypatch.setattr(history, "_stale", set())
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert inbox.sent == []  # the strategy's alert stands for the instrument
    fresh = [int(t.timestamp() * 1000) for t in pd.date_range(kept[-1] + H8, now, freq="8h")]
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [(t, 0.0001) for t in fresh if t >= start])
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [(level, kind) for level, kind, _ in inbox.sent] == [("info", "funding_stale_cleared")]
    assert inbox.sent[0][2].startswith(tag)


def test_paper_and_collector_share_the_instrument_tag():
    from sleeve_fund import funding

    assert funding.stale_tag("binance", PAIR) == f"[BINANCE {PAIR}]"
