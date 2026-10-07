"""The funding staleness episode and the backtest's baseline warning (CR on #163): one episode per instrument,
whether the collector or a paper strategy notices first; it stays open while any settlement is still missing; and a
backtest's one funding_fallback warning names a settlement a position was held through, never a flat one."""

from __future__ import annotations

import json

import pandas as pd

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


def test_a_second_missing_settlement_keeps_the_episode_open_when_the_first_arrives(tmp_path, monkeypatch, binance):
    """08:00 never arrives, 16:00 arrives at 16:30: one funding_stale for the whole stretch and no
    funding_stale_cleared, so the inbox never says the instrument is fine while a charge is still on the baseline."""
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 16:40", 1)),
                "2025-10-03 07:50", 530, step=5, rates={"2025-10-03 16:00": 0.0002},
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


def test_paper_and_collector_share_the_instrument_tag_without_the_venue():
    from sleeve_fund import funding

    assert funding.stale_tag("binance", PAIR) == funding.stale_tag("", PAIR) == f"[{PAIR}]"  # no venue name shown


def test_a_rate_beyond_the_published_cap_is_alerted_and_still_kept(tmp_path, monkeypatch):
    """The Advisor's guard until DA-11: a BTC perp rate past the instrument's published cap (0.3%) is alerted, never
    rejected or dropped; one within it says nothing."""
    from sleeve_fund import funding, history
    from sleeve_fund.venues import venue

    profile = venue("BINANCE")
    assert profile.published_funding_caps["BTC/USDT"] == 0.003
    t0 = pd.Timestamp("2025-10-03", tz="UTC")
    rows = [(int((t0 + i * pd.Timedelta(hours=8)).timestamp() * 1000), r) for i, r in enumerate((0.0001, 0.004, -0.0035))]
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [x for x in rows if x[0] >= start])
    monkeypatch.setattr(profile, "stats_loaders", {})
    inbox = _Journal()
    monkeypatch.setattr("sleeve_fund.store.Store", lambda *a, **k: inbox)
    history._warned.clear()
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    kept = funding.rates("BINANCE", "BTC/USDT", tmp_path)
    assert list(kept) == [0.0001, 0.004, -0.0035]  # kept and charged as settled
    (over,) = [m for lv, k, m in inbox.sent if k == "funding_over_cap" and lv == "warning"]
    assert "0.4000%" in over and "-0.3500%" in over and "0.30%" in over
    inbox.sent.clear()
    history._warned.clear()
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)  # nothing new: nothing said again
    assert not [k for _, k, _ in inbox.sent if k == "funding_over_cap"]


def test_an_episode_left_open_by_a_restart_is_closed_once_the_rates_keep_up_and_the_next_outage_alerts(
        tmp_path, monkeypatch):
    """The opener (a strategy, or the collector before a deploy) restarted before the rate came: nobody holds the
    episode in memory. The collector, seeing the rates keep up, writes its one clear; a later outage then alerts
    again rather than being held back by the old episode (CR on #163)."""
    from sleeve_fund import funding, history
    from sleeve_fund.venues import venue

    H8 = pd.Timedelta(hours=8)
    now = pd.Timestamp.now(tz="UTC").floor("8h")
    tag = funding.stale_tag("BINANCE", "BTC/USDT")
    inbox = _Journal(events=[{"kind": "funding_stale", "message": f"{tag} No settled funding rate from the venue",
                              "ts": now - 3 * H8}])
    monkeypatch.setattr("sleeve_fund.store.Store", lambda *a, **k: inbox)
    profile = venue("BINANCE")
    monkeypatch.setattr(profile, "stats_loaders", {})
    fresh = [int(t.timestamp() * 1000) for t in pd.date_range(now - 4 * H8, now, freq="8h")]
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [(t, 0.0001) for t in fresh if t >= start])
    history._warned.clear()
    monkeypatch.setattr(history, "_stale", set())  # a new collector process
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [k for _, k, _ in inbox.sent] == ["funding_stale_cleared"]
    assert inbox.sent[0][2].startswith(tag) and "BINANCE" not in inbox.sent[0][2].upper().replace(tag, "")
    # The venue stops publishing: the next outage is alerted, once.
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [])
    monkeypatch.setattr(funding.pd.Timestamp, "now", classmethod(lambda cls, tz=None: now + 5 * H8))
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [k for _, k, _ in inbox.sent] == ["funding_stale_cleared", "funding_stale"]


def test_the_collector_keeps_an_episode_open_until_the_settlement_it_opened_on_is_kept(tmp_path, monkeypatch):
    """Paper opened the episode at 08:15 for 08:00. The hub's store, newest 00:00, is not stale (one interval old),
    but 08:00 is still missing: no clear. Once 08:00 is kept, the next pass closes it (QA P1-O17a-8)."""
    from sleeve_fund import funding, history
    from sleeve_fund.venues import venue

    t8 = pd.Timestamp("2025-10-03 08:00", tz="UTC")
    kept = {t8 - 2 * pd.Timedelta(hours=8): 0.0001, t8 - pd.Timedelta(hours=8): 0.0001}
    tag = funding.stale_tag("BINANCE", "BTC/USDT")
    inbox = _Journal(events=[{"kind": "funding_stale", "message": f"{tag} No settled funding rate from the venue",
                              "ts": t8 + pd.Timedelta(minutes=15)}])
    monkeypatch.setattr("sleeve_fund.store.Store", lambda *a, **k: inbox)
    profile = venue("BINANCE")
    monkeypatch.setattr(profile, "stats_loaders", {})
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [
        (int(t.timestamp() * 1000), r) for t, r in kept.items() if int(t.timestamp() * 1000) >= start])
    real_stale = funding.stale
    history._warned.clear()
    monkeypatch.setattr(history, "_stale", set())
    for at in ("2025-10-03 08:30", "2025-10-03 11:30"):
        monkeypatch.setattr(funding, "stale", lambda v, p, root=None, now=None, at=at:
                            real_stale(v, p, root, now=pd.Timestamp(at, tz="UTC")))
        history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert inbox.sent == []  # 08:00 still missing: the episode stays open
    kept[t8] = 0.0002
    monkeypatch.setattr(funding, "stale", lambda v, p, root=None, now=None:
                        real_stale(v, p, root, now=pd.Timestamp("2025-10-03 12:30", tz="UTC")))
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [(k, m.split("newest rate ")[-1]) for _, k, m in inbox.sent] == [
        ("funding_stale_cleared", "2025-10-03 08:00 UTC")]


def test_a_strategy_still_missing_a_settlement_reopens_an_episode_closed_meanwhile(monkeypatch):
    """The collector closed the episode once the settlement it opened on was kept, but this strategy still waits on a
    later one (16:00): the strategy opens it again, once, so the inbox never reads clear while a charge is on the
    baseline (QA P1-O17a-8)."""
    from types import MethodType, SimpleNamespace

    from sleeve_fund import funding
    from sleeve_fund.strategies.base import LongFlatStrategy

    tag = funding.stale_tag("", PAIR)
    t16 = pd.Timestamp("2025-10-03 16:00", tz="UTC")
    inbox = _Journal(events=[{"kind": "funding_stale_cleared", "message": f"{tag} funding kept up again",
                              "ts": t16 + pd.Timedelta(minutes=30)}])
    s = SimpleNamespace(_funding_recheck=None, _funding_missing={t16}, instrument=None, FUNDING_RECHECK=pd.Timedelta(0),
                        _venue_rate=lambda terms, pair, when: None,
                        runtime=SimpleNamespace(store=inbox, now=lambda: t16 + pd.Timedelta(minutes=31)))
    s._funding_episode = MethodType(LongFlatStrategy._funding_episode, s)
    monkeypatch.setattr("sleeve_fund.strategies.base.pair_of", lambda instrument: PAIR)
    for minute in (31, 32):
        LongFlatStrategy._watch_funding_recovery(s, None, t16 + pd.Timedelta(minutes=minute))
    assert [k for _, k, _ in inbox.sent] == ["funding_stale"]
    assert "16:00" in inbox.sent[0][2] and inbox.sent[0][2].startswith(tag)
