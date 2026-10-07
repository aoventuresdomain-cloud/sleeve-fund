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


def test_a_second_missing_settlement_keeps_the_episode_open_when_the_first_arrives(tmp_path, monkeypatch, binance):
    """08:00 never arrives, 16:00 arrives at 16:30: one funding_stale for the whole stretch and no
    funding_stale_cleared, so the inbox never says the instrument is fine while a charge is still on the baseline."""
    # A trade every 5 s: with #146, a trade gap past UNSEEN_GAP_NS (15 s) defers funding while the minutes it missed
    # are still to come, so 30-second ticks would never settle in this harness.
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

    @property
    def alerts(self):
        """What was sent about episodes (funding_stale / funding_stale_cleared), the per-settlement marks left out."""
        return [x for x in self.sent if x[1].startswith("funding_stale")]

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
    assert inbox.alerts == []  # the strategy's alert stands for the instrument
    fresh = [int(t.timestamp() * 1000) for t in pd.date_range(kept[-1] + H8, now, freq="8h")]
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [(t, 0.0001) for t in fresh if t >= start])
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [(level, kind) for level, kind, _ in inbox.alerts] == [("info", "funding_stale_cleared")]
    assert inbox.alerts[0][2].startswith(tag)


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
    assert [k for _, k, _ in inbox.alerts] == ["funding_stale_cleared"]
    assert inbox.alerts[0][2].startswith(tag) and "BINANCE" not in inbox.alerts[0][2].upper().replace(tag, "")
    # The venue stops publishing: the next outage is alerted, once.
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [])
    monkeypatch.setattr(funding.pd.Timestamp, "now", classmethod(lambda cls, tz=None: now + 5 * H8))
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [k for _, k, _ in inbox.alerts] == ["funding_stale_cleared", "funding_stale"]


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
        monkeypatch.setattr(history, "_now", lambda at=at: pd.Timestamp(at, tz="UTC"))
        history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert inbox.alerts == []  # 08:00 still missing: the episode stays open
    kept[t8] = 0.0002
    monkeypatch.setattr(funding, "stale", lambda v, p, root=None, now=None:
                        real_stale(v, p, root, now=pd.Timestamp("2025-10-03 12:30", tz="UTC")))
    monkeypatch.setattr(history, "_now", lambda: pd.Timestamp("2025-10-03 12:30", tz="UTC"))
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [(k, m.split("newest rate ")[-1]) for _, k, m in inbox.alerts] == [
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
    for name in ("_newest_kept", "_funding_missed", "_funding_watch_unread"):
        setattr(s, name, MethodType(getattr(LongFlatStrategy, name), s))
    monkeypatch.setattr("sleeve_fund.strategies.base.pair_of", lambda instrument: PAIR)
    for minute in (31, 32):
        LongFlatStrategy._watch_funding_recovery(s, None, t16 + pd.Timedelta(minutes=minute))
    assert [k for _, k, _ in inbox.alerts] == ["funding_stale"]
    assert "16:00" in inbox.alerts[0][2] and inbox.alerts[0][2].startswith(tag)
    assert "episode from 2025-10-03 16:00 UTC" in inbox.alerts[0][2]  # opened on it, so only its own rate closes it


def _hub_pass(monkeypatch, binance, store, at, kept: dict, history_rates: dict | None = None):
    """One collector pass over the instrument at simulated time `at`, its store holding `kept`, alerting into the
    paper journal `store` at that time."""
    from types import SimpleNamespace

    import sleeve_fund.store as store_mod
    from o17_harness import _REAL_RATES, write_rates
    from sleeve_fund import funding, history

    write_rates(kept)
    real_stale = funding.stale
    with monkeypatch.context() as m:
        m.setattr(funding, "rates", _REAL_RATES)
        m.setattr(funding, "stale", lambda v, p, root=None, now=None: real_stale(v, p, root, now=utc(at)))
        m.setattr(history, "_now", lambda: utc(at))
        m.setattr(binance, "funding_loader", lambda pair, start: [  # the venue's history, for a backfill
            (int(utc(t).timestamp() * 1000), r) for t, r in sorted((history_rates or {}).items())
            if int(utc(t).timestamp() * 1000) >= start])
        m.setattr(binance, "stats_loaders", {})
        m.setattr(store_mod, "Store", lambda *a, **k: SimpleNamespace(
            events_of=store.events_of,
            event=lambda sleeve, level, kind, message, ts=None: store.event(sleeve, level, kind, message,
                                                                            ts=ts or utc(at).to_pydatetime())))
        history._warned.clear()
        history._refresh_funding(binance, PAIR, funding.DEFAULT_ROOT, None)
    funding._cache.clear()


def test_the_collector_does_not_close_an_episode_on_a_later_settlement_while_its_own_never_arrives(
        tmp_path, monkeypatch, binance):
    """08:00 never arrives, 16:00 arrives at 16:30. Paper opens the episode at 08:15; the hub's passes after 16:30 see
    16:00 kept but never 08:00, so they leave it open: one funding_stale and no clear, rather than an open/clear pair
    on every pass (CR on #163)."""
    from o17_harness import journal
    from sleeve_fund import history

    monkeypatch.setattr(history, "_stale", set())
    store = journal()
    paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 17:40", 1)), "2025-10-03 07:50", 530,
          step=5, rates={"2025-10-03 16:00": 0.0002}, store=store, name="w",
          published={"2025-10-03 08:00": NEVER, "2025-10-03 16:00": "2025-10-03 16:30"})
    kept = {"2025-10-02 16:00": 0.0001, "2025-10-03 00:00": 0.0001, "2025-10-03 16:00": 0.0002}
    for at in ("2025-10-03 16:45", "2025-10-03 17:45", "2025-10-03 18:45"):
        _hub_pass(monkeypatch, binance, store, at, kept)
    got = [e["kind"] for e in store.events(None, limit=500) if e["kind"].startswith("funding_stale")]
    assert got == ["funding_stale"], got
    kept["2025-10-03 08:00"] = 0.0001  # backfilled at last: the episode closes, once
    _hub_pass(monkeypatch, binance, store, "2025-10-03 19:45", kept)
    _hub_pass(monkeypatch, binance, store, "2025-10-03 20:45", kept)
    got = [e["kind"] for e in reversed(store.events(None, limit=500)) if e["kind"].startswith("funding_stale")]
    assert got == ["funding_stale", "funding_stale_cleared"], got


def test_the_collector_does_not_close_an_episode_while_a_later_settlement_is_due_and_missing(
        tmp_path, monkeypatch, binance):
    """08:00 is published late, at 17:00, and 16:00 never comes. The hub's pass at 17:30 keeps 08:00 but 16:00 is due
    and missing, so the episode stays open: no false clear (QA P1-O17a-10)."""
    from o17_harness import journal
    from sleeve_fund import history

    monkeypatch.setattr(history, "_stale", set())
    store = journal()
    paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 17:40", 1)), "2025-10-03 07:50", 590,
          step=5, rates={"2025-10-03 08:00": 0.0002}, store=store, name="w",
          published={"2025-10-03 08:00": "2025-10-03 17:00", "2025-10-03 16:00": NEVER})
    kept = {"2025-10-02 16:00": 0.0001, "2025-10-03 00:00": 0.0001, "2025-10-03 08:00": 0.0002}
    for at in ("2025-10-03 17:30", "2025-10-03 18:30"):
        _hub_pass(monkeypatch, binance, store, at, kept)
    got = [e["kind"] for e in store.events(None, limit=500) if e["kind"].startswith("funding_stale")]
    assert got == ["funding_stale"], got
    kept["2025-10-03 16:00"] = 0.0001  # backfilled: now every settlement due is kept, and the episode closes
    _hub_pass(monkeypatch, binance, store, "2025-10-03 19:30", kept)
    got = [e["kind"] for e in reversed(store.events(None, limit=500)) if e["kind"].startswith("funding_stale")]
    assert got == ["funding_stale", "funding_stale_cleared"], got


def test_paper_rebuilds_its_watched_settlements_after_a_restart():
    """A restart while 08:00 and 16:00 are still charged the baseline: the open episode's baseline rows from the
    journal are watched again, and older or settled rows are not (QA P1-O17a-10)."""
    from types import SimpleNamespace

    from sleeve_fund import funding
    from sleeve_fund.strategies.base import LongFlatStrategy

    t8, t16 = (pd.Timestamp(f"2025-10-03 {h}:00", tz="UTC") for h in ("08", "16"))
    tag = funding.stale_tag("BINANCE", PAIR)
    inbox = _Journal(events=[{"kind": "funding_stale", "message": f"{tag} No settled funding rate", "ts": t8}])
    rows = [{"ts": t16, "kind": "baseline"}, {"ts": t8, "kind": "baseline"}, {"ts": t8 - pd.Timedelta(hours=8),
            "kind": "settled"}, {"ts": t8 - pd.Timedelta(days=2), "kind": "baseline"}]
    inbox.funding = lambda sleeve, limit=1000: rows
    perp = SimpleNamespace(funding_venue="BINANCE", funding_hours=(0, 8, 16))
    s = SimpleNamespace(_cfg=SimpleNamespace(perp=perp), instrument=None, _funding_missing=set(),
                        runtime=SimpleNamespace(store=inbox, name="w"))
    import sleeve_fund.strategies.base as base

    orig, base.pair_of = base.pair_of, lambda instrument: PAIR
    try:
        LongFlatStrategy._rebuild_funding_missing(s)
    finally:
        base.pair_of = orig
    assert s._funding_missing == {t8, t16}


def test_a_restart_never_watches_a_settlement_already_reversed():
    """CR, #163: a baseline the venue's records showed was no settlement was reversed by its own row while another
    settlement kept the episode open. A restart must not watch it again, or the first watch would refund it a
    second time: a settlement with a reversal row, or whose rows net to zero, is skipped."""
    from types import SimpleNamespace

    from sleeve_fund import funding
    from sleeve_fund.strategies.base import LongFlatStrategy

    t8, t12, t16 = (pd.Timestamp(f"2025-10-03 {h}:00", tz="UTC") for h in ("08", "12", "16"))
    tag = funding.stale_tag("BINANCE", PAIR)
    inbox = _Journal(events=[{"kind": "funding_stale", "message": f"{tag} No settled funding rate", "ts": t8}])
    row = dict(qty=0.1, price=60_000.0, rate=0.0001)
    rows = [{"ts": t8, "kind": "baseline", "amount": -0.6, **row},  # still missing: keeps the episode open
            {"ts": t12, "kind": "baseline", "amount": -0.6, **row}, {"ts": t12, "kind": "reversal", "amount": 0.6, **row},
            {"ts": t16, "kind": "baseline", "amount": -0.6, **row}, {"ts": t16, "kind": "settled", "amount": 0.6, **row}]
    inbox.funding = lambda sleeve, limit=1000: rows
    perp = SimpleNamespace(funding_venue="BINANCE", funding_hours=(0, 8, 16))
    s = SimpleNamespace(_cfg=SimpleNamespace(perp=perp), instrument=None, _funding_missing=set(), _funding_paid={},
                        runtime=SimpleNamespace(store=inbox, name="w"))
    import sleeve_fund.strategies.base as base

    orig, base.pair_of = base.pair_of, lambda instrument: PAIR
    try:
        LongFlatStrategy._rebuild_funding_missing(s)
    finally:
        base.pair_of = orig
    assert s._funding_missing == {t8} and set(s._funding_paid) == {t8}, (s._funding_missing, s._funding_paid)


def test_the_recovery_watch_runs_while_funding_is_deferred(monkeypatch):
    """With trades over 15 s apart, #146 defers the funding charge; the watch on missing settlements still runs, so
    the strategy clears or reopens its episode on time (QA P1-O17a-12)."""
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import LongFlatStrategy

    now = pd.Timestamp("2025-10-03 16:30", tz="UTC").to_pydatetime()
    calls = []
    s = SimpleNamespace(_cfg=SimpleNamespace(perp=SimpleNamespace()), _funding_missing={now}, _backtest=False,
                        _entry_px=100.0, _awaiting=object(), _trade_ns=None, clock=SimpleNamespace(utc_now=lambda: now),
                        _watch_funding_recovery=lambda terms, at: calls.append(at))
    LongFlatStrategy._apply_funding(s, 100.0)
    assert calls == [now]


def _episode_kinds(store):
    return [(e["kind"], e["message"]) for e in reversed(store.events(None, limit=500))
            if e["kind"] in ("funding_stale", "funding_stale_cleared", "funding_never_published")]


def _opened(store, at, o):
    from sleeve_fund import funding

    store.event(None, "warning", "funding_stale", f"{funding.stale_tag('', PAIR)} No settled funding rate from the "
                f"venue; {funding.from_words(utc(o))}", ts=utc(at).to_pydatetime())


def test_a_settlement_never_published_closes_its_episode_after_a_day_and_a_later_outage_alerts(monkeypatch, binance):
    """08:00 never comes; 16:00 and the next day's are kept. Until 08:00 the next day the episode stays open; then it
    is marked never published once (a warning: the baseline stays) and the episode closes, so a later outage alerts
    again rather than sitting in an episode open forever (Advisor, 7 Oct 2026, QA P1-O17a-11)."""
    from o17_harness import journal
    from sleeve_fund import history

    monkeypatch.setattr(history, "_stale", set())
    store = journal()
    _opened(store, "2025-10-03 08:15", "2025-10-03 08:00")
    kept = {"2025-10-03 00:00": 0.0001, "2025-10-03 16:00": 0.0001, "2025-10-04 00:00": 0.0001}
    _hub_pass(monkeypatch, binance, store, "2025-10-04 07:30", kept)
    assert [k for k, _ in _episode_kinds(store)] == ["funding_stale"]  # under a day: still missing
    kept["2025-10-04 08:00"] = 0.0001
    for at in ("2025-10-04 08:30", "2025-10-04 09:30"):
        _hub_pass(monkeypatch, binance, store, at, kept)
    got = _episode_kinds(store)
    assert [k for k, _ in got] == ["funding_stale", "funding_never_published", "funding_stale_cleared"], got
    assert "2025-10-03 08:00 UTC rate never published; baseline kept, true-up impossible" in got[1][1]
    assert "episode from 2025-10-03 08:00 UTC" in got[2][1]
    # A later outage: the store stops at 08:00 on the 4th, so by 01:00 on the 5th it is stale and alerts anew
    _hub_pass(monkeypatch, binance, store, "2025-10-05 01:00", kept)
    assert [k for k, _ in _episode_kinds(store)][-1] == "funding_stale"


def test_a_missing_settlement_is_backfilled_from_the_venue_history_before_it_is_called_never_published(
        monkeypatch, binance):
    """The store never got 08:00, but the venue's history has it: the hub asks for it again, keeps it and closes the
    episode, with nothing marked never published (Advisor, 7 Oct 2026)."""
    from o17_harness import _REAL_RATES, journal
    from sleeve_fund import funding, history

    monkeypatch.setattr(history, "_stale", set())
    store = journal()
    _opened(store, "2025-10-03 08:15", "2025-10-03 08:00")
    kept = {"2025-10-03 00:00": 0.0001, "2025-10-03 16:00": 0.0001}
    _hub_pass(monkeypatch, binance, store, "2025-10-03 16:30", kept, history_rates={**kept, "2025-10-03 08:00": 0.0002})
    got = [k for k, _ in _episode_kinds(store)]
    assert got == ["funding_stale", "funding_stale_cleared"], got
    assert utc("2025-10-03 08:00") in _REAL_RATES("BINANCE", PAIR, funding.DEFAULT_ROOT).index  # kept now


def test_a_missing_settlement_after_a_published_one_opens_its_own_episode():
    """08:00 missing (an open episode), 16:00 published, then 00:00 missing: not contiguous with the open episode, so
    it opens its own and alerts again; 16:00 missing straight after 08:00 would have joined it (Advisor, 7 Oct 2026)."""
    from sleeve_fund import funding

    t = [utc(x) for x in ("2025-10-03 08:00", "2025-10-03 16:00", "2025-10-04 00:00")]
    state = {"open": {t[0]: {}}, "missing": {t[0], t[2]}, "never": set()}
    assert funding.episode_of(state, t[2], t) is None  # 16:00 was published in between
    state["missing"].add(t[1])
    assert funding.episode_of(state, t[2], t) == t[0]  # 08:00, 16:00, 00:00 all missing: one outage
    state = {"open": {t[0]: {}}, "missing": {t[0]}, "never": set()}
    assert funding.episode_of(state, t[1], t) == t[0]


def test_a_batch_mixing_flat_and_held_settlements_marks_every_one(monkeypatch):
    """CR minor 1: one charge covering 08:00 (flat), 16:00 and 00:00 (held), as a daily bar with a position opened
    mid-day gives it, marks all three, so funding.baseline_summary sees no hole in the record."""
    from types import MethodType, SimpleNamespace

    from sleeve_fund import funding, markets
    from sleeve_fund.strategies.base import LongFlatStrategy

    t = [utc(x).to_pydatetime() for x in ("2025-10-03 08:00", "2025-10-03 16:00", "2025-10-04 00:00")]
    terms = SimpleNamespace(funding_venue=None, funding_hours=(0, 8, 16), funding_rate=markets.LOW_FEE_PERP.funding_rate)
    ns = lambda d: int(d.timestamp()) * 1_000_000_000  # noqa: E731
    s = SimpleNamespace(_cfg=SimpleNamespace(perp=terms), _backtest=True, _entry_px=None, _funding_missing=set(),
                        _funding_since=utc("2025-10-03 01:00").to_pydatetime(), _intrabar=None, _funding_skip=None,
                        _held_at={ns(t[0]): (0.0, 100.0), ns(t[1]): (1.0, 100.0), ns(t[2]): (1.0, 100.0)},
                        _net_position=lambda: (1.0,), funding_marks=[], funding_log=[], _cash_adj=0.0, runtime=None,
                        _settled=None, _funding_fallback_said=False, instrument=None,
                        FUNDING_WAIT=LongFlatStrategy.FUNDING_WAIT, FUNDING_LOOKBACK=LongFlatStrategy.FUNDING_LOOKBACK,
                        clock=SimpleNamespace(utc_now=lambda: utc("2025-10-04 00:00").to_pydatetime()))
    for name in ("_settlements", "_funding_rate", "_mark_flat", "_book_funding", "_rescan_from"):
        setattr(s, name, MethodType(getattr(LongFlatStrategy, name), s))
    LongFlatStrategy._apply_funding(s, 100.0)
    assert [(utc(m[0]), m[2]) for m in s.funding_marks] == [(utc(t[0]), False), (utc(t[1]), True), (utc(t[2]), True)]
    assert funding.baseline_summary(s.funding_marks)[1] == 2  # two held, the flat one counted as flat


def test_paper_marks_a_settlement_never_published_after_a_day_once_and_closes_its_episode(monkeypatch):
    """08:00 never comes and 16:00 did: a day after 08:00, the strategy marks it never published (a warning; its
    baseline stays) and closes the episode. A second strategy on the instrument, or the hub, marks nothing again
    (Advisor, 7 Oct 2026, QA P1-O17a-11)."""
    from types import MethodType, SimpleNamespace

    from sleeve_fund import funding
    from sleeve_fund.strategies.base import LongFlatStrategy

    tag = funding.stale_tag("", PAIR)
    t8, t16 = utc("2025-10-03 08:00"), utc("2025-10-03 16:00")
    at = t8 + pd.Timedelta(hours=24, minutes=1)
    inbox = _Journal(events=[
        {"kind": "funding_missing", "message": f"{tag} 2025-10-03 08:00 UTC settlement missing", "ts": t8},
        {"kind": "funding_stale", "message": f"{tag} No settled funding rate; {funding.from_words(t8)}", "ts": t8}])
    monkeypatch.setattr("sleeve_fund.strategies.base.pair_of", lambda instrument: PAIR)

    def strategy():
        s = SimpleNamespace(_funding_recheck=None, _funding_missing={t8}, instrument=None, _funding_last_settled=t16,
                            FUNDING_RECHECK=pd.Timedelta(0), _venue_rate=lambda terms, pair, when: None,
                            runtime=SimpleNamespace(store=inbox, now=lambda: at))
        for name in ("_funding_episode", "_newest_kept", "_funding_missed", "_funding_watch_unread"):
            setattr(s, name, MethodType(getattr(LongFlatStrategy, name), s))
        return s

    first, second = strategy(), strategy()
    LongFlatStrategy._watch_funding_recovery(first, None, at)
    LongFlatStrategy._watch_funding_recovery(second, None, at)
    assert [k for _, k, _ in inbox.sent] == ["funding_never_published", "funding_stale_cleared"], inbox.sent
    assert "rate never published; baseline kept, true-up impossible" in inbox.sent[0][2]
    assert not first._funding_missing and not second._funding_missing


def test_paper_reverses_the_baseline_for_a_foreseen_settlement_the_venue_never_made(tmp_path, monkeypatch, binance):
    """4-hourly records to 08:00, then the venue goes back to 8-hourly. 12:00, foreseen from the 4 h step, is alerted
    and charged the baseline 15 minutes after it was due (QA P1-O17a-13). The 16:00 record alone leaves it provisionally
    missing (the step after the gap is still unknown); once 4 Oct 00:00 is kept, 8 h after 16:00, it shows 12:00 never
    settled, so the charge is reversed by its own journaled row (kind "reversal"), the original row untouched, and the
    episode closes (Advisor, 7 Oct 2026, (c) as amended 04:31)."""
    from o17_harness import journal

    from sleeve_fund import funding

    rates = {**{f"2025-10-03 {h}": 0.0001 for h in ("00:00", "04:00", "08:00", "16:00")}, "2025-10-04 00:00": 0.0001}
    store = journal()
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-04 00:40", 1)), "2025-10-03 07:50",
                1010, step=5, rates=rates, store=store, name="w", published={"2025-10-03 12:00": NEVER},
                stored={"2025-10-03 00:00": "2025-10-03 00:00", "2025-10-03 04:00": "2025-10-03 04:00",
                        "2025-10-03 08:00": "2025-10-03 08:00", "2025-10-03 16:00": "2025-10-03 16:01",
                        "2025-10-04 00:00": "2025-10-04 00:01"})
    at12 = sorted((r for r in out["funding"] if utc(r["ts"]) == utc("2025-10-03 12:00")), key=lambda r: r["id"])
    assert [r.get("kind") for r in at12] == ["baseline", "reversal"], at12
    assert at12[0]["amount"] < 0 and at12[1]["amount"] == pytest.approx(-at12[0]["amount"])
    stale = [e for e in store.events(None, limit=500) if e["kind"] == "funding_stale"]
    assert stale and utc("2025-10-03 12:15") <= pd.Timestamp(stale[-1]["ts"]) < utc("2025-10-03 12:17")
    # 16:00 foresees 20:00 at the 4 h step (the shorter, adverse one, counting 12:00): missed too, its own episode
    # (16:00 kept between them), and reversed with 12:00 once 00:00 shows the 8 h step
    at20 = [r.get("kind") for r in sorted(out["funding"], key=lambda r: r["id"])
            if utc(r["ts"]) == utc("2025-10-03 20:00")]  # by id: the journal orders rows at one time arbitrarily
    assert at20 == ["baseline", "reversal"], out["funding"]
    got = [e["kind"] for e in reversed(store.events(None, limit=500)) if e["kind"].startswith("funding_stale")]
    assert sorted(got) == ["funding_stale"] * 2 + ["funding_stale_cleared"] * 2, got
    assert funding.journal_state(store, "[BTC/USDT]")["open"] == {}, got
