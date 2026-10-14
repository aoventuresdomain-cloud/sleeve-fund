"""HUB-RESTART-FILL (FE finding 13 Oct, HoE): a hub restart never leaves the history with a hole for good while the
venue still has the candles. A clean restart was already refilled from the first live bar (last_closes seeds the
relay's gap state from the store); what stayed for good was a hole an earlier refill couldn't close (the venue's REST
down, or the outage longer than the candles it returns) or a refill a restart cut off, since the next start counted
only from the store's last minute. At start the hub now refills the store's holes from the last day through the same
refill path and writer (relay._fill, the store thread: #136), to the store only; a refill that fails or comes back
short is alerted and leaves the gap as it is, never made up."""

from types import SimpleNamespace

import pandas as pd
import pytest
from test_hub import BTC, T0, _bar, _candles, _Fan, _Now

from sleeve_fund.history import HistoryStore
from sleeve_fund.hub.__main__ import last_closes, start_holes
from sleeve_fund.hub.relay import MINUTE_NS, HubRelay, HubRelayConfig, store_sink

V, P = "BINANCE", "BTC/USDT"
PROFILE = SimpleNamespace(name=V)


def _venue(start="2026-10-05 10:00", n=150):
    """The venue's recent 1-minute candles by open time (the newest forming), 10:00 to 12:29 by default."""
    return lambda pair, minutes: _candles(start, n)


def _down(pair, minutes):
    raise ConnectionError("venue REST unavailable")


def _live(store, first_close, last_close):
    """Live bars the hub stored for the minutes closing first_close..last_close (ns)."""
    rows = [(c - MINUTE_NS, 60000.0, 60010.0, 59990.0, 60000.1, 1.25)
            for c in range(first_close, last_close + 1, MINUTE_NS)]
    store.append_bars(V, P, rows, "live")


def _start(store, recent, now=T0, since=None):
    """A hub started at `now` as hub.__main__.build starts it, its refills and writes run in line; on_start's refill of
    the store's holes done. Returns (relay, alerts)."""
    alerts, pairs = [], {BTC: P}
    r = HubRelay(HubRelayConfig(instrument_ids=(BTC,)))
    r.attach(_Fan(), sink=store_sink(store, V, pairs), pairs=pairs, recent=recent,
             last_close=last_closes(PROFILE, pairs, store),
             holes=start_holes(PROFILE, pairs, store, now=pd.Timestamp(now, unit="ns", tz="UTC")),
             alert=lambda key, message: alerts.append(message))
    r._refill = r._store = _Now()
    r._since[BTC] = since if since is not None else now - 30 * 1_000_000_000  # subscribed half way through a minute
    r._refill_holes()
    return r, alerts


def _missing(store):
    return [(f"{a:%H:%M}", f"{b:%H:%M}") for a, b in store.gaps(V, P)]


def test_a_restart_after_twenty_missed_minutes_leaves_no_gap(tmp_path):
    store = HistoryStore(tmp_path)
    _live(store, T0 - 40 * MINUTE_NS, T0 - 21 * MINUTE_NS)  # the hub stored to the minute closing 11:39, then stopped
    r, alerts = _start(store, _venue())
    r.on_bar(_bar(BTC, T0))  # the part bar after subscribing: the venue's candle stands in
    r.on_bar(_bar(BTC, T0 + MINUTE_NS))
    assert _missing(store) == [] and alerts == []
    one = store.read(V, P, 1)
    assert len(one.loc["2026-10-05 11:40":"2026-10-05 12:01"]) == 22


def test_a_hole_an_earlier_refill_could_not_close_is_refilled_at_the_next_start(tmp_path):
    store = HistoryStore(tmp_path)
    _live(store, T0 - 60 * MINUTE_NS, T0 - 41 * MINUTE_NS)  # to 11:19
    first, alerts = _start(store, _down, now=T0 - 30 * MINUTE_NS)  # back at 11:30 with the venue's REST down
    for k in range(-30, -20):
        first.on_bar(_bar(BTC, T0 + k * MINUTE_NS))
    assert _missing(store) == [("11:19", "11:29")], "set-up: the refill failed, the hole stays"
    (said,) = alerts
    # from the last stored minute, which the first live bar refetches (last_closes: it may have been forming)
    assert "12 minute(s) 2026-10-05 11:18 UTC to 2026-10-05 11:29 UTC could not be refilled" in said
    assert "venue REST unavailable" in said and "the next hub start tries again" in said

    second, alerts = _start(store, _venue())  # a deploy at 12:00, the venue's REST back
    assert _missing(store) == [], "the old hole is refilled at start, before any live bar"
    second.on_bar(_bar(BTC, T0))
    assert _missing(store) == [] and alerts == []
    refilled = store.read(V, P, 1).loc["2026-10-05 11:20":"2026-10-05 11:30", "close"]
    assert list(refilled) == [100.0 + 79 + k for k in range(11)]  # the venue's own candles, 11:19 to 11:29


def test_minutes_the_venues_candles_no_longer_reach_are_alerted_and_never_made_up(tmp_path):
    store = HistoryStore(tmp_path)
    _live(store, T0 - 60 * MINUTE_NS, T0 - 41 * MINUTE_NS)
    _live(store, T0 - 10 * MINUTE_NS, T0 - MINUTE_NS)  # 11:19 to 11:49 missing
    _, alerts = _start(store, _venue("2026-10-05 11:40", 50))  # the venue's candles go back to 11:40 only
    assert _missing(store) == [("11:19", "11:39")]
    (said,) = alerts
    assert "21 minute(s) 2026-10-05 11:19 UTC to 2026-10-05 11:39 UTC could not be refilled" in said
    assert "the venue's candles didn't cover them" in said and "never made up" in said


def test_the_start_refill_goes_to_the_store_only_never_to_clients():
    """Minutes long gone: a client connected already has passed them, and would replay them as late minutes."""
    r = HubRelay(HubRelayConfig(instrument_ids=(BTC,)))
    stored = []
    r.attach(_Fan(), sink=stored.extend, pairs={BTC: P}, recent=_venue(),
             holes={BTC: [(T0 - 40 * MINUTE_NS, T0 - 31 * MINUTE_NS)]})
    r._refill = r._store = _Now()
    r._refill_holes()
    assert len(stored) == 10 and all(m["refilled"] for m in stored)
    assert r.fanout.sent == []  # no bars, no gap and no filled message
    r._refill_holes()
    assert len(stored) == 10, "once per start"


def test_on_start_refills_the_stores_holes(monkeypatch):
    calls = []
    clock = SimpleNamespace(timestamp_ns=lambda: T0, set_timer=lambda *a, **k: None, set_timer_ns=lambda *a, **k: None)
    monkeypatch.setattr(HubRelay, "clock", clock, raising=False)
    monkeypatch.setattr(HubRelay, "_relay", lambda self, iid: None)
    monkeypatch.setattr(HubRelay, "_refill_holes", lambda self: calls.append(True))
    HubRelay(HubRelayConfig(instrument_ids=(BTC,))).on_start()
    assert calls == [True]


def test_start_holes_reads_the_last_day_only_and_a_store_it_cant_read_is_skipped(tmp_path, capfd):
    store = HistoryStore(tmp_path)
    day = 24 * 60 * MINUTE_NS
    _live(store, T0 - day - 30 * MINUTE_NS, T0 - day - 21 * MINUTE_NS)
    _live(store, T0 - day + 10 * MINUTE_NS, T0 - 41 * MINUTE_NS)  # a hole across the day's start
    _live(store, T0 - 30 * MINUTE_NS, T0 - MINUTE_NS)  # and one inside it
    now = pd.Timestamp(T0, unit="ns", tz="UTC")
    holes = start_holes(PROFILE, {BTC: P}, store, now=now)
    assert holes == {BTC: [(T0 - day + MINUTE_NS, T0 - day + 9 * MINUTE_NS),
                           (T0 - 40 * MINUTE_NS, T0 - 31 * MINUTE_NS)]}

    class Unreadable:
        def gaps(self, venue, pair):
            raise OSError("bad month file")

    assert start_holes(PROFILE, {BTC: P}, Unreadable(), now=now) == {}
    assert "can't read the history's gaps" in capfd.readouterr().out


def test_f_hrf_1_a_coverage_it_cant_read_never_stops_the_hubs_start_and_is_alerted(tmp_path, capfd):
    """F-HRF-1 (QT2, HoQA MUST FIX): build reads last_closes before the relay starts, so a torn coverage.json must
    leave that instrument out, alerted, never raise: no hub would cost every Strategy on the venue its data."""
    store = HistoryStore(tmp_path)
    _live(store, T0 - 30 * MINUTE_NS, T0 - MINUTE_NS)
    assert last_closes(PROFILE, {BTC: P}, store) == {BTC: T0 - 2 * MINUTE_NS}  # the last stored minute is re-fetched
    next(tmp_path.rglob("coverage.json")).write_text("{not json")
    alerts = []
    assert last_closes(PROFILE, {BTC: P}, store, alert=lambda key, message: alerts.append((key, message))) == {}
    ((key, message),) = alerts
    assert P in key and "can't read the stored history's coverage" in message
    assert "can't read the stored history's coverage" in capfd.readouterr().out


@pytest.mark.parametrize("pair", [None, P], ids=["no-pair", "no-venue-candles"])
def test_a_refill_with_nothing_to_refill_from_says_so(pair):
    r = HubRelay(HubRelayConfig(instrument_ids=(BTC,)))
    alerts = []
    r.attach(_Fan(), sink=lambda m: None, pairs={BTC: pair} if pair else {}, recent=None if pair else _venue(),
             alert=lambda key, message: alerts.append(message))
    r._refill = r._store = _Now()
    r._fill(BTC, T0, T0 + MINUTE_NS)
    assert len(alerts) == 1 and "2 minute(s)" in alerts[0]
    assert r.fanout.sent[-1] == {"t": "filled", "id": BTC, "since": T0, "until": T0 + MINUTE_NS}


def test_the_start_refill_never_replaces_a_stored_bar(tmp_path):
    """Only the store's holes are asked for, so a stored bar (live or refilled) is never offered again, let alone
    replaced: the venue's candles here differ from every stored bar, and none is touched."""
    store = HistoryStore(tmp_path)
    _live(store, T0 - 60 * MINUTE_NS, T0 - 41 * MINUTE_NS)
    _live(store, T0 - 30 * MINUTE_NS, T0 - MINUTE_NS)
    before = store.read(V, P, 1)
    _start(store, _venue())
    after = store.read(V, P, 1)
    assert _missing(store) == [] and len(after) == len(before) + 10
    pd.testing.assert_frame_equal(after.loc[before.index], before)
    assert {r["kind"] for r in store.provenance(V, P)} == {"refill"}  # recorded as refilled, nothing replaced


def _relaying(monkeypatch, seed):
    """A relay attached without the instrument (discover() finds it after start), its subscriptions stubbed."""
    monkeypatch.setattr(HubRelay, "subscribe_trades", lambda self, i: None)
    monkeypatch.setattr(HubRelay, "subscribe_quotes", lambda self, i: None)
    monkeypatch.setattr(HubRelay, "cache", SimpleNamespace(instrument=lambda i: None), raising=False)
    stored = []
    r = HubRelay(HubRelayConfig())
    r.attach(_Fan(), sink=stored.extend, pairs={BTC: P}, recent=_venue(), seed=seed, alert=lambda k, m: None)
    r._refill = r._store = _Now()
    return r, stored


def test_an_instrument_found_after_start_is_seeded_from_the_store_on_its_first_relay(tmp_path, monkeypatch):
    """P1-L16 (HoE: in scope): read when it is first relayed, so the minutes missed are refilled from its first bar
    and the store's holes at once, as for one known at start."""
    from sleeve_fund.hub.__main__ import seeder

    store = HistoryStore(tmp_path, clock=lambda: pd.Timestamp(T0, unit="ns", tz="UTC"))
    _live(store, T0 - 60 * MINUTE_NS, T0 - 41 * MINUTE_NS)
    _live(store, T0 - 30 * MINUTE_NS, T0 - 21 * MINUTE_NS)  # to 11:39, a hole 11:19 to 11:29 before it
    pairs = {BTC: P}
    r, _ = _relaying(monkeypatch, seeder(PROFILE, pairs, store))
    r.sink = store_sink(store, V, pairs)
    r._relay(BTC)
    assert r.gaps.last[BTC] == T0 - 22 * MINUTE_NS  # the last stored minute, by its open (as last_closes)
    assert _missing(store) == [], "its hole refilled on the first relay"
    r._since[BTC] = T0 - 30 * 1_000_000_000
    r.on_bar(_bar(BTC, T0))
    gap = next(m for m in r.fanout.sent if m["t"] == "gap")
    assert (gap["since"], gap["until"]) == (T0 - 21 * MINUTE_NS, T0)
    assert _missing(store) == []


def test_seeding_reads_the_store_once_and_never_over_what_the_relay_knows(monkeypatch):
    calls = []
    r, _ = _relaying(monkeypatch, lambda iid: calls.append(iid) or (T0 - 50 * MINUTE_NS, []))
    r.gaps.last[BTC] = T0 - 5 * MINUTE_NS  # seeded at start, or a bar already seen
    r._relay(BTC)
    assert calls == [] and r.gaps.last[BTC] == T0 - 5 * MINUTE_NS


def test_a_store_it_cant_read_on_first_relay_still_relays_the_instrument(monkeypatch, capfd):
    def seed(iid):
        raise OSError("bad coverage file")

    r, stored = _relaying(monkeypatch, seed)
    r._relay(BTC)
    assert BTC in r.relayed and BTC not in r.gaps.last and stored == []
    assert "can't read the stored history" in capfd.readouterr().out
