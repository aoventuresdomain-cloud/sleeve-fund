"""DA-10: open interest snapshots and funding rates are kept append-only, survive restarts, and report holes."""

import json
import time

import pandas as pd
import pytest

from sleeve_fund import funding, open_interest
from sleeve_fund.history import _refresh_funding
from sleeve_fund.venues import binance_open_interest, venue

T0 = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp() * 1000)
STEP = 300_000  # five minutes


def _venue_with(times, scale=1.0):
    """A loader over a venue holding a snapshot at each of `times`, paging 500 at a time as Binance does."""
    rows = [(t, 100.0 + i * scale, 6_000_000.0 + i) for i, t in enumerate(times)]

    def load(pair, start):
        return [r for r in rows if r[0] >= start][:500]
    return load


def test_binance_open_interest_parses_and_never_asks_before_the_window_it_keeps():
    asked = []
    now = T0 + 40 * 86_400_000
    rows = [{"symbol": "BTCUSDT", "sumOpenInterest": "81234.5", "sumOpenInterestValue": "5300000000.1",
             "timestamp": T0}]
    out = binance_open_interest("BTC/USDT", 0, get_json=lambda url: asked.append(url) or rows, now_ms=now)
    assert out == [(T0, 81234.5, 5300000000.1)]
    start = int(asked[0].split("startTime=")[1].split("&")[0])
    assert now - 30 * 86_400_000 < start < now
    assert "period=5m" in asked[0] and "symbol=BTCUSDT" in asked[0]


def test_backfill_pages_through_the_window_and_writes_each_snapshot_once(tmp_path):
    times = [T0 + i * STEP for i in range(1200)]  # more than two pages
    out = open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(times))
    assert out["written"] == 1200 and out["conflicts"] == 0
    kept = open_interest.snapshots("BINANCE", "BTC/USDT", root=tmp_path)
    assert len(kept) == 1200 and kept.index.is_monotonic_increasing and not kept.index.duplicated().any()
    assert open_interest.latest("BINANCE", "BTC/USDT", root=tmp_path) == pd.Timestamp(times[-1], unit="ms", tz="UTC")
    assert open_interest.gaps("BINANCE", "BTC/USDT", root=tmp_path) == []


def test_a_restart_neither_duplicates_nor_loses_snapshots(tmp_path):
    times = [T0 + i * STEP for i in range(30)]
    open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(times[:10]))
    # the process stops and starts again; the venue has moved on, and pages overlap what is kept
    again = open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path,
                                  loader=lambda pair, start: _venue_with(times)(pair, 0))
    assert again["written"] == 20 and again["unchanged"] == 10
    twice = open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(times))
    assert twice["written"] == 0
    kept = open_interest.snapshots("BINANCE", "BTC/USDT", root=tmp_path)
    assert list(kept.index) == [pd.Timestamp(t, unit="ms", tz="UTC") for t in times]


def test_a_kept_snapshot_is_never_replaced_and_a_differing_copy_goes_to_provenance(tmp_path):
    times = [T0 + i * STEP for i in range(5)]
    open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(times))
    out = open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path,
                                loader=lambda pair, start: _venue_with(times, scale=2.0)(pair, 0))
    assert out["written"] == 0 and out["conflicts"] == 4  # the first snapshot is the same in both
    assert open_interest.snapshots("BINANCE", "BTC/USDT", root=tmp_path)["contracts"].iloc[-1] == 104.0
    log = open_interest.provenance("BINANCE", "BTC/USDT", root=tmp_path)
    assert {e["series"] for e in log} == {"open_interest"} and log[-1]["stored"][0] == 104.0
    # The series' own file: the price series' provenance.jsonl, whose readers expect bar minutes, is untouched.
    assert not (tmp_path / "BINANCE" / "BTC-USDT" / "provenance.jsonl").exists()


def test_gaps_reports_the_periods_missing_between_snapshots(tmp_path):
    times = [T0, T0 + STEP, T0 + 5 * STEP, T0 + 6 * STEP]
    open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(times))
    assert open_interest.gaps("BINANCE", "BTC/USDT", root=tmp_path) == [
        (pd.Timestamp(T0 + 2 * STEP, unit="ms", tz="UTC"), pd.Timestamp(T0 + 4 * STEP, unit="ms", tz="UTC"))]


def test_funding_gaps_find_a_missed_settlement_but_not_an_interval_change(tmp_path):
    h = 3_600_000
    kept = [T0, T0 + 8 * h, T0 + 16 * h, T0 + 32 * h, T0 + 40 * h, T0 + 44 * h, T0 + 48 * h, T0 + 52 * h]
    funding.refresh("BINANCE", "BTC/USDT", root=tmp_path,
                    loader=lambda pair, start: [(t, 0.0001) for t in kept if t >= start])
    assert funding.gaps("BINANCE", "BTC/USDT", root=tmp_path) == [
        (pd.Timestamp(T0 + 16 * h, unit="ms", tz="UTC"), pd.Timestamp(T0 + 32 * h, unit="ms", tz="UTC"))]


def test_the_collector_keeps_both_and_logs_how_far_each_is_kept(tmp_path, capfd, monkeypatch):
    profile = venue("BINANCE")
    times = [T0 + i * STEP for i in range(3)]
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [(t, 0.0001) for t in (T0,) if t >= start])
    monkeypatch.setattr(profile, "stats_loaders", {"open_interest": _venue_with(times)})
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    line = capfd.readouterr().out.strip().splitlines()[-1]
    assert line == "BINANCE BTC/USDT: open interest to 2026-10-01 00:10 UTC (+3), funding to 2026-10-01 00:00 UTC"


def test_a_funding_failure_does_not_stop_open_interest(tmp_path, capfd, monkeypatch):
    profile = venue("BINANCE")

    def broken(pair, start):
        raise OSError("venue down")
    monkeypatch.setattr(profile, "funding_loader", broken)
    monkeypatch.setattr(profile, "stats_loaders", {"open_interest": _venue_with([T0])})
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    out = capfd.readouterr().out
    assert "funding refresh failed" in out and "open interest to 2026-10-01 00:00 UTC (+1), funding to none yet" in out


def test_each_snapshot_keeps_when_it_was_first_seen_and_the_lag_ignores_the_backfill(tmp_path, monkeypatch):
    monkeypatch.setattr(open_interest, "MIN_LIVE", 1)  # the lag from a handful of live captures
    stamps = [T0 + i * STEP for i in range(4)]
    clock = [None]
    monkeypatch.setattr(open_interest.pd.Timestamp, "now", staticmethod(lambda tz=None: clock[0]))
    for n, seen in ((2, T0 + 10 * 86_400_000),  # the backfill, days later
                    (3, stamps[2] + 7 * 60_000), (4, stamps[3] + 9 * 60_000)):  # then each one 7, then 9 minutes late
        clock[0] = pd.Timestamp(seen, unit="ms", tz="UTC")
        open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(stamps[:n]))
    kept = open_interest.snapshots("BINANCE", "BTC/USDT", root=tmp_path)
    assert list(kept["backfill"]) == [True, True, False, False]
    assert kept["first_seen"].iloc[2] - kept.index[2] == pd.Timedelta(minutes=7)
    assert pd.Timedelta(minutes=7) < open_interest.lag("BINANCE", "BTC/USDT", root=tmp_path) <= pd.Timedelta(minutes=9)


def test_a_week_without_new_snapshots_is_raised_once_a_day_before_any_are_lost(tmp_path):
    from sleeve_fund import history

    path = open_interest._path("BINANCE", "BTC/USDT", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"snapshots": [[T0, 1.0, 1.0, T0 + 3 * 60_000, 0]]}))  # received 3 minutes late
    received = pd.Timestamp(T0 + 3 * 60_000, unit="ms", tz="UTC")
    assert open_interest.at_risk("BINANCE", "BTC/USDT", root=tmp_path, now=received + pd.Timedelta(days=7)) is None
    problem = open_interest.at_risk("BINANCE", "BTC/USDT", root=tmp_path, now=received + pd.Timedelta(days=7, minutes=1))
    assert "last kept 2026-10-01 00:03 UTC" in problem and "lost for good from 2026-10-31 00:00 UTC" in problem

    class Inbox:
        events = []

        def event(self, sleeve, level, kind, message):
            self.events.append((level, kind))
    history._warned.clear()
    history._warn_at_risk(problem, store=Inbox())
    history._warn_at_risk(problem, store=Inbox())
    assert Inbox.events == [("error", "open_interest_at_risk")]


def test_as_of_hides_a_snapshot_until_its_measured_lag_has_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(open_interest, "MIN_LIVE", 1)  # the lag from a handful of live captures
    stamps = [T0 + i * STEP for i in range(3)]
    clock = [None]
    monkeypatch.setattr(open_interest.pd.Timestamp, "now", staticmethod(lambda tz=None: clock[0]))
    for n, late in ((1, 1), (2, 8), (3, 8)):
        clock[0] = pd.Timestamp(stamps[n - 1] + late * 60_000, unit="ms", tz="UTC")
        open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(stamps[:n]))
    assert open_interest.lag("BINANCE", "BTC/USDT", root=tmp_path) == pd.Timedelta(minutes=8)
    at = pd.Timestamp(stamps[2], unit="ms", tz="UTC") + pd.Timedelta(minutes=7)  # the newest is stored, not yet known
    assert open_interest.as_of("BINANCE", "BTC/USDT", at, root=tmp_path)["contracts"] == 101.0
    later = at + pd.Timedelta(minutes=1)
    assert open_interest.as_of("BINANCE", "BTC/USDT", later, root=tmp_path)["contracts"] == 102.0
    assert open_interest.as_of("BINANCE", "BTC/USDT", pd.Timestamp(T0, unit="ms", tz="UTC"), root=tmp_path) is None


def test_backfilled_snapshots_are_read_with_a_cautious_lag_until_one_is_measured(tmp_path):
    open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with([T0, T0 + STEP]))
    assert open_interest.lag("BINANCE", "BTC/USDT", root=tmp_path) is None
    at = pd.Timestamp(T0 + STEP, unit="ms", tz="UTC") + pd.Timedelta(minutes=19)
    assert open_interest.as_of("BINANCE", "BTC/USDT", at, root=tmp_path)["contracts"] == 100.0


def test_positioning_ratios_are_kept_as_their_own_unused_series(tmp_path, monkeypatch):
    from sleeve_fund.venues import BINANCE_STATS

    asked = []
    rows = [{"symbol": "BTCUSDT", "longShortRatio": "1.8", "longAccount": "0.6429", "shortAccount": "0.3571",
             "timestamp": T0}]
    out = BINANCE_STATS["long_short_top"]("BTC/USDT", 0, get_json=lambda url: asked.append(url) or rows, now_ms=T0)
    assert out == [(T0, 1.8, 0.6429, 0.3571)] and "/topLongShortPositionRatio?" in asked[0]
    profile = venue("BINANCE")
    monkeypatch.setattr(profile, "stats_loaders", {
        "open_interest": _venue_with([T0]),
        "long_short_global": lambda pair, start: [(T0, 1.2, 0.55, 0.45)] if start <= T0 else []})
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    kept = open_interest.snapshots("BINANCE", "BTC/USDT", root=tmp_path, series="long_short_global")
    assert list(kept.columns[:3]) == ["long_short_ratio", "long_share", "short_share"]
    assert kept["long_share"].iloc[0] == 0.55
    assert (tmp_path / "BINANCE" / "BTC-USDT" / "open_interest.json").exists()


def test_a_series_never_kept_is_raised_a_day_after_the_collector_started(tmp_path):
    started = pd.Timestamp("2026-10-01", tz="UTC")
    kw = dict(root=tmp_path, since=started)
    assert open_interest.at_risk("BINANCE", "BTC/USDT", now=started + pd.Timedelta(hours=23), **kw) is None
    problem = open_interest.at_risk("BINANCE", "BTC/USDT", now=started + pd.Timedelta(hours=25), **kw)
    assert "open interest last kept never" in problem and "2026-10-01 00:00" in problem
    # The once-a-day key is the instrument and series, the same as for a series that stopped.
    assert problem.split(" last kept")[0] == "BINANCE BTC/USDT: open interest"


def test_a_write_merges_what_another_writer_kept_while_this_one_fetched_and_is_flushed_to_disk(tmp_path, monkeypatch):
    """QA F3/F8 for the snapshots: the read-modify-write is under the series directory's cross-process lock and
    re-reads what is kept, so a manual refresh never overwrites the hub's newer rows (or their first_seen)."""
    from sleeve_fund import history

    times = [T0 + i * STEP for i in range(4)]
    open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(times[:2]))
    hub_first_seen = None

    def slow_loader(pair, start):  # while this refresh fetches, the hub keeps snapshot 3 first
        nonlocal hub_first_seen
        if hub_first_seen is None:
            open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(times[:3]))
            hub_first_seen = open_interest.snapshots("BINANCE", "BTC/USDT", root=tmp_path)["first_seen"].iloc[2]
        return _venue_with(times)(pair, start)

    synced = []
    real = history._durable_replace
    monkeypatch.setattr(open_interest, "_durable_replace", lambda tmp, path: synced.append(path) or real(tmp, path))
    out = open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=slow_loader)
    kept = open_interest.snapshots("BINANCE", "BTC/USDT", root=tmp_path)
    assert len(kept) == 4 and out["written"] == 1 and out["unchanged"] == 1
    assert kept["first_seen"].iloc[2] == hub_first_seen  # the hub's row stands
    assert synced and (tmp_path / "BINANCE" / "BTC-USDT" / ".write.lock").exists()
    assert not list((tmp_path / "BINANCE" / "BTC-USDT").glob("*.tmp"))


def _put(root, rows):
    path = open_interest._path("BINANCE", "BTC/USDT", root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"snapshots": rows}))


def test_the_lag_is_the_venues_not_an_outage_this_collector_caught_up_after(tmp_path):
    """QA P1-O4: snapshots fetched together on return from an outage share one first_seen; only the newest of a
    fetch measures the venue, so a two-day outage leaves the lag at the venue's three minutes."""
    rows = [[T0 + i * STEP, 1.0 + i, 1.0, T0 + i * STEP + 3 * 60_000, 0] for i in range(3000)]
    back = T0 + 2600 * STEP + 2 * 60_000
    for i in range(2000, 2576):  # 48 hours missed, fetched at once
        rows[i][3] = back
    _put(tmp_path, rows)
    assert open_interest.lag("BINANCE", "BTC/USDT", root=tmp_path) == pd.Timedelta(minutes=3)


def test_as_of_never_serves_a_live_snapshot_before_it_was_first_seen(tmp_path):
    """QA P1-O6, MAJOR per the Advisor: the p95 lag lets 5% through early; first_seen <= at holds every live one
    back. QA's case: the 12:30 snapshot reached us at 13:30, so as_of(12:32) serves the 12:25 one."""
    rows = [[T0 + i * STEP, 1.0 + i, 1.0, T0 + i * STEP + 2 * 60_000, 0] for i in range(288)]
    rows[150][3] = rows[150][0] + 60 * 60_000  # the venue was an hour late with the 12:30 one
    _put(tmp_path, rows)
    at = pd.Timestamp("2026-10-01 12:32", tz="UTC")
    assert open_interest.as_of("BINANCE", "BTC/USDT", at, root=tmp_path)["contracts"] == 150.0  # the 12:25 one
    later = pd.Timestamp("2026-10-01 13:30", tz="UTC")
    assert open_interest.as_of("BINANCE", "BTC/USDT", later, root=tmp_path)["contracts"] >= 151.0


def test_a_snapshot_that_is_not_a_finite_number_is_refused_and_recorded(tmp_path):
    loader = lambda pair, start: [(T0, float("nan"), 1e6), (T0 + STEP, 101.0, 1e6)]  # noqa: E731
    out = open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=loader)
    assert out["written"] == 1 and out["refused"] == 1
    assert list(open_interest.snapshots("BINANCE", "BTC/USDT", root=tmp_path)["contracts"]) == [101.0]
    assert [e["kind"] for e in open_interest.provenance("BINANCE", "BTC/USDT", root=tmp_path)] == ["refused"]


def test_never_kept_counts_from_the_first_try_across_restarts(tmp_path, monkeypatch):
    """QA P1-O7: when collection was first tried is kept in the series' file, so a restart doesn't reset it."""
    first = pd.Timestamp("2026-10-01", tz="UTC")
    monkeypatch.setattr(open_interest.pd.Timestamp, "now", staticmethod(lambda tz=None: first))
    open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=lambda pair, start: [])
    later = first + pd.Timedelta(days=3)
    assert "collected since 2026-10-01 00:00" in open_interest.at_risk("BINANCE", "BTC/USDT", root=tmp_path, now=later)
    assert open_interest.at_risk("BINANCE", "BTC/USDT", root=tmp_path, now=first + pd.Timedelta(hours=23)) is None
    assert "never tried" in open_interest.at_risk("BINANCE", "ETH/USDT", root=tmp_path, now=later)


def test_failing_prices_skip_neither_open_interest_nor_its_alert(tmp_path, monkeypatch):
    """QA P1-O5: the run loop refreshes funding and open interest in their own try."""
    from sleeve_fund import history

    old = T0 - 30 * 86_400_000  # received a month before the test data
    _put(tmp_path, [[old, 1.0, 1.0, old, 0]])
    raised = []

    class Stop(Exception):
        pass

    def fail(*a, **k):
        raise RuntimeError("venue 503")
    monkeypatch.setattr(history, "_pairs_in_use", lambda v, store=None: [("BTC/USDT", None)])
    monkeypatch.setattr(history, "refresh", fail)
    monkeypatch.setattr(funding, "refresh", fail)
    monkeypatch.setattr(open_interest, "refresh", fail)
    monkeypatch.setattr(history, "_warn_at_risk", lambda problem, store=None: raised.append(problem))
    monkeypatch.setattr(time, "sleep", lambda s: (_ for _ in ()).throw(Stop()))
    try:
        history.main(["--root", str(tmp_path), "run", "--venue", "binance"])
    except Stop:
        pass
    assert any(p and p.startswith("BINANCE BTC/USDT: open interest last kept") for p in raised)


def test_funding_gaps_find_two_missed_settlements_in_a_row(tmp_path):
    h = 3_600_000
    kept = [T0 + k * 8 * h for k in range(12) if k not in (3, 5)]
    funding.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=lambda pair, start: [(t, 0.0001) for t in kept if t >= start])
    assert [a for a, b in funding.gaps("BINANCE", "BTC/USDT", root=tmp_path)] == [
        pd.Timestamp(T0 + k * 8 * h, unit="ms", tz="UTC") for k in (2, 4)]


def test_the_lag_is_not_trusted_until_twenty_live_captures(tmp_path):
    """Advisor (15:07 6 Oct): keep the 20-minute fallback until there are enough live captures for a p95."""
    rows = [[T0 + i * STEP, 1.0 + i, 1.0, T0 + i * STEP + 3 * 60_000, 0] for i in range(open_interest.MIN_LIVE)]
    _put(tmp_path, rows[:-1])
    assert open_interest.lag("BINANCE", "BTC/USDT", root=tmp_path) is None
    _put(tmp_path, rows)
    assert open_interest.lag("BINANCE", "BTC/USDT", root=tmp_path) == pd.Timedelta(minutes=3)


def test_a_change_of_settlement_interval_is_reported_as_a_possible_hole_and_logged(tmp_path, capfd, monkeypatch):
    """QA P1-O9 / HoE: 8h to 16:00 then 4h from 00:00 reads the same as a missed 20:00, so it is flagged, not passed."""
    h = 3_600_000
    kept = [T0 + k * 8 * h for k in range(6)] + [T0 + 48 * h + k * 4 * h for k in range(6)]  # change at 40h -> 48h
    profile = venue("BINANCE")
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [(t, 0.0001) for t in kept if t >= start])
    monkeypatch.setattr(profile, "stats_loaders", {})
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert funding.gaps("BINANCE", "BTC/USDT", root=tmp_path) == []
    change = (pd.Timestamp(T0 + 40 * h, unit="ms", tz="UTC"), pd.Timestamp(T0 + 48 * h, unit="ms", tz="UTC"))
    assert funding.interval_changes("BINANCE", "BTC/USDT", root=tmp_path) == [change]
    assert "possible hole at interval change 2026-10-02 16:00 to 2026-10-03 00:00" in capfd.readouterr().out


def test_a_catch_up_of_many_pages_is_one_fetch_and_a_null_is_refused(tmp_path, monkeypatch):
    """Code Reviewer on P1-O4: first_seen is one time for the whole refresh, so a catch-up past 500 snapshots adds
    one row to the lag statistic, not one per page; a null value is refused like a NaN."""
    times = [T0 + i * STEP for i in range(1200)]
    open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(times[:1]))  # the first backfill
    open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=_venue_with(times))  # back after an outage
    kept = open_interest.snapshots("BINANCE", "BTC/USDT", root=tmp_path)
    assert kept["first_seen"].iloc[1:].nunique() == 1
    out = open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path,
                                loader=lambda pair, start: [(times[-1] + STEP, None, 1.0)])
    assert out["refused"] == 1 and out["written"] == 0


def test_a_steady_interval_with_millisecond_jitter_is_no_change(tmp_path):
    """Code Reviewer: the venue stamps settlements a few milliseconds late; that is not a change of interval."""
    h = 3_600_000
    kept = [T0 + k * 8 * h + (k * 7) % 5 for k in range(12)]
    funding.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=lambda pair, start: [(t, 0.0001) for t in kept if t >= start])
    assert funding.interval_changes("BINANCE", "BTC/USDT", root=tmp_path) == []
    assert funding.gaps("BINANCE", "BTC/USDT", root=tmp_path) == []


def test_a_null_from_the_venue_loses_only_its_own_row_not_the_page(tmp_path):
    """QA P1-O13: the loader passes values through, and refresh refuses a null row by row."""
    rows = [{"timestamp": T0 + i * STEP, "sumOpenInterest": None if i == 1 else str(100.0 + i),
             "sumOpenInterestValue": "6000000.0"} for i in range(3)]
    now = T0 + 86_400_000
    loader = lambda pair, start: binance_open_interest(pair, start, get_json=lambda url: rows if start <= T0 else [], now_ms=now)  # noqa: E731
    out = open_interest.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=loader)
    assert out["written"] == 2 and out["refused"] == 1
    assert list(open_interest.snapshots("BINANCE", "BTC/USDT", root=tmp_path)["contracts"]) == [100.0, 102.0]


def test_funding_gaps_and_possible_holes_reach_the_alerts_inbox_once_each_and_the_history_chip(tmp_path, monkeypatch):
    """QA P1-O11, P1-O15, P1-O16: each funding hole reaches the alerts inbox once, a second new hole the same day
    included, and the instrument's history chip says so in its text and colour, not only the log or hover."""
    from sleeve_fund import history
    from sleeve_fund.dashboard import development as dev

    h = 3_600_000
    kept = [T0 + k * 8 * h for k in range(6) if k != 3] + [T0 + 48 * h + k * 4 * h for k in range(6)]
    profile = venue("BINANCE")
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [(t, 0.0001) for t in kept if t >= start])
    monkeypatch.setattr(profile, "stats_loaders", {})
    sent = []

    class Inbox:
        def event(self, sleeve, level, kind, message):
            sent.append((level, kind, message))
    monkeypatch.setattr("sleeve_fund.store.Store", Inbox)
    monkeypatch.setattr("sleeve_fund.funding.stale", lambda *a, **k: None)  # fixed stamps, real clock
    monkeypatch.setattr(history, "_stale", set())  # no staleness episode left open by another test
    history._warned.clear()
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [(level, kind) for level, kind, _ in sent] == [("warning", "funding_gap")]
    assert "missed between 2026-10-01 16:00" in sent[0][2] and "possible hole at interval change" in sent[0][2]
    history._warned.clear()  # as after a restart: a pass that finds nothing new raises nothing (Code Reviewer)
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert len(sent) == 1
    # A second hole within the day still goes in, alone (QA P1-O16).
    kept += [T0 + 72 * h + k * 4 * h for k in range(6) if k != 2]
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert len(sent) == 2 and sent[1][2].count("missed between") == 1 and "interval change" not in sent[1][2]
    assert "2026-10-04 04:00" in sent[1][2]

    from sleeve_fund.dashboard.app import _funding_health
    health = _funding_health("BINANCE", "BTC/USDT", tmp_path, None)
    assert health == {"funding_gaps": 2, "funding_maybe": 1}
    first, last = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")
    chip = dev.history_chip({"first": first, "last": last, "state": "current", "gaps": [], **health})
    assert chip["title"] == "2 missed funding settlements; 1 possible funding hole at an interval change"
    assert chip["state"] == "stored" and chip["tone"] == "paused"
    assert chip["text"].endswith("no price gaps · 2 funding holes · 1 possible funding hole")  # QA P1-O19
    clean = dev.history_chip({"first": first, "last": last, "state": "current", "gaps": []})
    assert clean["tone"] == "running" and clean["text"].endswith("no gaps")


def test_a_null_funding_rate_is_refused_alone_and_left_as_a_hole(tmp_path):
    """QA P1-O13 for funding (HoE): a null rate loses only its settlement, which gaps() then reports."""
    from sleeve_fund.venues import binance_funding

    h = 3_600_000
    rows = [{"symbol": "BTCUSDT", "fundingTime": T0 + k * 8 * h, "fundingRate": None if k == 2 else "0.0001"}
            for k in range(6)]
    loader = lambda pair, start: binance_funding(pair, start, get_json=lambda url: [r for r in rows if r["fundingTime"] >= start])  # noqa: E731
    kept = funding.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=loader)
    assert len(kept) == 5 and not kept.isna().any()
    assert funding.gaps("BINANCE", "BTC/USDT", root=tmp_path) == [
        (pd.Timestamp(T0 + 8 * h, unit="ms", tz="UTC"), pd.Timestamp(T0 + 24 * h, unit="ms", tz="UTC"))]
    assert len(funding.fetch("BINANCE", "BTC/USDT", pd.Timestamp(T0, unit="ms", tz="UTC"), loader=loader)) == 5


def _inbox(monkeypatch, check=None):
    sent = []

    class Inbox:
        def event(self, sleeve, level, kind, message, ts=None):
            if check is not None:
                check()
            sent.append((level, kind, message))
    monkeypatch.setattr("sleeve_fund.store.Store", Inbox)
    return sent


def test_a_change_to_a_longer_interval_fed_one_settlement_a_pass_is_one_possible_hole(tmp_path, monkeypatch):
    """QA P1-O18: as the hub gets them, 4-hourly settlements then 8-hourly: a span is judged only once a later
    settlement is kept, so the change is one possible hole and nothing is called missed."""
    from sleeve_fund import history

    h = 3_600_000
    every = [T0 + k * 4 * h for k in range(8)] + [T0 + 28 * h + k * 8 * h for k in range(1, 8)]
    kept: list = []
    profile = venue("BINANCE")
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [(t, 0.0001) for t in kept if t >= start])
    monkeypatch.setattr(profile, "stats_loaders", {})
    sent = _inbox(monkeypatch)
    for t in every:
        kept.append(t)
        _refresh_funding(profile, "BTC/USDT", tmp_path, None)
        history._warned.clear()
    assert len(sent) == 1 and "possible hole at interval change" in sent[0][2] and "missed" not in sent[0][2], sent


def _one_hole(tmp_path, monkeypatch):
    h = 3_600_000
    kept = [T0 + k * 8 * h for k in range(6) if k != 3]
    profile = venue("BINANCE")
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [(t, 0.0001) for t in kept if t >= start])
    monkeypatch.setattr(profile, "stats_loaders", {})
    monkeypatch.setattr("sleeve_fund.funding.stale", lambda *a, **k: None)  # fixed stamps, real clock: holes only
    monkeypatch.setattr("sleeve_fund.history._stale", set())
    return profile, funding._path("BINANCE", "BTC/USDT", tmp_path).with_name("funding.alerted.json")


@pytest.mark.parametrize("content", ["null", "5", "[[1, 2]]"])
def test_an_alerted_file_of_the_wrong_shape_is_read_as_empty_and_said(tmp_path, monkeypatch, capfd, content):
    """QA P1-O20: the hole goes in once more and the file is rewritten, rather than the alerts stopping for good."""
    from sleeve_fund import history

    profile, path = _one_hole(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)
    path.write_text(content)
    history._warned.clear()
    sent = _inbox(monkeypatch)
    for _ in range(3):
        _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    out = capfd.readouterr().out
    assert [k for _, k, _ in sent] == ["funding_alerted_unreadable", "funding_gap"], sent  # an incident, then the hole
    assert "reading it as empty" in out and "funding refresh failed" not in out
    assert json.loads(path.read_text()) == ["missed between 2026-10-01 16:00 to 2026-10-02 08:00"]


def test_an_alerted_file_that_cannot_be_saved_is_not_sent_again_and_leaves_no_temp_file(tmp_path, monkeypatch, capfd):
    """QA P1-O20: the event went in; a failed save neither repeats it every pass nor leaves a .tmp behind."""
    from sleeve_fund import history

    profile, path = _one_hole(tmp_path, monkeypatch)
    monkeypatch.setattr(history, "_sent_holes", {})

    def refuse(tmp, dest):
        raise PermissionError("read-only")
    monkeypatch.setattr(history, "_durable_replace", refuse)
    sent = _inbox(monkeypatch)
    for _ in range(4):
        _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert len(sent) == 1 and "could not keep the funding holes already raised" in capfd.readouterr().out
    assert not list(path.parent.glob("*.tmp"))


def test_the_alert_is_sent_outside_the_stores_lock(tmp_path, monkeypatch):
    """QA P1-O21: a slow or unreachable database never holds up the store's other writes."""
    from sleeve_fund import history

    profile, _ = _one_hole(tmp_path, monkeypatch)
    held = []
    sent = _inbox(monkeypatch, check=lambda: held.append(history._lock.locked()))
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    path = funding._path("BINANCE", "BTC/USDT", tmp_path).with_name("funding.alerted.json")
    path.write_text("{not json")
    history._warned.clear()
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)  # the incident and the hole sent again, outside it too
    assert len(sent) == 3 and held == [False, False, False]


def test_two_refreshes_racing_send_a_new_hole_once(tmp_path, monkeypatch):
    """Code Reviewer on #163: a manual refresh while the hub runs; both see the same new hole, one event results."""
    import threading

    from sleeve_fund import history

    sent = _inbox(monkeypatch, check=lambda: time.sleep(0.2))  # a slow database widens the race
    go = threading.Barrier(2)

    def refresh():
        go.wait()
        history._alert_funding_holes("BINANCE", "BTC/USDT", tmp_path, ["missed between 2026-10-01 16:00 to 2026-10-02 08:00"])
    threads = [threading.Thread(target=refresh) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [k for _, k, _ in sent] == ["funding_gap"]


def test_an_unreadable_alerted_file_is_rewritten_so_its_incident_is_raised_once(tmp_path, monkeypatch):
    """Code Reviewer on #163: with nothing new to send, a corrupt file is still rewritten clean."""
    from sleeve_fund import history

    profile, path = _one_hole(tmp_path, monkeypatch)
    sent = _inbox(monkeypatch)
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    kept = json.loads(path.read_text())
    path.write_text(json.dumps(kept)[:-3])  # truncated
    for _ in range(3):
        history._warned.clear()  # a day apart
        _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [k for _, k, _ in sent] == ["funding_gap", "funding_alerted_unreadable", "funding_gap"]
    assert json.loads(path.read_text()) == kept
