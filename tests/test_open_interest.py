"""DA-10: open interest snapshots and funding rates are kept append-only, survive restarts, and report holes."""

import json

import pandas as pd

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
    log = [json.loads(line) for line in (tmp_path / "BINANCE" / "BTC-USDT" / "provenance.jsonl").read_text().splitlines()]
    assert {e["series"] for e in log} == {"open_interest"} and log[-1]["stored"][0] == 104.0


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
    monkeypatch.setattr(profile, "open_interest_loader", _venue_with(times))
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    line = capfd.readouterr().out.strip().splitlines()[-1]
    assert line == "BINANCE BTC/USDT: open interest to 2026-10-01 00:10 UTC (+3), funding to 2026-10-01 00:00 UTC"


def test_a_funding_failure_does_not_stop_open_interest(tmp_path, capfd, monkeypatch):
    profile = venue("BINANCE")

    def broken(pair, start):
        raise OSError("venue down")
    monkeypatch.setattr(profile, "funding_loader", broken)
    monkeypatch.setattr(profile, "open_interest_loader", _venue_with([T0]))
    _refresh_funding(profile, "BTC/USDT", tmp_path, None)
    out = capfd.readouterr().out
    assert "funding refresh failed" in out and "open interest to 2026-10-01 00:00 UTC (+1), funding to none yet" in out
