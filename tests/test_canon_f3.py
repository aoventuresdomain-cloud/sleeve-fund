"""CANON-F3 (HoE GO conditions, 7 Oct): the backfill can be bounded at a closed minute so the dry run and the real run
cover the same window, it reports the span it covered, canon() caps where canonise() caps (F2), and one writer per
series holds across processes, the funding file included."""

import multiprocessing as mp
import threading
import time

import pandas as pd
import pytest

from sleeve_fund import funding, history
from sleeve_fund.history import HistoryStore, _writing
from test_p1_1_canon import P, T0, V, _closes, _profile, _rows, _venue

MIN = pd.Timedelta(minutes=1)


def test_until_leaves_every_minute_from_it_alone(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(T0, 8), "live")  # closed to 12:07
    venue = _venue(12)
    out = history.canon(store, _profile(venue), P, T0, until=T0 + 4 * MIN)
    assert out["replaced"] == 4  # 12:00-12:03
    assert _closes(store) == venue["close"].iloc[:4].tolist() + [104.0, 105.0, 106.0, 107.0]
    assert (out["first"], out["last"]) == (T0.isoformat(), (T0 + 3 * MIN).isoformat())
    assert (out["offered"], out["stored"], out["until"]) == (4, 4, (T0 + 4 * MIN).isoformat())
    assert out["top"] == (T0 + 3 * MIN).isoformat()  # the window's newest minute, as the run took it (F212-2)


def test_a_dry_run_and_the_real_run_over_the_same_bound_count_the_same(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(T0, 8), "live")
    venue = _venue(12)
    dry = history.canon(store, _profile(venue), P, T0, until=T0 + 6 * MIN, dry_run=True)
    store.append_bars(V, P, _rows(T0 + 8 * MIN, 3), "live")  # the hub stores more minutes in between
    real = history.canon(store, _profile(venue), P, T0, until=T0 + 6 * MIN)
    assert {k: dry[k] for k in ("replaced", "written", "offered", "first", "last")} == \
        {k: real[k] for k in ("replaced", "written", "offered", "first", "last")}


def test_until_must_be_after_since(tmp_path):
    with pytest.raises(ValueError, match="must be after"):
        history.canon(HistoryStore(tmp_path), _profile(_venue(3)), P, T0, until=T0)


def test_an_until_past_the_newest_closed_minute_stored_is_refused(tmp_path):
    """F212-2: the store vouches only for minutes up to its newest closed one, so a later until would quietly cover
    less than asked, and the dry run and the real run could differ. Refused, nothing written."""
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(T0, 4), "live")  # closed to 12:03
    with pytest.raises(ValueError, match="past the newest closed minute"):
        history.canon(store, _profile(_venue(12)), P, T0, until=T0 + 5 * MIN)
    assert store.provenance(V, P) == []
    assert history.canon(store, _profile(_venue(12)), P, T0, until=T0 + 4 * MIN)["replaced"] == 4  # up to it: fine


def test_the_command_checks_every_instrument_before_it_writes_any(tmp_path, capfd):
    store = HistoryStore(tmp_path)
    store.append_bars(V, "BTC/USDT", _rows(T0, 8), "live")
    store.append_bars(V, "ETH/USDT", _rows(T0, 2), "live")  # closed to 12:01 only
    before = store.read(V, "BTC/USDT", 1)
    code = history.main(["--root", str(tmp_path), "canon", "BTC/USDT", "ETH/USDT", "--venue", "binance",
                         "--since", "2026-10-05T12:00", "--until", "2026-10-05T12:05"])
    assert code == 2 and "ETH/USDT: until" in capfd.readouterr().out
    pd.testing.assert_frame_equal(store.read(V, "BTC/USDT", 1), before)  # BTC, though fine, left untouched


def test_canon_caps_where_canonise_caps_when_no_minute_is_known_closed(tmp_path):
    """F2: with only the loader's minutes stored, its newest may be forming: canon() offers up to the one before it,
    as canonise() takes, so the span it reports is the span it changed."""
    store = HistoryStore(tmp_path)
    store.append(V, P, _venue(5, price=100.0), cursor="c")  # 12:00-12:04, 12:04 possibly forming
    assert store.coverage(V, P).closed is None
    out = history.canon(store, _profile(_venue(12)), P, T0)
    assert out["last"] == (T0 + 3 * MIN).isoformat() and out["replaced"] == 4
    assert max(r["minute"] for r in store.provenance(V, P)) == (T0 + 3 * MIN).isoformat()  # 12:04 left as it was


def _hold(d, ready, seconds):
    with _writing(d):
        ready.set()
        time.sleep(seconds)


def _held_elsewhere(d, seconds=1.0):
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=_hold, args=(d, ready, seconds))
    proc.start()
    assert ready.wait(30), "the other process never took the lock"
    return proc


def test_one_writer_per_series_across_processes(tmp_path):
    d = HistoryStore(tmp_path)._dir(V, P)
    proc = _held_elsewhere(d)
    t = time.monotonic()
    with _writing(d):
        waited = time.monotonic() - t
    proc.join(30)
    assert waited >= 0.5  # it waited for the other process's write to finish


def test_a_funding_refresh_waits_for_the_series_writer_in_another_process(tmp_path):
    path = funding._path(V, P, tmp_path)
    proc = _held_elsewhere(path.parent)
    t = time.monotonic()
    funding.refresh(V, P, root=tmp_path, since=T0, loader=lambda pair, start: [(int(T0.timestamp() * 1000), 0.0001)])
    waited = time.monotonic() - t
    proc.join(30)
    assert waited >= 0.5
    assert funding.rates(V, P, tmp_path).tolist() == [0.0001]
    assert not list(path.parent.glob("*.tmp"))  # its own temporary file, gone once renamed


def test_a_minute_write_is_not_held_up_by_a_funding_fetch(tmp_path):
    """F212-1: the venue is asked for funding outside the series lock, so the hub's minute writes don't queue behind
    a slow page (stale bars mean no new trades)."""
    store = HistoryStore(tmp_path)
    asked, release = threading.Event(), threading.Event()

    def slow(pair, start):
        asked.set()
        release.wait(30)
        return [(int(T0.timestamp() * 1000), 0.0001)]

    fetch = threading.Thread(target=funding.refresh, args=(V, P), kwargs={"root": tmp_path, "since": T0, "loader": slow})
    fetch.start()
    try:
        assert asked.wait(30)
        write = threading.Thread(target=store.append_bars, args=(V, P, _rows(T0, 2), "live"))
        write.start()
        write.join(5)
        assert not write.is_alive()  # written while the page is still awaited
    finally:
        release.set()
        fetch.join(30)
    assert funding.rates(V, P, tmp_path).tolist() == [0.0001] and len(store.read(V, P, 1)) == 2


def test_a_funding_backfill_asks_the_venue_outside_the_lock_and_keeps_what_is_new(tmp_path):
    path = funding._path(V, P, tmp_path)
    t = int(T0.timestamp() * 1000)
    h8 = 8 * 3_600_000
    funding.refresh(V, P, root=tmp_path, since=T0, loader=lambda pair, start: [(t, 0.0001), (t + 2 * h8, 0.0003)])
    d = HistoryStore(tmp_path)._dir(V, P)
    assert d == path.parent

    wrote = []

    def minute_write():
        with _writing(d):
            wrote.append(True)

    def page(pair, start):  # a minute write while the venue is asked goes straight through: the lock isn't held
        write = threading.Thread(target=minute_write, daemon=True)
        write.start()
        write.join(5)
        return [(t, 0.0001), (t + h8, 0.0002), (t + 2 * h8, 0.0003)]

    assert funding.backfill(V, P, T0, root=tmp_path, loader=page) == 1
    assert wrote == [True]
    assert funding.rates(V, P, tmp_path).tolist() == [0.0001, 0.0002, 0.0003]
