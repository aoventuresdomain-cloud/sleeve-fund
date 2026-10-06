"""QA's minor findings on the history store's write path (quant-review/v2-p1/hub-storage.md, F2-F8): temporary
files, a second writer process, bars not closed yet, `closed` moving back, the gap cache, repeated conflict
records, the refill record's span, and a crash between the month write and the coverage write."""

import multiprocessing as mp

import pandas as pd

from sleeve_fund import history as H
from sleeve_fund.history import HistoryStore

V, P = "BINANCE", "BTC/USDT"
NS = 60_000_000_000


def _rows(start, n, price=100.0):
    t0 = pd.Timestamp(start, tz="UTC").value
    return [(t0 + i * NS, price + i, price + i + 0.5, price + i - 0.5, price + i, 1.0) for i in range(n)]


def _rest(start, n, price=90.0):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq="1min")
    return pd.DataFrame({"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1.0}, index=idx)


def test_a_leftover_temporary_file_never_breaks_a_read(tmp_path):  # F2
    s = HistoryStore(tmp_path)
    s.append_bars(V, P, _rows("2026-10-05 12:00", 3), "live")
    d = s._dir(V, P)
    (d / "2026-09.tmp.npz").write_bytes(b"partial")  # the old temp name, left by a killed writer
    (d / ".2026-10.abc.tmp.npz").write_bytes(b"partial")  # the new one
    assert len(s.read(V, P, 1)) == 3 and s.report(V, P)["minutes"] == 3 and s.gaps(V, P) == []
    s.append_bars(V, P, _rows("2026-10-05 12:03", 1), "live")
    assert [p.name for p in d.glob(".*.tmp.npz")] == [".2026-10.abc.tmp.npz"]  # a write leaves none of its own


def _writer(root, start, n, q):
    s, errs = HistoryStore(root), 0
    for i in range(n):
        try:
            s.append_bars(V, P, _rows(str(pd.Timestamp(start) + pd.Timedelta(minutes=i)), 1), "live")
        except Exception:  # noqa: BLE001
            errs += 1
    q.put(errs)


def test_two_writer_processes_take_turns_and_lose_nothing(tmp_path):  # F3
    ctx = mp.get_context("fork")
    q, n = ctx.Queue(), 60
    ps = [ctx.Process(target=_writer, args=(str(tmp_path), t, n, q)) for t in ("2026-10-05 08:00", "2026-10-05 10:00")]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    assert q.get() + q.get() == 0
    stored = H._load(HistoryStore(tmp_path)._dir(V, P) / "2026-10.npz")
    assert len(stored) == 2 * n and not stored.index.duplicated().any()


def test_a_bar_for_a_minute_not_yet_closed_is_refused_and_recorded(tmp_path):  # F4
    now = pd.Timestamp("2026-10-05 12:03:30", tz="UTC")
    s = HistoryStore(tmp_path, clock=lambda: now)
    res = s.append_bars(V, P, _rows("2026-10-05 12:00", 5), "live")  # 12:03 is in progress, 12:04 is future
    assert res.written == 3 and s.coverage(V, P).closed == pd.Timestamp("2026-10-05 12:02", tz="UTC")
    refused = [e for e in s.provenance(V, P) if e["kind"] == "refused"]
    assert refused[0]["minutes"] == ["2026-10-05T12:03:00+00:00", "2026-10-05T12:04:00+00:00"]
    late = HistoryStore(tmp_path, clock=lambda: pd.Timestamp("2026-10-05 12:03:58.5", tz="UTC"))
    assert late.append_bars(V, P, _rows("2026-10-05 12:03", 1), "live").written == 1  # within the clock skew


def test_an_old_refill_on_a_loader_only_store_does_not_pull_closed_back(tmp_path):  # F5
    s = HistoryStore(tmp_path)
    s.append(V, P, _rest("2026-10-05 11:00", 61), cursor="c")  # 11:00-12:00, 12:00 forming
    s.append_bars(V, P, _rows("2026-10-05 11:00", 1, 90.0), "refill")
    assert s.coverage(V, P).closed == pd.Timestamp("2026-10-05 11:59", tz="UTC")


def test_the_gap_cache_sees_a_rewrite_with_the_same_size_and_time(tmp_path):  # F6
    import os

    s = HistoryStore(tmp_path)
    s.append_bars(V, P, _rows("2026-10-05 12:00", 5), "live")
    f = s._dir(V, P) / "2026-10.npz"
    assert s.gaps(V, P) == []
    st = f.stat()
    df = H._load(f).drop(pd.Timestamp("2026-10-05 12:02", tz="UTC"))
    H._save(f, pd.concat([df, df.iloc[[-1]].set_axis([pd.Timestamp("2026-10-05 12:09", tz="UTC")])]).sort_index())
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert (pd.Timestamp("2026-10-05 12:02", tz="UTC"),) * 2 in s.gaps(V, P)


def test_the_same_difference_offered_again_is_recorded_once(tmp_path):  # F7
    s = HistoryStore(tmp_path)
    s.append_bars(V, P, _rows("2026-10-05 12:00", 1), "live")
    for _ in range(3):
        assert s.append_bars(V, P, _rows("2026-10-05 12:00", 1, 99.0), "refill").conflicts == 1
    assert sum(e["kind"] == "conflict" for e in s.provenance(V, P)) == 1


def test_a_refill_record_spans_the_minutes_it_wrote(tmp_path):  # F8
    s = HistoryStore(tmp_path)
    s.append_bars(V, P, _rows("2026-10-05 12:00", 3), "live")
    s.append_bars(V, P, _rows("2026-10-05 12:00", 5), "refill")
    refill = [e for e in s.provenance(V, P) if e["kind"] == "refill"][-1]
    assert (refill["first"], refill["last"], refill["minutes"]) == (
        "2026-10-05T12:03:00+00:00", "2026-10-05T12:04:00+00:00", 2)


def test_a_crash_between_the_month_write_and_the_coverage_write_heals_on_the_restart_refill(tmp_path):
    s = HistoryStore(tmp_path)
    s.append_bars(V, P, _rows("2026-10-05 12:00", 3), "live")
    d = s._dir(V, P)
    before = (d / "coverage.json").read_text()
    s.append_bars(V, P, _rows("2026-10-05 12:03", 3), "live")
    (d / "coverage.json").write_text(before)  # the coverage write never happened
    assert s.read(V, P, 1).index[-1] == pd.Timestamp("2026-10-05 12:03", tz="UTC") and s.gaps(V, P) == []
    res = s.append_bars(V, P, _rows("2026-10-05 12:03", 3, 103.5), "refill")
    assert (res.written, res.conflicts) == (3, 0)
    assert s.coverage(V, P).closed == pd.Timestamp("2026-10-05 12:05", tz="UTC")
