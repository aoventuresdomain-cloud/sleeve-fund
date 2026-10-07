"""The hub's write path into the history store: closed bars, idempotent on (venue, instrument, minute),
the venue's own candle replaces a differing hub bar and both are recorded (P1-1-CANON), a live bar never replaces
a stored one, and holes stay holes until refilled."""

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.history import HistoryStore

V, P = "BINANCE", "BTC/USDT"


def _rows(start, n, price=100.0):
    t0 = pd.Timestamp(start, tz="UTC").value
    return [(t0 + i * 60_000_000_000, price + i, price + i + 0.5, price + i - 0.5, price + i, 1.0) for i in range(n)]


def test_live_bars_are_readable_up_to_the_last_closed_minute(tmp_path):
    store = HistoryStore(tmp_path)
    res = store.append_bars(V, P, _rows("2026-10-05 12:00", 3), "live")
    assert (res.written, res.unchanged, res.conflicts) == (3, 0, 0)
    one = store.read(V, P, 1)
    # All three minutes are closed, so the newest is read too (stamped at its close, 12:03).
    assert list(one.index) == list(pd.date_range("2026-10-05 12:01", periods=3, freq="1min", tz="UTC"))
    cov = store.coverage(V, P)
    assert cov.closed == cov.last == pd.Timestamp("2026-10-05 12:02", tz="UTC")


def test_writing_the_same_bars_again_changes_nothing(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows("2026-10-05 12:00", 5), "live")
    before = store.read(V, P, 1)
    res = store.append_bars(V, P, _rows("2026-10-05 12:00", 5), "live")
    assert (res.written, res.unchanged, res.conflicts) == (0, 5, 0)
    pd.testing.assert_frame_equal(store.read(V, P, 1), before)
    assert store.provenance(V, P) == []


def test_a_refill_replaces_a_differing_live_bar_and_records_both(tmp_path):
    """P1-1-CANON (HoE 7 Oct, test correction): a refill is the venue's own candle, the record of the minute, so it
    replaces a live bar that differs; the live bar's values stay in provenance."""
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows("2026-10-05 12:00", 3), "live")
    other = _rows("2026-10-05 12:01", 1, price=500.0)
    res = store.append_bars(V, P, other, "refill")
    assert (res.written, res.unchanged, res.conflicts, res.replaced) == (0, 0, 0, 1)
    assert store.read(V, P, 1).loc["2026-10-05 12:02", "close"] == 500.0  # the venue's candle
    (rec,) = store.provenance(V, P)
    assert rec["kind"] == "replaced" and rec["source"] == "refill" and rec["minute"].startswith("2026-10-05T12:01")
    assert rec["stored"][3] == 101.0 and rec["offered"][3] == 500.0


def test_a_hole_stays_a_hole_until_refilled(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows("2026-10-05 12:00", 2), "live")
    store.append_bars(V, P, _rows("2026-10-05 12:05", 2, price=105.0), "live")  # 12:02-12:04 missed
    hole = (pd.Timestamp("2026-10-05 12:02", tz="UTC"), pd.Timestamp("2026-10-05 12:04", tz="UTC"))
    assert store.gaps(V, P) == [hole]
    assert store.report(V, P)["missing"] == 3
    # A 15-minute bar over the hole is incomplete, so it is not served.
    assert store.read(V, P, 15).empty
    res = store.append_bars(V, P, _rows("2026-10-05 12:02", 3, price=102.0), "refill")
    assert res.written == 3 and store.gaps(V, P) == []
    (rec,) = store.provenance(V, P)
    assert rec["kind"] == "refill" and rec["minutes"] == 3


def test_a_forming_minute_from_the_rest_loader_is_replaced_by_the_closed_bar(tmp_path):
    store = HistoryStore(tmp_path)
    idx = pd.date_range("2026-10-05 11:58", periods=3, freq="1min", tz="UTC")
    store.append(V, P, pd.DataFrame({"open": 90.0, "high": 91.0, "low": 89.0, "close": 90.0, "volume": 1.0},
                                    index=idx), cursor="c")
    assert len(store.read(V, P, 1)) == 2  # 12:00 may still be forming
    res = store.append_bars(V, P, _rows("2026-10-05 12:00", 1), "live")
    assert (res.written, res.conflicts) == (1, 0)
    one = store.read(V, P, 1)
    assert len(one) == 3 and one.iloc[-1]["close"] == 100.0
    assert store.coverage(V, P).cursor == "c"  # the loader's resume point is untouched


def test_bars_cross_a_month_boundary(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows("2026-09-30 23:58", 4), "live")
    names = sorted(p.name for p in (tmp_path / V / "BTC-USDT").glob("*.npz"))
    assert names == ["2026-09.npz", "2026-10.npz"] and len(store.read(V, P, 1)) == 4


@pytest.mark.parametrize("bad, why", [
    ([(pd.Timestamp("2026-10-05 12:00:30", tz="UTC").value, 1, 1, 1, 1, 1)], "whole minutes"),
    ([(pd.Timestamp("2026-10-05 12:00", tz="UTC").value, 1, 0.5, 1, 1, 1)], "impossible"),
    ([(pd.Timestamp("2026-10-05 12:00", tz="UTC").value, 1, 1, 1, np.nan, 1)], "missing"),
    (_rows("2026-10-05 12:00", 1) * 2, "same minute twice"),
])
def test_bad_bars_are_refused_and_nothing_is_written(tmp_path, bad, why):
    store = HistoryStore(tmp_path)
    with pytest.raises(ValueError, match=why):
        store.append_bars(V, P, bad, "live")
    assert store.coverage(V, P) is None


def test_source_must_be_live_or_refill(tmp_path):
    with pytest.raises(ValueError, match="source"):
        HistoryStore(tmp_path).append_bars(V, P, _rows("2026-10-05 12:00", 1), "rest")


def test_a_stamp_a_few_ns_off_the_minute_is_refused(tmp_path):
    t = pd.Timestamp("2026-10-05 12:00", tz="UTC").value + 100  # rounds onto the minute through a float64
    with pytest.raises(ValueError, match="whole minutes"):
        HistoryStore(tmp_path).append_bars(V, P, [(t, 1.0, 1.0, 1.0, 1.0, 1.0)], "live")


def _loader_page(start, n, price):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 3.0}, index=idx)


def test_the_rest_loader_fills_holes_and_replaces_hub_bars_with_the_venues_candles(tmp_path):
    # The hub runs the REST backfill in the same process: a loader page overlapping the hub's bars. P1-1-CANON (HoE
    # 7 Oct, test correction): the venue's candles replace the hub's differing bars, recorded with both values.
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows("2026-10-05 12:00", 2), "live")
    store.append_bars(V, P, _rows("2026-10-05 12:04", 2, price=104.0), "live")  # 12:02-12:03 missed
    cov = store.append(V, P, _loader_page("2026-10-05 12:00", 7, price=500.0), cursor="next")
    one = store.read(V, P, 1)
    stamped = lambda m: pd.Timestamp(m, tz="UTC") + pd.Timedelta("1min")
    assert one.loc[stamped("2026-10-05 12:00"), "close"] == 500.0  # the venue's candle replaced the hub's bar
    assert one.loc[stamped("2026-10-05 12:02"), "close"] == 500.0  # the hole is filled from the loader
    assert store.gaps(V, P) == [] and cov.cursor == "next"
    kinds = [(r["kind"], r["source"]) for r in store.provenance(V, P)]
    assert kinds.count(("replaced", "loader")) == 4 and ("refill", "loader") in kinds
    assert ("conflict", "loader") not in kinds
    # 12:06, beyond the hub's newest closed minute, is the loader's newest and may still be forming.
    assert cov.closed == pd.Timestamp("2026-10-05 12:05", tz="UTC") and cov.last == pd.Timestamp("2026-10-05 12:06", tz="UTC")


def test_the_loader_does_not_fill_a_hub_hole_flat(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows("2026-10-05 12:00", 2), "live")
    store.append(V, P, _loader_page("2026-10-05 12:10", 2, price=110.0), cursor="c")  # loader ahead of the hub
    hole = (pd.Timestamp("2026-10-05 12:02", tz="UTC"), pd.Timestamp("2026-10-05 12:09", tz="UTC"))
    assert store.gaps(V, P) == [hole]


def _rest(start, n, price=90.0):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq="1min")
    return pd.DataFrame({"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1.0}, index=idx)


def test_rest_forming_minute_is_not_promoted_to_complete_when_the_hub_starts_one_minute_later(tmp_path):
    """QA P1-H1 (quant-review/v2-p1/hub-storage.md): the REST loader's newest minute (12:00) is a part bar. The
    hub's first stored bar is 12:01, its refill for 12:00 failed or is late. The part bar must not be served as
    complete, gaps() must show it, and the complete bar must replace it when it comes."""
    s = HistoryStore(tmp_path)
    s.append(V, P, _rest("2026-10-05 11:58", 3), cursor="c")  # 12:00 is still forming
    s.append_bars(V, P, _rows("2026-10-05 12:01", 2), "live")
    assert pd.Timestamp("2026-10-05 12:01", tz="UTC") not in s.read(V, P, 1).index  # 12:00, stamped at its close
    at = pd.Timestamp("2026-10-05 12:00", tz="UTC")
    assert (at, at) in s.gaps(V, P)
    res = s.append_bars(V, P, _rows("2026-10-05 12:00", 1, 95.0), "refill")
    assert (res.written, res.conflicts) == (1, 0)
    assert s.read(V, P, 1).loc[pd.Timestamp("2026-10-05 12:01", tz="UTC"), "close"] == 95.0
    assert s.gaps(V, P) == [] and s.coverage(V, P).forming is None


def test_the_loaders_next_page_completes_its_own_part_bar_after_the_hub_has_started(tmp_path):
    s = HistoryStore(tmp_path)
    s.append(V, P, _rest("2026-10-05 11:58", 3), cursor="c")
    s.append_bars(V, P, _rows("2026-10-05 12:01", 2), "live")
    s.append(V, P, _rest("2026-10-05 12:00", 2, 96.0), cursor="d")  # resumes on 12:00, now complete
    assert s.read(V, P, 1).loc[pd.Timestamp("2026-10-05 12:01", tz="UTC"), "close"] == 96.0
    assert s.read(V, P, 1).loc[pd.Timestamp("2026-10-05 12:02", tz="UTC"), "close"] == 100.0  # the hub's 12:01 kept
    assert s.coverage(V, P).forming is None and s.gaps(V, P) == []


def test_coverage_written_before_the_forming_minute_was_recorded_reads_as_it_did(tmp_path):
    import json

    s = HistoryStore(tmp_path)
    s.append(V, P, _rest("2026-10-05 11:58", 3), cursor="c")
    path = s._dir(V, P) / "coverage.json"
    raw = json.loads(path.read_text())
    raw.pop("forming")
    path.write_text(json.dumps(raw))
    assert s.coverage(V, P).forming == pd.Timestamp("2026-10-05 12:00", tz="UTC")


def test_a_loader_page_ahead_of_the_hub_keeps_the_venues_candles_and_records_the_hubs_later_bars(tmp_path):
    """Code Reviewer's question on #143: the loader writes past the hub's end; only its newest minute is forming.
    The hub's live bars arriving later for the minutes in between are conflicts, and the venue's candles stay."""
    s = HistoryStore(tmp_path)
    s.append_bars(V, P, _rows("2026-10-05 12:00", 6), "live")  # closed 12:05
    s.append(V, P, _rest("2026-10-05 12:04", 5, 200.0), cursor="c")  # 12:04-12:08; 12:08 forming
    res = s.append_bars(V, P, _rows("2026-10-05 12:06", 2, 106.0), "live")
    assert (res.written, res.conflicts) == (0, 2)
    assert s.read(V, P, 1).loc[pd.Timestamp("2026-10-05 12:07", tz="UTC"), "close"] == 200.0
    assert s.coverage(V, P).forming == pd.Timestamp("2026-10-05 12:08", tz="UTC")
