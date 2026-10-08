"""Hub bar parity (sleeve_fund.parity): the store's bars against the venue's candles, minute by minute."""

from types import SimpleNamespace

import pandas as pd

from sleeve_fund.history import HistoryStore
from sleeve_fund.parity import canon_sample, compare, markdown, markdown_sample, run, venue_minutes, window

V, P = "BINANCE", "BTC/USDT"
START = pd.Timestamp("2026-10-05 12:00", tz="UTC")


def _frame(n, price=100.0, start=START):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    c = price + pd.Series(range(n), index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 2.0}, index=idx)


def _rows(df):
    return [(ts.value, *row) for ts, row in zip(df.index, df.to_numpy())]


def _loader(df, page=4):
    """A venue loader paging `df` by open-time cursor in ms, like binance_minutes."""
    def load(pair, cursor):
        lo = pd.Timestamp(int(cursor), unit="ms", tz="UTC")
        chunk = df[df.index >= lo].iloc[:page]
        nxt = str(int(chunk.index[-1].timestamp() * 1000)) if len(chunk) else cursor
        return chunk, nxt, len(chunk) < page
    return load


def _profile(df):
    return SimpleNamespace(name=V, label="Binance", minute_loader=_loader(df),
                           minute_cursor_at=lambda ts: str(int(ts.timestamp() * 1000)))


def test_identical_bars_match(tmp_path):
    venue = _frame(10)
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(venue), "live")
    (p,) = run(store, _profile(venue), [P], START, START + pd.Timedelta("10min"))
    assert p.ok and p.both == 10 and p.worst_price == 0.0
    assert "| BTC/USDT | match | 10 |" in markdown("Binance", [p])


def test_missing_minutes_price_and_volume_differences_and_refills_are_reported(tmp_path):
    venue = _frame(10)
    ours = venue.drop(venue.index[[3, 4]]).copy()
    ours.loc[venue.index[6], "close"] += 0.25
    ours.loc[venue.index[7], "volume"] = 2.5
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(ours.iloc[:3]), "live")
    store.append_bars(V, P, _rows(ours.iloc[3:]), "live")
    store.append_bars(V, P, _rows(venue.iloc[[0]] * 1.01), "live")  # disagrees with a stored bar: kept (P1-1-CANON)
    (p,) = run(store, _profile(venue), [P], START, START + pd.Timedelta("10min"), late={P: (3, 1000)})
    assert not p.ok
    assert p.venue_only == list(venue.index[[3, 4]]) and p.store_only == []
    assert (p.price_diffs, p.volume_diffs, p.worst_price, p.conflicts) == (1, 1, 0.25, 1)
    md = markdown("Binance", [p])
    assert "DIFFERS" in md and "05 Oct 12:03 to 12:04" in md and "3 of 1000 (0.300%)" in md


def test_refilled_minutes_are_counted(tmp_path):
    venue = _frame(6)
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(venue.iloc[:2]), "live")
    store.append_bars(V, P, _rows(venue.iloc[4:]), "live")
    store.append_bars(V, P, _rows(venue.iloc[2:4]), "refill")
    (p,) = run(store, _profile(venue), [P], START, START + pd.Timedelta("6min"))
    assert p.ok and p.refilled == 2


def test_venue_pages_are_joined_and_cut_to_the_window():
    venue = _frame(20)
    got = venue_minutes(_loader(venue, page=3), P, START + pd.Timedelta("2min"), START + pd.Timedelta("9min"),
                        lambda ts: str(int(ts.timestamp() * 1000)))
    assert list(got.index) == list(venue.index[2:9])


def test_quiet_minutes_with_no_volume_on_both_sides_match():
    a = _frame(3)
    a["volume"] = 0.0
    p = compare(P, a, a.copy(), START, START + pd.Timedelta("3min"))
    assert p.ok and p.worst_volume == 0.0


def test_the_window_leaves_out_the_forming_minute():
    start, end = window(24, now=pd.Timestamp("2026-10-05 12:00:42", tz="UTC"))
    assert end == pd.Timestamp("2026-10-05 12:00", tz="UTC") and end - start == pd.Timedelta("24h")


def test_the_hubs_late_trade_counts_are_read_from_its_own_file_by_default(tmp_path):
    import json

    from sleeve_fund.history import HistoryStore
    from sleeve_fund.parity import late_counts

    store = HistoryStore(tmp_path)
    assert late_counts(store, "BINANCE") is None
    (tmp_path / "hub-late-BINANCE.json").write_text(json.dumps({"BTC/USDT": [3, 1200]}))
    assert late_counts(store, "BINANCE") == {"BTC/USDT": (3, 1200)}
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"BTC/USDT": [1, 10]}))
    assert late_counts(store, "BINANCE", other) == {"BTC/USDT": (1, 10)}


def test_the_minutes_whose_price_differs_are_listed(tmp_path):
    venue = _frame(6)
    store = HistoryStore(tmp_path)
    hub = venue.copy()
    hub.iloc[3, hub.columns.get_loc("close")] += 0.01
    store.append_bars(V, P, _rows(hub), "live")
    (p,) = run(store, _profile(venue), [P], START, START + pd.Timedelta("6min"))
    assert p.price_minutes == [START + pd.Timedelta("3min")]
    assert "OHLC differs at 05 Oct 12:03." in markdown("Binance", [p])


def test_the_canon_sample_shows_both_values_and_that_the_store_now_holds_the_venues_bar(tmp_path):
    """DA 8 Oct (HoE): the spot-check of the CANON run reads the provenance log and the store, and writes nothing."""
    venue = _frame(10)
    hub = venue.copy()
    hub["volume"] = 1.5  # every live bar differs from the venue's
    store = HistoryStore(tmp_path, clock=lambda: START + pd.Timedelta("1h"))
    store.append_bars(V, P, _rows(hub), "live")
    assert store.canonise(V, P, venue).replaced == 10
    before = sorted((f.name, f.stat().st_mtime_ns) for f in tmp_path.rglob("*") if f.is_file())
    rows = canon_sample(store, V, P, 3)
    assert [r["minute"] for r in rows] == [START, START + pd.Timedelta("4min"), START + pd.Timedelta("9min")]
    assert rows[0]["stored"] == [100.0, 100.5, 99.5, 100.0, 1.5] and rows[0]["offered"][-1] == 2.0
    assert all(r["now"] == r["offered"] and r["now_is_offered"] for r in rows)
    assert sorted((f.name, f.stat().st_mtime_ns) for f in tmp_path.rglob("*") if f.is_file()) == before
    md = markdown_sample(P, rows, 10)
    assert "3 of 10 minutes canon replaced" in md and "| 100 100.5 99.5 100 1.5 | 100 100.5 99.5 100 2 |" in md
    assert canon_sample(store, V, P, 0) == [] and len(canon_sample(store, V, P, 50)) == 10


def test_a_sampled_minute_that_no_longer_holds_the_venues_bar_is_flagged(tmp_path):
    venue = _frame(4)
    hub = venue.copy()
    hub["volume"] = 1.5
    store = HistoryStore(tmp_path, clock=lambda: START + pd.Timedelta("1h"))
    store.append_bars(V, P, _rows(hub), "live")
    store.canonise(V, P, venue)
    path = tmp_path.rglob("provenance.jsonl").__next__()
    path.write_text(path.read_text().replace('"offered": [100.0, 100.5, 99.5, 100.0, 2.0]',
                                             '"offered": [100.0, 100.5, 99.5, 100.0, 9.0]'))
    (row,) = canon_sample(store, V, P, 1)
    assert not row["now_is_offered"] and "| NO |" in markdown_sample(P, [row], 4)


def test_a_minute_canon_replaced_twice_is_sampled_once_by_its_latest_record(tmp_path):
    """CR F224-1: a later run offering a newer venue bar leaves the first record's offered bar stale; not an alarm."""
    venue = _frame(3)
    hub = venue.copy()
    hub["volume"] = 1.5
    store = HistoryStore(tmp_path, clock=lambda: START + pd.Timedelta("1h"))
    store.append_bars(V, P, _rows(hub), "live")
    store.canonise(V, P, venue)
    revised = venue.copy()
    revised["volume"] = 2.5
    assert store.canonise(V, P, revised).replaced == 3
    rows = canon_sample(store, V, P, 10)
    assert len(rows) == 3 and all(r["now_is_offered"] and r["stored"][-1] == 2.0 for r in rows)
