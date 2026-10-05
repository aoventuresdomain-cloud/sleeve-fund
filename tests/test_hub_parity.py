"""Hub bar parity (sleeve_fund.parity): the store's bars against the venue's candles, minute by minute."""

from types import SimpleNamespace

import pandas as pd

from sleeve_fund.history import HistoryStore
from sleeve_fund.parity import compare, markdown, run, venue_minutes, window

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
    store.append_bars(V, P, _rows(venue.iloc[[0]] * 1.01), "refill")  # disagrees with a stored bar
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
