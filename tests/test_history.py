"""The price-history store: whole minutes, quiet minutes kept, bars stamped at their close."""

import numpy as np
import pandas as pd
import pytest

from sleeve_fund import venues
from sleeve_fund.history import HistoryStore, _load, _save, refresh, trades_to_minutes


def _minutes(start, n, price=100.0):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    c = price + np.arange(n, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1.0}, index=idx)


def test_daily_bars_are_complete_and_stamped_at_the_close(tmp_path):
    store = HistoryStore(tmp_path)
    store.append("X", "ABC/USD", _minutes("2026-01-01", 3 * 1440 + 10), cursor="c1")
    days = store.read("X", "ABC/USD", 1440)
    # Three complete days, each stamped when it closed; the 10 minutes of 4 Jan are not a day.
    assert list(days.index) == [pd.Timestamp(d, tz="UTC") for d in ("2026-01-02", "2026-01-03", "2026-01-04")]
    first = days.iloc[0]
    assert first["open"] == 100.0 and first["close"] == 100.0 + 1439 and first["volume"] == 1440.0
    hours = store.read("X", "ABC/USD", 60)
    assert hours.index[0] == pd.Timestamp("2026-01-01 01:00", tz="UTC")  # 00:00-01:00 is known at 01:00


def test_bars_read_month_by_month_match_one_resample_of_all_minutes(tmp_path):
    store = HistoryStore(tmp_path)
    store.append("X", "ABC/USD", _minutes("2026-01-30 07:00", 4 * 1440), cursor="c")  # spans January and February
    one = store.read("X", "ABC/USD", 1)
    for minutes in (5, 60, 1440):
        opened = one.set_axis(one.index - pd.Timedelta("1min"))
        g = opened.resample(f"{minutes}min", origin="epoch", label="left", closed="left")
        whole = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        whole = whole[g["close"].count() == minutes]
        whole = whole.set_axis(whole.index + pd.Timedelta(minutes=minutes))
        pd.testing.assert_frame_equal(store.read("X", "ABC/USD", minutes), whole, check_names=False, check_freq=False)


def test_a_bar_missing_a_minute_inside_the_series_is_kept_and_only_part_bars_at_the_ends_are_dropped(tmp_path):
    store = HistoryStore(tmp_path)
    bars = _minutes("2026-01-30 07:00", 4 * 1440)  # spans two months; part-days at both ends
    gone = [pd.Timestamp("2026-01-31 12:34", tz="UTC"), pd.Timestamp("2026-02-01 00:00", tz="UTC")]
    store.append("X", "ABC/USD", bars, cursor="c")
    for month in ("2026-01", "2026-02"):  # minutes lost to a write that never finished
        path = store._dir("X", "ABC/USD") / f"{month}.npz"
        df = _load(path)
        _save(path, df[~df.index.isin(gone)])
    # 31 Jan, 1 and 2 Feb are whole days; 30 Jan from 07:00 and 3 Feb to 06:58 (06:59 may still be forming) are
    # not, so the part bars at both ends go: 68 + 288 + 27 quarter-hours, 4 + 18 + 1 four-hour bars, 3 days.
    for minutes, n in ((15, 383), (240, 23), (1440, 3)):
        assert len(store.read("X", "ABC/USD", minutes)) == n, minutes
    days = store.read("X", "ABC/USD", 1440)
    jan31, feb1 = days.loc[pd.Timestamp("2026-02-01", tz="UTC")], days.loc[pd.Timestamp("2026-02-02", tz="UTC")]
    # Each a minute short and built from the 1,439 it has; 1 Feb opens at its second minute.
    assert jan31["volume"] == feb1["volume"] == 1439.0
    assert jan31["open"] == bars.loc["2026-01-31 00:00", "open"] and jan31["close"] == bars.loc["2026-01-31 23:59", "close"]
    assert feb1["open"] == bars.loc["2026-02-01 00:01", "open"]


def test_appends_continue_across_months_and_fill_quiet_minutes(tmp_path):
    store = HistoryStore(tmp_path)
    store.append("X", "ABC/USD", _minutes("2026-01-31 23:50", 5), cursor="a")
    later = _minutes("2026-02-01 00:10", 5, price=200.0)  # nothing traded 23:55 to 00:09
    cov = store.append("X", "ABC/USD", later, cursor="b")
    assert cov.cursor == "b" and cov.first == pd.Timestamp("2026-01-31 23:50", tz="UTC")
    rep = store.report("X", "ABC/USD")
    assert rep["missing"] == 0 and rep["duplicates"] == 0 and rep["minutes"] == 25
    one = store.read("X", "ABC/USD", 1)
    quiet = one.loc["2026-02-01 00:00":"2026-02-01 00:09"]  # stamped at close
    assert (quiet["volume"] == 0).all() and (quiet["close"] == 104.0).all()
    assert sorted(p.name for p in (tmp_path / "X" / "ABC-USD").glob("*.npz")) == ["2026-01.npz", "2026-02.npz"]


def test_merge_combines_a_minute_split_across_pages(tmp_path):
    store = HistoryStore(tmp_path)
    t = pd.Timestamp("2026-03-01 12:00", tz="UTC")
    page1 = trades_to_minutes(pd.DataFrame({"price": [10.0, 12.0], "volume": [1.0, 2.0]},
                                           index=[t, t + pd.Timedelta("20s")]))
    page2 = trades_to_minutes(pd.DataFrame({"price": [9.0, 11.0, 11.5], "volume": [1.0, 1.0, 1.0]},
                                           index=[t + pd.Timedelta("40s"), t + pd.Timedelta("50s"),
                                                  t + pd.Timedelta("70s")]))
    store.append("X", "ABC/USD", page1, cursor="1", merge=True)
    store.append("X", "ABC/USD", page2, cursor="2", merge=True)
    bar = store.read("X", "ABC/USD", 1).iloc[0]
    assert (bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]) == (10.0, 12.0, 9.0, 11.0, 5.0)


def test_rejects_bars_that_would_corrupt_a_backtest(tmp_path):
    store = HistoryStore(tmp_path)
    bad = _minutes("2026-01-01", 3)
    bad.iloc[1, bad.columns.get_loc("high")] = 1.0
    with pytest.raises(ValueError, match="impossible"):
        store.append("X", "ABC/USD", bad, cursor="")
    naive = _minutes("2026-01-01", 3).tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        store.append("X", "ABC/USD", naive, cursor="")
    with pytest.raises(KeyError):
        store.read("X", "NONE/USD")


def test_kraken_refresh_pages_through_trade_history(tmp_path):
    t0 = pd.Timestamp("2026-04-01", tz="UTC").timestamp()
    trades = [[str(100 + i % 7), "0.5", t0 + i * 7.0, "b", "m", "", i] for i in range(2500)]
    calls = []

    def get_json(url):
        if "AssetPairs" in url:
            return {"result": {"XXBTZUSD": {"wsname": "XBT/USD"}}}
        since = float(url.split("since=")[1].split("&")[0])
        page = [r for r in trades if r[2] * 1e9 > since][:1000]
        calls.append(since)
        return {"error": [], "result": {"XXBTZUSD": page, "last": str(int(page[-1][2] * 1e9)) if page else str(int(since))}}

    venues._KRAKEN_KEYS.clear()
    profile = venues.VenueProfile(**{**venues.KRAKEN.__dict__,
                                     "minute_loader": lambda pair, cur: venues.kraken_minutes(pair, cur, get_json)})
    store = HistoryStore(tmp_path)
    out = refresh(store, profile, "BTC/USD", sleep=lambda s: None, log=lambda m: None)
    assert out["pages"] == 3 and calls[0] == 0
    rep = store.report("KRAKEN", "BTC/USD")
    assert rep["missing"] == 0 and rep["duplicates"] == 0
    total = store.read("KRAKEN", "BTC/USD", 1)["volume"].sum()
    last_minute = pd.Timestamp(t0 + 2499 * 7.0, unit="s", tz="UTC").floor("1min")
    in_last = sum(1 for r in trades if pd.Timestamp(r[2], unit="s", tz="UTC").floor("1min") == last_minute)
    assert total == pytest.approx(0.5 * (2500 - in_last))  # every trade once; the forming minute held back
    again = refresh(store, profile, "BTC/USD", sleep=lambda s: None, log=lambda m: None)
    assert again["pages"] == 1 and store.read("KRAKEN", "BTC/USD", 1)["volume"].sum() == pytest.approx(total)


def test_backtest_page_reads_the_store_once_it_has_caught_up(tmp_path, monkeypatch):
    from sleeve_fund import history
    from sleeve_fund.dashboard import preview

    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path)
    store = HistoryStore(tmp_path)
    now = pd.Timestamp.now(tz="UTC").floor("1min")
    store.append("KRAKEN", "ABC/USD", _minutes(now - pd.Timedelta(days=100), 100 * 1440), cursor="x")
    monkeypatch.setattr(venues.KRAKEN, "daily_history", lambda pair: pytest.fail("should read the store"))
    preview._history.clear()
    assert len(preview.history("ABC/USD")) >= 98

    stale = HistoryStore(tmp_path)
    stale.append("KRAKEN", "OLD/USD", _minutes("2017-01-01", 100 * 1440), cursor="x")
    monkeypatch.setattr(venues.KRAKEN, "daily_history", lambda pair: "recent candles")
    assert preview.history("OLD/USD") == "recent candles"  # still backfilling: use the venue's recent candles


def test_long_stretches_without_trades_are_reported_not_hidden(tmp_path):
    from sleeve_fund.history import quiet_runs

    store = HistoryStore(tmp_path)
    before, after = _minutes("2026-01-01", 600), _minutes("2026-01-01 13:00", 600, price=800.0)
    store.append("X", "ABC/USD", before, cursor="c1")
    store.append("X", "ABC/USD", after, cursor="c2")  # 10:00 to 13:00 had no trades: a 3-hour hole
    rep = store.report("X", "ABC/USD")
    assert rep["missing"] == 0  # every minute is stored...
    q = rep["quiet_over_an_hour"]  # ...but the quiet stretch is reported
    assert q["count"] == 1 and q["longest_minutes"] == 180 and q["longest_end"] == pd.Timestamp("2026-01-01 12:59", tz="UTC")
    hours = store.read("X", "ABC/USD", 60)
    assert quiet_runs(hours, 60)["longest_minutes"] == 180
    assert quiet_runs(hours, 60, at_least=240)["count"] == 0


def test_the_collector_backfills_what_research_asked_for_from_its_start(tmp_path):
    """Review round 8, R8-M6: an instrument asked for on the Research page is collected like a
    strategy's, starting where the request says rather than at the instrument's listing."""
    from dataclasses import replace

    from sleeve_fund.history import _pairs_in_use, refresh
    from sleeve_fund.store import Store
    from sleeve_fund.venues import KRAKEN

    store = Store(f"sqlite:///{tmp_path / 'j.db'}")
    since = pd.Timestamp("2021-10-04", tz="UTC")
    assert store.request_history("KRAKEN", "ADA/EUR", since.to_pydatetime())
    assert not store.request_history("kraken", "ADA/EUR", since.to_pydatetime())  # asked once
    pairs = dict(_pairs_in_use("KRAKEN", store))
    assert all(pairs[p] is None for p in KRAKEN.core_pairs) and pairs["ADA/EUR"] == since
    assert "ADA/EUR" not in dict(_pairs_in_use("OTHER", store))

    asked = []

    def loader(pair, cursor):
        asked.append(cursor)
        idx = pd.date_range(since, periods=3, freq="1min", tz="UTC")
        return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx), "c2", True

    hist = HistoryStore(tmp_path / "h")
    refresh(hist, replace(KRAKEN, minute_loader=loader), "ADA/EUR", since=since)
    assert asked == [str(int(since.timestamp()))]  # Kraken's `since` in seconds
    refresh(hist, replace(KRAKEN, minute_loader=loader), "ADA/EUR", since=since)
    assert asked[-1] == "c2"  # then it resumes from its own cursor
