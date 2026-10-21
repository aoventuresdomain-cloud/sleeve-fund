"""R-22 (HoE, 21 Oct): the venue's taker-buy volume is kept beside the stored minutes, additively: in
taker_buy/YYYY-MM.npz, so no OHLCV month file is rewritten and every candle re-reads byte for byte. Only a closed stored minute whose volume
the offer matches takes a figure; one below 0 or above its volume is refused; a kept figure never changes; a dry run
writes nothing. Expected values are written out from the fixtures, not read from the code under test."""
from types import SimpleNamespace

import pandas as pd

from sleeve_fund import history
from sleeve_fund.history import HistoryStore
from test_p1_1_canon import P, V, _rows

T0 = pd.Timestamp("2026-09-30 23:58", tz="UTC")  # six minutes across a month end: 23:58 .. 00:03
MIN = pd.Timedelta(minutes=1)


def _store(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(T0, 6), "live")  # volume 1.0 each, closed to 00:03
    return store


def _offer(taker, volume=None, start=T0):
    idx = pd.date_range(start, periods=len(taker), freq="1min", tz="UTC")
    return pd.DataFrame({"volume": volume or [1.0] * len(taker), "taker_buy": taker}, index=idx)


def _bytes(d):
    return {p.name: p.read_bytes() for p in sorted(d.glob("*.npz"))}


def test_adding_taker_volume_leaves_every_candle_file_and_read_byte_identical(tmp_path):
    store = _store(tmp_path)
    d = store._dir(V, P)
    before, read = _bytes(d), store.read(V, P, 1)
    assert sorted(before) == ["2026-09.npz", "2026-10.npz"]  # set-up: two month files
    out = store.add_taker_buy(V, P, _offer([0.25, 0.5, 0.0, 1.0, 0.75, 0.5]))
    assert out == {"added": 6, "held": 0, "not_stored": 0, "mismatched": 0, "refused": 0}
    assert _bytes(d) == before
    pd.testing.assert_frame_equal(store.read(V, P, 1), read)
    assert sorted(p.name for p in (d / "taker_buy").glob("*.npz")) == ["2026-09.npz", "2026-10.npz"]
    got = store.taker_buy(V, P)
    assert got.tolist() == [0.25, 0.5, 0.0, 1.0, 0.75, 0.5]
    assert got.index[0] == T0 and got.index[-1] == T0 + 5 * MIN


def test_only_a_stored_minute_with_the_same_volume_and_a_figure_within_it_is_kept(tmp_path):
    store = _store(tmp_path)
    offer = _offer([0.5, 2.0, -0.1, 0.5, 0.5, 0.5, 0.5], volume=[1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0])
    out = store.add_taker_buy(V, P, offer)
    # 23:58 kept; 23:59 above its volume and 00:00 below 0 refused; 00:01 another volume; 00:04 not stored
    assert out == {"added": 3, "held": 0, "not_stored": 1, "mismatched": 1, "refused": 2}
    assert store.taker_buy(V, P).index.tolist() == [T0, T0 + 4 * MIN, T0 + 5 * MIN]


def test_a_kept_figure_never_changes_and_a_dry_run_writes_nothing(tmp_path):
    store = _store(tmp_path)
    dry = store.add_taker_buy(V, P, _offer([0.5] * 6), dry_run=True)
    assert dry["added"] == 6 and not (store._dir(V, P) / "taker_buy").exists()
    store.add_taker_buy(V, P, _offer([0.5] * 3))
    again = store.add_taker_buy(V, P, _offer([0.9] * 6))
    assert (again["added"], again["held"]) == (3, 3)
    assert store.taker_buy(V, P).tolist() == [0.5, 0.5, 0.5, 0.9, 0.9, 0.9]


def test_the_backfill_pages_the_venue_and_its_dry_run_counts_what_the_run_adds(tmp_path):
    store = _store(tmp_path)
    venue = _offer([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])  # one more than stored: 00:04, the forming minute

    def load(pair, cursor, page=4):
        chunk = venue[venue.index >= pd.Timestamp(int(cursor), unit="ms", tz="UTC")].iloc[:page]
        return chunk, str(int(chunk.index[-1].timestamp() * 1000)), len(chunk) < page

    profile = SimpleNamespace(name=V, label="Binance", taker_loader=load, request_interval=0,
                              minute_cursor_at=lambda ts: str(int(ts.timestamp() * 1000)))
    dry = history.taker_backfill(store, profile, P, T0, dry_run=True)
    run = history.taker_backfill(store, profile, P, T0)
    assert dry["added"] == run["added"] == 6 and run["complete"] and run["refused"] == run["mismatched"] == 0
    assert store.taker_buy(V, P).tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert history.taker_backfill(store, profile, P, T0, dry_run=True)["added"] == 0


def test_the_venue_reads_the_klines_taker_buy_base_volume_and_its_candles_are_unchanged():
    from sleeve_fund.venues import _binance_klines, binance_taker_minutes

    # open time, o, h, l, c, volume, close time, quote volume, trades, taker-buy base, taker-buy quote, ignore
    rows = [[1_760_000_000_000 + i * 60_000, "10", "11", "9", "10.5", "4.0", 0, "40", 7, str(1.5 + i), "15", "0"]
            for i in range(3)]
    plain = _binance_klines("BTC/USDT", "1m", 0, get_json=lambda url: rows)
    assert plain.columns.tolist() == ["open", "high", "low", "close", "volume"]
    bars, nxt, caught_up = binance_taker_minutes("BTC/USDT", "0", get_json=lambda url: rows)
    assert bars["taker_buy"].tolist() == [1.5, 2.5, 3.5] and bars["volume"].tolist() == [4.0] * 3
    assert nxt == str(1_760_000_000_000 + 120_000) and caught_up
