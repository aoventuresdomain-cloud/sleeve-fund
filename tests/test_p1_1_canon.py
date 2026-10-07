"""P1-1-CANON (board 7 Oct 16:44 UK): the store keeps the venue's own 1-minute candle as the record of each minute.
The hub's live bar is provisional: a settled venue candle replaces it, each replacement is recorded with both values,
a live bar never replaces anything, and every order's journalled signal keeps the bar its decision used."""

from types import SimpleNamespace

import pandas as pd
import pytest

from sleeve_fund import history
from sleeve_fund.history import CANON_SETTLE, HistoryStore
from sleeve_fund.parity import run

V, P = "BINANCE", "BTC/USDT"
T0 = pd.Timestamp("2026-10-05 12:00", tz="UTC")


def _rows(start, n, price=100.0):
    t0 = (pd.Timestamp(start) if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")).value
    return [(t0 + i * 60_000_000_000, price + i, price + i + 0.5, price + i - 0.5, price + i, 1.0) for i in range(n)]


def _venue(n, price=500.0, start=T0):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    c = price + pd.Series(range(n), index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 3.0}, index=idx)


def _loader(df, page=4):
    def load(pair, cursor):
        lo = pd.Timestamp(int(cursor), unit="ms", tz="UTC")
        chunk = df[df.index >= lo].iloc[:page]
        nxt = str(int(chunk.index[-1].timestamp() * 1000)) if len(chunk) else cursor
        return chunk, nxt, len(chunk) < page
    return load


def _profile(df, merge=False):
    return SimpleNamespace(name=V, label="Binance", minute_loader=_loader(df), merge_minutes=merge,
                           minute_cursor_at=lambda ts: str(int(ts.timestamp() * 1000)), request_interval=0)


def _closes(store):
    return store.read(V, P, 1)["close"].tolist()


def test_the_loader_pages_newest_minute_never_replaces_a_hub_bar(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(T0, 5), "live")  # closed to 12:04
    store.append(V, P, _venue(3), cursor="c")  # 12:00-12:02; 12:02 is the page's newest: may be forming
    assert _closes(store) == [500.0, 501.0, 102.0, 103.0, 104.0]
    kinds = [r["kind"] for r in store.provenance(V, P)]
    assert kinds == ["replaced", "replaced", "conflict"]
    store.append(V, P, _venue(5), cursor="d")  # the next page offers 12:02 again, closed now
    assert _closes(store)[:4] == [500.0, 501.0, 502.0, 503.0]


def test_a_minute_closed_less_than_the_settle_time_ago_is_not_replaced(tmp_path):
    now = T0 + pd.Timedelta(minutes=2) + CANON_SETTLE - pd.Timedelta(seconds=1)  # 12:01 closed 9 s ago
    store = HistoryStore(tmp_path, clock=lambda: now)
    store.append_bars(V, P, _rows(T0, 2), "live")
    res = store.append_bars(V, P, _rows(T0, 2, price=300.0), "refill")
    assert (res.replaced, res.conflicts) == (1, 1)
    assert _closes(store) == [300.0, 101.0]


def test_a_live_bar_never_replaces_a_venue_candle(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(T0, 3), "live")
    store.append_bars(V, P, _rows(T0, 3, price=500.0), "refill")
    res = store.append_bars(V, P, _rows(T0, 3), "live")
    assert (res.written, res.replaced, res.conflicts) == (0, 0, 3)
    assert _closes(store) == [500.0, 501.0, 502.0]


def test_a_trade_built_loader_keeps_the_hub_bars(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(T0, 5), "live")
    store.append(V, P, _venue(4), cursor="c", merge=True)
    assert _closes(store) == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert {r["kind"] for r in store.provenance(V, P)} == {"conflict"}


def test_the_backfill_makes_the_stored_minutes_the_venues_and_leaves_the_cursor(tmp_path):
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(T0, 3), "live")
    store.append_bars(V, P, _rows(T0 + pd.Timedelta(minutes=5), 3, price=105.0), "live")  # 12:03-12:04 missed
    before = store.coverage(V, P)
    venue = _venue(12)  # the venue has minutes past the hub's newest closed one too
    out = history.canon(store, _profile(venue), P, T0)
    assert out["replaced"] == 6 and out["written"] == 2
    assert _closes(store) == venue["close"].iloc[:8].tolist()  # nothing past 12:07 (the hub's newest closed)
    assert store.coverage(V, P) == before and store.gaps(V, P) == []
    # a second run finds nothing to change
    again = history.canon(store, _profile(venue), P, T0)
    assert (again["replaced"], again["written"]) == (0, 0)


def test_the_backfill_refuses_a_trade_built_venue(tmp_path):
    with pytest.raises(ValueError, match="from trades"):
        history.canon(HistoryStore(tmp_path), _profile(_venue(3), merge=True), P, T0)


def test_parity_counts_hub_bars_the_venue_replaced(tmp_path):
    venue = _venue(8)  # the venue is two minutes ahead: its newest (forming) candle is never used
    store = HistoryStore(tmp_path)
    store.append_bars(V, P, _rows(T0, 6), "live")
    history.canon(store, _profile(venue), P, T0)
    (p,) = run(store, _profile(venue), [P], T0, T0 + pd.Timedelta("6min"))
    assert p.ok and p.replaced == 6  # the store now matches the venue; the hub's differences are still counted


def test_every_order_journals_the_bar_its_decision_used(prices, instrument):
    from sleeve_fund.research.runner import run_backtest

    res = run_backtest("trend_filter", prices, instrument, {"fast": 10, "slow": 30})
    assert len(res.fills)
    for oid in res.fills.index:
        sig = res.decisions[oid]["signal"]
        bar = sig["bar"]
        assert bar["src"] == "store" and bar["c"] == pytest.approx(sig["close"])
        assert bar["l"] <= bar["c"] <= bar["h"] and pd.Timestamp(bar["close_ts"]).tzinfo is not None
