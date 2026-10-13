"""START-GAPS (HoE 13 Oct, a start gate): a strategy doesn't start while the stored minutes its warm-up reads have
gaps, as a backtest already waits (UI v2, item 10). Its indicators would otherwise start on missing bars. A
refused start is journaled on the strategy; a gap older than the warm-up window, or a store too stale to warm up
from, holds nothing back."""

import numpy as np
import pandas as pd
import pytest
from test_dashboard import AUTH, SAME, _new, client  # noqa: F401 - the fixture
from test_history_badge import _hole

from sleeve_fund import history
from sleeve_fund.history import HistoryStore

VENUE, PAIR = "KRAKEN", "BTC/USD"  # the default venue and the instrument test_dashboard._new starts on


def _minutes(days: int, end: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(end - pd.Timedelta(days=days), end, freq="1min", tz="UTC", inclusive="left")
    c = 60_000 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.0008, len(idx))))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 5.0},
                        index=idx)


@pytest.fixture
def stored(tmp_path, monkeypatch):
    """60 days of minutes up to the last whole minute (current, so the store is what warms the strategy up), and a
    way to cut a 30-minute hole `back` before its newest minute."""
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    store = HistoryStore()
    bars = _minutes(60, pd.Timestamp.now(tz="UTC").floor("1min"))
    store.append(VENUE, PAIR, bars, cursor="x")

    def hole(back: pd.Timedelta) -> pd.Timestamp:
        start = bars.index[-1] - back
        _hole(store, VENUE, PAIR, start.strftime("%Y-%m"), start, 30)
        assert store.gaps(VENUE, PAIR)
        return start

    return store, bars, hole


def test_a_new_strategy_is_refused_while_its_warm_up_window_has_a_gap(client, stored):  # noqa: F811
    c, store = client
    _, _, hole = stored
    start = hole(pd.Timedelta(hours=3))
    r = _new(c)
    where = r.headers["location"]
    assert r.status_code == 303 and where.startswith("/sleeves/new?")
    from urllib.parse import parse_qs, urlparse

    error = parse_qs(urlparse(where).query)["error"][0]
    end = start + pd.Timedelta(minutes=29)
    assert error == (f"history gaps in warm-up window: BTC/USD {start:%d %b %H:%M}-{end:%H:%M} UTC; "
                     "it can start once they are filled")  # names the hole, by its missing minutes
    assert [s.name for s in store.sleeves()] == []  # nothing was created


def test_a_gap_older_than_the_warm_up_window_holds_nothing_back(client, stored):  # noqa: F811
    c, store = client
    _, _, hole = stored
    hole(pd.Timedelta(days=50))  # the hourly model's warm-up reads well under 50 days
    r = _new(c)
    assert r.headers["location"] == "/sleeves/btc-test"
    assert store.sleeve("btc-test").warmup_bars * 60 < 50 * 1440  # so the hole really is outside the window


def test_a_start_is_refused_and_journaled_while_the_window_has_a_gap_and_starts_once_the_hub_refills_it(client, stored):  # noqa: F811
    c, store = client
    hs, bars, hole = stored
    assert _new(c).headers["location"] == "/sleeves/btc-test"  # created on clean history
    c.post("/sleeves/btc-test/command", data={"command": "stop", "reason": "x"}, auth=AUTH, headers=SAME)
    gap = hole(pd.Timedelta(hours=3))

    def start():
        return c.post("/sleeves/btc-test/command", data={"command": "start", "reason": "go"}, auth=AUTH,
                      headers=SAME, follow_redirects=False).headers["location"]

    assert "command_error=history" in start() and store.sleeve("btc-test").desired_state == "stopped"
    (ev,) = [e for e in store.events("btc-test") if e["kind"] == "start_refused"]
    end = gap + pd.Timedelta(minutes=29)
    assert ev["level"] == "warning"
    assert ev["message"] == f"history gaps in warm-up window: BTC/USD {gap:%d %b %H:%M}-{end:%H:%M} UTC"
    assert not store.decisions("btc-test", action="start")  # a refused start is no decision
    refill = bars[(bars.index >= gap) & (bars.index < end + pd.Timedelta(minutes=1))]
    hs.append_bars(VENUE, PAIR, [(t.value, r.open, r.high, r.low, r.close, r.volume) for t, r in refill.iterrows()],
                   "refill")  # the hub's refill (its write path, stamped at the open) closes the hole
    assert hs.gaps(VENUE, PAIR) == []
    assert "command_error" not in start() and store.sleeve("btc-test").desired_state == "running"


def test_a_store_too_stale_to_warm_up_from_holds_nothing_back(client, tmp_path, monkeypatch):  # noqa: F811
    """Ten hours old, past paper.node.HISTORY_MAX_LAG: the strategy warms up on the venue's own candles, so the
    store's gaps aren't what its indicators read."""
    c, _ = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    hs = HistoryStore()
    bars = _minutes(60, pd.Timestamp.now(tz="UTC").floor("1min") - pd.Timedelta(hours=10))
    hs.append(VENUE, PAIR, bars, cursor="x")
    start = bars.index[-1] - pd.Timedelta(hours=3)
    _hole(hs, VENUE, PAIR, start.strftime("%Y-%m"), start, 30)
    assert hs.gaps(VENUE, PAIR)
    assert _new(c).headers["location"] == "/sleeves/btc-test"
