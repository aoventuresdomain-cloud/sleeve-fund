"""Each instrument's history badge and the gap check (UI v2, item 10): first and last candle and any gaps in what
the store holds, the badge's words, and backtests and studies held back while there are gaps. No new collector."""

import pandas as pd
import pytest
from test_dashboard import AUTH, SAME, client  # noqa: F401 - the fixture
from test_research import _stored_minutes

from sleeve_fund import history
from sleeve_fund.dashboard.development import blocked_by_gaps, history_badge
from sleeve_fund.history import HistoryStore, _load, _save


def _hole(store, venue, pair, month, start, minutes):
    """Take minutes out of a stored month, as a write that never finished would leave it."""
    path = store._dir(venue, pair) / f"{month}.npz"
    df = _load(path)
    cut = pd.date_range(start, periods=minutes, freq="1min")
    _save(path, df[~df.index.isin(cut)])


def test_a_whole_series_has_no_gaps_across_its_months(tmp_path):
    store = HistoryStore(tmp_path)
    store.append("KRAKEN", "ETH/USD", _stored_minutes(40), cursor="x")  # 40 days: two or three month files
    assert len(list(store._dir("KRAKEN", "ETH/USD").glob("*.npz"))) >= 2
    assert store.gaps("KRAKEN", "ETH/USD") == []
    assert store.gaps("KRAKEN", "SOL/USD") == []  # nothing stored: nothing missing


def test_missing_minutes_are_found_inside_a_month_and_where_a_month_is_gone(tmp_path):
    store = HistoryStore(tmp_path)
    bars = _stored_minutes(40)
    store.append("KRAKEN", "ETH/USD", bars, cursor="x")
    first = bars.index[0]
    start = first + pd.Timedelta(days=2, minutes=7)
    _hole(store, "KRAKEN", "ETH/USD", start.strftime("%Y-%m"), start, 5)
    assert store.gaps("KRAKEN", "ETH/USD") == [(start, start + pd.Timedelta(minutes=4))]
    # The last month's file lost: everything from the month before's end to the coverage's last minute.
    files = sorted(store._dir("KRAKEN", "ETH/USD").glob("*.npz"))
    files[-1].unlink()
    cov = store.coverage("KRAKEN", "ETH/USD")
    gaps = store.gaps("KRAKEN", "ETH/USD")
    assert len(gaps) == 2 and gaps[-1][1] == cov.last
    assert gaps[-1][0] == _load(files[-2]).index[-1] + pd.Timedelta(minutes=1)


def test_an_unchanged_month_is_read_once(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path)
    store.append("KRAKEN", "ETH/USD", _stored_minutes(40), cursor="x")
    store.gaps("KRAKEN", "ETH/USD")
    reads = []
    real = history.np.load
    monkeypatch.setattr(history.np, "load", lambda *a, **k: reads.append(a[0]) or real(*a, **k))
    store.gaps("KRAKEN", "ETH/USD")
    assert reads == []


NOW = pd.Timestamp("2026-10-05 14:31", tz="UTC")


@pytest.mark.parametrize("row, state, text", [
    (None, "none", "not stored yet"),
    ({"pair": "SOL/USD", "first": None, "gaps": []}, "filling", "being filled"),  # asked for, nothing yet
    ({"pair": "SOL/USD", "first": NOW, "last": NOW, "state": "catching up", "gaps": []}, "filling", "being filled"),
    ({"pair": "SOL/USD", "first": NOW, "last": NOW, "state": "current", "gaps": [(NOW, NOW)] * 3},
     "gaps", "3 gaps · backtests wait until filled"),
    ({"pair": "SOL/USD", "first": NOW, "last": NOW, "state": "current", "gaps": []},
     "stored", "stored · last candle 14:31 · no gaps"),
])
def test_the_badge_says_which_of_the_four_it_is(row, state, text):
    assert history_badge(row) == {"state": state, "text": text}


def test_only_gaps_hold_a_backtest_back():
    assert blocked_by_gaps("ETH/USD", [(NOW, NOW)]) == "ETH/USD: 1 gap · backtests wait until filled"
    assert blocked_by_gaps("ETH/USD", []) is None


def test_a_backtest_and_a_study_wait_while_the_history_has_gaps(client, tmp_path, monkeypatch):  # noqa: F811
    c, _ = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    store = HistoryStore()
    bars = _stored_minutes(130)
    store.append("KRAKEN", "ETH/USD", bars, cursor="x")
    start = bars.index[0] + pd.Timedelta(days=3)
    _hole(store, "KRAKEN", "ETH/USD", start.strftime("%Y-%m"), start, 30)
    words = "ETH/USD: 1 gap · backtests wait until filled"
    page = c.get("/backtest?run=1&instrument=ETH/USD&strategy=buy_and_hold", auth=AUTH).text
    assert words in page
    form = {"strategy": "buy_and_hold", "instrument": "eth/usd", "minutes": "240", "risk_profile": "conservative",
            "train_days": "60", "test_days": "30", "holdout_days": "30"}
    r = c.post("/research/run", data=form, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 200 and words in r.text
    # The pick-list's data: the instrument with its first and last candle, its gap and the badge.
    got = c.get("/api/history/coverage?venue=kraken", auth=AUTH).json()
    eth = next(i for i in got["instruments"] if i["pair"] == "ETH/USD")
    assert eth["state"] == "gaps" and eth["text"] == "1 gap · backtests wait until filled" and len(eth["gaps"]) == 1
    assert c.get("/api/history/coverage?venue=nope", auth=AUTH).status_code == 400
