"""P1-3s wiring: the candles API carries the strategy's own indicator lines (charts.indicators), recomputed from the
history store's closed candles, for the chart's range, with warm-up flagged and each point the value the model read
when it decided."""

import numpy as np
import pandas as pd
import pytest

from sleeve_fund import history
from sleeve_fund.dashboard import charts
from sleeve_fund.history import HistoryStore
from sleeve_fund.instruments import history_price_decimals
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.venues import KRAKEN
from test_dashboard import AUTH, _new, client  # noqa: F401

START = pd.Timestamp("2026-08-01", tz="UTC")


def _minutes(days=30, seed=4):
    n = days * 1440
    rng = np.random.default_rng(seed)
    c = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.0015, n))), 2)  # priced in cents, as a venue's tick
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.date_range(START, periods=n, freq="1min")  # stamped at the open, as the store keeps them
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.05, "low": np.minimum(o, c) - 0.05, "close": c,
                         "volume": rng.uniform(1, 3, n)}, index=idx)


@pytest.fixture
def stored(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    charts._drawn.clear()
    m = _minutes()
    HistoryStore().append(KRAKEN.name, "SOL/USD", m, cursor="t")
    hourly = m.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return hourly  # the venue's own hourly candles, stamped at the open, as the chart gets them


def test_the_candles_api_carries_the_strategys_lines_over_the_charts_range(client, stored, monkeypatch):  # noqa: F811
    c, _ = client
    _new(c, name="sol-x", instrument="SOL/USD")  # trend_filter, fast 10 and slow 30, on 1-hour candles
    chart = stored.iloc[-200:]
    charts._cache.clear()
    monkeypatch.setattr(KRAKEN, "ohlc_history", lambda pair, minutes: chart)
    d = c.get("/api/sleeves/sol-x/candles", auth=AUTH).json()
    lines = {x["key"]: x for x in d["indicators"]}
    assert set(lines) == {"sma_10", "sma_30"} and "indicators_note" not in d
    first_close = int((chart.index[0] + pd.Timedelta("1h")).timestamp())
    for x in lines.values():
        t = [p[0] for p in x["points"]]
        # One per chart candle, at its close; the newest is left out, as the store can't vouch its last minute closed.
        assert t == [int(i.timestamp()) + 3600 for i in chart.index[:-1]]
        assert x["settled_from"] <= first_close and all(p[1] is not None for p in x["points"])

    # The points are the model's own: a backtest over the store's candles journals the same values when it trades.
    df = HistoryStore().read(KRAKEN.name, "SOL/USD", 60)
    inst = KRAKEN.instrument("SOL", "USD", price_precision=history_price_decimals(df["close"]))  # as the backtest page
    res = run_backtest("trend_filter", df, inst, {"fast": 10, "slow": 30}, bar_minutes=60, half_spread=0)
    checked = 0
    for order, dec in res.decisions.items():
        t = int(res.fills.loc[order, "ts_last"].timestamp()) if order in res.fills.index else None
        pts = dict(map(tuple, lines["sma_30"]["points"]))
        if t in pts and "sma_30" in (dec.get("signal") or {}):
            assert pts[t] == pytest.approx(dec["signal"]["sma_30"], abs=1e-8)
            checked += 1
    assert checked >= 2

    # On another interval the lines would sit on the wrong candles: none, and the chart says why.
    d = c.get("/api/sleeves/sol-x/candles?interval=4h", auth=AUTH).json()
    assert d["indicators"] == [] and "own 1h candles" in d["indicators_note"]


def test_the_warm_up_inside_the_chart_is_flagged(stored):
    from types import SimpleNamespace

    s = SimpleNamespace(name="w", strategy="trend_filter", params={"fast": 10, "slow": 30}, bar_spec="1-HOUR-LAST-INTERNAL",
                        venue=None, instrument="SOL/USD")
    lines, why = charts.indicators(s, stored.iloc[:100], 60)  # the chart starts where the history does
    assert why is None
    slow = next(x for x in lines if x["key"] == "sma_30")
    assert slow["settled_from"] == int(stored.index[29].timestamp()) + 3600  # the 30th candle's close
    assert slow["points"][0][1] is None  # nothing before the average has its 30 candles


def test_without_stored_history_the_chart_still_draws_and_says_why(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "empty")
    charts._drawn.clear()
    s = SimpleNamespace(name="n", strategy="trend_filter", params={}, bar_spec="1-HOUR-LAST-INTERNAL", venue=None,
                        instrument="SOL/USD")
    chart = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0]},
                         index=pd.DatetimeIndex([START]))
    lines, why = charts.indicators(s, chart, 60)
    assert lines == [] and "No stored history for SOL/USD" in why


def test_the_first_visible_point_is_the_models_own_after_its_warm_up_is_read_in_front_of_the_window(stored):
    """A Wilder RSI carries a trace of where it started. The chart reads twice the model's warm-up of the strategy's
    own candles before its window and trims them, so from the first visible candle each point is what the model read
    on the store's whole history (what a backtest over it journals), to well under the journal's last digit."""
    from types import SimpleNamespace

    from sleeve_fund.strategies.series import indicator_series

    params = {"rsi_period": 14}
    s = SimpleNamespace(name="r", strategy="rsi_bands", params=params, bar_spec="1-HOUR-LAST-INTERNAL", venue=None,
                        instrument="SOL/USD")
    chart = stored.iloc[400:520]  # starts well inside the history: 400 hours of candles before the window
    lines, why = charts.indicators(s, chart, 60)
    assert why is None
    shown = dict(map(tuple, lines[0]["points"]))
    assert min(shown) == int(chart.index[0].timestamp()) + 3600  # the window's first candle, at its close
    assert lines[0]["settled_from"] <= min(shown)  # settled before the window opens: nothing in it is warm-up
    df = HistoryStore().read(KRAKEN.name, "SOL/USD", 60)
    inst = KRAKEN.instrument("SOL", "USD", price_precision=8)
    whole = dict(map(tuple, indicator_series("rsi_bands", df, inst, params, 60)[0]["points"]))
    assert all(abs(v - whole[t]) < 1e-6 for t, v in shown.items())  # RSI points: far under the journal's 4 decimals
