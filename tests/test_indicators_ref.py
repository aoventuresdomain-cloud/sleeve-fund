"""Every indicator block against an independent pandas reference, on recorded market data and on a long
synthetic minute series (several UTC days, quiet bars with no volume). The references are written the way
the textbook defines each one, not the way the block streams it, so a shared mistake is unlikely."""

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.indicators import ExponentialMovingAverage

from sleeve_fund.strategies.indicators import (
    BLOCKS, DAY_OF_MINUTE_BARS, FLAT_BAND, NS_PER_DAY, Atr, Bollinger, EfficiencyRatio, Ema, RelativeVolume, Rsi, Sma, Vwap,
    Wma, make_block, settle_bars, warmup_for,
)

REPLAY = Path(__file__).parent / "data" / "replay"
NS_PER_MIN = 60 * 1_000_000_000


def _recorded_bars() -> pd.DataFrame:
    """1m bars built from the trades of the recorded venue sessions, stamped at their close. Minutes with no
    trade are skipped, as a candle feed with no trades would skip them; the sessions are joined end to end."""
    frames = []
    for path in sorted(REPLAY.glob("*-rec-1m-*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            trades = [r for r in map(json.loads, f) if r.get("k") == "t"]
        t = pd.DataFrame({"ts": [r["e"] for r in trades], "p": [float(r["p"]) for r in trades],
                          "s": [float(r["s"]) for r in trades]})
        t["close_ts"] = (t["ts"] // NS_PER_MIN + 1) * NS_PER_MIN
        g = t.groupby("close_ts")
        frames.append(pd.DataFrame({"high": g["p"].max(), "low": g["p"].min(), "close": g["p"].last(),
                                    "volume": g["s"].sum()}))
    bars = pd.concat(frames).reset_index().rename(columns={"close_ts": "ts"})
    assert len(bars) > 100, "the recorded sessions should give over 100 minute bars"
    return bars


def _synthetic_bars(days: float = 3.5, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(days * 1440)
    close = 85_000 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    open_ = np.concatenate([[85_000], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0005, n))
    volume = rng.exponential(2.0, n)
    volume[rng.random(n) < 0.05] = 0.0  # quiet minutes
    volume[100:140] = 0.0  # and a quiet stretch longer than most windows
    start = 1_790_000_000 * 1_000_000_000 // NS_PER_DAY * NS_PER_DAY + 7 * 3_600 * 1_000_000_000  # 07:00 UTC
    ts = start + NS_PER_MIN * np.arange(1, n + 1)
    return pd.DataFrame({"ts": ts, "high": high, "low": low, "close": close, "volume": volume})


DATA = {"recorded": _recorded_bars(), "synthetic": _synthetic_bars()}


def _stream(block, bars: pd.DataFrame, feed, key=None) -> tuple[np.ndarray, np.ndarray]:
    """Feed every bar; return each bar's output (nan before the block is initialized) and the initialized flags."""
    out, ready = [], []
    for row in bars.itertuples(index=False):
        feed(block, row)
        ready.append(bool(block.initialized))
        out.append((block.values[key] if key else block.value) if block.initialized else np.nan)
    return np.array(out), np.array(ready)


def _close(b, r):
    b.update_raw(r.close)


def _hlc(b, r):
    b.update_raw(r.high, r.low, r.close)


def _ohlcv(b, r):
    b.update_raw(r.high, r.low, r.close, r.volume, int(r.ts))


def _vol(b, r):
    b.update_raw(r.volume)


def _check(got: np.ndarray, ready: np.ndarray, want: pd.Series) -> None:
    want = want.to_numpy(dtype=float)
    assert ready.any(), "the block never initialized"
    assert not np.isnan(got[ready]).any()
    np.testing.assert_allclose(got[ready], want[ready], rtol=1e-9, atol=1e-9)


def _wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder's average: the mean of the first n values, then (previous x (n - 1) + value) / n."""
    seeded = x.copy()
    seeded.iloc[: n - 1] = np.nan
    seeded.iloc[n - 1] = x.iloc[:n].mean()
    out = seeded.iloc[n - 1:].ewm(alpha=1 / n, adjust=False).mean()
    return out.reindex(x.index)


@pytest.fixture(params=sorted(DATA))
def bars(request):
    return DATA[request.param]


# ---- references ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 14, 60])
def test_sma(bars, n):
    _check(*_stream(Sma(n), bars, _close), bars["close"].rolling(n).mean())


@pytest.mark.parametrize("n", [2, 14, 60])
def test_ema(bars, n):
    _check(*_stream(Ema(n), bars, _close), bars["close"].ewm(span=n, adjust=False).mean())


@pytest.mark.parametrize("n", [2, 14, 200, 2000])
def test_ema_matches_the_engines_ema_that_rsi_pullback_trades(n):
    ours, theirs = Ema(n), ExponentialMovingAverage(n)
    for c in DATA["synthetic"]["close"]:
        ours.update_raw(c)
        theirs.update_raw(c)
        assert ours.initialized == theirs.initialized
        assert ours.value == pytest.approx(theirs.value, rel=1e-12)


@pytest.mark.parametrize("n", [1, 5, 30])
def test_wma(bars, n):
    w = np.arange(1, n + 1, dtype=float)
    ref = bars["close"].rolling(n).apply(lambda a: a @ w / w.sum(), raw=True)
    _check(*_stream(Wma(n), bars, _close), ref)


@pytest.mark.parametrize("n", [1, 15, 60])
def test_rolling_vwap(bars, n):
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    pv, v = (tp * bars["volume"]).rolling(n).sum(), bars["volume"].rolling(n).sum()
    ref = (pv / v).where(v > 0, tp)
    _check(*_stream(Vwap("rolling", n), bars, _ohlcv), ref)


def test_day_vwap():
    bars = DATA["synthetic"]
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    day = (bars["ts"] - 1) // NS_PER_DAY
    pv, v = (tp * bars["volume"]).groupby(day).cumsum(), bars["volume"].groupby(day).cumsum()
    ref = (pv / v).where(v > 0, tp)
    got, ready = _stream(Vwap("day"), bars, _ohlcv)
    _check(got, ready, ref)
    # Not initialized on the part-day it started in; initialized from the first bar of the next UTC day.
    first_day = day == day.iloc[0]
    assert not ready[first_day.to_numpy()].any() and ready[~first_day.to_numpy()].all()


def test_day_vwap_counts_the_bar_closing_at_midnight_in_the_old_day():
    v = Vwap("day")
    midnight = 20_000 * NS_PER_DAY
    v.update_raw(10, 10, 10, 1, midnight - NS_PER_MIN)
    v.update_raw(20, 20, 20, 1, midnight)  # the 23:59 to 00:00 bar
    assert v.value == 15 and not v.initialized
    v.update_raw(40, 40, 40, 1, midnight + NS_PER_MIN)  # first bar of the new day
    assert v.value == 40 and v.initialized


@pytest.mark.parametrize("n,k", [(2, 1.0), (20, 2.0), (60, 2.5)])
def test_bollinger(bars, n, k):
    c = bars["close"]
    # Two-pass per window: pandas' own rolling std loses about 1% on two nearly equal prices near 85,000.
    windows = np.lib.stride_tricks.sliding_window_view(c.to_numpy(), n)
    mid = pd.Series(np.concatenate([np.full(n - 1, np.nan), windows.mean(axis=1)]), index=c.index)
    sd = pd.Series(np.concatenate([np.full(n - 1, np.nan), windows.std(axis=1)]), index=c.index)
    sd = sd.where(sd > FLAT_BAND * mid.abs(), 0.0)
    upper, lower = mid + k * sd, mid - k * sd
    refs = {"mid": mid, "upper": upper, "lower": lower, "width": (upper - lower) / mid,
            "pct_b": ((c - lower) / (upper - lower)).where(upper > lower, 0.5)}
    for key, ref in refs.items():
        got, ready = _stream(Bollinger(n, k), bars, _close, key)
        # A close sitting on a band puts %B at 0 - 0 at price scale: equal to a millionth of the band is equal.
        atol = 1e-6 if key == "pct_b" else 1e-7
        np.testing.assert_allclose(got[ready], ref.to_numpy()[ready], rtol=1e-7, atol=atol, err_msg=key)


def test_bollinger_on_a_flat_window_has_no_band():
    b = Bollinger(5)
    for _ in range(5):
        b.update_raw(100.0)
    assert b.values == {"mid": 100.0, "upper": 100.0, "lower": 100.0, "width": 0.0, "pct_b": 0.5}


@pytest.mark.parametrize("n", [1, 20, 50])
def test_relative_volume(bars, n):
    vol = bars["volume"]
    avg = vol.rolling(n).mean().shift(1)
    ref = (vol / avg).where(avg > 0, 0.0)
    _check(*_stream(RelativeVolume(n), bars, _vol), ref)


@pytest.mark.parametrize("n", [1, 10, 30])
def test_efficiency_ratio(bars, n):
    c = bars["close"]
    path = c.diff().abs().rolling(n).sum()
    ref = (c.diff(n).abs() / path).where(path > 0, 0.0)
    _check(*_stream(EfficiencyRatio(n), bars, _close), ref)


@pytest.mark.parametrize("n", [1, 14, 60])
def test_atr(bars, n):
    prev = bars["close"].shift(1)
    tr = np.maximum(bars["high"], prev.fillna(bars["high"])) - np.minimum(bars["low"], prev.fillna(bars["low"]))
    _check(*_stream(Atr(n), bars, _hlc), tr.rolling(n).mean())


@pytest.mark.parametrize("n", [2, 14, 30])
def test_rsi(bars, n):
    change = bars["close"].diff().iloc[1:]
    gain, loss = _wilder(change.clip(lower=0), n), _wilder((-change).clip(lower=0), n)
    ref = (100 - 100 / (1 + gain / loss)).where(loss > 0, np.where(gain > 0, 100.0, 50.0)).reindex(bars.index)
    _check(*_stream(Rsi(n), bars, _close), ref)


# ---- the interface every block keeps -----------------------------------------------------------------------

def _all_blocks():
    """One of each block, with the feed its update_raw takes."""
    return [(Sma(20), _close), (Ema(20), _close), (Wma(20), _close), (Vwap("day"), _ohlcv),
            (Vwap("rolling", 30), _ohlcv), (Rsi(14), _close), (Bollinger(20), _close), (Atr(14), _hlc),
            (RelativeVolume(20), _vol), (EfficiencyRatio(10), _close)]


def test_every_named_block_is_tested_here():
    assert {type(b) for b, _ in _all_blocks()} == set(BLOCKS.values())


@pytest.mark.parametrize("i", range(10))
def test_no_look_ahead(i):
    """A value at a bar is the same whether or not later bars exist: each prefix reads as the full run did."""
    block, feed = _all_blocks()[i]
    bars = DATA["synthetic"]
    full, _ = _stream(copy.deepcopy(block), bars, feed)
    for cut in (500, 1500, 3000):
        part, _ = _stream(copy.deepcopy(block), bars.iloc[:cut], feed)
        np.testing.assert_array_equal(part, full[:cut])


@pytest.mark.parametrize("i", range(10))
def test_reset_gives_a_fresh_block(i):
    block, feed = _all_blocks()[i]
    fresh = copy.deepcopy(block)
    for row in DATA["synthetic"].iloc[:300].itertuples(index=False):
        feed(block, row)
    block.reset()
    for row in DATA["synthetic"].iloc[300:900].itertuples(index=False):
        feed(block, row)
        feed(fresh, row)
    assert block.values == fresh.values and block.initialized == fresh.initialized


@pytest.mark.parametrize("i", range(10))
def test_settled_after_its_warmup(i):
    """Fed warmup_bars bars, every block is initialized: warm-up from warmup_bars is always enough."""
    block, feed = _all_blocks()[i]
    for row in DATA["synthetic"].iloc[: block.warmup_bars].itertuples(index=False):
        feed(block, row)
    assert block.initialized


def test_warmup_bars_from_settings():
    assert Sma(50).warmup_bars == 50
    assert Ema(20).warmup_bars == settle_bars(20)
    assert Rsi(14).warmup_bars == settle_bars(14)
    assert Atr(14).warmup_bars == 15
    assert Wma(9).warmup_bars == 9
    assert Bollinger(20).warmup_bars == 20
    assert RelativeVolume(20).warmup_bars == 21
    assert EfficiencyRatio(10).warmup_bars == 11
    assert Vwap("rolling", 30).warmup_bars == 30
    assert Vwap("day").warmup_bars == DAY_OF_MINUTE_BARS + 1
    assert warmup_for([Sma(50), Ema(20), RelativeVolume(20)]) == settle_bars(20)
    assert warmup_for([]) == 0


@pytest.mark.parametrize("kind", ["sma", "ema", "wma", "rsi", "bollinger", "efficiency_ratio"])
def test_peek_reads_the_forming_bar_without_moving_the_block(kind):
    block = make_block(kind) if kind in ("rsi", "bollinger", "efficiency_ratio") else make_block(kind, period=20)
    closes = DATA["synthetic"]["close"].iloc[:400]
    for c in closes:
        block.update_raw(c)
    before = copy.deepcopy(block.values)
    peeked = block.peek(closes.iloc[-1] * 1.01)
    assert block.values == before
    block.update_raw(closes.iloc[-1] * 1.01)
    assert peeked == block.value


def test_make_block_by_name():
    assert isinstance(make_block("vwap", anchor="rolling", period=5), Vwap)
    assert make_block("bollinger", period=10, k=1.5).k == 1.5
    with pytest.raises(ValueError, match="no indicator block"):
        make_block("macd")


@pytest.mark.parametrize("kind,settings", [
    ("ema", {"period": 0}), ("wma", {"period": 2.5}), ("bollinger", {"period": 1}), ("bollinger", {"k": 0}),
    ("vwap", {"anchor": "week"}), ("vwap", {"anchor": "rolling"}), ("vwap", {"anchor": "day", "period": 5}),
    ("relative_volume", {"period": True}), ("efficiency_ratio", {"period": -1}),
])
def test_bad_settings_are_refused(kind, settings):
    with pytest.raises(ValueError):
        make_block(kind, **settings)


def test_day_vwap_needs_close_times():
    with pytest.raises(ValueError, match="close time"):
        Vwap("day").update_raw(1, 1, 1, 1)
