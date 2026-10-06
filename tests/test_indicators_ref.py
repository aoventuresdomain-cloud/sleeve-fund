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
    BLOCKS, DAY_OF_MINUTE_BARS, FLAT_BAND, NS_PER_DAY, Atr, AtrSma, Bollinger, Donchian, EfficiencyRatio, Ema, Keltner, RelativeVolume, Rsi, RsiDivergence, Sma, Stochastic, Vwap,
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
    """Pinned reference (P1-I4): alpha 2 / (n + 1), seeded with the first close (the engine's convention), not
    with an n-bar simple average."""
    want = bars["close"].ewm(span=n, adjust=False).mean().to_numpy()
    e = Ema(n)
    for i, c in enumerate(bars["close"]):
        e.update_raw(c)
        assert e._value == pytest.approx(want[i], rel=1e-9)  # the arithmetic, before it settles too
        assert e.initialized == (i + 1 >= settle_bars(n)) and (e.value is None) != e.initialized


@pytest.mark.parametrize("n", [2, 14, 200, 2000])
def test_ema_matches_the_engines_ema_that_rsi_pullback_trades(n):
    ours, theirs = Ema(n), ExponentialMovingAverage(n)
    for c in DATA["synthetic"]["close"]:
        ours.update_raw(c)
        theirs.update_raw(c)
        # Ours is initialized only once settled (QA, F1); the engine's after `period` values.
        assert ours.initialized <= theirs.initialized and ours.initialized == (ours.count >= settle_bars(n))
        assert ours._value == pytest.approx(theirs.value, rel=1e-12)  # before it is initialized too


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
    assert v._value == 15 and v.value is None and not v.initialized
    v.update_raw(40, 40, 40, 1, midnight + NS_PER_MIN)  # first bar of the new day
    assert v.value == 40 and v.initialized


@pytest.mark.parametrize("n,k", [(2, 1.0), (20, 2.0), (60, 2.5)])
def test_bollinger(bars, n, k):
    """Pinned reference (P1-I4): the population standard deviation (divide by n, not n - 1) of the last n
    closes, as Bollinger defined the bands."""
    c = bars["close"]
    # Two-pass per window: pandas' own rolling std loses about 1% on two nearly equal prices near 85,000.
    windows = np.lib.stride_tricks.sliding_window_view(c.to_numpy(), n)
    mid = pd.Series(np.concatenate([np.full(n - 1, np.nan), windows.mean(axis=1)]), index=c.index)
    sd = pd.Series(np.concatenate([np.full(n - 1, np.nan), windows.std(axis=1)]), index=c.index)
    sd = sd.where(sd > FLAT_BAND * mid.abs(), 0.0)
    upper, lower = mid + k * sd, mid - k * sd
    # %B from the close's distance to the mid, as the block works it: upper minus lower at 85,000 keeps too few
    # digits for a band a millionth of the price wide.
    refs = {"mid": mid, "upper": upper, "lower": lower, "width": 2 * k * sd / mid,
            "pct_b": (0.5 + (c - mid) / (2 * k * sd)).where(sd > 0, 0.5)}
    for key, ref in refs.items():
        got, ready = _stream(Bollinger(n, k), bars, _close, key)
        np.testing.assert_allclose(got[ready], ref.to_numpy()[ready], rtol=1e-9, atol=1e-9, err_msg=key)


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


def _true_range(bars):
    prev = bars["close"].shift(1)
    return np.maximum(bars["high"], prev.fillna(bars["high"])) - np.minimum(bars["low"], prev.fillna(bars["low"]))


@pytest.mark.parametrize("n", [1, 3, 6])  # 20 lengths to settle (QA P1-A1): 13 outran the 137 recorded candles
def test_atr_is_wilders(bars, n):
    """Pinned reference (P1-I4): `atr` is Wilder's ATR, seeded with the mean of the first n true ranges, the
    first bar's range its high minus low."""
    _check(*_stream(Atr(n), bars, _hlc), _wilder(_true_range(bars), n))


@pytest.mark.parametrize("n", [1, 14, 60])
def test_atr_sma(bars, n):
    prev = bars["close"].shift(1)
    tr = np.maximum(bars["high"], prev.fillna(bars["high"])) - np.minimum(bars["low"], prev.fillna(bars["low"]))
    _check(*_stream(AtrSma(n), bars, _hlc), tr.rolling(n).mean())


@pytest.mark.parametrize("n", [2, 14, 30])
def test_rsi(bars, n):
    change = bars["close"].diff().iloc[1:]
    gain, loss = _wilder(change.clip(lower=0), n), _wilder((-change).clip(lower=0), n)
    ref = (100 - 100 / (1 + gain / loss)).where(loss > 0, np.where(gain > 0, 100.0, 50.0)).reindex(bars.index)
    _check(*_stream(Rsi(n), bars, _close), ref)


@pytest.mark.parametrize("n", [1, 20, 55])
@pytest.mark.parametrize("source", ["high_low", "close"])
def test_donchian(bars, n, source):
    hi, lo = (bars["close"], bars["close"]) if source == "close" else (bars["high"], bars["low"])
    upper, lower = hi.shift(1).rolling(n).max(), lo.shift(1).rolling(n).min()
    for key, ref in {"upper": upper, "lower": lower, "mid": (upper + lower) / 2}.items():
        _check(*_stream(Donchian(n, source), bars, _hlc, key), ref)


@pytest.mark.parametrize("n,smooth,d", [(1, 1, 1), (14, 3, 3), (30, 5, 4)])
def test_stochastic(bars, n, smooth, d):
    top, bottom = bars["high"].rolling(n).max(), bars["low"].rolling(n).min()
    raw = (100 * (bars["close"] - bottom) / (top - bottom)).where(top > bottom, 50.0).where(top.notna())
    k = raw.rolling(smooth).mean()
    for key, ref in {"k": k, "d": k.rolling(d).mean()}.items():
        _check(*_stream(Stochastic(n, smooth, d), bars, _hlc, key), ref)


def test_stochastic_on_a_flat_range_reads_50():
    s = Stochastic(3, 1, 1)
    for _ in range(3):
        s.update_raw(100.0, 100.0, 100.0)
    assert s.values == {"k": 50.0, "d": 50.0}


@pytest.mark.parametrize("n,atr_n,k", [(1, 1, 1.0), (5, 10, 2.0), (12, 3, 1.5)])
def test_keltner(bars, n, atr_n, k):
    """Pinned reference (P1-I4): the middle line is the EMA of the close, the range Wilder's ATR."""
    mid = bars["close"].ewm(alpha=2 / (n + 1), adjust=False).mean()  # the engine EMA: starts at the first close
    atr = _wilder(_true_range(bars), atr_n)
    for key, ref in {"mid": mid, "upper": mid + k * atr, "lower": mid - k * atr}.items():
        got, ready = _stream(Keltner(n, atr_n, k), bars, _hlc, key)
        np.testing.assert_allclose(got[ready], ref.to_numpy()[ready], rtol=1e-9, atol=1e-7, err_msg=key)


def _rsi_ref(close: pd.Series, n: int) -> pd.Series:
    change = close.diff().iloc[1:]
    gain, loss = _wilder(change.clip(lower=0), n), _wilder((-change).clip(lower=0), n)
    return (100 - 100 / (1 + gain / loss)).where(loss > 0, np.where(gain > 0, 100.0, 50.0)).reindex(close.index)


def _divergence_ref(bars: pd.DataFrame, rsi_period: int, left: int, right: int, max_gap: int) -> pd.DataFrame:
    """Swings found over the whole series with rolling windows, then compared pairwise: an independent route to
    the same definition. Each signal is placed on the bar that confirms its swing (swing bar + right)."""
    rsi = _rsi_ref(bars["close"], rsi_period)
    low, high = bars["low"], bars["high"]
    later_min = low[::-1].rolling(right).min()[::-1].shift(-1)  # min of the `right` bars after
    later_max = high[::-1].rolling(right).max()[::-1].shift(-1)
    swing_low = (low <= low.shift(1).rolling(left).min()) & (low < later_min) & rsi.notna()
    swing_high = (high >= high.shift(1).rolling(left).max()) & (high > later_max) & rsi.notna()
    out = pd.DataFrame({"bullish": 0, "bearish": 0}, index=bars.index)
    for flags, price, key, beyond in ((swing_low, low, "bullish", np.less), (swing_high, high, "bearish", np.greater)):
        idx = list(np.flatnonzero(flags.to_numpy()))
        for a, b in zip(idx, idx[1:]):
            rsi_moved = rsi.iloc[b] > rsi.iloc[a] if key == "bullish" else rsi.iloc[b] < rsi.iloc[a]
            if b - a <= max_gap and beyond(price.iloc[b], price.iloc[a]) and rsi_moved:
                out.iloc[b + right, out.columns.get_loc(key)] = 1
    return out


@pytest.mark.parametrize("left,right,max_gap", [(3, 3, 50), (5, 2, 30), (1, 1, 10)])
def test_rsi_divergence(bars, left, right, max_gap):
    ref = _divergence_ref(bars, 14, left, right, max_gap)
    got = {"bullish": [], "bearish": []}
    block = RsiDivergence(14, left, right, max_gap)
    ready = []
    for row in bars.itertuples(index=False):
        block.update_raw(row.high, row.low, row.close)
        ready.append(block.initialized)
        for k in got:
            got[k].append(block.values[k] or 0)  # None before its warm-up
    ready = np.array(ready)
    assert ready.sum() == max(len(bars) - block.warmup_bars + 1, 0)  # initialized from its warm-up on
    for k in got:
        np.testing.assert_array_equal(np.array(got[k])[ready], ref[k].to_numpy()[ready], err_msg=k)
    if len(bars) > 1000:
        assert ref["bullish"].sum() > 5 and ref["bearish"].sum() > 5, "the test series should hold divergences"


def test_rsi_divergence_signals_only_once_the_swing_is_confirmed():
    d = RsiDivergence(rsi_period=2, left=1, right=2, max_gap=20)
    # Falling, a first low at 90, a bounce, a lower low at 88 on a gentler fall (RSI higher), then a bounce.
    closes = [100, 98, 96, 94, 92, 90, 95, 97, 96, 93, 91, 89.5, 88.8, 88, 93, 95, 96]
    flags = []
    for c in closes:
        d.update_raw(c + 0.5, c - 0.5, c)
        flags.append(d._vals["bullish"])  # what it works out; it reads None until its warm-up
    swing = closes.index(88)
    assert flags.index(1) == swing + d.confirm_lag and sum(flags) == 1


# ---- the interface every block keeps -----------------------------------------------------------------------

def _all_blocks():
    """One of each block, with the feed its update_raw takes."""
    return [(Sma(20), _close), (Ema(20), _close), (Wma(20), _close), (Vwap("day"), _ohlcv),
            (Vwap("rolling", 30), _ohlcv), (Rsi(14), _close), (Bollinger(20), _close), (AtrSma(14), _hlc), (Atr(14), _hlc),
            (RelativeVolume(20), _vol), (EfficiencyRatio(10), _close), (Donchian(20), _hlc),
            (RsiDivergence(), _hlc), (Stochastic(), _hlc), (Keltner(), _hlc)]


def test_every_named_block_is_tested_here():
    assert {type(b) for b, _ in _all_blocks()} == set(BLOCKS.values())


@pytest.mark.parametrize("i", range(15))
def test_no_look_ahead(i):
    """A value at a bar is the same whether or not later bars exist: each prefix reads as the full run did."""
    block, feed = _all_blocks()[i]
    bars = DATA["synthetic"]
    full, _ = _stream(copy.deepcopy(block), bars, feed)
    for cut in (500, 1500, 3000):
        part, _ = _stream(copy.deepcopy(block), bars.iloc[:cut], feed)
        np.testing.assert_array_equal(part, full[:cut])


@pytest.mark.parametrize("i", range(15))
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


@pytest.mark.parametrize("i", range(15))
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
    assert AtrSma(14).warmup_bars == 15
    assert Wma(9).warmup_bars == 9
    assert Bollinger(20).warmup_bars == 20
    assert RelativeVolume(20).warmup_bars == 21
    assert EfficiencyRatio(10).warmup_bars == 11
    assert Vwap("rolling", 30).warmup_bars == 30
    assert Vwap("day").warmup_bars == DAY_OF_MINUTE_BARS + 1
    assert warmup_for([Sma(50), Ema(20), RelativeVolume(20)]) == settle_bars(20)
    assert Donchian(20).warmup_bars == 21
    assert RsiDivergence(14, 3, 3, 50).warmup_bars == settle_bars(14) + 56
    assert Atr(14).warmup_bars == 20 * 14  # QA P1-A1: a range average settles over twenty lengths
    assert Stochastic(14, 3, 3).warmup_bars == 18
    assert Keltner(20, 10).warmup_bars == settle_bars(20) and Keltner(2, 30).warmup_bars == 20 * 30
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
    ("vwap", {"anchor": "week"}), ("ema", {"lookback": 5}), ("bollinger", {"k": 50}), ("sma", {"period": 10**6}), ("vwap", {"anchor": "rolling"}), ("vwap", {"anchor": "day", "period": 5}),
    ("relative_volume", {"period": True}), ("efficiency_ratio", {"period": -1}), ("donchian", {"source": "open"}),
    ("stochastic", {"smooth": 0}), ("stochastic", {"d_period": 1.5}), ("keltner", {"k": 0}), ("keltner", {"atr_period": 0}),
    ("rsi_divergence", {"right": 0}), ("rsi_divergence", {"rsi_period": 1}), ("rsi_divergence", {"left": 51}),
])
def test_bad_settings_are_refused(kind, settings):
    with pytest.raises(ValueError):
        make_block(kind, **settings)


@pytest.mark.parametrize("i", range(15))
def test_update_ohlcv_and_handle_bar_match_update_raw(i):
    """Any block can be fed whole bars, ignoring what it doesn't use, and reads as update_raw fed it."""
    block, feed = _all_blocks()[i]
    raw, whole_bar, from_bar = block, copy.deepcopy(block), copy.deepcopy(block)

    class _P:
        def __init__(self, x):
            self.x = x

        def as_double(self):
            return self.x

    class _Bar:
        def __init__(self, r):
            self.open, self.high, self.low = _P(r.close), _P(r.high), _P(r.low)
            self.close, self.volume, self.ts_event = _P(r.close), _P(r.volume), int(r.ts)

    for row in DATA["synthetic"].iloc[:2000].itertuples(index=False):
        feed(raw, row)
        whole_bar.update_ohlcv(row.close, row.high, row.low, row.close, row.volume, ts_ns=int(row.ts))
        if type(block) not in (Sma, AtrSma, Rsi):  # their handle_bar predates update_ohlcv and reads a real Bar
            from_bar.handle_bar(_Bar(row))
        assert whole_bar.values == raw.values
    if type(block) not in (Sma, AtrSma, Rsi):
        assert from_bar.values == raw.values


@pytest.mark.parametrize("i", range(15))
def test_values_are_none_until_initialized_and_never_nan(i):
    block, feed = _all_blocks()[i]
    assert set(block.values) == set(type(block).OUTPUTS) and all(v is None for v in block.values.values())
    for row in DATA["synthetic"].itertuples(index=False):
        feed(block, row)
        vals = block.values
        if block.initialized:
            assert all(v is not None and np.isfinite(v) for v in vals.values())
        else:
            assert all(v is None for v in vals.values())
            if type(block) not in (Sma, AtrSma, Rsi):  # their `value` is the one the hand-coded models trade
                assert block.value is None


@pytest.mark.parametrize("i", range(15))
def test_warmup_from_the_class_matches_the_block(i):
    block, _ = _all_blocks()[i]
    assert type(block).warmup_bars(**block.settings) == block.warmup_bars
    assert warmup_for([(k, s) for k, s in [(name, block.settings) for name, cls in BLOCKS.items()
                                            if cls is type(block)]]) == block.warmup_bars


def test_every_block_lists_settings_with_defaults_inside_their_limits():
    for kind, cls in BLOCKS.items():
        assert cls.SETTINGS, kind
        for s in cls.SETTINGS:
            if s.default is not None:
                assert s.check(s.default) == s.default
        assert make_block(kind, **({"anchor": "rolling", "period": 5} if kind == "vwap" else {})).initialized is False
        block = make_block(kind, **({"anchor": "rolling", "period": 5} if kind == "vwap" else {}))
        assert block.confirm_lag == (block.right if kind == "rsi_divergence" else 0)


def test_day_vwap_needs_close_times():
    with pytest.raises(ValueError, match="close time"):
        Vwap("day").update_raw(1, 1, 1, 1)


# ---- QA round on P1-3 (quant-review/v2-p1/indicators.md) ----------------------------------------------------

@pytest.mark.parametrize("i", range(15))
def test_settled_exactly_at_its_warmup(i):
    """F1: every block reads as settled from its warm-up on, and not before; library blocks are not initialized
    before it either. Sma, AtrSma and Rsi keep the `initialized` the hand-coded models trade on."""
    block, feed = _all_blocks()[i]
    for n, row in enumerate(DATA["synthetic"].iloc[: block.warmup_bars + 5].itertuples(index=False), start=1):
        feed(block, row)
        if n < block.warmup_bars:
            assert not block.settled
            if type(block) not in (Sma, AtrSma, Rsi, Vwap):  # a day VWAP is exact from a day start, before 1,441
                assert not block.initialized
    assert block.settled


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("i", range(15))
def test_non_finite_input_is_refused(i, bad):
    """F5: a NaN or infinite input raises rather than poisoning every later value."""
    block, feed = _all_blocks()[i]
    row = DATA["synthetic"].iloc[0]
    with pytest.raises(ValueError, match="finite"):
        block.update_ohlcv(bad, bad, bad, bad, bad, ts_ns=int(row.ts))
    feed(block, next(DATA["synthetic"].itertuples(index=False)))  # still usable afterwards


def test_negative_volume_and_a_rolling_vwap_with_no_period_are_refused():
    with pytest.raises(ValueError, match="negative"):
        RelativeVolume(5).update_raw(-1.0)
    with pytest.raises(ValueError, match="needs a period"):
        warmup_for([("vwap", {"anchor": "rolling"})])


def test_a_restart_on_its_warmup_reads_as_a_long_run_does():
    """F1/F2: started on warmup_bars of history, a block reads what one fed all history does (to the precision
    the warm-up was set for: a tenth of a basis point of price, a tenth of an RSI point)."""
    bars = DATA["synthetic"]
    for make, feed, tol in ((lambda: Ema(50), _close, 1e-5), (lambda: Keltner(20, 10), _hlc, 1e-5),
                            (lambda: Bollinger(20), _close, 1e-12), (lambda: Stochastic(), _hlc, 1e-12)):
        long_run, restart = make(), make()
        for row in bars.itertuples(index=False):
            feed(long_run, row)
        for row in bars.iloc[-restart.warmup_bars:].itertuples(index=False):
            feed(restart, row)
        assert restart.initialized
        for key, v in long_run.values.items():
            assert restart.values[key] == pytest.approx(v, rel=tol, abs=1e-9), (type(long_run).__name__, key)
