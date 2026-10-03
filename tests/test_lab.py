import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd

from sleeve_fund.lab import blocks, classifier_test, data


def _zip(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("k.csv", "\n".join(",".join(map(str, r)) for r in rows))
    return buf.getvalue()


def test_archive_parse_handles_milliseconds_and_microseconds_and_updates_incrementally(tmp_path):
    jan = [[1735689600000000 + i * 60_000_000, 1, 2, 0.5, 1.5, 10, 0, 0, 0, 0, 0, 0] for i in range(3)]  # 2025, µs
    dec = [[1733011200000 + i * 60_000, 1, 2, 0.5, 1.5, 10, 0, 0, 0, 0, 0, 0] for i in range(2)]  # 2024, ms
    calls = []

    def get(url):
        calls.append(url)
        if "2024-12.zip" in url:
            return _zip(dec)
        if "2025-01.zip" in url:
            return _zip(jan)
        return None

    data.update("BTC/USD", tmp_path, today=date(2025, 2, 1), get=get, log=lambda m: None)
    df = data.load("BTC/USD", tmp_path)
    assert len(df) == 5 and str(df.index[0]) == "2024-12-01 00:00:00+00:00" and df.index.is_monotonic_increasing
    calls.clear()
    data.update("BTC/USD", tmp_path, today=date(2025, 2, 1), get=get, log=lambda m: None)
    assert len(data.load("BTC/USD", tmp_path)) == 5  # no duplicates on a re-run
    assert all("2017" not in c for c in calls)  # only the last stored month onwards is fetched again


def _walk(days=40, seed=3):
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    idx = pd.date_range("2024-03-01", periods=n, freq="1min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.0007, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0002, "low": np.minimum(o, c) * 0.9998, "close": c,
                         "volume": rng.lognormal(0, 1, n)}, index=idx)


def test_blocks_and_classifier_use_no_future_bars():
    b = data.resample(_walk(), 5)
    full = blocks.with_session_blocks(b, "us_open")
    labels = blocks.classify(full)
    cut = b.index[len(b) // 2 + 41]
    part = blocks.with_session_blocks(b[b.index <= cut], "us_open")
    for col in ("vwap", "vwap_sd", "relvol", "atr_d", "or_high", "or_low", "above_share", "vwap_crosses"):
        assert np.allclose(full.loc[part.index, col], part[col], equal_nan=True), col
    assert (labels.loc[part.index] == blocks.classify(part)).all()


def test_session_vwap_matches_its_definition():
    b = data.resample(_walk(5), 15)
    out = blocks.with_session_blocks(b, "utc")
    day = out[out["session"] == out["session"].iloc[200]]
    tp = (day["high"] + day["low"] + day["close"]) / 3
    expect = (tp * day["volume"]).cumsum() / day["volume"].cumsum()
    assert np.allclose(day["vwap"], expect)
    assert (out.groupby("session")["elapsed"].first() == 15).all()


def test_a_clean_trend_day_is_classified_trend_and_continues():
    b = data.resample(_walk(30), 5)
    # Make the last full session a strong, steady up-move on rising volume.
    out = blocks.with_session_blocks(b, "utc")
    last = out["session"].unique()[-2]
    m = (out["session"] == last).to_numpy()
    k = np.arange(m.sum())
    base = b.loc[m, "open"].iloc[0]
    b.loc[m, "close"] = base * (1 + 0.0015 * k)
    b.loc[m, "open"] = b.loc[m, "close"] / 1.0005
    b.loc[m, "high"] = b.loc[m, "close"] * 1.0003
    b.loc[m, "low"] = b.loc[m, "open"] * 0.9999
    b.loc[m, "volume"] *= 3
    so = classifier_test.session_outcomes(b, anchor="utc", decide_at=60)
    row = so[so["session"] == last].iloc[0]
    assert row["label"] == blocks.TREND_UP and row["cont_session"] > 0
