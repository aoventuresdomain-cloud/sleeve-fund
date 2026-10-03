import pandas as pd
import pytest

from sleeve_fund.data import load_kraken_ohlcvt, synthetic_ohlcv, validate_ohlcv


def test_kraken_bars_are_stamped_at_close(tmp_path):
    # Kraken timestamps are bar OPEN times; a daily bar opening 1 Jan is known at 2 Jan 00:00.
    csv = tmp_path / "XBTUSD_1440.csv"
    csv.write_text("1704067200,100,110,90,105,12.5,40\n1704153600,105,120,100,118,9.0,30\n")
    df = load_kraken_ohlcvt(csv)
    assert df.index[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.assign(close=d["close"].where(d.index != d.index[3], -1.0)),
        lambda d: d.assign(high=d["low"] * 0.5),
        lambda d: pd.concat([d, d.iloc[[5]]]),
        lambda d: d.assign(open=d["open"].where(d.index != d.index[2], float("nan"))),
    ],
    ids=["negative", "high_below_low", "duplicate", "nan"],
)
def test_bad_data_is_rejected(mutate):
    with pytest.raises(ValueError):
        validate_ohlcv(mutate(synthetic_ohlcv(days=20)))
