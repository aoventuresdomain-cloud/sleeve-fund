"""Download the public BTCUSDT perpetual 1-minute klines (monthly files) up to the end of development, as one CSV
stamped at each minute's CLOSE, as the history store reads bars. The holdout months are never downloaded."""
import io
import sys
import urllib.request
import zipfile

import pandas as pd

URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{y}-{m:02d}.zip"


def month(y, m):
    raw = urllib.request.urlopen(URL.format(y=y, m=m), timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(df.iloc[0, 0]).isdigit():  # a header row
        df = df.iloc[1:]
    t = df[0].astype("int64")
    unit = "us" if t.iloc[0] > 10**14 else "ms"
    idx = pd.to_datetime(t, unit=unit, utc=True) + pd.Timedelta(minutes=1)
    out = pd.DataFrame({"open": df[1].astype(float).values, "high": df[2].astype(float).values,
                        "low": df[3].astype(float).values, "close": df[4].astype(float).values,
                        "volume": df[5].astype(float).values}, index=idx.values)
    out.index.name = "timestamp"
    return out


def main(start, end, path):
    frames = []
    for p in pd.period_range(start, end, freq="M"):
        frames.append(month(p.year, p.month))
        print(p, len(frames[-1]), flush=True)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()]
    df.to_csv(path)
    print("rows", len(df), df.index[0], df.index[-1])


if __name__ == "__main__":
    main(*sys.argv[1:4])
