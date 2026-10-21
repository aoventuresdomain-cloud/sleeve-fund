"""(d) FINALIST-CORR (HoE 21 Oct, the Advisor's spec): the correlation of the finalists' OUT-OF-SAMPLE daily returns,
net of costs, pair by pair over the calendar days both have (joined on the date, never by position), each with its
day count, blank under 60 days; a day out of the market is a 0 return, never a gap; and each finalist's correlation
to the plain BTC price return beside it. Expected values are worked out here from the fixtures, not read from the
code under test."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import fund_summary as fs
from sleeve_fund.research.run import oos_returns_path, write_oos_returns


def _days(n, start="2025-01-01"):
    return pd.date_range(start, periods=n, freq="1D")


def _walk(n, seed=0):
    return np.random.default_rng(seed).normal(0, 0.02, n)


def test_identical_series_correlate_1_and_a_series_and_its_negation_minus_1():
    x = pd.Series(_walk(100), index=_days(100))
    corr, days = fs.pairwise({"a": x, "b": x.copy(), "c": -x})
    assert corr.loc["a", "b"] == pytest.approx(1.0) and corr.loc["a", "c"] == pytest.approx(-1.0)
    assert days.loc["a", "b"] == 100 and days.loc["b", "c"] == 100


def test_misaligned_dates_are_joined_on_the_calendar_date_not_by_position():
    x = _walk(120, seed=1)
    a = pd.Series(x[:100], index=_days(120)[:100])  # 1 Jan to 10 Apr
    b = pd.Series(x[20:], index=_days(120)[20:])  # 21 Jan to 30 Apr: the same values on the same dates
    corr, days = fs.pairwise({"a": a, "b": b})
    assert days.loc["a", "b"] == 80  # 21 Jan to 10 Apr
    assert corr.loc["a", "b"] == pytest.approx(1.0)
    assert abs(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1]) < 0.5  # set-up: by position they would not agree


def test_a_pair_under_60_shared_days_is_blank_with_its_count_shown():
    x = pd.Series(_walk(100, seed=2), index=_days(100))
    corr, days = fs.pairwise({"a": x.iloc[:70], "b": x.iloc[11:]})  # 59 shared days
    assert days.loc["a", "b"] == 59 and np.isnan(corr.loc["a", "b"])
    corr, days = fs.pairwise({"a": x.iloc[:70], "b": x.iloc[10:]})  # 60: shown
    assert days.loc["a", "b"] == 60 and corr.loc["a", "b"] == pytest.approx(1.0)


def test_a_day_out_of_the_market_is_a_zero_return_counted_in_the_pair():
    x = _walk(80, seed=3)
    flat = x.copy()
    flat[::2] = 0.0  # out of the market every other day
    a, b = pd.Series(x, index=_days(80)), pd.Series(flat, index=_days(80))
    corr, days = fs.pairwise({"a": a, "b": b})
    assert days.loc["a", "b"] == 80
    assert corr.loc["a", "b"] == pytest.approx(np.corrcoef(x, flat)[0, 1])  # the zeros are in it


def test_each_finalist_s_correlation_to_the_plain_btc_price_return():
    btc = pd.Series(_walk(90, seed=4), index=_days(90))
    corr, days = fs.to_price({"long": 2 * btc, "short": -btc.iloc[10:], "brief": btc.iloc[:30]}, btc)
    assert corr["long"] == pytest.approx(1.0) and days["long"] == 90
    assert corr["short"] == pytest.approx(-1.0) and days["short"] == 80
    assert np.isnan(corr["brief"]) and days["brief"] == 30


def _result(oos, closes, research_end):
    return SimpleNamespace(oos_returns=oos, research_closes=closes, research_end=research_end)


def test_the_out_of_sample_file_holds_the_strategy_and_its_price_each_test_day(tmp_path):
    idx = pd.date_range("2026-01-01", periods=6, freq="1D", tz="UTC")
    closes = pd.Series([100.0, 110.0, 99.0, 99.0, 108.9, 50.0], index=idx)
    oos = pd.Series([0.0, 0.05, -0.02], index=idx[2:5])  # the test windows' days only, a research day after them
    write_oos_returns(_result(oos, closes, idx[-1]), tmp_path / "s.md")
    got = pd.read_csv(oos_returns_path(tmp_path / "s.md"), index_col="date", parse_dates=["date"])
    assert list(got.index.strftime("%Y-%m-%d")) == ["2026-01-03", "2026-01-04", "2026-01-05"]
    assert got["return"].tolist() == [0.0, 0.05, -0.02]
    assert got["price"].tolist() == pytest.approx([99 / 110 - 1, 0.0, 108.9 / 99 - 1])


def test_an_out_of_sample_day_past_the_research_period_is_refused_and_nothing_written(tmp_path):
    idx = pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC")
    r = _result(pd.Series([0.01, 0.02], index=idx[2:]), pd.Series(100.0, index=idx), idx[2])
    with pytest.raises(ValueError, match="holdout"):
        write_oos_returns(r, tmp_path / "s.md")
    assert not oos_returns_path(tmp_path / "s.md").exists()


def _sheet(d, name, dataset, oos, price=None):
    path = d / f"{name}.md"
    path.write_text(f"# {name}\n\nDataset `{dataset}` · research period\n", encoding="utf-8")
    price = oos * 0 if price is None else price
    pd.DataFrame({"return": oos, "price": price}, index=pd.Index(oos.index, name="date")).to_csv(
        oos_returns_path(path), date_format="%Y-%m-%d")
    return path


END = pd.Timestamp("2025-06-01", tz="UTC")


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """120 days of BTC/USDT minute bars on BINANCE in a history store, to 1 Jun 2025, and BTC's daily return each day
    worked out here from the minutes to END: a day stamped D closes on the minute opening at D - 1 minute."""
    from sleeve_fund.history import HistoryStore

    # a day of minutes past END: the store keeps its last stored minute as forming, so END's is the last whole day
    idx = pd.date_range(END - pd.Timedelta(days=120), END + pd.Timedelta(days=1, minutes=-1), freq="1min", tz="UTC")
    c = 30_000 * np.exp(np.cumsum(np.random.default_rng(8).normal(0, 0.0008, len(idx))))
    o = np.r_[c[0], c[:-1]]
    minutes = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 5.0},
                           index=idx)
    hist = HistoryStore(tmp_path_factory.mktemp("hist"))
    hist.append("BINANCE", "BTC/USDT", minutes, cursor="x")
    stamps = pd.date_range(END - pd.Timedelta(days=119), END, freq="1D")
    close = pd.Series(minutes["close"].reindex(stamps - pd.Timedelta(minutes=1)).to_numpy(), index=stamps)
    want = (close / close.shift(1) - 1).dropna()
    want.index = want.index.tz_localize(None)
    return hist, want


def test_btc_s_price_return_is_read_from_the_store_for_the_same_days(stored):
    hist, want = stored
    days = want.index[-90:]
    got, why = fs.btc_returns(hist, "BINANCE", "BTC/USDT", days)
    assert why == "" and list(got.index) == list(days)
    assert got.tolist() == pytest.approx(want.reindex(days).tolist())


def test_btc_reads_n_a_with_its_reason_when_the_store_does_not_cover_the_days(stored):
    hist, want = stored
    days = pd.date_range(want.index[-30], periods=40, freq="1D")  # ten days past the store's end
    got, why = fs.btc_returns(hist, "BINANCE", "BTC/USDT", days)
    assert got is None and "10 of the 40" in why and "BINANCE" in why
    got, why = fs.btc_returns(hist, "KRAKEN", "BTC/USD", days)
    assert got is None and "no stored history" in why


def test_the_section_shows_each_pair_with_its_days_and_btc_from_the_store(tmp_path, stored):
    hist, want = stored
    days = want.index[-100:]
    eth = pd.Series(_walk(100, seed=6), index=days)
    trend = 2 * want.reindex(days) + 0.001
    a = _sheet(tmp_path, "trend_btc", "binance-btcusdt-store-240m", trend)
    b = _sheet(tmp_path, "dip_eth", "binance-ethusdt-store", -eth.iloc[30:])
    text = "\n".join(fs.oos_section([b, a], hist))  # the ETH sheet first
    pair = np.corrcoef(trend.iloc[30:], -eth.iloc[30:])[0, 1]
    assert f"| 1 dip_eth | 1.00 (70 d) | {pair:.2f} (70 d) | " in text
    to_btc = np.corrcoef(-eth.iloc[30:], want.reindex(days).iloc[30:])[0, 1]
    assert text.splitlines()[[i for i, x in enumerate(text.splitlines()) if x.startswith("| 1 dip_eth")][0]].endswith(
        f"| {to_btc:.2f} (70 d) |")
    assert "| 2 trend_btc | " in text and "| 1.00 (100 d) | 1.00 (100 d) |" in text
    assert "out of sample only" in text.lower() and "blank under 60" in text
    assert "BTC/USDT on BINANCE" in text  # where the BTC column came from


def test_the_btc_column_reads_n_a_with_the_reason_when_the_store_falls_short(tmp_path, stored):
    hist, want = stored
    days = pd.date_range(want.index[-70], periods=80, freq="1D")  # past the store's end
    a = _sheet(tmp_path, "trend_btc", "binance-btcusdt-store", pd.Series(_walk(80, seed=9), index=days))
    text = "\n".join(fs.oos_section([a], hist))
    assert text.rstrip().splitlines()[-1].endswith("| n/a |")
    assert "BTC column: n/a" in text and "10 of the 80" in text
    text = "\n".join(fs.oos_section([a], None))
    assert "BTC column: n/a" in text and "no stored history" in text


def test_the_section_says_which_sheets_have_no_out_of_sample_file(tmp_path):
    btc = pd.Series(_walk(70, seed=7), index=_days(70))
    a = _sheet(tmp_path, "trend_btc", "binance-btcusdt-store", btc, btc)
    old = tmp_path / "old_run.md"
    old.write_text("# old\n", encoding="utf-8")
    text = "\n".join(fs.oos_section([a, old], None))
    assert "old_run.md" in text and "re-run" in text.lower()


def test_a_real_study_writes_its_test_windows_with_a_price_on_every_day(tmp_path):
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC
    from sleeve_fund.venues import venue

    prices = synthetic_ohlcv(days=900, seed=11)
    r = run_study(SPEC, prices, venue("KRAKEN").instrument("BTC", "USD"), dataset="syn",
                  ledger=IdeaLedger(tmp_path / "l.jsonl"), synthetic=True, holdout_days=0, train_days=300,
                  test_days=200)
    got = pd.read_csv(write_oos_returns(r, tmp_path / "s.md"), index_col="date", parse_dates=["date"],
                      float_precision="round_trip")  # repr in the file: the float itself
    assert len(got) == len(r.oos_returns) > 300  # set-up: about 600 test days
    assert got["return"].tolist() == r.oos_returns.tolist()
    close = prices["close"].astype(float)
    close.index = close.index.tz_localize(None) if close.index.tz is not None else close.index
    want = (close / close.shift(1) - 1).reindex(got.index)
    assert got["price"].notna().all() and got["price"].tolist() == pytest.approx(want.tolist())


def test_sheets_naming_two_venues_read_n_a_unless_one_is_given(tmp_path, stored):
    hist, want = stored
    days = want.index[-80:]
    a = _sheet(tmp_path, "trend_btc", "binance-btcusdt-store", pd.Series(_walk(80, seed=10), index=days))
    b = _sheet(tmp_path, "trend_eth", "kraken-ethusd-store", pd.Series(_walk(80, seed=11), index=days))
    text = "\n".join(fs.oos_section([a, b], hist))
    assert "BTC column: n/a, the sheets don't name one venue's stored history" in text
    text = "\n".join(fs.oos_section([a, b], hist, "BINANCE"))
    assert "BTC: BTC/USDT on BINANCE" in text and "| n/a |" not in text


def test_empty_out_of_sample_files_read_n_a_not_an_error(tmp_path, stored):
    hist, _ = stored
    got, why = fs.btc_returns(hist, "BINANCE", "BTC/USDT", pd.DatetimeIndex([]))
    assert got is None and "no days" in why


def test_a_store_study_writes_its_out_of_sample_file_beside_the_sheet(tmp_path):
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.research.run import StudyRequest, run_store_study

    idx = pd.date_range("2020-01-01", periods=191 * 24, freq="60min", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.004, len(idx))))
    hourly = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    hist = HistoryStore(tmp_path / "hist")
    hist.append("KRAKEN", "BTC/USD", hourly.resample("1min").ffill(), cursor="x")
    sheet = run_store_study(StudyRequest("buy_and_hold", "BTC/USD", venue="KRAKEN", holdout_days=100, train_days=60,
                                         test_days=30), ledger_path=tmp_path / "l.jsonl", out_dir=tmp_path, history=hist)
    got = pd.read_csv(oos_returns_path(sheet), index_col="date", parse_dates=["date"])
    assert len(got) >= 20 and got["price"].notna().all()  # set-up: a 30-day test window inside 90 research days


def test_two_sheets_with_the_same_name_are_refused_in_words(tmp_path):
    days = _days(70)
    (tmp_path / "btc").mkdir()
    (tmp_path / "eth").mkdir()
    a = _sheet(tmp_path / "btc", "trend", "binance-btcusdt-store", pd.Series(_walk(70, seed=12), index=days))
    b = _sheet(tmp_path / "eth", "trend", "binance-ethusdt-store", pd.Series(_walk(70, seed=13), index=days))
    with pytest.raises(ValueError, match="trend.*share one name"):
        fs.oos_section([a, b], None)
    with pytest.raises(ValueError, match="trend.*share one name"):
        fs.page([a, b])
