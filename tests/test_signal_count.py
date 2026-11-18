"""SIGNAL-COUNT (HoE 18 Nov): the dry run's train-window entry count per asset, for the Advisor's batch pre-checks
(at least 60 expected trades, at most half from one asset). It reads no bar on or after the first test day, runs
each asset alone, writes nothing and spends no try. Figures in fixtures are synthetic."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import rescore
from sleeve_fund.research.pinned import parse
from sleeve_fund.research.signal_count import (
    MAX_ASSET_SHARE,
    MIN_EXPECTED,
    AssetCount,
    SignalCount,
    count_signals,
    days_tested,
    tally,
    train_bars,
    train_window,
)

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "h0-fast-gate-list.md"
PINNED = parse(LIST.read_text())
PAIRS = "BTC/USDT,ETH/USDT,SOL/USDT"


def _bars(minutes=240, first="2024-01-01", last="2026-10-31", seed=0):
    """Synthetic candles indexed by close time, as the history store returns them, from the data start to past the
    holdout's end."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(first, tz="UTC") + pd.Timedelta(minutes=minutes), pd.Timestamp(last, tz="UTC"),
                        freq=f"{minutes}min")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx))))
    o = np.r_[c[0], c[:-1]]
    hi, lo = np.maximum(o, c) * 1.004, np.minimum(o, c) * 0.996
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1e9}, index=idx)


class _History:
    """The stored history, read-only: every read is recorded."""

    def __init__(self, bars):
        self.bars, self.reads = bars, []

    def read(self, venue, pair, minutes, start=None, end=None):
        self.reads.append((pair, minutes))
        return self.bars[pair]


def _request(pairs=PAIRS):
    name, minutes = "qrh0_trend_pullback", 240
    return rescore.request(name, minutes, "binance", pairs, PINNED)


def test_the_train_window_is_the_data_start_to_the_day_before_the_first_test_day():
    assert train_window(PINNED) == (date(2024, 1, 1), date(2024, 12, 31))
    assert days_tested(PINNED) == (date(2026, 4, 30) - date(2025, 1, 1)).days + 1 == 485


def test_no_bar_on_or_after_the_first_test_day_is_read():
    bars = _bars()
    kept = train_bars(bars, PINNED)
    assert kept.index[0] == pd.Timestamp("2024-01-01 04:00", tz="UTC")  # the first 4h candle of the data start
    assert kept.index[-1] == pd.Timestamp("2025-01-01 00:00", tz="UTC")  # the last 2024 candle, closing at midnight
    assert len(kept) == 366 * 6
    with pytest.raises(ValueError, match="not the train window 2024-01-01 to 2024-12-31"):
        train_bars(_bars(first="2024-03-01"), PINNED)


def _entry(ts, side="long", sized_by="1x leverage cap", factor=1.0):
    return {"intent": "entry", "signal": {"bar": {"close_ts": ts}, "rule": f"{side}.entry", "sized_by": sized_by,
                                          "vol_factor": factor}}


def _exit(ts, rule="long.exit"):
    return {"intent": "exit", "signal": {"rule": rule, "bars": {"4h": ["2024-01-01T04:00:00+00:00", ts]}}}


def test_each_asset_runs_alone_on_its_train_bars_and_its_entries_are_counted():
    bars = {p: _bars(seed=i) for i, p in enumerate(PAIRS.split(","))}
    seen = {}

    def backtest(strategy, prices, instrument, params, bar_minutes, half_spread):
        seen[instrument.id.symbol.value] = prices
        assert (float(instrument.maker_fee), float(instrument.taker_fee), half_spread) == (0, 0, 0)  # no fee or spread
        n = {"BTC": 30, "ETH": 12, "SOL": 5}[instrument.id.symbol.value.split("USDT")[0]]
        ds = {f"e{i}": _entry(f"2024-09-{1 + i % 28:02d}T12:00:00+00:00") for i in range(n)}
        return SimpleNamespace(decisions=ds | {"x": {"intent": "liquidation"}}, rule_counts={})

    req = _request()
    got = count_signals(req, history=_History(bars), backtest=backtest)
    assert got.entries == {"BTC/USDT": 30, "ETH/USDT": 12, "SOL/USDT": 5} and got.total == 47
    assert len(seen) == 3
    for prices in seen.values():
        assert prices.index[-1] < pd.Timestamp("2025-01-01 00:00:01", tz="UTC")
    # H0 decides after its daily SMA(200)'s warm-up: 201 days, so from 2024-07-20, 165 days of the train year
    assert got.decided_first == date(2024, 7, 20) and got.decided_days == 165
    assert got.expected == pytest.approx(47 * 485 / 165)
    lines = got.lines()
    assert lines[0] == ("train window 2024-01-01 to 2024-12-31 (366 days), decided from 2024-07-20 after its warm-up, "
                        "165 days; entries: BTC 30, ETH 12, SOL 5; 47 in all")
    assert lines[1] == "expected over the 485-day test window, pro-rated: 138.2"
    assert lines[2] == f"pre-check at least {MIN_EXPECTED} expected trades: PASS"
    assert lines[3] == (f"pre-check at most {MAX_ASSET_SHARE:.0%} from one asset: FAIL "
                        "(top BTC; shares BTC 64%, ETH 26%, SOL 11%)")
    assert lines[4] == ("first entries: 2024-09-01 12:00 UTC BTC long, 2024-09-01 12:00 UTC BTC long, "
                        "2024-09-01 12:00 UTC ETH long")
    assert lines[5] == "exit-rule exits on a later UTC day than their entry: 0 of 0"
    # no volatility target on H0: no sizing line; the fake runs filled nothing, so no closed trade to read
    assert lines[6] == ("gross edge, closed trades from the decided day (P&L before fees and funding over entry "
                        "notional, no spread; figures not printed: public log): BTC 0, ETH 0, SOL 0; 0 in all")
    assert lines[7] == "pre-check pooled gross edge at least 15 bps a trade: FAIL (no closed trade)"
    assert lines[8] == "pre-check gross edge positive on at least 2 of 3 assets: FAIL (positive: none)"
    assert len(lines) == 9


def _count(entries: dict, decided=date(2024, 1, 1), vol=None) -> SignalCount:
    assets = {p: AssetCount(entries=[(pd.Timestamp("2024-06-01 12:00", tz="UTC"), "long")] * n)
              for p, n in entries.items()}
    return SignalCount(train_first=date(2024, 1, 1), train_last=date(2024, 12, 31), decided_first=decided,
                       test_days=485, assets=assets, vol_target=vol)


@pytest.mark.parametrize("entries, enough, spread", [
    ({"BTC/USDT": 15, "ETH/USDT": 15, "SOL/USDT": 15}, "FAIL", "PASS"),  # 59.6 expected: short of 60
    ({"BTC/USDT": 16, "ETH/USDT": 15, "SOL/USDT": 15}, "PASS", "PASS"),  # 61.0
    ({"BTC/USDT": 23, "ETH/USDT": 23, "SOL/USDT": 0}, "PASS", "PASS"),  # exactly half from one asset is allowed
    ({"BTC/USDT": 24, "ETH/USDT": 22, "SOL/USDT": 0}, "PASS", "FAIL"),
    ({"BTC/USDT": 0, "ETH/USDT": 0, "SOL/USDT": 0}, "FAIL", "PASS"),
])
def test_the_pre_checks_read_the_pro_rated_count_and_the_top_assets_share(entries, enough, spread):
    got = _count(entries)
    assert got.lines()[2].endswith(f": {enough}")
    assert got.lines()[3].split("one asset: ")[1].startswith(spread)


def test_the_count_is_pro_rated_over_the_days_after_the_warm_up_only():
    """S4's day VWAP holds its first decision to the end of August: its months of entries stand for a year's rate."""
    late = _count({"BTC/USDT": 10, "ETH/USDT": 10, "SOL/USDT": 10}, decided=date(2024, 9, 1))
    assert late.decided_days == 122 and late.expected == pytest.approx(30 * 485 / 122)
    never = _count({"BTC/USDT": 0}, decided=None)
    assert never.expected == 0 and "its warm-up outlasts the train window" in never.lines()[0]


def test_an_exit_rule_exit_on_a_later_utc_day_is_counted_as_after_a_reset():
    """S4: the share of VWAP exits after the day VWAP restarted. A candle closing at 00:00 still belongs to the day
    before, as the VWAP reads it."""
    decisions = {"a": _entry("2024-06-01T16:00:00+00:00"), "b": _exit("2024-06-01T20:00:00+00:00"),
                 "c": _entry("2024-06-01T20:00:00+00:00"), "d": _exit("2024-06-02T00:00:00+00:00"),
                 "e": _entry("2024-06-02T16:00:00+00:00"), "f": _exit("2024-06-03T04:00:00+00:00"),
                 "g": _entry("2024-06-04T16:00:00+00:00"), "h": _exit("2024-06-05T00:00:00+00:00", "long.time_stop")}
    got = tally(decisions, vol_on=False)
    assert len(got.entries) == 4 and got.rule_exits == 3 and got.rule_exits_next_day == 1


def test_only_entries_from_the_first_counted_day_are_counted():
    """CR F1 on #368: a rule whose own blocks settle before the definition's longest warm-up can enter earlier; such an
    entry, and its exit, fall outside the days the count is pro-rated over."""
    decisions = {"a": _entry("2024-05-20T12:00:00+00:00"), "b": _exit("2024-05-21T04:00:00+00:00"),
                 "c": _entry("2024-05-21T04:00:00+00:00"), "d": _exit("2024-05-21T08:00:00+00:00")}
    got = tally(decisions, vol_on=False, decided_first=date(2024, 5, 21))
    assert [ts.isoformat() for ts, _ in got.entries] == ["2024-05-21T04:00:00+00:00"] and got.rule_exits == 1
    assert len(tally(decisions, vol_on=False).entries) == 2


def test_with_a_volatility_target_each_assets_median_size_and_share_capped_are_printed():
    from sleeve_fund.strategies.rules import VOL_SIZE_LIMIT

    btc = tally({"a": _entry("2024-06-01T12:00:00+00:00", factor=1.0),
                 "b": _entry("2024-06-02T12:00:00+00:00", sized_by=VOL_SIZE_LIMIT, factor=0.8),
                 "c": _entry("2024-06-03T12:00:00+00:00", factor=1.0)}, vol_on=True)
    sol = tally({"a": _entry("2024-06-01T12:00:00+00:00", "short", VOL_SIZE_LIMIT, 0.5),
                 "b": _entry("2024-06-02T12:00:00+00:00", "short", VOL_SIZE_LIMIT, 0.4)}, vol_on=True)
    sol.vol_skipped = 2
    got = SignalCount(train_first=date(2024, 1, 1), train_last=date(2024, 12, 31), decided_first=date(2024, 3, 1),
                      test_days=485, assets={"BTC/USDT": btc, "ETH/USDT": AssetCount(), "SOL/USDT": sol},
                      vol_target={"annual": 0.4, "lookback_days": 30})
    assert got.lines()[-4] == ("sizing to 40% a year over 30 days (size against the uncut size; capped = the target "
                               "didn't cut it): BTC median 1.00x, capped 67%, ETH no entries, SOL median 0.45x, "
                               "capped 0%; 2 entries skipped for want of the daily closes")
    assert "SOL short" in got.lines()[4]


def test_the_dry_run_prints_the_counts_and_runs_nothing(capfd):
    bars = {p: _bars(seed=i) for i, p in enumerate(PAIRS.split(","))}
    history = _History(bars)

    def no_run(*a, **k):
        raise AssertionError("a signal count runs no study")

    code = rescore.main([str(LIST), "--venue", "binance", "--pair", PAIRS, "--count-signals"], run=no_run,
                        journal=no_run, history=history)
    out = capfd.readouterr().out
    assert code == 0
    assert "row  1 qrh0_trend_pullback: train window 2024-01-01 to 2024-12-31 (366 days), decided from 2024-07-20 " in out
    assert "row  1 qrh0_trend_pullback: pre-check at least 60 expected trades: " in out
    assert "dry run: 1 checked, 0 run, train-window signals counted (no try)" in out
    assert {m for _, m in history.reads} == {240}  # the decision candles only: no minute bars


def test_without_the_flag_the_dry_run_is_as_before(capfd):
    def no_run(*a, **k):
        raise AssertionError("a dry run runs nothing")

    assert rescore.main([str(LIST), "--venue", "binance", "--pair", PAIRS, "--dry-run"], run=no_run,
                        journal=no_run, history=_History({})) == 0
    out = capfd.readouterr().out
    assert "train window" not in out and "pre-check" not in out
    assert out.rstrip().endswith("dry run: 1 checked, 0 run")


def test_the_workflow_passes_the_flag_only_when_asked():
    text = (ROOT / ".github" / "workflows" / "study.yml").read_text()
    assert "      count_signals:\n" in text and "default: false" in text.split("count_signals:")[1].split("configs_from")[0]
    assert "DRY: ${{ inputs.count_signals && '--count-signals' || (inputs.dry_run && '--dry-run' || '') }}" in text


def _gross(per_asset: dict) -> SignalCount:
    assets = {p: AssetCount(gross_bps=list(xs)) for p, xs in per_asset.items()}
    return SignalCount(train_first=date(2024, 1, 1), train_last=date(2024, 12, 31), decided_first=date(2024, 1, 1),
                       test_days=485, assets=assets)


@pytest.mark.parametrize("per_asset, pooled, spread", [
    ({"BTC/USDT": [15.0, 15.0], "ETH/USDT": [15.0], "SOL/USDT": [15.0]}, "PASS", "PASS"),  # exactly 15: enough
    ({"BTC/USDT": [14.9], "ETH/USDT": [15.0], "SOL/USDT": [15.0]}, "FAIL", "PASS"),  # 14.97 pooled
    ({"BTC/USDT": [90.0], "ETH/USDT": [-10.0], "SOL/USDT": [-10.0]}, "PASS", "FAIL"),  # one asset carries it
    ({"BTC/USDT": [40.0], "ETH/USDT": [0.0], "SOL/USDT": [10.0]}, "PASS", "PASS"),  # zero is not positive
    ({"BTC/USDT": [40.0], "ETH/USDT": [], "SOL/USDT": [-1.0]}, "PASS", "FAIL"),  # no closed trade is not positive
])
def test_the_gross_edge_pre_checks_read_the_pooled_mean_and_the_positive_assets(per_asset, pooled, spread):
    lines = _gross(per_asset).lines()
    assert f"pre-check pooled gross edge at least 15 bps a trade: {pooled}" in lines
    assert any(line.startswith(f"pre-check gross edge positive on at least 2 of 3 assets: {spread} ") for line in lines)


def test_no_edge_figure_is_printed():
    """The count prints to a public log: the checks' verdicts and trade counts only, never the bps themselves."""
    got = _gross({"BTC/USDT": [123.4, 56.7], "ETH/USDT": [-8.9], "SOL/USDT": [42.0]})
    text = "\n".join(got.lines())
    assert got.pooled_bps == pytest.approx((123.4 + 56.7 - 8.9 + 42.0) / 4)
    assert got.gross_bps("ETH/USDT") == pytest.approx(-8.9) and got.positive_assets == ["BTC/USDT", "SOL/USDT"]
    for figure in ("123", "56.7", "8.9", "42", "53.3"):
        assert figure not in text
    assert "BTC 2, ETH 1, SOL 1; 4 in all" in text


def _fills(rows):
    """A fills report as run_backtest gives it: side, quantity, price, fee and time, by order id."""
    return pd.DataFrame({"side": [r[0] for r in rows], "filled_qty": [r[1] for r in rows],
                         "avg_px": [r[2] for r in rows], "commissions": [0.0] * len(rows),
                         "ts_last": [pd.Timestamp(r[3], tz="UTC") for r in rows]},
                        index=[f"O-{i}" for i in range(len(rows))])


def test_gross_edge_is_each_closed_trades_pnl_over_its_entry_notional_from_the_decided_day():
    from sleeve_fund.research.signal_count import gross_bps

    fills = _fills([("BUY", 1, 100.0, "2024-02-01 04:00"), ("SELL", 1, 101.0, "2024-02-02 04:00"),  # before: out
                    ("BUY", 1, 100.0, "2024-07-20 04:00"), ("SELL", 1, 100.5, "2024-07-21 04:00"),  # +50 bps
                    ("SELL", 2, 200.0, "2024-08-01 04:00"), ("BUY", 2, 201.0, "2024-08-02 04:00"),  # short: -50 bps
                    ("BUY", 1, 100.0, "2024-12-31 20:00")])  # still open at the window's end: out
    assert gross_bps(fills, date(2024, 7, 20)) == pytest.approx([50.0, -50.0])
    assert gross_bps(fills, None) == [] and gross_bps(None, date(2024, 1, 1)) == []
