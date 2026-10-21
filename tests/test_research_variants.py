"""R-06 and R-07 (HoE 21 Oct; the PM approved the research round): the funding rate decile block and sizing to a target
volatility, for the pre-registered trend-long variants V1-V4 (research/quant-researcher/trend-variants-preregistration.md).
Studies only: paper and live refuse both on Start until the PM approves."""

import math
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleeve_fund import funding
from sleeve_fund.research.runner import FUNDING_BLOCK_NEEDS, run_backtest
from sleeve_fund.research.variants import PREREGISTERED, VOL_TARGET
from sleeve_fund.strategies import check_study_only, study_only_uses
from sleeve_fund.strategies.definitions import (DAY_NS, FUNDING_COVERAGE, FundingDecile, check_definition,
                                                funding_decile_of, to_params)
from sleeve_fund.strategies.rules import VOL_SIZE_LIMIT, realised_vol
from sleeve_fund.venues import venue
from test_dashboard import client  # noqa: F401 - a fixture

H = 3_600_000_000_000
EIGHT = 8 * H
DEFS = Path(__file__).resolve().parent.parent / "configs" / "definitions"


def _block(times, rates, window_days=7, interval=EIGHT):
    b = FundingDecile(window_days=window_days)
    b.source = (np.asarray(times, dtype=np.int64), np.asarray(rates, dtype=float), interval)
    return b


def _settlements(days=10, start=0):
    return [start + i * EIGHT for i in range(days * 3 + 1)]


# ---- the decile ------------------------------------------------------------------------------------------------------

def test_the_decile_is_one_more_than_ten_times_the_share_below_and_the_highest_reads_ten():
    w = list(range(1, 11))  # 1..10
    assert [funding_decile_of(w, x) for x in w] == list(range(1, 11))
    assert funding_decile_of([5.0] * 20, 5.0) == 1  # ties take the lower decile
    assert funding_decile_of(list(range(100)), 99) == 10 and funding_decile_of(list(range(100)), 90) == 10
    assert funding_decile_of(list(range(100)), 89) == 9


def test_no_look_ahead_a_settlement_after_the_close_is_never_read():
    """Red first: at a close between settlements, the next one (extreme) must not move the value."""
    times = _settlements()
    rates = [0.0001 * (i % 5) for i in range(len(times))]  # the latest at the close is mid-range
    close = times[-2] + 4 * H  # 4 h after the second-last settlement, 4 h before the last
    seen = _block(times[:-1], rates[:-1])
    seen.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=close)
    future = _block(times, rates[:-1] + [0.05])  # the next settlement, after the close, is the highest ever
    future.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=close)
    assert seen.value is not None and future.value == seen.value
    assert future.value < 10


def test_a_settlement_stamped_at_the_close_is_not_read_on_it():
    """Red first (Advisor, 21 Oct ~10:20 UK): only settlements strictly before the close. An 08:00 decision reads the
    00:00 settlement, never the 08:00 one, stamped at the close or a few seconds either side of it."""
    times = _settlements()
    rates = [0.0001 * (i % 5) for i in range(len(times) - 1)] + [0.05]  # the one at the close is the highest ever
    before = _block(times[:-1], rates[:-1])
    before.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1])
    assert before.value is not None and before.value < 10
    for stamp in (0, -30_000_000_000, 30_000_000_000):  # at the close, 30 s early, 30 s late
        b = _block(times[:-1] + [times[-1] + stamp], rates)
        b.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1])
        assert b.value == before.value, stamp
    for after in (4 * H, EIGHT):  # a later close, mid-interval or on the next settlement, reads it
        later = _block(times, rates)
        later.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1] + after)
        assert later.value == 10


def test_the_window_is_the_trailing_days_only_never_the_full_sample():
    """A spike older than the window no longer counts: the latest is again the window's highest."""
    times = _settlements(days=20)
    rates = [0.0001] * len(times)
    rates[:10] = [0.05] * 10  # over 16 days before the end, outside a 7-day window: on the full sample, decile 9
    rates[-1] = 0.0002
    b = _block(times, rates, window_days=7)
    b.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1] + 4 * H)
    assert b.value == 10
    wide = _block(times, rates, window_days=30)
    wide.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1] + 4 * H)
    assert wide.value is None  # a 30-day window over 20 days of settlements is under FUNDING_COVERAGE


def test_with_no_stored_rate_it_gives_no_value_and_counts_the_candle():
    times = _settlements()
    rates = [0.0001] * len(times)
    missing = _block(times[:-2] + times[-1:], rates[:-1])  # the settlement due before the close isn't stored
    missing.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1])
    assert missing.value is None and missing.no_rate == 1
    sparse = _block(times[::2], rates[::2])  # half the window's settlements: under the coverage it needs
    assert FUNDING_COVERAGE > 0.5
    sparse.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1] + 4 * H)  # the latest is there: only too few
    assert sparse.value is None and sparse.no_rate == 1
    paper = FundingDecile(window_days=7)  # never handed a source: it never reads a value
    paper.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1])
    assert paper.value is None and paper.no_rate == 1


def test_only_the_top_decile_is_counted_as_top():
    times = _settlements()[:-1]
    nine = [0.0] * (len(times) - 3) + [1.0, 1.0, 0.5]  # two of the window above the latest: under 90% below it
    ten = [0.0] * (len(times) - 1) + [0.5]
    for rates, value, top in ((nine, 9, 0), (ten, 10, 1)):
        b = _block(times, rates)
        b.update_ohlcv(0, 0, 0, 1.0, 0, ts_ns=times[-1] + 4 * H)
        assert (b.value, b.top_decile) == (value, top)


def test_the_block_takes_no_timeframe():
    defn = {"version": 1, "reason": "a fixture", "blocks": {"f": {"kind": "funding_decile", "timeframe": "1d"}},
            "long": {"entry": {"left": "f", "op": "<", "right": 10}, "exit": {"left": "f", "op": ">=", "right": 10}}}
    with pytest.raises(ValueError, match="takes no timeframe"):
        check_definition(defn)


# ---- end to end: the gate in a backtest -------------------------------------------------------------------------------

GATED = {"version": 1, "reason": "a fixture", "blocks": {"avg": {"kind": "sma", "period": 3}, "fund": {"kind": "funding_decile",
                                                                                 "window_days": 7}},
         "long": {"entry": {"all": [{"left": "close", "op": ">", "right": "avg"},
                                    {"left": "fund", "op": "<", "right": 10}]},
                  "exit": {"left": "close", "op": "<", "right": "avg"}}}
PERP = {"market": "perp"}


def _hourly(days=14):
    idx = pd.date_range("2026-01-01 01:00", periods=days * 24, freq="1h", tz="UTC")
    c = 60_000.0 * (1 + 0.02 * np.sin(np.arange(len(idx)) / 6))
    c = np.round(c, 1)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1e9},
                        index=idx)


def _gated(monkeypatch, rates):
    monkeypatch.setattr(funding, "rates", lambda v, pair, root=None: rates)
    bars = _hourly()
    return run_backtest("rules", bars, venue("BINANCE").instrument("BTC", "USDT", price_precision=1),
                        params={**to_params(GATED), **PERP}, bar_minutes=60, half_spread=0, starting_capital=10_000)


def _rates(values):
    idx = pd.date_range("2025-12-20", periods=len(values), freq="8h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def _entries(res):
    return [d for d in res.decisions.values() if d.get("intent") == "entry"]


def test_a_backtest_opens_while_the_rate_is_low_and_never_while_it_is_in_its_top_decile(monkeypatch):
    n = 4 * 3 * 14
    flat = _gated(monkeypatch, _rates([0.0001] * n))  # every rate equal: the latest is in the lowest decile
    rising = _gated(monkeypatch, _rates([0.0001 * (1 + i / n) for i in range(n)]))  # the latest is always the highest
    assert _entries(flat) and not _entries(rising)
    assert flat.rule_counts["funding_no_rate"] == 0


def test_a_backtest_with_no_stored_rate_opens_nothing_and_the_sheet_counts_the_candles(monkeypatch):
    res = _gated(monkeypatch, pd.Series(dtype=float))
    assert not _entries(res)
    assert res.rule_counts["funding_no_rate"] == len(_hourly())
    assert res.rule_counts["funding_top_decile"] == 0


def test_candles_in_the_top_decile_are_counted_apart_from_those_with_no_rate(monkeypatch):
    """Advisor (21 Oct ~10:20 UK): the sheet reports the two reasons an entry is blocked separately."""
    n = 4 * 3 * 14
    rising = _gated(monkeypatch, _rates([0.0001 * (1 + i / n) for i in range(n)]))
    flat = _gated(monkeypatch, _rates([0.0001] * n))
    assert rising.rule_counts["funding_no_rate"] == 0 and rising.rule_counts["funding_top_decile"] == len(_hourly())
    assert flat.rule_counts == {"funding_no_rate": 0, "funding_top_decile": 0}


def test_a_simulated_perp_or_spot_is_refused_up_front():
    bars = _hourly()
    with pytest.raises(ValueError, match="funding rate decile block reads the venue's stored"):
        run_backtest("rules", bars, venue("KRAKEN").instrument("BTC", "USD", price_precision=1),
                     params=to_params(GATED), bar_minutes=60, half_spread=0)
    assert "perpetual" in FUNDING_BLOCK_NEEDS


# ---- volatility-targeted sizing ---------------------------------------------------------------------------------------

def test_realised_vol_of_alternating_returns_is_known():
    r = 0.03
    closes = [100.0]
    for i in range(30):
        closes.append(closes[-1] * math.exp(r if i % 2 == 0 else -r))
    assert realised_vol(closes) == pytest.approx(r * math.sqrt(30 / 29 * 365), rel=1e-12)
    assert realised_vol(closes[:2]) is None  # one return can't give a sample deviation


def _daily(r, days=60, tail=True):
    idx = pd.date_range("2026-01-02", periods=days, freq="1D", tz="UTC")
    lr = np.array([r if i % 2 == 0 else -r for i in range(days)])
    if tail:
        lr[-12:] = [0.02, 0.02, 0.02, -0.001, -0.02, -0.02, -0.02, 0.02, 0.02, 0.02, 0.02, 0.02]  # a dip, then a rise
    c = np.round(60_000.0 * np.exp(np.cumsum(lr)), 1)  # at the instrument's 0.1 tick, as the strategy reads them
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1e12},
                        index=idx)


TREND = {"version": 1, "reason": "a fixture", "blocks": {"avg": {"kind": "sma", "period": 3}},
         "long": {"entry": {"left": "close", "op": "crosses_above", "right": "avg"},
                  "exit": {"left": "close", "op": "<", "right": "avg"}}}


def _sized(bars, vol=None):
    params = {**to_params(TREND), **({"vol_target_size": vol} if vol else {})}
    return run_backtest("rules", bars, venue("KRAKEN").instrument("BTC", "USD", price_precision=1), params=params,
                        bar_minutes=1440, half_spread=0, starting_capital=10_000)


def _first_entry(res) -> dict:
    return next(d["signal"] for d in res.decisions.values() if d.get("intent") == "entry")


def test_an_entry_is_sized_to_the_target_over_the_realised_volatility():
    """Red first: on a known-volatility fixture the entry's budget is the fund's equity x 40% / that volatility."""
    bars = _daily(0.03)
    sig = _first_entry(_sized(bars, VOL_TARGET))
    assert sig["sized_by"] == VOL_SIZE_LIMIT
    # The entry candle's own close and the 30 before it: every daily close at or before the decision, none after.
    k = bars.index.get_loc(pd.Timestamp(sig["bar"]["close_ts"]))
    vol = realised_vol(list(bars["close"].iloc[k - 30:k + 1]))
    assert vol == pytest.approx(0.03 * math.sqrt(30 / 29 * 365), rel=1e-4)  # the fixture's own (to the tick)
    assert sig["budget"] == pytest.approx(10_000 * 0.40 / vol, abs=0.01)
    assert "against a 40% target" in sig["vol_target"]


def test_the_target_never_sizes_above_todays_size():
    """A calm fixture: 40% over its volatility is several times the capital, so today's limits size it, unchanged."""
    bars = _daily(0.002)
    on, off = _sized(bars, VOL_TARGET), _sized(bars)
    assert _first_entry(on)["sized_by"] == _first_entry(off)["sized_by"] != VOL_SIZE_LIMIT
    assert _first_entry(on)["budget"] == _first_entry(off)["budget"]
    assert on.fills["quantity"].iloc[0] == off.fills["quantity"].iloc[0]


ALTERNATING = math.sqrt(30 / 29 * 365)  # an alternating +-r daily return's annual volatility over 30 days, per r


def test_at_80_percent_volatility_the_entry_is_half_the_fund_at_30_percent_todays_size_never_more():
    """Advisor cells (21 Oct ~10:20 UK): sigma 80% against the 40% target sizes at 0.5x the fund's equity, under today's
    size; sigma 30% would allow 1.33x, so today's size stands, unchanged."""
    high = _first_entry(_sized(_daily(0.80 / ALTERNATING, tail=False), VOL_TARGET))
    today = _first_entry(_sized(_daily(0.80 / ALTERNATING, tail=False)))["budget"]  # its first entry, at 10,000 too
    vol = 0.40 * 10_000 / high["budget"]
    assert vol == pytest.approx(0.80, abs=0.002) and high["sized_by"] == VOL_SIZE_LIMIT
    assert high["budget"] == pytest.approx(0.5 * 10_000, rel=0.003)
    assert high["vol_factor"] == pytest.approx(high["budget"] / today, abs=1e-6) and high["vol_factor"] < 1
    perp = venue("BINANCE").instrument("BTC", "USDT", price_precision=1)  # the perpetual's own sizing path
    on, off = ({**to_params(TREND), **PERP, **v} for v in ({"vol_target_size": VOL_TARGET}, {}))
    kw = dict(bar_minutes=1440, half_spread=0, starting_capital=10_000)
    bars = _daily(0.80 / ALTERNATING, tail=False)
    p_high = _first_entry(run_backtest("rules", bars, perp, params=on, **kw))
    p_today = _first_entry(run_backtest("rules", bars, perp, params=off, **kw))["budget"]
    assert p_high["sized_by"] == VOL_SIZE_LIMIT and p_high["budget"] == pytest.approx(0.5 * 10_000, rel=0.003)
    assert p_high["vol_factor"] == pytest.approx(p_high["budget"] / p_today, abs=1e-6) and p_high["vol_factor"] < 1
    calm = _first_entry(_sized(_daily(0.30 / ALTERNATING, tail=False), VOL_TARGET))
    assert calm["sized_by"] != VOL_SIZE_LIMIT and calm["vol_factor"] == 1.0
    assert calm["budget"] == today  # the same equity, the same limits: today's size, not 1.33x


ALWAYS = {"version": 1, "reason": "a fixture", "blocks": {},
          "long": {"entry": {"left": "close", "op": ">", "right": 0}, "exit": {"left": "close", "op": "<", "right": 0}}}


def test_with_29_days_of_returns_there_is_no_entry_and_with_30_there_is():
    """Advisor cell: 30 complete UTC days (31 daily closes) or no entry, and the skips counted on their own."""
    bars = _daily(0.03, days=40, tail=False)
    params = {**to_params(ALWAYS), "vol_target_size": VOL_TARGET}
    res = run_backtest("rules", bars, venue("KRAKEN").instrument("BTC", "USD", price_precision=1), params=params,
                       bar_minutes=1440, half_spread=0, starting_capital=10_000)
    [first] = [d["signal"] for d in res.decisions.values() if d.get("intent") == "entry"][:1]
    assert pd.Timestamp(first["bar"]["close_ts"]) == bars.index[30]  # its 31st close: 30 days of returns
    assert res.rule_counts == {"vol_skipped": 30}  # every candle before, the 30th close's (29 returns) included


def _four_hourly(r, days=60):
    """The daily fixture's closes at 00:00 UTC, with five 4h candles inside each day wobbling half a percent around
    the day's open: realised volatility reads only the closes of complete days, never the day under way."""
    d = _daily(r, days)
    rows, prev = [], float(d["open"].iloc[0])
    for ts, close in d["close"].items():
        for k in range(1, 6):
            rows.append((ts - pd.Timedelta(hours=24 - 4 * k), round(prev * (1 + 0.005 * (-1) ** k), 1)))
        rows.append((ts, float(close)))
        prev = float(close)
    c = pd.Series(dict(rows))
    o = np.r_[c.iloc[0], c.to_numpy()[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c.to_numpy(),
                         "volume": 1e12}, index=c.index)


def test_on_intraday_candles_only_complete_days_are_read():
    bars = _four_hourly(0.03)
    params = {**to_params(TREND), "vol_target_size": VOL_TARGET}
    res = run_backtest("rules", bars, venue("KRAKEN").instrument("BTC", "USD", price_precision=1), params=params,
                       bar_minutes=240, half_spread=0, starting_capital=10_000)
    inside = [d["signal"] for d in res.decisions.values()
              if d.get("intent") == "entry" and pd.Timestamp(d["signal"]["bar"]["close_ts"]).hour != 0]
    assert inside  # decided inside a day: that day's candles so far are not a daily close
    for sig in inside:
        at = pd.Timestamp(sig["bar"]["close_ts"])
        days = bars["close"][(bars.index.hour == 0) & (bars.index <= at)]
        assert sig["vol_target"].startswith(f"volatility {realised_vol(list(days.iloc[-31:])):.2%} a year over 30 days")


def test_without_the_daily_closes_to_read_it_the_entry_is_skipped_and_counted():
    bars = _daily(0.03, days=20)  # 20 days: the 31 closes it needs never come
    on, off = _sized(bars, VOL_TARGET), _sized(bars)
    assert any(d.get("intent") == "entry" for d in off.decisions.values())
    assert not any(d.get("intent") == "entry" for d in on.decisions.values())
    assert on.rule_counts["vol_skipped"] > 0


# ---- studies only ------------------------------------------------------------------------------------------------------

def test_paper_and_live_refuse_both_on_start_in_words():
    gated = to_params(GATED)
    with pytest.raises(ValueError, match="funding rate decile block, which runs in studies only until the PM"):
        check_study_only("rules", gated)
    with pytest.raises(ValueError, match="sizing to a target volatility"):
        check_study_only("rules", {**to_params(TREND), "vol_target_size": VOL_TARGET})
    check_study_only("rules", to_params(TREND))  # neither: unchanged
    assert study_only_uses({}) == []


def test_the_start_check_names_a_strategy_using_either(tmp_path):
    from sleeve_fund.start_check import FAIL, PASS, check_study_only as section
    from sleeve_fund.store import Store

    store = Store.in_memory()
    store.create_sleeve(name="plain", strategy="rules", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=1_000, params=to_params(TREND))
    assert [s for s, _ in section(store).lines] == [PASS]
    store.create_sleeve(name="v3", strategy="rules", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=1_000, params={**to_params(TREND), "vol_target_size": VOL_TARGET})
    [(status, words)] = section(store).lines
    assert status == FAIL and words.startswith("v3: uses sizing to a target volatility")


# CR M1 (11:48 UK): every place paper or live would build one refuses it, each pinned.
def _vol_sleeve(name="v3"):
    from sleeve_fund.paper.config import SleeveConfig

    return SleeveConfig(name=name, strategy="rules", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=1_000, params={**to_params(TREND), "vol_target_size": VOL_TARGET})


def test_a_holder_with_study_only_settings_runs_for_its_exits_only_and_says_why():
    """base: a Strategy the supervisor let through because it holds a position (node.py) is put on exits only by the
    model itself (_refused_to_trade), never trades in full."""
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import LongFlatStrategy

    held = SimpleNamespace(strategy="rules", params={**to_params(TREND), "vol_target_size": VOL_TARGET},
                           risk_profile="balanced")
    assert "sizing to a target volatility" in LongFlatStrategy._refused_to_trade(held)
    assert "funding rate decile" in LongFlatStrategy._refused_to_trade(
        SimpleNamespace(strategy="rules", params=to_params(GATED), risk_profile="balanced"))
    assert LongFlatStrategy._refused_to_trade(
        SimpleNamespace(strategy="rules", params=to_params(TREND), risk_profile="balanced")) is None


def test_the_whole_start_check_fails_on_a_study_only_strategy_and_names_it(tmp_path):
    from sleeve_fund import start_check
    from sleeve_fund.store import Store

    url = f"sqlite:///{tmp_path}/j.db"
    store = Store(url)
    store.create_sleeve(name="v3", strategy="rules", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=1_000, params={**to_params(TREND), "vol_target_size": VOL_TARGET})
    store.engine.dispose()
    _, page = start_check.run(url, environ={})
    assert "Studies-only settings: FAIL" in page and "Start check: FAIL" in page
    assert "v3: uses sizing to a target volatility" in page and "its Start is refused" in page


def test_the_dashboards_start_and_its_form_refuse_one_in_words(client):  # noqa: F811
    from test_dashboard import AUTH, SAME

    from sleeve_fund.dashboard.app import _check_strategy_params

    c, store = client
    store.create_sleeve(name="v3", strategy="rules", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=1_000, params={**to_params(TREND), "vol_target_size": VOL_TARGET},
                        desired_state="stopped")
    r = c.post("/sleeves/v3/command", data={"command": "start", "reason": "try it"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert "command_error" in r.headers["location"] and "studies+only" in r.headers["location"], r.headers["location"]
    assert store.sleeve("v3").desired_state == "stopped"
    with pytest.raises(ValueError, match="runs in studies only until the PM approves it for paper"):
        _check_strategy_params(_vol_sleeve())  # create, clone and a settings edit all build through it


def test_a_file_config_node_refuses_one_before_it_connects(monkeypatch):
    from sleeve_fund.paper import node

    monkeypatch.setattr(node, "load_sleeve", lambda path: _vol_sleeve())
    monkeypatch.setattr(node, "build_node", lambda *a, **k: pytest.fail("built a node for a study-only Strategy"))
    with pytest.raises(ValueError, match="sizing to a target volatility"):
        node.main(["configs/x.toml"])


# ---- the pre-registered variants run exactly their registered settings ------------------------------------------------

def _load(name):
    from sleeve_fund.research.variants import definition_path

    with open(definition_path(name), "rb") as f:
        return tomllib.load(f)


def test_the_four_variants_are_exactly_the_preregistered_settings():
    assert VOL_TARGET == {"annual": 0.40, "lookback_days": 30}
    for vid in ("V1", "V2"):
        v = PREREGISTERED[vid]
        defn, parent = _load(v["definition"]), _load(v["parent"])
        assert v["params"] == {}
        assert defn["blocks"].pop("fund") == {"kind": "funding_decile", "window_days": 90}
        gate = {"left": "fund", "op": "<", "right": 10}  # no new long in the top decile
        entry = defn["long"]["entry"]["all"]
        assert gate in entry
        entry.remove(gate)
        rest = entry[0] if len(entry) == 1 else {"all": entry}
        assert rest == parent["long"]["entry"]
        defn["long"]["entry"] = rest
        for k in ("name", "title"):
            defn.pop(k), parent.pop(k, None)
        assert defn == parent  # exits, stops, blocks and candles are the parent's
        assert study_only_uses(to_params(_load(v["definition"])))
        check_definition(_load(v["definition"]), bar_minutes=240)  # its own file passes the builder checks on 4h
    for vid in ("V3", "V4"):
        v = PREREGISTERED[vid]
        assert v["definition"] == v["parent"] and v["params"] == {"vol_target_size": VOL_TARGET}


def test_the_supervisor_and_a_process_started_by_hand_refuse_it_saying_why(tmp_path, monkeypatch):
    from sleeve_fund.paper.node import main
    from sleeve_fund.store import Store
    from sleeve_fund.supervisor import Supervisor

    url = f"sqlite:///{tmp_path}/r07.db"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("SLEEVE_LOCK_DIR", str(tmp_path / "locks"))
    store = Store(url)
    store.create_sleeve(name="v3", strategy="rules", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=1_000, params={**to_params(TREND), "vol_target_size": VOL_TARGET})
    assert main(["--db-sleeve", "v3"]) == 1
    assert Supervisor(store, python="true")._refused("v3")
    refused = [e["message"] for e in store.events("v3") if e["kind"] == "start_refused"]
    assert refused and all("runs in studies only until the PM approves it for paper" in m for m in refused), refused
    assert store.sleeve("v3").status == "stopped"


def test_a_study_sheet_counts_the_candles_with_no_rate_and_states_the_vol_target_and_exposure(tmp_path, monkeypatch):
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research import study
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.strategies.rules import SPEC as RULES
    from test_binance import INFO, binance_contract  # the venue's contract limits without the network

    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")  # no stored rate anywhere
    params = {**to_params(GATED), **PERP, "vol_target_size": VOL_TARGET}
    prices = synthetic_ohlcv(days=200, seed=4)
    r = study.run_study(RULES, prices, b.instrument("BTC", "USDT"), dataset="syn", ledger=IdeaLedger(tmp_path / "l"),
                        synthetic=True, holdout_days=0, train_days=60, test_days=60, risk_profile="balanced",
                        default_params=params)
    [no_rate] = [n for n in r.notes if n.startswith("Funding rate decile: ")]
    assert no_rate.startswith(f"Funding rate decile: {len(prices):,} of {len(prices):,} decision candles had no usable")
    assert any(n.startswith("Entries are sized to 40% a year of volatility, read over 30 days") for n in r.notes)
    assert any(n.startswith("Average gross exposure over the research period: ") for n in r.notes)


# ---- against the parent at the same exposure (Advisor, 21 Oct ~10:20 UK) ----------------------------------------------

def test_k_is_the_variants_average_gross_exposure_over_the_parents_overall_and_by_year():
    from sleeve_fund.research.equal_exposure import compare, exposure_ratio

    idx = pd.date_range("2024-12-31", periods=4, freq="1D", tz="UTC")
    variant, parent = pd.Series([0.2, -0.2, 0.3, 0.0], index=idx), pd.Series([0.4, 0.4, 0.6, 0.0], index=idx)
    k, by_year = exposure_ratio(variant, parent)
    assert k == pytest.approx(0.7 / 1.4) and by_year == {2024: pytest.approx(0.5), 2025: pytest.approx(0.5)}
    with pytest.raises(ValueError, match="same test bars"):
        exposure_ratio(variant, parent.iloc[1:])
    days = pd.date_range("2025-01-01", periods=40, freq="1D", tz="UTC")
    ret = pd.Series(0.01 * np.sin(np.arange(40)), index=days)
    e = compare(variant, parent, ret * 0.5, ret, 60.0, 55.0)
    assert e.scaled_exposure == pytest.approx(e.variant_exposure)  # the scaled parent holds what the variant holds
    assert e.scaled["cagr"] == pytest.approx(e.variant["cagr"]) and e.scaled["max_drawdown"] == pytest.approx(
        e.variant["max_drawdown"])  # the parent at half its size is this variant exactly
    assert e.parent_sharpe == pytest.approx(e.variant["sharpe"])  # Sharpe: against the parent unscaled


def test_a_vol_sized_study_shows_the_parent_scaled_to_its_exposure(tmp_path, monkeypatch):
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research import study
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.tearsheet import render
    from sleeve_fund.strategies.rules import SPEC as RULES

    ledger = IdeaLedger(tmp_path / "l")
    prices = synthetic_ohlcv(days=240, seed=4)
    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    kw = dict(dataset="syn", ledger=ledger, synthetic=True, holdout_days=0, train_days=60, test_days=60)
    r = study.run_study(RULES, prices, inst, default_params={**to_params(TREND), "vol_target_size": VOL_TARGET}, **kw)
    plain = study.run_study(RULES, prices, inst, default_params=to_params(TREND), **kw)
    e = r.equal_exposure
    assert e is not None and plain.equal_exposure is None
    assert 0 < e.k <= 1 + 1e-9  # it only shrinks
    assert e.scaled_exposure == pytest.approx(e.variant_exposure, rel=1e-12)
    stats = lambda x: sum(f.gross_exposure for f in x.folds) / sum(f.window_bars for f in x.folds)  # noqa: E731
    assert e.variant_exposure == pytest.approx(stats(r)) and e.parent_exposure == pytest.approx(stats(plain))
    assert e.parent_sharpe == pytest.approx(study.summary(plain.oos_returns)["sharpe"])
    sizes = [x for f in r.folds for _, x in f.trip_sizes]  # random entry sizes its trips as they were sized
    assert sizes and all(0 < x <= 1 for x in sizes) and min(sizes) < 1
    assert not [x for f in plain.folds for _, x in f.trip_sizes]
    sheet = render(r, ledger)
    assert "## Sizing to a target volatility, against the parent at the same exposure" in sheet
    assert f"k = {e.k:.3f}" in sheet


def test_with_vol_sizing_off_the_entry_journal_carries_no_new_field():
    """The engine is paper's and live's too (HoE, 21 Oct): without vol_target_size an entry's signal is as on main."""
    for inst, extra in ((venue("KRAKEN").instrument("BTC", "USD", price_precision=1), {}),
                        (venue("BINANCE").instrument("BTC", "USDT", price_precision=1), PERP)):
        res = run_backtest("rules", _daily(0.03), inst, params={**to_params(TREND), **extra}, bar_minutes=1440,
                           half_spread=0, starting_capital=10_000)
        signals = [d["signal"] for d in res.decisions.values() if d.get("intent") == "entry"]
        assert signals and not [s for s in signals if {"vol_factor", "vol_target"} & set(s)]


def test_the_study_runner_runs_each_variant_by_its_row_and_the_strategy_tab_never_lists_them():
    from sleeve_fund.research import rescore

    v1, v3 = (rescore.request(n, 240, "binance", "BTC/USDT") for n in ("v1", "v3"))
    assert "fund" in v1.params["definition"]["blocks"] and "vol_target_size" not in v1.params
    assert v3.params["vol_target_size"] == VOL_TARGET and v3.label == "v3"
    assert not list(DEFS.glob("*funding*.toml"))  # the Strategy tab's files: none of the variants'


NOISE_EDGE = 0.00026


def test_better_than_the_parent_needs_all_three_and_a_tight_paired_sharpe():
    """Advisor (21 Oct ~10:46 UK): Sharpe higher in 90% of paired stationary-bootstrap resamples, drawdown no worse
    than the scaled parent's, and its own edge. A 0.03 Sharpe edge is not better."""
    from sleeve_fund.research.equal_exposure import SHARPE_SHARE, compare, paired_sharpe_share
    from sleeve_fund.research.metrics import summary

    days = pd.date_range("2023-01-01", periods=600, freq="1D", tz="UTC")
    rng = np.random.default_rng(1)
    parent = pd.Series(rng.normal(0.0015, 0.02, len(days)), index=days)
    assert paired_sharpe_share(parent, parent) == 0.0  # never higher than itself
    calmer = parent * 0.5 + 0.0004  # the same signals at half the size, plus a steady edge: higher in every resample
    assert paired_sharpe_share(calmer, parent) == 1.0
    noisy = parent + pd.Series(rng.normal(0, 0.004, len(days)), index=days) + NOISE_EDGE  # a few hundredths
    assert 0 < summary(noisy)["sharpe"] - summary(parent)["sharpe"] < 0.05  # higher, but not by much
    assert paired_sharpe_share(noisy, parent) < SHARPE_SHARE
    from sleeve_fund.research.equal_exposure import SHARPE_BLOCK_DAYS, stationary_indices

    idx = stationary_indices(5000, SHARPE_BLOCK_DAYS, 2, np.random.default_rng(0))
    runs = (np.diff(idx, axis=1) != 1).sum() + 2  # blocks across both resamples (a wrap-round reads as 1 step)
    assert 5000 * 2 / runs == pytest.approx(SHARPE_BLOCK_DAYS, rel=0.1)  # blocks average 20 days
    assert paired_sharpe_share(calmer, parent) == paired_sharpe_share(calmer, parent)  # a fixed seed
    exp = pd.Series(0.5, index=days)
    yes = compare(exp * 0.5, exp, calmer, parent, 97.0, 90.0, own_edge=True)
    assert yes.better and yes.sharpe_better and yes.drawdown_better
    assert not compare(exp * 0.5, exp, calmer, parent, 97.0, 90.0, own_edge=False).better  # no edge of its own
    tight = compare(exp, exp, noisy, parent, 97.0, 90.0, own_edge=True)
    assert tight.variant["sharpe"] > tight.parent_sharpe and not tight.sharpe_better and not tight.better  # paired
    deeper = calmer.copy()
    deeper.iloc[300:320] = -0.02  # a drawdown deeper than the parent's at half its size
    worse = compare(exp * 0.5, exp, deeper, parent, 97.0, 90.0, own_edge=True)
    assert not worse.drawdown_better and not worse.better
    from sleeve_fund.research.equal_exposure import section

    line = next(x for x in section(yes) if x.startswith("**Variant against parent"))
    assert yes.better and line.startswith(f"**Variant against parent, no verdict while close-priced** (Sharpe ✓ "
                                          f"p={yes.sharpe_share:.2f}, drawdown ✓, own edge ✓)")
