"""qr01b V3 (research/quant-researcher/qr01b-v3-adx-preregistration.md, Advisor cleared 24 Oct): the `adx` block is
regime_split.adx worked bar by bar; on a 4h strategy its 1d candles show only completed days; an entry condition with
no_value = "pass" imposes nothing while the ADX warms up and nothing else changes; the V3 definition is each parent leg
plus that one condition; its one-row study list parses."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import regime_split as rs
from sleeve_fund.research import rescore
from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import (Compiled, Env, check_definition, definition_hash, load_definition_file,
                                                side_exits, to_params)
from sleeve_fund.strategies.indicators import BLOCKS, Adx, make_block, settle_bars
from sleeve_fund.strategies.rules import Rules
from sleeve_fund.strategies.timeframes import SlowerCandles

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "qr01b-v3-adx-list.md"
H = 3_600_000_000_000
GATE = {"left": "adx", "op": "<=", "right": 25, "no_value": "pass"}
BASE = {"version": 1, "reason": "A test case, written before it is run.",
        "blocks": {"adx": {"kind": "adx", "period": 14, "timeframe": "1d"}},
        "long": {"entry": {"all": [{"left": "close", "op": ">", "right": 0}, GATE]}}}


def _daily(n=500, seed=0, flat_from=None):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    h, lo = c * (1 + rng.uniform(0, 0.03, n)), c * (1 - rng.uniform(0, 0.03, n))
    if flat_from is not None:  # days with no range at all: 0/0 directional moves
        h[flat_from:flat_from + 10] = lo[flat_from:flat_from + 10] = c[flat_from:flat_from + 10] = c[flat_from - 1]
    idx = pd.date_range("2021-01-02", periods=n, freq="D", tz="UTC")  # stamped at each day's close
    return pd.DataFrame({"open": c, "high": h, "low": lo, "close": c}, index=idx)


def _four_hour(n_days=200, seed=1):
    """4h candles stamped at their close, 00:00 included, as the engine's decision candles are."""
    rng = np.random.default_rng(seed)
    n = 6 * n_days
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.date_range("2024-01-01 04:00", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "volume": 1e3}, index=idx)


@pytest.mark.parametrize("period, seed, flat", [(14, 0, None), (14, 1, 50), (5, 2, None), (3, 3, 200)])
def test_the_block_reads_regime_split_adx_bit_for_bit_on_the_same_daily_bars(period, seed, flat):
    d = _daily(seed=seed, flat_from=flat)
    ref = rs.adx(d, period).to_numpy()
    b, raw, read = Adx(period), [], []
    for r in d.itertuples():
        b.update_raw(r.high, r.low, r.close)
        raw.append(b._value)
        read.append(b.value)
    have = ~np.isnan(ref)
    assert have.sum() == len(d) - 2 * period  # regime_split reads it from bar 2n on
    np.testing.assert_array_equal(np.array(raw)[have], ref[have])
    settle = settle_bars(period)
    assert all(v is None for v in read[: settle - 1])  # a Wilder average: initialized after ten lengths
    np.testing.assert_array_equal(np.array(read[settle - 1:], dtype=float), ref[settle - 1:])


def test_on_4h_candles_each_decision_reads_the_last_completed_day_only():
    bars = _four_hour()
    ref = rs.adx(rs.daily(bars))  # the report's own daily bars, stamped at each day's close
    block = Adx(14)
    slow = SlowerCandles(1440, 240, blocks=[block])
    days = ref.index.as_unit("ns").asi8
    seen = 0
    for ts, r in zip(bars.index.as_unit("ns").asi8, bars.itertuples()):
        slow.update(r.open, r.high, r.low, r.close, r.volume, int(ts))
        last = int(np.searchsorted(days, ts, side="right")) - 1  # the latest day closed at or before this decision
        if last >= settle_bars(14) - 1:
            assert block.value == ref.iloc[last]
            seen += 1
    assert seen > 300
    # A decision mid-day (12:00) reads the day that closed at 00:00, whatever the candles since then did.
    noon = bars.index[bars.index.hour == 12][-5]
    for wild in (False, True):
        part = bars.loc[:noon].copy()
        if wild:
            part.iloc[-3:, :4] *= 1.5  # the 04:00, 08:00 and 12:00 candles of the day still forming
        b = Adx(14)
        s = SlowerCandles(1440, 240, blocks=[b])
        for ts, r in zip(part.index.as_unit("ns").asi8, part.itertuples()):
            s.update(r.open, r.high, r.low, r.close, r.volume, int(ts))
        assert s.last.end == noon.normalize().value and b.value == ref.loc[noon.normalize()]


def _env(adx_value):
    """A compiled BASE with its ADX block reading `adx_value` (None: not settled yet)."""
    c = Compiled(check_definition(BASE, bar_minutes=240))
    block = c.blocks["adx"]
    if adx_value is not None:
        for _ in range(block.warmup_bars):
            block.update_raw(1, 1, 1)
        block._value = adx_value
    return c, Env(ohlcv=(1.0, 1.0, 1.0, 1.0, 1.0), blocks=c.blocks)


@pytest.mark.parametrize("adx, holds", [(None, True), (10.0, True), (25.0, True), (25.01, False), (60.0, False)])
def test_the_gate_holds_at_or_below_25_and_while_the_adx_has_no_value(adx, holds):
    c, env = _env(adx)
    assert c.sides[1]["entry"].test(env) is holds


def test_without_no_value_a_condition_still_fails_while_its_block_warms_up():
    defn = copy.deepcopy(BASE)
    defn["long"]["entry"]["all"][1] = {k: v for k, v in GATE.items() if k != "no_value"}
    c = Compiled(check_definition(defn, bar_minutes=240))
    assert c.sides[1]["entry"].test(Env(ohlcv=(1.0,) * 5, blocks=c.blocks)) is False  # as every block did before


def test_in_a_backtest_entries_open_through_the_warm_up_and_are_filtered_once_the_adx_reads(instrument):
    from sleeve_fund.research.runner import run_backtest

    n = 6 * 200
    idx = pd.date_range("2024-01-01 04:00", periods=n, freq="4h", tz="UTC")
    c = np.linspace(100, 400, n)  # a steady climb: ADX far above 25 once it reads
    df = pd.DataFrame({"open": c - 0.1, "high": c + 0.2, "low": c - 0.3, "close": c, "volume": 1e3}, index=idx)
    defn = {**copy.deepcopy(BASE), "exits": {"time_stop": {"bars": 1}}}
    res = run_backtest("rules", df, instrument, params=to_params(defn), bar_minutes=240, half_spread=0)
    entries = [res.fills.loc[o, "ts_last"] for o in res.fills.index if res.decisions[o]["intent"] == "entry"]
    settled = idx[0].normalize() + pd.Timedelta(days=settle_bars(14))  # the 140th completed day's close
    assert len(entries) > 100 and max(entries) < settled  # no filter in the warm-up, then the climb is filtered out


@pytest.mark.parametrize("defn, words", [
    ({**BASE, "blocks": {"adx": {"kind": "adx", "period": 0}}}, "at least 1"),
    ({**BASE, "blocks": {"adx": {"kind": "adx", "period": 2.5}}}, "whole number"),
    ({**BASE, "blocks": {"adx": {"kind": "adx", "length": 14}}}, "no setting length"),
    ({**BASE, "blocks": {"adx": {"kind": "adx", "input": "close"}}}, "reads several prices"),
    ({**BASE, "long": {"entry": {**GATE, "no_value": "fail"}}}, "no_value is \"pass\""),
    ({**BASE, "long": {"entry": {**GATE, "op": "crosses_below"}}}, "not a cross"),
    ({**BASE, "long": {"entry": {"left": "close", "op": ">", "right": 0}, "exit": GATE}}, "belongs in an entry rule"),
    ({**BASE, "long": {"entry": {"only_when": GATE, "then": {"left": "close", "op": ">", "right": 0}}}},
     "belongs in an entry rule"),
])
def test_bad_adx_settings_and_no_value_forms_are_refused(defn, words):
    with pytest.raises(ValueError, match=words):
        check_definition(defn, bar_minutes=240)


@pytest.mark.parametrize("settings", [{"period": 0}, {"period": -3}, {"period": True}, {"period": 10**6}])
def test_bad_periods_are_refused_by_the_library(settings):
    with pytest.raises(ValueError):
        make_block("adx", **settings)


def test_the_adx_is_a_library_block_and_definitions_without_no_value_keep_their_hashes():
    assert BLOCKS["adx"] is Adx and Adx.warmup_bars(period=14) == 140
    long_ = load_definition_file(definition_path("qr01b_trend_multispeed_long"))
    assert definition_hash(long_) == "e122ddf1851edf73"  # the parent the pre-registration names


def _legs(defn: dict, side: str) -> dict:
    return {"entry": defn[side]["entry"], "exit": defn[side].get("exit"), "exits": side_exits(defn.get("exits"), side)}


def test_v3_is_each_parent_leg_plus_only_the_daily_adx_entry_gate():
    v = PREREGISTERED["V3_ADX"]
    defn = load_definition_file(definition_path(v["definition"]))
    assert definition_path(v["definition"]).parent.name == "variants" and v["params"] == {}
    assert defn["blocks"].pop("adx") == {"kind": "adx", "period": 14, "timeframe": "1d"}
    for side, parent_name in (("long", v["parent"]), ("short", v["short_parent"])):
        parent = load_definition_file(definition_path(parent_name))
        assert set(parent) - {"long", "short"} == set(defn) - {"long", "short"} and side in parent
        assert defn["blocks"] == parent["blocks"]
        mine = _legs(defn, side)
        entry = mine["entry"]["all"]
        assert entry.count(GATE) == 1
        entry.remove(GATE)  # state-based: an entry condition only, so it never forces an exit
        assert mine == _legs(parent, side)  # exit rule and 3 ATR stop are the parent's
    full = load_definition_file(definition_path(v["definition"]))
    checked = check_definition(full, bar_minutes=240)
    assert checked.slower == {1440: 140}
    parent = load_definition_file(definition_path(v["parent"]))
    assert Rules.warmup_needed({"definition": full}, 240) == Rules.warmup_needed({"definition": parent}, 240)
    assert "adx" not in str(full["long"]["exit"]) + str(full["short"]["exit"]) + str(full["exits"])


def test_v3_reads_in_words_with_its_gate():
    from sleeve_fund.dashboard import strategy_words as words

    full = load_definition_file(definition_path("qr01b_trend_multispeed_ls_v3_adx"))
    long_, short = words.rules(check_definition(full), 240)
    for s in (long_, short):
        assert s["entry"].endswith("and ADX(14) daily is at or below 25 (or it has no value yet)")


def test_the_study_list_is_one_btc_row_on_the_round1_dates_and_runs_v3_long_and_short_on_the_perpetual(capfd):
    from sleeve_fund.research.pinned import parse

    text = LIST.read_text()
    round1 = (ROOT / "configs" / "studies" / "qr-round1-list.md").read_text()
    assert rescore.rows_of(text) == [(1, "v3_adx", 240)]
    assert parse(text) == parse(round1) and parse(text).holdout_days == rescore.HOLDOUT_DAYS
    assert rescore.CONTINUOUS_LINE.search(text) is None  # the round 1 walk-forward, as the parent's
    req = rescore.request("v3_adx", 240, "binance", "BTC/USDT", parse(text))
    defn = load_definition_file(definition_path("qr01b_trend_multispeed_ls_v3_adx"))
    assert req.params["definition"] == defn and req.params["market"] == "perp" and req.params["allow_short"] is True
    assert (req.use_holdout, req.holdout_days, req.train_days, req.test_days) == (False, 365, 730, 365)
    req.validate()
    workflow = (ROOT / ".github" / "workflows" / "study.yml").read_text()
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}\.md", LIST.name) and "^[a-z0-9][a-z0-9_-]{0,63}\\.md$" in workflow
    assert rescore.main([str(LIST), "--dry-run"], run=lambda *a, **k: None, journal=lambda: None) == 0
    assert "dry run: 1 checked, 0 run" in capfd.readouterr().out
