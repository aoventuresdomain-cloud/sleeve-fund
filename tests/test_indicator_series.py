"""P1-3s: the chart's indicator overlay (strategies.series). Each line is the model's own indicator as it stood when
the model decided on that candle: the value its decision journal records, its warm-up marked as not settled, and
never a value from a candle not yet closed."""

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.series import build, indicator_series

DAILY = synthetic_ohlcv(days=700, seed=3)

# A rule-builder strategy reading a band (price pane, grouped), a slower candle, and an average fed an RSI (drawn
# with the RSI, in the lower pane).
DEFINITION = {
    "version": 1, "reason": "A test case, written before it is run.",
    "blocks": {"rsi": {"kind": "rsi", "period": 14}, "avg": {"kind": "sma", "period": 3, "input": "rsi"},
               "bb": {"kind": "bollinger", "period": 20, "k": 2.0}, "trend": {"kind": "sma", "period": 20,
                                                                             "timeframe": "4h"}},
    "long": {"entry": {"all": [{"left": "avg", "op": "<=", "right": 40}, {"left": "close", "op": ">", "right": "trend"}]},
             "exit": {"left": "close", "op": ">=", "right": "bb.upper"}},
    "short": {"entry": {"all": [{"left": "rsi", "op": ">=", "right": 65}, {"left": "close", "op": "<", "right": "trend"}]},
              "exit": {"left": "close", "op": "<=", "right": "bb.lower"}},
}

CASES = [  # (model, params, candle minutes)
    ("rsi_cross", {"rsi_period": 14, "trend_sma": 20, "long_entry": 50.0, "long_exit": 60.0}, 60),
    ("dip_buy", {"trend_sma_days": 50}, 1440),
    ("rsi_bands", {}, 1440),
    ("rsi_pullback", {"ema_period": 20, "rsi_entry": 50, "vol_mult": 0.8}, 1440),
    ("trend_filter", {"fast": 10, "slow": 40}, 1440),
    ("donchian", {}, 1440),
    ("rules", {"definition": DEFINITION}, 60),
]


def _hourly(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    c = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.date_range("2025-06-01 01:00", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + rng.uniform(0, 3e-3, n)),
                         "low": np.minimum(o, c) * (1 - rng.uniform(0, 3e-3, n)), "close": c,
                         "volume": rng.uniform(5, 15, n)}, index=idx)


HOURLY = _hourly()


def _candles(minutes):
    return DAILY if minutes == 1440 else HOURLY


@pytest.mark.parametrize("model, params, minutes", CASES, ids=[c[0] for c in CASES])
def test_every_line_is_the_value_the_model_journalled_when_it_decided(model, params, minutes, instrument):
    df = _candles(minutes)
    series = indicator_series(model, df, instrument, params, minutes)
    assert series and all(s["kind"] == "line" and s["pane"] in ("price", "lower") for s in series)
    by_key = {s["key"]: dict(map(tuple, s["points"])) for s in series}
    res = run_backtest(model, df, instrument, params, bar_minutes=minutes, half_spread=0)
    checked = 0
    for order, d in res.decisions.items():
        if order not in res.fills.index:
            continue
        signal = d.get("signal") or {}
        journal = signal.get("blocks", {}) if model == "rules" else signal
        t = int(res.fills.loc[order, "ts_last"].timestamp())  # filled at the close of the candle it decided on
        for key, points in by_key.items():
            if isinstance(journal.get(key), (int, float)):
                assert points[t] == pytest.approx(journal[key], rel=1e-12, abs=1e-8), (key, t)  # the journal keeps 8 decimals
                checked += 1
    assert checked >= 4  # the model traded, and its journal names the drawn indicators


@pytest.mark.parametrize("model, params, minutes", CASES, ids=[c[0] for c in CASES])
def test_no_point_reads_a_candle_after_its_own(model, params, minutes, instrument):
    """Drawn over the first k candles, every line is the full run's cut at k: nothing later reached it."""
    df = _candles(minutes)
    k = len(df) * 2 // 3
    full = {s["key"]: s for s in indicator_series(model, df, instrument, params, minutes)}
    cut = indicator_series(model, df.iloc[:k], instrument, params, minutes)
    last = int(df.index[k - 1].timestamp())
    for s in cut:
        assert s["points"] == [p for p in full[s["key"]]["points"] if p[0] <= last]
        assert s["points"][-1][0] == last
    t = [p[0] for p in cut[0]["points"]]
    assert t == sorted(set(t)) and t == [int(x.timestamp()) for x in df.index[:k]]  # one point per closed candle


@pytest.mark.parametrize("model, params, minutes", CASES[:-1], ids=[c[0] for c in CASES[:-1]])
def test_a_models_warm_up_is_marked_not_settled(model, params, minutes, instrument):
    """Hand-coded models: settled from the candle their warm-up (settle_bars_needed) completes, as their fills'
    "unsettled" flag reads it; any value drawn before that is warm-up, never a normal line."""
    df = _candles(minutes)
    need = build(model, instrument, params, minutes).settle_bars_needed
    assert 1 < need < len(df)
    settled = int(df.index[need - 1].timestamp())
    for s in indicator_series(model, df, instrument, params, minutes):
        first = next(t for t, v in s["points"] if v is not None and t >= settled)
        assert s["settled_from"] == first, s["key"]


def test_a_rule_builder_strategy_marks_each_block_settled_when_its_rules_could_read_it(instrument):
    series = {s["key"]: s for s in indicator_series("rules", HOURLY, instrument, {"definition": DEFINITION}, 60)}
    assert set(series) == {"rsi", "avg", "bb.mid", "bb.upper", "bb.lower", "bb.width", "bb.pct_b", "trend"}
    hour = 3600
    first = int(HOURLY.index[0].timestamp())
    assert series["rsi"]["settled_from"] == first + (140 - 1) * hour  # Wilder: ten lengths
    assert series["avg"]["settled_from"] == first + (140 - 1 + 3 - 1) * hour  # fed from the RSI's first settled value
    assert series["bb.upper"]["settled_from"] == first + (20 - 1) * hour
    assert series["trend"]["settled_from"] > series["bb.upper"]["settled_from"]  # twenty 4-hour candles
    rsi = series["rsi"]
    assert any(v is not None for t, v in rsi["points"] if t < rsi["settled_from"])  # warm-up is drawn, marked


def test_a_rule_builder_strategys_lines_carry_their_panes_groups_levels_and_timeframe(instrument):
    series = {s["key"]: s for s in indicator_series("rules", HOURLY.iloc[:300], instrument,
                                                    {"definition": DEFINITION}, 60)}
    assert {k: s["pane"] for k, s in series.items()} == {
        "rsi": "lower", "avg": "lower", "bb.mid": "price", "bb.upper": "price", "bb.lower": "price",
        "bb.width": "lower", "bb.pct_b": "lower", "trend": "price"}
    assert {series[k]["group"] for k in ("bb.mid", "bb.upper", "bb.lower")} == {"bb"}
    assert series["rsi"]["levels"] == [65.0] and series["avg"]["levels"] == [40.0]
    assert series["trend"]["tf"] == "4h" and series["rsi"]["tf"] is None
    # At first only what the rules read: the RSI, its average, the 4h trend and the outer bands, not mid, width or %b.
    assert {k for k, s in series.items() if s["shown"]} == {"avg", "rsi", "trend", "bb.upper", "bb.lower"}


def test_a_rule_reading_a_band_by_its_bare_id_shows_the_bands_mid(instrument):
    """CR #174: a rule may read a multi-output block by its bare id, which reads its primary output (a band's mid);
    that line starts shown, and the band's other outputs stay hidden."""
    bare = {**DEFINITION, "long": {**DEFINITION["long"], "exit": {"left": "close", "op": ">=", "right": "bb"}},
            "short": {**DEFINITION["short"], "exit": {"left": "close", "op": "<=", "right": "trend"}}}
    series = {s["key"]: s for s in indicator_series("rules", HOURLY.iloc[:300], instrument, {"definition": bare}, 60)}
    assert {k for k, s in series.items() if s["shown"]} == {"avg", "rsi", "trend", "bb.mid"}
