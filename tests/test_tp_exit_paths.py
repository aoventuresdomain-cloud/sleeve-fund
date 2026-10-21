"""TP-EXIT (HoE, Advisor 21 Oct 2026): a take-profit exit in a definition, {r = N} or {level = "<block>.<output>"}, and
one rule for reaching it in all three paths:
1. a taker fill at the level less max(half spread, 0.05 %), with the taker fee (target_fill_px);
2. reached on a TOUCH (>= for a long, <= for a short) in the backtest, paper's tick and the outage replay: one shared
   target_reached, its operator one parameter, and one target_level on the instrument's tick, so an exact touch is a
   touch in each (the parity cell below has a bar whose high is the level exactly);
3. never better than the level, a gap included;
4. the stop goes first;
5. {r}: R is the initial stop distance, frozen at entry;
6. {level}: worked from closed candles and refreshed each candle; on the wrong side it doesn't fire on that candle;
   the form is logged;
7. every exit records its kind."""
from __future__ import annotations

import copy
import os

os.environ.setdefault("BACKTEST_ISOLATE", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from sleeve_fund.instruments import target_level, target_reached  # noqa: E402
from sleeve_fund.strategies.base import replay_missed  # noqa: E402
from sleeve_fund.strategies.definitions import check_definition, to_params  # noqa: E402
from test_hub_146_qa import _probe  # noqa: E402,F401 - the probe strategy the parity cells trade

TP = 0.02


# --- the shared rule ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_target_is_reached_on_a_touch_and_not_a_tick_short_of_it(side):
    level = 61_200.0
    assert target_reached(side, level, level, level)  # an exact touch
    assert target_reached(side, level + side * 0.1, level + side * 0.1, level)  # through it
    assert not target_reached(side, level - side * 0.1, level - side * 0.1, level)  # a tick short
    # a long reads the high, a short the low: the other extreme never reaches it
    hi, lo = (level, level - 50) if side > 0 else (level + 50, level)
    assert target_reached(side, hi, lo, level)
    assert not target_reached(side, lo if side > 0 else hi, lo if side > 0 else hi, level)


def test_the_target_level_is_on_the_instruments_tick():
    assert target_level(60_000.06, 1, TP, 1) == 61_200.1
    assert target_level(60_000.06, -1, TP, 1) == 58_800.1


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_the_outage_replay_takes_an_exact_touch_and_never_books_better_than_the_level(side):
    level = 61_200.0 if side > 0 else 58_800.0
    flat = 60_000.0
    touch = (1, flat, level if side > 0 else flat, flat if side > 0 else level)
    hit = replay_missed([touch], side, {}, {}, level, target_px=level * (1 - side * 0.0005))
    assert hit is not None and hit[0] == "take_profit" and hit[2] == level
    assert side * (hit[1] - level) <= 0  # never better than the level
    gap = (1, level + side * 300, level + side * 400, level + side * 200)  # opens through it
    hit = replay_missed([gap], side, {}, {}, level, target_px=level * (1 - side * 0.0005))
    assert hit[0] == "take_profit" and side * (hit[1] - level) <= 0, hit  # a gap books the level's fill, not the open
    both = (1, flat, max(level, flat * 1.03), min(level, flat * 0.97))  # the stop and the target in one minute
    hit = replay_missed([both], side, {"stop_loss": flat * (1 - side * 0.01)}, {}, level)
    assert hit[0] == "stop_loss", hit  # the stop goes first


# --- parity: an exact touch fires in the backtest, paper and the replay -------------------------------------------

def _touched(prices, level, at=(7.5, 7 + 50 / 60)):
    p = prices.copy()
    s = np.arange(len(p))
    p[(s >= at[0] * 60) & (s < at[1] * 60)] = level
    return p


SETUPS = [("spot-long", False, "aggressive", 1), ("perp-3x-long", True, "aggressive", 1),
          ("perp-3x-short", True, "aggressive", -1)]


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=[s[0] for s in SETUPS])
@pytest.mark.parametrize("path", ["backtest", "paper", "replay"])
def test_an_exact_touch_of_the_target_fires_in_every_path(path, label, perp, profile, side):
    """Advisor rule 2: a bar (or tick) whose high (a long's) or low (a short's) is the level exactly takes the target,
    in the backtest, paper's tick and the outage replay, booked at the level less the taker's slippage."""
    import test_hub_146_qa as qa

    def run(p):
        kw = {"side": side, "perp": perp, "profile": profile, "tp": TP, "stop": 0.01, "leave": 25}
        if path == "backtest":
            return None, qa.backtest(p, **kw)
        if path == "paper":
            r = qa.paper(p, **kw)
        else:
            r = qa.restart(p, 6, 12, qty=0.05, **kw)
        return r, r.sequence()

    flat = qa.flat_prices(30)
    _, seq = run(flat)
    entry = next(r[3] for r in seq if r[0] == "entry")
    level = target_level(entry, side, TP, 1)
    p = _touched(flat, level)
    m = qa.minutes_of(p).iloc[7]
    assert float(m["high" if side > 0 else "low"]) == level  # an exact touch, nothing through it
    _, seq = run(p)
    assert next(r[3] for r in seq if r[0] == "entry") == entry  # the same entry, so the same level
    ex = qa.first_exit(seq)
    assert ex[0] == "take_profit", (path, seq)
    assert side * (ex[3] - level) <= 0, (ex, level)  # never better than the level
    if path != "paper":  # paper's market order fills at the market; the backtest and replay book the model
        assert ex[3] == pytest.approx(level * (1 - side * 0.0005), rel=1e-6), (ex, level)


# --- the definition -------------------------------------------------------------------------------------------------

BASE = {"version": 1, "reason": "A test case, written before it is run.",
        "blocks": {"atr": {"kind": "atr", "period": 14}, "dc": {"kind": "donchian", "period": 20}},
        "long": {"entry": {"left": "close", "op": ">", "right": 0}},
        "exits": {"stop": {"level": {"sub": ["close", {"mul": [3.0, "atr"]}]}}}}


def _with_tp(tp, exits=None):
    d = copy.deepcopy(BASE)
    if exits is not None:
        d["exits"] = exits
    d["exits"]["take_profit"] = tp
    return d


@pytest.mark.parametrize("tp", [{"r": 3}, {"r": 0.5}, {"level": "dc.upper"}])
def test_a_take_profit_in_r_or_at_a_level_is_accepted(tp):
    check_definition(_with_tp(tp), bar_spec="15-MINUTE-LAST-INTERNAL")


@pytest.mark.parametrize("tp, exits, words", [
    ({"r": 3}, {}, "take_profit needs a stop"),
    ({"r": 3, "level": "dc.upper"}, None, "take_profit is"),
    ({}, None, "take_profit is"),
    ({"r": 0}, None, "take_profit.r"),
    ({"r": 51}, None, "take_profit.r"),
    ({"r": True}, None, "take_profit.r"),
    ({"level": "nope.upper"}, None, "unknown reference"),
])
def test_a_malformed_take_profit_is_refused_with_its_reason(tp, exits, words):
    with pytest.raises(ValueError, match=words):
        check_definition(_with_tp(tp, exits), bar_spec="15-MINUTE-LAST-INTERNAL")


def _bars(closes, highs=None, start="2025-10-03 00:01"):
    idx = pd.date_range(start, periods=len(closes), freq="1min", tz="UTC")
    c = np.asarray(closes, dtype=float)
    h = np.asarray(highs if highs is not None else c, dtype=float)
    return pd.DataFrame({"open": c, "high": np.maximum(h, c), "low": c, "close": c, "volume": 1e3}, index=idx)


def _run(defn, df, instrument):
    from sleeve_fund.research.runner import run_backtest

    return run_backtest("rules", df, instrument, params=to_params(defn), bar_minutes=1, half_spread=0)


def _exits(res):
    f = res.fills.sort_values("ts_last")
    return [(res.decisions[o]["intent"], res.decisions[o].get("signal") or {}) for o in f.index
            if res.decisions[o]["intent"] != "entry"]


def _entry_signal(res):
    f = res.fills.sort_values("ts_last")
    return next(res.decisions[o].get("signal") or {} for o in f.index if res.decisions[o]["intent"] == "entry")


def _entry_fill(res):
    f = res.fills.sort_values("ts_last")
    return next(float(f.loc[o, "avg_px"]) for o in f.index if res.decisions[o]["intent"] == "entry")


def test_an_r_target_is_r_times_the_initial_stop_in_price_from_the_fill_and_frozen(instrument):
    """Rule 5 (Advisor, HoE 21 Oct): {r = 2} on a 1% trailing stop is the entry fill + 2 x (the fill - the stop level
    the entry was planned with), costs charged on the fills, not in the level; the trail then tightens with the price
    and the target stays where the fill set it."""
    defn = {"version": 1, "reason": "A test case, written before it is run.", "blocks": {},
            "long": {"entry": {"left": "close", "op": ">", "right": 0}},
            "exits": {"trail": {"level": {"mul": [0.99, "close"]}}, "take_profit": {"r": 2}}}
    closes = [100.0] * 3 + [100.3, 100.6, 100.9, 101.2, 101.5] + [104.0] * 5
    res = _run(defn, _bars(closes), instrument)
    entry, fill = _entry_signal(res), _entry_fill(res)
    stop_level = entry["close"] * (1 - entry["stop_frac"])  # 99: the trail's level when the entry was planned
    exits = _exits(res)
    assert exits and exits[0][0] == "take_profit", exits
    sig = exits[0][1]
    assert sig["target_form"] == "r=2"
    assert sig["target_px"] == pytest.approx(round(fill + 2 * (fill - stop_level), 1), abs=0.051), (sig, fill)


def test_a_level_target_is_refreshed_each_candle_and_never_fires_on_a_candle_it_is_on_the_wrong_side(instrument):
    """Rule 6: {level = "m.value"}, a 3-candle average of the closes, worked at each close for the next candle. After
    the 00:04 close (101) it is 100.33, under the close: the wrong side for a long's target, so the 00:05 candle's
    high of 103 doesn't take it. After the 00:07 close (98) it is 99, above: the 00:08 candle's high of 100 takes it,
    at 99 less the taker's slippage."""
    defn = {"version": 1, "reason": "A test case, written before it is run.",
            "blocks": {"m": {"kind": "sma", "period": 3}},
            "long": {"entry": {"left": "close", "op": ">", "right": 0}},
            "exits": {"stop": {"level": {"mul": [0.9, "close"]}}, "take_profit": {"level": "m"}}}
    closes = [100.0, 100.0, 100.0, 101.0, 101.0, 98.0, 98.0, 98.0, 98.0, 98.0]
    highs = list(closes)
    highs[4], highs[7] = 103.0, 100.0
    df = _bars(closes, highs)
    res = _run(defn, df, instrument)
    exits = _exits(res)
    assert exits and exits[0][0] == "take_profit", exits
    sig = exits[0][1]
    assert sig["target_form"] == "level=m" and sig["target_px"] == pytest.approx(99.0), sig
    f = res.fills.sort_values("ts_last")
    when = next(f.loc[o, "ts_last"] for o in f.index if res.decisions[o]["intent"] == "take_profit")
    assert when >= df.index[7], (when, "fired on the wrong-side candle")


def test_the_target_form_and_the_exit_kind_are_journaled(instrument):
    defn = {"version": 1, "reason": "A test case, written before it is run.", "blocks": {},
            "long": {"entry": {"left": "close", "op": ">", "right": 0}},
            "exits": {"stop": {"level": {"mul": [0.99, "close"]}}, "take_profit": {"r": 1}}}
    res = _run(defn, _bars([100.0] * 3 + [110.0] * 4), instrument)
    exits = _exits(res)
    assert exits[0][0] == "take_profit" and exits[0][1]["target_form"] == "r=1", exits
    assert exits[0][1]["exit_kind"] == "take_profit"


def test_qr01b_with_its_3_atr_stop_and_a_3r_target_reconciles(instrument):
    """The HoE's reconcile: qr01b's own definition with take_profit = {r = 3} beside its 3 ATR(14) stop, daily bars.
    Each target is 3R in price from its entry's fill (HoE 21 Oct); every take-profit fills on a touch of it, booked at the level
    less the taker's slippage, never better; and the journal's cash is the run's end equity."""
    import tomllib
    from pathlib import Path

    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.store import replay_book

    defn = tomllib.loads((Path(__file__).resolve().parents[1] / "configs/definitions/qr01b_trend_multispeed_long.toml")
                         .read_text())
    defn["exits"]["take_profit"] = {"r": 3}
    df = synthetic_ohlcv(days=3000, seed=11)
    res = run_backtest("rules", df, instrument, params=to_params(defn), bar_minutes=1440, half_spread=0.0002,
                       risk_profile="balanced")  # with a profile, so the run keeps its journal
    f = res.fills.sort_values("ts_last")
    rows = [(res.decisions[o]["intent"], res.decisions[o].get("signal") or {}, float(f.loc[o, "avg_px"]))
            for o in f.index]
    targets, entry = [], None
    for k, s, px in rows:
        if k == "entry":
            entry = (s, px)
        elif k == "take_profit":  # 3R in price from that entry's fill and its 3 ATR stop level as planned
            stop_level = entry[0]["close"] * (1 - entry[0]["stop_frac"])
            fill = s["entry_px"]  # the engine's entry, the fill before the report's cent rounding
            # to 0.05: the journaled stop share is rounded to 6 places (x 3R), the target to the tick
            assert s["target_px"] == pytest.approx(fill + 3 * (fill - stop_level), abs=0.05), (s, entry)
            targets.append((s, px))
    assert targets, [k for k, _, _ in rows]
    for s, px in targets:
        assert s["target_form"] == "r=3" and s["exit_kind"] == "take_profit"
        assert px < s["target_px"], (px, s)  # never better than the level (rule 3)
        # the level less the floor; the report carries the fee's cent rounding into the price (D13 pin e)
        assert px == pytest.approx(s["target_px"] * (1 - 0.0005), rel=2e-6), (px, s)
    book = replay_book(res.journal.fills_, 10_000.0, res.journal.funding_total("backtest"),
                       res.journal.insurance_total("backtest"))
    marked = book["cash"] + book["qty"] * float(df["close"].iloc[-1])
    assert marked == pytest.approx(float(res.equity.iloc[-1]), abs=0.05), (book, float(res.equity.iloc[-1]))


@pytest.mark.parametrize("tp, words", [({"r": 3}, "3R after costs, from the stop's distance as the trade opens"),
                                       ({"level": "dc.upper"}, "as set at each close")])
def test_the_strategy_tab_says_the_take_profit_in_words(tp, words):
    from sleeve_fund.dashboard import strategy_words

    (long_,) = strategy_words.rules(check_definition(_with_tp(tp)), 15)
    assert ("Take-profit" in dict(long_["levels"])) and words in dict(long_["levels"])["Take-profit"], long_["levels"]


@pytest.mark.parametrize("study", [{"take_profit_r": 2.0}, {"take_profit": 0.05}])
def test_a_definitions_take_profit_and_a_study_wide_one_are_refused_together(study, instrument):
    """HoE 21 Oct: the definition's {r} is R in price, the study-wide take_profit_r R after costs; a run may carry one
    target, never both, so which one applied is never in doubt."""
    from sleeve_fund.research.runner import run_backtest

    defn = _with_tp({"r": 3})
    with pytest.raises(ValueError, match="one take-profit"):
        run_backtest("rules", _bars([100.0] * 30), instrument, params={**to_params(defn), **study}, bar_minutes=1,
                     half_spread=0)


def test_the_sheet_says_a_definitions_r_is_in_price():
    from sleeve_fund.research.study import _definition_target_words, _exit_words

    assert _definition_target_words(to_params(_with_tp({"r": 3}))) == ", take-profit at 3R (price)"
    assert _definition_target_words(to_params(_with_tp({"level": "dc.upper"}))) == ", take-profit at the level dc.upper"
    assert _exit_words({"take_profit_r": 3}) == "take-profit making 3R after costs"  # the study-wide one, unchanged
