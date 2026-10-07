"""QA Tester 2: pre-build pins for LEG-RE (Advisor 22:37, owned by the Quant Developer, before strategy testing), written
BEFORE the build. Copy this file into tests/ to run it (it uses the suite's own `instrument` and `prices` fixtures).

Ruling (eng-board.md 23:55, Advisor 22:37): the legacy models rsi_cross and dip_buy get NO same-side re-entry on a
close-fired exit bar (time stops included); k+1's close is the earliest. Opposite-side entries on the exit bar stay as
declared reversals (rsi_bands, ping_pong, rsi_cross's opposite cross). The affected trials are re-run; the old rows are
kept in N, marked "superseded (same-candle re-entry)". The p1-5 rsi_cross port pin 1 then flips to exact parity.

How the bug shows today (measured on #161's base 0272f18, where an exit and a same-candle re-entry net out, the P1-5-X1
"swallowed exit"): no explicit re-entry fill appears; the leg simply runs PAST its time stop. rsi_cross on 3,000 one-
minute bars, time stop 6: legs held up to 21 bars (26-36 legs over the limit per seed); dip_buy on 4-hour candles, time
stop 2: legs held 6 bars. So the strict pins are "no leg is held longer than its time stop" and, after the fix, the
fills-based pins (no same-side entry on the candle a flatten filled) keep it honest. Those fills-based pins and the
parity pin PASS today (the bug hides the re-entry) and must keep passing: they are unmarked controls.

Marks: `xfail(strict=True, raises=AssertionError, reason="LEG-RE")`. No interface is assumed except one, named in the
trials pin (TrialsRegister.supersede), which PE or QD should rename freely.
"""

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.runner import run_backtest

LEG = pytest.mark.xfail(strict=True, raises=AssertionError, reason="LEG-RE")
PERP = {"market": "perp", "allow_short": True}
START = pd.Timestamp("2025-10-03", tz="UTC")


def _minutes(n, seed):
    """1-minute bars stamped at their close: a drifting walk with sharp heavy-volume dips every 170 minutes (the data
    of quant-review/p1-5-xfails, so legacy models trade on it)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.00008, 0.0012, n)
    vol = rng.uniform(0.5, 1.5, n)
    for k in range(300, n, 170):
        r[k:k + 6] -= 0.0025
        vol[k:k + 6] *= 4
    c = 60_000 * np.exp(np.cumsum(r))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + rng.uniform(0, 4e-4, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 4e-4, n))
    idx = pd.date_range(START + pd.Timedelta(minutes=1), periods=n, freq="1min")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": vol * 1e3}, index=idx)


def _candles(m, minutes):
    return m.resample(f"{minutes}min", closed="right", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def _fills(res):
    """(intent, side, close time, qty) per fill, in time order."""
    if res.fills is None or len(res.fills) == 0:
        return []
    f = res.fills.sort_values(["ts_last", "ts_init"], kind="stable")
    return [(res.decisions[o]["intent"], f.loc[o, "side"], f.loc[o, "ts_last"], float(f.loc[o, "filled_qty"]))
            for o in f.index]


def _legs(fills):
    """[(open time, close time, direction)] for each leg. Robust to the order of fills that share a timestamp (a declared
    reversal's exit and its opposite entry fill on the same close): within one timestamp the non-entry fills come first,
    and a leg also ends when the position reaches flat or changes sign (one fill through flat), so two legs are never
    joined by a sort order. (Found on a12561b: an entry that sorted before its reversal's exit joined two legs.)"""
    order = sorted(range(len(fills)), key=lambda i: (fills[i][2], fills[i][0] == "entry", i))
    pos, t0, d, out = 0.0, None, 0, []
    for i in order:
        _intent, side, t, qty = fills[i]
        before, pos = pos, pos + (qty if side == "BUY" else -qty)
        if abs(pos) < 1e-9:
            pos = 0.0
        if before and (not pos or before * pos < 0):
            out.append((t0, t, d))
        if pos and (not before or before * pos < 0):
            t0, d = t, 1 if pos > 0 else -1
    return out


def _same_side_reentries(fills, minutes):
    """Entries in a leg's own direction that fill on the decision candle (close - minutes, close] in which that leg's
    flattening fill happened. Candle-aware, so it holds for 15-minute and 4-hour decisions with 1-minute execution."""
    def candle(t):
        return pd.Timestamp(t).ceil(f"{minutes}min")

    out, last = [], None
    pos = 0.0
    for i, (_intent, side, t, qty) in enumerate(fills):
        before, pos = pos, pos + (qty if side == "BUY" else -qty)
        if abs(pos) < 1e-9:
            pos = 0.0
        if before and not pos:
            last = (candle(t), before > 0)
        elif not before and pos and last == (candle(t), pos > 0):
            out.append(i)
    return out


def _over_stop(res, step_minutes, stop_bars):
    """Legs held longer than the time stop, in decision bars (no bars are missing in this data)."""
    return [(a, b) for a, b, _ in _legs(_fills(res)) if (b - a) / pd.Timedelta(minutes=step_minutes) > stop_bars]


def test_a_reversal_at_one_timestamp_gives_two_legs_in_either_sort_order():
    """Control for `_legs`: a short closed and a long opened on the same close are two legs, whichever fill comes first,
    and so is one fill through flat."""
    t0, t1, t2 = (pd.Timestamp("2025-10-03 00:00", tz="UTC") + pd.Timedelta(minutes=m) for m in (1, 2, 4))
    short_in, cover, long_in, long_out = (("entry", "SELL", t0, 1.0), ("exit", "BUY", t1, 1.0),
                                          ("entry", "BUY", t1, 0.9), ("exit", "SELL", t2, 0.9))
    for fills in ([short_in, cover, long_in, long_out], [short_in, long_in, cover, long_out]):
        assert _legs(fills) == [(t0, t1, -1), (t1, t2, 1)]
    assert _legs([("entry", "SELL", t0, 1.0), ("entry", "BUY", t1, 1.9), ("exit", "SELL", t2, 0.9)]) == [
        (t0, t1, -1), (t1, t2, 1)]


# ---------------------------------------------------------------------------------------------------- rsi_cross
RSI = {"rsi_period": 3, "trend_sma": 0, "long_entry": 30.0, "long_exit": 80.0}


@pytest.mark.parametrize("seed, stop, perp", [(1, 2, True), (1, 6, True), (3, 6, True), (2, 6, False)],
                         ids=["seed1-stop2-perp", "seed1-stop6-perp", "seed3-stop6-perp", "seed2-stop6-spot"])
def test_a_rsi_cross_leg_is_never_held_past_its_time_stop(seed, stop, perp, instrument):
    """[Advisor 22:37] The time stop fires at bar `stop`; a same-side re-entry on that bar must not swallow it. Today
    legs run on (up to 21 bars on a 6-bar stop)."""
    res = run_backtest("rsi_cross", _minutes(3000, seed), instrument,
                       {**RSI, "time_stop_bars": stop, **(PERP if perp else {})}, bar_minutes=1, half_spread=0)
    assert len(_legs(_fills(res))) >= 100, "the case needs legs"
    assert _over_stop(res, 1, stop) == []


@pytest.mark.parametrize("seed, stop", [(1, 2), (3, 6)])
def test_rsi_cross_opens_no_same_side_entry_on_the_candle_its_leg_was_flattened_on(seed, stop, instrument):
    """Control, passes today (the re-entry hides inside the swallowed exit) and must keep passing after the fix,
    on 15-minute decisions with 1-minute execution as well as on 1-minute ones."""
    m = _minutes(15 * 800, seed)
    res = run_backtest("rsi_cross", _candles(m, 15), instrument, {**RSI, "time_stop_bars": stop, **PERP},
                       bar_minutes=15, exec_prices=m, exec_minutes=1, half_spread=0)
    f = _fills(res)
    assert len(_legs(f)) >= 20
    assert _same_side_reentries(f, 15) == []
    res1 = run_backtest("rsi_cross", m.iloc[:3000], instrument, {**RSI, "time_stop_bars": stop, **PERP},
                        bar_minutes=1, half_spread=0)
    assert _same_side_reentries(_fills(res1), 1) == []


# ----------------------------------------------------------------------------------------------------- dip_buy
DIP = {"rsi_period": 2, "rsi_entry": 40.0, "dip_atr": 0.1, "dip_atr_bars": 5, "exit_sma": 50, "trend_sma_days": 5,
       "trend_ema_days": 3, "stop_atr": 20.0}


def _dip_run(seed, stop, instrument):
    m = _minutes(1440 * 60, seed)
    return run_backtest("dip_buy", _candles(m, 240), instrument, {**DIP, "time_stop_bars": stop, **PERP},
                        bar_minutes=240, half_spread=0)


@pytest.mark.parametrize("seed, stop", [(1, 2), (3, 2), (1, 6)])
def test_a_dip_buy_leg_is_never_held_past_its_time_stop(seed, stop, instrument):
    res = _dip_run(seed, stop, instrument)
    assert len(_legs(_fills(res))) >= 5, "the case needs legs"
    assert _over_stop(res, 240, stop) == []


@pytest.mark.parametrize("seed, stop", [(1, 2), (3, 2)])
def test_dip_buy_opens_no_same_side_entry_on_the_candle_its_leg_was_flattened_on(seed, stop, instrument):
    """Control, passes today and must keep passing."""
    f = _fills(_dip_run(seed, stop, instrument))
    assert len(_legs(f)) >= 5
    assert _same_side_reentries(f, 240) == []


# -------------------------------------------------------------------------- declared reversals stay (controls)
def test_a_declared_reversal_still_closes_and_opens_the_other_way_on_the_same_candle(prices, instrument):
    """Control, passes today and must keep passing after the fix: ping_pong turns from long to short (and back) on one
    candle: its exit and the opposite entry share a close. (tests/test_long_short.py's own path.)"""
    from test_backtest import _path

    closes = [100.0, 100.5, 100.8, 101.5, 101.3, 101.2, 100.9, 101.0, 101.5, 102.0, 102.0]
    res = run_backtest("ping_pong", _path(prices, closes), instrument, PERP, half_spread=0)
    f = _fills(res)
    assert [x[0] for x in f] == ["entry", "exit", "entry", "exit", "entry", "exit", "entry"]
    assert [x[1] for x in f] == ["BUY", "SELL", "SELL", "BUY", "BUY", "SELL", "SELL"]
    assert f[1][2] == f[2][2] and f[3][2] == f[4][2] and f[5][2] == f[6][2], "each reversal shares its exit's close"


# --------------------------------------------------------------------------------------- the p1-5 pin flips to exact
# Moved to the P1-5 master (p1-5-xfails/test_p1_5_xfails.py::test_the_rsi_cross_port_matches_rsi_cross_exactly_again),
# HoE 7 Oct: #172 merges before #161, and #161 carries the rule builder and the p1-5 helpers this pin needs.


# ----------------------------------------------------------------------------------------------- the trials register
@LEG
def test_the_old_rows_of_the_affected_models_are_kept_in_n_and_marked_superseded(tmp_path):
    """[Advisor 22:37] The affected trials are re-run; the old rows stay in N, marked 'superseded (same-candle
    re-entry)'. ASSUMED interface (QD to rename): TrialsRegister.supersede(strategy, reason) marks every counted row of
    that model and returns how many. Pinned: N is unchanged by superseding; the rows carry the marking; other models'
    rows are untouched; a re-run adds one row that is counted and is not superseded; superseding twice marks nothing
    new."""
    from sleeve_fund.research.trials import TrialsRegister, record_model_run, run_setup
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/t.db")
    reg = TrialsRegister(store)
    setup = run_setup(risk_profile="balanced", fee=0.0005)
    for strategy, p in (("rsi_cross", {"time_stop_bars": 6}), ("rsi_cross", {"time_stop_bars": 12}),
                        ("dip_buy", {"time_stop_bars": 6}), ("rsi_bands", {})):
        record_model_run(store, strategy=strategy, params=p, dataset="d", source="backtest", setup=setup, sharpe=0.5)
    before = reg.counts()
    assert before["evaluations"] == 4
    supersede = getattr(reg, "supersede", None)
    assert supersede is not None, "not built: TrialsRegister.supersede"
    reason = "superseded (same-candle re-entry)"
    n1 = supersede("rsi_cross", reason) + supersede("dip_buy", reason)
    assert n1 == 3
    assert reg.counts()["evaluations"] == before["evaluations"] and reg.counts()["variants"] == before["variants"]
    status = {(t["definition_name"], t["settings"]): t.get("status") for t in store.trials()}
    marked = [k for k, v in status.items() if v == reason]
    assert sorted(k[0] for k in marked) == ["dip_buy", "rsi_cross", "rsi_cross"]
    assert status[("rsi_bands", "{}")] != reason
    assert supersede("rsi_cross", reason) == 0, "superseding again marks nothing new"
    record_model_run(store, strategy="rsi_cross", params={"time_stop_bars": 6}, dataset="d", source="backtest",
                     setup=setup, sharpe=0.4)
    assert reg.counts()["evaluations"] == 5
    assert sum(1 for t in store.trials() if t.get("status") == reason) == 3, "the re-run is not superseded"
