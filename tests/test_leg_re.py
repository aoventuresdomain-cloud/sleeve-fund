"""LEG-RE (Advisor 6 Oct 22:37): rsi_cross and dip_buy no longer re-enter the same side on the candle a close-fired
exit (time stops included) closed them on; the earliest re-entry is the next candle. Before, such a re-entry restarted
the leg with nothing traded, so the position silently outlived its time stop. The opposite side on the exit candle is a
declared reversal and stays."""

import numpy as np
import pandas as pd

from sleeve_fund.research.runner import run_backtest
from test_sprint_models import _dip
from test_test_strategies import _cross, _walk

PERP = {"market": "perp", "allow_short": True}


def test_rsi_cross_closes_on_its_time_stop_even_when_the_same_side_crosses_again_on_that_candle():
    """A long from the cross at 31; three candles on, the time stop and a fresh cross back above 30 land together. The
    leg ends there (it used to restart, held), and the next candle's cross may open a new one."""
    s = _cross(time_stop_bars=3)
    assert _walk(s, [25, 31, 40, 29, 31, 29, 31]) == [0, 1, 1, 1, 0, 0, 1]
    #                                  exit candle ^   ^ 29: no cross   ^ next fresh cross: a new long
    s = _cross(time_stop_bars=3)
    _walk(s, [25, 31, 40, 29])
    assert s.target_side(31.0, 29.0) == 0  # time stop + same-side cross on one candle: flat
    assert "time stop" in s._why[0] and "a new long waits for the next candle" in s._why[0]


def test_rsi_cross_short_leg_is_held_to_the_same_rule():
    s = _cross(time_stop_bars=2)
    assert _walk(s, [75, 69, 72, 69]) == [0, -1, -1, 0]  # time stop at the 2nd candle, with a fresh cross below 70


def test_rsi_cross_reverses_on_an_opposite_cross_on_the_exit_candle():
    """A declared reversal: the long ends on its time stop, and RSI crossing back below 70 on that candle opens the
    short there."""
    s = _cross(long_exit=95.0, short_entry=70.0, time_stop_bars=2)
    assert _walk(s, [25, 31, 75, 69]) == [0, 1, 1, -1]


def test_dip_buy_closes_on_its_time_stop_while_the_dip_still_holds_and_buys_again_a_candle_later():
    s = _dip(time_stop_bars=2)
    dip = (95.0, 4.0, 2.0, 100.0, 94.0, 98.0, 1)  # RSI 4, 2.5 ATR under the high, daily up-trend, under the average
    assert [s.target_side(*dip) for _ in range(4)] == [1, 1, 0, 1]
    #                                          time stop ^  ^ the next candle: a new long, the dip still there
    s = _dip(time_stop_bars=2)
    s.target_side(*dip), s.target_side(*dip)
    assert s.target_side(*dip) == 0
    assert "time stop" in s._why[0] and "a new long waits for the next candle" in s._why[0]


def _minutes(n=3000, seed=5):
    """QA's p1-5 walk (quant-review/p1-5-xfails, _minutes): drifting 1-minute bars with sharp dips every 170."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.00008, 0.0012, n)
    vol = rng.uniform(0.5, 1.5, n)
    for k in range(300, n, 170):
        r[k:k + 6] -= 0.0025
        vol[k:k + 6] *= 4
    c = 60_000 * np.exp(np.cumsum(r))
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.date_range(pd.Timestamp("2025-10-01", tz="UTC") + pd.Timedelta(minutes=1), periods=n, freq="1min")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + rng.uniform(0, 4e-4, n)),
                         "low": np.minimum(o, c) * (1 - rng.uniform(0, 4e-4, n)), "close": c, "volume": vol * 1e3},
                        index=idx)


def test_in_a_backtest_no_rsi_cross_leg_outlives_its_time_stop(instrument):
    """On QA's walk, legacy rsi_cross held 13 of its 252 legs past the 6-candle time stop (up to 18), each restarted by
    a same-side cross on its exit candle. Now every leg ends by its time stop at the latest: one round trip each."""
    df = _minutes()
    res = run_backtest("rsi_cross", df, instrument, {"rsi_period": 5, "time_stop_bars": 6, "trend_sma": 0, **PERP},
                       bar_minutes=1, half_spread=0)
    at = {t: i for i, t in enumerate(df.index)}
    f = res.fills.sort_values("ts_last")
    pos, start, held = 0.0, None, []
    for o in f.index:
        qty = float(f.loc[o, "filled_qty"]) * (1 if f.loc[o, "side"] == "BUY" else -1)
        before, pos = pos, pos + qty
        pos = 0.0 if abs(pos) < 1e-12 else pos
        if before and (not pos or (before > 0) != (pos > 0)):
            held.append(at[f.loc[o, "ts_last"]] - at[start])
        if pos and (not before or (before > 0) != (pos > 0)):
            start = f.loc[o, "ts_last"]
    assert len(held) > 200
    assert max(held) <= 6, sorted(held)[-5:]
