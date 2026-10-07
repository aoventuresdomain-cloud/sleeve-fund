"""QA Tester 2, #161 delta round: a strict xfail repro. Copy into tests/ of a worktree (needs tests/m_p15.py = the p1-5 master for D, B, C, SIDE), never run it in this folder."""
import pandas as pd
import pytest
from types import SimpleNamespace
from nautilus_trader.model import Bar
from sleeve_fund.data import bar_type_for
from sleeve_fund.strategies.definitions import to_params
from sleeve_fund.strategies.rules import Rules, RulesConfig
from tests.m_p15 import D, B, C, SIDE


def test_flip_after_target_ignores_the_slower_candle_hold(instrument):
    d = D({"ema1h": B("ema", period=3, timeframe="1h")}, long=SIDE(C("close", ">", "ema1h"), C("close", "<", "ema1h")),
          short=SIDE(C("close", "<", "ema1h"), C("close", ">", "ema1h")))
    bt = bar_type_for(instrument, 15)
    s = Rules(RulesConfig(instrument_id=instrument.id, bar_type=bt, assumed_taker_fee=0.001, **{**to_params(d), "market": "perp", "allow_short": True}))
    ts = pd.Timestamp("2025-10-03 01:00", tz="UTC").value; p = instrument.make_price
    bar = Bar(bt, p(100.0), p(101.0), p(99.0), p(100.0), instrument.make_qty(15), ts, ts)
    s.want_side = lambda b: -1  # the signal has turned short while a long's target traded
    s._pending_exit = "O-target"
    # the slower candles' warm-up is NOT met (a restart whose history is short): no entry may be decided
    s._short_history = "its 1-HOUR candles need 30 closed ones of history and 2 loaded"
    s._slower = [SimpleNamespace(count=2, need=30)]
    s._flip_after_target(bar, 1)
    print("\nFLIP", s._flip is not None, "held notice:", "entry_held" in s._noted)
    assert s._flip is None, "a reversal after the target must respect the slower-candle hold (_entry_held), as the other entry paths do"
