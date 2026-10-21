"""RANDOM-ENTRY (HoE 14:08 and 14:43 UK 21 Oct; Advisor; board RANDOM-ENTRY): the random-entry check priced the
strategy's own trips on decision-candle closes, so a stop or trail exited at the close, not its fill; a trip opened
and closed on one candle was pushed out a candle; neither side paid funding. Now the strategy's side is its realised
trips (fees, fills, funding), reconciled to its equity within 0.5 points or NOT JUDGED, and each draw is filled as the
engine fills the definition's own exits (sleeve_fund.research.fills)."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import study as st
from sleeve_fund.research.fills import DrawFills, Pricer, unmodelled
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.random_entry import (MARGINAL, NOT_JUDGED, STOP_FILL_LINE, RandomEntryResult, Trade,
                                               random_entry_filled)
from sleeve_fund.strategies.definitions import to_params
from sleeve_fund.strategies.rules import SPEC as RULES
from sleeve_fund.venues import venue

INSTRUMENT = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)


def _stop_heavy(atr_mult=1.0):
    return {"version": 1, "reason": "A stop-heavy test case, written before it is run.",
            "blocks": {"fast": {"kind": "ema", "period": 10}, "slow": {"kind": "ema", "period": 30},
                       "atr": {"kind": "atr", "period": 14}},
            "long": {"entry": {"left": "fast", "op": "crosses_above", "right": "slow"},
                     "exit": {"left": "fast", "op": "<", "right": "slow"}},
            "exits": {"stop": {"level": {"sub": ["close", {"mul": [atr_mult, "atr"]}]}}}}


def _wicky(seed=5, days=700):
    """4h candles whose lows often wick 2-5% below and close back up: a stop fills deep in the wick, while the
    candle's close says nothing happened."""
    idx = pd.date_range("2022-01-01 04:00", periods=days * 6, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    c = np.round(30_000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx))
                                           + 0.006 * np.sin(np.arange(len(idx)) / 60))), 1)
    o = np.r_[c[0], c[:-1]]
    wick = np.where(rng.random(len(idx)) < 0.3, rng.uniform(0.02, 0.05, len(idx)), 0.002)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * (1 - wick),
                         "close": c, "volume": 1e9}, index=idx)


def _study(definition, prices, close_priced=False, monkeypatch=None):
    if close_priced:
        monkeypatch.setattr(st, "_draw_fill", lambda *a, **k: None)
    with tempfile.TemporaryDirectory() as d:
        return st.run_study(RULES, prices, INSTRUMENT, dataset="syn", ledger=IdeaLedger(Path(d) / "l.jsonl"),
                            synthetic=True, holdout_days=120, train_days=365, test_days=90,
                            default_params=to_params(definition))


@pytest.fixture(scope="module")
def stop_heavy():
    return _study(_stop_heavy(), _wicky())


def test_a_stop_heavy_strategys_close_priced_percentile_flattered_it(stop_heavy, monkeypatch):
    """Red first: on closes its stopped trips looked like small losses and its timing beat 81% of the draws; at its
    fills (deep in the wicks) and against draws stopped as it is, it beats 3%."""
    old = _study(_stop_heavy(), _wicky(), close_priced=True, monkeypatch=monkeypatch).random_entry
    new = stop_heavy.random_entry
    assert old.return_percentile > 80 and new.return_percentile < 10
    assert new.strategy_return < old.strategy_return
    assert new.verdict == "FAIL" and new.stopped_share > 0.3


def test_the_strategys_side_is_its_realised_trips_and_they_reconcile_to_its_equity(stop_heavy):
    re_ = stop_heavy.random_entry
    realised = [t["ret"] for f in stop_heavy.folds for t in f.trip_returns]
    assert re_.strategy_return == pytest.approx(float(np.prod(1 + np.array(realised)) - 1), abs=1e-12)
    trips, equity = re_.reconciled
    assert abs(trips - equity) <= 1e-6 and not re_.unreconciled  # exact: every flow over a trip is in its pnl
    assert "realised trips (fees, fills and funding)" in re_.words and "filled on the candles' lows" in re_.words


def test_every_counted_trip_is_compared_one_opened_on_the_close_before_the_window_too(stop_heavy):
    assert stop_heavy.random_entry.trades == sum(f.test_trades for f in stop_heavy.folds) > 0


def test_a_trip_opened_and_closed_on_one_candle_keeps_a_hold_of_0():
    """The old placement pushed it out a candle (out = max(bar(closed), entry + 1)); a stop inside a candle is out at
    that candle; nothing is placed past the window's end."""
    idx = pd.date_range("2024-01-01 04:00", periods=6, freq="4h", tz="UTC")
    at = idx[2]
    assert st._trip_bars(idx, at, at, 5) == (2, 2)
    assert st._trip_bars(idx, at, at + pd.Timedelta(hours=1), 5) == (2, 3)  # stopped inside candle 3
    assert st._trip_bars(idx, at, idx[5] + pd.Timedelta(hours=1), 4) == (2, 4)


def test_trips_that_dont_reconcile_to_the_equity_are_not_judged():
    def result(reconciled):
        return RandomEntryResult(99.0, 99.0, 0.5, 0.0, 10, 0.3, 1000, False, False, reconciled=reconciled)

    assert result((0.100, 0.1049)).verdict == "PASS"
    assert result((0.100, 0.1051)).verdict == NOT_JUDGED
    assert "not judged" in result((0.10, 0.12)).words


def test_a_hand_coded_model_keeps_the_close_priced_comparison_and_says_so():
    r = RandomEntryResult(50.0, 50.0, 0.1, 0.0, 10, 0.3, 1000, False, False)
    assert "priced on the decision candles' closes" in r.words and r.verdict == "FAIL"


# ---- the fill model ------------------------------------------------------------------------------------------------

def _defn(stop=None, trail=None, ratchet=True, short=False):
    exits = {}
    if stop is not None:
        exits["stop"] = {"level": stop}
    if trail is not None:
        exits["trail"] = {"level": trail, "ratchet": ratchet}
    rule = {"entry": {"left": "close", "op": ">", "right": 0}}
    return {"version": 1, "reason": "A test case, written before it is run.", "blocks": {},
            ("short" if short else "long"): rule, **({"exits": exits} if exits else {})}


def _candles(rows):
    idx = pd.date_range("2024-01-01 01:00", periods=len(rows), freq="1h", tz="UTC")
    o, h, lo, c = zip(*rows)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1.0}, index=idx)


FLAT = (100.0, 100.0, 100.0, 100.0)
FLOOR = 0.0005  # a stop's exit pays the taker's slippage, max(half spread, 0.05%), as the engine's stop does


def test_a_stop_fills_by_the_engines_own_replay_at_its_level_or_at_the_open_it_gapped_through(monkeypatch):
    """The draws call strategies.base.replay_missed (Advisor: the same function, never a copy): a candle reaching the
    level books it there; one that opened through books at the open, never better than the level."""
    p = _candles([FLAT, (100, 101, 97, 100.5), FLAT])
    pr = Pricer(_defn(stop={"mul": ["close", 0.98]}), p, 60, 0.001)
    r, how = pr.trip(0, 2, 1)
    assert how == "stopped" and r == pytest.approx(98 / 100 - 1 - 0.002 - FLOOR)
    gap = _candles([FLAT, (96, 97, 95, 96.5), FLAT])
    r, how = Pricer(_defn(stop={"mul": ["close", 0.98]}), gap, 60, 0.001).trip(0, 2, 1)
    assert how == "stopped" and r == pytest.approx(96 / 100 - 1 - 0.002 - FLOOR)
    from sleeve_fund.research import fills

    seen = []
    monkeypatch.setattr(fills, "replay_missed", lambda *a, **k: seen.append(a) or ("stop_loss", 90.0, 98.0, 0, 90.0))
    assert pr.trip(0, 2, 1) == (pytest.approx(90 / 100 - 1 - 0.002 - FLOOR), "stopped") and seen  # its booking, as is


def test_a_stop_fills_on_the_first_execution_bar_through_it():
    p = _candles([FLAT, (100, 101, 97, 100.5)])
    x_idx = pd.date_range("2024-01-01 00:15", periods=8, freq="15min", tz="UTC")  # four per candle
    x = pd.DataFrame({"open": [100] * 4 + [100, 98, 97.5, 98], "high": [100] * 4 + [101, 98.2, 98, 100.5],
                      "low": [100] * 4 + [99.5, 97.8, 97, 98], "close": [100] * 4 + [98, 97.5, 98, 100.5]}, index=x_idx)
    r, how = Pricer(_defn(stop={"mul": ["close", 0.985]}), p, 60, 0.0, exec_prices=x).trip(0, 1, 1)
    # The second exec bar opened through 98.5, so it books at that open; the candle alone would say the level.
    assert how == "stopped" and r == pytest.approx(98 / 100 - 1 - FLOOR)


def test_a_trail_ratchets_and_a_close_beyond_a_level_ends_the_trip_at_that_close():
    rows = [FLAT, (100, 106, 100, 105), (105, 105, 103, 104), (104, 104, 101.2, 101.5), FLAT]
    trail = {"sub": ["close", 3]}
    r, how = Pricer(_defn(trail=trail), _candles(rows), 60, 0.0).trip(0, 4, 1)
    assert how == "stopped" and r == pytest.approx(102 / 100 - 1 - FLOOR)  # trail 97 -> 102, kept past 101 at close 2
    loose = Pricer(_defn(trail=trail, ratchet=False), _candles(rows), 60, 0.0).trip(0, 4, 1)
    assert loose[1] == "closed" and loose[0] == pytest.approx(100 / 100 - 1)  # follows down; held to the cap
    beyond = Pricer(_defn(stop={"mul": ["close", 0.9]}, trail={"mul": ["close", 1.0]}), _candles([FLAT, FLAT]),
                    60, 0.0)
    assert beyond.trip(0, 1, 1) == (pytest.approx(0.0), "held back")  # a trail at the close: the engine holds it back


def test_a_draw_holds_one_of_the_strategys_periods_at_most_and_may_close_on_its_own_candle():
    p = _candles([FLAT, (100, 101, 99.5, 101), (101, 102, 100.5, 102)])
    pr = Pricer(_defn(stop={"mul": ["close", 0.9]}), p, 60, 0.001)
    assert pr.trip(0, 1, 1) == (pytest.approx(101 / 100 - 1 - 0.002), "closed")
    assert pr.trip(0, 0, 1) == (pytest.approx(-0.002), "closed")  # opened and closed on one candle: its costs


def test_funding_is_charged_at_the_settlements_a_draw_holds_through():
    p = _candles([FLAT] * 6)
    ts = p.index.as_unit("ns").asi8
    funding = (np.array([ts[0], ts[2], ts[4]]), np.array([0.001, 0.002, -0.0005]), np.array([False, False, True]))
    pr = Pricer(_defn(stop={"mul": ["close", 0.9]}), p, 60, 0.0, funding=funding)
    assert pr.trip(0, 4, 1)[0] == pytest.approx(-(0.002 + 0.0005))  # not the one at its entry close; the baseline
    short = Pricer(_defn(stop={"mul": ["close", 1.1]}, short=True), p, 60, 0.0, funding=funding)
    assert short.trip(0, 4, -1)[0] == pytest.approx(0.002 - 0.0005)  # a short receives a rate, pays a baseline


def test_a_shorts_stop_rests_above_and_fills_on_the_high():
    p = _candles([FLAT, (100, 103, 99.5, 100), FLAT])
    pr = Pricer(_defn(stop={"mul": ["close", 1.02]}, short=True), p, 60, 0.0)
    r, how = pr.trip(0, 2, -1)
    assert how == "stopped" and r == pytest.approx(-(102 / 100 - 1) - FLOOR)


def test_a_levered_draw_is_liquidated_by_the_engines_replay_before_a_stop_past_it():
    """With no stop, a 3x long whose candle wicks through its liquidation price is booked as random_entry books one:
    the margin, and the fees."""
    p = _candles([FLAT, (100, 100, 60, 99), FLAT])
    r, how = Pricer(_defn(trail={"mul": ["close", 0.5]}), p, 60, 0.001).trip(0, 2, 1, leverage=3.0)
    assert how == "liquidated" and r < -1 / 3


def test_an_exit_the_draws_dont_model_is_refused_in_words():
    slow = _defn(stop={"sub": ["close", "atr"]})
    slow["blocks"] = {"atr": {"kind": "atr", "period": 14, "timeframe": "1d"}}
    assert "slower timeframe" in unmodelled(slow, 60)
    with pytest.raises(ValueError, match="can't be filled as the engine fills"):
        Pricer(slow, _candles([FLAT] * 3), 60, 0.0)
    fill = st.DrawFill(None, 60, None, None, 0.001, refused="its draws can't be filled: test")
    assert fill.refused and math.isfinite(fill.run_taker)


# ---- the STOP-FILL gap (Advisor ruling (A), HoE 21 Oct): reported, never promoted -----------------------------------

def test_the_sheet_says_how_each_side_books_a_stop_and_counts_each_sides_stop_exits(stop_heavy):
    re_ = stop_heavy.random_entry
    stopped = [t for f in stop_heavy.folds for t in f.trip_returns if t.get("stopped")]
    assert re_.strategy_stops == len(stopped) > 0 and re_.draw_stops > 0
    assert STOP_FILL_LINE in re_.words and f"{re_.strategy_stops} stop exits in the Strategy's trips" in re_.words
    assert f"{re_.draw_stops:.1f} per draw" in re_.words


def test_the_gap_is_the_strategys_stop_exits_re_priced_at_the_draws_fill(stop_heavy):
    """The backtest books a long's stop at the bar's low less a slippage, below its level; the draws at the level less
    the taker's slippage (Advisor, PR E round 1). Each gap is what the Strategy gives up to the draws' fill, and the
    gap in total return is the realised trips compounded with them less without."""
    from sleeve_fund.instruments import taker_slippage

    re_ = stop_heavy.random_entry
    slip = float(taker_slippage(venue(str(INSTRUMENT.id.venue)).assumed_half_spread))
    trips = [t for f in stop_heavy.folds for t in f.trip_returns]
    at_level = np.array([t.get("stop_gap", 0.0) for t in trips])
    gaps = at_level - np.array([t.get("stop_level", 0.0) for t in trips]) * slip
    realised = np.array([t["ret"] for t in trips])
    assert (gaps[[bool(t.get("stopped")) for t in trips]] > 0).all() and gaps.sum() > 0
    assert re_.gap_return == pytest.approx(float(np.prod(1 + realised + gaps) - np.prod(1 + realised)), abs=1e-12)
    assert re_.gap_return < float(np.prod(1 + realised + at_level) - np.prod(1 + realised))  # not the bare level
    assert re_.gap_percentile >= re_.return_percentile
    assert "percentile points" in re_.words


def test_a_stop_exits_gap_is_side_times_level_less_fill_over_entry():
    long_ = {"side": 1, "entry_px": 100.0, "exit_px": 94.9}
    stop = {"intent": "stop_loss", "signal": {"entry_px": 100.0, "stop_loss": 0.05, "trigger": 95.0}}
    assert st._stop_gap(long_, stop)["stop_gap"] == pytest.approx(0.001)
    assert st._stop_gap({"side": -1, "entry_px": 100.0, "exit_px": 105.1},
                        {"intent": "stop_loss", "signal": {"stop_loss": 0.05}})["stop_gap"] == pytest.approx(0.001)
    assert st._stop_gap(long_, {"intent": "exit", "signal": {}}) == {} and st._stop_gap(long_, None) == {}


def _gapped(return_percentile, gap_percentile, reconciled=(0.0, 0.0), why_not=""):
    return RandomEntryResult(return_percentile, return_percentile, 0.1, 0.0, 10, 0.3, 1000, False, False,
                             reconciled=reconciled, why_not=why_not, strategy_stops=4, draw_stops=3.0,
                             gap_return=0.02, gap_percentile=gap_percentile)


def test_a_fail_the_gap_would_turn_into_a_pass_reads_marginal_and_stays_a_fail():
    r = _gapped(90.0, 96.0)
    assert r.verdict == "FAIL" and r.marginal and r.words.startswith(MARGINAL)
    assert "+6 percentile points" in r.words
    for other in (_gapped(90.0, 94.0), _gapped(96.0, 99.0), _gapped(90.0, 96.0, (0.0, 0.01)),
                  _gapped(90.0, 96.0, why_not="test"), _gapped(90.0, None, None)):
        assert not other.marginal and MARGINAL not in other.words


class _Steps:
    """A stub pricer: each draw's return is set by its entry bar, so the draws are fixed whatever the gap."""

    def draws(self, entries, holds, sides, levs):
        return DrawFills(np.asarray(entries) % 20 / 1000 - 0.01, 0, 1)


def test_the_gap_percentile_is_read_against_the_same_draws():
    trades, windows = [Trade(10, 12, 1)], [(0, 99)]
    def run(gap):
        return random_entry_filled(trades, np.array([0.0]), windows, _Steps(), (0.0, 0.0), draws=500,
                                   gaps=np.array([gap]), stops=np.array([gap > 0]))
    none, small, big = run(0.0), run(0.003), run(0.05)
    assert none.gap_percentile == none.return_percentile and none.strategy_stops == 0
    assert none.return_percentile < small.gap_percentile < 95 <= big.gap_percentile == 100
    assert big.strategy_return == none.strategy_return and big.verdict == "FAIL" and big.marginal
    assert big.gap_return == pytest.approx(0.05) and big.strategy_stops == 1 and big.draw_stops == 1


# ---- PR E round 1 (Advisor): the draws pay what the engine pays; too many held back is NOT JUDGED -----------------

def test_a_draws_stop_exit_pays_the_takers_slippage_floor_and_a_market_exit_the_half_spread():
    """The engine's market entries and signal exits pay the half spread; its stop pays max(half spread, 0.05%)."""
    stop = [FLAT, (100, 101, 97, 100.5), FLAT]
    wide = Pricer(_defn(stop={"mul": ["close", 0.98]}), _candles(stop), 60, 0.0011, half_spread=0.001)
    assert wide.trip(0, 2, 1) == (pytest.approx(98 / 100 - 1 - 0.0011 - 0.0011), "stopped")  # the spread is wider
    tight = Pricer(_defn(stop={"mul": ["close", 0.98]}), _candles(stop), 60, 0.0003, half_spread=0.0002)
    assert tight.trip(0, 2, 1) == (pytest.approx(98 / 100 - 1 - 0.0003 - 0.0006), "stopped")  # fee + the floor
    closed = Pricer(_defn(stop={"mul": ["close", 0.9]}), _candles([FLAT, FLAT]), 60, 0.0003, half_spread=0.0002)
    assert closed.trip(0, 1, 1) == (pytest.approx(-0.0006), "closed")  # entry and close exit: fee + half spread


def test_a_study_passes_its_half_spread_to_the_draws(monkeypatch):
    seen = []
    real = st.DrawFill.pricer
    monkeypatch.setattr(st.DrawFill, "pricer", lambda self, p, c, h=0.0: seen.append(h) or real(self, p, c, h))
    _study(_stop_heavy(), _wicky())
    assert seen and all(np.all(np.asarray(h) > 0) for h in seen)


def test_too_many_held_back_draws_are_not_judged():
    """A long whose exit_at_level sits at the open less 0.5 is on the wrong side of the close on most down candles,
    so the engine would hold back many draws; priced flat, they pull the draws to zero."""
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    o = np.r_[100.0, c[:-1]]
    rows = list(zip(o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c))
    pr = Pricer(_defn(stop={"mul": ["close", 0.9]}) | {"exits": {"stop": {"level": {"mul": ["close", 0.9]}},
                                                                  "exit_at_level": {"level": {"sub": ["open", 0.5]}}}},
                _candles(rows), 60, 0.001)
    trades = [Trade(10 + 20 * k, 15 + 20 * k, 1) for k in range(15)]
    r = random_entry_filled(trades, np.zeros(15), [(0, 399)], pr, (0.0, 0.0), draws=200)
    assert r.held_back_share > 0.02 and r.verdict == NOT_JUDGED and "too many draws held back" in r.words
    ok = RandomEntryResult(99.0, 99.0, 0.5, 0.0, 10, 0.3, 1000, False, False, reconciled=(0.0, 0.0), held_back=2,
                           held_back_share=0.02)
    assert ok.verdict == "PASS" and "(2.0%) priced flat" in ok.words


# ---- CR, PR E early findings (4363afb4) -----------------------------------------------------------------------------

class _Flat:
    """A stub pricer: every draw makes nothing, so only the strategy's side moves."""

    def draws(self, entries, holds, sides, levs):
        return DrawFills(np.zeros(len(entries)), 0, 0)


def _one_trip_benchmark(taker, charged=True, charge=0.0):
    from types import SimpleNamespace

    idx = pd.date_range("2024-01-01 04:00", periods=40, freq="4h", tz="UTC")
    prices = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    trip = {"opened": idx[15], "closed": idx[18], "side": 1, "ret": 0.05, "exit_ratio": 1.1}
    fold = SimpleNamespace(test_end=idx[-1], trip_returns=[trip], trip_sizes=[], chosen={},
                           trip_charges=[(idx[15], charge)] if charge else [], trip_reconcile=None, window_bars=30,
                           in_market_bars=3)
    fill = SimpleNamespace(refused="", definition={"stub": True}, run_taker=0.001,
                           pricer=lambda prices, cost, half=0.0: _Flat())
    return st._benchmarks(prices, [fold], 30, taker, 0.0, False, None, None, charged, fill)[0]


def test_a_rungs_fee_takes_the_place_of_the_runs_on_both_legs_of_each_realised_trip():
    """RE-COST at a cost-ladder rung: a trip that paid the run's 0.1% a side pays the rung's 0.3% instead, on its entry
    notional and on its exit's (1.1 times it here): 0.05 - 0.002 x (1 + 1.1)."""
    assert _one_trip_benchmark(0.001).strategy_return == pytest.approx(0.05)
    assert _one_trip_benchmark(0.003).strategy_return == pytest.approx(0.05 - 0.002 * 2.1)


def test_the_uncharged_comparison_adds_back_the_absorb_gap_charge():
    assert _one_trip_benchmark(0.001, charged=False, charge=0.01).strategy_return == pytest.approx(0.06)
    assert _one_trip_benchmark(0.001, charged=True, charge=0.01).strategy_return == pytest.approx(0.05)


def test_a_draw_whose_level_isnt_known_yet_is_held_back_and_counted():
    p = _candles([FLAT, FLAT, (100, 100, 97, 100), FLAT])
    pr = Pricer(_defn(stop={"prev": {"mul": ["close", 0.98]}}), p, 60, 0.001)
    assert pr.trip(0, 2, 1) == (0.0, "held back")  # no close before the first: no level yet
    assert pr.trip(1, 2, 1)[1] == "stopped"
    fills_ = pr.draws(np.array([0, 1, 0]), np.array([2, 2, 1]), np.array([1, 1, 1]))
    assert (fills_.held_back, fills_.stopped) == (2, 1)
