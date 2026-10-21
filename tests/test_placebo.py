"""R-24 (HoE 10:28 UK 21 Oct; Advisor 10:49 UK): a filtered variant (V1, V2, D) against a placebo of its parent, the
same share of the parent's entry signals blocked at random, 1,000 seeded draws on a fast path calibrated once per
variant against the full engine (parent, variant, three gated draws: the same trips and a net return within 0.5 pp,
else NOT JUDGED; the tolerance is never widened)."""

import copy
import math
import tomllib
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.instruments import FeeSchedule
from sleeve_fund.research import placebo as P
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.metrics import fills_to_rows, trades
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.research.study import CONTINUOUS, run_study
from sleeve_fund.research.tearsheet import render
from sleeve_fund.strategies import check_study_only
from sleeve_fund.strategies.definitions import to_params
from sleeve_fund.strategies.rules import SPEC as RULES
from sleeve_fund.venues import venue

ROOT = Path(__file__).resolve().parents[1]
QR_B1 = tomllib.loads((ROOT / "configs" / "definitions" / "qr01_trend_ema_long.toml").read_text())
QR_B2 = tomllib.loads((ROOT / "configs" / "definitions" / "qr01b_trend_multispeed_long.toml").read_text())
INSTRUMENT = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
FEE = 0.001
SIZE = 1 - 0.01 - FEE  # the engine's invested share on spot: less its 1% cash buffer and the entry fee
START = 6 * 60  # the first traded 4h candle: 60 days of warm-up
QUIET, LOUD = 1e6, 5e6


def _hours(days=300, seed=1):
    rng = np.random.default_rng(seed)
    n = 24 * days
    idx = pd.date_range("2023-01-01 01:00", periods=n, freq="1h", tz="UTC")
    c = 20_000 * np.exp(np.cumsum(rng.normal(0.00005, 0.006, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": QUIET}, index=idx).round(1)


def _candles(hours):
    bars = hours.resample("4h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return bars[bars.index <= hours.index[-1]]


HOURS = _hours()
BARS = _candles(HOURS)  # 4 x QUIET a candle
FAST = dict(exec_prices=HOURS, stop_slippage=0.0005, size=SIZE)


def _loud(candles) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The fixture with these 4h candles' hours trading LOUD, as (bars, hours, the fast path's options), for the fast
    path's own cells (the engine reads its book's share of the volume, so its cells use _spiky)."""
    hours = HOURS.copy()
    for i in candles:
        inside = (hours.index > BARS.index[i] - pd.Timedelta(hours=4)) & (hours.index <= BARS.index[i])
        hours.loc[inside, "volume"] = LOUD
    return _candles(hours), hours, {**FAST, "exec_prices": hours}


def _spiky(candles) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The fixture with a spike in each of these 4h candles' last hour, its high 3% over the candle's close, as
    (bars, hours, the fast path's options): a filter both the engine and the fast path read alike (the engine's
    candle volume is its book's share, not the stored one)."""
    hours = HOURS.copy()
    for i in candles:
        at = hours.index.get_loc(BARS.index[i])
        hours.iloc[at, hours.columns.get_loc("high")] = round(BARS["close"].iloc[i] * 1.03, 1)
    return _candles(hours), hours, {**FAST, "exec_prices": hours}


def _calm(parent: dict) -> dict:
    """The parent with one more entry condition: the candle's high within 2% of its close."""
    v = copy.deepcopy(parent)
    v["long"]["entry"] = {"all": [parent["long"]["entry"],
                                  {"left": "high", "op": "<", "right": {"mul": ["close", 1.02]}}]}
    return v


def _filtered(parent: dict, cap: float = 8 * QUIET) -> dict:
    """The parent with one more entry condition: the candle's volume under `cap` (a filter the test sets candles for)."""
    v = copy.deepcopy(parent)
    v["long"]["entry"] = {"all": [parent["long"]["entry"], {"left": "volume", "op": "<", "right": cap}]}
    return v


def _engine(bars, hours=HOURS):
    """The full engine on the same window, as a study's window runs it: trading from START, on hourly execution bars."""
    def full(defn, blocked):
        p = {**to_params(defn), "trade_from": int(bars.index[START - 1].value) + 1}
        if blocked:
            p["blocked_entries"] = [int(bars.index[b].value) for b in blocked]
        r = run_backtest("rules", bars, INSTRUMENT, params=p, bar_minutes=240, half_spread=0, starting_capital=10_000,
                         exec_prices=hours, exec_minutes=60,
                         fees=FeeSchedule(maker=Decimal(str(FEE)), taker=Decimal(str(FEE))))
        t = trades(fills_to_rows(r.fills), r.shorts, open_trip=True)
        return ([(x["opened"], x["closed"], x["side"]) for x in t if x["closed"] is not None],
                float(r.equity.iloc[-1] / 10_000 - 1))
    return full


def _as_fills(run: P.FastRun, index):
    """A fast run's trips as the engine's fill times: at the candle's close, a level a minute inside it."""
    return [(index[e], index[x] - pd.Timedelta(minutes=1) if s else index[x], 1)
            for (e, x, _), s in zip(run.trips, run.stopped)]


# ---- (2) red first: a filter that blocks nothing, and a known best subset -------------------------------------------

def test_a_filter_that_blocks_nothing_gives_the_parents_result_in_every_draw():
    r = P.placebo(QR_B1, _filtered(QR_B1, cap=1e12), BARS, 240, START, FEE, draws=50, **FAST)
    parent = P.fast_path(QR_B1, BARS, 240, START, FEE, **FAST)
    assert r.signals > 5 and r.blocked == 0 and r.share == 0
    assert np.array_equal(r.returns, np.full(50, parent.net)) and r.variant_return == parent.net
    assert r.percentile == 0  # the same as every draw is beating none of them


def test_on_a_fixture_with_a_known_best_subset_the_percentile_is_exact():
    """QR-B1's entries are crosses, one candle each, so blocking one removes exactly its trip and no other: a draw's
    return is the parent's trips less the blocked ones. The filter blocks exactly the losers: the best subset of that
    size, so no draw beats it, and the percentile is the share of draws that didn't block those very trips."""
    parent = P.fast_path(QR_B1, BARS, 240, START, FEE, **FAST)
    entries = [e for e, _, _ in parent.trips]
    losers = [e for (e, _, _), r in zip(parent.trips, parent.returns) if r < 0]
    assert len(entries) >= 6 and 2 <= len(losers) < len(entries)
    bars, _, fast = _loud(losers)
    r = P.placebo(QR_B1, _filtered(QR_B1), bars, 240, START, FEE, draws=400, seed=3, **fast)
    signals = np.flatnonzero(parent.holds & (np.arange(len(bars)) >= START))
    assert r.signals == len(signals) and r.blocked == len(losers)
    by_entry = dict(zip(entries, parent.returns))
    still = parent.open_return
    open_entry = next((i for i in signals if i not in by_entry), None)  # a trip still open at the end, if any

    def closed_form(blocked):
        kept = np.prod([1 + x for e, x in by_entry.items() if e not in blocked])
        return float(kept * (1 + (still if open_entry not in blocked else 0.0)) - 1)

    mask = (bars["volume"] >= 8 * QUIET).to_numpy()  # the filter's own mask: the candles it fails on
    offsets = P.shift_offsets(len(bars) - START, 240, 400, 3)
    want = np.array([closed_form(P.shifted_blocks(mask, parent.holds, START, o)) for o in offsets])
    assert np.allclose(r.returns, want, atol=1e-12)
    assert math.isclose(r.variant_return, closed_form(set(losers)), abs_tol=1e-12)
    assert (want <= r.variant_return + 1e-12).all()  # the best subset: no draw beats it
    assert r.percentile == pytest.approx(100 * float((want < r.variant_return).mean()))
    assert r.percentile > 95


def test_each_draw_shifts_the_filters_whole_mask_by_a_seeded_offset_of_at_least_30_days():
    """Advisor 11:28 UK: a circular shift keeps the mask's share and its runs (top-decile funding comes in runs of
    days), where a uniform draw by candle would scatter them."""
    k = 30 * 6  # 30 days of 4h candles
    offsets = P.shift_offsets(1_000, 240, 200, 0)
    assert offsets == P.shift_offsets(1_000, 240, 200, 0) != P.shift_offsets(1_000, 240, 200, 1)
    assert min(offsets) >= k and max(offsets) <= 1_000 - k and len(set(offsets)) > 100
    mask = np.zeros(1_100, dtype=bool)
    mask[300:360] = mask[700:712] = True  # two runs, inside the scored window from 100
    holds = np.ones(1_100, dtype=bool)
    got = P.shifted_blocks(mask, holds, 100, 250)
    assert sorted(got) == list(range(550, 610)) + list(range(950, 962))  # the same runs, 250 candles on
    wrapped = P.shifted_blocks(mask, holds, 100, 450)
    assert sorted(wrapped) == list(range(150, 162)) + list(range(750, 810))  # round the window, never before it
    holds[::2] = False
    assert P.shifted_blocks(mask, holds, 100, 250) == frozenset(i for i in got if i % 2)  # signals only
    with pytest.raises(ValueError, match="under 60 days"):
        P.shift_offsets(2 * k, 240, 1, 0)


def test_only_a_variant_that_is_its_parent_with_entry_conditions_added_is_a_filter():
    from sleeve_fund.research.variants import PREREGISTERED, definition_path

    for vid in ("V1", "V2"):
        v = PREREGISTERED[vid]
        defn, parent = (tomllib.loads(definition_path(v[k]).read_text()) for k in ("definition", "parent"))
        only = P.filter_of(parent, defn)
        assert only["long"]["entry"] == {"left": "fund", "op": "<", "right": 10}
        assert only["long"]["exit"] == parent["long"]["exit"] and only["exits"] == parent["exits"]
        moved = copy.deepcopy(defn)
        moved["exits"]["stop"] = {"level": {"sub": ["close", {"mul": [2.0, "atr"]}]}}
        assert P.filter_of(parent, moved) is None  # a changed stop isn't a filter
    assert P.filter_of(QR_B1, QR_B1) is None
    assert P.filter_of(QR_B1, _calm(QR_B1))["long"]["entry"]["left"] == "high"


# ---- (1) calibration against the full engine -------------------------------------------------------------------------

@pytest.mark.parametrize("parent", [QR_B1, QR_B2], ids=["QR-B1", "QR-B2"])
def test_the_fast_path_matches_the_full_engine_on_the_parent_the_variant_and_three_draws(parent):
    holds = P.fast_path(parent, BARS, 240, START, FEE, **FAST).holds
    every = np.flatnonzero(holds & (np.arange(len(BARS)) >= START))
    bars, hours, fast = _spiky(every[::3])  # the filter: a third of the signals (and any spiky enough already)
    r = P.placebo(parent, _calm(parent), bars, 240, START, FEE, full=_engine(bars, hours), draws=30, **fast)
    assert [c.run for c in r.calibration] == ["parent", "variant", "draw 0", "draw 1", "draw 2"]
    assert all(c.same_trips and c.full_trips > 0 for c in r.calibration)
    assert all(c.gap <= P.CALIBRATION_PP for c in r.calibration), [c.gap for c in r.calibration]
    assert r.judged and not r.why_not and r.blocked >= len(every[::3]) and r.share < 1
    assert math.isclose(r.full_return, r.calibration[1].full_return)
    sheet = "\n".join(P.section(r))
    assert "| draw 2 |" in sheet and ("Placebo: PASS" in sheet or "Placebo: FAIL" in sheet)
    assert f"blocked {r.blocked:,} of them" in sheet


def test_a_stopped_trip_fills_where_the_engine_fills_its_stop_and_nothing_trades_before_trade_from():
    """Each trip as the engine booked it: a stop at the first execution bar through it (its low, the open where it
    opened through), less the stop slippage; and a later start trades nothing before it, as the engine."""
    late = 900
    fast = P.fast_path(QR_B1, BARS, 240, late, FEE, **FAST)
    assert any(fast.stopped) and min(e for e, _, _ in fast.trips) >= late
    p = {**to_params(QR_B1), "trade_from": int(BARS.index[late - 1].value) + 1}
    r = run_backtest("rules", BARS, INSTRUMENT, params=p, bar_minutes=240, half_spread=0, starting_capital=10_000,
                     exec_prices=HOURS, exec_minutes=60, fees=FeeSchedule(maker=Decimal(str(FEE)), taker=Decimal(str(FEE))))
    rows = [x for x in trades(fills_to_rows(r.fills), r.shorts, open_trip=True) if x["closed"] is not None]
    assert fast.matches(BARS.index, [(x["opened"], x["closed"], x["side"]) for x in rows])
    for ret, row in zip(fast.returns, rows):
        equity = float(r.equity[r.equity.index <= row["opened"]].iloc[-1])
        assert ret == pytest.approx(row["pnl"] / equity, abs=2e-4), (ret, row["pnl"] / equity)


def test_a_trip_still_open_at_the_end_is_marked_at_the_last_close_as_the_engine_marks_it():
    whole = P.fast_path(QR_B1, BARS, 240, START, FEE, **FAST)
    e, x, _ = whole.trips[-2]
    cut = BARS.iloc[: (e + x) // 2 + 1]  # ends inside a trip
    hours = HOURS[HOURS.index <= cut.index[-1]]
    fast = P.fast_path(QR_B1, cut, 240, START, FEE, **{**FAST, "exec_prices": hours})
    assert fast.open_return != 0 and fast.net == pytest.approx(np.prod(1 + fast.returns) * (1 + fast.open_return) - 1)
    trips, net = _engine(cut, hours)(QR_B1, frozenset())
    assert fast.matches(cut.index, trips) and abs(fast.net - net) <= P.CALIBRATION_PP


def test_a_calibration_miss_is_not_judged_and_the_tolerance_is_never_widened():
    holds = P.fast_path(QR_B1, BARS, 240, START, FEE, **FAST).holds
    bars, _, _ = _loud(np.flatnonzero(holds & (np.arange(len(BARS)) >= START))[::2])

    def stub(off=0.0, drop=False):
        """The full engine as the fast path itself sees it, draw 1's (its fourth run's) net return moved by `off`, or
        its first trip dropped."""
        calls = []

        def full(defn, blocked):
            calls.append(1)
            run = P.fast_path(defn, bars, 240, START, FEE, blocked=blocked, **FAST)
            got = _as_fills(run, bars.index)
            fourth = len(calls) == 4
            return (got[1:] if drop and fourth else got), run.net + (off if fourth else 0.0)
        return full

    variant = _filtered(QR_B1)
    inside = P.placebo(QR_B1, variant, bars, 240, START, FEE, full=stub(0.0049), draws=10, **FAST)
    outside = P.placebo(QR_B1, variant, bars, 240, START, FEE, full=stub(0.0051), draws=10, **FAST)
    dropped = P.placebo(QR_B1, variant, bars, 240, START, FEE, full=stub(0.0, drop=True), draws=10, **FAST)
    assert inside.judged and P.CALIBRATION_PP == 0.005
    assert not outside.judged and "didn't match the full engine" in outside.why_not
    assert not dropped.judged and not dropped.calibration[3].same_trips
    assert "Placebo: NOT JUDGED" in "\n".join(P.section(outside))
    assert not P.placebo(QR_B1, variant, bars, 240, START, FEE, draws=10, **FAST).judged  # never calibrated


def test_a_variant_that_isnt_a_filter_or_a_definition_the_fast_path_doesnt_model_is_not_judged():
    widened = copy.deepcopy(QR_B1)
    widened["long"]["entry"] = {"any": [QR_B1["long"]["entry"], {"left": "volume", "op": ">", "right": 8 * QUIET}]}
    bars, _, fast = _loud([START + 7])
    r = P.placebo(QR_B1, widened, bars, 240, START, FEE, draws=10, **fast)
    assert r.why_not.startswith("the variant isn't its parent with conditions added") and not r.judged
    donchian = tomllib.loads((ROOT / "configs" / "definitions" / "qr01c_trend_donchian_long.toml").read_text())
    r = P.placebo(donchian, donchian, BARS, 1440, 60, FEE, draws=10)
    assert "breakout slippage" in r.why_not and not r.judged


def test_a_perpetuals_trips_pay_the_settled_funding_they_hold_through():
    plain = P.fast_path(QR_B1, BARS, 240, START, FEE, **FAST)
    times = pd.date_range(BARS.index[0], BARS.index[-1], freq="8h").as_unit("ns").asi8
    rates = np.full(len(times), 0.0001)
    charged = P.fast_path(QR_B1, BARS, 240, START, FEE, funding=(times, rates, 8 * 3_600 * 10**9), **FAST)
    assert charged.trips == plain.trips
    ts = BARS.index.as_unit("ns").asi8
    held = np.array([((times > ts[e]) & (times <= ts[x])).sum() for e, x, _ in plain.trips])
    assert held.sum() > 10
    assert np.allclose(plain.returns - charged.returns, SIZE * 0.0001 * held, atol=1e-12)  # a long pays a positive rate


# ---- the study-only hook the gated full runs use ---------------------------------------------------------------------

def test_the_engine_opens_nothing_on_a_blocked_candle_and_start_refuses_blocked_entries():
    full = _engine(BARS)
    trips, _ = full(QR_B1, frozenset())
    fast = P.fast_path(QR_B1, BARS, 240, START, FEE, **FAST)
    first = fast.trips[0][0]
    gated, _ = full(QR_B1, frozenset({first}))
    assert trips[0] not in gated and trips[1:] == gated[: len(trips) - 1]
    with pytest.raises(ValueError, match="blocked entries"):
        check_study_only("rules", {**to_params(QR_B1), "blocked_entries": [1]})
    check_study_only("rules", to_params(QR_B1))


# ---- (3) the V1 and V2 sheets print it ------------------------------------------------------------------------------

def _calm_prices(days=700, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01 04:00", periods=6 * days, freq="4h", tz="UTC")
    c = np.round(30_000 * np.exp(np.cumsum(rng.normal(0.00002, 0.01, len(idx)))), 1)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c,
                         "volume": 1e9}, index=idx)


def test_a_walk_forward_study_is_scored_as_before_and_its_placebo_reads_not_judged(tmp_path):
    from tests.test_continuous_window import EMA, INSTRUMENT as KRAKEN, PRICES

    kw = dict(dataset="syn", synthetic=True, holdout_days=120, train_days=365, test_days=180)
    r = run_study(RULES, PRICES, KRAKEN, ledger=IdeaLedger(tmp_path / "l.jsonl"), default_params=to_params(_calm(EMA)),
                  placebo_parent=EMA, **kw)
    plain = run_study(RULES, PRICES, KRAKEN, ledger=IdeaLedger(tmp_path / "p.jsonl"),
                      default_params=to_params(_calm(EMA)), **kw)
    assert len(r.folds) > 1 and r.score_window == "walk_forward"
    assert r.placebo.why_not == "the placebo runs on a continuous scoring window only" and not r.placebo.judged
    assert [f.test for f in r.folds] == [f.test for f in plain.folds]  # scored, not refused, and unchanged by it
    assert r.oos_returns.equals(plain.oos_returns)


def test_a_walk_forward_study_reads_not_judged_for_the_placebo_even_with_one_window(tmp_path):
    from tests.test_continuous_window import EMA, INSTRUMENT as KRAKEN

    # 700 days less a 120-day holdout leave 580: room for one 365 + 180-day window and no second
    r = run_study(RULES, _calm_prices(), KRAKEN, dataset="syn", ledger=IdeaLedger(tmp_path / "l.jsonl"),
                  synthetic=True, holdout_days=120, train_days=365, test_days=180,
                  default_params=to_params(_calm(EMA)), placebo_parent=EMA)
    assert len(r.folds) == 1
    assert r.placebo.why_not == "the placebo runs on a continuous scoring window only" and not r.placebo.judged


def test_v1_and_v2_rows_carry_their_parent_and_only_on_a_continuous_window():
    from sleeve_fund.research import rescore

    for row, parent in (("v1", "qr01b-trend-multispeed-long"), ("v2", "qr01-trend-ema-long")):
        assert rescore.request(row, 240, "bybit", "BTC/USDT", continuous=True).placebo_parent["name"] == parent
        assert rescore.request(row, 240, "bybit", "BTC/USDT").placebo_parent is None
    assert rescore.request("v3", 240, "bybit", "BTC/USDT", continuous=True).placebo_parent is None
    from sleeve_fund.research.run import StudyRequest

    with pytest.raises(ValueError, match="continuous scoring window only"):
        StudyRequest(strategy="rules", pair="BTC/USDT", placebo_parent=QR_B1).validate()


def test_a_continuous_study_of_a_filtered_variant_prints_the_placebo_and_its_calibration(tmp_path):
    from tests.test_continuous_window import EMA, INSTRUMENT as KRAKEN

    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-01 04:00", periods=6 * 700, freq="4h", tz="UTC")
    c = np.round(30_000 * np.exp(np.cumsum(rng.normal(0.00002, 0.01, len(idx)))), 1)
    o = np.r_[c[0], c[:-1]]
    prices = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c,
                           "volume": 1e9}, index=idx)
    holds = P.fast_path(EMA, prices, 240, 0, 0.0).holds
    spikes = np.flatnonzero(holds)[::2]
    prices.iloc[spikes, prices.columns.get_loc("high")] = prices["close"].iloc[spikes] * 1.03
    kw = dict(dataset="syn", synthetic=True, holdout_days=120, train_days=365, test_days=180, score_window=CONTINUOUS)
    r = run_study(RULES, prices, KRAKEN, ledger=IdeaLedger(tmp_path / "l.jsonl"), default_params=to_params(_calm(EMA)),
                  placebo_parent=EMA, **kw)
    assert r.placebo is not None and r.placebo.blocked > 0 and len(r.placebo.calibration) == 5
    assert r.placebo.judged, [(c.run, c.same_trips, c.fast_return, c.full_return) for c in r.placebo.calibration]
    assert r.placebo.draws == 1000 and 0 <= r.placebo.percentile <= 100
    sheet = render(r, IdeaLedger(tmp_path / "l.jsonl"))
    assert "## The filter against a placebo" in sheet and "| variant |" in sheet and "| draw 2 |" in sheet
    assert r.placebo.where == "BTC/USD spot" and 0.9 < r.placebo.size < 1  # the variant run's own invested share
    assert f"each trip holds {r.placebo.size:.1%} of equity" in sheet and "on BTC/USD spot" in sheet
    assert "at that bar's low (the engine's pessimistic fill inside the bar)" in sheet
    assert "blocked " in sheet and "on average (" in sheet  # the draws' blocked share, mean and range
    assert "marked at its last close, as the engine marks it, for the variant and every draw alike" in sheet
    plain = run_study(RULES, prices, KRAKEN, ledger=IdeaLedger(tmp_path / "p.jsonl"), default_params=to_params(EMA), **kw)
    assert plain.placebo is None and "against a placebo" not in render(plain, IdeaLedger(tmp_path / "p.jsonl"))
