"""A10 (Independent Quant Advisor ruling, 15 Oct 07:02 UK; Done when in hoe/bgl-plan.md): at every warm-up each
recursive indicator is seeded N bars back (its settle) and 2N bars back and the two compared at the last loaded bar:
RSI within 0.1 points, an EMA within 1 bp of its value, Wilder's ATR within 0.5%. Apart, both move back (2N and the
most the store holds, up to 1,000 bars); still apart, or the store too short, the Strategy opens nothing new, says so
once, and runs its exits, until the two agree on a live bar."""

import math

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model import BarType

from sleeve_fund.strategies.indicators import BLOCKS, Atr, Ema, Keltner, Rsi, RsiDivergence, Sma, make_block
from sleeve_fund.strategies.indicators import seeding

MIN = 60_000_000_000
T0 = 1_760_000_000 * 1_000_000_000  # Oct 2025: no candle at the epoch, which a strategy takes as already seen


def _walk(n, seed=1, vol=0.001, start=60_000.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return [(c, c * (1 + vol), c * (1 - vol), c, 1.0, T0 + (k + 1) * MIN) for k, c in enumerate(close)]


def _crash_then_creep():
    """720 bars wobbling ±1, 140 falling 1,000 each, 140 rising 0.01 each. RSI(14) seeded 140 bars back sees only
    rises (100); seeded 280 back it still carries the fall (about 24); seeded 1,000 back it reads what 280 does."""
    closes, x = [], 200_000.0
    for k in range(720):
        x += 1 if k % 2 else -1
        closes.append(x)
    for _ in range(140):
        x -= 1_000
        closes.append(x)
    for _ in range(140):
        x += 0.01
        closes.append(x)
    return [(c, c, c, c, 1.0, T0 + (k + 1) * MIN) for k, c in enumerate(closes)]


def _creep(after, n, step=0.01):
    x, ts = after[3], after[5]
    return [(x + step * k, x + step * k, x + step * k, x + step * k, 1.0, ts + k * MIN) for k in range(1, n + 1)]


# --- the check itself ----------------------------------------------------------------------------------------------

def test_converging_seeds_settle_at_n_and_2n():
    r = seeding.check(Rsi(14), _walk(1_000), "rsi")
    assert r.settled and r.seeds == (140, 280) and r.gap is None and r.feed_bars == 280  # fed the farther seed (Advisor 08:49 UK)


def test_a_gap_above_tolerance_moves_both_seeds_back_and_the_extension_settles_it():
    candles = _crash_then_creep()
    near = seeding._seeded(Rsi(14), candles[-140:])
    far = seeding._seeded(Rsi(14), candles[-280:])
    assert near.value == 100.0 and far.value < 30  # set-up: N and 2N are far apart
    r = seeding.check(Rsi(14), candles, "rsi")
    assert r.settled and r.seeds == (280, 1_000) and r.feed_bars == 1_000  # fed the farther agreeing seed


def test_seeds_still_apart_at_the_cap_hold_and_name_the_gap_in_points():
    candles = _crash_then_creep()
    r = seeding.check(Rsi(14), candles, "rsi", cap=280)  # no room to move back
    assert not r.settled and r.seeds == (140, 280) and r.gap.label == "RSI(14)" and r.gap.gap > 70
    assert r.why().startswith("rsi seeded 140 and 280 bars back reads RSI(14) ") and "points apart (tolerance 0.1)" in r.why()


def test_a_store_too_short_holds_and_says_why():
    r = seeding.check(Rsi(14), _walk(100), "rsi")
    assert not r.settled and r.short and "holds 100 bars" in r.why() and "140 and 280 bars back" in r.why()


def test_a_held_pair_settles_once_the_two_agree_on_live_bars_and_not_before():
    candles = _crash_then_creep()
    r = seeding.check(Rsi(14), candles, "rsi", cap=280)
    live = _creep(candles[-1], 200)
    settled_at = next(k for k, c in enumerate(live, 1) if r.pair.update(c))
    assert 100 < settled_at < 130  # the fall's trace in the far seed takes about 108 bars to drop below 0.1 points
    assert abs(r.pair.near.value - r.pair.far.value) <= 0.1


def test_a_pair_from_a_short_store_needs_its_n_bars_before_it_can_settle():
    r = seeding.check(Rsi(14), _walk(100), "rsi")
    live = _walk(400, seed=2)[100:]
    settled_at = next(k for k, c in enumerate(live, 1) if r.pair.update(c))
    assert settled_at == 140 - 50  # the nearer seed began 50 bars back: it needs 90 more for its 140


def test_an_ema_200_seeded_at_the_1000_bar_cap_is_within_1bp_of_one_seeded_on_twenty_times_that():
    """Advisor cell: the cap is enough. An EMA(200) keeps (1 - 2/201)^1000, about 5e-5, of its start after 1,000
    bars, so even a start 100% from the value reads under 1 bp off."""
    candles = _walk(21_000, seed=5, vol=0.002)
    for end in range(20_000, 21_001, 100):
        capped = seeding._seeded(Ema(200), candles[end - 1_000:end])
        long = seeding._seeded(Ema(200), candles[:end])
        assert abs(capped._value - long._value) / long._value <= 1e-4, end
    r = seeding.check(Ema(200), candles, "ema")
    assert r.seeds == (500, 1_000)


def test_an_atr_seeded_differently_moves_the_stop_by_no_more_than_its_tolerance():
    """Advisor cell: a 3 x ATR(14) stop read from seeds 280 and 1,000 bars back is the same stop to within 0.5% of
    its distance; seeded only 14 bars back it is not, and the check says so."""
    candles = _walk(1_000, seed=7, vol=0.003)
    close = candles[-1][3]
    r = seeding.check(Atr(14), candles, "atr")
    assert r.settled and r.seeds == (280, 560)
    a, b = (seeding._seeded(Atr(14), candles[-n:])._value for n in (280, 1_000))
    assert abs((close - 3 * a) - (close - 3 * b)) <= 0.005 * 3 * b
    short = seeding._seeded(Atr(14), candles[-14:])
    assert seeding.worst(short, seeding._seeded(Atr(14), candles[-1_000:])) is not None


def test_every_block_is_classified_and_composites_are_checked_on_their_recursive_parts():
    for kind, cls in BLOCKS.items():
        block = make_block(kind)
        assert seeding.uncovered(block) == [], kind
        assert (cls in seeding.TOLERANCE) or isinstance(block, seeding.WINDOW) or seeding.leaves(block), kind
    assert [seeding.name(x) for x in seeding.leaves(Keltner(20, 10))] == ["EMA(20)", "ATR(10)"]
    assert [seeding.name(x) for x in seeding.leaves(RsiDivergence(14))] == ["RSI(14)"]
    assert seeding.leaves(Sma(50)) == [] and seeding.need(Keltner(20, 10)) == 200
    ema = ExponentialMovingAverage(20)
    assert seeding.leaves(ema) == [ema] and seeding.need(ema) == 200
    assert seeding.check(ema, _walk(1_000), "ema").settled


def test_an_unknown_recursive_average_is_named_uncovered_and_dip_buys_daily_ema_is_covered_as_an_ema():
    from sleeve_fund.strategies.dip_buy import _Ema

    class Smoothed:  # a model's own average the check doesn't know
        def __init__(self):
            self.initialized, self.value = False, 0.0

        def update_raw(self, x):
            self.value = x

    assert seeding.uncovered(Smoothed()) == ["Smoothed"]
    assert seeding.uncovered(_Ema(20)) == [] and seeding.TOLERANCE[_Ema] == (seeding.RELATIVE, 1e-4)
    assert seeding.name(_Ema(20)) == "EMA(20)" and seeding.need(_Ema(20)) == 200


# --- the Strategy ----------------------------------------------------------------------------------------------------

def _rsi_bands(candles):
    from sleeve_fund.data import to_bars
    from sleeve_fund.strategies.rsi_bands import RsiBands, RsiBandsConfig
    from sleeve_fund.venues import venue

    instrument = venue("kraken").instrument("BTC", "USD")
    bar_type = BarType.from_str("BTC/USD.KRAKEN-1-MINUTE-LAST-INTERNAL")
    df = pd.DataFrame([c[:5] for c in candles], columns=["open", "high", "low", "close", "volume"],
                      index=pd.to_datetime([c[5] - MIN for c in candles], unit="ns", utc=True))
    bars = to_bars(df, instrument, bar_type)
    asked = []

    def loader(inst, bt, limit):
        asked.append(limit)
        return bars[-limit:]

    cfg = RsiBandsConfig(instrument_id=instrument.id, bar_type=bar_type, assumed_taker_fee=0.001, warmup_bars=140)
    s = RsiBands(cfg).attach_history(loader)
    s.instrument = instrument
    events = []
    s.runtime = type("R", (), {"name": "s1", "backtest": True, "now": lambda self: None,
                               "store": type("S", (), {"event": lambda self, *a, **k: events.append(a)})()})()
    return s, events, bars, asked


def _bar_at(s, c):
    from sleeve_fund.data import to_bars

    df = pd.DataFrame([c[:5]], columns=["open", "high", "low", "close", "volume"],
                      index=pd.to_datetime([c[5] - MIN], unit="ns", utc=True))
    return to_bars(df, s.instrument, s._cfg.bar_type)[0]


def test_a_strategy_whose_seeds_agree_warms_up_as_before_and_opens_entries():
    s, events, bars, asked = _rsi_bands(_walk(1_000))
    s._warm_from_history()
    assert s._decision_bars == 280 and s._seed_hold is None  # fed the farther agreeing seed (Advisor 08:49 UK)
    assert not [e for e in events if e[2] == "seed_unsettled"]
    assert not s._entry_held(bars[-1], "long entry")
    assert max(asked) == seeding.CAP


def test_a_strategy_the_extension_settles_is_fed_the_farther_agreeing_seed():
    candles = _crash_then_creep()
    s, events, bars, _ = _rsi_bands(candles)
    s._warm_from_history()
    assert s._seed_hold is None and s._decision_bars == 1_000
    assert s.rsi.value == pytest.approx(seeding._seeded(Rsi(14), candles[-1_000:]).value, abs=1e-6)  # bars at the pair's price precision
    assert not [e for e in events if e[2] == "seed_unsettled"]


def test_a_strategy_still_apart_holds_entries_says_so_once_and_opens_once_the_seeds_agree_live():
    candles = _crash_then_creep()[-280:]  # the store holds 2N: no room to move back
    s, events, bars, _ = _rsi_bands(candles)
    s._warm_from_history()
    (name, level, kind, msg), = [e for e in events if e[2] == "seed_unsettled"]
    assert level == "error" and msg.startswith("No new entries: RSI(14) seeded 140 and 280 bars back reads ")
    assert "points apart (tolerance 0.1)" in msg and "Stops, targets and exits still run" in msg
    assert s._entry_held(bars[-1], "long entry") and s._entry_held(bars[-1], "addition")
    assert [e[2] for e in events].count("entry_held") == 1  # said once, not every bar
    live = _creep(candles[-1], 130)
    for c in live[:50]:
        s._accept(_bar_at(s, c))
    assert s._entry_held(bars[-1], "long entry")
    for c in live[50:]:
        s._accept(_bar_at(s, c))
    assert s._seed_hold is None and "entry_held" not in s._noted  # a later hold is said again
    assert not s._entry_held(bars[-1], "long entry")
    assert [e[2] for e in events][-1] == "seed_settled"


def test_a_strategy_on_a_store_too_short_holds_and_names_why():
    s, events, bars, _ = _rsi_bands(_walk(100))
    s._warm_from_history()
    (_, level, _, msg), = [e for e in events if e[2] == "seed_unsettled"]
    assert "the history store holds 100 bars" in msg and s._entry_held(bars[-1], "long entry")


def test_a_strategy_held_on_its_seeds_still_exits():
    """Only an entry is held: the stop and the model's own exit still run (as for a short slower warm-up)."""
    from test_timing import _late_strategy

    s, bar, events, sold, opened = _late_strategy(side_now=1, wants=0)
    s._seed_hold = "RSI(14) seeded 140 and 280 bars back reads RSI(14) 0.40 points apart (tolerance 0.1)"
    s._on_bar_sided(bar)  # the model wants out: it exits
    assert len(sold) == 1 and sold[0][0] == "exit"
    s, bar, events, sold, opened = _late_strategy(side_now=1)
    s._seed_hold = "RSI(14) apart"
    s._entry_px, s._entry_side, s._stop_frac, s._tp_frac, s._restore = 100.0, 1, 0.02, 0.05, None
    s.runtime.backtest, s.runtime.now = False, lambda: None
    s._mark = lambda: (0.0, 0.0, 0.0, 97.5)
    assert s._check_exits(97.5)
    assert sold[-1][0] == "stop_loss"
    s, bar, events, sold, opened = _late_strategy(side_now=0, wants=1)
    s._seed_hold = "RSI(14) apart"
    s.runtime.now = lambda: None
    s._on_bar_sided(bar)  # flat and the model wants in: held
    assert opened == [] and [e[2] for e in events] == ["entry_held"]


def test_every_hand_coded_model_names_its_recursive_indicators_and_none_is_uncovered():
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.strategies.series import build
    from sleeve_fund.venues import venue

    instrument = venue("kraken").instrument("BTC", "USD")
    found = {}
    for name in REGISTRY:
        if name == "rules":
            continue
        try:
            s = build(name, instrument, {}, 1)
        except ValueError:  # a model on daily candles only
            s = build(name, instrument, {}, 1440)
        units = s.seeding_units() + [u for us in s.seeding_timeframes().values() for u in us]
        found[name] = (sorted(seeding.name(x) for _, u in units for x in seeding.leaves(u)),
                       sorted(f"{k}: {u}" for k, o in units for u in seeding.uncovered(o)))
    assert found["rsi_bands"] == (["RSI(14)"], [])
    assert found["rsi_pullback"][0] == sorted(["RSI(14)", f"EMA({build('rsi_pullback', instrument, {}, 1).c.ema_period})"])
    assert found["dip_buy"][0] == ["EMA(20)", "RSI(2)"]
    assert all(not unc for _, unc in found.values()), found


def test_a_rules_strategy_checks_its_decision_blocks_chained_as_they_are_fed():
    from sleeve_fund.strategies.series import build
    from sleeve_fund.venues import venue

    definition = {"version": 1, "reason": "A10 cell.", "blocks": {"k": {"kind": "keltner"}, "r": {"kind": "rsi"}},
                  "long": {"entry": {"left": "r", "op": "<", "right": 30}}}
    s = build("rules", venue("kraken").instrument("BTC", "USD"), {"definition": definition}, 1)
    (label, unit), = s.seeding_units()
    assert sorted(seeding.name(x) for x in seeding.leaves(unit)) == ["ATR(10)", "EMA(20)", "RSI(14)"]
    assert seeding.check(unit, _walk(1_000), label).settled
    assert all(seeding.read(x) is None for x in seeding.leaves(unit))  # the strategy's own blocks are untouched


# --- slower candles ----------------------------------------------------------------------------------------------------

def _rules_on_hourly_rsi(tmp_path, hours):
    """A rule-builder Strategy deciding on 15-minute candles with an RSI(14) on 1-hour ones, its store holding
    `hours` of minutes."""
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.paper.node import history_loader
    from sleeve_fund.strategies.definitions import to_params
    from sleeve_fund.strategies.rules import Rules, RulesConfig
    from sleeve_fund.venues import venue

    n = hours * 60
    now = pd.Timestamp.now(tz="UTC").floor("4h")
    idx = pd.date_range(end=now, periods=n, freq="1min", tz="UTC")
    c = 60_000 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.0005, n)))
    store = HistoryStore(tmp_path)
    store.append("KRAKEN", "BTC/USD", pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0},
                                                   index=idx), cursor="c")
    instrument = venue("kraken").instrument("BTC", "USD")
    defn = {"version": 1, "reason": "A10 cell.", "blocks": {"r": {"kind": "rsi", "timeframe": "1h"}},
            "long": {"entry": {"left": "r", "op": "<", "right": 30}}}
    s = Rules(RulesConfig(instrument_id=instrument.id,
                          bar_type=BarType.from_str("BTC/USD.KRAKEN-15-MINUTE-LAST-INTERNAL"),
                          assumed_taker_fee=0.001, warmup_bars=40, **to_params(defn)))
    s.attach_history(history_loader("KRAKEN", "BTC/USD", store))
    s.instrument = instrument
    events = []
    s.runtime = type("R", (), {"name": "s1", "backtest": True, "now": lambda self: None,
                               "store": type("S", (), {"event": lambda self, *a, **k: events.append(a)})()})()
    return s, events


def test_a_slower_candles_rsi_is_seeded_twice_and_a_store_between_n_and_2n_holds(tmp_path):
    s, events = _rules_on_hourly_rsi(tmp_path, 200)  # 200 hourly candles: past its 140, short of 280
    s._warm_slower()
    assert s._short_history is None  # its own warm-up is met, as before
    (_, level, _, msg), = [e for e in events if e[2] == "seed_unsettled"]
    assert msg.startswith("No new entries: its 1-hour candles' indicators: the history store holds 199 bars")
    bar = type("B", (), {"ts_event": 0})()
    assert s._entry_held(bar, "long entry")


def test_a_slower_candles_rsi_the_store_covers_twice_over_settles(tmp_path):
    s, events = _rules_on_hourly_rsi(tmp_path, 400)
    s._warm_slower()
    assert s._seed_hold is None and not [e for e in events if e[2] == "seed_unsettled"]
    assert s._slow[60].count >= 140


def test_a_slower_pair_is_fed_only_on_its_own_closed_candles():
    from sleeve_fund.strategies.timeframes import SlowerCandles

    s, bar, events, sold, opened = __import__("test_timing")._late_strategy(side_now=0, wants=1)
    slow = SlowerCandles(60, 15)
    r = seeding.check(Rsi(14), _walk(100), "rsi")
    s._seed_pairs, s._seed_hold = {"rsi": (r.pair, slow, 100 * MIN)}, "rsi: short"
    fed = r.pair.fed
    s._feed_seeds(None, (1, 1, 1, 1, 1, 101 * MIN))  # a decision candle: not this pair's
    assert r.pair.fed == fed
    s._feed_seeds(slow, (1, 1, 1, 1, 1, 50 * MIN))  # one loaded already
    assert r.pair.fed == fed
    s._feed_seeds(slow, (1, 1, 1, 1, 1, 160 * MIN))
    assert r.pair.fed == fed + 1


def test_a_forced_restart_at_several_points_matches_one_continuous_run_on_every_signal():
    """Done when 4: restarted from the store at several points, the warm-up's seeds agree and the model decides
    exactly what one continuous run decides on every later candle (its leg put back, as the journal gives it)."""
    candles = _walk(4_000, seed=11, vol=0.002)
    whole, _, all_bars, _ = _rsi_bands(candles)
    signals = []
    for b in all_bars:
        whole._accept(b)
        signals.append((whole.want_side(b), round(whole.rsi.value, 6)))
    for cut in (1_200, 2_000, 2_750, 3_500):
        cont, _, bars, _ = _rsi_bands(candles)
        for b in bars[:cut]:
            cont._accept(b)
            cont.want_side(b)
        s, events, _, _ = _rsi_bands(candles[:cut])  # the store, at the restart, ends at the cut
        s._warm_from_history()
        assert s._seed_hold is None, events
        s._side = cont._side  # the journal's leg (resume_leg)
        for k in range(cut, len(all_bars)):
            assert s._accept(all_bars[k])
            assert s.want_side(all_bars[k]) == signals[k][0], (cut, k)
            assert abs(s.rsi.value - signals[k][1]) <= 0.1, (cut, k)  # within the tolerance the check held it to
    assert {sig for sig, _ in signals} >= {-1, 0, 1}  # long, short and flat all occur


# --- an indicator fed on a longer boundary: dip_buy's daily EMA ----------------------------------------------------

def _dip_buy(days_loaded, daily_candles):
    """dip_buy on 1-hour decision candles: its warm-up loads `days_loaded` days of them; the store holds
    `daily_candles` daily ones, to the same last close."""
    from sleeve_fund.data import to_bars
    from sleeve_fund.strategies.dip_buy import DipBuy, DipBuyConfig
    from sleeve_fund.venues import venue

    instrument = venue("kraken").instrument("BTC", "USD")
    hourly = BarType.from_str("BTC/USD.KRAKEN-1-HOUR-LAST-INTERNAL")
    n = max(days_loaded, daily_candles) * 24
    end = pd.Timestamp("2026-10-15", tz="UTC")
    rng = np.random.default_rng(9)
    close = 60_000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    idx = pd.date_range(end=end - pd.Timedelta(hours=1), periods=n, freq="1h", tz="UTC")  # by open time
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})

    def loader(inst, bt, limit):
        if str(bt).startswith("BTC/USD.KRAKEN-1-DAY"):
            return to_bars(daily.tail(min(limit, daily_candles)), inst, bt)
        return to_bars(df.tail(min(limit, days_loaded * 24)), inst, bt)

    cfg = DipBuyConfig(instrument_id=instrument.id, bar_type=hourly, assumed_taker_fee=0.001,
                       warmup_bars=days_loaded * 24)
    s = DipBuy(cfg).attach_history(loader)
    s.instrument = instrument
    events = []
    s.runtime = type("R", (), {"name": "s1", "backtest": True, "now": lambda self: None,
                               "store": type("S", (), {"event": lambda self, *a, **k: events.append(a)})()})()
    return s, events


def test_dip_buys_daily_ema_settles_when_the_stores_daily_candles_cover_twice_its_settle():
    s, events = _dip_buy(201, 600)
    s._warm_from_history()
    assert s._seed_hold is None and not [e for e in events if e[2] == "seed_unsettled"], events
    assert s.day_ema.count == 201


def test_dip_buys_daily_ema_holds_when_the_store_has_too_few_daily_candles_and_says_so():
    s, events = _dip_buy(201, 300)  # 300 daily candles: short of 2 x 200
    s._warm_from_history()
    (_, _, _, msg), = [e for e in events if e[2] == "seed_unsettled"]
    assert "EMA(20) on daily closes: the history store holds 300 bars" in msg
    assert s._entry_held(type("B", (), {"ts_event": 0})(), "long entry")


def test_dip_buys_daily_pair_is_fed_only_on_day_closes():
    s, events = _dip_buy(201, 300)
    s._warm_from_history()
    (pair, of, after), = s._seed_pairs.values()
    assert of == 1440
    fed = pair.fed
    s._feed_seeds(None, (1, 1, 1, 61_000.0, 1, after + 3_600_000_000_000))  # an hour after: not a day close
    assert pair.fed == fed
    s._feed_seeds(None, (1, 1, 1, 61_000.0, 1, after + 86_400_000_000_000))
    assert pair.fed == fed + 1


def test_a_boundary_indicator_whose_own_warm_up_is_shorter_than_the_agreeing_seed_holds():
    """The model can't be fed more of a boundary indicator's candles than its warm-up gives it: settled only further
    back than that, its own reading is the nearer seed, so it holds until a pair agrees live."""
    from sleeve_fund.strategies.dip_buy import _Ema

    s, events = _dip_buy(201, 600)
    candles = _walk(600, seed=4, vol=0.03)
    s._check_seeds([("EMA(20) on daily closes", _Ema(20))], candles, 0, 1440, fed=20)
    (pair, of, _), = s._seed_pairs.values()
    assert of == 1440 and pair.fed == 20 and "seeded 20 and 600 bars back" in s._seed_hold
    s._seed_pairs, s._seed_hold = {}, None
    s._check_seeds([("EMA(20) on daily closes", _Ema(20))], candles, 0, 1440, fed=400)
    assert s._seed_hold is None
