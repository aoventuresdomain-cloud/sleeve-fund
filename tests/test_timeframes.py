"""v2 P1-4: a strategy reads slower candles of its instrument (e.g. 4-hour ones under 15-minute decisions), built
from its own decision candles in backtest and paper alike, each visible only once closed."""

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model import BarType

from sleeve_fund.strategies.indicators import Sma
from sleeve_fund.strategies.timeframes import Candle, SlowerCandles
from test_dashboard import AUTH, SAME, client  # noqa: F401

M = 60_000_000_000
H4 = 240 * M
T0 = 1_791_201_600_000_000_000  # 12:00 UTC on 5 Oct 2026: a 4-hour boundary


def _feed(s, closes, step=15, start=T0, skip=()):
    """15-minute decision candles closing after `start`, one per close; `skip`: indexes the feed missed."""
    out = []
    for k, c in enumerate(closes):
        if k not in skip:
            out.append((start + (k + 1) * step * M, s.update(c, c + 1, c - 1, c, 1.0, start + (k + 1) * step * M)))
    return out


def test_a_slower_value_is_never_visible_before_its_candle_closes_and_changes_exactly_then():
    sma = Sma(2)
    s = SlowerCandles(240, 15, [sma])
    changed = []
    for k in range(16 * 4):  # 15-minute decision candles from 12:15 to 04:00
        ts, c = T0 + (k + 1) * 15 * M, 100.0 + k
        before = (s.last, sma.value if sma.initialized else None)
        closed = s.update(c, c + 1, c - 1, c, 1.0, ts)
        assert s.last is None or s.last.end <= ts  # never a candle whose close is still ahead
        assert (closed is not None) == (ts % H4 == 0)  # it closes on the decision candle stamped at its end
        if (s.last, sma.value if sma.initialized else None) != before:
            changed.append(ts)
    assert changed == [T0 + k * H4 for k in (1, 2, 3, 4)]
    assert s.last == Candle(148.0, 164.0, 147.0, 163.0, 16.0, T0 + 4 * H4)  # the 16 decision candles in it


def test_a_candle_whose_closing_decision_candle_is_missing_closes_on_the_next_one_stamped_at_its_end():
    s = SlowerCandles(240, 15)
    closed = dict(_feed(s, np.arange(20, dtype=float) + 100, skip={15}))  # the 16:00 decision candle never came
    assert T0 + H4 not in closed and s.count == 1
    late = closed[T0 + H4 + 15 * M]
    assert late.end == T0 + H4 and late.close == 114 and late.volume == 15  # built from the 15 it has


def test_a_part_candle_at_the_start_is_dropped_and_one_missing_candles_inside_is_kept():
    s = SlowerCandles(60, 15)
    _feed(s, [1.0, 2.0, 3.0], start=T0 + 15 * M)  # began at 12:15: a part hour, dropped
    assert s.count == 0
    _feed(s, [4.0, 5.0, 6.0, 7.0], start=T0 + 60 * M, skip={1})  # 13:00 to 14:00, missing 13:30
    assert s.count == 1 and s.last.open == 4.0 and s.last.volume == 3


@pytest.mark.parametrize("minutes, step, why", [(250, 15, "divide a day"), (240, 25, "whole multiple"),
                                                (15, 15, "whole multiple")])
def test_slower_candles_must_fit_the_day_and_the_decision_candles(minutes, step, why):
    with pytest.raises(ValueError, match=why):
        SlowerCandles(minutes, step)


def test_store_resampled_candles_match_strategy_built_ones_including_a_gap(tmp_path):
    """A warm-up seeds the slower candles from the store's own resample; they must be the ones built here."""
    from sleeve_fund.history import HistoryStore

    rng = np.random.default_rng(3)
    idx = pd.date_range(pd.Timestamp(T0, tz="UTC") + pd.Timedelta(minutes=1), periods=3 * 24 * 60, freq="1min")
    c = 60_000 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(idx))))
    df = pd.DataFrame({"open": c * (1 + rng.normal(0, 1e-4, len(c))), "high": c * 1.001, "low": c * 0.999,
                       "close": c, "volume": rng.uniform(0.5, 2, len(c))}, index=idx)
    df = df.drop(df.index[1000:1030])  # a half-hour hole inside a 4-hour candle
    store = HistoryStore(tmp_path)
    store.append("KRAKEN", "BTC/USD", df, cursor="c")
    stored = store.read("KRAKEN", "BTC/USD", 240)
    q = store.read("KRAKEN", "BTC/USD", 15)
    s = SlowerCandles(240, 15)
    built = [x for ts, row in q.iterrows() if (x := s.update(row.open, row.high, row.low, row.close, row.volume,
                                                             ts.value)) is not None]
    assert len(built) == len(stored) >= 17
    for b, (ts, row) in zip(built, stored.iterrows()):
        assert b.end == ts.value
        assert (b.open, b.high, b.low, b.close) == pytest.approx((row.open, row.high, row.low, row.close), rel=1e-12)
        assert b.volume == pytest.approx(row.volume, rel=1e-9)


def test_seeded_candles_carry_on_into_the_decision_candles_after_them():
    seeded, fresh = SlowerCandles(240, 15, [Sma(3)]), SlowerCandles(240, 15, [Sma(3)])
    closes = np.arange(16 * 5, dtype=float) + 100
    _feed(fresh, closes)
    built = [x for _, x in _feed(SlowerCandles(240, 15), closes[:48]) if x is not None]
    seeded.seed(built)  # the store's three closed candles to 00:00
    _feed(seeded, closes)  # the decision candles loaded since overlap them: only the ones after count
    assert seeded.count == fresh.count == 5 and seeded.last == fresh.last
    assert seeded.blocks[0].value == fresh.blocks[0].value


def _strategy(tmp_path, n_minutes, sma):
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.paper.node import history_loader
    from sleeve_fund.strategies.rsi_cross import RsiCross, RsiCrossConfig
    from sleeve_fund.venues import venue

    now = pd.Timestamp.now(tz="UTC").floor("4h")
    idx = pd.date_range(end=now, periods=n_minutes, freq="1min", tz="UTC")
    c = 100 + np.arange(n_minutes, dtype=float)
    store = HistoryStore(tmp_path)
    store.append("KRAKEN", "BTC/USD", pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0},
                                                   index=idx), cursor="c")
    instrument = venue("kraken").instrument("BTC", "USD")
    cfg = RsiCrossConfig(instrument_id=instrument.id, bar_type=BarType.from_str("BTC/USD.KRAKEN-15-MINUTE-LAST-INTERNAL"),
                         assumed_taker_fee=0.008, warmup_bars=40, trend_sma=sma, trend_minutes=240)
    s = RsiCross(cfg).attach_history(history_loader("KRAKEN", "BTC/USD", store))
    events = []
    s.instrument = instrument
    # backtest: no m13-E3 hold-back of the last warm-up candle (#146), which this test isn't about
    s.runtime = type("R", (), {"name": "s1", "backtest": True, "store": type("S", (), {"event": lambda self, *a, **k: events.append(a)})()})()
    return s, events


def test_a_strategy_warms_its_slower_candles_from_the_store_at_their_own_size(tmp_path):
    s, events = _strategy(tmp_path, 4 * 24 * 60, sma=5)
    s._warm_slower()
    s._warm_from_history()
    assert s._short_history is None and s.trend.initialized and s._trend_candles.count >= 5
    from sleeve_fund.history import HistoryStore

    assert s._trend_close == HistoryStore(tmp_path).read("KRAKEN", "BTC/USD", 240).close.iloc[-1]
    assert not [e for e in events if e[1] == "error"]


def test_a_strategy_whose_slower_warm_up_the_store_cant_meet_says_so_and_opens_nothing_until_it_fills(tmp_path):
    """Independent Quant Advisor (5 Oct): a short slower warm-up at start holds new entries and additions only,
    until the candles have closed live; a position held across the restart is still managed."""
    s, events = _strategy(tmp_path, 12 * 60, sma=5)  # half a day: 2 closed 4-hour candles of the 5 needed
    s._warm_slower()
    assert s._short_history is not None
    (name, level, kind, msg), = [e for e in events if e[2] == "warmup_short"]
    assert level == "error" and "4-hour candles need 5 closed ones of history and 2 loaded" in msg
    assert msg.startswith("No new entries") and "Stops, targets and exits still run" in msg
    bar = type("B", (), {"ts_event": T0})()
    s.runtime.now = lambda: None
    assert s._entry_held(bar, "long entry") and s._entry_held(bar, "addition")
    assert [e[2] for e in events].count("entry_held") == 1  # said once, not every bar
    s._trend_candles.seed(Candle(1, 1, 1, 1, 1, s._trend_candles.last.end + k * H4) for k in (1, 2, 3))
    s._lag = None
    assert not s._entry_held(bar, "long entry") and s._short_history is None  # five closed now: entries open
    assert [e[2] for e in events][-1] == "warmup_met"


def test_a_position_held_across_a_restart_with_a_short_slower_warm_up_still_exits():
    """Done when (Advisor, 5 Oct): restarted holding, with the slower warm-up short, the stop and the model's
    own exit still run; only an entry is held."""
    from test_timing import _late_strategy

    s, bar, events, sold, opened = _late_strategy(side_now=1, wants=0)
    s._short_history, s._slower = "its 4-hour candles need 5", [SlowerCandles(240, 60)]
    s._slower[0].need = 5
    s._on_bar_sided(bar)  # the model wants out: it exits
    assert len(sold) == 1 and sold[0][0] == "exit"
    s, bar, events, sold, opened = _late_strategy(side_now=1)
    s._short_history, s._slower = "its 4-hour candles need 5", [SlowerCandles(240, 60)]
    s._slower[0].need = 5
    s._entry_px, s._entry_side, s._stop_frac, s._tp_frac, s._restore = 100.0, 1, 0.02, 0.05, None
    s.runtime.backtest, s.runtime.now = False, lambda: None
    assert s._check_exits(97.5)  # past the 2% stop: it sells
    assert sold[-1][0] == "stop_loss"
    s, bar, events, sold, opened = _late_strategy(side_now=0, wants=1)
    s._short_history, s._slower = "its 4-hour candles need 5", [SlowerCandles(240, 60)]
    s._slower[0].need = 5
    s.runtime.now = lambda: None
    s._on_bar_sided(bar)  # flat and the model wants in: held
    assert opened == [] and [e[2] for e in events] == ["entry_held"]


def test_slower_candles_align_to_the_venues_daily_anchor():
    from sleeve_fund.venues import VENUES

    assert {v.daily_anchor_minutes for v in VENUES.values()} == {0}  # 00:00 UTC everywhere so far (Advisor, 5 Oct)
    s = SlowerCandles(1440, 60, anchor=8 * 60)  # a venue whose day started 08:00 UTC
    closed = [x for _, x in _feed(s, np.arange(48, dtype=float), step=60, start=T0 - 4 * 60 * M) if x is not None]
    assert [pd.Timestamp(c.end, tz="UTC").hour for c in closed] == [8, 8]  # two days, 08:00 to 08:00, not midnight


def test_a_strategy_with_a_slower_filter_trades_the_same_in_a_backtest_and_a_paper_replay(tmp_path):
    """Done when (P1-4): decisions on short candles with a filter on slower ones, here 1-minute RSI under a
    15-minute trend average to keep the recording short, make the same trades on both paths."""
    from test_tick_bar_parity import _both, _same_trades

    s = np.arange(6 * 60 * 60)  # six hours, a trade a second: a slow trend under fast swings
    prices = 60_000 * (1 + 0.03 * np.sin(s / 4000) + 0.004 * np.sin(s / 110))
    params = {"rsi_period": 5, "trend_sma": 3, "trend_minutes": 15, "time_stop_bars": 20}
    ticks, bar = _both(tmp_path, prices, "rsi_cross", params, "aggressive")
    unfiltered, _ = _both(tmp_path / "u", prices, "rsi_cross", {**params, "trend_sma": 0}, "aggressive")
    assert len(ticks) >= 6 and [f[:3] for f in ticks] != [f[:3] for f in unfiltered]  # the filter decided some
    _same_trades(ticks, bar)


def test_a_strategy_the_store_cant_warm_up_is_refused_when_created(client, tmp_path, monkeypatch):  # noqa: F811
    from urllib.parse import parse_qs, urlparse

    from sleeve_fund import history
    from sleeve_fund.dashboard.app import _venue_name

    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "history")
    c, store = client
    form = {"name": "rsi-4h", "strategy": "rsi_cross", "instrument": "BTC/USD", "bar_spec": "15-MINUTE-LAST-INTERNAL",
            "starting_balance": "5000", "risk_profile": "balanced", "warmup_bars": "0", "reason": "test",
            "p_rsi_cross__trend_sma": "50", "p_rsi_cross__trend_minutes": "240"}
    post = lambda: c.post("/sleeves/new", data=form, auth=AUTH, headers=SAME, follow_redirects=False)  # noqa: E731
    r = post()
    error = parse_qs(urlparse(r.headers["location"]).query)["error"][0]
    assert error == ("history: the model's 4-hour candles need 50 closed ones of stored history and the store holds "
                     "0 for BTC/USD; load more history first") and store.sleeves() == []
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"), periods=51 * 240, freq="1min")
    c1 = np.full(len(idx), 100.0)
    history.HistoryStore().append(_venue_name(None), "BTC/USD", pd.DataFrame(
        {"open": c1, "high": c1, "low": c1, "close": c1, "volume": 1.0}, index=idx), cursor="c")
    assert post().headers["location"] == "/sleeves/rsi-4h"


def test_a_hole_inside_a_slower_candle_gives_the_same_candle_and_decisions_in_a_backtest_and_hub_paper(monkeypatch):
    """Board rule 5a's parity check: minutes neither the hub nor the store has, inside a 15-minute trend candle
    under 1-minute decisions. Hub-fed paper (the hub client's bars, built into slower candles in the strategy)
    and the backtest on the store's minutes build the same slower candles from the minutes present and make the
    same trades."""
    from test_hub_path_parity import START, WAVE, _assert_same, backtest, hub_paper, per_order

    secs = np.arange(4 * 60 * 60)
    prices = WAVE(secs) * (1 + 0.004 * np.sin(secs / 90))
    hole = set(range(93 * 60, 99 * 60))  # 01:33 to 01:39, inside the 01:30-01:45 trend candle
    prices[sorted(hole)] = np.nan
    params = {"rsi_period": 5, "trend_sma": 3, "trend_minutes": 15, "time_stop_bars": 20}
    candles, emit = [], SlowerCandles._emit
    monkeypatch.setattr(SlowerCandles, "_emit", lambda self, c: (candles.append(c), emit(self, c))[1])

    o, f, dec, _ = hub_paper(prices, params, strategy="rsi_cross", gone=hole)
    on_hub, candles[:] = list(candles), []
    bo, bf = backtest(prices, params, strategy="rsi_cross")
    hub, bt = per_order(o, f), per_order(bo, bf)
    price = lambda cs: [(c.end, c.open, c.high, c.low, c.close) for c in cs]  # noqa: E731 - volumes are scaled apart
    assert price(on_hub) == price(candles) and len(candles) == 4 * 4  # every 15-minute candle of the four hours
    holed = next(c for c in on_hub if c.end == START + 105 * M)  # 01:30 to 01:45
    assert holed.volume == 9 * 60.0  # built from the nine minutes present, nothing made up for the six missing
    assert len(hub) >= 6 and dec.late == 0
    _assert_same(hub, bt)
