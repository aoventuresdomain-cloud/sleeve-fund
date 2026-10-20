"""B43-PRECISION (a) (HoE, 20 Oct): a feed whose prices carry another precision than its instrument's is refused at
load, loudly, naming the instrument and both precisions: a 2 dp feed into BTC's 0.1 (1 place) instrument never fills,
so it never runs silently. Covered here: the guard itself (data.check_feed_precision), a backtest or study start
(runner.feed_legs, which both run through), a replay start (replay.replay), and the live start check (a venue now
quoting BTC at 0.01 against the kept 0.1 FAILs, start_check.check_price_ticks). The engine's runtime backstop is QD's
(b). Expected values are written out, not read from the code."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from test_multi_asset_replay import offline  # noqa: F401 - the venue offline, for the multi-leg cell

from sleeve_fund.venues import venue

pytestmark = [pytest.mark.no_maintenance_tables]
REPLAY = sorted((Path(__file__).parent / "data" / "replay").glob("*.jsonl.gz"))[0]
CLOSES = [60_000.12, 60_010.37, 59_990.55, 60_005.01]  # 2 places, as stored history keeps them


def _btc():
    inst = venue("BINANCE").instrument("BTC", "USDT", price_precision=2)
    assert inst.price_precision == 1  # set-up: B-43's kept tick 0.1
    return inst


def _frame():
    idx = pd.date_range("2026-10-20", periods=len(CLOSES), freq="1min", tz="UTC")
    return pd.DataFrame({"open": CLOSES, "high": CLOSES, "low": CLOSES, "close": CLOSES, "volume": [1.0] * 4}, index=idx)


def _bars_at(inst, places: int):
    """The frame's bars at `places`, whatever the instrument's: a feed built on another path than to_bars."""
    from nautilus_trader.model import Bar, Price, Quantity

    from sleeve_fund.data import bar_type_for

    bt, df = bar_type_for(inst, 1), _frame()
    return [Bar(bt, Price(c, places), Price(c, places), Price(c, places), Price(c, places), Quantity(1, 8), int(t), int(t))
            for c, t in zip(df["close"], df.index.as_unit("ns").asi8)]


def _refused(exc) -> None:
    words = str(exc.value)
    assert "BTCUSDT" in words and "2 decimal places" in words and "has 1" in words, words


def test_a_2dp_feed_into_a_0_1_instrument_is_refused_naming_both_and_its_own_precision_passes():
    from sleeve_fund.data import FeedPrecisionError, check_feed_precision, to_bars

    inst = _btc()
    with pytest.raises(FeedPrecisionError) as exc:
        check_feed_precision(inst, _bars_at(inst, 2))
    _refused(exc)
    check_feed_precision(inst, _bars_at(inst, 1))
    check_feed_precision(inst, to_bars(_frame(), inst))


def test_a_backtest_or_study_start_refuses_a_2dp_feed_before_the_engine_gets_it(monkeypatch):
    from sleeve_fund.data import FeedPrecisionError
    from sleeve_fund.research import runner

    inst = _btc()
    given = []

    class Engine:
        def add_data(self, data):
            given.append(data)

        def run(self, streaming=False):
            pass

        def clear_data(self):
            pass

        def end(self):
            pass

    leg = runner.Leg(plan=None, instrument=inst, strategy=None, fee_model=None, feed=_frame(), feed_type=None,
                     coarse=False, exec_minutes=None)
    runner.feed_legs(Engine(), [leg])  # set-up: the real path loads at the instrument's own 1 place
    assert len(given) == 1
    monkeypatch.setattr(runner, "to_bars", lambda df, i, bt=None: _bars_at(i, 2))
    for legs in ([leg], [leg, leg]):  # one leg, and a portfolio's slices
        with pytest.raises(FeedPrecisionError) as exc:
            runner.feed_legs(Engine(), legs)
        _refused(exc)
    assert len(given) == 1  # nothing more reached the engine


def test_a_replay_start_refuses_a_feed_off_its_recorded_instruments_precision(monkeypatch):
    from nautilus_trader.model import Price

    from sleeve_fund.data import FeedPrecisionError
    from sleeve_fund.research import replay

    real = replay._data

    def at_2dp(instrument, rows):  # the recording's quotes at 2 places, its instrument at 1
        assert instrument.price_precision == 1  # set-up
        return [type(d)(d.instrument_id, Price(d.bid_price.as_double(), 2), Price(d.ask_price.as_double(), 2),
                        d.bid_size, d.ask_size, d.ts_event, d.ts_init) if hasattr(d, "bid_price") else d
                for d in real(instrument, rows)]

    monkeypatch.setattr(replay, "_data", at_2dp)
    with pytest.raises(FeedPrecisionError) as exc:
        replay.replay(REPLAY)
    words = str(exc.value)
    assert "BTC/USD" in words and "2 decimal places" in words and "has 1" in words, words


def test_the_live_start_check_fails_a_venue_quoting_btc_at_2dp_against_the_kept_0_1():
    from sleeve_fund import start_check
    from sleeve_fund.store import Store

    store = Store.in_memory()
    store.create_sleeve(name="p1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, venue="BINANCE",
                        params={"market": "perp", "allow_short": True, "stop_loss": 0.02})
    sec = start_check.check_price_ticks(store, live_tick=lambda v, p: Decimal("0.01"))
    assert sec.failed
    (line,) = [w for k, w in sec.lines if k == "FAIL"]
    assert "p1" in line and "0.01" in line and "0.1" in line, line


def test_a_multi_leg_replay_start_refuses_one_legs_feed_off_its_instruments_precision(tmp_path, monkeypatch, offline):
    """CR M1: replay_legs checks each leg at load too: the first leg's recorded quotes at 2 places against BTC's 1."""
    from nautilus_trader.model import Price
    from test_multi_asset_replay import ASSETS, NAME, _record, _store

    from sleeve_fund.data import FeedPrecisionError
    from sleeve_fund.research import replay

    real = replay._data

    def btc_at_2dp(instrument, rows):
        data = real(instrument, rows)
        if instrument.price_precision != 1:  # the other legs are left as recorded
            return data
        return [type(d)(d.instrument_id, Price(d.bid_price.as_double(), 2), Price(d.ask_price.as_double(), 2),
                        d.bid_size, d.ask_size, d.ts_event, d.ts_init) if hasattr(d, "bid_price") else d for d in data]

    store = _store()
    rows = [NAME, *store._asset_legs(NAME)]
    paths = []
    for row, asset in zip(rows, ASSETS):
        paths.append(tmp_path / f"{row.replace('~', '-')}.jsonl.gz")
        _record(paths[-1], row, asset, [(1, 0.0)])
    monkeypatch.setattr(replay, "_data", btc_at_2dp)
    with pytest.raises(FeedPrecisionError) as exc:
        replay.replay_legs(paths, store=store)
    _refused(exc)
