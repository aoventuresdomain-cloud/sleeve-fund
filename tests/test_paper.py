import re
from pathlib import Path

import pytest
from nautilus_trader.common import Environment
from nautilus_trader.model import Bar, BarType, InstrumentId, Price, Quantity

from sleeve_fund.paper.config import SleeveConfig, load_sleeve
from sleeve_fund.paper.node import build_node
from sleeve_fund.paper.safety import PaperSafetyError, assert_keyless
from sleeve_fund.strategies import TrendFilter, TrendFilterConfig

ROOT = Path(__file__).resolve().parent.parent
PAPER_SRC = ROOT / "sleeve_fund" / "paper"
SLEEVES = sorted((ROOT / "configs" / "sleeves").glob("*.toml"))


def test_paper_code_cannot_build_a_real_execution_client():
    # The only execution client allowed in paper is the local sandbox.
    for path in PAPER_SRC.glob("*.py"):
        src = path.read_text()
        assert "ExecutionClientFactory" not in src.replace("SandboxExecutionClientFactory", ""), path
        assert not re.search(r"\.add_exec_client\(", src), path
        assert "api_key=" not in src and "api_secret=" not in src, path


@pytest.mark.parametrize("var", ["KRAKEN_SPOT_API_KEY", "KRAKEN_SPOT_API_SECRET", "binance_api_key",
                                 "DERIBIT_TESTNET_API_KEY", "DERIBIT_TESTNET_API_SECRET"])
def test_refuses_to_start_with_credentials(var):
    with pytest.raises(PaperSafetyError):
        assert_keyless({var: "x"})
    assert_keyless({var: ""})  # empty is fine
    assert_keyless({"PATH": "/usr/bin"})


@pytest.mark.parametrize("path", SLEEVES, ids=lambda p: p.stem)
def test_shipped_sleeves_build_in_sandbox(path, monkeypatch):
    for k in list(__import__("os").environ):
        if k.upper().startswith("KRAKEN_"):
            monkeypatch.delenv(k)
    node = build_node(load_sleeve(path), log_level="ERROR", asset_fetch=dict)
    try:
        assert node.environment == Environment.SANDBOX
    finally:
        node.dispose()


def _sleeve(**over):
    base = dict(
        name="t", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
        starting_balance=1000.0,
    )
    return SleeveConfig(**{**base, **over})


@pytest.mark.parametrize(
    "over",
    [{"strategy": "nope"}, {"bar_spec": "1-TICK-LAST-INTERNAL"}, {"instrument": "BTCUSD"}, {"instrument": "SUI/"},
     {"instrument": "SUI/USD; rm"},
     {"starting_balance": 0}, {"max_notional": -1}, {"warmup_bars": -1}],
)
def test_bad_sleeve_config_rejected(over):
    with pytest.raises(ValueError):
        _sleeve(**over)


@pytest.mark.parametrize("pair,expected", [("SUI/USD", "SUI/USD"), (" xrp/gbp ", "XRP/GBP"), ("ETH/USDT", "ETH/USDT")])
def test_any_kraken_pair_accepted(pair, expected):
    cfg = _sleeve(instrument=pair)
    assert cfg.instrument == expected and cfg.instrument_id == f"{expected}.KRAKEN"


def test_typo_in_strategy_params_rejected():
    with pytest.raises(TypeError):
        TrendFilterConfig(
            instrument_id=InstrumentId.from_str("BTC/USD.KRAKEN"),
            bar_type=BarType.from_str("BTC/USD.KRAKEN-1-MINUTE-LAST-INTERNAL"),
            fsat=5,
        )


def test_warmup_and_live_bars_never_double_count():
    bt = BarType.from_str("BTC/USD.KRAKEN-1-MINUTE-LAST-INTERNAL")
    cfg = TrendFilterConfig(instrument_id=InstrumentId.from_str("BTC/USD.KRAKEN"), bar_type=bt, fast=2, slow=3,
                            assumed_taker_fee=0.008)
    s = TrendFilter(cfg)

    def bar(i, px):
        p = Price(px, 2)
        return Bar(bt, p, p, p, p, Quantity(1, 8), i * 60_000_000_000, i * 60_000_000_000)

    history = [bar(i, 100 + i) for i in range(1, 4)]
    for b in history:
        assert s._accept(b)
    # The same bar arriving again (overlap between warm-up and live) is ignored.
    assert not s._accept(history[-1])
    assert s.slow.value == pytest.approx(102.0)  # (101+102+103)/3; 102.67 if double-counted
    assert s._accept(bar(4, 110))
    assert s.slow.value == pytest.approx(105.0)


def test_kraken_asset_codes_use_the_venue_names():
    from sleeve_fund.venues import kraken_asset_codes

    pairs = {"result": {
        "XXBTZUSD": {"wsname": "XBT/USD", "base": "XXBT", "quote": "ZUSD"},
        "SUIUSD": {"wsname": "SUI/USD", "base": "SUI", "quote": "ZUSD"},
        "ETHUSDT": {"wsname": "ETH/USDT", "base": "XETH", "quote": "USDT"},
    }}
    assert kraken_asset_codes("BTC/USD", fetch=lambda: pairs) == ("XXBT", "ZUSD")
    assert kraken_asset_codes("SUI/USD", fetch=lambda: pairs) == ("SUI", "ZUSD")
    assert kraken_asset_codes("ETH/USDT", fetch=lambda: pairs) == ("XETH", "USDT")
    assert kraken_asset_codes("ABC/GBP", fetch=lambda: pairs) == ("ABC", "GBP")  # unlisted: plain codes

    def down():
        raise OSError("no network")

    assert kraken_asset_codes("SUI/USD", fetch=down) == ("SUI", "USD")


def _store_with_minutes(tmp_path, end, n):
    import numpy as np
    import pandas as pd

    from sleeve_fund.history import HistoryStore

    idx = pd.date_range(end=pd.Timestamp(end).floor("min"), periods=n, freq="1min", tz="UTC")
    c = 100 + np.arange(n, dtype=float)
    store = HistoryStore(tmp_path)
    store.append("KRAKEN", "BTC/USD", pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0},
                                                   index=idx), cursor="c")
    return store


def test_sleeve_on_trade_built_bars_warms_up_from_the_history_store(tmp_path):
    import pandas as pd

    from sleeve_fund.paper.node import history_loader
    from sleeve_fund.venues import venue

    now = pd.Timestamp.now(tz="UTC").floor("h")
    store = _store_with_minutes(tmp_path, now + pd.Timedelta(minutes=20), 10 * 60)
    instrument = venue("kraken").instrument("BTC", "USD")
    bt = BarType.from_str("BTC/USD.KRAKEN-1-HOUR-LAST-INTERNAL")
    bars = history_loader("KRAKEN", "BTC/USD", store)(instrument, bt, 3)
    # The latest three complete hours, each stamped at its close; the forming hour is left to live trades.
    assert [pd.Timestamp(b.ts_event, tz="UTC") for b in bars] == [now - pd.Timedelta(hours=h) for h in (2, 1, 0)]
    assert all(b.bar_type == bt for b in bars)

    events = []
    runtime = type("R", (), {"name": "s1", "store": type("S", (), {"event": lambda self, *a: events.append(a)})()})()
    cfg = TrendFilterConfig(instrument_id=instrument.id, bar_type=bt, fast=2, slow=3, assumed_taker_fee=0.008,
                            warmup_bars=3)
    s = TrendFilter(cfg).attach_history(history_loader("KRAKEN", "BTC/USD", store))
    s.instrument, s.runtime = instrument, runtime
    s._warm_from_history()
    assert s.slow.initialized and s.slow.value == pytest.approx(sum(b.close.as_double() for b in bars) / 3)
    assert events == [("s1", "info", "warmup", "Loaded 3 of 3 warm-up bars from the history store")]


def test_warm_up_runs_up_to_now_and_says_any_hole_left(tmp_path):
    """R3-M7: the store can be up to 6 h behind. The venue's own recent candles fill the gap up to the
    first live bar; without them the warm-up event says how many bars are missing."""
    import numpy as np
    import pandas as pd

    from sleeve_fund.paper.node import history_loader
    from sleeve_fund.venues import venue

    now = pd.Timestamp.now(tz="UTC").floor("h")
    store = _store_with_minutes(tmp_path, now - pd.Timedelta(hours=3) + pd.Timedelta(minutes=1), 10 * 60)
    instrument = venue("kraken").instrument("BTC", "USD")
    bt = BarType.from_str("BTC/USD.KRAKEN-1-HOUR-LAST-INTERNAL")

    def recent(pair, minutes):  # the venue's last 12 hourly candles by open time, the newest forming
        assert (pair, minutes) == ("BTC/USD", 60)
        idx = pd.date_range(end=now, periods=12, freq="1h", tz="UTC")
        c = np.full(12, 500.0)
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)

    bars = history_loader("KRAKEN", "BTC/USD", store, recent=recent)(instrument, bt, 6)
    closes = [pd.Timestamp(b.ts_event, tz="UTC") for b in bars]
    assert closes == [now - pd.Timedelta(hours=h) for h in range(5, -1, -1)]  # no hole, nothing forming
    assert [b.close.as_double() for b in bars][-3:] == [500.0] * 3  # the venue's candles after the store's

    events = []
    runtime = type("R", (), {"name": "s1", "store": type("S", (), {"event": lambda self, *a: events.append(a)})()})()
    cfg = TrendFilterConfig(instrument_id=instrument.id, bar_type=bt, fast=2, slow=3, assumed_taker_fee=0.008,
                            warmup_bars=6)
    s = TrendFilter(cfg).attach_history(history_loader("KRAKEN", "BTC/USD", store))  # no venue candles
    s.instrument, s.runtime = instrument, runtime
    s._warm_from_history()
    (_, level, kind, msg), = events
    assert level == "warning" and kind == "warmup" and "3 bars before the first live bar are missing" in msg


def test_a_stale_history_store_is_not_used_for_warm_up(tmp_path):
    import pandas as pd

    from sleeve_fund.paper.node import history_loader
    from sleeve_fund.venues import venue

    store = _store_with_minutes(tmp_path, pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=7), 300)
    instrument = venue("kraken").instrument("BTC", "USD")
    bt = BarType.from_str("BTC/USD.KRAKEN-1-HOUR-LAST-INTERNAL")
    with pytest.raises(LookupError, match="hours old"):
        history_loader("KRAKEN", "BTC/USD", store)(instrument, bt, 3)
    with pytest.raises(LookupError, match="no stored history"):
        history_loader("KRAKEN", "ETH/USD", store)(instrument, bt, 3)


def test_a_quiet_instrument_is_valued_from_its_quotes_until_it_trades(monkeypatch):
    # e.g. SUI at night: quotes arrive at once, the first trade can take minutes, and until then
    # the sleeve could neither mark nor reconcile.
    bt = BarType.from_str("SUI/USD.KRAKEN-1-MINUTE-LAST-INTERNAL")
    cfg = TrendFilterConfig(instrument_id=InstrumentId.from_str("SUI/USD.KRAKEN"), bar_type=bt, fast=2, slow=3,
                            assumed_taker_fee=0.008)
    last = {"px": None}
    monkeypatch.setattr(TrendFilter, "cache", property(lambda self: type("C", (), {
        "price": lambda _self, _iid, _kind: last["px"]})()))
    s = TrendFilter(cfg)
    assert s._price() == 0.0
    s._bid, s._ask = 1.17, 1.19
    assert s._price() == pytest.approx(1.18)
    last["px"] = Price(1.2, 4)
    assert s._price() == pytest.approx(1.2)  # a trade price wins once there is one


def test_a_strategy_always_warms_up_on_what_its_indicators_need():
    """PM, 5 Oct 2026: the warm-up is automatic, at least what the model's indicators need to settle (ten
    lengths for Wilder's RSI and an EMA), whatever was asked for, within what can load."""
    from sleeve_fund.paper.config import MAX_WARMUP_BARS, VENUE_WARMUP_BARS, SleeveConfig, auto_warmup

    base = dict(name="x", instrument="BTC/USD", starting_balance=1000)
    rsi = SleeveConfig(**base, strategy="rsi_bands", bar_spec="15-MINUTE-LAST-INTERNAL", warmup_bars=42)
    assert rsi.warmup_bars == 140
    more = SleeveConfig(**base, strategy="rsi_bands", bar_spec="15-MINUTE-LAST-INTERNAL", warmup_bars=500)
    assert more.warmup_bars == 500  # more can be asked for
    assert auto_warmup("trend_filter", {"slow": 200, "ema": True}, "1-MINUTE-LAST-INTERNAL") == 2000
    assert auto_warmup("trend_filter", {"slow": 200}, "1-MINUTE-LAST-INTERNAL") == 200  # a simple average
    assert auto_warmup("trend_filter", {"slow": 200, "ema": True}, "1-DAY-LAST-EXTERNAL") == VENUE_WARMUP_BARS
    assert auto_warmup("rsi_pullback", {}, "1-MINUTE-LAST-INTERNAL") == 2000  # its EMA(200)
    assert auto_warmup("ping_pong", {}, "1-MINUTE-LAST-INTERNAL") == 0
    assert auto_warmup("trend_filter", {"slow": 20_000, "ema": True}, "1-MINUTE-LAST-INTERNAL") == MAX_WARMUP_BARS


def _gap_strategy(loader, backtest=False):
    from sleeve_fund.venues import venue

    instrument = venue("kraken").instrument("BTC", "USD")
    bt = BarType.from_str("BTC/USD.KRAKEN-1-HOUR-LAST-INTERNAL")
    events = []
    store = type("S", (), {"event": lambda self, *a, **k: events.append(a)})()
    runtime = type("R", (), {"name": "s1", "store": store, "backtest": backtest, "now": lambda self: None})()
    cfg = TrendFilterConfig(instrument_id=instrument.id, bar_type=bt, fast=2, slow=3, assumed_taker_fee=0.008)
    s = TrendFilter(cfg).attach_gap_loader(loader)
    s.instrument, s.runtime, s._maybe_tick = instrument, runtime, lambda: None

    def bar(hour, close, volume):
        ts = int((1_790_000_000 // 3600 + hour) * 3600 * 1e9)
        return Bar(bt, Price(close, 1), Price(close, 1), Price(close, 1), Price(close, 1), Quantity(volume, 8), ts, ts)

    def step(b):  # what on_bar does with a decision bar before deciding
        if s._hold_gap(b):
            return None
        b = s._fill_gap(b)
        s._accept(b)
        return b

    return s, bar, step, events


def test_candles_built_while_no_trades_arrived_are_rebuilt_from_the_venue_before_deciding():
    """PM, 5 Oct 2026: after a dropped feed, paper decided on flat candles at the last price. A candle with no
    volume is now held; when trades are back, the venue's own candles replace the held ones before the model
    decides, and the bar it decides on is the venue's when the venue saw more of it."""
    asked = []
    s, bar, step, events = _gap_strategy(lambda inst, bt, since, until: (asked.append((since, until)), venue)[1])
    venue = [bar(2, 110.0, 5), bar(3, 120.0, 4), bar(4, 131.0, 9)]
    assert step(bar(1, 100.0, 1)) is not None
    assert step(bar(2, 100.0, 0)) is None and step(bar(3, 100.0, 0)) is None  # held, not decided on
    assert s.slow.value == 0 or not s.slow.initialized
    decided = step(bar(4, 130.0, 1))
    assert decided.close.as_double() == 131.0 and asked == [(bar(2, 0, 0).ts_event, bar(4, 0, 0).ts_event)]
    assert s.slow.value == pytest.approx((110 + 120 + 131) / 3)
    (_, level, kind, msg), = events
    assert (level, kind) == ("warning", "feed_gap")
    assert "No trades reached this process for 2 candles" in msg and "2 rebuilt from the venue's own candles" in msg


def test_a_quiet_market_is_used_flat_without_a_warning_and_an_unreachable_venue_says_so():
    s, bar, step, events = _gap_strategy(lambda *a: [bar(1, 100.0, 3), bar(2, 100.0, 0), bar(3, 101.0, 2)])
    for b in (bar(1, 100.0, 3), bar(2, 100.0, 0), bar(3, 101.0, 1)):
        step(b)
    assert events == [] and s.slow.value == pytest.approx((100 + 100 + 101) / 3)  # the venue had none either

    def down(*a):
        raise ConnectionError("503")

    s, bar, step, events = _gap_strategy(down)
    for b in (bar(1, 100.0, 3), bar(2, 100.0, 0), bar(3, 101.0, 1)):
        step(b)
    (_, level, kind, msg), = events
    assert kind == "feed_gap" and "1 couldn't be checked (couldn't fetch the venue's candles: 503)" in msg
    assert s.slow.value == pytest.approx((100 + 100 + 101) / 3)  # the held candle stood


def test_a_backtest_never_holds_a_candle():
    s, bar, step, _ = _gap_strategy(lambda *a: [], backtest=True)
    assert step(bar(1, 100.0, 0)) is not None


def test_the_gap_loader_returns_the_venues_closed_candles_stamped_at_their_close():
    import pandas as pd

    from sleeve_fund.paper.node import gap_loader
    from sleeve_fund.venues import venue

    instrument = venue("kraken").instrument("BTC", "USD")
    bt = BarType.from_str("BTC/USD.KRAKEN-1-HOUR-LAST-INTERNAL")
    idx = pd.date_range("2026-10-05 00:00", periods=6, freq="1h", tz="UTC")  # by open time, the last forming
    c = [10.0, 11, 12, 13, 14, 15]
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    since, until = (pd.Timestamp(t, tz="UTC").value for t in ("2026-10-05 02:00", "2026-10-05 05:00"))
    bars = gap_loader("BTC/USD", lambda pair, minutes: df)(instrument, bt, since, until)
    assert [(pd.Timestamp(b.ts_event, tz="UTC").hour, b.close.as_double()) for b in bars] == [(2, 11.0), (3, 12.0),
                                                                                             (4, 13.0), (5, 14.0)]


def test_a_restart_with_no_stored_history_warms_up_on_the_venues_candles_so_a_long_can_exit(tmp_path):
    """PM, 5 Oct 2026: rsi-bands-ls-binance held a long it should have exited. Binance's history store was empty,
    so after each deploy the model rebuilt RSI from live bars: no exit for its first 15 bars, then an unsettled
    RSI unlike the chart's. With no stored bars it now warms up on the venue's own recent candles, so RSI is
    ready and settled on the first live bar and the restored long ends when RSI has recovered."""
    import numpy as np
    import pandas as pd

    from sleeve_fund.history import HistoryStore
    from sleeve_fund.paper.node import history_loader
    from sleeve_fund.strategies.indicators import Rsi
    from sleeve_fund.strategies.rsi_bands import RsiBands, RsiBandsConfig
    from sleeve_fund.venues import venue

    now = pd.Timestamp.now(tz="UTC").floor("min")
    instrument = venue("kraken").instrument("BTC", "USD")
    bt = BarType.from_str("BTC/USD.KRAKEN-1-MINUTE-LAST-INTERNAL")
    # The venue's last 300 one-minute candles by open time, the newest still forming: a dip, then a recovery.
    closes = np.concatenate([np.linspace(86_000, 85_800, 150), np.linspace(85_800, 86_100, 150)])

    def recent(pair, minutes):
        assert minutes == 1
        idx = pd.date_range(end=now, periods=len(closes), freq="1min", tz="UTC")
        return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0}, index=idx)

    empty = HistoryStore(tmp_path)
    with pytest.raises(LookupError, match="no stored history"):  # without venue candles: nothing, as before
        history_loader("BINANCE", "BTC/USD", empty)(instrument, bt, 140)
    load = history_loader("BINANCE", "BTC/USD", empty, recent=recent)
    bars = load(instrument, bt, 140)
    assert len(bars) == 140 and pd.Timestamp(bars[-1].ts_event, tz="UTC") == now  # the forming candle left out

    events = []
    runtime = type("R", (), {"name": "rb", "book": {"qty": 0.076, "entry_px": 85_878.2},
                             "store": type("S", (), {"event": lambda self, *a: events.append(a)})()})()
    s = RsiBands(RsiBandsConfig(instrument_id=instrument.id, bar_type=bt, assumed_taker_fee=0.0005, warmup_bars=140))
    s.attach_history(load)
    s.instrument, s.runtime = instrument, runtime
    s._side = 1  # the long restored from the journal (on_start)
    s._warm_from_history()
    assert events == [("rb", "info", "warmup", "Loaded 140 of 140 warm-up bars from the venue's recent candles")]
    expected = Rsi(14)
    for b in bars:
        expected.update_raw(b.close.as_double())
    assert s.rsi.initialized and s.rsi.value == pytest.approx(expected.value) and s.rsi.value >= 55
    assert s.target_side(s.rsi.value) != 1  # the long ends on the first live bar, not 15 bars later


def _restarted(strategy, closes, orders, book_qty=0.0, **params):
    """A model after a deploy restart: the journal holds `orders` (oldest first) and the warm-up is `closes`, one
    1-minute bar each; what on_start does with them before the first live bar."""
    from datetime import datetime, timezone

    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.venues import venue

    instrument = venue("kraken").instrument("BTC", "USD")
    bt = BarType.from_str("BTC/USD.KRAKEN-1-MINUTE-LAST-INTERNAL")
    t0 = 1_790_000_000 // 60 * 60

    def at(i, seconds=0):  # bar i closes at t0 + i minutes; an order goes out a few seconds after its bar
        return datetime.fromtimestamp(t0 + 60 * i + seconds, tz=timezone.utc)

    rows = [{"intent": intent, "side": side, "ts": at(i, sec)} for intent, side, i, sec in orders][::-1]
    store = type("S", (), {"orders": lambda self, name, limit=500: rows, "event": lambda self, *a, **k: None})()
    runtime = type("R", (), {"name": "s1", "backtest": False, "store": store,
                             "book": {"qty": book_qty, "entry_px": 100.0 if book_qty else None}})()
    cls, cfg = REGISTRY[strategy]
    s = cls(cfg(instrument_id=instrument.id, bar_type=bt, assumed_taker_fee=0.0005, market="perp", allow_short=True,
                **params))
    s.instrument, s.runtime = instrument, runtime
    bars = [Bar(bt, Price(c, 1), Price(c, 1), Price(c, 1), Price(c, 1), Quantity(1, 8), int(at(i).timestamp() * 1e9),
                int(at(i).timestamp() * 1e9)) for i, c in enumerate(closes)]
    s._plan_resume()
    s.on_historical_bars(bars)
    return s


FALLING = [100_000.0 - 10 * i for i in range(200)]  # RSI near 0, every close below its recent average


@pytest.mark.parametrize("strategy", ["rsi_cross", "dip_buy"])
def test_a_restart_counts_the_time_stop_from_the_journals_entry_not_from_the_restart(strategy):
    """Review round 13, E13-3: after a deploy restart rsi_cross and dip_buy put the leg back with its time stop at 0,
    so it ran on past where the uninterrupted run exited. The leg's bars now count from the entry's own bar."""
    entry = [("entry", "BUY", 150, 2)]  # decided on bar 150, the leg's rules unmet since (no exit)
    s = _restarted(strategy, FALLING, entry, book_qty=0.1, time_stop_bars=100)
    assert (s._side, s._held) == (1, 49)  # bars 151 to 199
    s = _restarted(strategy, FALLING, entry, book_qty=0.1, time_stop_bars=40)
    assert s._side == 0  # the time stop fell on bar 190, during the warm-up: the first live bar exits


@pytest.mark.parametrize("strategy, params", [("rsi_bands", {}), ("rsi_cross", {"time_stop_bars": 0})])
def test_a_restart_after_a_stop_keeps_the_leg_and_its_exit_lock_until_the_signal_moves_off(strategy, params):
    """Review round 13, E13-4: a restart after a stop or target forgot the leg and the exit lock, and the model
    re-entered the trade the uninterrupted run declined. Both now stand until the signal moves off that side."""
    rising = [90_000.0 + 10 * i for i in range(200)]  # RSI near 100: the short's exit (RSI down to 50) never comes
    journal = [("entry", "SELL", 150, 1), ("stop_loss", "BUY", 160, 30)]
    s = _restarted(strategy, rising, journal, **params)
    assert (s._side, s._exit_lock) == (-1, -1)
    # The same, but RSI falls through the short's exit after the stop: the leg ends and the lock clears.
    s = _restarted(strategy, rising[:180] + [rising[179] - 50 * i for i in range(1, 21)], journal, **params)
    assert s._side != -1 and s._exit_lock is False


def test_a_restart_with_no_entry_in_the_journal_puts_the_held_leg_back():
    s = _restarted("rsi_bands", FALLING, [], book_qty=-0.1)
    assert s._side == -1 and s._exit_lock is False
