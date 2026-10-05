"""The strategy page's Signals tab: each model's rules for a side (conditions), which must agree with the
decision the model makes on the same bar, published by the paper process on the forming candle and drawn by
the dashboard. Display only (PM, 5 Oct 2026, "Entry conditions panel", option A)."""

import copy
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from nautilus_trader.model import BarType, InstrumentId

from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.store import Store, utcnow
from sleeve_fund.strategies.base import Condition
from sleeve_fund.strategies.indicators import Rsi
from sleeve_fund.strategies.ping_pong import PingPong, PingPongConfig
from sleeve_fund.strategies.rsi_bands import RsiBands, RsiBandsConfig

AUTH = ("pm", "test-pw")
IID = InstrumentId.from_str("BTC/USD.KRAKEN")
BAR = BarType.from_str("BTC/USD.KRAKEN-15-MINUTE-LAST-INTERNAL")


class _FixedRsi:
    """An RSI that reads `value` now and on the forming candle, to put the model at any reading."""

    def __init__(self, value):
        self.value, self.initialized = value, True

    def peek(self, close):
        return self.value


def _bands(**params):
    return RsiBands(RsiBandsConfig(instrument_id=IID, bar_type=BAR, assumed_taker_fee=0.0005, **params))


def _pong(**params):
    return PingPong(PingPongConfig(instrument_id=IID, bar_type=BAR, assumed_taker_fee=0.0005, **params))


RSI_READINGS = [x / 2 for x in range(0, 201)] + [29.999999, 30.000001, 54.999999, 69.999999, 50.000001]


@pytest.mark.parametrize("params", [{}, {"long_entry": 60.0, "long_exit": 80.0, "short_entry": 40.0, "short_exit": 20.0}],
                         ids=["default bands", "overlapping bands"])
@pytest.mark.parametrize("leg", [0, 1, -1])
def test_rsi_bands_conditions_agree_with_its_decision(params, leg):
    """For every reading and every leg: a side's conditions are all met exactly when the model's decision on
    that bar is that side (an entry), and a leg's exit conditions exactly when it leaves it."""
    for rsi in RSI_READINGS:
        for side in (1, -1):
            s = _bands(**params)
            s.rsi, s._side = _FixedRsi(rsi), leg
            conds = s.conditions(side, price=1.0)
            assert conds and all(isinstance(c, Condition) for c in conds)
            assert s._side == leg  # reading the conditions changes nothing the model decides with
            every = all(c.met for c in conds)
            decided = s.target_side(rsi)
            if leg == side:
                assert all(c.exit for c in conds)
                assert every == (decided != side), (rsi, leg, side)
            else:
                assert not any(c.exit for c in conds)
                assert every == (decided == side), (rsi, leg, side)


def test_rsi_bands_every_condition_met_is_an_entry():
    s = _bands()
    s.rsi = _FixedRsi(28.5)
    conds = s.conditions(1, price=1.0)
    assert [c.name for c in conds] == ["RSI(14) at or below 30"]
    assert conds[0].met and conds[0].op == "<=" and conds[0].unit == "pts" and (conds[0].gauge_min, conds[0].gauge_max) == (0, 100)
    assert s.target_side(28.5) == 1
    # On the long leg the long card lists its exit; the short side needs the long to end first.
    assert [c.name for c in s.conditions(1, price=1.0)] == ["RSI(14) at or above 55"]
    assert [c.name for c in s.conditions(-1, price=1.0)] == ["RSI(14) at or above 55", "RSI(14) at or above 70"]


def test_rsi_bands_reads_the_forming_candle_without_touching_its_rsi():
    s = _bands(rsi_period=5)
    closes = [100, 101, 99, 98, 97, 96, 95, 96, 94]
    for c in closes:
        s.rsi.update_raw(c)
    before = copy.deepcopy(s.rsi.__dict__)
    cond = s.conditions(1, price=90.0)[0]
    assert s.rsi.__dict__ == before
    twin = copy.deepcopy(s.rsi)
    twin.update_raw(90.0)
    assert cond.value == twin.value != s.rsi.value
    assert s.conditions(1)[0].value == s.rsi.value  # no price: the last closed bar


def test_rsi_peek_is_the_next_value_and_changes_nothing():
    r = Rsi(3)
    assert r.peek(10) is None
    for c in (10, 11, 12):
        r.update_raw(c)
    assert r.peek(13) is not None and not r.initialized
    r.update_raw(11)
    state = copy.deepcopy(r.__dict__)
    peeked = r.peek(9.5)
    assert r.__dict__ == state
    r.update_raw(9.5)
    assert peeked == r.value


PONG_CLOSES = [98 + i * 0.05 for i in range(81)] + [101.0, 99.5, 100.99999, 99.50001]


@pytest.mark.parametrize("leg", [None, 1, -1])
def test_ping_pong_conditions_agree_with_its_decision(leg):
    for close in PONG_CLOSES:
        for side in (1, -1):
            s = _pong()
            s._side, s._ref = leg, (None if leg is None else 100.0)
            conds = s.conditions(side, price=close)
            assert conds and (s._side, s._ref) == (leg, None if leg is None else 100.0)
            every = all(c.met for c in conds)
            decided = s.target_side(close)
            if leg == side:
                assert all(c.exit for c in conds) and every == (decided != side), (close, leg, side)
            else:
                assert every == (decided == side), (close, leg, side)


def test_ping_pong_every_condition_met_is_an_entry():
    s = _pong()
    s._side, s._ref = -1, 100.0
    conds = s.conditions(1, price=99.4)
    assert [c.name for c in conds] == ["Close 0.5% below the close it sold on"]
    assert conds[0].met and conds[0].unit == "%" and conds[0].value == pytest.approx(-0.6) and conds[0].threshold == -0.5
    assert s.target_side(99.4) == 1
    # Long now: its card lists the 1% rise as the exit, the short side the same rise as its entry.
    exit_ = s.conditions(1, price=99.4 * 1.0042)[0]
    assert exit_.exit and exit_.name == "Close 1.0% above the entry close" and not exit_.met
    assert exit_.value == pytest.approx(0.42) and exit_.threshold == 1.0 and exit_.note == "bought on the 99.4 close"


def test_models_without_conditions_list_none():
    from sleeve_fund.strategies.trend_filter import TrendFilter, TrendFilterConfig

    s = TrendFilter(TrendFilterConfig(instrument_id=IID, bar_type=BAR, assumed_taker_fee=0.0005))
    assert s.conditions(1) is None and s.signal_state(price=100.0) is None


def test_signal_state_payload_has_both_sides_and_the_stop():
    s = _pong(stop_loss=0.02)
    s._side, s._ref = 1, 100.0
    s._entry_px, s._entry_side, s._stop_frac = 100.0, 1, 0.02
    p = s.signal_state(price=99.0)
    assert p["bar_minutes"] == 15 and p["held"] == 1
    assert p["long"][0]["exit"] and p["short"][0]["name"] == "Close 1.0% above the long's entry close"
    assert p["guards"] == [{"name": "Stop-loss 2.0% below the entry", "value": -1.0, "threshold": -2.0, "op": "<=",
                            "met": False, "unit": "%", "gauge_min": -4.0, "gauge_max": 4.0, "exit": True,
                            "note": "stop at 98"}]


def test_store_keeps_one_signal_row_per_strategy(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="pp", strategy="ping_pong", instrument="BTC/USD", bar_spec="15-MINUTE-LAST-INTERNAL",
                        starting_balance=1000)
    assert store.signal_state("pp") is None
    store.set_signal_state("pp", {"long": [1]})
    store.set_signal_state("pp", {"long": [2]})
    row = store.signal_state("pp")
    assert row["payload"] == {"long": [2]} and row["ts"].tzinfo is not None


def test_a_backtest_never_publishes_signals():
    rt = SleeveRuntime.for_backtest(strategy="ping_pong", instrument="BTC/USD", bar_spec="15-MINUTE-LAST-EXTERNAL",
                                    starting_balance=1000, risk_profile="balanced")
    rt.publish_signals({"long": []})  # the in-memory journal has no table for them; nothing is written
    s = _pong().attach_runtime(rt)
    s._publish_signals()  # returns before reading anything: a backtest's runtime
    assert s._signals_ns == 0


def test_paper_publishes_the_forming_candles_conditions(tmp_path):
    """The paper path (recorded ticks replayed through the paper runtime) writes the Signals row, and the
    model trades the same as without it."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    from sleeve_fund.paper.recorder import Recorder
    from sleeve_fund.research.replay import replay
    from sleeve_fund.venues import venue
    from test_replay import START

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    path = tmp_path / "pp.jsonl.gz"
    rec = Recorder(path)
    rec.meta = {"balances": ["10000.00 USD"],
                "sleeve": {"name": "ping-pong-test", "strategy": "ping_pong", "instrument": "BTC/USD",
                           "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000,
                           "risk_profile": "balanced", "params": {"rise": 0.01, "dip": 0.005},
                           "maker_fee": "0.004", "taker_fee": "0.008", "tick_seconds": 30}}
    rec.start(inst)
    px = 60_000.0
    for s in range(4 * 60):
        px *= 1.00002
        t = START + s * 1_000_000_000
        rec.quote(QuoteTick(inst.id, Price(px - 0.5, 1), Price(px + 0.5, 1), Quantity(1, 8), Quantity(1, 8), t, t + 1000))
        rec.trade(TradeTick(inst.id, Price(px, 1), Quantity(0.05, 8), AggressorSide.BUY, TradeId(str(s)), t + 2000,
                            t + 3000))
    rec.close()
    store = Store.in_memory()
    orders = replay(path, store=store)
    assert [o["side"] for o in orders] == ["BUY"]
    row = store.signal_state("ping-pong-test")
    assert row is not None
    p = row["payload"]
    assert p["held"] == 1 and p["bar_minutes"] == 1
    assert p["long"][0]["name"] == "Close 1.0% above the entry close" and p["long"][0]["exit"]
    assert 0 < p["long"][0]["value"] < 1.0  # the price has risen a little from the close it bought on


# --- the dashboard ---------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    from sleeve_fund.dashboard import app as app_mod

    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store


def _sleeve(store, name="rsi-15m", strategy="rsi_bands", params=None):
    store.create_sleeve(name=name, strategy=strategy, instrument="BTC/USD", bar_spec="15-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=params or {})
    store.record_equity(name, equity=10_000, cash=10_000, qty=0, price=60_000, benchmark=10_000)


def _rsi_payload(rsi, held=0, bar_ts=None):
    s = _bands()
    s.rsi = _FixedRsi(rsi)
    s._side = held
    p = s.signal_state(price=60_000.0)
    p["held"] = held
    p["bar_ts"] = bar_ts
    return p


def test_signals_tab_sits_after_overview_with_the_long_lights(client):
    c, store = client
    _sleeve(store)
    store.set_signal_state("rsi-15m", _rsi_payload(30.6))
    html = c.get("/sleeves/rsi-15m", auth=AUTH).text
    tabs = html[html.index('data-tabs'):html.index('</nav>', html.index('data-tabs'))]
    assert tabs.index('data-tab="overview"') < tabs.index('data-tab="signals"') < tabs.index('data-tab="positions"')
    assert 'aria-label="Long side: 0 of 1 met"' in tabs and tabs.count("sig-lamp off") == 1
    panel = html[html.index('id="tab-signals"'):html.index('id="tab-positions"')]
    assert 'data-live="signals"' in panel and "<form" not in panel
    assert "RSI(14) at or below 30" in panel and "30.6" in panel and "≤ 30.0" in panel and "0.6 pts to go" in panel
    assert "0 of 1 met" in panel and "held flat on spot" in panel
    assert "Lights are live · model acts at the 15-minute close in" in panel


def test_signals_all_met_says_it_acts_at_close(client):
    c, store = client
    _sleeve(store, params={"market": "perp", "allow_short": True})
    store.set_signal_state("rsi-15m", _rsi_payload(29.14))
    html = c.get("/sleeves/rsi-15m", auth=AUTH).text
    panel = html[html.index('id="tab-signals"'):html.index('id="tab-positions"')]
    assert "All met · acts at close" in panel and "met · 0.9 pts inside" in panel
    assert "held flat on spot" not in panel and "RSI(14) at or above 70" in panel


def test_signals_show_exit_conditions_while_holding(client):
    c, store = client
    _sleeve(store)
    store.set_signal_state("rsi-15m", _rsi_payload(41.0, held=1))
    html = c.get("/sleeves/rsi-15m", auth=AUTH).text
    panel = html[html.index('id="tab-signals"'):html.index('id="tab-positions"')]
    assert "Exit conditions" in panel and "RSI(14) at or above 55" in panel and "14.0 pts to go" in panel


@pytest.mark.parametrize("age, words", [(None, "hasn't sent its conditions yet"), (90, "1 min ago")])
def test_signals_wait_for_the_model_without_a_fresh_reading(client, age, words):
    c, store = client
    _sleeve(store)
    if age is not None:
        store.set_signal_state("rsi-15m", _rsi_payload(30.6), ts=utcnow() - timedelta(seconds=age))
    html = c.get("/sleeves/rsi-15m", auth=AUTH).text
    panel = html[html.index('id="tab-signals"'):html.index('id="tab-positions"')]
    assert "Waiting for the model." in panel and words in panel and "RSI(14)" not in panel


def test_signals_for_a_model_without_conditions(client):
    c, store = client
    _sleeve(store, name="trend", strategy="trend_filter")
    assert "This model doesn't list its conditions yet" in c.get("/sleeves/trend", auth=AUTH).text


def test_signals_tab_left_off_a_saved_backtest(client):
    c, store = client
    _sleeve(store, name="bt:abc123")
    assert 'data-tab="signals"' not in c.get("/sleeves/bt:abc123", auth=AUTH).text
