"""v2 P1-2: every paper or live order's bar close, arrival, decision, send, acceptance and fills, to the
microsecond, for close-to-fill times per strategy."""

from datetime import datetime, timezone

import pytest

from sleeve_fund.store import Store
from test_dashboard import AUTH, client  # noqa: F401
from test_sleeve_runtime import store as _store_fixture

journal = _store_fixture  # Postgres when TEST_DATABASE_URL is set

NS = 1_791_223_500_123_456_789  # 18:05:00.123456789 on 5 Oct 2026


def _store():
    db = Store.in_memory()
    db.create_sleeve(name="pp", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                     starting_balance=10_000)
    db.record_order("pp", order_id="O-1", side="BUY", qty=0.1, intent="entry", reason="test")
    return db


def test_an_orders_stamps_are_kept_to_the_microsecond_and_filled_in_as_they_come():
    db = _store()
    db.record_timing("pp", "O-1", accepted=NS)  # before the decision's row: nothing to fill in yet
    assert db.timings("pp") == []
    db.record_timing("pp", "O-1", bar_close=NS - 123_456_789, bar_recv=NS - 100_000_000, decided=NS, sent=NS + 1_000)
    db.record_timing("pp", "O-1", accepted=NS + 2_000_000)
    db.record_timing("pp", "O-1", fill=NS + 3_000_000, venue_ts=NS + 2_900_000)
    db.record_timing("pp", "O-1", fill=NS + 5_000_000, venue_ts=NS + 4_900_000)  # a second part fill
    (t,) = db.timings("pp")
    assert t["decided"] == datetime(2026, 10, 5, 18, 5, 0, 123456, tzinfo=timezone.utc)
    assert t["bar_close"] == datetime(2026, 10, 5, 18, 5, tzinfo=timezone.utc)
    assert (t["sent"] - t["decided"]).microseconds == 1 and (t["accepted"] - t["decided"]).microseconds == 2000
    assert (t["first_fill"] - t["decided"]).microseconds == 3000 and (t["last_fill"] - t["decided"]).microseconds == 5000
    assert (t["venue_ts"] - t["decided"]).microseconds == 4900


def test_an_order_nothing_decided_on_a_bar_gets_no_row():
    db = _store()
    db.record_timing("pp", "O-1", fill=NS)  # a risk stop or a restore: journaled, but not timed
    assert db.timings("pp") == []


def test_a_paper_run_times_each_decision_and_a_backtest_keeps_none(monkeypatch):
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.venues import venue

    prices = synthetic_ohlcv(days=300, seed=5, vol=0.03)
    inst = venue("kraken").instrument("ETH", "USD")
    db = Store.in_memory()
    db.create_sleeve(name="backtest", strategy="trend_filter", instrument="ETH/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                     starting_balance=10_000, risk_profile="aggressive")
    paper = SleeveRuntime(db, "backtest", tick_seconds=86_400)
    keep = SleeveRuntime.on_timing
    # run_backtest marks its runtime a backtest, which keeps no timings: record them anyway, as paper would.
    monkeypatch.setattr(SleeveRuntime, "on_timing", lambda self, oid, **st: self.store.record_timing(self.name, oid, **st))
    monkeypatch.setattr(SleeveRuntime, "on_order", lambda self, timing=None, **k: self.store.record_order(
        self.name, ts=self.now(), timing=timing, **k))
    run_backtest("trend_filter", prices, inst, params={"fast": 5, "slow": 20}, runtime=paper)
    entries = [o for o in db.orders("backtest", limit=10_000) if o["intent"] == "entry" and o["filled_qty"]]
    timed = {t["order_id"]: t for t in db.timings("backtest", limit=10_000)}
    assert entries and all(o["order_id"] in timed for o in entries)
    for o in entries:
        t = timed[o["order_id"]]
        assert t["bar_close"] is not None and t["bar_close"] <= t["decided"] <= t["sent"]
        assert t["first_fill"] is not None and t["first_fill"] <= t["last_fill"] and t["venue_ts"] is not None
    replay = SleeveRuntime.for_backtest(strategy="trend_filter", instrument="ETH/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                                        starting_balance=10_000, risk_profile="aggressive")
    replay.store = db  # a backtest's runtime on the same journal still writes no timings
    before = len(db.timings("backtest", limit=10_000))
    keep(replay, entries[0]["order_id"] + "-again", decided=NS)
    assert len(db.timings("backtest", limit=10_000)) == before


def test_the_strategy_page_shows_close_to_fill(client):  # noqa: F811
    from sleeve_fund.dashboard.trading import timing_view

    c, store = client
    store.create_sleeve(name="pp", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"rise": 0.01, "dip": 0.005})
    for k, lag_ms in enumerate([120, 80, 100, 400]):
        oid = f"O-{k}"
        store.record_order("pp", order_id=oid, side="BUY", qty=0.1, intent="entry", reason="test")
        close = NS + k * 60_000_000_000
        store.record_timing("pp", oid, bar_close=close, bar_recv=close + 5_000_000, decided=close + 6_000_000,
                            sent=close + 7_000_000)
        store.record_timing("pp", oid, fill=close + lag_ms * 1_000_000, venue_ts=close + lag_ms * 1_000_000)
    t = timing_view(store.timings("pp"))
    assert (t["n"], t["median"], t["p95"], t["ours"]) == (4, 120.0, 400.0, 7.0)
    page = c.get("/sleeves/pp", auth=AUTH).text
    assert "Candle close to fill" in page and "120 ms median, 400 ms p95" in page and "7 ms to send, 4 orders" in page
    assert timing_view([]) is None


def test_a_bar_that_closed_while_the_strategy_was_down_is_decided_once_if_still_the_latest():
    """m13-E3: a restart spanning a bar close decided nothing on that bar; the signal was lost."""
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import late_bar

    minute = 60_000_000_000
    bars = [SimpleNamespace(ts_event=NS - minute), SimpleNamespace(ts_event=NS)]
    at = lambda ns: datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)  # noqa: E731
    down = at(NS - 40_000_000_000)  # the previous process's last heartbeat, 40 s before the close
    assert late_bar(bars, NS + 20_000_000_000, minute, down, None) is bars[-1]  # down over the close: decide on it
    assert late_bar(bars, NS + 20_000_000_000, minute, down, at(NS - 30 * minute)) is bars[-1]
    assert late_bar(bars, NS + 20_000_000_000, minute, None, None) is None  # a first start: nothing was missed
    assert late_bar(bars, NS + 20_000_000_000, minute, at(NS + 1_000_000_000), None) is None  # alive at the close
    assert late_bar(bars, NS + minute, minute, down, None) is None  # a bar old: history, not a signal
    hour = 60 * minute
    hourly = [SimpleNamespace(ts_event=NS - hour), SimpleNamespace(ts_event=NS)]
    assert late_bar(hourly, NS + 90_000_000_000, hour, down, None) is hourly[-1]
    assert late_bar(hourly, NS + 50 * minute, hour, down, None) is hourly[-1]  # late: it may still exit (below)
    assert late_bar(bars, NS + 20_000_000_000, minute, down, at(NS + 1_000_000)) is None  # acted on before
    assert late_bar([], NS, minute, down, None) is None


def test_journal_writes_leave_the_decision_path_and_every_read_sees_them():
    import time

    from sleeve_fund.paper.queued import QueuedStore

    class Slow(Store):
        def record_order(self, *a, **k):
            time.sleep(0.2)
            super().record_order(*a, **k)

    slow = Slow.in_memory()
    slow.create_sleeve(name="pp", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                       starting_balance=10_000)
    q = QueuedStore(slow, retry_seconds=(0.01,))
    t0 = time.monotonic()
    q.record_order("pp", order_id="O-1", side="SELL", qty=0.1, intent="exit", reason="test")
    q.record_timing("pp", "O-1", decided=NS, sent=NS)  # after its order, in order
    assert time.monotonic() - t0 < 0.05  # an exit didn't wait for the database
    assert [o["order_id"] for o in q.orders("pp")] == ["O-1"] and len(q.timings("pp")) == 1  # reads wait
    q.record_order("pp", order_id="O-2", side="BUY", qty=0.1, intent="nonsense", reason="test")
    with pytest.raises(RuntimeError, match="journal write failed"):
        q.orders("pp")  # a failed write surfaces on the next read, after its retry
    assert [o["order_id"] for o in q.orders("pp")] == ["O-1"]
    (incident,) = [e for e in q.events("pp") if e["kind"] == "incident"]  # and was an alert at once
    assert incident["level"] == "error" and "O-2" in incident["message"]
    t0 = time.monotonic()
    q.record_order("pp", order_id="O-3", side="BUY", qty=0.1, intent="entry", reason="test")
    assert time.monotonic() - t0 >= 0.2  # an opening order is on record before it goes (QA P1-L5)


def test_an_opening_order_whose_journal_row_fails_is_never_sent_an_exit_is_and_its_row_is_retried():
    """QA P1-L5, HoE's ruling: nothing opens without its row; an exit goes whatever happens to its row, which is
    an incident and is retried."""
    from sleeve_fund.paper.queued import QueuedStore

    db = Store.in_memory()
    db.create_sleeve(name="q", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                     starting_balance=1_000)
    real, tries = db.record_order, []

    def flaky(*a, **k):
        tries.append(k["order_id"])
        if len(tries) < 3:
            raise RuntimeError("database gone")
        real(*a, **k)

    db.record_order = flaky
    q = QueuedStore(db, retry_seconds=(0.01, 0.01))
    with pytest.raises(RuntimeError, match="database gone"):
        q.record_order("q", order_id="O-1", side="BUY", qty=0.1, intent="entry", reason="t")
    q.record_order("q", order_id="O-2", side="SELL", qty=0.1, intent="stop_loss", reason="t")  # queued: no raise
    q.flush()
    assert tries == ["O-1", "O-2", "O-2"] and [o["order_id"] for o in q.orders("q")] == ["O-2"]
    assert [e["kind"] for e in q.events("q")] == ["incident"]


def test_a_run_journals_the_same_through_the_queue():
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.paper.queued import QueuedStore
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    prices = synthetic_ohlcv(days=300, seed=5, vol=0.03)
    inst = venue("kraken").instrument("ETH", "USD")

    def journal(wrap):
        db = Store.in_memory()
        db.create_sleeve(name="backtest", strategy="trend_filter", instrument="ETH/USD",
                         bar_spec="1-DAY-LAST-EXTERNAL", starting_balance=10_000, risk_profile="aggressive")
        rt = SleeveRuntime(wrap(db), "backtest", tick_seconds=86_400)
        run_backtest("trend_filter", prices, inst, params={"fast": 5, "slow": 20}, runtime=rt)
        if wrap is QueuedStore:
            rt.store.flush()
        return ([(o["side"], o["intent"], round(o["qty"], 8), o["status"], o["ts"]) for o in db.orders("backtest", limit=10_000)],
                [(f["side"], round(f["qty"], 8), f["price"]) for f in db.fills("backtest", limit=10_000)])

    direct = journal(lambda db: db)
    assert direct[0] and journal(QueuedStore) == direct


def test_an_orders_timing_is_journaled_with_the_order_itself_on_the_production_engine(journal):
    """DA-8: the order_timings row is written in the order's own transaction, so its foreign key never
    depends on the writer thread's ordering. Runs on Postgres (foreign keys enforced) when TEST_DATABASE_URL
    is set."""
    from sqlalchemy.exc import IntegrityError

    from sleeve_fund.paper.queued import QueuedStore
    from sleeve_fund.paper.runtime import SleeveRuntime

    store = journal
    store.create_sleeve(name="pp", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000)
    rt = SleeveRuntime(QueuedStore(store), "pp")
    rt.on_order(order_id="O-1", side="BUY", qty=0.1, intent="entry", reason="test", signal={},
                timing={"bar_close": NS - 2_000_000, "bar_recv": NS - 1_000_000, "decided": NS})
    rt.on_timing("O-1", sent=NS + 1_000)
    rt.store.flush()
    (t,) = store.timings("pp")
    assert t["order_id"] == "O-1" and (t["sent"] - t["decided"]).microseconds == 1
    if store.engine.dialect.name == "postgresql":  # a timing row without its order is refused
        with pytest.raises(IntegrityError):
            store.record_timing("pp", "O-none", decided=NS)


# --- m13-E3 as the Independent Quant Advisor ruled (5 Oct 2026): past 90 s after its bar's close, a decision
# opens and adds nothing, but exits and reductions always run; both are journaled with the lag.

def _late_strategy(lag_s=None, side_now=0, wants=1):
    from types import SimpleNamespace

    from nautilus_trader.model import BarType, InstrumentId

    from sleeve_fund.strategies import TrendFilter, TrendFilterConfig

    cfg = TrendFilterConfig(instrument_id=InstrumentId.from_str("BTC/USD.KRAKEN"),
                            bar_type=BarType.from_str("BTC/USD.KRAKEN-1-HOUR-LAST-INTERNAL"), fast=2, slow=3,
                            assumed_taker_fee=0.008, stop_loss=0.02, take_profit=0.05, allow_short=True,
                            market="perp")
    s = TrendFilter(cfg)
    events, sold, opened = [], [], []
    s.runtime = SimpleNamespace(name="s1", store=SimpleNamespace(event=lambda *a, **k: events.append(a)))
    s.want_side = lambda bar: wants
    s._busy = lambda: False
    s._pos_side = lambda: side_now
    s.explain = lambda bar, side: ("why", {})
    s._sell_all = lambda *a, **k: sold.append(a)
    s._open = lambda *a: opened.append(a)
    s._price = lambda: 101.0
    s._lag = None if lag_s is None else lag_s * 1_000_000_000
    bar = SimpleNamespace(ts_event=NS, close=SimpleNamespace(as_double=lambda: 100.0))
    return s, bar, events, sold, opened


def test_an_entry_decided_more_than_90_s_after_its_close_is_skipped_and_said_with_its_lag():
    s, bar, events, sold, opened = _late_strategy(lag_s=91)
    s._on_bar_sided(bar)
    assert opened == [] and sold == []
    (_, level, kind, msg), = events
    assert (level, kind) == ("warning", "late_entry_skipped")
    assert "Skipped a long entry on the 18:05 candle: decided 91.0 s after its close" in msg and "price now 101" in msg
    s, bar, events, sold, opened = _late_strategy(lag_s=None)  # on time: it opens
    s._on_bar_sided(bar)
    assert len(opened) == 1 and events == []


def test_an_exit_missed_by_any_amount_is_executed_and_said_with_its_lag():
    s, bar, events, sold, opened = _late_strategy(lag_s=50 * 60, side_now=1, wants=0)
    s._on_bar_sided(bar)
    assert len(sold) == 1 and sold[0][0] == "exit"
    (_, level, kind, msg), = events
    assert kind == "late_exit" and "3000 s after its close (candle close 100, price now 101)" in msg


def test_a_late_reversal_only_closes():
    s, bar, events, sold, opened = _late_strategy(lag_s=120, side_now=1, wants=-1)
    s._on_bar_sided(bar)
    assert len(sold) == 1 and s._flip is None and opened == []  # the short isn't opened once the long closes
    assert [e[2] for e in events] == ["late_entry_skipped", "late_exit"]
    assert "short entry after closing the long" in events[0][3]
    s, bar, events, sold, opened = _late_strategy(lag_s=None, side_now=1, wants=-1)  # on time: close, then open
    s._on_bar_sided(bar)
    assert len(sold) == 1 and s._flip is not None and s._flip[0] == -1


def test_a_missed_bar_is_decided_however_late_until_a_newer_bar_has_closed():
    s, bar, *_ = _late_strategy()
    decided, warmed = [], []
    s.on_bar, s.on_historical_bars = decided.append, warmed.append
    s._late, s._now_ns = bar, lambda: NS + 50 * 60_000_000_000
    s._decide_late()
    assert decided == [bar] and warmed == []  # 50 minutes late on an hourly bar: decided (exits only)
    s._late, s._now_ns = bar, lambda: NS + 60 * 60_000_000_000
    s._decide_late()
    assert decided == [bar] and warmed == [[bar]]  # a newer bar has closed: it decides instead


def _minutes(*hl, start=NS):
    from types import SimpleNamespace

    def px(v):
        return SimpleNamespace(as_double=lambda: v)

    return [SimpleNamespace(ts_event=start + (k + 1) * 60_000_000_000, open=px((h + lo) / 2), high=px(h), low=px(lo))
            for k, (h, lo) in enumerate(hl)]


def _booked(s, sold):
    """What the strategy sends, with the booking the replay hands its order (_outage_book, taken by _submit)."""
    s._sell_all = lambda intent, reason, values=None: sold.append((intent, reason, dict(s._outage_book or {})))


@pytest.mark.parametrize("minutes, hit, booked", [
    ([(100.5, 99.5), (100.2, 97.9), (101.0, 99.0)], "stop_loss", 98.0),  # 2% stop at 98 crossed in minute 2
    ([(103.0, 99.5), (105.1, 101.0)], "take_profit", 105.0),
    ([(105.5, 97.5)], "stop_loss", 98.0),  # both in one minute: the stop, adverse first (Advisor NA-2)
    ([(100.5, 99.5), (97.0, 96.0)], "stop_loss", 96.5),  # the second minute opens past the stop: at its open
    ([(101.0, 99.0)], None, None),
])
def test_a_stop_or_target_crossed_while_the_strategy_was_down_is_booked_where_the_venue_would_have_filled_it(
        minutes, hit, booked):
    """On restart the stored minutes since the previous process last saw market data are replayed as the venue's
    resting orders would have traded them (Advisor NA-1): a stop at its level, or the open on a gap; a target at
    its level; the market's price now beside it."""
    s, bar, events, sold, opened = _late_strategy()
    _booked(s, sold)
    s._entry_px, s._entry_side, s._stop_frac, s._tp_frac = 100.0, 1, 0.02, 0.05
    s._last_alive = s._last_seen = datetime.fromtimestamp(NS / 1e9, tz=timezone.utc)  # market data seen till then
    s.runtime.store.fills = lambda name, limit: [{"ts": datetime.fromtimestamp((NS - 600e9) / 1e9, tz=timezone.utc)}]
    s._mark = lambda: (1e4, 1e4, 0.0, 101.0)  # the perp's position isn't at a venue here: no liquidation price
    before = _minutes((90.0, 90.0), start=NS - 120_000_000_000)  # crossed before the last heartbeat: seen then
    s.history_loader = lambda instrument, bar_type, n: before + _minutes(*minutes)
    s.instrument = None
    assert s._check_outage_exits() is (hit is not None)
    if hit is None:
        assert sold == [] and events == []
        return
    (intent, reason, book), = sold
    assert intent == hit and book["book_px"] == pytest.approx(booked) and book["market_on_return"] == 101.0
    assert s._entry_px is None and s._outage_book is None
    assert [e[2] for e in events] == ["outage_exit"] and "while the strategy was down" in reason


def test_the_restart_replay_starts_at_the_last_market_data_seen_not_the_last_heartbeat():
    """QA P1-L3: a process keeps its heartbeat through a hub outage, so the replay starts from the last market data
    it saw; with none on record, from the open position's last fill."""
    from sleeve_fund.strategies.base import _ns

    s, *_ = _late_strategy()
    at = lambda m: datetime.fromtimestamp((NS + m * 60_000_000_000) / 1e9, tz=timezone.utc)  # noqa: E731
    s.runtime.store.fills = lambda name, limit: [{"ts": at(-30)}]
    s._last_alive, s._last_seen = at(0), at(-20)
    assert s._outage_since() == _ns(at(-20))
    s._last_seen = None
    assert s._outage_since() == _ns(at(-30))
    s._last_seen = at(-40)  # older than the fill: nothing before the fill is this position's
    assert s._outage_since() == _ns(at(-30))


@pytest.mark.parametrize("unseen, high, low, hit", [
    (True, 100.5, 97.9, "stop_loss"),  # no trade reached the strategy while the bar formed: the stop crossed in it
    (True, 105.2, 99.0, "take_profit"),
    (True, 101.0, 99.0, None),  # nothing crossed
    (False, 100.5, 97.9, None),  # trades all through it: the live checks saw every price, nothing to do here
])
def test_a_stop_or_target_inside_a_bar_no_trade_reached_the_strategy_for_is_replayed(unseen, high, low, hit):
    """QA P1-L1: a bar holding venue time with no trade (the hub or its venue away), late or on time: its high and
    low are replayed, not only its close."""
    from types import SimpleNamespace

    s, bar, events, sold, opened = _late_strategy(side_now=1)
    _booked(s, sold)
    s.runtime.backtest = False
    s._mark = lambda: (1e4, 1e4, 0.0, 101.0)
    s._entry_px, s._entry_side, s._stop_frac, s._tp_frac = 100.0, 1, 0.02, 0.05
    s._trade_ns = NS - (40 if unseen else 2) * 60_000_000_000 // 60
    late = SimpleNamespace(ts_event=NS, open=SimpleNamespace(as_double=lambda: 100.0),
                           high=SimpleNamespace(as_double=lambda: high), low=SimpleNamespace(as_double=lambda: low),
                           close=SimpleNamespace(as_double=lambda: 100.2))
    assert s._check_unseen_exits(late) is (hit is not None)
    if hit is None:
        assert sold == [] and events == []
        return
    (intent, reason, book), = sold
    assert intent == hit and book["outage_while"] == "while the market data feed was away"
    assert "reached while the market data feed was away: the price passed the" in reason and "in the minute to" in reason
    from sleeve_fund.strategies.base import outage_fill_note

    note = outage_fill_note({"intent": intent, "signal": book}, book["book_px"])
    assert "while the market data feed was away" in note and "the market was 101 on return" in note


def test_an_outage_exits_fill_is_said_against_its_level_once():
    from sleeve_fund.strategies.base import outage_fill_note

    decision = {"intent": "stop_loss", "signal": {"outage_level": 98.0, "breached_at": "18:07",
                                                  "market_on_return": 99.5}}
    assert outage_fill_note(decision, 97.02) == (
        "The stop-loss is booked at 97.02, -1.00% from its 98 level, as the venue would have filled it in the minute "
        "to 18:07, reached while the strategy was down; the market was 99.5 on return")
    assert outage_fill_note(decision, 97.0) is None  # a later part fill
    assert outage_fill_note({"intent": "exit", "signal": {}}, 97.0) is None and outage_fill_note(None, 1.0) is None
