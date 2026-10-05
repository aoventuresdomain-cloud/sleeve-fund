"""v2 P1-2: every paper or live order's bar close, arrival, decision, send, acceptance and fills, to the
microsecond, for close-to-fill times per strategy."""

from datetime import datetime, timezone

from sleeve_fund.store import Store
from tests.test_dashboard import AUTH, client  # noqa: F401

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
    assert late_bar(bars, NS + 20_000_000_000, minute, down, at(NS + 1_000_000)) is None  # acted on before
    assert late_bar([], NS, minute, down, None) is None
