"""DAY-RET (A), Advisor option 2 (20 Oct): the day's opening book in money, read on demand (no column): "Opened today
at X; deposits/withdrawals today Y; P&L today Z", from the book's minute marks and the fund's flows. A display and
reconciliation line only: the T5 brake keeps DAY-RET (B)'s scaled start. Z must match Home's day P&L (the Strategies'
sum); more than 1 USDT apart is shown as unreconciled and journaled once a day as a warning."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pytest
from sqlalchemy import insert

from sleeve_fund.dashboard.compliance import DAY_OPEN_UNRECONCILED, day_open
from sleeve_fund.store import Store, book_marks_t

NOW = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)  # 16:00 UK; the day opened 00:00 UTC, 01:00 UK


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path}/d.db", portfolio_gate=True)


def _mark(store, ts, equity):
    e = D(equity)
    with store.engine.begin() as c:
        c.execute(insert(book_marks_t), [{"ts": ts, "equity": e, "gross": D(0), "net_max": D(0), "margin_used": D(0),
                                          "open_risk": D(0), "hwm": e, "reference_equity": e, "day_start_equity": e,
                                          "status": "ok", "profile_version": 1}])


def _flow(store, kind, amount, at):
    fid = store.fund_flow(kind, amount, actor="pm", reason="cell", now=at)
    store.settle_flow(fid, at)


def test_nothing_marked_yet_has_no_line(store):
    assert day_open(store, NOW) is None


def test_a_deposit_mid_day_leaves_the_opening_and_is_not_p_and_l(store):
    _mark(store, NOW.replace(hour=0) - timedelta(minutes=1), "100000")
    _flow(store, "deposit", "10000", NOW - timedelta(hours=3))
    _mark(store, NOW - timedelta(minutes=1), "110500")
    line = day_open(store, NOW)
    assert (line["open"], line["flows"], line["pnl"]) == (D(100_000), D(10_000), D(500))
    assert line["opened"] == "01:00 UK" and line["last_mark"] is None  # a mark just before midnight: no gap to name


def test_a_withdrawal_counts_negative_and_one_before_the_open_isnt_today(store):
    _flow(store, "deposit", "5000", NOW.replace(hour=0) - timedelta(hours=2))  # yesterday's
    _mark(store, NOW.replace(hour=0) - timedelta(minutes=1), "100000")
    _flow(store, "withdraw", "2000", NOW - timedelta(hours=2))
    _mark(store, NOW - timedelta(minutes=1), "97700")
    line = day_open(store, NOW)
    assert (line["flows"], line["pnl"]) == (D(-2000), D(-300))


def test_a_gap_over_midnight_names_the_last_mark_it_opened_at(store):
    _mark(store, NOW.replace(hour=0) - timedelta(hours=3), "100000")  # down from 22:00 UTC, back after midnight
    _mark(store, NOW - timedelta(minutes=1), "100200")
    line = day_open(store, NOW)
    assert line["open"] == D(100_000) and line["last_mark"] == "22:00 UK"  # 21:00 UTC on 19 Oct is 22:00 UK


def test_the_opening_is_shown_in_uk_time_from_the_utc_day_across_the_clock_change(store):
    after = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)  # GMT from Sun 25 Oct: the day opens 00:00 UK
    _mark(store, after.replace(hour=0) - timedelta(minutes=1), "100000")
    _mark(store, after, "100000")
    assert day_open(store, after)["opened"] == "00:00 UK"


def test_a_p_and_l_more_than_1_usdt_from_the_strategies_is_unreconciled_and_journaled_once_a_day(store):
    _mark(store, NOW.replace(hour=0) - timedelta(minutes=1), "100000")
    _mark(store, NOW - timedelta(minutes=1), "100500")
    assert day_open(store, NOW, day_pnl=499.5)["unreconciled"] is None  # within 1 USDT
    assert not store.events_of((DAY_OPEN_UNRECONCILED,))
    line = day_open(store, NOW, day_pnl=480.0)
    assert line["unreconciled"] == D("20.0")
    day_open(store, NOW + timedelta(minutes=5), day_pnl=480.0)
    [e] = store.events_of((DAY_OPEN_UNRECONCILED,))
    assert e["level"] == "warning" and "20.00" in e["message"]
    day_open(store, NOW + timedelta(days=1), day_pnl=480.0)  # a new day warns again
    assert len(store.events_of((DAY_OPEN_UNRECONCILED,))) == 2
