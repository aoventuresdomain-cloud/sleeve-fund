"""DAY-RET (A) on Home (HoE, D4): inside the fund, the Today tile says what the day opened at, its deposits and
withdrawals and its P&L, from QD's compliance.day_open, called exactly once a render (it may journal its unreconciled
warning). The day starts 00:00 UTC, shown in UK time across the clock change. More than 1 USDT between the book's P&L
today and the Strategies' sum shows as unreconciled. On Postgres too (TEST_DATABASE_URL)."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

from test_day_open import _mark
from test_fe_b20_fund_held_wording import _long, _summaries, _words, c  # noqa: F401
from test_halt_ref import store  # noqa: F401

from sleeve_fund.dashboard import compliance
from sleeve_fund.dashboard.compliance import DAY_OPEN_UNRECONCILED, day_open
from sleeve_fund.portfolio.gate import next_utc_midnight


def _day(store, off):  # noqa: F811
    """The fund held from before today's open, one Strategy holding a long: 100,000 at the open, and now that plus the
    Strategies' day P&L plus `off` (the book's P&L the Strategies don't have), no flow today. Returns their day P&L."""
    from sleeve_fund.dashboard import book as bookm

    now = datetime.now(UTC)
    opened = next_utc_midnight(now) - timedelta(days=1)
    at = opened - timedelta(hours=2)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="the PM's reset", rebase=True, now=at), at)
    _mark(store, opened - timedelta(seconds=30), "100000")
    _long(store)
    xs, frames = _summaries(store)
    day = D(str(bookm.book_view(store, xs, frames)["day_pnl"]))
    _mark(store, now - timedelta(seconds=5), str(D(100000) + day + D(off)))
    return day


def _tile(c):  # noqa: F811
    words, _ = _words(c, "/")
    start = words.index("Today")
    return words[start:words.index("Month to date")]


def test_home_calls_day_open_exactly_once_a_render(c, store, monkeypatch):  # noqa: F811
    _day(store, "0.50")
    calls = []
    real = compliance.day_open
    monkeypatch.setattr(compliance, "day_open", lambda *a, **k: calls.append(k) or real(*a, **k))
    _tile(c)
    assert len(calls) == 1 and "day_pnl" in calls[0], calls
    _tile(c)
    assert len(calls) == 2


def test_the_today_tile_shows_the_recorded_opening_in_uk_time(c, store):  # noqa: F811
    day = _day(store, "0.50")
    line = day_open(store, day_pnl=day)
    tile = _tile(c)
    assert f"Opened today at 100,000.00 ({line['opened']})" in tile, tile
    pnl = day + D("0.50")
    assert "deposits and withdrawals today 0.00" in tile and f"P&L today {'+' if pnl > 0 else '−'}{abs(pnl):,.2f}" in tile, tile
    assert "unreconciled" not in tile.lower(), tile


def test_the_unreconciled_case_renders_and_is_journaled_once(c, store):  # noqa: F811
    _day(store, "250")  # the book made 250 today that no Strategy did
    tile = _tile(c)
    assert "Unreconciled: the book's P&L today is 250.00 away from the Strategies'" in tile, tile
    _tile(c)
    assert len(store.events_of((DAY_OPEN_UNRECONCILED,), limit=10)) == 1


def test_the_day_opens_at_00_00_utc_shown_in_uk_time_either_side_of_the_clock_change(store):  # noqa: F811
    for now, uk in ((datetime(2026, 10, 24, 12, tzinfo=UTC), "01:00 UK"), (datetime(2026, 10, 26, 12, tzinfo=UTC),
                                                                          "00:00 UK")):
        _mark(store, now.replace(hour=0) - timedelta(minutes=1), "100000")
        _mark(store, now, "100000")
        assert day_open(store, now, day_pnl=0)["opened"] == uk, now


def test_outside_the_fund_home_shows_no_opening_line(c, store):  # noqa: F811
    _long(store, "own-long", balance=10_000)
    assert "Opened today" not in _tile(c)
