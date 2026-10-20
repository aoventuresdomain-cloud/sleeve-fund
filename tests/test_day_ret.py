"""DAY-RET (Advisor (B), 20 Oct): today's return is the daily pause's own ratio, equity over the day's start scaled by
each flow (gate.apply_flow), the time-weighted return; the page and the brake read the one function."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pytest

from sleeve_fund.portfolio.gate import MemoryLedger, apply_flow, day_ret, mark_book
from sleeve_fund.portfolio.limits import book_breach, day_return
from sleeve_fund.risk import PortfolioProfile

OPEN = datetime(2026, 10, 20, 0, 5, tzinfo=UTC)
PROFILE = PortfolioProfile()  # the 3% daily pause


def _day(*steps):
    """A book opened at 100,000 then the steps in order: ("mark", equity) or ("flow", amount). Returns the ledger and
    what each mark did."""
    led = MemoryLedger(equity=D(100_000))
    mark_book(led, D(100_000), OPEN - timedelta(minutes=10), PROFILE)  # yesterday's last mark: today's start
    mark_book(led, D(100_000), OPEN, PROFILE)
    acts, t, book = [], OPEN, D(100_000)
    for kind, x in steps:
        t += timedelta(minutes=5)
        if kind == "flow":
            apply_flow(led, D(x), now=t)
            book += D(x)
        else:
            book = D(x)
            acts.append(mark_book(led, book, t, PROFILE))
    return led, acts


def _pct(r):
    return float(round(r, 6))


def test_100k_open_then_94k_is_minus_6_percent():
    led, _ = _day(("mark", 94_000))
    assert _pct(day_ret(led.state())) == -0.06


def test_a_10k_withdrawal_with_flat_pnl_is_0_percent_and_never_pauses():
    led, acts = _day(("flow", -10_000), ("mark", 90_000))
    assert _pct(day_ret(led.state())) == 0.0 and acts == [None] and led.state().paused is None


def test_a_10k_deposit_then_a_3k_loss_is_minus_2_73_percent_and_no_pause():
    led, acts = _day(("flow", 10_000), ("mark", 107_000))
    assert _pct(day_ret(led.state())) == pytest.approx(-0.027273, abs=1e-6)
    assert acts == [None] and led.state().paused is None


def test_a_3k_loss_then_a_10k_deposit_is_minus_3_percent_and_the_pause_fires():
    led, acts = _day(("mark", 97_000), ("flow", 10_000), ("mark", 107_000))
    assert acts[0] == "pause" and led.state().paused
    assert _pct(day_ret(led.state())) == pytest.approx(-0.03, abs=1e-6)


def test_a_3k_loss_then_a_10k_withdrawal_is_minus_3_percent():
    led, _ = _day(("mark", 97_000), ("flow", -10_000), ("mark", 87_000))
    assert _pct(day_ret(led.state())) == pytest.approx(-0.03, abs=1e-6)


def test_the_page_figure_is_the_brakes_byte_for_byte(monkeypatch):
    """The compliance page's T5 day use and the pause's own words come from the one function."""
    from sleeve_fund.dashboard import compliance as page
    from sleeve_fund.portfolio.limits import Book

    led, _ = _day(("flow", 10_000), ("mark", 106_000))
    st = led.state()
    r = day_ret(st)
    breach = book_breach(PROFILE, st.equity, st.reference(), st.day_start_equity)
    assert breach.action == "pause" and f"lost {-r:.1%} since 00:00 UTC" in breach.reason
    monkeypatch.setattr(page, "_gate_book", lambda store: (Book(equity=st.equity), st))
    used = page.use_now(None, {})["fund_day_loss"]["used"]
    assert used == -r  # the same Decimal, not a second sum
    assert str(used) == str(-day_return(st.equity, st.day_start_equity))


def test_no_day_start_or_one_at_0_has_no_return():
    assert day_return(D(100), None) is None and day_return(None, D(100)) is None
    assert day_return(D(100), D(0)) is None and day_return(D(100), D(-1)) is None


def test_the_page_reads_the_pauses_function_never_its_own_sum(monkeypatch):
    """FE wires the page to the one function: a page doing its own division would match the numbers above and drift
    the day either changes."""
    from sleeve_fund.dashboard import compliance as page
    from sleeve_fund.portfolio.limits import Book

    led, _ = _day(("mark", 98_000))
    st = led.state()
    monkeypatch.setattr(page, "_gate_book", lambda store: (Book(equity=st.equity), st))
    monkeypatch.setattr(page, "day_return", lambda equity, day_start: D("-0.0123"))
    assert page.use_now(None, {})["fund_day_loss"]["used"] == D("0.0123")


def test_after_a_day_pause_rebase_the_label_says_since_when_and_the_figure_is_the_brakes():
    """Advisor (B) condition: after a B-31 re-base mid-day the figure measures from the re-base, so its label says so;
    the figure is still the brake's own ratio."""
    from dataclasses import replace

    from sleeve_fund.portfolio.gate import day_figure

    led, _ = _day(("mark", 98_000))
    at = OPEN + timedelta(hours=13)  # 14:05 UK
    st = replace(led.state(), day_start_equity=D(98_000), equity=D(97_000))  # re-based to the book then, then a loss
    fig = day_figure(st, at + timedelta(hours=1), rebased_at=at)
    assert fig["label"] == "Today (since re-base at 14:05 UK)"
    assert fig["value"] == day_return(D(97_000), D(98_000))
    breach_words = book_breach(PortfolioProfile(daily_loss=0.01), st.equity, st.reference(), st.day_start_equity)
    assert f"{-fig['value']:.1%}" in breach_words.reason
    assert day_figure(st, at, rebased_at=at - timedelta(days=1))["label"] == "Today"  # yesterday's re-base: the open


def test_on_a_flow_day_the_page_shows_the_pauses_figure_not_the_curves_change():
    """Advisor condition 2: after a flow the time-weighted figure and the book's money change differ; the page shows
    the brake's."""
    from sleeve_fund.portfolio.gate import day_figure

    led, _ = _day(("flow", 10_000), ("mark", 107_000))
    st = led.state()
    fig = day_figure(st, OPEN + timedelta(hours=2))
    money_change = D(107_000) / D(100_000) - 1  # +7%: the deposit read as a gain
    assert fig["value"] == day_ret(st) and _pct(fig["value"]) == pytest.approx(-0.027273, abs=1e-6)
    assert fig["value"] != money_change


@pytest.mark.parametrize("now, resets", [(datetime(2026, 10, 20, 12, tzinfo=UTC), "01:00 UK"),
                                         (datetime(2026, 10, 26, 12, tzinfo=UTC), "00:00 UK")])
def test_the_day_resets_at_00_00_utc_said_in_uk_time_across_the_clock_change(now, resets):
    """The reset is 00:00 UTC (risk.trading_day): 01:00 UK in summer, 00:00 UK from Sun 25 Oct, worked out, never
    written as a fixed hour."""
    from sleeve_fund.portfolio.gate import PortfolioState, day_figure

    assert day_figure(PortfolioState(), now)["resets"] == resets


def test_the_pages_today_figure_reads_the_funds_state_and_todays_day_pause_rebase(tmp_path):
    from sleeve_fund.dashboard.compliance import day_figure
    from sleeve_fund.gate_ledger import DbLedger
    from sleeve_fund.portfolio.gate import PortfolioState
    from sleeve_fund.risk import trading_day
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/d.db", portfolio_gate=True)
    now = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)
    DbLedger(store, lambda: ()).set_state(PortfolioState(equity=D(97_000), mark_ts=now, hwm=D(100_000),
                                                         day=trading_day(now), day_start_equity=D(98_000)))
    assert day_figure(store, now)["label"] == "Today"
    store.event(None, "info", "portfolio_brake_rebased", "T5 drawdown halt switched back on: ...", ts=now)
    assert day_figure(store, now)["label"] == "Today"  # the drawdown's re-base isn't the day's
    store.event(None, "info", "portfolio_brake_rebased", "T5 daily loss pause switched back on: ...",
                ts=now - timedelta(minutes=30))
    fig = day_figure(store, now)
    assert fig["label"] == "Today (since re-base at 15:30 UK)" and fig["value"] == day_return(D(97_000), D(98_000))


def test_a_busy_day_keeps_the_day_pause_rebase_label(tmp_path):
    """CR DAY-RET follow-up (a): the re-base is found by its kind, not among the last few hundred events."""
    from sleeve_fund.dashboard.compliance import day_figure
    from sleeve_fund.gate_ledger import DbLedger
    from sleeve_fund.portfolio.gate import PortfolioState
    from sleeve_fund.risk import trading_day
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/d.db", portfolio_gate=True)
    now = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)
    DbLedger(store, lambda: ()).set_state(PortfolioState(equity=D(97_000), mark_ts=now, hwm=D(100_000),
                                                         day=trading_day(now), day_start_equity=D(98_000)))
    store.event(None, "info", "portfolio_brake_rebased", "T5 daily loss pause switched back on: ...",
                ts=now - timedelta(minutes=30))
    for i in range(250):
        store.event(None, "info", "mark", f"busy {i}", ts=now)
    assert day_figure(store, now)["label"] == "Today (since re-base at 15:30 UK)"
