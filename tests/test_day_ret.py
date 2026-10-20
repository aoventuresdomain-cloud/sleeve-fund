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
