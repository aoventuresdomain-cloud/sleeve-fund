"""A-20 (HoE, 15 Oct 20:37 UK): the PM's Portfolio read "Book value 0". Once the fund holds the cash (the re-base
deposit applied, supervisor.bridge_start) every Strategy is made with 0 (CUT-A (a)), so a book summed from Strategies'
starting balances and equity is 0 while the fund's book is its own cash put in plus what the Strategies made
(supervisor.fund_equity). Home and Risk, the book's curve and the reports read the fund's book; a store from before the
bridge reads as it did. On Postgres too (TEST_DATABASE_URL)."""

import html as htmlmod
import math
import re
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

from sleeve_fund import supervisor
from sleeve_fund.dashboard.metrics import sleeve_summary

AUTH = ("pm", "pw")


@pytest.fixture
def c(store, tmp_path, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    return TestClient(app_mod.create_app(store))


def _hold(store, amount=100_000):  # noqa: F811
    at = datetime.now(UTC) - timedelta(minutes=5)
    store.settle_flow(store.fund_flow("deposit", amount, actor="pm", reason="the PM's reset", rebase=True, now=at), at)


def _make(store, name, balance):  # noqa: F811
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=balance, venue="BINANCE", desired_state="stopped")


def _text(html: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def _kpi(html: str, label: str) -> str:
    m = re.search(rf'<div class="k">{label}</div><div class="v[^"]*">(.*?)</div>', html)
    assert m, f"no {label} tile"
    return _text(m.group(1))


def test_fund_held_with_only_zero_capital_strategies_home_shows_the_funds_book(c, store):  # noqa: F811
    _hold(store)
    _make(store, "one", 0)
    _make(store, "two", 0)
    assert float(supervisor.fund_equity(store)) == 100_000
    html = htmlmod.unescape(c.get("/", auth=AUTH).text)
    assert _kpi(html, "Book value") == "100,000.00"
    assert _kpi(html, "Since start") in ("+0.00%", "0.00%")
    assert "started at 100,000.00" in html and "The fund's own cash and every Strategy's equity" in html


def test_fund_held_the_book_figures_read_the_fund_on_home_and_risk(c, store):  # noqa: F811
    from sleeve_fund.dashboard import book as bookm

    _hold(store)
    _make(store, "one", 0)
    frames = {s.name: bookm.daily(store, s.name) for s in store.sleeves()}
    summaries = [bookm.sleeve_extras(store, sleeve_summary(store, s), frames[s.name]) for s in store.sleeves()]
    book = bookm.book_view(store, summaries, frames)
    assert (book["start"], book["equity"], book["pnl"], book["ret"]) == (100_000, 100_000, 0, 0)
    assert book["capital"] == 100_000 and book["cash"] == 100_000
    assert book["exposure_pct"] == 0 and book["net_pct"] == 0
    for path in ("/", "/risk", "/reports"):
        r = c.get(path, auth=AUTH)
        assert r.status_code == 200, path
        words = _text(r.text)
        assert not re.search(r"\b(nan|inf|-inf)\b", words, re.IGNORECASE), path
    eq = c.get("/api/book/equity", auth=AUTH).json()
    assert eq["start"] == 100_000
    assert all(math.isfinite(v) for v in eq["equity"] + eq["drawdown"])


def test_not_held_the_book_is_the_strategies_own_as_before(c, store):  # noqa: F811
    _make(store, "one", 20_000)
    _make(store, "two", 5_000)
    html = c.get("/", auth=AUTH).text
    assert _kpi(html, "Book value") == "25,000.00"
    assert c.get("/api/book/equity", auth=AUTH).json()["start"] == 25_000


def test_fund_held_the_book_curve_starts_from_the_funds_cash(c, store):  # noqa: F811
    """The curve, its drawdown and the Change baseline count the fund's cash too: a Strategy made with 0 that has made
    500 puts the book at 100,500, not 500, and its first mark is no 100% fall."""
    _hold(store)
    _make(store, "one", 0)
    now = datetime.now(UTC)
    store.record_equity("one", equity=0, cash=0, qty=0, price=60_000, benchmark=0, ts=now - timedelta(days=2))
    store.record_equity("one", equity=500, cash=500, qty=0, price=60_000, benchmark=0, ts=now - timedelta(minutes=1))
    eq = c.get("/api/book/equity", auth=AUTH).json()
    assert eq["start"] == 100_000 and eq["equity"][0] == 100_000 and eq["equity"][-1] == 100_500
    assert max(eq["drawdown"]) == 0
    day = c.get("/api/book/equity?days=1", auth=AUTH).json()
    assert day["equity"][-1] == 100_500 and max(day["drawdown"]) == 0
    assert all(math.isfinite(v) for v in day["equity"] + day["drawdown"])


def test_a_strategy_put_away_after_an_error_is_not_halted_and_waiting_on_home(c, store):  # noqa: F811
    """With B-29's top bar (CR on B-29): Home's own line counted the archived candle-step Strategy left in "error" on
    15 Oct as "1 strategy halted and waiting for you"."""
    _hold(store)
    for name in ("candle-step-btcusdt-1m", "timed-test"):
        _make(store, name, 0)
    store.set_status("candle-step-btcusdt-1m", "error", "node exited 1")
    assert "1 strategy halted and waiting for you" in _text(c.get("/", auth=AUTH).text)  # a live one waits for the PM
    store.archive("candle-step-btcusdt-1m")
    assert "halted and waiting" not in _text(c.get("/", auth=AUTH).text)
