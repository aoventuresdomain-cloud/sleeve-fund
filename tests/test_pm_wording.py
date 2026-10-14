"""PM wording sweep (HoE 14 Oct 16:05 UK, before the PM watches the first test): no PM-facing page or alert says
"sleeve", "coin" or "crypto", and none names a venue outside Setup and Accounts (the only pages that name venues: QA U8,
P1-U23). Rendered on a re-based book as paper runs it: 19,887.78, one running 1x Strategy, one paused Strategy on a
test account outside the fund's book, with fills, orders, alerts and decisions whose stored words name venues, and a
research result. Visible text only: the words, plus the hover and screen-reader text the PM reads (title, aria-label,
placeholder, alt); links, ids, form values and scripts are not read."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from html import unescape

import pytest
from fastapi.testclient import TestClient
from test_development import _sheet

from sleeve_fund.store import Store

AUTH = ("pm", "test-pw")
T0 = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
BOOK = 19_887.78
# Every banned word, any case, as a word ("Sleeves", "coins", "cryptocurrency"; never "Bitcoin" or a key's "_sleeve").
BANNED = re.compile(r"(?<![\w-])(?:sleeves?|coins?|crypto\w*)(?![\w-])", re.IGNORECASE)
VENUES = re.compile(r"(?<![\w-])(?:binance|kraken|bybit|deribit)\w*(?![\w-])", re.IGNORECASE)
# The only page that names venues: Setup, where the PM sets up an account and its key (QA U8). Moved pages (/ops,
# /decisions, /reports, /accounts, /settings) are read where they land.
NAMES_VENUES = ("/setup",)
PAGES = ("/", "/risk", "/alerts", "/sleeves/new", "/sleeves/trend-btc", "/sleeves/tst", "/research",
         "/strategies/trend_filter", "/records", "/backtest", "/trades", "/orders", "/setup")
SHEET = "trend_filter_binance-btcusdt-store-60m_20261005-100000"


def visible(html: str) -> str:
    """The text the PM reads: the page's words and the hover and screen-reader text, not links, ids or scripts."""
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    said = re.findall(r'\s(?:title|aria-label|placeholder|alt)="([^"]*)"', html)
    words = re.sub(r"<[^>]+>", " ", html)
    return " ".join(unescape(" ".join([words, *said])).split())


@pytest.fixture
def book(tmp_path, monkeypatch):
    """The re-based paper book: trend-btc running at 1x (19,887.78 after the re-base), tst paused on the test account
    (outside the fund's book), each with the words a venue sends back stored in its journal."""
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund import history
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="trend-btc", strategy="trend_filter", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-EXTERNAL",
                        starting_balance=20_000, params={"fast": 10, "slow": 30, "market": "perp", "leverage": "1"},
                        venue="BINANCE")
    store.create_sleeve(name="tst", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-HOUR-LAST-EXTERNAL",
                        starting_balance=3_000, params={"fast": 10, "slow": 30})
    store.record_order("trend-btc", order_id="o1", side="BUY", qty=0.1, intent="entry", reason="fast MA crossed above slow",
                       ts=T0)
    store.record_fill("trend-btc", side="BUY", qty=0.1, price=60_000.0, fee=3.0, order_id="o1", trade_id="t1", ts=T0)
    store.record_equity("trend-btc", equity=BOOK, cash=13_887.78, qty=0.1, price=60_000.0, benchmark=20_000,
                        ts=T0 + timedelta(minutes=1))
    store.record_equity("tst", equity=2_950, cash=2_950, qty=0.0, price=100_000.0, benchmark=3_000,
                        ts=T0 + timedelta(minutes=1))
    store.set_status("tst", "paused", "paused by the PM")
    for sleeve, level, kind, said in (
            ("trend-btc", "warning", "stale_price", "[BINANCE BTC/USDT] no trade from BTCUSDT-PERP.BINANCE for 90s"),
            ("trend-btc", "error", "order_rejected", "BinanceClientError: -2019 margin is insufficient"),
            ("tst", "warning", "reconcile_mismatch", "Kraken spot says 0.001 more than the journal"),
            (None, "error", "backup_problem", "pg_dump failed: connection refused")):
        store.event(sleeve, level, kind, said, ts=T0 + timedelta(minutes=2))
    store.decide("PM", "pause", "testing the binance account first", "tst")
    (tmp_path / f"{SHEET}.md").write_text(_sheet("trend_filter", "PASS", minutes=60))
    return TestClient(app_mod.create_app(store)), store


def _hits(text: str, pattern) -> list[str]:
    return [text[max(m.start() - 50, 0):m.end() + 50] for m in pattern.finditer(text)]


@pytest.mark.parametrize("path", PAGES + (f"/results/{SHEET}",))
def test_no_pm_page_says_sleeve_coin_crypto_or_names_a_venue(book, path):
    client, _ = book
    r = client.get(path, auth=AUTH)
    assert r.status_code == 200 and not r.history, path  # a moved page is checked where it lands
    text = visible(r.text)
    assert _hits(text, BANNED) == [], path
    if path not in NAMES_VENUES:
        assert _hits(text, VENUES) == [], path


def test_the_alert_message_sent_outside_says_none_of_them(book):
    from sleeve_fund import alerts

    _, store = book
    text = alerts.message(store.events(min_level="warning"))
    assert "Kraken spot says" not in text and "BinanceClientError" not in text  # the stored words were read
    assert _hits(text, BANNED) == [] and _hits(text, VENUES) == []


def test_the_test_account_strategy_reads_as_outside_the_funds_figures(book):
    """Done-when (4): wherever tst shows, it reads as outside the fund's book, and the book's figures are the fund's."""
    client, _ = book
    home = visible(client.get("/", auth=AUTH).text)
    assert "outside the fund's book: left out of its equity, P&L and open risk" in home
    assert "Book value 19,887.78" in home  # trend-btc's alone, not with tst's 2,950
    page = visible(client.get("/sleeves/tst", auth=AUTH).text)
    assert "On a test account, outside the fund's book: its equity, P&L and risk are not in the fund's figures" in page
    assert "outside the fund's book" not in visible(client.get("/sleeves/trend-btc", auth=AUTH).text)
    risk = client.get("/risk", auth=AUTH).text
    assert "left out of the book's size, drawdown and market-move figures" in visible(risk)
    limits = risk[risk.index('id="lim-h"'):].split("</table>")[0]
    tst_row = limits[limits.index('href="/sleeves/tst"'):].split("</tr>")[0]
    assert "Test account</span>" in tst_row
    assert "Test account</span>" not in limits[limits.index('href="/sleeves/trend-btc"'):].split("</tr>")[0]


def test_the_risk_pages_market_moves_leave_a_running_test_account_strategy_out(book, monkeypatch):
    """A running test-account Strategy with a position adds nothing to the book's P&L if the market moves."""
    from sleeve_fund.dashboard import riskops

    client, store = book
    store.record_equity("tst", equity=2_950, cash=950, qty=0.02, price=100_000.0, benchmark=3_000,
                        ts=T0 + timedelta(minutes=3))
    store.set_desired_state("tst", "running")
    seen, real = [], riskops.stress_bars
    monkeypatch.setattr(riskops, "stress_bars", lambda scenarios: seen.append(scenarios) or real(scenarios))
    assert client.get("/risk", auth=AUTH).status_code == 200
    down10 = next(sc for sc in seen[0] if abs(sc["shock"] + 0.10) < 1e-9)
    assert down10["loss"] == pytest.approx(600)  # trend-btc's 6,000 position alone, not + tst's 2,000
