"""B-23 display: Portfolio and Risk & health show the account's marked equity against the book at the same read, the
gap and how it is measured, and that the book sizes every entry; never a venue's name. Display only."""

import re
from datetime import UTC, datetime, timedelta

import pytest
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres too (TEST_DATABASE_URL)
)

from sleeve_fund import account_capital as ac

AUTH = ("pm", "pw")
NOW = datetime(2026, 10, 20, 19, tzinfo=UTC)


def _read(store, equity, book, at, anchor=False):
    store.record_account_read("paper", ok=True, at=at, anchor=anchor, payload={
        "coin": "USDT", "wallet": equity, "unrealised": "0", "equity": equity, "other": [], "book": book})


def _section(store, monkeypatch, path):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as dash

    store.create_sleeve(name="a", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp"})  # Portfolio's empty state has no figures
    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(dash, "utcnow", lambda: NOW)
    html = TestClient(dash.create_app(store)).get(path, auth=AUTH).text
    sect = re.search(r'aria-label="Account and book".*?</section>', html, re.DOTALL)
    assert sect, f"no account panel on {path}"
    return " ".join(re.sub(r"<[^>]+>", " ", sect.group(0)).split())


@pytest.mark.parametrize("path", ["/", "/risk"])
def test_before_any_read_the_panel_says_so_and_the_book_sizes(store, monkeypatch, path):  # noqa: F811
    words = _section(store, monkeypatch, path)
    assert "Not read yet" in words and "Size from the account figure: Off" in words and "Bybit" not in words


@pytest.mark.parametrize("path", ["/", "/risk"])
def test_account_book_and_drift_since_the_anchor(store, monkeypatch, path):  # noqa: F811
    _read(store, "50000", "100000", NOW - timedelta(minutes=10), anchor=True)
    _read(store, "50700", "100000", NOW - timedelta(minutes=1))
    words = _section(store, monkeypatch, path)
    assert "50,700.00" in words and "100,000.00" in words and "+700.00" in words
    assert "0.70% of book, drift since the anchor read at" in words and "alert above 500.00" in words
    assert ac.view(store, "paper", NOW).alerting
