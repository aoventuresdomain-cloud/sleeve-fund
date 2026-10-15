"""F-WN-2 end to end (HoE, CR F-WN2-1): with the fund holding the cash, a Strategy put away flat is journaled
withdrawn 0 by the store itself (PE2's F-WN-1), and its page and Portfolio say it left with nothing to move, its P&L
kept in the fund's cash; none of the fund's book is its own any more. SQLite, and Postgres with TEST_DATABASE_URL."""

from __future__ import annotations

from decimal import Decimal
from html import unescape

from fastapi.testclient import TestClient
from test_drop_a_fund_book import MIN, _rebased
from test_halt_ref import T0, _make, _mark, store  # noqa: F401

AUTH = ("pm", "pw")


def test_a_strategy_put_away_under_fund_held_cash_reads_as_leaving_with_nothing_to_move(store, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as dash

    _rebased(store)
    _make(store, "y", 0)
    _mark(store, T0 + MIN)
    store.archive("y")
    assert store.withdrawn() == {"y": Decimal(0)}  # the store's own journaled 0, no stub
    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    c = TestClient(dash.create_app(store))
    page = unescape(c.get("/sleeves/y", auth=AUTH).text)
    assert "Archived, it left with nothing to move: its P&L stays in the fund's cash, which isn't a loss." in page
    assert "its equity left the fund's book" not in page
    home = unescape(c.get("/", auth=AUTH).text)
    assert "stopped and put away; none of the fund's book is theirs any more</span>" in home
