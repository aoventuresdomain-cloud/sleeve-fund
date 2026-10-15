"""F-A1 (HoQA attribution rebind, Drop B): a Strategy's money P&L adds up in Decimal to the cent (F-A1-DECIMAL), and
Portfolio's attribution has a fund line, so the Strategies' contributions plus the fund line are the fund's return
exactly (F-A1-FUNDLINE). Hand figures: x and y start with 10,000 each (book 20,000); x makes +1,234.56 and y -234.55,
so the fund makes 1,000.01, 5.00005% of its book. SQLite, and Postgres with TEST_DATABASE_URL."""

from __future__ import annotations

import re
from decimal import Decimal as D
from html import unescape

from fastapi.testclient import TestClient
from test_drop_a_fund_book import MIN
from test_halt_ref import T0, _make, _mark, _trade, store  # noqa: F401

from sleeve_fund.dashboard import attribution

AUTH = ("pm", "pw")


def _book(store):  # noqa: F811
    _make(store, "x", 10_000)
    _make(store, "y", 10_000)
    _mark(store, T0)
    _trade(store, "x", 1_234.56, T0 + MIN)
    _trade(store, "y", -234.55, T0 + MIN)  # 765.45 - 1,000 in floats is -234.54999999999995
    _mark(store, T0 + 2 * MIN)


def test_a_strategys_pnl_is_summed_in_decimal_to_the_cent(store):  # noqa: F811
    _book(store)
    s = store.sleeve("y")
    a = attribution.strategy(store, "y", attribution.trips_of(store, s), T0 + 3 * MIN)
    assert isinstance(a["pnl"], D) and a["pnl"] == D("-234.55")
    assert a["contribution"] == D("-234.55") / D(20_000)


def test_float_trip_pnls_add_up_to_the_cent_not_to_a_float_tail(store):  # noqa: F811
    _make(store, "z", 10_000)
    _mark(store, T0)
    trips = [{"closed": T0 + MIN, "pnl": 0.1, "fees": 0.1, "funding": 0.0}, {"closed": T0 + MIN, "pnl": 0.2,
             "fees": 0.2, "funding": None}, {"closed": T0 + MIN, "pnl": -234.54999999999995, "fees": 0.0}]
    a = attribution.strategy(store, "z", trips, T0 + 2 * MIN)
    assert a["pnl"] == D("-234.25") and a["fees"] == D("0.3") and a["funding"] == 0


def test_the_strategies_plus_the_fund_line_are_the_funds_return_exactly(store):  # noqa: F811
    _book(store)
    f = attribution.fund(store, [store.sleeve("x"), store.sleeve("y")], T0 + 3 * MIN)
    rows = {r["name"]: r for r in f["rows"]}
    assert rows["x"]["pnl"] == D("1234.56") and rows["y"]["pnl"] == D("-234.55")
    assert rows["x"]["contribution"] == D("1234.56") / D(20_000)
    assert f["total"]["contribution"] == D("1000.01") / D(20_000) and f["total"]["pnl"] == D("1000.01")
    assert sum(r["contribution"] for r in f["rows"]) + f["fund"]["contribution"] == f["total"]["contribution"]
    assert sum(r["pnl"] for r in f["rows"]) + f["fund"]["pnl"] == f["total"]["pnl"]
    assert f["fund"]["pnl"] == 0


def test_portfolio_shows_the_rows_the_fund_line_and_the_total(store, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as dash

    _book(store)
    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    html = TestClient(dash.create_app(store)).get("/", auth=AUTH).text
    panel = html.split('aria-labelledby="attrib-h"')[1].split("</section>")[0]
    words = " ".join(re.sub(r"<[^>]+>", " ", unescape(panel)).split())
    assert "x 1 +1,234.56 +6.17%" in words and "y 1 −234.55 -1.17%" in words
    assert "Fund: open positions, costs and cash 0.00 +0.00%" in words
    assert "Fund total +1,000.01 +5.00%" in words


def test_before_the_books_first_mark_there_is_no_total_and_says_so(store):  # noqa: F811
    _make(store, "x", 10_000)
    f = attribution.fund(store, [store.sleeve("x")], T0)
    assert f["total"] is None and f["fund"] is None and f["rows"][0]["contribution"] is None
