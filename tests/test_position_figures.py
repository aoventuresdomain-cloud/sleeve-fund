"""Margin, liquidation, open risk and costs (UI v2, item 7): worked out once, in the view dicts, so Portfolio, the
strategy page, Risk & health and Trades show the same figures for the same position and no template does sums."""

import pytest
from fastapi.templating import Jinja2Templates
from test_dashboard import AUTH, client  # noqa: F401 - the fixture
from test_portfolio_page import _book

from sleeve_fund import markets
from sleeve_fund.dashboard.trading import risk_to_stop
from sleeve_fund.risk import profile

FIGURES = ("margin", "leverage", "liq_px", "to_liq", "risk_to_stop", "notional", "stop_px")


def _spy(monkeypatch) -> dict:
    """Records the context each page is rendered with, by template name (the latest one)."""
    seen = {}
    render = Jinja2Templates.TemplateResponse

    def spy(self, request, name, context=None, *a, **k):
        seen[name] = context
        return render(self, request, name, context, *a, **k)

    monkeypatch.setattr(Jinja2Templates, "TemplateResponse", spy)
    return seen


@pytest.fixture
def pages(client, monkeypatch):  # noqa: F811
    c, store = client
    seen = _spy(monkeypatch)
    _book(store)
    for url in ("/", "/sleeves/pp-short", "/sleeves/rsi-long", "/risk", "/trades"):
        assert c.get(url, auth=AUTH).status_code == 200, url
    return seen


def _by_page(seen, name):
    """One strategy's open position as each page holds it."""
    return {
        "portfolio": next(r for r in seen["home.html"]["positions"]["rows"] if r["sleeve"] == name),
        "risk": next(r["position"] for r in seen["risk.html"]["risk"]["rows"] if r["x"]["sleeve"].name == name),
        "trades": next(p for p in seen["trades.html"]["h"]["positions"] if p["sleeve"] == name),
    }


def test_every_page_shows_the_same_margin_and_liquidation_for_a_position(client, monkeypatch):  # noqa: F811
    c, store = client
    seen = _spy(monkeypatch)
    _book(store)
    lev = profile("balanced").max_leverage
    mm = markets.terms({"market": "perp"}, "binance").maintenance_margin
    for name, qty, entry, cash in (("pp-short", -0.05, 60_000, 12_998.5), ("rsi-long", 1.0, 3_000, 6_998.8)):
        for url in ("/", f"/sleeves/{name}", "/risk", "/trades"):
            assert c.get(url, auth=AUTH).status_code == 200, url
        page = {**_by_page(seen, name), "strategy": seen["sleeve.html"]["position"]}
        # Isolated: notional at entry over the leverage cap, and liquidated from that margin alone.
        margin = abs(qty) * entry / lev
        liq = (margin - qty * entry) / (mm * abs(qty) - qty)
        for where, p in page.items():
            assert p["margin"] == pytest.approx(margin), where
            assert p["leverage"] == pytest.approx(lev), where
            assert p["liq_px"] == pytest.approx(liq), where
            assert {k: p[k] for k in FIGURES} == {k: page["strategy"][k] for k in FIGURES}, where
        # The strategy page's perpetual panel takes the same two figures.
        perp = seen["sleeve.html"]["perp"]
        assert (perp["margin"], perp["liq_px"]) == (page["strategy"]["margin"], page["strategy"]["liq_px"])


def test_open_risk_is_the_sum_of_risk_to_stop_and_a_position_without_a_stop_is_named(pages):
    port, risk, trades = pages["home.html"]["positions"], pages["risk.html"]["risk"], pages["trades.html"]["h"]
    long_ = next(r for r in port["rows"] if r["sleeve"] == "rsi-long")
    short = next(r for r in port["rows"] if r["sleeve"] == "pp-short")
    # Long 1 at 3,000 with a 2% stop (2,940), marked at 2,970: 30 to lose if the stop is hit. The short has none.
    assert long_["risk_to_stop"] == pytest.approx(30.0) and short["risk_to_stop"] is None
    for view in (port, risk, trades):
        assert view["open_risk"] == pytest.approx(sum(r["risk_to_stop"] or 0.0 for r in port["rows"])) == 30.0
        assert view["unbounded"] == ["pp-short"]
        assert view["margin"] == pytest.approx(long_["margin"] + short["margin"])


def test_risk_to_stop_is_the_move_to_the_stop_for_either_side():
    assert risk_to_stop(2.0, 100.0, 95.0) == pytest.approx(10.0)  # long: a fall to the stop
    assert risk_to_stop(-2.0, 100.0, 104.0) == pytest.approx(8.0)  # short: a rise to the stop
    assert risk_to_stop(2.0, 94.0, 95.0) == 0.0  # already through it: the stop exit is due, nothing more to lose
    assert risk_to_stop(2.0, 100.0, None) is None  # no stop: unbounded


def test_fees_and_funding_as_a_share_of_gross_pnl(pages):
    x = next(x for x in pages["home.html"]["summaries"] if x["sleeve"].name == "pp-short")
    # 1.50 in fees, 0.30 of funding received: 1.20 of costs on a P&L of +48.50 after them, +49.70 before.
    assert x["funding"] == pytest.approx(0.3) and x["costs"] == pytest.approx(1.2)
    assert x["gross_pnl"] == pytest.approx(49.7) and x["cost_share"] == pytest.approx(1.2 / 49.7)
    book = pages["home.html"]["book"]
    # The book: 2.70 of fees less 0.30 received, on +17.30 after costs.
    assert book["funding"] == pytest.approx(0.3) and book["costs"] == pytest.approx(2.4)
    assert book["gross_pnl"] == pytest.approx(19.7) and book["cost_share"] == pytest.approx(2.4 / 19.7)
    flat = next(x for x in pages["home.html"]["summaries"] if x["sleeve"].name == "flat-one")
    assert flat["costs"] == 0.0 and flat["cost_share"] is None  # nothing made or paid yet


def test_the_strategy_page_limit_bar_measures_margin_against_the_profile_cap(pages):
    risk, x = pages["sleeve.html"]["risk"], pages["sleeve.html"]["x"]  # the last strategy page read: rsi-long
    cap = profile("balanced").max_position_pct * x["equity"]
    assert risk["margin"] == pytest.approx(1_500.0) and risk["margin_cap"] == pytest.approx(cap)
    assert risk["margin_used"] == pytest.approx(1_500.0 / cap)
    assert x["room"] > 0  # room to halt in money, as #116 left it
