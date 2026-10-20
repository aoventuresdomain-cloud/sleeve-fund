"""B-20 in the browser: on a fund-held store (the re-base deposit applied, a Strategy made with 0 holding a perp), each
page the PM opens says nothing of a Strategy's own halt, room, capital or drawdown, with no script error, and the New
Strategy summary the page's own script writes names the fund's equity and leverage, never "its capital"."""

import re

import pytest
from test_browser import (  # noqa: F401 - the browser fixture
    PASSWORD,
    _open,
    _Served,
    _step,
    browser,
)
from test_fe_b20_fund_held_wording import OLD_RULE, _hold, _long

PAGES = ["/", "/risk", "/setup", "/sleeves/held-long", "/reports", "/trades"]


@pytest.fixture
def held_site(tmp_path, monkeypatch):
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.dashboard import charts
    from sleeve_fund.store import Store

    monkeypatch.setenv("DASHBOARD_PASSWORD", PASSWORD)
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    listed = {"kraken": ["BTC/USD", "ETH/USD"], "binance": ["BTC/USDT", "ETH/USDT"]}
    monkeypatch.setattr(charts, "instruments", lambda get_json=None, venue=None: listed.get(venue, []))
    store = Store(f"sqlite:///{tmp_path}/b20.db")
    _hold(store)
    _long(store)
    with _Served(store) as base:
        yield base, store


@pytest.mark.parametrize("path", PAGES)
def test_b20_a_fund_held_page_states_no_strategys_own_halt_or_capital(held_site, browser, path):  # noqa: F811
    base, _ = held_site
    page, errors = _open(browser, f"{base}{path}")
    text = " ".join(page.inner_text("body").split())
    # The book's chart (Home, Risk, Reports) is the fund's own, so its worst drawdown is the fund's and stays.
    rules = OLD_RULE if path.startswith("/sleeves/") else tuple(w for w in OLD_RULE if w != r"Worst drawdown")
    hits = [w for w in rules if re.search(w, text)]
    assert hits == [], (path, hits)
    assert errors == [], (path, errors)
    page.context.close()


def test_b20_the_new_strategy_summary_names_the_funds_equity_and_leverage(held_site, browser):  # noqa: F811
    base, _ = held_site
    page, errors = _open(browser, f"{base}/sleeves/new?instrument=BTC/USDT&venue=binance")
    page.evaluate("() => { const f = document.getElementById('sleeve-form'); f.elements.market.value = 'perp';"
                  " f.elements.market.dispatchEvent(new Event('change')); }")
    _step(page, "Model")
    page.select_option("#strategy", "ping_pong")
    _step(page, "Orders and exits")
    page.select_option("#stop_kind", "pct")
    page.fill("#stop_loss_pct", "1")
    page.fill("#risk_per_trade_pct", "1")
    page.dispatch_event("#risk_per_trade_pct", "input")
    summary = page.inner_text("#summary")
    assert "of its capital" not in summary and "x its capital" not in summary, summary
    assert "leverage up to" in summary, summary
    assert "of the fund's equity if the stop is hit" in summary, summary
    assert errors == []
    page.context.close()
