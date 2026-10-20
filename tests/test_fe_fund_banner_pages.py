"""Head 4 (HoE, from QT3's item 3): while the fund's daily pause or drawdown halt is in force, every fund page that lists
positions or limits shows it as Portfolio does. The Risk page and Trades (both list the open positions, their margin
and liquidation) were missing the banner. The structural cell keeps a new page listing positions or limits from
missing it too."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_c7_fund_banner import _banner, _seed
from test_dashboard import AUTH, client  # noqa: F401 - the fixture
from test_strategy_page import _sleeve

from sleeve_fund.dashboard import app as app_mod

PAGES = ["/", "/risk", "/trades", "/sleeves/btc-x", "/portfolio-settings"]
STATES = [
    ({"state": "brake", "since": None, "until": datetime(2026, 10, 16, tzinfo=UTC), "reason": None},
     "New trades paused until the next day begins (01:00 UK)."),
    ({"state": "halt", "since": None, "until": None, "reason": "Drawdown 15% from the peak",
      "restart_url": "/fund/restart"}, "The fund is flattened and halted."),
]
# What marks a template as listing positions or the fund's limits: the shared positions table, the stop/liquidation
# and risk cells, the limits-after preview, or the limits table's own "Limit, share of the fund" heading.
LISTS = re.compile(r'_positions\.html|positions_table|risk_cell|stop_liq|limits_after|Limit, share of the fund')
TEMPLATES = Path(app_mod.__file__).parent / "templates"


@pytest.mark.parametrize("path", PAGES)
@pytest.mark.parametrize(("state", "words"), STATES, ids=["pause", "halt"])
def test_every_fund_page_listing_positions_or_limits_shows_the_brake(client, monkeypatch, path, state, words):  # noqa: F811
    c, store = client
    _sleeve(store)
    _seed(monkeypatch, state)
    page = c.get(path, auth=AUTH).text
    b = _banner(page)
    assert f'data-fund-state="{state["state"]}"' in b and words in b, path
    assert page.count("fund-banner") == 1, path  # one banner, at the top


@pytest.mark.parametrize("path", ["/risk", "/trades"])
def test_no_banner_on_risk_or_trades_while_the_fund_is_normal(client, monkeypatch, path):  # noqa: F811
    c, store = client
    _sleeve(store)
    _seed(monkeypatch, {"state": "normal", "since": None, "until": None, "reason": None})
    assert "fund-banner" not in c.get(path, auth=AUTH).text


def test_every_page_template_listing_positions_or_limits_is_a_banner_page():
    pages = {p.name for p in TEMPLATES.glob("*.html") if not p.name.startswith("_") and LISTS.search(p.read_text())}
    assert {"home.html", "risk.html", "trades.html", "sleeve.html", "portfolio_settings.html"} <= pages
    assert pages <= app_mod.FUND_BANNER_PAGES, sorted(pages - app_mod.FUND_BANNER_PAGES)
