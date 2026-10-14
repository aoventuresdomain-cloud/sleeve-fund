"""C8, the Strategy page fix (ui-v2/compliance-build-spec.md, the PM's look of 14 Oct 2026): one Position view under
the chart with every view a tab at a time; Signals and a read-only Settings beside the chart; Path to live a button
immediately left of Clone with changes; no Reset on the header; KPIs without an equity tile; the compliance bars
under the positions; a take-profit the engine dropped shown struck through with its reason."""

from __future__ import annotations

import re

from test_strategy_page import AUTH, _sleeve, client  # noqa: F401 - the fixture


def _head(page: str) -> str:
    return page[page.index('class="page-head"'):page.index("</div>", page.index('class="actions"'))]


def test_the_page_has_one_position_view_and_its_views_in_one_panel(client):  # noqa: F811
    c, store = client
    _sleeve(store, held=0.01, params={"stop_loss": 0.02, "risk_per_trade": 0.01})
    page = c.get("/sleeves/btc-x", auth=AUTH).text
    assert 'data-tabs' not in page and "desk-tabs" not in page  # no top tab row
    lower = page.split('class="sub-tabs" role="tablist"')[1].split("</div>")[0]
    assert re.findall(r'data-sub="(\w+)"', lower) == ["pos", "trades", "open", "history", "refused", "activity"]
    assert page.count('data-sub-panel="pos"') == 1 and 'id="pos-h"' not in page and "<h3>Size</h3>" not in page
    side = page.split('class="sub-tabs side"')[1].split("</div>")[0]
    assert re.findall(r'data-sub="(\w+)"', side) == ["signals", "settings"]
    settings = page.split('id="tab-settings-side"')[1].split("</aside>")[0]
    assert "Risk per trade: 1% of the fund" in settings and "Leverage at most 1× · SL required" in settings
    assert " cap" not in settings.lower() and "limit" not in settings.lower()
    assert 'data-open="dlg-settings"' in settings and 'id="dlg-settings"' in page and 'id="settings-form"' in page


def test_the_header_has_path_to_live_left_of_clone_and_no_reset(client):  # noqa: F811
    c, store = client
    _sleeve(store)
    head = _head(c.get("/sleeves/btc-x", auth=AUTH).text)
    assert "Reset" not in head and 'data-open="dlg-reset"' not in head
    buttons = re.findall(r'(data-open="dlg-path"|>Clone with changes<)', head)
    assert buttons == ['data-open="dlg-path"', ">Clone with changes<"]
    between = head[head.index('data-open="dlg-path"'):head.index(">Clone with changes<")]
    assert between.count("<button") == 0 and between.count("<a ") == 1  # nothing between the two
    assert re.search(r"Path to live <span class=\"faint\">\w[\w ]* · G2 \d+/\d+</span>", head)


def test_the_kpis_have_no_equity_and_show_the_contribution(client):  # noqa: F811
    c, store = client
    _sleeve(store)
    page = c.get("/sleeves/btc-x", auth=AUTH).text
    figures = page[page.index('aria-label="Strategy figures"'):page.index("</section>", page.index('aria-label="Strategy figures"'))]
    assert [k.strip() for k in re.findall(r'<div class="k">([^<{]+)', figures)] == [
        "P&amp;L since start", "Today", "Contribution to fund", "Unrealised", "Realised", "Fees and funding"]
    assert "Equity" not in figures


def test_the_bars_say_dash_until_the_fund_is_marked(client):  # noqa: F811
    c, store = client
    _sleeve(store, held=0.01, params={"stop_loss": 0.02, "risk_per_trade": 0.01})
    pos = c.get("/sleeves/btc-x", auth=AUTH).text.split('data-sub-panel="pos"')[1].split('data-sub-panel=')[0]
    keys = re.findall(r'data-bar="(\w+)"', pos)
    assert keys == ["risk_per_trade", "asset_net", "asset_stop_loss"]
    assert "Net on BTC" in pos and "BTC loss if stops hit" in pos and "The fund&#39;s book has no mark yet" in pos


def test_a_dropped_take_profit_shows_struck_through_with_its_reason(client):  # noqa: F811
    c, store = client
    store.create_sleeve(name="btc-x", strategy="rsi_bands", instrument="BTC/USD", bar_spec="15-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"stop_loss": 0.02})
    store.record_order("btc-x", order_id="E-1", side="BUY", qty=0.01, intent="entry", reason="RSI 28 at or below 30",
                       signal={"tp_dropped": "below round-trip cost", "tp_target": 0.001, "tp_break_even": 0.0011,
                               "tp_drop_code": "below_round_trip_cost", "tp_drop_step": "entry_target_check"})
    store.record_fill("btc-x", side="BUY", qty=0.01, price=60_000, fee=4.8, order_id="E-1", trade_id="T-1")
    store.record_equity("btc-x", equity=10_000, cash=9_400, qty=0.01, price=60_000, benchmark=10_000)
    store.heartbeat("btc-x")
    pos = c.get("/sleeves/btc-x", auth=AUTH).text.split('data-sub-panel="pos"')[1]
    assert "Take-profit <s>0.10%</s> dropped: below round-trip cost (break-even 0.11%)" in pos
