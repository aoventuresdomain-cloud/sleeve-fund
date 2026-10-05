"""The Portfolio page (combined build spec, part F1): KPI boxes, the bottom tabs in order, positions with their
unrealised, realised and fees and a totals row, and the get-started steps on an empty book."""

import re

from test_dashboard import AUTH, SAME, client  # noqa: F401 - the fixture

from sleeve_fund.dashboard import reasons

PERP = {"market": "perp", "allow_short": True}


def _book(store):
    """Two perpetual strategies holding positions, one long with a stop and one short without, and a flat one."""
    store.create_sleeve(name="pp-short", strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000,
                        params={"rise": 0.01, "dip": 0.005, **PERP})
    store.create_sleeve(name="rsi-long", strategy="rsi_bands", instrument="ETH/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000,
                        params={"rsi_period": 14, "long_entry": 30.0, "long_exit": 55.0, "short_entry": 70.0,
                                "short_exit": 50.0, "stop_loss": 0.02, **PERP})
    store.create_sleeve(name="flat-one", strategy="trend_filter", instrument="BTC/USD",
                        bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=5_000, params={"fast": 10, "slow": 30})
    store.record_order("pp-short", order_id="P-1", side="SELL", qty=0.05, intent="entry", reason="rose 1%, sold short")
    store.update_order("P-1", fill_qty=0.05, fill_px=60_000, fee=1.5)
    store.record_fill("pp-short", side="SELL", qty=0.05, price=60_000, fee=1.5, order_id="P-1", trade_id="T-1")
    # Equity 10,048.50: starting 10,000 + 50 unrealised (short 0.05 from 60,000, marked at 59,000) - 1.50 fee.
    store.record_equity("pp-short", equity=10_048.5, cash=12_998.5, qty=-0.05, price=59_000, benchmark=10_000)
    store.record_order("rsi-long", order_id="R-1", side="BUY", qty=1.0, intent="entry", reason="RSI 28 below 30")
    store.update_order("R-1", fill_qty=1.0, fill_px=3_000, fee=1.2)
    store.record_fill("rsi-long", side="BUY", qty=1.0, price=3_000, fee=1.2, order_id="R-1", trade_id="T-2")
    # Equity 9,968.80: 10,000 - 30 unrealised (long 1 from 3,000 at 2,970) - 1.20 fee.
    store.record_equity("rsi-long", equity=9_968.8, cash=6_998.8, qty=1.0, price=2_970, benchmark=10_000)
    store.record_funding("pp-short", qty=-0.05, price=59_500, rate=0.0001, amount=0.3)
    store.record_equity("flat-one", equity=5_000, cash=5_000, qty=0, price=60_000, benchmark=5_000)


def _panel(page, name):
    return page.split(f'data-panel="{name}"')[1].split('class="tab-panel"')[0]


def test_positions_show_unrealised_realised_and_fees_with_a_totals_row(client):  # noqa: F811
    c, store = client
    _book(store)
    page = c.get("/", auth=AUTH).text
    pos = _panel(page, "positions")
    table = pos.split('<table class="book-pos">')[1].split("</table>")[0]
    head = table.split("</thead>")[0]
    for col in ("Strategy", "Instrument", "Side", "Size", "Entry", "Mark", "Liq. price", "Stop", "Unrealised",
                "Realised", "Fees paid"):
        assert f">{col}</th>" in head, col
    rows = table.split("<tbody>")[1].split("</tbody>")[0].split("<tr>")[1:]
    assert len(rows) == 2 and "flat-one" not in table  # only open positions
    short = next(r for r in rows if "pp-short" in r)
    long_ = next(r for r in rows if "rsi-long" in r)
    assert ">Short</span>" in short and ">Long</span>" in long_
    # Unrealised on the open position; realised is the strategy's P&L less that (here, the entry fee); fees paid.
    assert "+50.00" in short and "−1.50" in short and ">1.50</td>" in short
    assert "−30.00" in long_ and "−1.20" in long_ and ">1.20</td>" in long_
    # The short has no stop: an amber "none". The long's 2% stop sits under its entry.
    assert '<span class="warn" title="No stop-loss on this position">none</span>' in short
    assert "2,940.00" in long_
    foot = table.split("<tfoot>")[1].split("</tfoot>")[0]
    assert "Total, 2 positions" in foot
    assert "+20.00" in foot and "−2.70" in foot and ">2.70</td>" in foot
    # Phones get a card per position with the three figures on one line, and the same totals.
    cards = pos.split('<ul class="pos-cards">')[1].split("</ul>")[0]
    assert cards.count('<div class="g3">') == 3 and "Unrealised<b>" in cards and "Fees paid<b>" in cards


def test_close_reuses_the_strategy_flatten_with_a_reason(client):  # noqa: F811
    c, store = client
    _book(store)
    page = c.get("/", auth=AUTH).text
    dlg = page.split('<dialog id="dlg-close-pp-short"')[1].split("</dialog>")[0]
    assert 'action="/sleeves/pp-short/command"' in dlg and 'name="command" value="flatten"' in dlg
    assert "then pauses the strategy" in dlg
    # The Flatten / Close position reason list plus Other, as on the strategy page; confirm waits for a pick.
    for r, _ in reasons.ACTION_REASONS["close"]:
        assert f'name="reason_pick" value="{r}"' in dlg
    assert 'value="Other"' in dlg and 'name="reason_for" value="close"' in dlg
    assert "data-needs-reason disabled" in dlg
    assert 'data-open="dlg-close-pp-short"' in page
    r = c.post("/sleeves/pp-short/command", data={"command": "flatten", "reason_for": "close",
                                                  "reason_pick": "Reducing exposure", "reason_note": ""},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and store.pending_commands("pp-short")[0]["command"] == "flatten"
    assert store.decisions("pp-short", limit=1)[0]["reason"] == "Reducing exposure"
    # A flatten already waiting: Close is disabled rather than queueing a second one.
    again = c.get("/", auth=AUTH).text
    button = again.split('data-open="dlg-close-pp-short"')[1].split(">")[0]
    assert "disabled" in button


def test_bottom_tabs_come_in_the_spec_order_and_are_hash_driven(client):  # noqa: F811
    c, store = client
    _book(store)
    page = c.get("/", auth=AUTH).text
    bar = page.split('aria-label="Book views"')[1].split("</nav>")[0]
    assert re.findall(r'data-tab="(\w+)"', bar) == ["positions", "strategies", "orders", "history", "funding",
                                                   "allocation"]
    labels = re.findall(r'role="tab" data-tab="\w+">([^<]+)<', bar)
    assert labels == ["Positions", "Strategies", "Open orders", "Trade history", "Funding", "Allocation"]
    assert 'href="#positions"' in bar and bar.startswith(" data-tabs")
    assert "Console.tabs()" in page
    # Each tab has its panel, in the same order, and its panel is a live region.
    assert re.findall(r'data-panel="(\w+)"', page) == ["positions", "strategies", "orders", "history", "funding",
                                                      "allocation"]
    assert 'aria-labelledby="positions-h"' in page and 'data-live="book-tabs"' in page
    # Trade history: the book's fills with their reasons; Funding: the perpetual's payment; Allocation and
    # correlation where they were.
    assert "rose 1%, sold short" in _panel(page, "history") and "RSI 28 below 30" in _panel(page, "history")
    assert "+0.30" in _panel(page, "funding")
    assert "Allocation" in _panel(page, "allocation") and "How the strategies move together" in page
    assert "Room to halt" in _panel(page, "strategies")


def test_open_orders_tab_lists_working_orders_only(client):  # noqa: F811
    c, store = client
    _book(store)
    store.record_order("pp-short", order_id="P-2", side="BUY", qty=0.05, intent="take_profit",
                       reason="cover after a 0.5% dip", order_type="LIMIT")
    store.update_order("P-2", status="accepted")
    orders = _panel(c.get("/", auth=AUTH).text, "orders")
    assert "cover after a 0.5% dip" in orders and "rose 1%, sold short" not in orders  # the filled one isn't open


def test_kpi_row_has_the_nine_boxes(client):  # noqa: F811
    c, store = client
    _book(store)
    page = c.get("/", auth=AUTH).text
    kpis = page.split('<section class="kpis" aria-label="Book figures">')[1].split("</section>")[0]
    names = re.findall(r'<div class="k">([^<]+?) <', kpis)
    assert names == ["Book equity", "Today", "Month to date", "Since start", "Gross exposure", "Cash", "Drawdown",
                     "Sharpe", "Fees paid"]
    assert kpis.count('<div class="kpi') == 9 and 'class="kpi lead"' in kpis
    assert "25,017.30" in kpis  # book equity: 10,048.50 + 9,968.80 + 5,000
    assert "nearest halt" in kpis and "3 fills" not in kpis and "2 fills" in kpis


def test_an_empty_book_still_shows_get_started(client):  # noqa: F811
    c, _ = client
    page = c.get("/", auth=AUTH).text
    assert "Get started in three steps" in page
    assert 'aria-label="Book views"' not in page and 'class="kpis"' not in page
