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
    # Its 2% stop journaled with the entry, as paper journals a fixed stop (base.py), which the open-risk limit reads.
    store.record_order("rsi-long", order_id="R-1", side="BUY", qty=1.0, intent="entry", reason="RSI 28 below 30",
                       signal={"stop_frac": 0.02})
    store.update_order("R-1", fill_qty=1.0, fill_px=3_000, fee=1.2)
    store.record_fill("rsi-long", side="BUY", qty=1.0, price=3_000, fee=1.2, order_id="R-1", trade_id="T-2")
    # Equity 9,968.80: 10,000 - 30 unrealised (long 1 from 3,000 at 2,970) - 1.20 fee.
    store.record_equity("rsi-long", equity=9_968.8, cash=6_998.8, qty=1.0, price=2_970, benchmark=10_000)
    store.record_funding("pp-short", qty=-0.05, price=59_500, rate=0.0001, amount=0.3)
    store.record_equity("flat-one", equity=5_000, cash=5_000, qty=0, price=60_000, benchmark=5_000)


def _panel(page, name):
    return page.split(f'data-panel="{name}"')[1].split('class="tab-panel"')[0]


COLUMNS = ["Instrument", "Side", "Leverage", "Size", "Notional", "Entry", "Mark", "SL", "TP", "Liq. price",
           "Unrealised", "Risk to stop", "Fees"]


def _head(table):
    return re.findall(r"<th[^>]*>([^<]+)</th>", table.split("</thead>")[0])


def test_positions_table_has_the_v2_columns_and_no_percentages(client):  # noqa: F811
    c, store = client
    _book(store)
    page = c.get("/", auth=AUTH).text
    pos = _panel(page, "positions")
    table = pos.split('<table class="book-pos">')[1].split("</table>")[0]
    assert _head(table) == ["Strategy", *COLUMNS]
    rows = table.split("<tbody>")[1].split("</tbody>")[0].split("<tr>")[1:]
    assert len(rows) == 2 and "flat-one" not in table  # only open positions
    short = next(r for r in rows if "pp-short" in r)
    long_ = next(r for r in rows if "rsi-long" in r)
    assert ">Short</span>" in short and ">Long</span>" in long_
    # Leverage in its own column; notional in money; unrealised and fees; no realised (it moved to Trades).
    from sleeve_fund.dashboard import trading

    # The position's own leverage (item 7: notional at entry over its isolated margin, the profile's 2x cap)
    # once the figures exist; until then notional over the strategy's whole equity.
    lev = "2×" if hasattr(trading, "open_risk") else "0.3×"
    assert f'class="lev">{lev}' in short and "2,950.00" in short and "+50.00" in short and ">1.50</td>" in short
    assert "−30.00" in long_ and ">1.20</td>" in long_ and "Realised" not in table
    # The short has neither stop nor target: an amber "none" stop, "none" target, and an unbounded risk.
    assert '<span class="warn" title="No stop-loss on this position">none</span>' in short
    assert '<span class="faint" title="No take-profit on this position">none</span>' in short and " est.</span>" in short
    assert "2,940.00" in long_  # the long's 2% stop under its 3,000 entry
    # No cell contains a % (a distance or share goes in a title).
    cells = re.sub(r'title="[^"]*"', "", table)
    assert "%" not in cells
    foot = table.split("<tfoot>")[1].split("</tfoot>")[0]
    assert "Total, 2 positions" in foot and "5,920.00" in foot and "+20.00" in foot and ">2.70</td>" in foot
    # Phones get a card per position with the same figures.
    cards = pos.split('<ul class="pos-cards">')[1].split("</ul>")[0]
    assert cards.count('<div class="g3">') == 2 and "Risk to stop<b>" in cards


def test_strategy_page_positions_use_the_same_table(client):  # noqa: F811
    c, store = client
    _book(store)
    page = c.get("/sleeves/rsi-long", auth=AUTH).text
    table = page.split('data-sub-panel="pos"')[1].split('<table class="book-pos">')[1].split("</table>")[0]
    assert _head(table) == COLUMNS  # no Strategy column on the strategy's own page
    assert 'data-open="dlg-close"' in table and "2,940.00" in table and "<tfoot>" not in table


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
    assert re.findall(r'data-tab="(\w+)"', bar) == ["positions", "strategies", "orders", "history", "funding"]
    labels = re.findall(r'role="tab" data-tab="\w+">([^<]+)<', bar)
    assert labels == ["Positions", "Strategies", "Open orders", "Trade history", "Funding"]
    assert 'href="#positions"' in bar and bar.startswith(" data-tabs")
    assert "Console.tabs()" in page
    # Each tab has its panel, in the same order, and its panel is a live region.
    assert re.findall(r'data-panel="(\w+)"', page) == ["positions", "strategies", "orders", "history", "funding"]
    assert 'aria-labelledby="positions-h"' in page and 'data-live="book-tabs"' in page
    # Trade history: the book's fills with their reasons; Funding: the perpetual's payment. Allocation is on the
    # page now, not a tab, and the correlation grid moved to Risk & health (UI v2).
    assert "rose 1%, sold short" in _panel(page, "history") and "RSI 28 below 30" in _panel(page, "history")
    assert "+0.30" in _panel(page, "funding")
    assert "How the strategies move together" not in page
    assert "How the strategies move together" in c.get("/risk", auth=AUTH).text
    assert "Room to halt" in _panel(page, "strategies")


def test_open_orders_tab_lists_working_orders_only(client):  # noqa: F811
    c, store = client
    _book(store)
    store.record_order("pp-short", order_id="P-2", side="BUY", qty=0.05, intent="take_profit",
                       reason="cover after a 0.5% dip", order_type="LIMIT")
    store.update_order("P-2", status="accepted")
    orders = _panel(c.get("/", auth=AUTH).text, "orders")
    assert "cover after a 0.5% dip" in orders and "rose 1%, sold short" not in orders  # the filled one isn't open


def test_kpi_ledger_has_the_eight_tiles_with_the_detail_on_hover(client):  # noqa: F811
    c, store = client
    _book(store)
    page = c.get("/", auth=AUTH).text
    kpis = page.split('<section class="kpis ledger8" aria-label="Book figures">')[1].split("</section>")[0]
    names = re.findall(r'<div class="k">([^<]+)</div>', kpis)
    assert names == ["Book value", "Today", "Month to date", "Since start", "Margin used", "Open risk", "Drawdown",
                     "Fees and funding"]
    # A label and a number only: no sub-lines, no help buttons; each tile's detail is its title.
    assert 'class="s"' not in kpis and 'class="help"' not in kpis
    assert kpis.count('<div class="kpi') == kpis.count(' title="') == 8 and 'class="kpi lead"' in kpis
    assert "25,017.30" in kpis  # book value is the sum of strategy equities: 10,048.50 + 9,968.80 + 5,000
    # Fees and funding: 2.70 of fees, and the short received 0.30 of funding, so 2.40 net cost.
    assert ">2.40</div>" in kpis and "funding received 0.30" in kpis
    # Margin used and Open risk come from the position figures (item 7): until those exist they wait and say
    # so, rather than show the old whole-equity margin.
    from sleeve_fund.dashboard import trading

    if hasattr(trading, "open_risk"):
        assert 'title="3,000.00 isolated margin across open positions"><div class="k">Margin used</div><div class="v">12%' in kpis
        # FE v2: the stopless perp short is counted at the limit's estimate and named (Advisor 7 Oct 18:50 UK).
        assert "pp-short: no stop, counted at the larger of 10% and 3 daily ATRs" in kpis
        assert re.search(r'<div class="v warn">[\d,.]+ · 1 estimated</div>', kpis)
    else:
        assert kpis.count('class="kpi pending"') == 2
    assert "gross exposure" not in page.lower() and "Sharpe" not in kpis


def test_needs_you_bar_shows_only_while_an_alert_waits(client):  # noqa: F811
    c, store = client
    _book(store)
    assert 'class="needs-you"' not in c.get("/", auth=AUTH).text
    store.event("pp-short", "error", "halted", "hit its drawdown limit")
    store.event("rsi-long", "warning", "mark_unavailable", "price feed quiet")
    bar = c.get("/", auth=AUTH).text.split('class="needs-you"')[1].split("</section>")[0]
    assert "price feed quiet" in bar and "1 more" in bar and 'action="/alerts/' in bar  # the newest first


def test_pnl_by_strategy_rows_add_up_to_the_book_row_for_each_period(client):  # noqa: F811
    c, store = client
    _book(store)
    page = c.get("/", auth=AUTH).text
    panel = page.split('aria-labelledby="move-h" data-periods>')[1].split("</section>")[0]
    assert re.findall(r'data-period="(\w+)"', panel) == ["day", "week", "mtd"]
    for key in ("day", "week", "mtd"):
        body = panel.split(f'data-period-rows="{key}"')[1].split("</tbody>")[0]
        cells = re.findall(r'<td class="num">(?:<span class="\w*">)?([+−]?[\d,.]+)', body)
        vals = [float(v.replace("−", "-").replace(",", "")) for v in cells]
        *rows, total = vals
        assert len(rows) == 3 and abs(sum(rows) - total) < 0.011, key
        assert "<th scope=\"row\">Book</th>" in body
    # Today shows first; the others wait behind the switch. Biggest contributor first.
    assert 'data-period-rows="day">' in panel and 'data-period-rows="week" hidden' in panel
    day = panel.split('data-period-rows="day"')[1].split("</tbody>")[0]
    assert day.index("pp-short") < day.index("flat-one") < day.index("rsi-long")


def test_an_archived_strategy_gets_one_row_so_the_rows_still_add_up_to_book(client):  # noqa: F811
    """QA U1: the book's figures count an archived strategy, so its P&L goes in one Archived row."""
    c, store = client
    _book(store)
    store.set_desired_state("pp-short", "stopped")
    store.archive("pp-short")
    page = c.get("/", auth=AUTH).text
    panel = page.split('aria-labelledby="move-h" data-periods>')[1].split("</section>")[0]
    for key in ("day", "week", "mtd"):
        body = panel.split(f'data-period-rows="{key}"')[1].split("</tbody>")[0]
        cells = re.findall(r'<td class="num">(?:<span class="\w*">)?([+−]?[\d,.]+)', body)
        *rows, total = [float(v.replace("−", "-").replace(",", "")) for v in cells]
        assert len(rows) == 3 and abs(sum(rows) - total) < 0.011, key
        assert 'class="archived-row"' in body and 'title="pp-short"' in body and "/sleeves/pp-short" not in body
    assert 'id="archived"' in page


def test_a_positions_fees_are_its_own_not_the_strategys_since_it_started(client):  # noqa: F811
    """QA U4: after a closed trip, the Fees cell is the open position's fees, and the footer adds those up."""
    c, store = client
    _book(store)
    store.record_fill("rsi-long", side="SELL", qty=1.0, price=3_010, fee=1.0, order_id="R-2", trade_id="T-3")
    store.record_fill("rsi-long", side="BUY", qty=1.0, price=2_990, fee=0.7, order_id="R-3", trade_id="T-4")
    store.record_equity("rsi-long", equity=9_986.1, cash=6_996.1, qty=1.0, price=2_970, benchmark=10_000)
    table = _panel(c.get("/", auth=AUTH).text, "positions")
    row = next(r for r in table.split("<tr") if "rsi-long" in r)
    assert re.findall(r'<td class="num">([\d,.]+)</td>', row)[-1] == "0.70"  # not 2.90, the strategy's fees
    foot = table.split("<tfoot")[1]
    assert re.findall(r'<td class="num">([\d,.]+)</td>', foot)[-1] == "2.20"  # 0.70 + pp-short's 1.50


def test_allocation_nets_holdings_by_instrument_and_flags_crossing(client):  # noqa: F811
    c, store = client
    _book(store)
    # A second strategy long the instrument pp-short is short: one netted row, flagged crossing.
    store.create_sleeve(name="btc-long", strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000,
                        params={"rise": 0.01, "dip": 0.005, **PERP})
    store.record_order("btc-long", order_id="B-1", side="BUY", qty=0.02, intent="entry", reason="dipped")
    store.update_order("B-1", fill_qty=0.02, fill_px=59_000, fee=0.5)
    store.record_fill("btc-long", side="BUY", qty=0.02, price=59_000, fee=0.5, order_id="B-1", trade_id="T-3")
    store.record_equity("btc-long", equity=9_999.5, cash=8_819.5, qty=0.02, price=59_000, benchmark=10_000)
    page = c.get("/", auth=AUTH).text
    panel = page.split('aria-labelledby="alloc-h"')[1].split("</section>")[0]
    rows = panel.split("<tbody>")[1].split("</tbody>")[0].split("<tr>")[1:]
    assert len(rows) == 2  # BTC/USDT netted across two strategies, ETH/USDT
    eth, btc = rows  # largest notional first: ETH 2,970 against BTC net short 0.03 x 59,000 = 1,770
    assert "ETH/USDT" in eth and ">Long</span>" in eth and "2,970.00" in eth
    assert "BTC/USDT" in btc and ">Short</span>" in btc and "1,770.00" in btc
    assert "crossing" in btc and "crossing" not in eth and "pp-short" in btc and "btc-long" in btc
    # Weight of book is notional over book value (35,016.80).
    assert ">8%" in eth and ">5%" in btc
    # The bar's segments follow the rows, in the same order and colours.
    seg = re.findall(r'<i style="flex:[\d.]+;background:(var\(--[\w-]+\))"', panel)
    sw = re.findall(r'class="swatch" style="background:(var\(--[\w-]+\))"', panel)
    assert seg == sw and len(seg) == 2


def test_allocation_shows_at_most_ten_holdings():
    from sleeve_fund.dashboard.book import holdings

    rows = [{"pair": f"I{i}/USDT", "perp": None, "qty": 1.0, "value": 100.0 + i, "unrealised": 0.0, "sleeve": f"s{i}",
             "side": 1} for i in range(12)]
    h = holdings(rows, 10_000)
    assert len(h["rows"]) == 10 and h["count"] == 12 and h["rows"][0]["instrument"] == "I11/USDT"
    assert abs(sum(r["share"] for r in holdings(rows[:3], 10_000)["rows"]) - 1) < 1e-9


def test_chart_js_is_gone_and_no_page_says_gross_exposure(client):  # noqa: F811
    from pathlib import Path

    c, store = client
    _book(store)
    for path in ("/", "/sleeves/rsi-long", "/trades", "/backtest", "/sleeves/new"):
        page = c.get(path, auth=AUTH).text
        assert "chart.umd" not in page and "<canvas" not in page, path
    for path in ("/", "/sleeves/rsi-long", "/trades"):
        assert "gross exposure" not in c.get(path, auth=AUTH).text.lower(), path
    static = Path(__file__).parents[1] / "sleeve_fund" / "dashboard" / "static"
    assert not (static / "chart.umd.min.js").exists() and "new Chart(" not in (static / "console.js").read_text()


def test_an_empty_book_still_shows_get_started(client):  # noqa: F811
    c, _ = client
    page = c.get("/", auth=AUTH).text
    assert "Get started in three steps" in page
    assert 'aria-label="Book views"' not in page and 'class="kpis"' not in page
