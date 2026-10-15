"""F-B5-PAGE (HoQA, Drop B MUST): a multi-asset Strategy's page covers every asset, not only its first. One row per
asset with its net from Store.asset_book (the journal), the Strategy's total, every asset's trades, and fees and
funding over every leg. The hand case is HoQA's b5: "m" on BTC, ETH and SOL, all closed."""

import re
from datetime import timedelta
from decimal import Decimal as D
from html import unescape

import pytest
from test_halt_ref import T0, store  # noqa: F401

AUTH = ("pm", "pw")
MIN = timedelta(minutes=1)
ASSETS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
PERP = {"market": "perp", "allow_short": True, "stop_loss": 0.02}


def _row(db, asset):
    return {a["instrument"]: a["sleeve"] for a in db.strategy_assets("m")}[asset]


def _fill(db, asset, side, qty, price, fee, at, oid):
    row = _row(db, asset)
    db.record_order(row, order_id=oid, side=side, qty=qty, intent="entry" if side == "BUY" else "exit", reason="r",
                       ts=at)
    db.record_fill(row, side=side, qty=qty, price=price, fee=fee, order_id=oid, trade_id=f"t-{oid}", ts=at)
    db.update_order(oid, status="filled")


@pytest.fixture
def m(store):
    """BTC net +120.66, ETH -68.83, SOL -0.29: the Strategy +51.54, fees 3.91, funding -0.12 (paid)."""
    at = T0 - timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", D(100_000), actor="pm", reason="QA", rebase=True, now=at), at)
    store.create_legs(name="m", strategy="ping_pong", bar_spec="1-MINUTE-LAST-INTERNAL", assets=list(ASSETS),
                      params=dict(PERP), starting_balance=0, universe_date="2026-10-15", venue="BINANCE",
                      desired_state="running")
    _fill(store, "BTC/USDT", "BUY", 0.1, 60_000.0, 1.20, T0, "b1")
    _fill(store, "BTC/USDT", "SELL", 0.1, 61_234.5, 1.22, T0 + MIN, "b2")
    store.record_funding(_row(store, "BTC/USDT"), qty=0.0, price=1.0, rate=0.0, amount=-0.37, ts=T0 + MIN)
    _fill(store, "ETH/USDT", "BUY", 2.0, 3_000.0, 0.60, T0, "e1")
    _fill(store, "ETH/USDT", "SELL", 2.0, 2_966.055, 0.59, T0 + MIN, "e2")
    store.record_funding(_row(store, "ETH/USDT"), qty=0.0, price=1.0, rate=0.0, amount=0.25, ts=T0 + MIN)
    _fill(store, "SOL/USDT", "BUY", 10.0, 150.0, 0.15, T0, "s1")
    _fill(store, "SOL/USDT", "SELL", 10.0, 150.001, 0.15, T0 + MIN, "s2")
    return store


@pytest.fixture
def page(m, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    html = unescape(TestClient(dash.create_app(m)).get("/sleeves/m", auth=AUTH).text)
    return html, " ".join(re.sub(r"<[^>]+>", " ", html).split()).replace("−", "-")


def test_each_asset_has_its_row_with_its_net_and_the_strategy_total_is_their_sum(page):
    html, _words = page
    rows = {r.group(1): r.group(0) for r in re.finditer(r'<tr data-asset="(\w+)"[^>]*>.*?</tr>', html, re.DOTALL)}
    assert set(rows) == {"BTC", "ETH", "SOL", "_total"}
    for asset, net in (("BTC", "+120.66"), ("ETH", "-68.83"), ("SOL", "-0.29"), ("_total", "+51.54")):
        assert net in rows[asset].replace("−", "-"), asset


def test_the_header_and_the_kpis_cover_the_whole_strategy(page):
    _html, words = page
    assert "Ping pong · BTC, ETH and SOL" in words
    assert "P&L since start +51.54" in words
    assert "Fees and funding 4.03" in words  # fees 3.91 and funding paid 0.12, every leg


def test_the_trade_list_holds_every_assets_trades(page):
    html, _words = page
    assert re.search(r'data-sub="trades"[^>]*>Trades<span class="n">3</span>', html)
    trades = re.findall(r'<td class="num" data-m="hide">[\d.]+ (BTC|ETH|SOL)', html)
    assert sorted(trades) == ["BTC", "ETH", "SOL"]


# ---------------------------------------------------------------- F-B5-FLATTEN (HoE): Close and Flatten, every leg

HDR = {"Origin": "http://testserver"}


@pytest.fixture
def held(m, monkeypatch):
    """"m" holding 0.1 BTC long and 2 ETH short, SOL flat; "solo", another Strategy, holding 1 BTC."""
    from test_halt_ref import _make

    from sleeve_fund.dashboard import app as dash

    _fill(m, "BTC/USDT", "BUY", 0.1, 60_000.0, 1.20, T0 + 2 * MIN, "b3")
    _fill(m, "ETH/USDT", "SELL", 2.0, 3_000.0, 0.60, T0 + 2 * MIN, "e3")
    _make(m, "solo", 0)
    m.record_fill("solo", side="BUY", qty=1.0, price=60_000.0, fee=0.0, order_id="o1", trade_id="t1", ts=T0)
    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    from fastapi.testclient import TestClient

    return m, TestClient(dash.create_app(m))


def _dialog(client, name, cmd):
    html = unescape(client.get(f"/sleeves/{name}", auth=AUTH).text)
    body = re.search(rf'<dialog[^>]*id="dlg-{cmd}".*?</dialog>', html, re.DOTALL)
    return " ".join(re.sub(r"<[^>]+>", " ", body.group(0)).split()), html


def test_the_flatten_dialog_names_every_open_leg_with_its_asset_and_quantity(held):
    _db, client = held
    words, html = _dialog(client, "m", "flatten")
    assert "It sells 0.1 BTC and buys back 2 ETH at market, paying the taker fee, then pauses the strategy." in words
    assert "SOL" not in words  # flat: nothing of it to close
    assert re.search(r'data-open="dlg-flatten"\s*>Flatten</button>', html)  # offered: a leg holds


def test_a_flatten_reaches_every_leg_of_the_strategy_and_nothing_else_in_the_fund(held):
    db, client = held
    r = client.post("/sleeves/m/command", data={"command": "flatten", "reason": "done for today"}, auth=AUTH,
                    headers=HDR, follow_redirects=False)
    assert r.status_code == 303 and "command_error" not in r.headers["location"]
    told = {n for n in [s.name for s in db.sleeves()] if any(c["command"] == "flatten"
                                                            for c in db.pending_commands(n))}
    assert told == {a["sleeve"] for a in db.strategy_assets("m")}  # every asset's row, each its own runtime
    assert "solo" not in told


def test_a_single_asset_strategys_flatten_words_are_unchanged(held):
    _db, client = held
    words, _html = _dialog(client, "solo", "flatten")
    assert "it starts only to sell 1 BTC (about" in words and "every leg" not in words


def test_the_book_kill_switch_flattens_every_open_leg_of_a_multi_asset_strategy(held):
    """HoE: the kill switch tells a Strategy through its first row (legs()[0]); every asset leg, ETH and SOL
    included, gets the flatten, as every one of the Strategy's positions must close."""
    db, client = held
    r = client.post("/book/flatten", data={"reason": "kill switch drill"}, auth=AUTH, headers=HDR,
                    follow_redirects=False)
    assert r.status_code == 303 and "kill_error" not in r.headers["location"]
    told = {n for n in [s.name for s in db.sleeves()] if any(c["command"] == "flatten"
                                                            for c in db.pending_commands(n))}
    assert {a["sleeve"] for a in db.strategy_assets("m")} <= told
