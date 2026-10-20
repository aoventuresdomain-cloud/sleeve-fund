"""B-28 (HoE, Head 5): a Strategy's Positions tab is the Portfolio's positions table, the same component with the same
columns, order and formatting, filtered to that Strategy. On a fund-held store (the re-base deposit applied, Strategies
made with 0) a perp's margin, liquidation and stop there equal the Portfolio row for it, and the margin is never 0 (B-36);
quantity, entry and unrealised P&L equal the journal's book. A flat Strategy reads the Portfolio's empty message and a
put-away one shows no live rows. On Postgres too (TEST_DATABASE_URL)."""

import html as htmlmod
import re

import pytest
from fastapi.testclient import TestClient
from test_browser import (  # noqa: F401 - the browser fixture
    PASSWORD,
    _open,
    _Served,
    browser,
)
from test_fe_b20_fund_held_wording import _engine_liquidation, _hold, _summaries
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

AUTH = ("pm", "pw")
PERP = {"market": "perp", "allow_short": True}
ENTRY, MARK, FEE, STOP = 60_000.0, 59_000.0, 3.3, 0.02
EMPTY = "All in cash. A position appears here as soon as a Strategy buys or sells short."


@pytest.fixture
def c(store, tmp_path, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    return TestClient(app_mod.create_app(store))


def _long(store, name, qty=0.1):  # noqa: F811
    """A fund-held perp long (made with 0) of `qty` at 60,000 with a 2% stop, marked at 59,000, journaled as the
    engine journals it."""
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0, desired_state="running",
                        params={"rise": 0.01, "dip": 0.005, "stop_loss": STOP, **PERP})
    store.record_order(name, order_id=f"{name}-1", side="BUY", qty=qty, intent="entry", reason="dipped 0.5%")
    store.update_order(f"{name}-1", fill_qty=qty, fill_px=ENTRY, fee=FEE)
    store.record_fill(name, side="BUY", qty=qty, price=ENTRY, fee=FEE, order_id=f"{name}-1", trade_id=f"{name}-T")
    cash = -qty * ENTRY - FEE
    store.record_equity(name, equity=cash + qty * MARK, cash=cash, qty=qty, price=MARK, benchmark=0)
    return store.sleeve(name)


def _html(c, path):
    r = c.get(path, auth=AUTH)
    assert r.status_code == 200, path
    return htmlmod.unescape(r.text)


def _table(page):
    m = re.search(r'<table class="book-pos">.*?</table>', page, re.DOTALL)
    return m.group(0) if m else ""


def _thead(page):
    m = re.search(r"<thead>.*?</thead>", _table(page), re.DOTALL)
    assert m, "no positions table"
    return m.group(0)


def _cells(page, name):
    """The row's cells for Strategy `name`, the Close cell left out (it opens each page's own dialog)."""
    m = re.search(rf'<tr data-pos="{re.escape(name)}">(.*?)</tr>', _table(page), re.DOTALL)
    assert m, f"no row for {name}"
    cells = re.findall(r"<td[^>]*>(.*?)</td>", m.group(1), re.DOTALL)
    return [" ".join(re.sub(r"<[^>]+>", " ", x).split()) for x in cells if "close-btn" not in x]


def _words(page):
    return " ".join(re.sub(r"<[^>]+>", " ", page).split())


def test_the_strategy_tab_is_the_portfolios_table(c, store):  # noqa: F811
    _hold(store)
    _long(store, "a")
    home, tab = _html(c, "/"), _html(c, "/sleeves/a")
    assert _thead(tab) == _thead(home)  # same columns, same order, same headings


def test_a_fund_held_perps_row_on_its_tab_is_its_portfolio_row(c, store):  # noqa: F811
    """Margin, liquidation, stop and every other figure, written the same; the margin is the engine's, never 0."""
    _hold(store)
    s = _long(store, "a")
    home, tab = _html(c, "/"), _html(c, "/sleeves/a")
    assert _cells(tab, "a") == _cells(home, "a")
    liq, posted = _engine_liquidation(store, s)
    row = _cells(tab, "a")
    assert f"margin {posted:,.2f}" in " ".join(row), row
    assert "margin 0.00" not in " ".join(row)
    assert f"{liq:,.2f}" in row[-4] or f"{round(liq):,}" in row[-4], row  # the Liq. price cell
    assert f"{ENTRY * (1 - STOP):,.2f}" in " ".join(row), row  # SL


def test_the_tabs_quantity_entry_and_unrealised_are_the_ledgers(c, store):  # noqa: F811
    from sleeve_fund.dashboard import trading

    _hold(store)
    _long(store, "a")
    ledger = store.journal_book("a", 0)
    summaries, _ = _summaries(store)
    row = trading.book_positions(store, summaries)["rows"][0]
    assert row["qty"] == pytest.approx(float(ledger["qty"]))
    assert row["entry_px"] == pytest.approx(float(ledger["entry_px"]))
    assert row["unrealised"] == pytest.approx(float(ledger["qty"]) * (MARK - float(ledger["entry_px"])))
    words = " ".join(_cells(_html(c, "/sleeves/a"), "a"))
    assert f"{float(ledger['entry_px']):,.2f}" in words and f"{float(ledger['qty']) * (MARK - ENTRY):,.2f}".replace("-", "\u2212") in words


def test_a_flat_strategy_reads_the_portfolios_empty_message(c, store):  # noqa: F811
    _hold(store)
    store.create_sleeve(name="flat", strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0, desired_state="stopped",
                        params={"rise": 0.01, "dip": 0.005})
    assert EMPTY in _words(_html(c, "/"))
    tab = _words(_html(c, "/sleeves/flat"))
    assert EMPTY in tab and "The strategy is stopped" in tab


def test_a_put_away_strategy_shows_no_live_rows(c, store):  # noqa: F811
    _hold(store)
    store.create_sleeve(name="gone", strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0, desired_state="stopped",
                        params={"rise": 0.01, "dip": 0.005})
    store.archive("gone")
    tab = _html(c, "/sleeves/gone")
    assert 'data-pos="' not in _table(tab) and EMPTY in _words(tab)


@pytest.fixture
def shared_site(tmp_path, monkeypatch):
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.store import Store

    monkeypatch.setenv("DASHBOARD_PASSWORD", PASSWORD)
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    db = Store(f"sqlite:///{tmp_path}/b28.db")
    _hold(db)
    _long(db, "one", 0.1)
    _long(db, "two", 0.05)
    with _Served(db) as base:
        yield base, db


def _rows(page):
    return page.evaluate("""() => [...document.querySelectorAll('table.book-pos tbody tr')].map(r => ({
        name: r.dataset.pos, size: parseFloat(r.querySelectorAll('td')[4].innerText.replace(/,/g, ''))}))""")


def test_b28_two_strategies_on_one_asset_each_tab_shows_its_own_rows(shared_site, browser):  # noqa: F811
    base, _ = shared_site
    page, errors = _open(browser, f"{base}/")
    book = _rows(page)
    assert sorted(r["name"] for r in book) == ["one", "two"]
    tabs = []
    for name in ("one", "two"):
        page.goto(f"{base}/sleeves/{name}")
        rows = _rows(page)
        assert [r["name"] for r in rows] == [name], rows
        tabs += rows
    assert sum(r["size"] for r in tabs) == pytest.approx(sum(r["size"] for r in book)) == pytest.approx(0.15)
    assert errors == []
    page.context.close()
