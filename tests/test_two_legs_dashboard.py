"""P2-8 step 7 (spec pe2-plans/p2-8-step7-dashboard.md): how the dashboard shows a two-leg Strategy. Display only.

One row under its first leg's name, its figures the two legs summed, the legs nested under it; the second leg's page
says which Strategy it belongs to and takes no command; the Strategy's page shows the pair's state; a red flag while one
of the pair's alerts waits to be acknowledged; "Strategy" and "leg", never the venue clone."""

from __future__ import annotations

import re
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.store import Store, utcnow

AUTH = ("pm", "test-pw")
# Spot long and perp short on one instrument: a carry. Each leg's share of the 10,000 is 5,000 (1x perp).
FIRST = {"instrument": "BTC/USDT", "params": {}, "venue": "BINANCE", "side": 1}
SECOND = {"instrument": "BTC/USDT", "params": {"market": "perp", "allow_short": True}, "venue": "BINANCE", "side": -1}


@pytest.fixture
def carry(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_two_legs(name="carry", strategy="trend_filter", bar_spec="1-HOUR-LAST-INTERNAL",
                          starting_balance=10_000, first=FIRST, second=SECOND)
    store.create_sleeve(name="solo", strategy="trend_filter", instrument="ETH/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=10_000, params={}, venue="BINANCE")
    # Leg 1 long 0.05 at 60,000, now 61,000 (+50); leg 2 short 0.05 at 60,100, now 61,000 (-45), fee 2 each.
    for name, side, px, cash, qty, equity in (("carry", "BUY", 60_000, 1_998.0, 0.05, 5_048.0),
                                              ("carry~2", "SELL", 60_100, 4_998.0, -0.05, 4_953.0)):
        store.record_fill(name, side=side, qty=0.05, price=px, fee=2.0, order_id=f"{name}-o1", trade_id=f"{name}-t1")
        store.record_equity(name, equity=equity, cash=cash, qty=qty, price=61_000, benchmark=5_000)
    return TestClient(app_mod.create_app(store)), store


def _get(client, path):
    r = client.get(path, auth=AUTH)
    assert r.status_code == 200, path
    return r.text


def _visible(html: str) -> str:
    """The page's words: scripts and attributes (links, ids, form values) dropped, so a row's name in a URL doesn't
    count."""
    html = re.sub(r"<script\b.*?</script>", "", html, flags=re.DOTALL)
    return re.sub(r'\s[\w:-]+="[^"]*"', "", html)


def _strategies_tab(html: str) -> str:
    return html.split('id="tab-strategies"')[1].split("</table>")[0]


def test_lists_show_a_two_leg_strategy_once_with_its_legs_summed_and_nested(carry):
    client, _ = carry
    html = _get(client, "/")
    tab = _strategies_tab(html)
    rows = tab.split("<tbody>")[1].split("<tr")[1:]
    assert len(rows) == 2 and "carry" in rows[0] and "solo" in rows[1]  # one row for the pair, none for its leg
    row = rows[0]
    assert ">10,001.00<" in row  # equity: 5,048 + 4,953
    assert "Two legs, flat" in row
    assert "Leg 1</a> · BTC/USDT spot" in row and "Leg 2</a> · BTC/USDT perp" in row
    assert re.search(r'Leg 2</a> · BTC/USDT perp · <span class="side sell">Short</span> 0\.05', row)
    assert "fees 2.00" in row and "funding" in row.split("Leg 2")[1]
    assert '<a href="/sleeves/carry~2">Leg 2</a>' in row  # the leg's own page, one click away
    # The header and the rail count Strategies, not rows.
    assert "Paper · 2 strategies, " in html
    # Positions: both legs under the Strategy's name, one Close for the pair, which closes both.
    pos = html.split('id="tab-positions"')[1].split("</table>")[0]
    assert pos.count('<a href="/sleeves/carry">carry</a><span class="sub2">leg ') == 2
    assert pos.index("leg 1") < pos.index("leg 2")
    assert pos.count(">Close</button>") == 1 and 'title="Closes both legs">Close' in pos
    assert "Close carry's legs" in html and "Close both legs and pause" in html
    assert 'id="dlg-close-carry~2"' not in html


def test_the_shell_counts_and_lists_the_pair_as_one_strategy(carry):
    client, _ = carry
    html = _get(client, "/alerts")
    assert html.count('href="/sleeves/carry~2"') == 0  # not in the rail
    assert html.count('href="/sleeves/carry"') >= 1


def test_the_second_legs_page_names_its_strategy_and_has_no_buttons(carry):
    client, _ = carry
    html = _get(client, "/sleeves/carry~2")
    assert '<h1>carry, leg 2</h1>' in html
    assert 'Leg 2 of the two-leg Strategy <a href="/sleeves/carry">carry</a>' in html
    head = html.split('class="page-head"')[1].split("</div>")[0]
    assert "<button" not in head and 'class="actions"' not in head
    pos = html.split('id="tab-positions"')[1] if 'id="tab-positions"' in html else html
    assert ">Close</button>" not in pos.split("</table>")[0]


def test_the_strategys_page_shows_the_pair_state_and_while_opening_its_trade_and_deadline_in_uk_time(carry):
    client, store = carry
    html = _get(client, "/sleeves/carry")
    assert "Two legs · Flat" in html and "done or unwound by" not in html
    assert "Both legs: equity 10,001.00" in html and "The figures below are leg 1's." in html
    deadline = utcnow().replace(hour=7, minute=30, second=15, microsecond=0) + timedelta(days=1)
    assert store.set_pair_state("carry", expected=None, state="flat")
    assert store.set_pair_state("carry", expected="flat", state="legging", trade_id="T1", deadline=deadline)
    html = _get(client, "/sleeves/carry")
    assert "Two legs · Opening" in html
    uk = deadline.astimezone(ZoneInfo("Europe/London")).strftime("%H:%M:%S UK")
    assert f"Opening trade T1: done or unwound by {uk}" in html
    assert store.set_pair_state("carry", expected="legging", state="hedged", trade_id="T1", expected_trade_id="T1")
    html = _get(client, "/sleeves/carry")
    assert "Two legs · Hedged" in html and "done or unwound by" not in html
    assert "Two legs, hedged" in _strategies_tab(_get(client, "/"))


@pytest.mark.parametrize("kind", ["two_leg_no_pair", "two_leg_broken", "two_leg_liq_distance", "pair_state_unsaved"])
def test_a_pair_alert_flags_the_strategy_in_red_until_acknowledged(carry, kind):
    client, store = carry
    assert "Legs need you" not in _get(client, "/")
    store.event("carry~2", "warning", kind, "Two-leg trade T1: something to look at")
    home, page = _get(client, "/"), _get(client, "/sleeves/carry")
    row = _strategies_tab(home).split("<tbody>")[1].split("<tr")[1]
    assert '<a class="chip halted" href="/alerts"' in row and "Legs need you" in row
    assert "Legs need you" in page.split('class="page-head"')[1].split("</div>")[0]
    for a in store.alerts():
        store.ack(a["id"], "pm")
    assert "Legs need you" not in _get(client, "/") and "Legs need you" not in _get(client, "/sleeves/carry")


def test_an_info_level_pair_event_or_another_strategys_alert_raises_no_flag(carry):
    client, store = carry
    store.event("carry", "info", "two_leg_broken", "an info note")
    store.event("solo", "warning", "two_leg_broken", "not a pair")
    row = _strategies_tab(_get(client, "/")).split("<tbody>")[1].split("<tr")[1]
    assert "Legs need you" not in row


def test_no_page_shows_the_venue_clone_or_a_legs_row_name(carry):
    client, store = carry
    store.event("carry~2", "warning", "two_leg_broken",
                "Two-leg trade T1: BTCUSDT-PERP.BINANCE_P2 couldn't fill on BINANCE_P2, so leg 1 was unwound")
    for path in ("/", "/risk", "/trades", "/orders", "/alerts", "/sleeves/carry", "/sleeves/carry~2"):
        words = _visible(_get(client, path))
        assert "_P2" not in words, path
        assert "carry~2" not in words, path


def test_book_figures_still_count_both_legs(carry):
    """Folding is display only: the book's value, P&L and positions keep each leg's own book."""
    client, _ = carry
    html = _get(client, "/")
    assert "Book value</div><div class=\"v\">20,001.00" in html  # carry 10,001 + solo's 10,000 start


def test_a_second_leg_whose_first_is_put_away_still_shows():
    """Nothing held is hidden: a leg row with no first-leg row in the list stays a row of its own."""
    from types import SimpleNamespace as NS

    from sleeve_fund.dashboard import legs

    leg = {"sleeve": NS(name="x~2", parent="x")}
    assert legs.fold(None, [leg]) == [leg]
    assert legs.label("x~2") == "x, leg 2" and legs.label("x") == "x"


@pytest.mark.parametrize(("text", "reads"), [
    ("BTCUSDT-PERP.BINANCE_P2 filled", "BTCUSDT-PERP filled"),  # an instrument id: the clone's suffix goes
    ("ethusdt.binance_p2 stale", "ethusdt stale"),
    ("an order on BINANCE_P2 was refused", "an order on the perpetual venue was refused"),  # the clone alone
    ("BTCUSDT-PERP.BINANCE", "BTCUSDT-PERP"),  # the venue's own suffix, as before
])
def test_the_venue_clone_reads_as_its_venue(text, reads):
    from sleeve_fund.wording import no_venues

    assert no_venues(text) == reads


def test_the_outside_alert_chat_never_names_the_venue_clone():
    from sleeve_fund.alerts import message

    out = message([{"level": "warning", "sleeve": "carry", "kind": "two_leg_broken",
                    "message": "Two-leg trade T1: BTCUSDT-PERP.BINANCE_P2 couldn't fill on BINANCE_P2"}])
    assert "_P2" not in out and "BINANCE" not in out.upper()
    assert "BTCUSDT-PERP couldn't fill on the perpetual venue" in out
