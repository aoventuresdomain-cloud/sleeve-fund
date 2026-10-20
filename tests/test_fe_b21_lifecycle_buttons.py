"""B-21 (board, from Setup Desk via CoS): Wind down and Close now on the Strategy page. The engine and
POST /sleeves/{name}/lifecycle exist (P2-6, sleeve_fund.lifecycle); the page had no button. Each opens a dialog that
says what happens, takes a reason as every other command does, and posts to the existing route: Close now on a
Strategy holding a position winds it down due now (flattened at the next candle close through the exit path, retired
once flat), and on a flat one retires it. On Postgres too (TEST_DATABASE_URL)."""

import re

from test_browser import browser  # noqa: F401 - the browser fixture
from test_fe_b20_browser import held_site  # noqa: F401 - a fund-held store, served
from test_fe_b20_fund_held_wording import (  # noqa: F401 - the fixtures
    AUTH,
    _hold,
    _long,
    _words,
    c,
)
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

SAME = {"Origin": "http://testserver"}


def _flat(store, name="flat-one"):  # noqa: F811
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0, params={"rise": 0.01, "dip": 0.005})
    return store.sleeve(name)


def _dialog(html: str, cmd: str) -> str:
    m = re.search(rf'<dialog id="dlg-{cmd}".*?</dialog>', html, re.DOTALL)
    assert m, f"no {cmd} dialog"
    return m.group(0)


def _post(c, name, action, pick):  # noqa: F811
    return c.post(f"/sleeves/{name}/lifecycle", data={"action": action, "reason_pick": pick}, auth=AUTH, headers=SAME,
                  follow_redirects=False)


def test_b21_an_active_strategy_offers_both_with_a_dialog_each_posting_to_the_lifecycle_route(c, store):  # noqa: F811
    _hold(store)
    s = _long(store)
    _, html = _words(c, f"/sleeves/{s.name}")
    assert 'data-open="dlg-wind_down"' in html and 'data-open="dlg-close_now"' in html
    for cmd, button in (("wind_down", "Wind down"), ("close_now", "Close now")):
        d = _dialog(html, cmd)
        assert f'action="/sleeves/{s.name}/lifecycle"' in d and f'name="action" value="{cmd}"' in d
        assert 'name="reason_pick"' in d and "data-needs-reason" in d and button in d
    wind = " ".join(re.sub(r"<[^>]+>", " ", _dialog(html, "wind_down")).split())
    assert "Opens nothing new from now" in wind and "its exits, trims and stops keep running" in wind
    close = " ".join(re.sub(r"<[^>]+>", " ", _dialog(html, "close_now")).split())
    assert "sells 0.1 BTC at market at the next candle close" in close and "retires once flat" in close


def test_b21_wind_down_saves_the_reason_and_the_page_shows_it_winding_down(c, store):  # noqa: F811
    _hold(store)
    s = _long(store)
    r = _post(c, s.name, "wind_down", "Rotating capital")
    assert r.status_code == 303, r.text
    assert store.sleeve(s.name).lifecycle == "winding_down"
    assert any(d["reason"].startswith("Rotating capital") for d in store.decisions(s.name))
    words, html = _words(c, f"/sleeves/{s.name}")
    assert "Winding down" in words and "deadline" in words.lower()
    assert 'data-open="dlg-wind_down"' not in html and 'data-open="dlg-close_now"' in html


def test_b21_close_now_winds_a_holder_down_due_now_and_retires_a_flat_one(c, store):  # noqa: F811
    _hold(store)
    s = _long(store)
    assert _post(c, s.name, "close_now", "Kill rule hit").status_code == 303
    held = store.sleeve(s.name)
    assert held.lifecycle == "winding_down" and held.wind_down_deadline <= held.wind_down_started
    f = _flat(store)
    assert _post(c, f.name, "close_now", "Kill rule hit").status_code == 303
    assert store.sleeve(f.name).lifecycle == "retired" and f.name in store.archived()
    _, html = _words(c, f"/sleeves/{f.name}")
    assert 'data-open="dlg-close_now"' not in html and 'data-open="dlg-wind_down"' not in html


def test_b21_no_reason_no_move(c, store):  # noqa: F811
    _hold(store)
    s = _long(store)
    r = c.post(f"/sleeves/{s.name}/lifecycle", data={"action": "wind_down"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 400 and store.sleeve(s.name).lifecycle == "active"


def test_b21_in_the_browser_close_now_waits_for_a_reason_then_winds_it_down(held_site, browser):  # noqa: F811
    from test_browser import _open

    base, store = held_site  # noqa: F811
    page, errors = _open(browser, f"{base}/sleeves/held-long")
    page.click("button[data-open=dlg-close_now]")
    confirm = page.locator("#dlg-close_now button[name=action][value=close_now]")
    assert confirm.is_disabled()
    page.locator("#dlg-close_now label.opt:has-text('Kill rule hit')").click()
    assert confirm.is_enabled()
    with page.expect_navigation():
        confirm.click()
    assert store.sleeve("held-long").lifecycle == "winding_down"
    assert "Winding down" in page.inner_text("main")
    assert errors == []
    page.context.close()
