"""B-17 (PM 17:10 UK, via CoS): "Acknowledge all" on Alerts. One button acks every open alert in one action through
the existing ack path; a confirm step names how many; each cleared alert keeps its row with who and when and the note
"cleared with Acknowledge all", and still shows under "All, including acknowledged"; an alert raised after the confirm
step was shown is not cleared (its ids are fixed there). On Postgres too (TEST_DATABASE_URL)."""

import re
from html import unescape

import pytest
from fastapi.testclient import TestClient
from test_browser import browser  # noqa: F401 - the browser fixture
from test_fe_b20_browser import held_site  # noqa: F401 - a fund-held store, served
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

AUTH = ("pm", "pw")
SAME = {"Origin": "http://testserver"}
NOTE = "cleared with Acknowledge all"


@pytest.fixture
def c(store, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    return TestClient(app_mod.create_app(store))


def _text(html: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", html)).split())


def _raise(store, n, start=0):  # noqa: F811
    store.create_sleeve(name=f"s{start}", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, venue="BINANCE", desired_state="stopped")
    for i in range(n):
        store.event(f"s{start}", "warning" if i % 2 else "error", "x", f"alert {start + i}")


def test_b17_the_button_shows_only_with_open_alerts(c, store):  # noqa: F811
    assert "Acknowledge all" not in _text(c.get("/alerts", auth=AUTH).text)
    _raise(store, 3)
    page = c.get("/alerts", auth=AUTH).text
    assert 'href="/alerts/ack-all"' in page and "Acknowledge all" in _text(page)


def test_b17_the_confirm_step_names_the_count_and_fixes_the_ids(c, store):  # noqa: F811
    _raise(store, 3)
    r = c.get("/alerts/ack-all", auth=AUTH)
    assert r.status_code == 200
    assert "Acknowledge all 3 open alerts?" in _text(r.text)
    assert len(re.findall(r'<input type="hidden" name="ids" value="\d+">', r.text)) == 3
    assert store.open_alert_count() == 3  # showing it writes nothing


def test_b17_confirm_acks_those_ids_only_and_keeps_every_row(c, store):  # noqa: F811
    _raise(store, 3)
    ids = re.findall(r'name="ids" value="(\d+)"', c.get("/alerts/ack-all", auth=AUTH).text)
    _raise(store, 1, start=10)  # raised after the confirm step was shown
    r = c.post("/alerts/ack-all", data={"ids": ids}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/alerts?cleared=3"
    left = store.alerts(include_acked=False)
    assert [a["message"] for a in left] == ["alert 10"]
    acked = [a for a in store.alerts(include_acked=True) if a["acked_at"]]
    assert len(acked) == 3 and all(a["acked_by"] == "PM" and a["ack_note"] == NOTE for a in acked)
    page = _text(c.get("/alerts?cleared=3", auth=AUTH).text)
    assert "Acknowledged 3 alerts." in page
    every = _text(c.get("/alerts?show=all", auth=AUTH).text)
    assert every.count(NOTE) == 3 and "alert 10" in every


def test_b17_needs_the_login_and_the_same_origin_and_real_alert_ids(c, store):  # noqa: F811
    _raise(store, 2)
    ids = re.findall(r'name="ids" value="(\d+)"', c.get("/alerts/ack-all", auth=AUTH).text)
    assert c.post("/alerts/ack-all", data={"ids": ids}, auth=("pm", "wrong"), headers=SAME,
                  follow_redirects=False).status_code == 401
    assert c.post("/alerts/ack-all", data={"ids": ids}, auth=AUTH, headers={"Origin": "http://evil.example"},
                  follow_redirects=False).status_code in (400, 403)
    assert c.post("/alerts/ack-all", data={"ids": ["abc"]}, auth=AUTH, headers=SAME,
                  follow_redirects=False).status_code == 400
    assert store.open_alert_count() == 2


def test_b17_in_the_browser_acknowledge_all_then_confirm_empties_the_open_list(held_site, browser):  # noqa: F811
    from test_browser import _open

    base, store = held_site  # noqa: F811
    for i in range(3):
        store.event("held-long", "warning", "x", f"browser alert {i}")
    page, errors = _open(browser, f"{base}/alerts")
    with page.expect_navigation():
        page.click("a:has-text('Acknowledge all')")
    assert "Acknowledge all 3 open alerts?" in page.inner_text("main")
    with page.expect_navigation():
        page.click("button:has-text('Acknowledge all 3')")
    main = page.inner_text("main")
    assert "Acknowledged 3 alerts." in main and "No open alerts" in main
    assert store.open_alert_count() == 0
    assert errors == []
    page.context.close()
