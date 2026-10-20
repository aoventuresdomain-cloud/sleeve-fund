"""B-31 locked half, the pages (Advisor B31-BRAKE-SWITCH, with the Advisor's 17:00 UK correction: only the order rules
have no off switch). QD's 75f96a6 lets T3 and the T5 brakes switch off for paper testing and re-bases a brake switched
back on. Here, what the PM reads: the review step says what each stops doing (a pause or halt in force stays), says the
re-based level and the true drawdown before a brake goes back on, and refuses a switch-off while the exchange order
client is on; a brake off shows a banner that stays, with since when and why, on every fund page; switching one back
on offers its saved limit, never a default; the order rules say they have no off switch; no page says "locked"; each
switch shows in the version history and as an alert. On Postgres too (TEST_DATABASE_URL)."""

from __future__ import annotations

import re
from datetime import UTC
from html import unescape
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from test_dashboard import AUTH, SAME
from test_fe_b20_fund_held_wording import _mark
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)
from test_portfolio_settings import _form, _n, _text

from sleeve_fund import compliance_rules as rules

UK = ZoneInfo("Europe/London")
BRAKES = ("fund_stop_loss", "fund_day_loss", "fund_drawdown")
PAGES = ("/", "/risk", "/trades", "/portfolio-settings")


@pytest.fixture
def client(store, tmp_path, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", AUTH[1])
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    return TestClient(app_mod.create_app(store)), store


def _apply(c, version, reason="testing the fund", **over):
    r = c.post("/portfolio-settings/apply", data={**_form(expected_version=str(version), **over), "confirm_off": "on",
                                                   "reason_pick": "Other", "reason_note": reason},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303, r.text
    return r


def _review(c, version, **over):
    return c.post("/portfolio-settings/review", data=_form(expected_version=str(version), **over), auth=AUTH,
                  headers=SAME)


def _off(key):
    return {f"{key}.enabled": None}


@pytest.mark.parametrize("key", BRAKES)
def test_the_review_says_what_switching_a_brake_off_stops(client, key):
    c, _ = client
    page = _text(_review(c, 1, **_off(key)).text)
    words = {"fund_stop_loss": "no longer trimmed to a loss for the whole fund if every stop is hit",
             "fund_day_loss": "today's loss no longer pauses new trades. A pause already in force stays until the "
                              "next day begins",
             "fund_drawdown": "a fall from the peak no longer closes everything and halts. A halt already in force "
                              "stays until you restart the fund"}[key]
    assert f"{rules.name(key)} off: " in page and words in page, page[:2000]
    assert "for paper testing only" in page


def test_a_pause_in_force_stays_when_its_brake_is_switched_off(client):
    from sleeve_fund.paper.portfolio_gate import PaperGate
    from sleeve_fund.portfolio.gate import block_reason

    c, store = client  # noqa: F811
    _mark(store, 94_000, hwm=100_000)  # 6% down today: past the 3% pause
    ledger = PaperGate(store).ledger
    assert ledger.state().paused_until is not None
    _apply(c, 1, **_off("fund_day_loss"))
    st = ledger.state()
    assert st.paused_until is not None and block_reason(st, st.mark_ts) is not None
    assert 'data-fund-state="brake"' in c.get("/", auth=AUTH).text


def test_the_review_shows_the_rebased_level_and_the_true_drawdown_before_a_brake_goes_back_on(client):
    c, store = client  # noqa: F811
    _apply(c, 1, **_off("fund_drawdown"), **_off("fund_day_loss"))
    _mark(store, 94_000, hwm=100_000)
    page = _text(_review(c, 2).text)
    assert ("T5 Drawdown halt back on: it will measure from 94,000.00, the book now. The book is 6.0% below its peak "
            "of 100,000.00, which is kept.") in page, page[:3000]
    assert "T5 Daily loss pause back on: today's loss will count from 94,000.00. The day is 6.0% down now." in page


def test_the_review_says_so_when_the_book_has_no_mark_to_rebase_from(client):
    c, _ = client
    _apply(c, 1, **_off("fund_drawdown"))
    page = _text(_review(c, 2).text)
    assert "T5 Drawdown halt back on: the fund's book has no mark yet, so it measures from the first mark." in page


def test_switching_a_brake_off_is_refused_on_the_review_while_the_order_client_is_on(client, monkeypatch):
    c, store = client  # noqa: F811
    for k, v in {"EXEC_MODE": "demo", "EXEC_HOST": "api-demo.bybit.com", "BYBIT_DEMO_API_KEY": "k",
                 "BYBIT_DEMO_API_SECRET": "s"}.items():
        monkeypatch.setenv(k, v)
    r = _review(c, 1, **_off("fund_drawdown"))
    page = _text(r.text)
    assert "Not saved." in page and "can't be switched off while the exchange order client is on" in page
    assert "Apply these changes" not in page and _n(store) == 1


def _since(store, version):  # noqa: F811
    at = next(v["created_at"] for v in store.compliance_versions() if v["version"] == version)
    at = at if at.tzinfo else at.replace(tzinfo=UTC)
    return f"{at.astimezone(UK):%d %b %H:%M} UK"


@pytest.mark.parametrize("path", PAGES)
def test_a_brake_off_shows_a_banner_that_stays_with_since_when_and_why(client, path):
    c, store = client  # noqa: F811
    _apply(c, 1, reason="testing the fund", **_off("fund_drawdown"))
    _apply(c, 2, reason="T1 tighter", **_off("fund_drawdown"), **{"asset_stop_loss.limit": "3"})  # keeps its off time
    page = unescape(c.get(path, auth=AUTH).text)
    m = re.search(r'<div class="banner warn brake-off"[^>]*data-brake-off="fund_drawdown"[^>]*>(.*?)</div>', page,
                  re.DOTALL)
    assert m, path
    words = " ".join(re.sub(r"<[^>]+>", "", m.group(1)).split())
    assert words == f"T5 Drawdown halt OFF since {_since(store, 2)}, reason: Other: testing the fund.", words
    assert "data-once" not in m.group(0)
    assert page.count('class="banner warn brake-off"') == 1


def test_no_brake_banner_while_every_brake_is_on(client):
    c, _ = client
    _apply(c, 1, **_off("fund_gross"))  # T4 off is not a brake
    for path in PAGES:
        assert "brake-off" not in c.get(path, auth=AUTH).text, path


def test_switching_a_brake_back_on_offers_its_saved_limit_never_the_default(client):
    c, store = client  # noqa: F811
    _apply(c, 1, **_off("fund_drawdown"), **{"fund_drawdown.limit": "12"})
    edit = c.get("/portfolio-settings?edit=1", auth=AUTH).text
    assert re.search(r'<input name="fund_drawdown\.limit" value="12"', edit), edit[:200]
    assert not re.search(r'name="fund_drawdown\.enabled" checked', edit)
    _apply(c, 2, **{"fund_drawdown.limit": "12"})
    [t] = [t for t in store.compliance().tests if t["test"] == "fund_drawdown"]
    assert t["enabled"] and float(t["limit"]) == pytest.approx(0.12)


def test_the_order_rules_have_no_off_switch_and_no_page_says_locked(client):
    c, _ = client
    for path in ("/portfolio-settings", "/portfolio-settings?edit=1"):
        html = c.get(path, auth=AUTH).text
        t = _text(html)
        assert ("The order rules apply to every trade and have no off switch: a stop, or 1x leverage without one; a "
                "stop within half the distance to liquidation; and the leverage cap.") in t, path
        assert "Locked on" not in t and "locked" not in html.split("<main", 1)[-1].lower().replace(
            "live trading locked", ""), path
    edit = c.get("/portfolio-settings?edit=1", auth=AUTH).text
    for key in rules.KEYS:  # every compliance test, T1 to T5, has its switch
        assert re.search(rf'<input type="checkbox" class="sw" name="{key}\.enabled"', edit), key


def test_each_brake_switch_shows_in_the_history_and_as_an_alert(client):
    c, store = client  # noqa: F811
    _apply(c, 1, reason="testing the fund", **_off("fund_day_loss"))
    _apply(c, 2, reason="testing done")
    hist = _text(c.get("/portfolio-settings", auth=AUTH).text)
    assert "T5 Daily loss pause: on → off" in hist and "T5 Daily loss pause: off → on" in hist
    alerts = [a for a in store.alerts(include_acked=True) if a["kind"] == "portfolio_brake_switch"]
    assert [a["level"] for a in alerts] == ["warning", "warning"]
    assert alerts[1]["message"] == "T5 Daily loss pause switched off (compliance tests v2): Other: testing the fund"
    assert alerts[0]["message"] == "T5 Daily loss pause switched back on (compliance tests v3): Other: testing done"
    page = _text(c.get("/alerts", auth=AUTH).text)
    assert "T5 Daily loss pause switched off" in page and "T5 Daily loss pause switched back on" in page


def test_a_non_brake_switch_raises_no_brake_alert(client):
    c, store = client  # noqa: F811
    _apply(c, 1, **_off("fund_gross"))
    assert not [a for a in store.alerts(include_acked=True) if a["kind"] == "portfolio_brake_switch"]


@pytest.mark.parametrize("key", BRAKES)
def test_the_server_refuses_a_brake_switch_off_without_the_confirm_tick(client, key):
    """TEC-B31-TICK now covers T3 and the T5 brakes too (HoE, TEC-B31-MERGE), and a reason stays required."""
    c, store = client  # noqa: F811
    data = {**_form(**_off(key)), "reason_pick": "Other", "reason_note": "testing the fund"}
    r = c.post("/portfolio-settings/apply", data=data, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 400 and rules.name(key) in r.text and "confirm tick" in r.text, r.text
    r = c.post("/portfolio-settings/apply", data={**_form(**_off(key)), "confirm_off": "on"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert r.status_code == 400 and "a reason is required" in r.text, r.text
    assert _n(store) == 1


@pytest.mark.parametrize("key", ["order_rules", "stop_rule", "leverage_cap"])
def test_an_order_rule_switched_off_through_the_server_is_refused_and_changes_nothing(client, key):
    """Advisor cell f (HoE ruling): the order rules have no off switch. Made-up keys posted to the switch endpoint, asked
    off, are refused in words, with the stored settings and the journal unchanged."""
    c, store = client  # noqa: F811
    tests, journal = store.compliance().tests, store.decisions(limit=500)
    data = {**_form(), f"{key}.enabled": "", f"{key}.limit": "0", "confirm_off": "on", "reason_pick": "Other",
            "reason_note": "testing the fund"}
    for path in ("/portfolio-settings/review", "/portfolio-settings/apply"):
        r = c.post(path, data=data, auth=AUTH, headers=SAME, follow_redirects=False)
        assert r.status_code in (200, 400), (path, r.status_code)
        words = _text(r.text)
        assert f"{key} isn't a compliance test, and the order rules have no off switch" in words, (path, words[:400])
        assert "Apply these changes" not in words
    assert _n(store) == 1 and store.compliance().tests == tests and store.decisions(limit=500) == journal
