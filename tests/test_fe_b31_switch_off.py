"""B-31, the unlocked half (HoE v3 20 Oct; QD the engine side, B-32): the PM switches T1, T2 or a T4 row off on
Portfolio Settings for testing. The review step says, for each row switched off, what new trades are no longer checked
against, that the order rules still apply (a stop or 1x, a stop within half the distance to liquidation, the leverage
cap), and asks for a confirm tick before Apply. An off row keeps its saved limit and shows it, so switching it back on
restores that limit."""

from __future__ import annotations

import re
from decimal import Decimal as D
from html import unescape

from test_browser import browser  # noqa: F401 - the browser fixture
from test_dashboard import (  # noqa: F401 - the fixtures
    AUTH,
    SAME,
    _strategy_accounts,
    client,
)
from test_fe_b20_browser import held_site  # noqa: F401 - a fund-held store, served
from test_portfolio_settings import _form, ps  # noqa: F401 - the fixture

OFF = {  # each unlocked row and what the PM is told it stops checking
    "asset_stop_loss": "T1 Per asset off: new trades are no longer trimmed to a loss per asset if every stop is hit.",
    "type_stop_loss": "T2 Per asset group off: new trades are no longer trimmed to a loss per asset group if every stop "
                      "is hit, and an asset with no group can take a trade.",
    "fund_gross": "T4 Gross off: new trades are no longer trimmed to a gross size for the fund.",
    "asset_net": "T4 Net per asset off: new trades are no longer trimmed to a net size on one asset.",
    "fund_margin": "T4 Margin off: new trades are no longer trimmed to the margin the fund may post.",
}
ORDER_RULES = ("The order rules still apply to every trade: a stop, or 1x leverage without one; a stop within half "
               "the distance to liquidation; and the leverage cap.")


def _text(html: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", html)).split())


def _review(c, **over):
    r = c.post("/portfolio-settings/review", data=_form(**over), auth=AUTH, headers=SAME)
    assert r.status_code == 200
    return r.text


def test_b31_each_row_switched_off_is_named_with_what_stops_being_checked(ps):  # noqa: F811
    c, _ = ps
    for key, words in OFF.items():
        html = _review(c, **{f"{key}.enabled": None})
        t = _text(html)
        assert words in t, (key, t[:600])
        assert ORDER_RULES in t
        assert re.search(r'<input type="checkbox" name="confirm_off" [^>]*required', html), key
        assert [w for k, w in OFF.items() if k != key and w in t] == []  # only the row switched off


def test_b31_all_unlocked_rows_off_together_and_nothing_off_needs_no_tick(ps):  # noqa: F811
    c, _ = ps
    t = _text(_review(c, **{f"{k}.enabled": None for k in OFF}))
    assert all(w in t for w in OFF.values())
    calm = _review(c, **{"asset_stop_loss.limit": "3"})
    assert 'name="confirm_off"' not in calm and ORDER_RULES not in _text(calm)


def test_b31_an_off_row_keeps_and_shows_its_limit_and_switching_back_on_restores_it(ps):  # noqa: F811
    c, store = ps
    carried = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">',
                              _review(c, **{"fund_gross.enabled": None, "fund_gross.limit": "2.5"})))
    r = c.post("/portfolio-settings/apply", data={**carried, "confirm_off": "on", "reason_pick": "Tighter while testing"},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303
    gross = next(t for t in store.compliance().tests if t["test"] == "fund_gross")
    assert not gross["enabled"] and gross["limit"] == D("2.5")
    view = c.get("/portfolio-settings", auth=AUTH).text
    row = _text(view.split('data-test="fund_gross"', 1)[1].split("</tr>", 1)[0])
    assert "Off" in row and "saved 2.50x" in row and "not checked" in row, row
    edit = c.get("/portfolio-settings?edit=1", auth=AUTH).text
    assert 'name="fund_gross.limit" value="2.5"' in edit
    back = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">',
                           _review(c, **{"expected_version": "2", "fund_gross.limit": "2.5"})))
    assert "T4 Gross: off → on" in _text(_review(c, **{"expected_version": "2", "fund_gross.limit": "2.5"}))
    r = c.post("/portfolio-settings/apply", data={**back, "reason_pick": "Tighter while testing"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert r.status_code == 303
    gross = next(t for t in store.compliance().tests if t["test"] == "fund_gross")
    assert gross["enabled"] and gross["limit"] == D("2.5")


def test_b31_in_the_browser_apply_waits_for_the_confirm_tick(held_site, browser):  # noqa: F811
    from test_browser import _open

    base, _ = held_site
    page, errors = _open(browser, f"{base}/portfolio-settings?edit=1")
    page.uncheck("input[name='fund_gross.enabled']")
    with page.expect_navigation():
        page.click("button:has-text('Review changes')")
    assert "T4 Gross off:" in page.inner_text("main")
    valid = "() => document.querySelector('form[action=\"/portfolio-settings/apply\"]').checkValidity()"
    page.evaluate("""() => { const f = document.querySelector('form[action="/portfolio-settings/apply"]');
        const r = f.querySelector('input[name=reason_pick]'); if (r) r.checked = true; }""")
    assert page.evaluate(valid) is False  # the reason alone doesn't apply a switch-off
    page.check("input[name=confirm_off]")
    assert page.evaluate(valid) is True
    assert errors == []
    page.context.close()
