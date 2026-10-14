"""F-T-2 (Advisor Addition A): a Strategy whose fill handler failed is held from new entries until the PM resumes it.
Its page shows a "Fill hold" badge beside the status, and its Resume says it clears the hold and nothing else."""

from __future__ import annotations

from html import unescape

from test_dashboard import AUTH, _new, _strategy_accounts, client  # noqa: F401 - the fixtures


def _resume_dialog(page: str) -> str:
    return unescape(page.split('id="dlg-resume"')[1].split("</dialog>")[0])


def test_a_held_strategy_shows_the_badge_and_a_resume_that_clears_the_hold_only(client):  # noqa: F811
    c, store = client
    _new(c)
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "fill-hold" not in page
    store.event("btc-test", "error", "fill_hold", "new entries held: a fill handler failed")
    store.set_status("btc-test", "paused", "fill handler failed")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "fill-hold" in page
    dlg = _resume_dialog(page)
    assert "Resuming clears the fill hold only" in dlg and "Clear the fill hold" in dlg
    store.event("btc-test", "info", "fill_hold_cleared", "fill-handler hold cleared by the PM: checked")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "fill-hold" not in page and "Resuming clears the fill hold only" not in _resume_dialog(page)
