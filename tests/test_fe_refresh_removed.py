"""REFRESH-RESET-REMOVE (HoE, Drop 3a, 15 Oct): the PM's clean slate is the new reset (reset_book: put every Strategy
away, then the 100,000 deposit) and nothing else. The old refresh, Setup's "Reset all" and its confirm, is gone from the
page and as routes. A Strategy's own reset route stays for today with nothing on any page reaching it; it goes in the
next drop (REFRESH-REMOVE-2, HoE)."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_halt_ref import _make, store  # noqa: F401

AUTH, SAME = ("pm", "pw"), {"Origin": "http://testserver"}


@pytest.fixture
def client(store, monkeypatch):
    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    _make(store, "a", 10_000)
    return TestClient(dash.create_app(store))


def test_setup_has_no_refresh_control(client):
    page = client.get("/setup", auth=AUTH).text
    assert "dlg-reset-all" not in page and 'action="/book/reset' not in page
    assert "Start the paper book afresh" not in page and "Reset all" not in page


@pytest.mark.parametrize("path", ["/book/reset", "/book/reset/confirm"])
def test_the_old_reset_routes_are_gone_and_reset_nothing(client, store, path):
    r = client.post(path, data={"reason": "old path", "liquidated": []}, auth=AUTH, headers=SAME,
                    follow_redirects=False)
    assert r.status_code in (404, 405), (path, r.status_code)
    assert store.pending_reset("a") is None


def test_no_page_links_or_posts_to_a_removed_or_retiring_reset_route():
    """HoE: no template reaches the two book routes or a Strategy's own reset (kept as a route for today only)."""
    from sleeve_fund.dashboard import app as dash

    hits = [(t.name, m.group(0)) for t in sorted((Path(dash.__file__).parent / "templates").glob("*.html"))
            for m in re.finditer(r"/book/reset[\w/]*|/sleeves/[^\s\"'<>]*/reset\b", t.read_text())]
    assert hits == []
