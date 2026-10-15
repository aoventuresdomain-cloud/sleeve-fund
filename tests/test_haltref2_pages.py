"""F-HALTREF-2 on the dashboard (FE): a Strategy the portfolio halt flattened reads so on every page, with one full
stop, and its activity row is titled for the halt, not the PM (the event stays pm_flatten: only the words change)."""

import re
from html import unescape

import pytest
from test_dashboard import (
    _strategy_accounts,  # noqa: F401 - in the book, as on the paper account (F229-6)
)
from test_f_haltref_2 import HALT_FLATTEN, HELD, WHY, T, _runtime, store  # noqa: F401

PAGES = ("/", "/sleeves/s1", "/orders", "/alerts", "/records", "/risk", "/trades")


def _visible(html: str) -> str:
    html = re.sub(r"<script\b.*?</script>", "", html, flags=re.DOTALL)
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", html)).split())


@pytest.fixture
def pages(store, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(dash, "TEARSHEETS", tmp_path)

    def flatten(reason, actor):
        rt = _runtime(store, [T])
        store.command("s1", "flatten", reason, actor=actor)
        assert rt.tick(**HELD) == "flatten"
        client = TestClient(dash.create_app(store))
        return {p: _visible(client.get(p, auth=("pm", "pw")).text) for p in PAGES}

    return flatten


def test_a_halt_flatten_reads_as_the_portfolio_halt_on_every_page(pages):
    out = pages(f"{HALT_FLATTEN}{WHY}", "supervisor")
    for path, words in out.items():
        assert "flattened by PM" not in words and "Flattened by PM" not in words and "Pm flatten" not in words, path
    assert f"Paused: flattened by the portfolio halt: {WHY}. " in out["/sleeves/s1"]  # the banner: one full stop
    # Main's Action needed list ends its lines without a full stop: the halt's words, never two stops.
    assert f"s1 is paused: flattened by the portfolio halt: {WHY}" in out["/risk"] and f"{WHY}.." not in out["/risk"]
    assert f"Flattened by the portfolio halt {HALT_FLATTEN}{WHY}" in out["/sleeves/s1"]  # the activity row


def test_a_pm_flatten_still_reads_as_the_pms(pages):
    out = pages("Book kill switch: drawdown", "PM")
    assert "Paused: flattened by PM: Book kill switch: drawdown. " in out["/sleeves/s1"]
    assert "Flattened by PM Book kill switch: drawdown" in out["/sleeves/s1"] and "portfolio halt" not in out["/sleeves/s1"]
