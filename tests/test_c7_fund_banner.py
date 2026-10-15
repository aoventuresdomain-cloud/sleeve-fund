"""C7 (ui-v2/compliance-build-spec.md): one banner at the top of Portfolio, a Strategy page and Portfolio Settings,
only when it applies, from the engine's fund state (PE2): the daily brake, the drawdown halt, a failed book check and
a fund value out of date. Seeded here; the engine's own cells pin the state itself."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from test_dashboard import AUTH, client  # noqa: F401 - the fixture
from test_strategy_page import _sleeve

PAGES = ["/", "/sleeves/btc-x", "/portfolio-settings"]


def _seed(monkeypatch, state):
    from sleeve_fund.dashboard import compliance

    monkeypatch.setattr(compliance, "fund_state", lambda store: state)


def _banner(page: str) -> str:
    return page.split('class="banner')[1].split("</div>")[0] if "fund-banner" in page else ""


@pytest.mark.parametrize("path", PAGES)
def test_no_banner_while_the_fund_is_normal_or_its_state_is_not_there_yet(client, monkeypatch, path):  # noqa: F811
    c, store = client
    _sleeve(store)
    for state in (None, {"state": "normal", "since": None, "until": None, "reason": None}):
        _seed(monkeypatch, state)
        assert "fund-banner" not in c.get(path, auth=AUTH).text


@pytest.mark.parametrize("path", PAGES)
def test_the_daily_brake_says_until_when_in_uk_time(client, monkeypatch, path):  # noqa: F811
    c, store = client
    _sleeve(store)
    # 00:00 UTC on 16 Oct is 01:00 UK (summer time): the UK time comes from the date, never a fixed 01:00.
    _seed(monkeypatch, {"state": "brake", "since": None, "until": datetime(2026, 10, 16, tzinfo=UTC), "reason": None})
    b = _banner(c.get(path, auth=AUTH).text)
    assert 'data-fund-state="brake"' in b and "New trades paused until the next day begins (01:00 UK)." in b
    _seed(monkeypatch, {"state": "brake", "since": None, "until": datetime(2026, 11, 2, tzinfo=UTC), "reason": None})
    assert "(00:00 UK)" in _banner(c.get(path, auth=AUTH).text)  # winter: UK is UTC



def test_a_brake_end_the_engine_gives_without_a_zone_is_read_as_utc(client, monkeypatch):  # noqa: F811
    import time

    c, store = client
    _sleeve(store)
    monkeypatch.setenv("TZ", "America/New_York")  # the server's own zone must never move the UK time
    time.tzset()
    try:
        _seed(monkeypatch, {"state": "brake", "since": None, "until": datetime(2026, 10, 16),  # noqa: DTZ001 - naive on purpose
                             "reason": None})
        assert "(01:00 UK)" in _banner(c.get("/portfolio-settings", auth=AUTH).text)
    finally:
        monkeypatch.undo()
        time.tzset()

def test_the_drawdown_halt_gives_its_reason_and_a_restart(client, monkeypatch):  # noqa: F811
    c, store = client
    _sleeve(store)
    _seed(monkeypatch, {"state": "halt", "since": None, "until": None, "reason": "Drawdown 15% from the peak",
                        "restart_url": "/fund/restart"})
    b = _banner(c.get("/sleeves/btc-x", auth=AUTH).text)
    assert "The fund is flattened and halted." in b and "Drawdown 15% from the peak" in b
    assert 'action="/fund/restart"' in b and ">Restart the fund</button>" in b


def test_a_failed_book_check_blocks_new_trades_and_resume_says_why(client, monkeypatch):  # noqa: F811
    c, store = client
    _sleeve(store)
    _seed(monkeypatch, {"state": "book_check", "since": None, "until": None,
                        "reason": "The book is out by 12.40 against the account"})
    b = _banner(c.get("/portfolio-settings", auth=AUTH).text)
    assert "The book check failed, so new trades are blocked." in b
    assert '<button type="button" class="secondary" disabled title="The book is out by 12.40 against the account">Resume' in b


def test_a_fund_value_out_of_date_refuses_new_trades_until_it_updates(client, monkeypatch):  # noqa: F811
    c, store = client
    _sleeve(store)
    _seed(monkeypatch, {"state": "stale", "since": None, "until": None, "reason": None})
    b = _banner(c.get("/", auth=AUTH).text)
    assert "The fund&#39;s value is out of date, so new trades are refused until it updates." in b
    assert "Restart" not in b and "Resume" not in b
