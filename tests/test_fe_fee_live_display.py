"""FEE-LIVE display (HoE, head 3; Advisor's ruling FEE-UI, 15 Oct): Setup's Costs and data shows, per instrument in use,
the fee and the margin tiers as two rows, each with its own source and fetch time (UK), amber past 24 hours, red past 72
hours or never fetched (entries refused), the out-of-range line when the latest fetch was refused, the published
schedule labelled when it stands in, and the research-fee line. Display only: each row is the engine's own reading
(venue_terms.live_terms) of DA's venue_terms_fetch rows; nothing is computed here. TEC-FE-FEE-UI (HoE): the cells drive
real journal rows instead of stubbing the store's readers; every intent is unchanged."""

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sleeve_fund import venue_terms
from sleeve_fund.store import PUBLISHED_FEES, Store
from sleeve_fund.venues import venue

pytestmark = [pytest.mark.no_maintenance_tables, pytest.mark.venue_terms_live]
AUTH = ("pm", "pw")
H = timedelta(hours=1)
NOW = datetime(2026, 10, 15, 12, tzinfo=UTC)
PAPER = venue("BINANCE")
PAIR = "BTC/USDT"
RECORDED = Path(__file__).parent / "data" / "fee_live" / "risk-limit-BTCUSDT-2026-10-15.json"  # the engine's table


@pytest.fixture
def db():
    store = Store.in_memory()
    store.create_sleeve(name="a", strategy="ping_pong", instrument=PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
    return store


def _fee(db, age, *, taker="0.00055"):
    db.record_venue_fetch("BINANCE", "fee", PAIR, source=venue_terms.FEE_RATE, ok=True, account=db.account_of("a"),
                          at=NOW - age, bounds=PAPER.fee_bounds,
                          payload={"symbol": "BTCUSDT", "maker": "0.0002", "taker": taker})


def _tiers(db, age):
    db.record_venue_fetch("BINANCE", "tiers", PAIR, source=venue_terms.RISK_LIMIT, ok=True, at=NOW - age,
                          payload=venue_terms.tier_payload(json.loads(RECORDED.read_text())["result"], "BTCUSDT"))


def _view(db):
    from sleeve_fund.dashboard import fee_terms

    return fee_terms.view(db, NOW)


def _words(db, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(dash, "utcnow", lambda: NOW)
    html = TestClient(dash.create_app(db)).get("/setup", auth=AUTH).text
    sect = re.search(r'id="fee-terms".*?</section>', html, re.DOTALL)
    assert sect, "no fee terms panel on Setup"
    return " ".join(re.sub(r"<[^>]+>", " ", sect.group(0)).split())


def test_fresh_fee_and_tiers_are_two_rows_each_with_its_source_and_uk_time(db, monkeypatch):
    _fee(db, 2 * H)
    _tiers(db, 5 * H)
    (line,) = _view(db)["instruments"]
    assert line["fee"]["state"] == "fresh" and line["tiers"]["state"] == "fresh"
    assert line["fee"]["maker"] == "0.0002" and line["fee"]["taker"] == "0.00055"  # as fetched
    words = _words(db, monkeypatch)
    assert "Fees: 0.0002 maker, 0.00055 taker" in words and "from the account" in words
    assert f"Margin tiers: {len(PAPER.maintenance_tiers[PAIR])} tiers" in words and "public risk-limit table" in words
    assert f"fetched {venue_terms.uk(NOW - 2 * H)}" in words and " UK" in words
    assert "Research tests use the standard published fee or higher, never a lower one." in words
    assert not re.search(r"binance|bybit", words, re.IGNORECASE)


@pytest.mark.parametrize("kind", ["fee", "tiers"])
def test_past_24_hours_a_row_is_amber_on_its_own(db, monkeypatch, kind):
    (_fee if kind == "fee" else _tiers)(db, 30 * H)
    (_tiers if kind == "fee" else _fee)(db, 1 * H)
    other = "tiers" if kind == "fee" else "fee"
    line = _view(db)["instruments"][0]
    assert line[kind]["state"] == "stale" and line[other]["state"] == "fresh"
    assert "30 hours ago, over 24 hours; trading goes on" in _words(db, monkeypatch)


@pytest.mark.parametrize("age", [73, None])
def test_past_72_hours_or_never_fetched_entries_are_refused(db, monkeypatch, age):
    if age:
        _fee(db, age * H)
    _tiers(db, 1 * H)
    assert _view(db)["instruments"][0]["fee"]["state"] == "refused"
    words = _words(db, monkeypatch)
    assert "entries are refused" in words and "exits and stops still run" in words
    assert ("never fetched" in words) is (age is None)


def test_a_fetch_outside_the_published_range_is_not_used_and_says_so(db, monkeypatch):
    _fee(db, 3 * H)
    _tiers(db, H)
    _fee(db, H, taker="0.0007")  # above VIP0: DA's store keeps it as failed, "out of bounds: ..."
    line = _view(db)["instruments"][0]["fee"]
    assert line["out_of_range"] and line["state"] == "fresh" and line["taker"] == "0.00055"
    assert (f"latest fetched fee was outside the published range and was not used; the last good fee, fetched "
            f"{venue_terms.uk(NOW - 3 * H)}, is still the one in use" in _words(db, monkeypatch))


def test_the_published_schedule_is_labelled_never_as_the_accounts_fee(db, monkeypatch):
    db.record_published_fees("BINANCE", PAIR, account=db.account_of("a"), bounds=PAPER.fee_bounds, at=NOW - H)
    _tiers(db, H)
    assert _view(db)["instruments"][0]["fee"]["state"] == "fresh"
    words = _words(db, monkeypatch)
    assert "published, not account" in words and "from the account" not in words
    assert PUBLISHED_FEES not in words
