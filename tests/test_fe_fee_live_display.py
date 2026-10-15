"""FEE-LIVE display (HoE, head 3; Advisor's ruling FEE-UI, 15 Oct): Setup's Costs and data shows, per instrument in use,
the fee and the margin tiers as two rows, each with its own source and fetch time (UK), amber past 24 hours, red past 72
hours or never fetched (entries refused), the out-of-range line when the latest fetch was refused, the published VIP0
schedule labelled when it stands in, and the research-fee line. Display only: every figure is the reader's decimal
string (DA's venue_terms_fetch: Store.latest_good_venue_fetch and latest_venue_fetch); nothing is computed here."""

import re
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest
from test_halt_ref import T0, _make, store  # noqa: F401

AUTH = ("pm", "pw")
H = timedelta(hours=1)
UK = ZoneInfo("Europe/London")
NOW = T0 + 100 * H


BOUNDS = {"maker": ("0.0001", "0.0002"), "taker": ("0.0004", "0.00055")}  # the published [lowest, VIP0]
FEE = {"symbol": "BTCUSDT", "maker": "0.0002", "taker": "0.00055"}
TIERS = {"symbol": "BTCUSDT", "tiers": [
    {"ceiling": "300000", "mmr": "0.0033", "imr": "0.0066", "max_leverage": "150", "deduction": "0"},
    {"ceiling": "800000", "mmr": "0.005", "imr": "0.01", "max_leverage": "100", "deduction": "510"}]}


def _fetch(db, kind, at, *, ok=True, source=None, payload=None, reason=""):
    """One fetch through DA's write path (store.record_venue_fetch), as FEE-LIVE's fetcher writes it."""
    return db.record_venue_fetch("BINANCE", kind, "BTC/USDT", source=source or ("fee-rate" if kind == "fee" else
                                 "risk-limit"), ok=ok, reason=reason, at=at, bounds=BOUNDS if kind == "fee" else None,
                                 payload=payload if payload is not None else (FEE if kind == "fee" else TIERS))


@pytest.fixture
def fetched(store):
    _make(store, "a", 10_000)
    return store


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


def test_fresh_fee_and_tiers_are_two_rows_each_with_its_source_and_uk_time(fetched, monkeypatch):
    db = fetched
    _fetch(db, "fee", NOW - 2 * H)
    _fetch(db, "tiers", NOW - 5 * H)
    v = _view(db)
    (line,) = v["instruments"]
    assert line["fee"]["state"] == "fresh" and line["tiers"]["state"] == "fresh"
    assert line["fee"]["maker"] == "0.0002" and line["fee"]["taker"] == "0.00055"  # the reader's strings, as stored
    words = _words(db, monkeypatch)
    assert "Fees: 0.0002 maker, 0.00055 taker, from the account" in words
    assert "Margin tiers: 2 tiers, from the public tier list" in words
    uk = (NOW - 2 * H).astimezone(UK).strftime("%d %b %H:%M UK")
    assert f"fetched {uk}" in words
    assert "Research tests use the standard published fee or higher, never a lower one." in words
    assert not re.search(r"binance|bybit", words, re.IGNORECASE)


@pytest.mark.parametrize("kind", ["fee", "tiers"])
def test_past_24_hours_a_row_is_amber_on_its_own(fetched, monkeypatch, kind):
    db = fetched
    other = "tiers" if kind == "fee" else "fee"
    _fetch(db, kind, NOW - 30 * H)
    _fetch(db, other, NOW - 1 * H)
    line = _view(db)["instruments"][0]
    assert line[kind]["state"] == "stale" and line[other]["state"] == "fresh"
    assert "30 hours old; the last good value is still in use" in _words(db, monkeypatch)


@pytest.mark.parametrize("age", [73, None])
def test_past_72_hours_or_never_fetched_entries_are_refused(fetched, monkeypatch, age):
    db = fetched
    if age:
        _fetch(db, "fee", NOW - age * H)
    _fetch(db, "tiers", NOW - 1 * H)
    assert _view(db)["instruments"][0]["fee"]["state"] == "refused"
    words = _words(db, monkeypatch)
    assert "new entries are refused until a fresh fee is fetched" in words
    assert ("Never fetched" in words) is (age is None)


def test_a_fetch_outside_the_published_range_is_not_used_and_says_so(fetched, monkeypatch):
    db = fetched
    _fetch(db, "fee", NOW - 3 * H)
    _fetch(db, "fee", NOW - 1 * H, payload={"symbol": "BTCUSDT", "maker": "0.0002", "taker": "0.0007"})  # > VIP0
    _fetch(db, "tiers", NOW - H)
    assert not db.latest_venue_fetch("BINANCE", "fee", "BTC/USDT")["ok"]  # DA's write path refused it
    line = _view(db)["instruments"][0]["fee"]
    assert line["out_of_range"] and line["state"] == "fresh" and line["taker"] == "0.00055"
    uk = (NOW - 3 * H).astimezone(UK).strftime("%d %b %H:%M UK")
    assert (f"latest fetched fee was outside the published range and was not used; using the last good fee from {uk}"
            in _words(db, monkeypatch))


def test_the_published_vip0_schedule_is_labelled_never_as_the_accounts_fee(fetched, monkeypatch):
    from sleeve_fund.store import PUBLISHED_FEES

    db = fetched
    _fetch(db, "fee", NOW - H, source=PUBLISHED_FEES)
    _fetch(db, "tiers", NOW - H)
    words = _words(db, monkeypatch)
    assert "from the published schedule (VIP0), as the account has no fee endpoint" in words
    assert "from the account" not in words
