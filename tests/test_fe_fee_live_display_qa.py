"""FEE-LIVE display QA (HoE, head 3, 15:24 UK): the Setup panel "Fees and margin tiers in use" must tell the PM the
same thing the engine acts on. For each case the panel's state per instrument and kind (fresh, stale, refused) is set
against venue_terms.live_terms, the reader the start check, the alerts and the entry refusal use, and the panel's words
are checked for the source the engine names. Real journal rows (record_venue_fetch) and the recorded risk-limit
response only; FEE-LIVE's rules as they are (venue_terms_live)."""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path

import pytest

from sleeve_fund import venue_terms
from sleeve_fund.store import Store
from sleeve_fund.venues import venue

pytestmark = [pytest.mark.no_maintenance_tables, pytest.mark.venue_terms_live]
DATA = Path(__file__).parent / "data" / "fee_live"
PAPER, SPOT = venue("BINANCE"), venue("KRAKEN")
PAIR = "BTC/USDT"
NOW = datetime(2026, 10, 15, 12, tzinfo=UTC)
H = timedelta(hours=1)
AUTH = ("pm", "pw")


def _tiers(scale: str = "1") -> dict:
    a = json.loads((DATA / "risk-limit-BTCUSDT-2026-10-15.json").read_text())
    for r in a["result"]["list"]:
        r["maintenanceMargin"] = str(D(r["maintenanceMargin"]) * D(scale))
        r["mmDeduction"] = str(D(r["mmDeduction"] or "0") * D(scale))
    return venue_terms.tier_payload(a["result"], "BTCUSDT")


def _perp() -> Store:
    store = Store.in_memory()
    store.create_sleeve(name="p1", strategy="ping_pong", instrument=PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=20_000, venue="BINANCE", params={"market": "perp"})
    return store


def _fee(store, age):
    store.record_venue_fetch("BINANCE", "fee", PAIR, source=venue_terms.FEE_RATE, ok=True,
                             account=store.account_of("p1"), at=NOW - age, bounds=PAPER.fee_bounds,
                             payload={"symbol": "BTCUSDT", "maker": "0.0002", "taker": "0.00055"})


def _tier_row(store, age, *, scale="1"):
    store.record_venue_fetch("BINANCE", "tiers", PAIR, source=venue_terms.RISK_LIMIT, ok=True, at=NOW - age,
                             payload=_tiers(scale))


def _shown(store) -> dict:
    """{(instrument, kind): state} as the Setup panel's view gives it."""
    from sleeve_fund.dashboard import fee_terms

    out = {}
    for i in fee_terms.view(store, NOW)["instruments"]:
        for kind in ("fee", "tiers"):
            if i.get(kind):
                out[(i["instrument"], kind)] = i[kind]["state"]
    return out


def _engine(store) -> dict:
    return {(t.instrument, t.kind): t.state for t in venue_terms.live_terms(store, NOW)}


def _words(store, monkeypatch) -> str:
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(dash, "utcnow", lambda: NOW)
    html = TestClient(dash.create_app(store)).get("/setup", auth=AUTH).text
    sect = re.search(r'id="fee-terms".*?</section>', html, re.DOTALL)
    assert sect, "no fee terms panel on Setup"
    return " ".join(re.sub(r"<[^>]+>", " ", sect.group(0)).split())


CASES = {
    "fresh": {"fee": 2 * H, "tiers": 3 * H},
    "fee 25 hours": {"fee": 25 * H, "tiers": 3 * H},
    "tiers 25 hours": {"fee": 2 * H, "tiers": 25 * H},
    "fee 73 hours": {"fee": 73 * H, "tiers": 3 * H},
    "tiers 73 hours": {"fee": 2 * H, "tiers": 73 * H},
    "fee never": {"fee": None, "tiers": 3 * H},
    "tiers never": {"fee": 2 * H, "tiers": None},
    "fee exactly 24 hours": {"fee": 24 * H, "tiers": 3 * H},
    "fee exactly 72 hours": {"fee": 72 * H, "tiers": 3 * H},
}


@pytest.mark.parametrize("case", CASES)
def test_the_panel_states_each_row_as_the_engine_does(case):
    store = _perp()
    c = CASES[case]
    if c["fee"] is not None:
        _fee(store, c["fee"])
    if c["tiers"] is not None:
        _tier_row(store, c["tiers"])
    assert _shown(store) == _engine(store)


def test_a_pulled_tier_table_that_differs_from_the_engines_shows_entries_refused(monkeypatch):
    """Advisor TIER-DIFF (b): the engine refuses new entries on the instrument until the table is re-checked; a panel
    that reads the table as fresh tells the PM the opposite."""
    store = _perp()
    _fee(store, 2 * H)
    _tier_row(store, H, scale="1.5")
    assert _engine(store)[(PAIR, "tiers")] == venue_terms.REFUSED
    assert _shown(store)[(PAIR, "tiers")] == venue_terms.REFUSED
    assert "new entries refused on BTC/USDT" in _words(store, monkeypatch)


def test_the_pasted_hand_checked_table_is_not_called_the_public_tier_list(monkeypatch):
    store = _perp()
    _fee(store, 2 * H)
    venue_terms.seed_pasted_tiers(store, PAPER)
    words = _words(store, monkeypatch)
    assert "public tier list" not in words and "hand-checked" in words


def test_a_spot_strategy_not_yet_recorded_is_not_shown_as_refused(monkeypatch):
    """The engine charges a spot venue's published schedule and trades on (STALE, "not recorded yet"); a panel line
    saying new entries are refused would send the PM after a fault that is not there."""
    store = Store.in_memory()
    store.create_sleeve(name="k1", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="KRAKEN")
    assert _engine(store)[("BTC/USD", "fee")] != venue_terms.REFUSED
    assert _shown(store) == _engine(store)
    assert "refused" not in _words(store, monkeypatch)


def test_a_spot_published_row_says_published_not_account_and_never_vip0(monkeypatch):
    """Kraken's stand-in is its published Tier 1, not a VIP 0 schedule; the engine's words say "published, not
    account"."""
    store = Store.in_memory()
    store.create_sleeve(name="k1", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="KRAKEN")
    store.record_published_fees("KRAKEN", "BTC/USD", account=store.account_of("k1"), bounds=SPOT.fee_bounds, at=NOW - H)
    words = _words(store, monkeypatch)
    assert "published, not account" in words and "VIP0" not in words and "from the account" not in words
    assert not re.search(r"kraken|binance|bybit", words, re.IGNORECASE)
