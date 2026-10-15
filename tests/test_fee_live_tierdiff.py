"""FEE-LIVE tier table changes (Advisor TIER-DIFF, 15 Oct, ruling (b) for tonight) and the no-silent-fallback rule for a
venue with no fee endpoint (HoE). A pulled table is compared with the one liquidation is priced with by value, tier by
tier, so formatting or order is no difference. One that differs refuses new entries on its instrument; an open
position's stop is held inside half the way to the NEARER liquidation of the two tables, clamped once with the warning
(STOP-LOOSEN-1), never flattened, and the alert names the positions it touches. Kraken spot, with no fee endpoint read,
charges its published Tier 1 from a labelled row written each attempt, alerted and on the start check. Recorded
responses only."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace

import pytest

from sleeve_fund import markets, start_check, venue_terms
from sleeve_fund.fees import resolve
from sleeve_fund.store import PUBLISHED_FEES, Store
from sleeve_fund.strategies.base import LongFlatStrategy
from sleeve_fund.venues import venue

pytestmark = [pytest.mark.no_maintenance_tables, pytest.mark.venue_terms_live]
DATA = Path(__file__).parent / "data" / "fee_live"
PAPER = venue("BINANCE")
PAIR = "BTC/USDT"
T = datetime(2026, 10, 15, 12, tzinfo=UTC)


def _answer():
    return json.loads((DATA / "risk-limit-BTCUSDT-2026-10-15.json").read_text())


def _store(qty=0.0):
    store = Store.in_memory()
    store.create_sleeve(name="p1", strategy="ping_pong", instrument=PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=20_000, venue="BINANCE", params={"market": "perp"})
    if qty:
        store.record_equity("p1", equity=20_000, cash=20_000 - qty * 60_000, qty=qty, price=60_000, benchmark=20_000)
    return store


def _doubled():
    """The venue's table with every maintenance rate and deduction half as much again: still continuous, the top rate
    0.9 under 1, and nearer for every position."""
    a = _answer()
    for r in a["result"]["list"]:
        r["maintenanceMargin"] = str(D(r["maintenanceMargin"]) * D("1.5"))
        r["mmDeduction"] = str(D(r["mmDeduction"] or "0") * D("1.5"))
    return a


def test_the_same_table_formatted_and_ordered_differently_is_no_difference():
    """TIER-DIFF 1: trailing zeros, a blank deduction written "0" and the tiers listed top first are the same values:
    no refusal, no alert."""
    from sleeve_fund.alerts import Forwarder

    store = _store()
    a = _answer()
    for r in a["result"]["list"]:
        r["maintenanceMargin"] += "000"
        r["riskLimitValue"] += ".00"
        r["mmDeduction"] = r["mmDeduction"] or "0"
    a["result"]["list"].reverse()
    venue_terms.fetch_tiers(store, PAPER, PAIR, get=lambda url: a, now=T)
    t = venue_terms.term(store, PAPER, "tiers", PAIR, T + timedelta(hours=1))
    assert (t.state, t.source) == ("fresh", "risk-limit"), t.words
    assert venue_terms.differing_tiers(store, PAPER, PAIR) is None
    sent = []
    store.record_venue_fetch("BINANCE", "fee", PAIR, source="fee-rate", ok=True, at=T, bounds=PAPER.fee_bounds,
                             account=store.account_of("p1"),
                             payload={"symbol": "BTCUSDT", "maker": "0.0002", "taker": "0.00055"})
    Forwarder(store, environ={"ALERT_WEBHOOK_URL": "https://hooks.example.com/x"},
              send=lambda url, text: sent.append(text), now=lambda: T + timedelta(hours=1)).step()
    assert not any("tier" in s for s in sent), sent


def _position(store, liq_now):
    """Just enough of a long of 1 at 60,000 at 3x on BTC/USDT for the stop cap: the engine's own clamp code runs."""
    s = SimpleNamespace(_backtest=False, _cfg=SimpleNamespace(perp=object()), _margin=True, _safety_stop=None,
                        _entry_px=60_000.0, _entry_side=1, _last_close=60_000.0, _stop_frac=None, _stop_capped=False,
                        stops_clamped=0, _net_position=lambda: (1.0, 60_000.0),
                        _mark=lambda: (None, -40_000.0, 1.0, None), _liq_now=lambda cash, qty: liq_now,
                        runtime=SimpleNamespace(store=store, name="p1", venue="BINANCE", now=lambda: T,
                                                profile=SimpleNamespace(stop_to_liquidation=0.5)))
    for f in ("_pulled_liq", "_capped_stop", "_stop_clamped", "_hold_stop_capped"):
        setattr(s, f, (lambda f: lambda *a: getattr(LongFlatStrategy, f)(s, *a))(f))
    return s


def test_a_changed_table_holds_an_open_stop_to_the_nearer_liquidation_clamped_once_never_flattened(monkeypatch):
    """TIER-DIFF 2 and FEE-LIVE-DW's tier-change cell (Advisor correction: clamp, not refuse). Long 1 at 60,000 at 3x
    with 20,000 of margin. The pulled table raises maintenance by half, so its liquidation is nearer (higher) than the priced
    table's; a 30% stop is held at half the way to the nearer one, with one stop_clamped_liquidation warning, and the
    position stays open (the stub has no way to flatten)."""
    monkeypatch.setattr(LongFlatStrategy, "_funded", lambda self: 0.0)
    monkeypatch.setattr(LongFlatStrategy, "_max_leverage", lambda self: 3.0)
    priced = markets.isolated_liquidation(-40_000.0, 1.0, 60_000.0, 3.0, PAPER.maintenance_tiers[PAIR])
    store = _store(qty=1.0)
    venue_terms.seed_pasted_tiers(store, PAPER)
    s = _position(store, priced)
    assert s._pulled_liq(-40_000.0, 1.0) is None  # the pasted table is the priced one: nothing nearer
    assert s._capped_stop(0.30)[1] == pytest.approx(priced)

    venue_terms.fetch_tiers(store, PAPER, PAIR, get=lambda url: _doubled(), now=T)
    pulled = markets.isolated_liquidation(-40_000.0, 1.0, 60_000.0, 3.0, venue_terms.differing_tiers(store, PAPER, PAIR))
    assert pulled > priced  # nearer for the long
    held, liq = s._capped_stop(0.30)
    assert liq == pytest.approx(pulled)
    assert held == pytest.approx(1 - (60_000 - 0.5 * (60_000 - pulled)) / 60_000)
    assert s._capped_stop(held * 0.9) == (held * 0.9, None)  # a tighter stop is kept as it is
    assert s._hold_stop_capped(0.30) and s._stop_frac == pytest.approx(held)
    assert not s._hold_stop_capped(0.30)  # at the cap already: nothing more
    warned = [e for e in store.events("p1") if e["kind"] == "stop_clamped_liquidation"]
    assert len(warned) == 1 and warned[0]["level"] == "warning", warned


def test_the_alert_names_the_positions_a_changed_table_touches():
    from sleeve_fund.alerts import Forwarder

    store = _store(qty=0.1)
    venue_terms.fetch_tiers(store, PAPER, PAIR, get=lambda url: _doubled(), now=T)
    sent = []
    Forwarder(store, environ={"ALERT_WEBHOOK_URL": "https://hooks.example.com/x"},
              send=lambda url, text: sent.append(text), now=lambda: T + timedelta(hours=1)).step()
    said = [x for x in sent if "tier table changed" in x]
    assert said and "new entries refused on BTC/USDT until re-checked" in said[0] and "p1" in said[0], sent
    sec = start_check.check_venue_terms(store, now=T + timedelta(hours=1))
    assert sec.failed and any("tier table changed" in w for k, w in sec.lines if k == "FAIL"), sec.lines


def test_a_kraken_strategys_fees_come_from_a_labelled_published_row_alerted_and_on_the_start_check():
    """HoE: no venue falls back silently. Kraken spot reads no fee endpoint here, so each hourly attempt writes its
    published Tier 1 as a row labelled published, not account; paper is charged from it, the alert says so once a day,
    and the start check shows it."""
    from sleeve_fund import supervisor
    from sleeve_fund.alerts import Forwarder

    store = Store.in_memory()
    store.create_sleeve(name="k1", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="KRAKEN")
    kraken, account = venue("KRAKEN"), store.account_of("k1")
    supervisor.Supervisor(store).check_venue_terms({}, now=T, get=lambda url: pytest.fail("spot has no tiers"))
    row = store.latest_good_venue_fetch("KRAKEN", "fee", "BTC/USD", account=account)
    assert (row["source"], row["payload"]["maker"], row["payload"]["taker"]) == (PUBLISHED_FEES, "0.0040", "0.0080")
    q = resolve("KRAKEN", store, "BTC/USD", now=T + timedelta(minutes=5), account=account)
    assert (q.fees.maker, q.fees.taker, q.source) == (kraken.fees.maker, kraken.fees.taker, "published")
    assert "published, not account" in q.text and "Kraken" not in q.text
    sec = start_check.check_venue_terms(store, now=T + timedelta(minutes=5))
    assert not sec.failed and any(k == "NOTE" and "published, not account" in w for k, w in sec.lines), sec.lines
    sent = []
    Forwarder(store, environ={"ALERT_WEBHOOK_URL": "https://hooks.example.com/x"},
              send=lambda url, text: sent.append(text), now=lambda: T + timedelta(minutes=5)).step()
    assert any("BTC/USD fee rates" in x and "published, not account" in x for x in sent), sent
    assert not any("Kraken" in x or "KRAKEN" in x for x in sent)
