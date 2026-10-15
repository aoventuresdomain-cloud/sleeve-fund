"""FEE-LIVE wiring (Advisor FEE-LIVE-DW): the instrument's fee rates and tiers reach the start check (FAIL when entries
are refused, NOTE when stale or published), the alerts (once a day while it lasts), what paper is charged
(fees.resolve per instrument) and the entry path (refused past 72 h or never fetched; exits never ask). Recorded
responses only."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from sleeve_fund import start_check, venue_terms
from sleeve_fund.fees import resolve
from sleeve_fund.store import PUBLISHED_FEES, Store
from sleeve_fund.venues import venue

pytestmark = [pytest.mark.no_maintenance_tables, pytest.mark.venue_terms_live]
PAPER = venue("BINANCE")
PAIR = "BTC/USDT"
T = datetime(2026, 10, 15, 12, tzinfo=UTC)
D1 = T + timedelta(days=1)  # the pasted table (15 Oct 00:00) is 36 h old: the daily pull is due
FEE = {"symbol": "BTCUSDT", "maker": "0.0002", "taker": "0.0004"}


def _store():
    store = Store.in_memory()
    store.create_sleeve(name="p1", strategy="ping_pong", instrument=PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
    return store


def _fee(store, at, source="fee-rate", payload=FEE):
    store.record_venue_fetch("BINANCE", "fee", PAIR, source=source, ok=True, payload=payload, at=at,
                             bounds=PAPER.fee_bounds, account=store.account_of("p1"))


def _tiers(store, at):
    venue_terms.seed_pasted_tiers(store, replace(PAPER, maintenance_read_on=at.date()))


def test_the_start_check_fails_while_entries_are_refused_and_passes_when_both_are_fresh():
    store = _store()
    sec = start_check.check_venue_terms(store, now=T)
    assert sec.failed and ("FAIL", "BTC/USDT fee rates: never fetched, so entries are refused; exits and stops still "
                                   "run") in sec.lines, sec.lines
    _fee(store, T)
    _tiers(store, T)
    sec = start_check.check_venue_terms(store, now=T + timedelta(hours=1))
    assert not sec.failed and sec.lines[-1][0] == "PASS", sec.lines
    late = start_check.check_venue_terms(store, now=T + timedelta(hours=25))
    assert not late.failed and any(k == "NOTE" and "over 24 hours" in w for k, w in late.lines), late.lines
    assert not any("BINANCE" in w or "Binance" in w for _, w in late.lines)


def test_the_published_rate_is_a_note_that_says_published_not_account():
    store = _store()
    _fee(store, T, source=PUBLISHED_FEES, payload={"symbol": "BTCUSDT", "maker": "0.0002", "taker": "0.00055"})
    _tiers(store, T)
    sec = start_check.check_venue_terms(store, now=T + timedelta(hours=1))
    assert not sec.failed and any(k == "NOTE" and "published, not account" in w for k, w in sec.lines), sec.lines


def test_the_whole_start_check_has_the_section(tmp_path):
    url = f"sqlite:///{tmp_path}/j.db"
    store = Store(url)
    store.create_sleeve(name="p1", strategy="ping_pong", instrument=PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
    store.engine.dispose()
    _, page = start_check.run(url, environ={})
    assert "Venue fees and tiers: FAIL" in page, page


def test_stale_fees_alert_once_a_day_while_they_last():
    from sleeve_fund.alerts import Forwarder

    store = _store()
    _fee(store, T)
    _tiers(store, T + timedelta(hours=20))
    at = {"now": T + timedelta(hours=23)}
    sent = []
    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": "https://hooks.example.com/x"},
                    send=lambda url, text: sent.append(text), now=lambda: at["now"])
    fwd.step()
    assert not any("fee rates" in t for t in sent), sent
    at["now"] = T + timedelta(hours=25)
    fwd.step()
    fwd.step()
    assert len([t for t in sent if "BTC/USDT fee rates" in t and "over 24 hours" in t]) == 1, sent
    at["now"] = T + timedelta(hours=49)  # the next day, still stale: said again
    fwd.step()
    assert len([t for t in sent if "BTC/USDT fee rates" in t]) == 2, sent
    assert not any("Binance" in t or "BINANCE" in t for t in sent)


def test_paper_is_charged_the_instruments_own_rate_and_the_published_vip0_until_it_is_fetched():
    store = _store()
    never = resolve("BINANCE", store, PAIR, now=T)
    assert never.fees.taker == D("0.00055") and "never fetched" in never.text
    _fee(store, T)
    q = resolve("BINANCE", store, PAIR, now=T + timedelta(hours=1))
    assert (q.fees.maker, q.fees.taker, q.source) == (D("0.0002"), D("0.0004"), "account")
    assert resolve("BINANCE", store).fees.taker == D("0.00055")  # no instrument: the published VIP 0


def test_the_entry_refusal_reads_the_strategys_instrument_and_account_and_a_backtest_never_asks():
    from sleeve_fund.strategies.base import LongFlatStrategy

    store = _store()
    rt = SimpleNamespace(store=store, name="p1", venue="BINANCE", now=lambda: T)
    s = SimpleNamespace(_backtest=False, runtime=rt)
    assert "never fetched" in LongFlatStrategy._venue_terms_refusal(s)
    _fee(store, T)
    _tiers(store, T)
    assert LongFlatStrategy._venue_terms_refusal(s) is None
    rt.now = lambda: T + timedelta(hours=73)
    assert "over 72 hours" in LongFlatStrategy._venue_terms_refusal(s)
    assert LongFlatStrategy._venue_terms_refusal(SimpleNamespace(_backtest=True, runtime=rt)) is None


def test_the_entry_path_stops_at_the_refusal_before_any_order(prices, instrument, full_margin, monkeypatch):
    """The perp entry path asks before it sizes the margin: refused, no entry is sent and the strategy says why."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies.base import LongFlatStrategy
    from test_backtest import _path

    monkeypatch.setattr(LongFlatStrategy, "_venue_terms_refusal", lambda self: "Entry refused: fee rates never fetched")
    feed = _path(prices, [100.0, 100.5, 100.8, 101.5, 101.5, 101.0, 100.0])
    res = run_backtest("ping_pong", feed, instrument, {"market": "perp", "allow_short": True}, half_spread=0,
                       risk_profile="aggressive")
    assert res.fills.empty
    assert "entry_refused_venue_terms" in {e["kind"] for e in res.journal.events_}


def test_the_supervisors_hourly_pass_fetches_fees_per_instrument_and_tiers_daily(monkeypatch):
    """Recorded answers only: the demo account's signed fee-rate call (on the demo host) and the public tier table."""
    import json
    from pathlib import Path

    from sleeve_fund import mirror, supervisor

    data = Path(__file__).parent / "data" / "fee_live"
    calls = []

    def call(self, method, path, params=None):
        calls.append((self.url, method, path, params))
        return json.loads((data / "fee-rate-BTCUSDT.json").read_text())["result"]

    monkeypatch.setattr(mirror.BybitDemo, "call", call)
    store = _store()
    urls = []

    def get(url):
        urls.append(url)
        return json.loads((data / "risk-limit-BTCUSDT-2026-10-15.json").read_text())

    sup = supervisor.Supervisor(store)
    env = {mirror.BYBIT_KEY_ENV: "k", mirror.BYBIT_SECRET_ENV: "s"}
    sup.check_venue_terms(env, now=D1, get=get)
    sup.check_venue_terms(env, now=D1 + timedelta(hours=1), get=get)  # tiers aren't pulled again inside 20 h
    assert [c[0] for c in calls] == ["https://api-demo.bybit.com"] * 2 and len(urls) == 1
    fee = venue_terms.term(store, PAPER, "fee", PAIR, D1 + timedelta(hours=1))
    tiers = venue_terms.term(store, PAPER, "tiers", PAIR, D1 + timedelta(hours=1))
    assert (fee.state, fee.source, tiers.state, tiers.source) == ("fresh", "fee-rate", "fresh", "risk-limit")


def test_with_no_demo_key_on_the_server_paper_is_charged_the_published_vip0_and_says_so():
    from sleeve_fund import supervisor

    store = _store()
    supervisor.Supervisor(store).check_venue_terms({}, now=D1, get=lambda url: {"retCode": 10006, "retMsg": "busy"})
    fee = venue_terms.term(store, PAPER, "fee", PAIR, D1)
    assert fee.source == PUBLISHED_FEES and "published, not account" in fee.words
    tiers = venue_terms.term(store, PAPER, "tiers", PAIR, D1)
    assert tiers.source == "hand-checked-pasted"  # the pull failed; the pasted table is still the last good one
    assert not store.latest_venue_fetch("BINANCE", "tiers", PAIR)["ok"]
