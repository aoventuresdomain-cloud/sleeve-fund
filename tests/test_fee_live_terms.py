"""FEE-LIVE (Advisor FEE-LIVE-DW, 15 Oct; HoE plan-tonight-2130): the paper venue's fee rates per instrument from the
account's fee-rate endpoint, and its margin tiers daily from the public risk-limit endpoint, each fetch kept as a row
(the DA's venue_terms_fetch). The newest good row is last-known-good: over 24 h it alerts and trades, over 72 h or never
fetched entries are refused and the start check fails; exits and stops always run. A rate outside the published
[lowest, VIP 0] is missing. A demo account with no fee endpoint charges paper the published VIP 0, a good row labelled
"published, not account", written at each attempt; a live account never uses it. Research never charges below VIP 0.
Every cell reads recorded venue responses (tests/data/fee_live), never the venue."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path

import pytest
from sqlalchemy import text

from sleeve_fund import venue_terms
from sleeve_fund.store import PUBLISHED_FEES, Store, make_engine, margin_tiers_of, metadata
from sleeve_fund.venues import venue

DATA = Path(__file__).parent / "data" / "fee_live"
T = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
PAPER = venue("BINANCE")  # the engine's paper perp venue profile
PAIR = "BTC/USDT"


def _recorded(name: str) -> dict:
    return json.loads((DATA / name).read_text())


class Recorded:
    """The account's signed V5 GET, answered from a recorded response (or refused as the venue refuses)."""

    def __init__(self, answer=None, refuse: str | None = None):
        self.answer, self.refuse, self.calls = answer, refuse, []

    def call(self, method, path, params=None):
        self.calls.append((method, path, dict(params or {})))
        if self.refuse:
            raise RuntimeError(self.refuse)
        return self.answer["result"]


@pytest.fixture
def store(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db")
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    s = Store(engine=eng)
    s.create_account("paper-main", "paper")
    s.create_account("live-main", "live")
    yield s
    metadata.drop_all(eng)


def test_the_profile_charges_the_published_vip0_and_keeps_its_bounds_with_their_source_and_date():
    assert (PAPER.fees.maker, PAPER.fees.taker) == (D("0.0002"), D("0.00055"))
    assert PAPER.fee_bounds == {"maker": (D("0"), D("0.0002")), "taker": (D("0.0003"), D("0.00055"))}
    assert "15 Oct 2026" in PAPER.fee_bounds_basis and "VIP 0" in PAPER.fee_basis


def test_the_recorded_fee_rate_and_risk_limit_responses_read_as_the_venues_own_strings():
    fee = venue_terms.fee_payload(_recorded("fee-rate-BTCUSDT.json")["result"], "BTCUSDT")
    assert fee == {"symbol": "BTCUSDT", "maker": "0.0002", "taker": "0.00055"}
    tiers = venue_terms.tier_payload(_recorded("risk-limit-BTCUSDT-2026-10-15.json")["result"], "BTCUSDT")
    assert len(tiers["tiers"]) == 35 and tiers["tiers"][0] == {
        "ceiling": "300000", "mmr": "0.0033", "imr": "0.0066", "max_leverage": "150.00", "deduction": "0"}
    assert margin_tiers_of({"payload": tiers}) == PAPER.maintenance_tiers[PAIR]  # the fetched table is the pasted one
    with pytest.raises(ValueError, match="ETHUSDT"):
        venue_terms.fee_payload(_recorded("fee-rate-BTCUSDT.json")["result"], "ETHUSDT")


def test_a_paper_fetch_keeps_the_accounts_rate_and_it_is_fresh(store):
    client = Recorded(_recorded("fee-rate-BTCUSDT.json"))
    venue_terms.fetch_fees(store, PAPER, PAIR, client=client, account="paper-main", live=False, now=T)
    assert client.calls == [("GET", "/v5/account/fee-rate", {"category": "linear", "symbol": "BTCUSDT"})]
    t = venue_terms.term(store, PAPER, "fee", PAIR, T + timedelta(hours=1))
    assert (t.state, t.source, t.value) == ("fresh", "fee-rate", (D("0.0002"), D("0.00055")))
    assert t.fetched_at == T and t.account == "paper-main"


def test_a_demo_account_with_no_fee_endpoint_charges_paper_the_published_vip0_labelled_not_account(store):
    venue_terms.fetch_fees(store, PAPER, PAIR, client=Recorded(refuse="refused (/v5/account/fee-rate): not supported"),
                           account="paper-main", live=False, now=T)
    last = store.latest_venue_fetch("BINANCE", "fee", PAIR)
    t = venue_terms.term(store, PAPER, "fee", PAIR, T)
    assert t.state == "fresh" and t.source == PUBLISHED_FEES and t.value == (D("0.0002"), D("0.00055"))
    assert "published, not account" in t.words and last["source"] == PUBLISHED_FEES
    failed = [r for r in _rows(store) if not r["ok"]]
    assert len(failed) == 1 and "not supported" in failed[0]["reason"]  # the account's refusal is kept too


def test_a_live_account_never_uses_the_published_rate_and_refuses_entries(store):
    venue_terms.fetch_fees(store, PAPER, PAIR, client=Recorded(refuse="refused: not supported"), account="paper-main",
                           live=False, now=T)  # a published row exists from paper's attempt
    venue_terms.fetch_fees(store, PAPER, PAIR, client=Recorded(refuse="refused: not supported"), account="live-main",
                           live=True, now=T)
    t = venue_terms.term(store, PAPER, "fee", PAIR, T, live=True)
    assert t.state == "refused" and "account" in t.words
    assert venue_terms.entry_refusal(store, PAPER, PAIR, T, live=True)


def test_a_rate_outside_the_published_bounds_is_missing_and_the_last_good_one_is_kept(store):
    good = Recorded(_recorded("fee-rate-BTCUSDT.json"))
    venue_terms.fetch_fees(store, PAPER, PAIR, client=good, account="paper-main", live=False, now=T)
    high = json.loads(json.dumps(_recorded("fee-rate-BTCUSDT.json")).replace('"0.00055"', '"0.0007"'))
    venue_terms.fetch_fees(store, PAPER, PAIR, client=Recorded(high), account="paper-main", live=False,
                           now=T + timedelta(hours=1))
    last = store.latest_venue_fetch("BINANCE", "fee", PAIR)
    assert not last["ok"] and last["reason"].startswith("out of bounds")
    t = venue_terms.term(store, PAPER, "fee", PAIR, T + timedelta(hours=2))
    assert t.value == (D("0.0002"), D("0.00055")) and t.fetched_at == T  # never silently widened


@pytest.mark.parametrize(("hours", "state", "refused"), [(23, "fresh", False), (25, "stale", False),
                                                         (73, "refused", True)])
def test_the_age_rules_alert_after_24h_and_refuse_entries_after_72h(store, hours, state, refused):
    venue_terms.fetch_fees(store, PAPER, PAIR, client=Recorded(_recorded("fee-rate-BTCUSDT.json")),
                           account="paper-main", live=False, now=T)
    _tiers(store, T)
    now = T + timedelta(hours=hours)
    t = venue_terms.term(store, PAPER, "fee", PAIR, now)
    assert t.state == state
    why = venue_terms.entry_refusal(store, PAPER, PAIR, now)
    assert bool(why) is refused
    if refused:
        assert "73 hours" in why and "fee" in why and "Binance" not in why and "BINANCE" not in why


def test_never_fetched_refuses_entries(store):
    t = venue_terms.term(store, PAPER, "fee", PAIR, T)
    assert t.state == "refused" and t.value is None and "never" in t.words
    assert "never" in venue_terms.entry_refusal(store, PAPER, PAIR, T)


def test_the_pasted_table_counts_as_a_fetch_dated_15_oct_and_refuses_from_18_oct(store):
    venue_terms.seed_pasted_tiers(store, PAPER)
    venue_terms.seed_pasted_tiers(store, PAPER)  # once only
    rows = [r for r in _rows(store) if r["kind"] == "tiers"]
    assert len(rows) == 1 and rows[0]["source"] == "hand-checked-pasted" and rows[0]["account"] is None
    t = venue_terms.term(store, PAPER, "tiers", PAIR, datetime(2026, 10, 16, 12, tzinfo=timezone.utc))
    assert t.state == "stale" and t.value == PAPER.maintenance_tiers[PAIR]
    assert venue_terms.term(store, PAPER, "tiers", PAIR, datetime(2026, 10, 18, 1, tzinfo=timezone.utc)).state == "refused"


def test_the_daily_tier_pull_reads_the_public_endpoint_and_supersedes_the_pasted_table(store):
    venue_terms.seed_pasted_tiers(store, PAPER)
    seen = []

    def get(url):
        seen.append(url)
        return _recorded("risk-limit-BTCUSDT-2026-10-15.json")

    venue_terms.fetch_tiers(store, PAPER, PAIR, get=get, now=T + timedelta(days=2))
    assert seen == ["https://api.bybit.com/v5/market/risk-limit?category=linear&symbol=BTCUSDT"]
    t = venue_terms.term(store, PAPER, "tiers", PAIR, T + timedelta(days=2, hours=1))
    assert (t.state, t.source) == ("fresh", "risk-limit") and t.value == PAPER.maintenance_tiers[PAIR]


def test_a_tier_table_the_engine_refuses_is_missing(store):
    bad = _recorded("risk-limit-BTCUSDT-2026-10-15.json")
    bad["result"]["list"][1]["mmDeduction"] = "999"  # not continuous at 300,000
    venue_terms.fetch_tiers(store, PAPER, PAIR, get=lambda url: bad, now=T)
    assert not store.latest_venue_fetch("BINANCE", "tiers", PAIR)["ok"]
    assert venue_terms.term(store, PAPER, "tiers", PAIR, T).state == "refused"


def test_research_never_charges_below_vip0_and_says_both_rates(store):
    store.record_venue_fetch("BINANCE", "fee", PAIR, source="fee-rate", ok=True, account="paper-main", at=T,
                             payload={"symbol": "BTCUSDT", "maker": "0.0001", "taker": "0.0004"},
                             bounds=PAPER.fee_bounds)
    q = venue_terms.research_fees(store, PAPER, PAIR)
    assert (q.fees.maker, q.fees.taker) == (D("0.0002"), D("0.00055"))
    assert "0.055%" in q.text and "0.040%" in q.text  # research's rate and the account's, side by side


def _tiers(store, at):
    store.record_venue_fetch("BINANCE", "tiers", PAIR, source="risk-limit", ok=True, at=at,
                             payload=venue_terms.tier_payload(_recorded("risk-limit-BTCUSDT-2026-10-15.json")["result"],
                                                              "BTCUSDT"))


def _rows(store):
    from sqlalchemy import select

    from sleeve_fund.store import venue_terms_fetch_t

    with store.engine.connect() as c:
        return [dict(r._mapping) for r in c.execute(select(venue_terms_fetch_t).order_by(venue_terms_fetch_t.c.id))]
