"""Venue terms check (DA, head 3; plan-tonight-2130.md): B8 also sets the engine's maintenance tiers and fees beside
the venue data FEE-LIVE fetched (venue_terms_fetch), with a verdict per row. Read-only: it reads the journal's fetch
rows and writes nothing. Without a journal (no store) B8 is as it was."""

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from sleeve_fund.store import PUBLISHED_FEES, Store, make_engine, metadata
from test_b8_instrument_terms import BTC_FUNDING, _sleeve, _symbol, _venue, b8, contract, hub  # noqa: F401 - autouse stubs

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
BOUNDS = {"maker": ("0", "0.0002"), "taker": ("0.0002", "0.00055")}
pytestmark = pytest.mark.no_maintenance_tables  # the venue's own tables (VS-1), not the tests' flat stand-in


def _tiers_payload():
    """The engine's kept BTC/USDT table as the venue's JSON would give it (strings; the last ceiling the venue's)."""
    from sleeve_fund.venues import venue

    out = []
    for t in venue("binance").maintenance_tiers["BTC/USDT"]:
        ceiling = "1200000000" if t.ceiling == float("inf") else f"{t.ceiling:.0f}"
        out.append({"ceiling": ceiling, "mmr": repr(t.rate), "imr": "1", "max_leverage": "1",
                    "deduction": f"{t.amount:.0f}"})
    return {"symbol": "BTCUSDT", "tiers": out}


@pytest.fixture
def store(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")  # CI also runs these against Postgres, as test_sleeve_runtime does
    if url:
        engine = make_engine(url)
        metadata.drop_all(engine)
        s = Store(engine=engine)
    else:
        s = Store(f"sqlite:///{tmp_path}/t.db")
    s.create_account("paper-main", "paper")
    return s


def _fees(store, at, taker="0.00055", maker="0.0002", source="fee-rate", **kw):
    store.record_venue_fetch("BINANCE", "fee", "BTC/USDT", source=source, ok=True, account="paper-main", at=at,
                             payload={"symbol": "BTCUSDT", "maker": maker, "taker": taker}, bounds=BOUNDS, **kw)


def _tiers(store, at, payload=None):
    store.record_venue_fetch("BINANCE", "tiers", "BTC/USDT", source="risk-limit", ok=True, at=at,
                             payload=payload or _tiers_payload())


def _run(tmp_path, capfd, store):
    _sleeve(tmp_path, "a", "BTC/USDT", "binance")
    code = b8.main([], get=_venue([_symbol("BTCUSDT")], BTC_FUNDING), pattern=str(tmp_path / "*.toml"), store=store,
                   now=NOW)
    out = capfd.readouterr().out
    return code, {line.split(None, 1)[1].rsplit(None, 3)[0].strip(): line.split()[-1]
                  for line in out.splitlines() if line.startswith("BTC/USDT")}, out


def test_fresh_fetches_equal_to_the_engines_terms_pass(tmp_path, capfd, store):
    _tiers(store, NOW - timedelta(hours=2))
    _fees(store, NOW - timedelta(hours=2))
    code, rows, out = _run(tmp_path, capfd, store)
    assert rows["maint. tiers (fetched)"] == "PASS" and rows["taker fee (fetched)"] == "PASS", out
    assert rows["maker fee (fetched)"] == "PASS" and code == 0, out


def test_never_fetched_is_a_must_never_a_pass(tmp_path, capfd, store):
    code, rows, out = _run(tmp_path, capfd, store)
    assert rows["maint. tiers (fetched)"] == "MUST" and rows["taker fee (fetched)"] == "MUST" and code == 1, out
    assert "never fetched" in out


def test_a_fetched_table_unlike_the_engines_is_a_must_and_names_the_tier(tmp_path, capfd, store):
    from decimal import Decimal

    payload = _tiers_payload()
    t = payload["tiers"]
    t[1]["mmr"] = str(Decimal(t[1]["mmr"]) + Decimal("0.001"))  # tier 2's rate differs from the engine's
    for a, b in zip(t, t[1:]):  # every deduction from tier 2 on kept continuous, so the fetch itself is good
        b["deduction"] = str(Decimal(a["deduction"]) + Decimal(a["ceiling"]) * (Decimal(b["mmr"]) - Decimal(a["mmr"])))
    _tiers(store, NOW - timedelta(hours=1), payload)
    _fees(store, NOW - timedelta(hours=1))
    code, rows, out = _run(tmp_path, capfd, store)
    assert rows["maint. tiers (fetched)"] == "MUST" and "tier 2" in out and code == 1, out


@pytest.mark.parametrize("hours, verdict", [(23, "PASS"), (25, "FOLLOW-UP"), (73, "MUST")])
def test_age_is_judged_as_fee_live_does(tmp_path, capfd, store, hours, verdict):
    _tiers(store, NOW - timedelta(hours=hours))
    _fees(store, NOW - timedelta(hours=hours))
    _, rows, out = _run(tmp_path, capfd, store)
    assert rows["maint. tiers (fetched)"] == verdict and rows["taker fee (fetched)"] == verdict, out


def test_the_fee_compared_is_what_paper_charges_the_instrument_not_the_published_vip0(tmp_path, capfd, store):
    """X1 (PE4 cross-check, HoE 15:33 UK): an account fetch below the published VIP 0 is what paper charges on the
    instrument (fees.resolve with the instrument), so the rows read equal and PASS, never "engine higher"."""
    from sleeve_fund import fees

    _tiers(store, NOW - timedelta(hours=1))
    _fees(store, NOW - timedelta(hours=1), taker="0.0004", maker="0.0001")  # the account pays less than published
    charged = fees.resolve("binance", store, "BTC/USDT", now=NOW, account="paper-main").fees
    assert (charged.taker, charged.maker) == (Decimal("0.0004"), Decimal("0.0001"))  # what paper's fills pay
    code, rows, out = _run(tmp_path, capfd, store)
    assert rows["taker fee (fetched)"] == "PASS" and rows["maker fee (fetched)"] == "PASS" and code == 0, out
    assert "account          equal" in out and "higher" not in out, out


def test_the_engine_charging_less_than_the_fetched_fee_is_a_must_and_more_a_follow_up(tmp_path, capfd, store,
                                                                                       monkeypatch):
    """The comparison can fire both ways against the instrument-aware rate: paper charging less than the venue is a
    MUST, more a FOLLOW-UP (TEC on the 15 Oct cell: it pinned the published VIP 0 as "the engine's")."""
    from sleeve_fund import fees

    _tiers(store, NOW - timedelta(hours=1))
    _fees(store, NOW - timedelta(hours=1), taker="0.0004", maker="0.0001")
    real, asked = fees.resolve, []

    def shifted(change):
        def resolve(venue=None, store=None, instrument=None, **kw):
            asked.append(instrument)
            return change(real(venue, store, instrument, **kw))
        return resolve

    monkeypatch.setattr(fees, "resolve", shifted(_cheaper))
    code, rows, out = _run(tmp_path, capfd, store)
    assert rows["taker fee (fetched)"] == "MUST" and code == 1 and "lower" in out, out
    assert asked and set(asked) == {"BTC/USDT"}  # asked for the instrument's rate, never the venue's published one
    monkeypatch.setattr(fees, "resolve", shifted(_dearer))
    code, rows, out = _run(tmp_path, capfd, store)
    assert rows["taker fee (fetched)"] == "FOLLOW-UP" and rows["maker fee (fetched)"] == "FOLLOW-UP", out


def _dearer(quote):
    from dataclasses import replace

    from sleeve_fund.instruments import FeeSchedule

    return replace(quote, fees=FeeSchedule(Decimal("0.0002"), Decimal("0.00055")))


def _cheaper(quote):
    from dataclasses import replace
    from decimal import Decimal

    from sleeve_fund.instruments import FeeSchedule

    return replace(quote, fees=FeeSchedule(Decimal("0.00005"), Decimal("0.0003")))


def test_the_published_stand_in_and_a_rejected_newer_fetch_are_follow_ups_that_say_so(tmp_path, capfd, store):
    _tiers(store, NOW - timedelta(hours=1))
    _fees(store, NOW - timedelta(hours=2), source=PUBLISHED_FEES)
    code, rows, out = _run(tmp_path, capfd, store)
    assert rows["taker fee (fetched)"] == "FOLLOW-UP" and "published, not account" in out, out
    _fees(store, NOW - timedelta(hours=1), taker="0.0009")  # out of bounds: kept as failed, never used
    code, rows, out = _run(tmp_path, capfd, store)
    assert "latest fetch not used: out of bounds" in out, out


def test_the_check_writes_nothing(tmp_path, capfd, store):
    _tiers(store, NOW - timedelta(hours=1))
    _fees(store, NOW - timedelta(hours=1))
    with store.engine.connect() as c:
        from sqlalchemy import text

        before = c.execute(text("SELECT count(*) FROM venue_terms_fetch")).scalar()
    _run(tmp_path, capfd, store)
    with store.engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM venue_terms_fetch")).scalar() == before


def test_an_accounts_fetched_rates_never_reach_the_public_log(tmp_path, capfd, store):
    """The job log is public (HoE, 12:31 UK): an account's fetched rates print as a relation to the engine's only."""
    _tiers(store, NOW - timedelta(hours=1))
    _fees(store, NOW - timedelta(hours=1), taker="0.000437", maker="0.000123")
    code, rows, out = _run(tmp_path, capfd, store)
    # X1: paper is charged the fetched rate, so the rows read equal (TEC: they read "more" against the published VIP 0)
    assert rows["taker fee (fetched)"] == "PASS" and rows["maker fee (fetched)"] == "PASS", out
    assert "437" not in out and "123" not in out, out
    assert "account          equal" in out


def test_the_published_stand_in_may_print_its_rates(tmp_path, capfd, store):
    _tiers(store, NOW - timedelta(hours=1))
    _fees(store, NOW - timedelta(hours=1), source=PUBLISHED_FEES)
    _, rows, out = _run(tmp_path, capfd, store)
    assert [line for line in out.splitlines() if "taker fee (fetched)" in line and "0.0005" in line], out
