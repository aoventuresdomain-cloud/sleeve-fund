"""Fees come from the connected exchange account; the published schedule is only the fallback."""

from decimal import Decimal

import pytest

from sleeve_fund import venues
from sleeve_fund.fees import resolve
from sleeve_fund.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path}/t.db")


def test_kraken_signature_matches_kraken_documented_example():
    # The worked example in Kraken's REST API authentication docs.
    secret = "kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HAZtuZuj6F1huXg=="
    data = "nonce=1616492376594&ordertype=limit&pair=XBTUSD&price=37500&type=buy&volume=1.25"
    sig = venues.kraken_sign("/0/private/AddOrder", data, "1616492376594", secret)
    assert sig == "4/dpxb3iT4tp/ZCVEwSnEsLxx0bqyhLpdfOpc6fn7OR8+UClSV5n9E6aSS8MPtnRfp32bAb0nmbRn6H8ndwLUQ=="


def test_kraken_account_fees_reads_trade_volume():
    seen = {}

    def post(path, data, headers):
        seen.update(path=path, data=data, headers=headers)
        return {"error": [], "result": {"currency": "ZUSD", "volume": "0.0000",
                                        "fees": {"XXBTZUSD": {"fee": "0.8000"}},
                                        "fees_maker": {"XXBTZUSD": {"fee": "0.4000"}}}}

    secret = "kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HAZtuZuj6F1huXg=="
    fees = venues.kraken_account_fees("key", secret, post=post)
    assert (fees.maker, fees.taker) == (Decimal("0.004"), Decimal("0.008"))
    assert seen["path"] == "/0/private/TradeVolume" and "API-Sign" in seen["headers"]
    assert "AddOrder" not in seen["path"]  # query only


def test_resolve_prefers_the_latest_fetched_account_schedule(store):
    q = resolve("KRAKEN", store)
    assert q.source == "published" and "no account on this venue is connected" in q.text
    store.create_account("kraken-main", "live")
    store.record_fees("KRAKEN", "kraken-main", 0.003, 0.006)
    q = resolve("KRAKEN", store)
    assert q.source == "account" and (q.fees.maker, q.fees.taker) == (Decimal("0.003"), Decimal("0.006"))
    assert "from account kraken-main" in q.text and "0.30% maker, 0.60% taker" in q.text
    with pytest.raises(ValueError):
        store.record_fees("KRAKEN", "kraken-main", 0.2, 0.3)  # a percent passed as a fraction


def test_supervisor_fetches_fees_for_connected_live_accounts_only(store, monkeypatch):
    from sleeve_fund.supervisor import Supervisor

    store.create_account("kraken-main", "live")
    store.create_account("kraken-nokey", "live")
    calls = []

    def fake_fetch(key, secret):
        calls.append(key)
        from sleeve_fund.instruments import FeeSchedule

        return FeeSchedule(Decimal("0.0025"), Decimal("0.004"))

    monkeypatch.setattr(venues.KRAKEN, "fetch_fees", fake_fetch)
    env = {"KRAKEN_API_KEY__KRAKEN_MAIN": "k1", "KRAKEN_API_SECRET__KRAKEN_MAIN": "s1"}
    Supervisor(store).check_fees(environ=env)
    assert calls == ["k1"]
    assert resolve("KRAKEN", store).fees.taker == Decimal("0.004")

    def failing(key, secret):
        raise ValueError(f"EAPI:Invalid key {key} {secret}")

    monkeypatch.setattr(venues.KRAKEN, "fetch_fees", failing)
    Supervisor(store).check_fees(environ=env)
    assert resolve("KRAKEN", store).fees.taker == Decimal("0.004")  # the last good schedule stays
    ev = [e for e in store.events(limit=10) if e["kind"] == "fee_fetch_failed"][0]
    assert "k1" not in ev["message"] and "s1" not in ev["message"]


def test_paper_processes_never_inherit_any_venue_key():
    from sleeve_fund.paper.safety import PaperSafetyError, assert_keyless, credential_var

    assert credential_var("KRAKEN_API_KEY__MAIN") and credential_var("IBKR_API_SECRET__X")
    with pytest.raises(PaperSafetyError):
        assert_keyless({"NEWVENUE_API_KEY__ACC": "x"})


def test_backtest_charges_the_fetched_account_fees(store):
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    store.create_account("kraken-main", "live")
    store.record_fees("KRAKEN", "kraken-main", 0.002, 0.004)
    preview._history.clear()
    res = preview.run("trend_filter", "BTC/USD", {"fast": 5, "slow": 20}, fee_quote=resolve("KRAKEN", store),
                      fetch=lambda pair: synthetic_ohlcv(days=200, seed=4, vol=0.03))
    assert res["fee_schedule"]["taker"] == 0.004 and res["fee_schedule"]["source"] == "account"
