"""CCY-8DP (Advisor 13 Oct 16:15 and 16:25 UK; HoE): fills are booked to 8 dp, so a Strategy whose quote currency the
engine holds to fewer places starts only on a named test account (the hub-fed USD test Strategies and the full-stack
check, on the paper account at the hub-fed venue), with a warning that its results are not Strategy evidence. Every
other account, live or a venue not named, is refused with its reason. Backtests never pass through here."""

from types import SimpleNamespace

import pytest

from sleeve_fund.store import Store
from sleeve_fund import venues
from sleeve_fund.supervisor import Supervisor, check_quote_places, is_test_currency_account


def _s(instrument, venue="KRAKEN"):
    return SimpleNamespace(instrument=instrument, venue=venue)


def test_a_quote_held_to_8_places_passes_on_any_account():
    for inst in ("BTC/USDT", "BTC/USDT-PERP"):
        assert check_quote_places(_s(inst, "BINANCE"), "paper") is None
        assert check_quote_places(_s(inst, "BINANCE"), "desk-live") is None


def test_a_usd_quote_on_the_named_test_account_runs_with_its_warning():
    assert set(venues.venue("KRAKEN").sub_8dp_test_accounts) == {"paper"}
    assert all(not p.sub_8dp_test_accounts for n, p in venues.VENUES.items() if n != "KRAKEN")
    words = check_quote_places(_s("BTC/USD"), "paper")
    assert "not Strategy evidence" in words and "USD is held to 2 places" in words
    assert check_quote_places(_s("SUI/USD", None), "paper")  # the default venue is the hub-fed one


@pytest.mark.parametrize("account, venue", [("desk-live", "KRAKEN"), ("pm-test", "KRAKEN"), ("paper", "BINANCE"),
                                            ("paper", "NEWVENUE")],
                         ids=["live", "another-paper-account", "the-paper-venue", "an-unknown-venue"])
def test_every_other_account_is_refused_with_its_reason(account, venue):
    with pytest.raises(ValueError, match=f"account {account} is not one on this venue") as got:
        check_quote_places(_s("BTC/USD", venue), account)
    assert venue not in str(got.value)  # the words reach the PM: no venue named


def test_a_euro_quote_is_held_to_2_places_too_and_refused_off_the_test_account():
    with pytest.raises(ValueError, match="EUR is held to 2 places"):
        check_quote_places(_s("BTC/EUR"), "desk-live")


@pytest.mark.parametrize("account, venue", [("paper", "KRAKEN"), ("desk-live", "BINANCE")],
                         ids=["the-test-account", "another-account"])
def test_a_quote_the_engine_does_not_list_is_refused_on_every_account(account, venue):
    """F-WED1-2: an unlisted currency would be made up at 8 places and pass; it is refused instead (fail closed)."""
    with pytest.raises(ValueError, match="quote currency ZZZ is not one the engine lists") as got:
        check_quote_places(_s("BTC/ZZZ", venue), account)
    assert venue not in str(got.value)
    assert check_quote_places(_s("BTC/USDT", venue), account) is None  # a listed 8-place quote still passes


def test_the_supervisor_starts_the_test_strategy_with_a_warning_and_refuses_a_live_one(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="usd-test", strategy="ping_pong", instrument="BTC/USD",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000)
    store.create_sleeve(name="usd-live", strategy="ping_pong", instrument="BTC/USD",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000)
    store.create_account("desk-live", "live", venue="KRAKEN")
    store.assign_account("usd-live", "desk-live")
    sup = Supervisor(store, python="true")
    assert sup._refused("usd-test") is False
    assert "not Strategy evidence" in store.last_event("usd-test", ("test_currency",))["message"]
    assert sup._refused("usd-live") is True
    assert store.sleeve("usd-live").desired_state == "stopped"
    assert "is not one" in store.last_event("usd-live", ("start_refused",))["message"]


def test_the_pm_views_helper_names_the_same_accounts_as_the_start_check():
    """CCY-8DP-UI (FE) keys on this helper, not a copy of the list."""
    assert is_test_currency_account("paper", "kraken") and is_test_currency_account("paper", None)
    assert not is_test_currency_account("paper", "BINANCE") and not is_test_currency_account("desk-live", "KRAKEN")
