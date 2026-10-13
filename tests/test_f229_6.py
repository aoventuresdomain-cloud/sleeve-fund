"""F229-6 (HoE 13 Oct 18:12 UK): a Strategy whose quote currency is held under 8 places, on the test account its venue
names (CCY-8DP), is outside the fund's book. One predicate (supervisor.outside_fund_book) leaves it out of the fund's
equity and of both fund_holdings, and keeps its entries off the book gate; its own per-Strategy checks still run."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from sleeve_fund import supervisor
from sleeve_fund.paper import portfolio_gate
from sleeve_fund.store import Store


def _store(tmp_path) -> Store:
    store = Store(f"sqlite:///{tmp_path}/f.db", portfolio_gate=True)
    for name, inst, venue in (("usd-test", "BTC/USD", "KRAKEN"), ("usdt", "BTC/USDT", "BINANCE")):
        store.create_sleeve(name=name, strategy="ping_pong", instrument=inst, bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, venue=venue)
    return store


def test_the_predicate_is_a_sub_8dp_quote_on_its_venues_named_test_account(tmp_path):
    store = _store(tmp_path)
    assert supervisor.outside_fund_book(store, store.sleeve("usd-test"))
    assert not supervisor.outside_fund_book(store, store.sleeve("usdt"))  # held to 8 places: the fund's
    store.create_account("desk-live", "live", venue="KRAKEN")
    store.assign_account("usd-test", "desk-live")  # not a named test account (it could never start there)
    assert not supervisor.outside_fund_book(store, store.sleeve("usd-test"))


def test_the_funds_equity_leaves_the_test_strategy_out(tmp_path):
    store = _store(tmp_path)
    assert supervisor.fund_equity(store) == Decimal(10_000)  # usdt's capital only


def test_both_fund_holdings_leave_the_test_strategys_position_out(tmp_path, monkeypatch):
    store = _store(tmp_path)
    held = {"usd-test": 0.1, "usdt": 0.0}
    monkeypatch.setattr(store, "journal_book", lambda name, bal: {"qty": held[name], "entry_px": 60_000.0,
                                                                  "fills": 1, "cash": 0.0})
    monkeypatch.setattr(store, "last_equity", lambda name: {"price": 60_000.0})
    assert portfolio_gate.fund_holdings(store, atr_pct=lambda s: 0.02) == ()
    assert supervisor.fund_holdings(store) == ()  # the cache row's stub doesn't see it either
    held.update({"usd-test": 0.0, "usdt": 0.1})
    assert len(portfolio_gate.fund_holdings(store, atr_pct=lambda s: 0.02)) == 1


def test_one_predicate_decides_all_of_them(tmp_path, monkeypatch):
    """Flip the predicate and every reader follows: equity, both holdings and the strategy's gate."""
    store = _store(tmp_path)
    monkeypatch.setattr(supervisor, "outside_fund_book", lambda store, s: True)
    assert supervisor.fund_equity(store) == 0
    monkeypatch.setattr(store, "journal_book", lambda name, bal: {"qty": 0.1, "entry_px": 60_000.0, "fills": 1,
                                                                  "cash": 0.0})
    monkeypatch.setattr(store, "last_equity", lambda name: {"price": 60_000.0})
    assert portfolio_gate.fund_holdings(store, atr_pct=lambda s: 0.02) == ()
    assert supervisor.fund_holdings(store) == ()
    assert _gate_of(store, "usdt") is None


@pytest.mark.parametrize("name, gated", [("usd-test", False), ("usdt", True)])
def test_the_test_strategys_entries_reserve_nothing_against_the_book_gate(tmp_path, name, gated):
    store = _store(tmp_path)
    assert (_gate_of(store, name) is not None) is gated


def _gate_of(store, name):
    from sleeve_fund.strategies.base import LongFlatStrategy

    s = LongFlatStrategy.__new__(LongFlatStrategy)
    s._portfolio, s._paper_gate = None, None
    s.runtime = SimpleNamespace(name=name, backtest=False, store=store)
    return LongFlatStrategy._paper_gated(s)
