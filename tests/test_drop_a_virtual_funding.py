"""CUT-A (Advisor Q3-MARGIN, HoE): once the fund holds the cash a Strategy's simulated account is funded virtually from
the fund's equity, so it can post isolated margin with no capital of its own, and that funding is netted out of the
engine's cash before anything compares it with the journal (whose cash starts at 0)."""

from __future__ import annotations

from types import SimpleNamespace
from decimal import Decimal as D

from test_drop_a_fund_book import _rebased
from test_halt_ref import T0, _make, _settle, store  # noqa: F401

from sleeve_fund.paper import node
from sleeve_fund.paper.runtime import SleeveRuntime


def _cfg(store, name):
    """What _balances reads of the Strategy's config: its balance and its market (a perp)."""
    return SimpleNamespace(starting_balance=store.sleeve(name).starting_balance, params={"market": "perp"})


def test_before_the_reset_nothing_is_funded_virtually(store):  # noqa: F811
    _make(store, "a", 10_000)
    rt = SleeveRuntime(store, "a")
    assert node.virtual_funding(rt) == 0.0 and rt.funded == 0.0
    bal = node._balances(_cfg(store, "a"), rt, "BTC", "USDT")
    assert [m.as_double() for m in bal] == [10_000.0]


def test_after_it_the_account_is_funded_from_the_fund_and_reconcile_nets_it_out(store):  # noqa: F811
    _rebased(store)
    _make(store, "x", 0)
    rt = SleeveRuntime(store, "x")
    bal = node._balances(_cfg(store, "x"), rt, "BTC", "USDT")
    assert rt.funded == 100_000.0 * node.VIRTUAL_HEADROOM and [m.as_double() for m in bal] == [rt.funded]
    assert rt.reconcile(cash=0.0, qty=0.0)  # the engine's cash net of the funding equals the journal's 0
    assert not [e for e in store.events("x", limit=20) if e["kind"] == "reconcile_mismatch"]
    assert store.fund_capital() == D(100_000)  # never journaled, never in the book
