"""F-FC-4 (Advisor 15 Oct ~02:35 UK): inside the fund a liquidation's halt reads "Position margin lost (liquidated):
X, Y% of the fund at entry", Y = X over the fund equity the position's first entry was sized off (net deposits +
realised + min(0, unrealised), at its mark), kept through adds and partial reductions, uncapped. The Strategy's own
equity starts at 0 there, so "of strategy equity" read "all". Outside the fund nothing changes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from sleeve_fund.paper.runtime import WIPED_OUT
from sleeve_fund.store import Store
from sleeve_fund.strategies.base import LongFlatStrategy

T0 = datetime(2026, 10, 15, 3, 0, tzinfo=timezone.utc)


def _journal(store, fills):
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=0, risk_profile="balanced")
    for k, (oid, side, qty, px, fee, intent, signal) in enumerate(fills):
        ts = T0 + timedelta(minutes=k)
        store.record_order("s1", order_id=oid, side=side, qty=qty, intent=intent, reason=intent, signal=signal, ts=ts)
        store.record_fill("s1", side=side, qty=qty, price=px, fee=fee, order_id=oid, trade_id=oid, ts=ts)
        store.update_order(oid, status="filled", fill_qty=qty, fill_px=px, fee=fee)


def _strategy(store, in_fund):
    rt = SimpleNamespace(store=store, name="s1", starting_balance=0.0, taker_fee=0.0005, backtest=False,
                         _last_equity=0.0, peak=0.0)
    return SimpleNamespace(runtime=rt, _paper_gated=lambda: object() if in_fund else None)


@pytest.fixture(autouse=True)
def _two_x(monkeypatch):
    monkeypatch.setattr(LongFlatStrategy, "_max_leverage", lambda self: 2.0)


ENTRY = ("e1", "BUY", 0.1, 60_000.0, 3.0, "entry", {"fund_equity": 100_000.0})
ADD = ("e2", "BUY", 0.1, 60_000.0, 3.0, "entry", {"fund_equity": 50_000.0})  # the fund lower by then: not Y's base
LIQ = ("l1", "SELL", 0.2, 30_100.0, 3.01, "liquidation", {})


def test_inside_the_fund_y_is_of_the_fund_equity_the_first_entry_was_sized_off_through_an_add():
    store = Store.in_memory()
    _journal(store, [ENTRY, ADD, LIQ])
    s = _strategy(store, in_fund=True)
    fees, taken, before, journaled = LongFlatStrategy._liquidation_figures(s, "l1")
    assert (before, taken, journaled) == (100_000.0, 0.2, True)
    text = LongFlatStrategy._margin_lost(s, taken, 60_000.0, fees, before)
    # X: the margin 0.2 x 60,000 / 2 = 6,000.00, the entry fees 6.00 and the liquidation's 3.01
    assert text == f"{WIPED_OUT}: 6,009.01, 6.0% of the fund at entry"


def test_the_advisors_example_reads_five_percent_of_the_fund():
    s = _strategy(Store.in_memory(), in_fund=True)
    text = LongFlatStrategy._margin_lost(s, 0.1, 99_751.6, 0.0, 99_751.6)  # X 4,987.58 over a 99,751.60 fund
    assert text == f"{WIPED_OUT}: 4,987.58, 5.0% of the fund at entry"


def test_inside_the_fund_an_entry_with_no_journaled_fund_figure_reads_all_never_the_strategys_own():
    store = Store.in_memory()
    _journal(store, [(*ENTRY[:6], {}), LIQ[:2] + (0.1,) + LIQ[3:]])
    s = _strategy(store, in_fund=True)
    fees, taken, before, _ = LongFlatStrategy._liquidation_figures(s, "l1")
    assert before is None
    assert LongFlatStrategy._margin_lost(s, taken, 60_000.0, fees, before).endswith(", all of the fund at entry")


def test_outside_the_fund_nothing_changes():
    store = Store.in_memory()
    _journal(store, [ENTRY, LIQ[:2] + (0.1,) + LIQ[3:]])
    s = _strategy(store, in_fund=False)
    s.runtime.starting_balance = 10_000.0
    fees, taken, before, _ = LongFlatStrategy._liquidation_figures(s, "l1")
    assert before == 10_000.0  # the journal's cash at the first fill, as ruled 20:37
    assert LongFlatStrategy._margin_lost(s, taken, 60_000.0, fees, before) == \
        f"{WIPED_OUT}: 3,006.01, 30% of strategy equity at entry"


@pytest.mark.parametrize("in_fund, journaled", [(True, {"fund_equity": 12_345.68}), (False, {})])
def test_an_entry_journals_the_fund_equity_it_was_sized_off_only_inside_the_fund(in_fund, journaled):
    s = _strategy(Store.in_memory(), in_fund)
    s._in_fund = lambda: LongFlatStrategy._in_fund(s)
    assert LongFlatStrategy._fund_at_entry(s, 12_345.678) == journaled
