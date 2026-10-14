"""The fund's marked equity the supervisor marks the portfolio with (P2-1b W1 on CASH-2; Advisor 3, HoE 13 Oct): every
strategy's starting capital and cash movement (Store.strategy_cash_pnl) and its position at its last mark; a run a reset
put away counts only its cash movement, so a reset doesn't move it. A strategy put away leaves with its whole equity as
one journaled withdrawal (F-C3B-A, Advisor 14 Oct), which scales the halt's reference with it (test_capital_withdrawal)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone, UTC
from decimal import Decimal as D

import pytest
from test_gate_ledger import _drop_schemas, _store  # noqa: F401

from sleeve_fund.supervisor import fund_equity

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


@pytest.fixture
def store():
    s = _store()
    for name, market in (("spot", {}), ("perp", {"market": "perp", "allow_short": True}), ("flat", {})):
        s.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=10_000, params=market)
    return s


def _trade(store, name, side, qty, px, fee, n, at):
    store.record_order(name, order_id=f"{name}-{n}", side=side, qty=qty, intent="entry" if n == 1 else "exit",
                       reason="test", ts=at)
    store.book_fill(name, side=side, qty=qty, price=px, fee=fee, order_id=f"{name}-{n}", trade_id=f"t{n}", ts=at)


def _mark(store, name, px, at):
    store.record_equity(name, equity=0.0, cash=0.0, qty=0.0, price=px, benchmark=0.0, ts=at)


def test_the_fund_is_every_strategys_capital_cash_moved_and_position_at_its_mark(store):
    _trade(store, "spot", "BUY", 0.1, 60_000.0, 6.0, 1, T0)
    _trade(store, "perp", "SELL", 0.2, 60_000.0, 4.8, 1, T0)  # a short
    _mark(store, "spot", 61_000.0, T0 + timedelta(minutes=1))
    _mark(store, "perp", 61_000.0, T0 + timedelta(minutes=1))
    # spot: 10,000 - 6,000 - 6 + 0.1 x 61,000 = 10,094; perp: 10,000 + 12,000 - 4.8 - 0.2 x 61,000 = 9,795.2
    assert fund_equity(store) == D(10094) + D("9795.2") + D(10000)


def test_a_position_with_no_mark_fails_the_fund_closed(store):
    _trade(store, "spot", "BUY", 0.1, 60_000.0, 6.0, 1, T0)
    with pytest.raises(RuntimeError, match="no mark"):
        fund_equity(store)


def test_a_reset_leaves_the_fund_and_archiving_takes_out_exactly_the_strategys_equity(store):
    """HoE 13 Oct: a reset would otherwise move the book under the 15% halt's reference. Archiving does move it, by
    exactly the strategy's equity, through the withdrawal that scales the reference with it (F-C3B-A)."""
    _trade(store, "spot", "BUY", 0.1, 60_000.0, 6.0, 1, T0)
    _trade(store, "spot", "SELL", 0.1, 62_000.0, 6.2, 2, T0 + timedelta(minutes=5))  # flat, 187.8 up
    before = fund_equity(store)
    assert before == D("30187.8")
    store.set_desired_state("spot", "stopped")
    store.request_reset("spot", "Test finished")
    store.split_run(store.pending_reset("spot"))
    assert fund_equity(store) == before  # the run keeps its 187.8, the strategy its 10,000
    store.set_desired_state("flat", "stopped")
    store.archive("flat")
    assert fund_equity(store) == before - D(10_000)  # put away, its 10,000 leaves with it
    assert store.withdrawn() == {"flat": D(10_000)}


def test_retiring_a_strategy_takes_its_start_and_everything_it_made_out_of_the_fund(store):
    """Retired today is archived (runtime.RETIRED is the archived cause); lifecycle.move only rules on the move and
    writes nothing, and neither starts a fresh run, so the withdrawal is the strategy's start plus everything it made
    (F-C3B-A: a strategy put away leaves the book with its equity; it was the fund's until 14 Oct, HoE 13 Oct)."""
    from sleeve_fund import lifecycle

    _trade(store, "spot", "BUY", 0.1, 60_000.0, 6.0, 1, T0)
    _trade(store, "spot", "SELL", 0.1, 59_000.0, 5.9, 2, T0 + timedelta(minutes=5))  # flat, 111.9 down
    before = fund_equity(store)
    assert before == D("29888.1")
    lifecycle.move(lifecycle.ACTIVE, lifecycle.RETIRED, lifecycle.PM, flat=True)
    store.set_desired_state("spot", "stopped")
    store.archive("spot")
    assert "spot" not in store.reset_runs()
    assert store.withdrawn() == {"spot": D("9888.1")}
    assert fund_equity(store) == before - D("9888.1") == D(20_000)


def test_dust_with_no_mark_is_held_at_its_entry_price_and_a_real_position_still_fails_closed(store):
    """HoQA F229-7: a position under DUST_NOTIONAL, never marked, doesn't fail the fund closed (it would block every
    entry over dust no order can close); it counts at its entry price. A real one with no mark still raises."""
    _trade(store, "spot", "BUY", 0.00001, 60_000.0, 0.0, 1, T0)  # 0.6 of notional: dust
    assert fund_equity(store) == D(30000)  # 10,000 - 0.6 + 0.00001 x 60,000, plus the two flat strategies
    _trade(store, "perp", "SELL", 0.2, 60_000.0, 4.8, 1, T0)
    with pytest.raises(RuntimeError, match="perp holds .* no mark"):
        fund_equity(store)
