"""F-ZERO-START (PE1, 15 Oct ~00:20 UK): once the fund holds the cash every Strategy is made with 0 (CUT-A (a)), and the
paper node builds its config from the stored row (paper.config.from_store). A 0 there was refused ("starting_balance
must be positive"), so after the PM's reset no Strategy made with 0 could start; the simulated account is funded from
the fund (paper.node.virtual_funding). F-PAIR-IN-FUND (HoE MUST): a two-leg Strategy is made with 0 inside the fund
too, both legs at 0; outside the fund its balance must still be above 0."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sleeve_fund.paper.config import from_store
from sleeve_fund.store import Store
from sleeve_fund.two_legs import TwoLegError

PERP = {"market": "perp", "allow_short": True, "stop_loss": 0.02}


def _fund(store):
    at = datetime.now(UTC) - timedelta(minutes=5)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="test", rebase=True, now=at), at)


def _pair(store, balance):
    return store.create_two_legs(
        name="carry", strategy="ping_pong", bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=balance,
        first={"instrument": "BTC/USDT", "params": PERP, "side": 1, "ratio": 1, "leverage": 1},
        second={"instrument": "ETH/USDT", "params": PERP, "side": -1, "ratio": 1, "leverage": 1})


def test_a_strategy_made_with_0_in_the_fund_builds_its_paper_config():
    store = Store.in_memory()
    _fund(store)
    store.create_sleeve(name="one", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, params=PERP)
    assert from_store(store.sleeve("one")).starting_balance == 0


def test_a_balance_below_0_is_still_refused():
    store = Store.in_memory()
    store.create_sleeve(name="one", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=1_000, params=PERP)
    row = store.sleeve("one")
    with pytest.raises(ValueError, match="below 0"):
        from_store(type(row)(**{**row.__dict__, "starting_balance": -1})
                   if hasattr(row, "__dict__") else row._replace(starting_balance=-1))


def test_a_two_leg_strategy_is_made_with_0_in_the_fund_both_legs_at_0():
    store = Store.in_memory()
    _fund(store)
    first, second = _pair(store, 0)
    assert (first.starting_balance, second.starting_balance) == (0, 0)
    assert second.parent == "carry"
    assert from_store(first).starting_balance == from_store(second).starting_balance == 0


def test_outside_the_fund_a_two_leg_strategy_still_needs_a_balance():
    with pytest.raises(TwoLegError, match="above 0"):
        _pair(Store.in_memory(), 0)
