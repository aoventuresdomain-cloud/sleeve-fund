"""P2-8 two legs, the store: a two-leg Strategy is made as two rows in one transaction, its balance split by what each
leg needs, a short spot leg refused (Advisor 13 Oct 23:55 UK), both legs' orders carry one trade id, and the
supervisor never runs the second leg's row as a Strategy of its own."""

import pytest

from sleeve_fund import two_legs
from sleeve_fund.store import Store
from sleeve_fund.supervisor import Supervisor

SPOT = {"instrument": "BTC/USDT", "params": {}, "venue": "KRAKEN", "side": 1}
PERP = {"instrument": "BTC/USDT", "params": {"market": "perp"}, "venue": "KRAKEN", "side": -1}


def _store(tmp_path):
    return Store(f"sqlite:///{tmp_path}/legs.db")


def _carry(store, **kw):
    return store.create_two_legs(name="carry", strategy="buy_and_hold", bar_spec="1-DAY-LAST-EXTERNAL",
                                 starting_balance=10_000, first=kw.get("first", SPOT), second=kw.get("second", PERP))


def test_a_carry_is_two_rows_with_the_balance_split_by_what_each_leg_needs(tmp_path):
    store = _store(tmp_path)
    one, two = _carry(store)
    assert (one.name, one.parent, two.name, two.parent) == ("carry", None, "carry~2", "carry")
    assert (float(one.starting_balance), float(two.starting_balance)) == (5_000.0, 5_000.0)  # spot 1, perp 1 at 1x
    assert two.params["market"] == "perp" and two.params["leg"] == {"side": -1, "ratio": "1", "leverage": "1"}
    assert [d["action"] for d in store.decisions("carry")] == ["two_legs"]


def test_the_split_follows_ratio_and_leverage_and_adds_up_to_the_cent():
    D = two_legs.Decimal
    assert two_legs.split_capital(10_000, (SPOT, {**PERP, "leverage": 2})) == (D("6666.66"), D("3333.34"))
    a, b = two_legs.split_capital("1000.01", (SPOT, {**PERP, "ratio": "0.5"}))
    assert (a, b) == (D("666.67"), D("333.34")) and a + b == D("1000.01")


@pytest.mark.parametrize("second, why", [
    ({**PERP, "params": {}}, "a short leg must be a perp"),
    ({**PERP, "leverage": 3}, "from 1x to 2x"),
    ({**PERP, "venue": "BINANCE"}, "one venue"),
    ({**PERP, "ratio": 0}, "above 0"),
    ({**SPOT}, "same instrument on the same market"),
])
def test_a_pair_that_isnt_allowed_is_refused_and_nothing_is_made(tmp_path, second, why):
    store = _store(tmp_path)
    with pytest.raises(two_legs.TwoLegError, match=why):
        _carry(store, second=second)
    assert store.sleeves() == []


def test_both_legs_are_made_or_neither(tmp_path):
    store = _store(tmp_path)
    _carry(store)
    with pytest.raises(Exception):
        _carry(store)  # the names are taken: the second attempt makes nothing
    assert [s.name for s in store.sleeves()] == ["carry", "carry~2"]


def test_both_legs_orders_carry_one_trade_id(tmp_path):
    store = _store(tmp_path)
    _carry(store)
    for leg, oid in (("carry", "a1"), ("carry~2", "b1")):
        store.record_order(leg, order_id=oid, side="BUY", qty=0.1, intent="entry", reason="r", parent_trade_id="t1")
    assert {o["sleeve"] for o in store.orders(None) if o.get("parent_trade_id") == "t1"} == {"carry", "carry~2"}


def test_listings_can_leave_the_second_leg_out_and_the_supervisor_never_runs_it(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _carry(store)
    assert [s.name for s in store.sleeves(include_legs=False)] == ["carry"]
    started = []
    sup = Supervisor(store)
    monkeypatch.setattr(sup, "_start", lambda name, proc: started.append(name))
    sup.step()
    assert started == ["carry"]
