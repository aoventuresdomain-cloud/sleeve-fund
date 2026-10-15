"""Multi-asset Strategies: same-close ideas are gated one at a time in a fixed order (multi-asset spec §4 and §8 cells 2,
3 and 9; Advisor MA-SPEC). A Strategy's legs are taken together, strongest idea first, then by the asset's place in its
list: never by which feed's bar came first. Each idea is checked against the book plus the earlier ideas' room."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sleeve_fund.research.portfolio import CloseBatch, Pending, strategy_of

TS = 1_000_000_000


class Clock:
    def set_time_alert_ns(self, name, at, callback):
        pass


def _run(order, posts, rotate=False, room=None):
    """posts: (strategy, strength) in arrival order. room: units the book has; each approval takes 1 (a reservation),
    so a later idea sees the earlier ones'. Returns [(strategy, approved)] in gate order."""
    left = [room]
    seen = []

    def gate(strategy, intent, ts):
        ok = left[0] is None or left[0] >= 1
        if ok and left[0] is not None:
            left[0] -= 1
        seen.append((strategy, 1 if ok else 0))
        return SimpleNamespace(approved_qty=1 if ok else 0)

    batch = CloseBatch(tuple(order), gate, rotate=rotate)
    for name, strength in posts:
        batch.post(Clock(), TS, Pending(name, {}, lambda i, d: None, lambda i, d: None, strength))
    batch.run(TS)
    return seen


LEGS = ("t", "t~ETH", "t~SOL")


def test_equal_strength_falls_back_to_the_asset_list_whatever_arrived_first():
    assert [s for s, _ in _run(LEGS, [("t~SOL", 0), ("t", 0), ("t~ETH", 0)])] == list(LEGS)


def test_the_strongest_idea_is_gated_first():
    assert [s for s, _ in _run(LEGS, [("t", 0.5), ("t~ETH", 2.0), ("t~SOL", 1.0)])] == ["t~ETH", "t~SOL", "t"]


def test_with_room_for_one_the_stronger_is_approved_and_the_other_refused():
    assert _run(LEGS, [("t~SOL", 1.0), ("t~ETH", 3.0)], room=1) == [("t~ETH", 1), ("t~SOL", 0)]


def test_other_strategies_keep_their_place_and_the_legs_go_together():
    order = ("a", "t", "b", "t~ETH")
    got = [s for s, _ in _run(order, [("t~ETH", 5.0), ("b", 0), ("t", 0), ("a", 0)])]
    assert got == ["a", "t~ETH", "t", "b"]


def test_rotation_moves_strategies_never_a_strategys_own_asset_order():
    for ts_order in range(5):
        got = [s for s, _ in _run(("a",) + LEGS, [("t~SOL", 0), ("t~ETH", 0), ("t", 0), ("a", 0)], rotate=True)]
        legs = [s for s in got if strategy_of(s) == "t"]
        assert legs == list(LEGS), got


@pytest.mark.parametrize("name, strategy", [("t~ETH", "t"), ("carry~2", "carry~2"), ("one", "one")])
def test_a_leg_belongs_to_its_strategy_and_a_pair_leg_to_none(name, strategy):
    assert strategy_of(name) == strategy
