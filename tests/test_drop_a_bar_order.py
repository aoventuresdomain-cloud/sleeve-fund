"""BRIDGE-3-QA 2 (Advisor): ideas on one decision mark are gated in an order that rotates per bar by hash(strategy,
bar), so arrival order and the run's listing never decide who gets the room. Ordering only."""

from __future__ import annotations

from collections import Counter

import pytest

from sleeve_fund.research.portfolio import CloseBatch, Pending, bar_order

NAMES = ("a", "b", "c", "d")
BAR = 1_760_000_000_000_000_000
HOUR = 3_600_000_000_000


def test_the_order_is_every_strategy_once_and_the_same_for_the_same_bar():
    assert sorted(bar_order(NAMES, BAR)) == sorted(NAMES)
    assert bar_order(NAMES, BAR) == bar_order(reversed(NAMES), BAR)  # the listing never decides


def test_over_many_bars_every_strategy_is_first_about_as_often():
    firsts = Counter(bar_order(NAMES, BAR + i * HOUR)[0] for i in range(4_000))
    assert set(firsts) == set(NAMES) and min(firsts.values()) > 800  # 1,000 each expected


def test_a_repeated_name_is_refused():
    with pytest.raises(ValueError):
        bar_order(("a", "a"), BAR)


def test_a_rotating_batch_gates_in_the_bars_order_and_keeps_each_strategys_own():
    seen = []
    batch = CloseBatch(NAMES, lambda s, intent, ts: seen.append((s, intent)), rotate=True)
    for s in reversed(NAMES):
        for k in (1, 2):
            batch.pending.setdefault(BAR, []).append(Pending(s, k, lambda *a: None))
    batch.run(BAR)
    order = list(dict.fromkeys(s for s, _ in seen))
    assert order == list(bar_order(NAMES, BAR))
    assert all([k for s2, k in seen if s2 == s] == [1, 2] for s in NAMES)
