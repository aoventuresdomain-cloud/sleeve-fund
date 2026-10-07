"""What the portfolio's book is (v2 P2-2, Advisor 3, [R-BOOK]): the whole fund's marked equity, every strategy's
plus unallocated cash, never only the running ones. A single-strategy backtest has no other strategies, so its book
is its equity over its allocation share, as if the others were flat, and it says so."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal

from sleeve_fund.money import money


def book_equity(strategy_equities: Iterable, unallocated) -> Decimal:
    """The fund's book: every strategy's marked equity, running or not, plus the cash no strategy holds."""
    return sum((money(e, "strategy equity") for e in strategy_equities), money(unallocated, "unallocated"))


def backtest_book(strategy_equity, allocation_share: float) -> tuple[Decimal, str]:
    """A single-strategy backtest's book and its label: the strategy's equity over its share of the fund."""
    if not 0 < allocation_share <= 1:
        raise ValueError(f"allocation share is in (0, 1], not {allocation_share!r}")
    book = money(strategy_equity, "strategy equity") / Decimal(repr(float(allocation_share)))
    return book, f"book = strategy equity / {allocation_share:.0%} allocation (others flat)"
