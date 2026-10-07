"""Money in the phase 2 cores (day-0 interface note v4, the Data Architect's Decimal rules for DA-9).

- Money and quantity are `Decimal`, built from strings, ints or NUMERIC, never from a float: a float passed for money
  is refused with a TypeError, so float money never gets in silently.
- NaN and Infinity are refused wherever money enters a core.
- A ratio (a float: a rate, share, weight or multiple) times money goes through `scale`, the one place that turns a
  ratio into a Decimal. No core calls `Decimal(ratio)` itself.
- The global decimal context is never touched; nothing here rounds. Money is quantised only where it is stored.
"""

from __future__ import annotations

import math
from decimal import Decimal


def money(value: Decimal | int | str, name: str = "amount") -> Decimal:
    """`value` as a finite Decimal. A float (or bool) is refused with a TypeError; NaN and Infinity with a
    ValueError."""
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, str)):
        raise TypeError(f"{name} is money: pass a Decimal, int or str, not {type(value).__name__}")
    d = value if isinstance(value, Decimal) else Decimal(value)
    if not d.is_finite():
        raise ValueError(f"{name} must be a finite amount, not {d}")
    return d


def scale(amount: Decimal, ratio: float) -> Decimal:
    """amount x ratio, exactly as the ratio is written: Decimal(repr(0.1)) is 0.1, where Decimal(0.1) is
    0.1000000000000000055511151231257827021181583404541015625."""
    amount = money(amount)
    if isinstance(ratio, bool) or not isinstance(ratio, (float, int)) or not math.isfinite(ratio):
        raise ValueError(f"a ratio must be a finite number, not {ratio!r}")
    return amount * Decimal(repr(ratio))
