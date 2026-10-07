"""Money in the phase 2 cores (day-0 interface note v4, the Data Architect's Decimal rules for DA-9).

- Money and quantity are `Decimal`, built from strings, ints or NUMERIC, never from a float: a float passed for money
  is refused with a TypeError, so float money never gets in silently.
- NaN and Infinity are refused wherever money enters a core.
- A ratio (a float: a rate, share, weight or multiple) times money goes through `scale`, the one place that turns a
  ratio into a Decimal. No core calls `Decimal(ratio)` itself.
- The global decimal context is never touched. Nothing here rounds but `stored`: money is quantised only where it is stored
  (the column's own rounding).

Exact money in the journal (DA-9). Money and quantities are stored as NUMERIC(38, 18): 20 places before the point and
18 after. A float that reaches the journal's edge is taken as the decimal it prints as (Decimal(repr(x))), so 0.1 is
0.1. Money is quantised ROUND_HALF_EVEN at that scale before it is written (Postgres would round half away from zero);
a quantity on a lot grid has 18 places or fewer and so is never re-rounded. The paper engine computes in floats: its
figures become exact at ONE place, EngineJournal (sleeve_fund.exact, with the column type), and code that computes
in floats reads through FloatView. This module stays pure: no database, so the phase 2 cores can import it.
"""

from __future__ import annotations

import math
import numbers
from decimal import ROUND_HALF_EVEN, Decimal


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


PRECISION, SCALE = 38, 18
QUANTUM = Decimal(1).scaleb(-SCALE)


class Money(Decimal):
    """An exact Decimal that also takes a float operand, as the decimal the float prints as (rule 1), so code and
    checks written in floats keep working on the journal's exact figures. Arithmetic with a Money stays a Money;
    comparing with a float compares with the decimal it prints as (Money("0.1") == 0.1)."""

    __slots__ = ()

    @staticmethod
    def _of(other):
        if isinstance(other, float):
            return to_decimal(other)
        return other

    def _wrap(op):
        def method(self, other=None, *rest):
            other = Money._of(other)
            r = op(self, other, *rest) if other is not None or rest else op(self)
            return Money(r) if isinstance(r, Decimal) and not isinstance(r, Money) else r
        return method

    __add__, __radd__ = _wrap(Decimal.__add__), _wrap(Decimal.__radd__)
    __sub__, __rsub__ = _wrap(Decimal.__sub__), _wrap(Decimal.__rsub__)
    __mul__, __rmul__ = _wrap(Decimal.__mul__), _wrap(Decimal.__rmul__)
    __truediv__, __rtruediv__ = _wrap(Decimal.__truediv__), _wrap(Decimal.__rtruediv__)
    __floordiv__, __rfloordiv__ = _wrap(Decimal.__floordiv__), _wrap(Decimal.__rfloordiv__)
    __mod__, __rmod__ = _wrap(Decimal.__mod__), _wrap(Decimal.__rmod__)
    __pow__, __rpow__ = _wrap(Decimal.__pow__), _wrap(Decimal.__rpow__)
    __eq__, __ne__ = _wrap(Decimal.__eq__), _wrap(Decimal.__ne__)
    __lt__, __le__ = _wrap(Decimal.__lt__), _wrap(Decimal.__le__)
    __gt__, __ge__ = _wrap(Decimal.__gt__), _wrap(Decimal.__ge__)
    del _wrap
    __hash__ = Decimal.__hash__

    def __repr__(self) -> str:
        # As a float's repr reads: the decimal itself, so Decimal(repr(x)) and messages work as they did on floats.
        return str(self)

    def __neg__(self):
        return Money(Decimal.__neg__(self))

    def __pos__(self):
        return Money(Decimal.__pos__(self))

    def __abs__(self):
        return Money(Decimal.__abs__(self))

    def __round__(self, n=None):
        r = Decimal.__round__(self, n) if n is not None else Decimal.__round__(self)
        return Money(r) if isinstance(r, Decimal) else r

    def quantize(self, exp, rounding=None, context=None):
        return Money(Decimal.quantize(self, exp, rounding=rounding, context=context))


def to_decimal(x, what: str = "amount") -> Money:
    """x as an exact Decimal (a Money): a Decimal or an int as it is, a str parsed, a float (or a numpy float) as the
    decimal it prints as. Raises TypeError for anything else (a bool included) and ValueError for NaN or infinity."""
    if isinstance(x, Money):
        d = x
    elif isinstance(x, bool) or not isinstance(x, (Decimal, numbers.Real, str)):
        raise TypeError(f"{what} must be a number, not {type(x).__name__}")
    elif isinstance(x, (Decimal, int, str)):
        d = Money(x)
    else:  # float, numpy floats: as printed
        f = float(x)
        if not math.isfinite(f):
            raise ValueError(f"{what} is {x}: not a number a journal can hold")
        return Money(repr(f))
    if not d.is_finite():
        raise ValueError(f"{what} is {d}: not a number a journal can hold")
    return d


def stored(x, what: str = "amount") -> Money:
    """x as the column holds it: exact, rounded half-even to 18 places."""
    return to_decimal(x, what).quantize(QUANTUM, rounding=ROUND_HALF_EVEN)


def as_floats(x):
    """x with every Decimal in it (in dicts, lists and tuples) as a float: for the display edge only."""
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, dict):
        return {k: as_floats(v) for k, v in x.items()}
    if isinstance(x, list):
        return [as_floats(v) for v in x]
    if isinstance(x, tuple) and not hasattr(x, "_fields"):
        return tuple(as_floats(v) for v in x)
    return x
