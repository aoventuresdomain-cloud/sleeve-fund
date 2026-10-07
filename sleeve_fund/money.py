"""Exact money (DA-9). Money and quantities are Decimals built from strings or NUMERIC, never from a float's binary
value: a float that reaches the edge is taken as the decimal it prints as (Decimal(repr(x))), so 0.1 is 0.1.

Stored as NUMERIC(38, 18): 20 places before the point and 18 after. Money is quantised ROUND_HALF_EVEN at that scale
before it is written (Postgres would round half away from zero); a quantity or price on a lot or tick grid has 18
places or fewer and so is never re-rounded. NaN and infinity are refused. The global decimal context is never
touched."""

from __future__ import annotations

import math
import numbers
from decimal import ROUND_HALF_EVEN, Decimal

from sqlalchemy import Numeric
from sqlalchemy.types import TypeDecorator

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


def scale(amount: Decimal, ratio: float) -> Decimal:
    """A ratio (a rate, a share, a fraction) times money: the ratio taken as the decimal it prints as."""
    return to_decimal(amount) * to_decimal(ratio, "ratio")


class _Exact(TypeDecorator):
    """NUMERIC(38, 18) read and written as exact Decimals on every database. SQLite has no exact decimal type and
    keeps a REAL: its value comes back as the decimal the float prints as, not 18 places of binary noise."""

    impl = Numeric(PRECISION, SCALE)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "sqlite":
            return dialect.type_descriptor(Numeric(PRECISION, SCALE, asdecimal=False))
        return dialect.type_descriptor(Numeric(PRECISION, SCALE))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        d = stored(value)
        return float(d) if dialect.name == "sqlite" else d

    def process_result_value(self, value, dialect):
        return None if value is None else to_decimal(value)


# The column type: plain NUMERIC(38, 18) in the schema (what a migration and the drift check compare), exact on read
# and write on each database the store runs on.
EXACT = Numeric(PRECISION, SCALE).with_variant(_Exact(), "postgresql", "sqlite")


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


class FloatView:
    """A journal read as floats, for code that computes in floats (the dashboard, the supervisor, the mirror). It
    only changes what is read: every Decimal in a result comes back as a float. A write passes through to the store
    untouched, and the store takes it exactly (to_decimal)."""

    def __init__(self, store) -> None:
        self._exact = store

    def __getattr__(self, name: str):
        attr = getattr(self._exact, name)
        if not callable(attr):
            return attr  # the store's own state (an engine, a url, a backtest's lists), as it is

        def call(*args, **kwargs):
            return as_floats(attr(*args, **kwargs))

        return call


def float_view(store):
    """store read as floats (FloatView), once however often it is asked for."""
    return store if isinstance(store, FloatView) else FloatView(store)


# The engine's money writes, and the grid each figure lies on: "lot" a venue's order or fill quantity, "money" anything
# else. A mark's or a funding row's quantity is the engine's float position, a sum of fills, kept exact as it is. Prices are kept
# exact rather than on the tick: a backtest books its taker fills at the bid or ask with the half spread in the price
# (Advisor 6 Oct 20:55) and a stop's fill with its slippage (P1-D13), neither on the tick, and a mark or a mid may fall
# between ticks.
ENGINE_WRITES = {
    "record_fill": {"qty": "lot", "price": "money", "fee": "money"},
    "book_fill": {"qty": "lot", "price": "money", "fee": "money"},
    "record_order": {"qty": "lot"},
    "update_order": {"qty": "lot", "fill_qty": "lot", "fill_px": "money", "fee": "money"},
    "record_equity": {"equity": "money", "cash": "money", "qty": "money", "price": "money", "benchmark": "money"},
    "record_funding": {"qty": "money", "price": "money", "amount": "money"},
    "record_insurance": {"price": "money", "amount": "money"},
    "set_exit_plan": {"risk_amount": "money"},
}


LOT_NOISE_ULPS = 4  # float steps a venue quantity can be off its lot by once read as a float


class EngineJournal(FloatView):
    """The paper engine's journal (DA-9, Advisor 7 Oct 20:32 UK). The engine computes in floats; this is the ONE
    place its figures become exact: each money or quantity argument of an ENGINE_WRITES call is taken as
    Decimal(str(x)), and a quantity must lie on the instrument's lot once the strategy has set it (set_grid): one that
    doesn't is refused loudly, never rounded. Reads are the float view's."""

    def __init__(self, store) -> None:
        super().__init__(store)
        self._grid: dict[str, Decimal] = {}

    def set_grid(self, *, lot_decimals: int | None) -> None:
        """The instrument's size decimals, from the strategy's instrument as it starts."""
        self._grid = {"lot": Decimal(1).scaleb(-lot_decimals)} if lot_decimals is not None else {}

    def exact(self, kind: str, x, what: str, sleeve=None) -> Decimal:
        """x as the journal books it. A quantity must already lie on the lot (Advisor 21:17 UK). A float within a few
        float steps of a lot multiple is that multiple: the venue's quantity, read as a float (303.82504499 can
        read 303.82504499000004), so the engine's quantity is unchanged. Anything further off is never rounded: it is
        refused with ValueError, and an error event says so."""
        d = to_decimal(x, what)
        step = self._grid.get(kind)
        if step is None:
            return d
        q = d.quantize(step, rounding=ROUND_HALF_EVEN)
        if q == d or (isinstance(x, float) and abs(q - d) <= LOT_NOISE_ULPS * Decimal(math.ulp(x))):
            return q
        why = (f"The engine's {what} {x!r} is not a whole number of lots ({step}). The journal never rounds a "
               "quantity, so this write was refused; check the order against the venue")
        try:
            self._exact.event(sleeve, "error", "qty_off_lot", why)
        except Exception:  # noqa: BLE001 - the refusal below is what matters; the journal may be what failed
            pass
        raise ValueError(why)

    def __getattr__(self, name: str):
        fields = ENGINE_WRITES.get(name)
        if fields is None:
            return super().__getattr__(name)
        write = getattr(self._exact, name)

        floats = getattr(self._exact, "keeps_floats", False)  # a backtest's in-memory journal

        def call(*args, **kwargs):
            sleeve = args[0] if args and name != "update_order" else None
            for k, kind in fields.items():
                if kwargs.get(k) is not None:
                    d = self.exact(kind, kwargs[k], k, sleeve)
                    kwargs[k] = float(d) if floats else d  # a float prints as d, so a saved copy is d again
            return write(*args, **kwargs)

        return call
