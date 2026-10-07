"""The journal's exact columns and its two float edges (DA-9). The column type (EXACT, NUMERIC(38, 18)); the ONE
place the paper engine's float figures become exact (EngineJournal, Advisor 7 Oct 20:32 UK); and the read-only float
view for code that computes in floats (FloatView: the dashboard, the supervisor, the mirror, replay). The arithmetic
rules are sleeve_fund.money's."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Decimal

from sqlalchemy import Numeric
from sqlalchemy.types import TypeDecorator

from sleeve_fund.money import PRECISION, SCALE, as_floats, stored, to_decimal


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

    def exact(self, kind: str, x, what: str, sleeve=None, order_id=None) -> Decimal:
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
        on = f" on order {order_id}" if order_id else ""
        why = (f"The engine's {what} {x!r}{on} is not a whole number of lots ({step}). The journal never rounds a "
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
            order_id = args[0] if args and name == "update_order" else kwargs.get("order_id")
            for k, kind in fields.items():
                if kwargs.get(k) is not None:
                    d = self.exact(kind, kwargs[k], k, sleeve, order_id)
                    kwargs[k] = float(d) if floats else d  # a float prints as d, so a saved copy is d again
            return write(*args, **kwargs)

        return call
