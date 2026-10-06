"""Instrument definitions with the venue fee schedule baked in.

Fees and venue are mandatory: there is no way to build an instrument here without them, and no
venue is assumed. Each venue's rates live in its profile (sleeve_fund.venues).
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

from nautilus_trader.execution import FeeModel
from nautilus_trader.model import (Currency, CryptoPerpetual, CurrencyPair, InstrumentId, Money, Price, Quantity, Symbol,
                                   Venue)

@dataclass(frozen=True)
class FeeSchedule:
    maker: Decimal
    taker: Decimal

    def __post_init__(self) -> None:
        for name, value in (("maker", self.maker), ("taker", self.taker)):
            if not Decimal("0") <= value < Decimal("0.05"):
                raise ValueError(f"{name} fee {value} outside sane range [0, 5%)")


def lot_decimals(instrument) -> int:
    """Decimals an order size can carry: the instrument's, but no more than the account holds the base
    currency in. Nautilus keeps XRP and ADA at 6 decimals while venues list 8 lot decimals, so an
    8-decimal buy is held as 6 and a full exit leaves the difference in the journal (review round 9, B9-1)."""
    return min(instrument.size_precision, instrument.base_currency.precision)


MAX_PRICE_DECIMALS = 9
MAX_TICK_SHARE = 0.0005  # a price step coarser than 0.05% of the price distorts stops and fills


def price_decimals(price: float) -> int:
    """Price decimals for an instrument built from stored history, where the venue's own increment isn't
    to hand: 2 at 100 or more, 4 from 1, else at least 6, and enough that a step is at most about 0.01% of
    the price (7 at $0.005, 9 at $0.00002). The 2/4/6 rule alone left sub-cent instruments on a step of 5% of
    the price (review rounds 9 and 10, B9-2, B10-2). Studies, the backtest page and the tables share it."""
    if price >= 100:
        return 2
    if price >= 1:
        return 4
    if price <= 0 or price != price:
        return 6
    return min(MAX_PRICE_DECIMALS, max(6, 4 - math.floor(math.log10(price))))


def history_price_decimals(closes) -> int:
    """Decimals for an instrument built from these closes: set by the lowest, so the cheapest stretch is
    still priced finely. Refuses prices so low that even the finest step the engine keeps is coarser than
    MAX_TICK_SHARE of the price, since stops and fills there would be noise (review round 10, B10-2)."""
    low = float(closes[closes > 0].min())
    d = price_decimals(low)
    if 10 ** -d / low > MAX_TICK_SHARE:
        raise ValueError(f"prices as low as {low:.3g} can't be tested: the finest price step kept "
                         f"({10 ** -d:g}) is {10 ** -d / low:.2%} of the price")
    return d


def spot_pair(
    base: str,
    quote: str,
    fees: FeeSchedule,
    venue: Venue,
    price_precision: int = 2,
    size_precision: int = 8,
) -> CurrencyPair:
    """Build a spot CurrencyPair. Use VenueProfile.instrument() so the venue's fees come with it.

    Sizes never carry more decimals than the engine keeps the base currency in (see lot_decimals)."""
    base_ccy = Currency.from_str(base)
    quote_ccy = Currency.from_str(quote)
    size_precision = min(size_precision, base_ccy.precision)
    symbol = Symbol(f"{base}/{quote}")
    return CurrencyPair(
        instrument_id=InstrumentId(symbol=symbol, venue=venue),
        raw_symbol=symbol,
        base_currency=base_ccy,
        quote_currency=quote_ccy,
        price_precision=price_precision,
        size_precision=size_precision,
        price_increment=Price(10**-price_precision, precision=price_precision),
        size_increment=Quantity(10**-size_precision, precision=size_precision),
        min_quantity=Quantity(10**-size_precision, precision=size_precision),
        min_notional=Money(1, quote_ccy),
        min_price=Price(10**-price_precision, precision=price_precision),
        margin_init=Decimal(0),
        margin_maint=Decimal(0),
        maker_fee=fees.maker,
        taker_fee=fees.taker,
        ts_event=0,
        ts_init=0,
    )


def perpetual(
    base: str,
    quote: str,
    fees: FeeSchedule,
    venue: Venue,
    symbol: str,
    price_precision: int = 2,
    size_precision: int = 3,
    min_quantity: float | None = None,
    min_notional: float = 5.0,
) -> CryptoPerpetual:
    """Build a linear (quote-settled) perpetual as the venue lists it, under the venue's own symbol, so a
    backtest trades the same instrument paper receives from the venue's data client. Use
    VenueProfile.instrument(), which fills in the venue's symbol, fees and contract limits."""
    base_ccy, quote_ccy = Currency.from_str(base), Currency.from_str(quote)
    size_precision = min(size_precision, base_ccy.precision)
    step = 10**-size_precision
    sym = Symbol(symbol)
    return CryptoPerpetual(
        instrument_id=InstrumentId(symbol=sym, venue=venue),
        raw_symbol=sym,
        base_currency=base_ccy,
        quote_currency=quote_ccy,
        settlement_currency=quote_ccy,
        is_inverse=False,
        price_precision=price_precision,
        size_precision=size_precision,
        price_increment=Price(10**-price_precision, precision=price_precision),
        size_increment=Quantity(step, precision=size_precision),
        min_quantity=Quantity(max(min_quantity or step, step), precision=size_precision),
        min_notional=Money(min_notional, quote_ccy),
        min_price=Price(10**-price_precision, precision=price_precision),
        margin_init=Decimal(0),
        margin_maint=Decimal(0),
        maker_fee=fees.maker,
        taker_fee=fees.taker,
        ts_event=0,
        ts_init=0,
    )


def pair_of(instrument) -> str:
    """BASE/QUOTE for an instrument, whatever the venue calls it (BTCUSDT-PERP is BTC/USDT)."""
    symbol = str(instrument.id.symbol)
    return symbol if "/" in symbol else f"{instrument.base_currency.code}/{instrument.quote_currency.code}"


# A stop's slippage in a backtest: the larger of half the spread and this, IN PLACE of the half spread (Advisor, P1-D13
# 18:36). A stop is a market order sent into a moving price, so it pays at least this even where the quoted spread is
# tighter. Paper books its real fills; this is the backtest's model of them.
STOP_SLIPPAGE_FLOOR = 0.0005


def stop_slippage(half_spread: float) -> float:
    return max(float(half_spread), STOP_SLIPPAGE_FLOOR)


class ExecBars:
    """The bars a backtest's venue matched orders against, looked up by their close time, for booking a resting
    exit's fill when the bar alone can't say what traded first (ScheduleFeeModel.exit_price)."""

    def __init__(self, df) -> None:
        import numpy as np

        idx = df.index if df.index.tz is None else df.index.tz_convert("UTC")
        self.ts = idx.as_unit("ns").asi8  # nanoseconds, as the engine's clock reads, whatever unit the index holds
        self.ohlc = np.column_stack([df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close")])

    def at(self, ts_ns: int) -> tuple[float, float, float, float] | None:
        import numpy as np

        i = int(np.searchsorted(self.ts, ts_ns))
        return tuple(self.ohlc[i]) if i < len(self.ts) and self.ts[i] == ts_ns else None


def exit_price(kind: str, side: int, trigger: float, bar: tuple, rested: bool, stop: float | None = None
               ) -> tuple[str, float]:
    """Where a resting exit that traded inside `bar` (open, high, low, close) is booked, before a stop's slippage, when
    only the bar is known (bars only, or execution bars longer than a minute): (what it is booked as, price). Pessimistic
    (Advisor, P1-D13 18:16 and 18:36):
    - a stop (side: the position's, 1 long) fills at the open when the price gapped through it there, else at the bar's
      adverse extreme (its low for a long), never at its trigger: inside the bar the price may have run on past it;
    - a target fills at its trigger, never better, also when the open had passed it;
    - with the target AND the stop (`stop`, its trigger) both inside the bar, the stop came first unless the open was
      already past the target (adverse first, NA-2): the target's fill is then booked as the stop.
    `rested`: the order rested before the bar opened. One placed inside the bar (its entry filled there) can't have
    gapped: it is stopped out at the extreme."""
    o, h, low, _ = bar
    adverse = low if side > 0 else h

    def stopped(level: float) -> tuple[str, float]:
        gap = rested and (o <= level if side > 0 else o >= level)
        return "stop", (o if gap else adverse)

    if kind == "target":
        open_past = o > trigger if side > 0 else o < trigger
        if stop is not None and not open_past and (low <= stop if side > 0 else h >= stop):
            return stopped(stop)
        return "target", trigger
    return stopped(trigger)


class ScheduleFeeModel(FeeModel):
    """Charges a FeeSchedule on every fill, regardless of what the instrument says.

    Used in both backtest and paper so the fee code path is identical. Market
    orders pay taker; anything else pays taker too unless it is post-only, which
    is the only way to be sure of the maker rate.
    """

    def __init__(self, fees: FeeSchedule, half_spread: float = 0.0) -> None:
        super().__init__()
        self.fees = fees
        # Backtests only (their bars carry trade prices, not quotes): half the bid-ask spread, charged
        # with the commission on fills that take liquidity, and kept apart per order so reports can
        # show the venue's fee and the spread separately. Paper fills on real quotes and passes 0.
        self.half_spread = Decimal(str(half_spread))
        self.spread_paid: dict[str, float] = {}
        # The venue's own fee on those same orders, unrounded, so a report can show it alone to the cent and put
        # what rounding left in the charged commission into the price with the spread (QA m-G7).
        self.fee_paid: dict[str, float] = {}
        # Paper only: the market orders that carry a post-only order's maker fills, each with the order's
        # limit price and side (LongFlatStrategy._slice_maker). Paper's simulated venue would fill a
        # post-only order whole on the first trade through its price; a backtest fills BOOK_SHARE of
        # what trades through, a slice at a time. So paper keeps the order itself and sends each slice
        # at market as it earns it, and this charges the slice as filled at the limit with the maker fee:
        # the commission carries the difference from the price the venue filled at (review round 9, M9-3).
        self.maker_slices: dict[str, tuple[Decimal, bool]] = {}
        # Paper on a perp: the order that puts a position carried over a restart back at the simulated
        # venue (LongFlatStrategy._send_restore). Not a trade, so it pays nothing.
        self.free_orders: set[str] = set()
        # The account keeps the quote currency to its own decimals (USD to the cent), so each fee is rounded.
        # Rounding every one alone charged equal small fills the same way: $1.17 slices paid nothing and
        # $2.44 slices 0.41% (sanity S-1). The rounding left over is carried into the next fee instead, so
        # the total charged stays within a cent of the schedule however small the fills.
        self._carry = Decimal(0)
        # Backtests: how each resting exit's fill is booked. exit_info(order) (the strategy's) says what an order is:
        # {"kind": "stop" | "target", "side": the position's, "trigger", "rested", "stop"} or, for an exit the strategy
        # sent at market to be booked elsewhere, {"kind": "stop", "side", "base"}; None for anything else. Every stop
        # pays stop_slippage in place of the half spread, and a target no half spread (a resting limit, Advisor 19:00).
        # With `bars` (bars only, or execution bars longer than a minute) a stop or target is booked at exit_price
        # against the bar it traded in; `now` gives that bar's close time.
        self.exit_info = None
        self.bars: ExecBars | None = None
        self.now = None
        self.rebooked: dict[str, str] = {}  # target fills booked as the stop (adverse first), by order
        self.booked: dict[str, tuple[float, float]] = {}  # each exit's last fill: (price booked, venue fee), for the journal
        self.intrabar: set[str] = set()  # "fill" / "liq": a booking relied on the order inside a bar (labels)

    def _charge(self, exact: Decimal, currency) -> Money:
        total = exact + self._carry
        charged = total.quantize(Decimal(10) ** -currency.precision, rounding=ROUND_HALF_EVEN)
        self._carry = total - charged
        return Money(charged, currency)

    def rate_for(self, order) -> Decimal:
        return self.fees.maker if getattr(order, "is_post_only", False) else self.fees.taker

    def get_commission(self, order, fill_quantity, fill_px, instrument) -> Money:
        if str(order.client_order_id) in self.free_orders:
            return Money(0, instrument.quote_currency)
        notional = fill_quantity.as_decimal() * fill_px.as_decimal()
        maker_slice = self.maker_slices.get(str(order.client_order_id))
        if maker_slice is not None:
            limit, buy = maker_slice
            qty = fill_quantity.as_decimal()
            # A buy that filled below its limit pays the difference here, a sell above it gives it back.
            shift = qty * (limit - fill_px.as_decimal()) * (1 if buy else -1)
            return self._charge(qty * limit * self.fees.maker + shift, instrument.quote_currency)
        info = self.exit_info(order) if self.exit_info is not None else None
        if info is not None:
            return self._exit_commission(order, info, fill_quantity, fill_px, instrument)
        charge = notional * self.rate_for(order)
        if self.half_spread and not getattr(order, "is_post_only", False):
            spread = notional * self.half_spread
            coid = str(order.client_order_id)
            self.spread_paid[coid] = self.spread_paid.get(coid, 0.0) + float(spread)
            self.fee_paid[coid] = self.fee_paid.get(coid, 0.0) + float(charge)
            charge += spread
        return self._charge(charge, instrument.quote_currency)

    def _exit_commission(self, order, info: dict, fill_quantity, fill_px, instrument) -> Money:
        """A resting exit's fill, booked at its price (exit_price), the stop's slippage included: the venue's fee is
        charged on that price, and the difference from the venue's fill rides with it, to be moved into the price as
        the half spread is (runner._spread_into_prices). Never booked better than the venue filled it."""
        coid = str(order.client_order_id)
        qty, filled = fill_quantity.as_decimal(), float(fill_px.as_decimal())
        side, kind = info["side"], info["kind"]
        base = info.get("base")
        bar = self.bars.at(self.now()) if self.bars is not None and self.now is not None else None
        if base is not None:
            self.intrabar.add("fill")
        elif bar is not None:
            booked_as, base = exit_price(kind, side, info["trigger"], bar, info.get("rested", True), info.get("stop"))
            if booked_as != kind:
                self.rebooked[coid] = booked_as
            kind = booked_as
            self.intrabar.add("liq" if info.get("liq") else "fill")
        else:
            base = filled if kind == "stop" else info["trigger"]
        px = base * (1 - side * stop_slippage(self.half_spread)) if kind == "stop" else base
        px = min(px, filled) if side > 0 else max(px, filled)  # an exit sells a long: never above the venue's fill
        fee = qty * Decimal(str(px)) * self.rate_for(order)
        moved = qty * Decimal(str((filled - px) * side))  # paid on top of the venue's price, as the spread is
        self.spread_paid[coid] = self.spread_paid.get(coid, 0.0) + float(moved)
        self.fee_paid[coid] = self.fee_paid.get(coid, 0.0) + float(fee)
        self.booked[coid] = (px, float(fee))
        return self._charge(fee + moved, instrument.quote_currency)


def fill_model():
    """How resting limit orders fill, the same in backtests and paper: only when the price trades
    through the limit. A price that merely touches it gets no fill, since a real order joining the
    queue at that price is behind everyone already there."""
    from nautilus_trader.execution import DefaultFillModel

    return DefaultFillModel(prob_fill_on_limit=0.0, prob_slippage=0.0)


# The share of the volume that trades through a resting order's price which that order may take in a
# backtest (sleeve_fund.research.runner), and in paper (LongFlatStrategy._slice_maker).
BOOK_SHARE = 0.2
