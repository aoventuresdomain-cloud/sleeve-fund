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
        # Backtests: a target the strategy judged on a bar the venue had matched, adverse side first (Advisor NA-2,
        # LongFlatStrategy._bar_target), sent at market and booked at its level with that level's price and side.
        # The commission carries the difference from the price the market order filled at; fee_paid keeps the
        # venue's fee apart so the report can move the rest into the price, as it does the spread.
        self.booked: dict[str, tuple[Decimal, bool]] = {}
        # The account keeps the quote currency to its own decimals (USD to the cent), so each fee is rounded.
        # Rounding every one alone charged equal small fills the same way: $1.17 slices paid nothing and
        # $2.44 slices 0.41% (sanity S-1). The rounding left over is carried into the next fee instead, so
        # the total charged stays within a cent of the schedule however small the fills.
        self._carry = Decimal(0)

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
        shift = Decimal(0)
        booked = self.booked.get(str(order.client_order_id))
        if booked is not None:
            level, buy = booked
            qty = fill_quantity.as_decimal()
            # A sell that filled below its level gets the difference back here, a buy above it the same.
            shift = qty * (level - fill_px.as_decimal()) * (1 if buy else -1)
            notional = qty * level
        charge = notional * self.rate_for(order)
        coid = str(order.client_order_id)
        if booked is not None or (self.half_spread and not getattr(order, "is_post_only", False)):
            self.fee_paid[coid] = self.fee_paid.get(coid, 0.0) + float(charge)
        if self.half_spread and not getattr(order, "is_post_only", False):
            spread = notional * self.half_spread
            self.spread_paid[coid] = self.spread_paid.get(coid, 0.0) + float(spread)
            charge += spread
        return self._charge(charge + shift, instrument.quote_currency)


def fill_model():
    """How resting limit orders fill, the same in backtests and paper: only when the price trades
    through the limit. A price that merely touches it gets no fill, since a real order joining the
    queue at that price is behind everyone already there."""
    from nautilus_trader.execution import DefaultFillModel

    return DefaultFillModel(prob_fill_on_limit=0.0, prob_slippage=0.0)


# The share of the volume that trades through a resting order's price which that order may take in a
# backtest (sleeve_fund.research.runner), and in paper (LongFlatStrategy._slice_maker).
BOOK_SHARE = 0.2
