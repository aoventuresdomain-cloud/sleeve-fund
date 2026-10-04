"""Instrument definitions with the venue fee schedule baked in.

Fees and venue are mandatory: there is no way to build an instrument here without them, and no
venue is assumed. Each venue's rates live in its profile (sleeve_fund.venues).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from nautilus_trader.execution import FeeModel
from nautilus_trader.model import Currency, CurrencyPair, InstrumentId, Money, Price, Quantity, Symbol, Venue

@dataclass(frozen=True)
class FeeSchedule:
    maker: Decimal
    taker: Decimal

    def __post_init__(self) -> None:
        for name, value in (("maker", self.maker), ("taker", self.taker)):
            if not Decimal("0") <= value < Decimal("0.05"):
                raise ValueError(f"{name} fee {value} outside sane range [0, 5%)")


def spot_pair(
    base: str,
    quote: str,
    fees: FeeSchedule,
    venue: Venue,
    price_precision: int = 2,
    size_precision: int = 8,
) -> CurrencyPair:
    """Build a spot CurrencyPair. Use VenueProfile.instrument() so the venue's fees come with it."""
    base_ccy = Currency.from_str(base)
    quote_ccy = Currency.from_str(quote)
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
        # Paper only: maker_cap(order, fill_qty) -> (the part a backtest would fill at the maker fee, half
        # the spread now). Paper's simulated venue fills a post-only order whole once the price trades
        # through it; a backtest fills at most BOOK_SHARE of the volume that traded through. The rest is
        # charged as the market order it would have been: the taker fee and half the spread (review
        # round 8, M8-5). Set by the paper node to the strategy's LongFlatStrategy.maker_allowance.
        self.maker_cap = None
        # Paper only: fees to give back on the next fill, from post-only orders that a backtest would have
        # filled at the maker fee by the end of their wait (LongFlatStrategy._settle_maker).
        self.pending_credit = 0.0

    def rate_for(self, order) -> Decimal:
        return self.fees.maker if getattr(order, "is_post_only", False) else self.fees.taker

    def get_commission(self, order, fill_quantity, fill_px, instrument) -> Money:
        notional = fill_quantity.as_decimal() * fill_px.as_decimal()
        credit, self.pending_credit = Decimal(str(self.pending_credit)), 0.0
        if self.maker_cap is not None and getattr(order, "is_post_only", False):
            maker_qty, half = self.maker_cap(order, fill_quantity.as_double())
            maker = min(Decimal(str(maker_qty)), fill_quantity.as_decimal())
            taker = (fill_quantity.as_decimal() - maker) * fill_px.as_decimal()
            charge = maker * fill_px.as_decimal() * self.fees.maker + taker * (self.fees.taker + Decimal(str(half)))
            return Money(float(charge - credit), instrument.quote_currency)
        charge = notional * self.rate_for(order) - credit
        if self.half_spread and not getattr(order, "is_post_only", False):
            spread = notional * self.half_spread
            coid = str(order.client_order_id)
            self.spread_paid[coid] = self.spread_paid.get(coid, 0.0) + float(spread)
            charge += spread
        return Money(float(charge), instrument.quote_currency)


def fill_model():
    """How resting limit orders fill, the same in backtests and paper: only when the price trades
    through the limit. A price that merely touches it gets no fill, since a real order joining the
    queue at that price is behind everyone already there."""
    from nautilus_trader.execution import DefaultFillModel

    return DefaultFillModel(prob_fill_on_limit=0.0, prob_slippage=0.0)


# The share of the volume that trades through a resting order's price which that order may take in a
# backtest (sleeve_fund.research.runner). Paper's simulated venue fills a post-only order in full once
# the price trades through, so paper charges the maker fee on this share only and the rest as a market
# order (ScheduleFeeModel.maker_cap, LongFlatStrategy.maker_allowance).
BOOK_SHARE = 0.2
