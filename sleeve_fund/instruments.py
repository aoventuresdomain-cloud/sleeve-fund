"""Instrument definitions with the venue fee schedule baked in.

Fees are mandatory: there is no way to build an instrument here without them.
Defaults are Kraken's UK entry tier from 9 July 2026 (0.40% maker, 0.80% taker).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from nautilus_trader.execution import FeeModel
from nautilus_trader.model import Currency, CurrencyPair, InstrumentId, Money, Price, Quantity, Symbol, Venue

KRAKEN = Venue("KRAKEN")


@dataclass(frozen=True)
class FeeSchedule:
    maker: Decimal
    taker: Decimal

    def __post_init__(self) -> None:
        for name, value in (("maker", self.maker), ("taker", self.taker)):
            if not Decimal("0") <= value < Decimal("0.05"):
                raise ValueError(f"{name} fee {value} outside sane range [0, 5%)")


KRAKEN_UK_ENTRY = FeeSchedule(maker=Decimal("0.0040"), taker=Decimal("0.0080"))


def spot_pair(
    base: str,
    quote: str,
    fees: FeeSchedule = KRAKEN_UK_ENTRY,
    venue: Venue = KRAKEN,
    price_precision: int = 2,
    size_precision: int = 8,
) -> CurrencyPair:
    """Build a spot CurrencyPair, e.g. spot_pair("BTC", "USD")."""
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
    is the only way to be sure of maker on Kraken.
    """

    def __init__(self, fees: FeeSchedule = KRAKEN_UK_ENTRY) -> None:
        super().__init__()
        self.fees = fees

    def rate_for(self, order) -> Decimal:
        return self.fees.maker if getattr(order, "is_post_only", False) else self.fees.taker

    def get_commission(self, order, fill_quantity, fill_px, instrument) -> Money:
        notional = fill_quantity.as_decimal() * fill_px.as_decimal()
        return Money(float(notional * self.rate_for(order)), instrument.quote_currency)
