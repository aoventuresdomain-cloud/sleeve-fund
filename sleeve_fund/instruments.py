"""Instrument definitions with the venue fee schedule baked in.

Fees and venue are mandatory: there is no way to build an instrument here without them, and no
venue is assumed. Each venue's rates live in its profile (sleeve_fund.venues).
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

from nautilus_trader.backtest import SimulationModule
from nautilus_trader.execution import FeeModel
from nautilus_trader.model import (Bar, Currency, CryptoPerpetual, CurrencyPair, InstrumentId, Money, OrderSide, Price,
                                   Quantity, Symbol, Venue)

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


# Advisor L12 (6 Oct 19:30): a phase-1 take-profit is a market order on touch, so a taker. A backtest books every
# target fill, touched or gapped through, at its level less the taker's slippage: half the spread, at least 0.05%.
# Never at a better open, never with price improvement. One helper for every backtest target (D13 shares it).
TAKER_SLIPPAGE_FLOOR = Decimal("0.0005")


def taker_slippage(half_spread) -> Decimal:
    """A market order's slippage as a share of the price: half the spread, at least TAKER_SLIPPAGE_FLOOR."""
    return max(Decimal(str(half_spread)), TAKER_SLIPPAGE_FLOOR)


def target_fill_px(level, long: bool, half_spread) -> Decimal:
    """Where a backtest books a take-profit at `level` for a position that is `long` (or short): the level less
    the taker's slippage, exact (not rounded to the tick)."""
    level, slip = Decimal(str(level)), taker_slippage(half_spread)
    return level * (1 - slip) if long else level * (1 + slip)


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


def exit_price(side: int, trigger: float, bar: tuple, rested: bool) -> float:
    """Where a resting stop that traded inside `bar` (open, high, low, close) is booked, before its slippage, when only
    the bar is known (bars longer than a minute, or execution bars longer than a minute). Pessimistic (Advisor, P1-D13
    18:16 and 18:36): at the open when the price gapped through it there, else at the bar's adverse extreme (its low
    for a long; side is the position's), never at its trigger: inside the bar the price may have run on past it.
    `rested`: the stop rested before the bar opened. One placed inside the bar (its entry filled there) can't have
    gapped: it is stopped out at the extreme."""
    o, h, low, _ = bar
    gap = rested and (o <= trigger if side > 0 else o >= trigger)
    return o if gap else (low if side > 0 else h)


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
        # Backtests with measured spreads (SPREAD-PIT): the series BarOpens reads half_spread from at each bar's open,
        # so a fill pays the spread in force when its bar began, never a later measurement.
        self.spread_series = None
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
        # Backtests only: extra slippage on an order, as a share of its price, by client order id (a rule-builder
        # entry on a breakout candle, LongFlatStrategy._submit). Charged and reported as the half spread is.
        self.slippage: dict[str, Decimal] = {}
        # Backtests: a target the strategy judged on a bar the venue had matched, adverse side first (Advisor NA-2,
        # LongFlatStrategy._bar_target), sent at market and booked at its level less the taker's slippage
        # (target_fill_px, Advisor L12), with that price and side.
        # The commission carries the difference from the price the market order filled at; fee_paid keeps the
        # venue's fee apart so the report can move the rest into the price, as it does the spread.
        self.booked: dict[str, tuple[Decimal, bool]] = {}
        # Those targets by their level, priced when the market order fills, with the half spread in force then (SPREAD-PIT:
        # the bar it fills in, not the bar it was decided on); the price goes into booked for the strategy's journal.
        self.booked_targets: dict[str, tuple[Decimal, bool]] = {}
        # Backtests: a resting stop's target level, by the stop's order id. A bar that opens through the target
        # takes the target before anything later in the bar can reach the stop (Advisor NA-2): if the venue fills
        # the stop in such a bar, it is booked as the target (rebooked), at target_fill_px, never the open (L12),
        # instead. bar_open is the open
        # of the bar the venue is matching, handed over before it matches (BarOpens).
        # open_targets: stop id -> (target level, the bar count when the target was first set); a target only counts
        # for bars that opened after it was set, never the bar its entry filled in (CR on #146).
        self.open_targets: dict[str, tuple[Decimal, int]] = {}
        # By stop id, per fill, (target level, booked price): taken by the strategy as it journals the fill.
        self.rebooked: dict[str, tuple[float, float]] = {}
        self.bar_open: Decimal | None = None
        self.bar_seq = 0  # bars the venue has been handed (BarOpens)
        # The account keeps the quote currency to its own decimals (USD to the cent), so each fee is rounded.
        # Rounding every one alone charged equal small fills the same way: $1.17 slices paid nothing and
        # $2.44 slices 0.41% (sanity S-1). The rounding left over is carried into the next fee instead, so
        # the total charged stays within a cent of the schedule however small the fills.
        self._carry = Decimal(0)
        # Backtests: how each resting stop's fill is booked. exit_info(order) (the strategy's) says what an order is:
        # {"kind": "stop", "side": the position's, "trigger", "rested", "liq"} or, for a stop-out the strategy sent at
        # market to be booked elsewhere, {"kind": "stop", "side", "base"}; None for anything else. Every stop pays
        # taker_slippage in place of the half spread. With `bars` (bars or execution bars longer than a minute) it is
        # booked at exit_price against the bar it traded in; `now` gives that bar's close time.
        self.exit_info = None
        self.bars: ExecBars | None = None
        self.now = None
        # Each taker fill's (price booked, the part of its charge moved into that price), in fill order: the venue
        # can match several slices before the strategy hears of the first.
        self.booked_fills: dict[str, list[tuple[float, float]]] = {}
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
        shift = Decimal(0)
        booked = self.booked.get(str(order.client_order_id))
        if booked is None and str(order.client_order_id) in self.booked_targets:
            level, buy = self.booked_targets.pop(str(order.client_order_id))
            booked = self.booked[str(order.client_order_id)] = (target_fill_px(level, not buy, self.half_spread), buy)
        target, since = self.open_targets.get(str(order.client_order_id), (None, None))
        if booked is None and target is not None and self.bar_open is not None and since < self.bar_seq:
            buy = order.side == OrderSide.BUY  # a short's stop buys back; its target sits below
            if (self.bar_open <= target) if buy else (self.bar_open >= target):
                booked = (target_fill_px(target, not buy, self.half_spread), buy)
                self.rebooked[str(order.client_order_id)] = (float(target), float(booked[0]))
        if booked is not None:
            level, buy = booked
            qty = fill_quantity.as_decimal()
            # A sell that filled below its level gets the difference back here, a buy above it the same.
            shift = qty * (level - fill_px.as_decimal()) * (1 if buy else -1)
            notional = qty * level
        if booked is None:
            info = self.exit_info(order) if self.exit_info is not None else None
            if info is not None:
                return self._exit_commission(order, info, fill_quantity, fill_px, instrument)
        coid = str(order.client_order_id)
        paid = self.half_spread + self.slippage.get(coid, 0)  # with a rule-builder order's extra slippage
        taker_spread = booked is None and paid and not getattr(order, "is_post_only", False)
        if taker_spread:
            # Filled at the ask or the bid, mid plus or minus half the spread, in the price (Advisor 20:55): the venue's
            # fee is on that price, the journal and every level derived from the fill take it, and the spread is
            # never charged again as a cost of its own. Any extra slippage the order carries goes the same way.
            filled, buy = float(fill_px.as_decimal()), order.side == OrderSide.BUY
            half = float(paid)
            px = filled * (1 + half) if buy else filled * (1 - half)
            spread = notional * paid
            charge = (notional + spread) * self.rate_for(order) if buy else (notional - spread) * self.rate_for(order)
            self.fee_paid[coid] = self.fee_paid.get(coid, 0.0) + float(charge)
            self.spread_paid[coid] = self.spread_paid.get(coid, 0.0) + float(spread)
            self.booked_fills.setdefault(coid, []).append((px, float(spread)))
            return self._charge(charge + spread, instrument.quote_currency)
        charge = notional * self.rate_for(order)
        if booked is not None:
            self.fee_paid[coid] = self.fee_paid.get(coid, 0.0) + float(charge)
        return self._charge(charge + shift, instrument.quote_currency)

    @staticmethod
    def _liquidation_close(info: dict, filled: float) -> bool:
        """A liquidation: the market close at the liquidation price, or a risk stop filled at or through it."""
        if info.get("kind") == "liquidation" or info.get("liquidation"):
            return True
        liq = info.get("liq_px")
        return liq is not None and (filled <= liq if info["side"] > 0 else filled >= liq)

    def _exit_commission(self, order, info: dict, fill_quantity, fill_px, instrument) -> Money:
        """A resting exit's fill, booked at its price (exit_price), the stop's slippage included: the venue's fee is
        charged on that price, and the difference from the venue's fill rides with it, to be moved into the price as
        the half spread is (runner._spread_into_prices). Never booked better than the venue filled it."""
        coid = str(order.client_order_id)
        qty, filled = fill_quantity.as_decimal(), float(fill_px.as_decimal())
        side = info["side"]
        if self._liquidation_close(info, filled):
            # Advisor 7 Oct 03:23 (D13-F2): a liquidation close, gapped or not, pays no half spread and no 0.05 %
            # floor: booked at the venue's fill, as before D13. Inside a bar's range it still relied on the
            # liquidation check, so the run keeps its label.
            if self.bars is not None and self.now is not None and self.bars.at(self.now()) is not None:
                self.intrabar.add("liq")
            fee = qty * fill_px.as_decimal() * self.rate_for(order)
            self.fee_paid[coid] = self.fee_paid.get(coid, 0.0) + float(fee)
            self.booked_fills.setdefault(coid, []).append((filled, 0.0))
            return self._charge(fee, instrument.quote_currency)
        base = info.get("base")
        bar = self.bars.at(self.now()) if self.bars is not None and self.now is not None else None
        if base is not None:
            self.intrabar.add("fill")
        elif bar is not None:
            base = exit_price(side, info["trigger"], bar, info.get("rested", True))
            self.intrabar.add("liq" if info.get("liq") else "fill")
        else:
            base = filled
        px = base * (1 - side * float(taker_slippage(self.half_spread)))
        px = min(px, filled) if side > 0 else max(px, filled)  # an exit sells a long: never above the venue's fill
        fee = qty * Decimal(str(px)) * self.rate_for(order)
        moved = qty * Decimal(str((filled - px) * side))  # paid on top of the venue's price, as the spread is
        self.spread_paid[coid] = self.spread_paid.get(coid, 0.0) + float(moved)
        self.fee_paid[coid] = self.fee_paid.get(coid, 0.0) + float(fee)
        self.booked_fills.setdefault(coid, []).append((px, float(moved)))
        return self._charge(fee + moved, instrument.quote_currency)


class BarOpens(SimulationModule):
    """Backtests: hands the fee model the open of each bar before the simulated venue matches it, so a resting
    stop filled in a bar that opened through the target is booked at the target (ScheduleFeeModel.open_targets), and
    the half spread in force when the bar opened (ScheduleFeeModel.spread_series)."""

    def __init__(self, fee_model: ScheduleFeeModel) -> None:
        self.fee_model = fee_model

    def pre_process(self, data) -> None:
        if isinstance(data, Bar):
            fm = self.fee_model
            fm.bar_open = data.open.as_decimal()
            fm.bar_seq += 1
            if fm.spread_series is not None:
                opened = int(data.ts_event) - int(data.bar_type.spec.timedelta.total_seconds()) * 1_000_000_000
                now = fm.spread_series.at(opened)
                if now != float(fm.half_spread):
                    fm.half_spread = Decimal(str(now))

    def process(self, ts_now, context):
        return None


def fill_model():
    """How resting limit orders fill, the same in backtests and paper: only when the price trades
    through the limit. A price that merely touches it gets no fill, since a real order joining the
    queue at that price is behind everyone already there."""
    from nautilus_trader.execution import DefaultFillModel

    return DefaultFillModel(prob_fill_on_limit=0.0, prob_slippage=0.0)


# The share of the volume that trades through a resting order's price which that order may take in a
# backtest (sleeve_fund.research.runner), and in paper (LongFlatStrategy._slice_maker).
BOOK_SHARE = 0.2
