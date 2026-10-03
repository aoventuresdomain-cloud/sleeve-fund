"""Venue profiles: everything that differs between venues, in one place.

The engine, strategies, runtime and research never name a venue. A sleeve names its venue, and
every mode (research, the backtest page, paper, later live) reads that venue's profile for its fee
schedule, price history, live market data, instrument naming and trading calendar. Adding a venue
means registering a profile here, not changing the engine.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

import pandas as pd
from nautilus_trader.model import CurrencyPair, Venue

from sleeve_fund.instruments import FeeSchedule, spot_pair


@dataclass
class VenueProfile:
    name: str  # the venue id in instrument ids, e.g. KRAKEN in BTC/USD.KRAKEN
    label: str  # how people name it
    fees: FeeSchedule
    fee_basis: str  # where the rates come from, shown beside them
    # pair -> daily candles, closed bars only, indexed by close time
    daily_history: Callable[[str], pd.DataFrame]
    # (pair, minutes) -> candles including the forming one, for charts
    ohlc_history: Callable[[str, int], pd.DataFrame] | None = None
    # (pair, fetch=None) -> (base, quote) as the venue's own instrument data names them
    asset_codes: Callable[..., tuple[str, str]] = lambda pair, fetch=None: tuple(pair.split("/"))  # type: ignore[assignment]
    # () -> (factory, config) for the live market data client; None if paper can't run here yet
    data_client: Callable[[], tuple] | None = None
    calendar: str = "24/7"

    @property
    def venue(self) -> Venue:
        return Venue(self.name)

    def instrument(self, base: str, quote: str, price_precision: int = 2, size_precision: int = 8) -> CurrencyPair:
        """A spot instrument at this venue, carrying this venue's fees."""
        return spot_pair(base, quote, fees=self.fees, venue=self.venue, price_precision=price_precision,
                         size_precision=size_precision)

    def fee_text(self) -> str:
        return f"{self.label}: {float(self.fees.maker):.2%} maker, {float(self.fees.taker):.2%} taker ({self.fee_basis})"


VENUES: dict[str, VenueProfile] = {}
DEFAULT_VENUE = "KRAKEN"


def register(profile: VenueProfile) -> VenueProfile:
    VENUES[profile.name] = profile
    return profile


def venue(name: str | None = None) -> VenueProfile:
    name = (name or DEFAULT_VENUE).upper()
    if name not in VENUES:
        raise ValueError(f"unknown venue {name!r}; known: {sorted(VENUES)}")
    return VENUES[name]


# --- Kraken spot -----------------------------------------------------------------------------


def _kraken_daily(pair: str) -> pd.DataFrame:
    from sleeve_fund.data import fetch_kraken_daily

    return fetch_kraken_daily(pair)


def _kraken_ohlc(pair: str, minutes: int) -> pd.DataFrame:
    from sleeve_fund.data import fetch_kraken_ohlc

    return fetch_kraken_ohlc(pair, minutes)


# Kraken names some assets differently in its instrument data than in the pair (USD is ZUSD,
# BTC is XXBT). The sandbox account must hold cash in the instrument's own quote currency
# or every buy is rejected, so look the codes up from Kraken's public pair list.
ASSET_PAIRS_URL = "https://api.kraken.com/0/public/AssetPairs"
_ALIASES = {"XBT": "BTC", "XDG": "DOGE"}


def _norm(code: str) -> str:
    return _ALIASES.get(code, code)


def kraken_asset_codes(pair: str, fetch=None) -> tuple[str, str]:
    """(base, quote) as Kraken's instrument data names them, e.g. SUI/USD -> (SUI, ZUSD).

    Falls back to the pair's own codes if Kraken can't be reached or doesn't list it.
    """
    base, quote = pair.split("/")
    try:
        if fetch is None:
            with urllib.request.urlopen(ASSET_PAIRS_URL, timeout=15) as r:  # public endpoint, no key
                data = json.load(r)
        else:
            data = fetch()
        for info in data.get("result", {}).values():
            ws = info.get("wsname", "")
            if "/" in ws and tuple(_norm(x) for x in ws.split("/")) == (_norm(base), _norm(quote)):
                return info["base"], info["quote"]
    except Exception as exc:  # noqa: BLE001 - fall back, the sleeve still runs and logs a mark warning
        print(f"asset code lookup failed for {pair}: {exc!r}", file=sys.stderr)
    return base, quote


def _kraken_data_client() -> tuple:
    from nautilus_trader.adapters.kraken import (
        KrakenDataClientConfig,
        KrakenDataClientFactory,
        KrakenEnvironment,
        KrakenProductType,
    )

    # Real Kraken prices (LIVE is Kraken's production feed; DEMO is futures-only).
    # No api_key/api_secret: public market data only.
    return KrakenDataClientFactory(), KrakenDataClientConfig(product_type=KrakenProductType.SPOT,
                                                             environment=KrakenEnvironment.LIVE)


KRAKEN = register(VenueProfile(
    name="KRAKEN",
    label="Kraken spot",
    # Entry tier (under $10k 30-day volume), Kraken's UK schedule from 9 July 2026.
    fees=FeeSchedule(maker=Decimal("0.0040"), taker=Decimal("0.0080")),
    fee_basis="entry tier",
    daily_history=_kraken_daily,
    ohlc_history=_kraken_ohlc,
    asset_codes=kraken_asset_codes,
    data_client=_kraken_data_client,
))
