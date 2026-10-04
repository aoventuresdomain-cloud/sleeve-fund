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
    # (get_json=None) -> every BASE/QUOTE pair the venue lists, for the chart's instrument dropdown
    list_instruments: Callable[..., list[str]] | None = None
    # (pair, fetch=None) -> (base, quote) as the venue's own instrument data names them
    asset_codes: Callable[..., tuple[str, str]] = lambda pair, fetch=None: tuple(pair.split("/"))  # type: ignore[assignment]
    # () -> (factory, config) for the live market data client; None if paper can't run here yet
    data_client: Callable[[], tuple] | None = None
    # (api_key, api_secret) -> FeeSchedule the venue charges that account. Query-only; supervisor only.
    fetch_fees: Callable[[str, str], FeeSchedule] | None = None
    # (pair, cursor) -> (1-minute bars by open time, next cursor, caught_up), for the history store
    minute_loader: Callable[[str, str], tuple] | None = None
    merge_minutes: bool = False  # the loader's pages split minutes (bars built from trades)
    # time -> the loader's cursor for history starting then, so a backfill need not start at listing
    minute_cursor_at: Callable[[pd.Timestamp], str] | None = None
    # pair -> None, raising ValueError when the venue doesn't list it (public data, no key)
    check_listed: Callable[[str], None] | None = None
    request_interval: float = 1.0  # seconds between loader requests, within the venue's rate limit
    calendar: str = "24/7"
    # Half the bid-ask spread a backtest charges on orders that take liquidity, until a paper sleeve
    # on this venue has measured the instrument's own (sleeve_fund.spreads). Deliberately cautious.
    assumed_half_spread: float = 0.0005

    @property
    def venue(self) -> Venue:
        return Venue(self.name)

    def instrument(self, base: str, quote: str, price_precision: int = 2, size_precision: int = 8,
                   fees: FeeSchedule | None = None) -> CurrencyPair:
        """A spot instrument at this venue, carrying the given fees (sleeve_fund.fees.resolve: the
        connected account's) or, without them, the venue's published schedule."""
        return spot_pair(base, quote, fees=fees or self.fees, venue=self.venue, price_precision=price_precision,
                         size_precision=size_precision)

    def key_env(self, account: str) -> tuple[str, str]:
        """The two server environment variables holding a live account's key and secret on this venue."""
        suffix = account.upper().replace("-", "_")
        return f"{self.name}_API_KEY__{suffix}", f"{self.name}_API_SECRET__{suffix}"

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


def kraken_instruments(get_json=None) -> list[str]:
    """Every pair Kraken lists, named as people write them (XBT/USD -> BTC/USD), from its public pair list."""
    from sleeve_fund.data import _get_json

    data = (get_json or _get_json)(ASSET_PAIRS_URL)
    if data.get("error"):
        raise ValueError(f"Kraken: {'; '.join(data['error'])}")
    return sorted({"/".join(_norm(c) for c in info["wsname"].split("/"))
                   for info in data.get("result", {}).values() if "/" in info.get("wsname", "")})


def kraken_sign(path: str, data: str, nonce: str, secret: str) -> str:
    """Kraken's API-Sign header: HMAC-SHA512 of path + SHA256(nonce + POST data), keyed by the
    base64-decoded secret."""
    import base64
    import hashlib
    import hmac

    digest = hashlib.sha256((nonce + data).encode()).digest()
    mac = hmac.new(base64.b64decode(secret), path.encode() + digest, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


def kraken_account_fees(api_key: str, api_secret: str, post=None) -> FeeSchedule:
    """The spot maker and taker fees Kraken charges this account now (its TradeVolume endpoint,
    which needs only a query permission). Rates are for BTC/USD, Kraken's reference pair; spot
    pairs share one tier schedule."""
    import time
    import urllib.parse

    path = "/0/private/TradeVolume"
    nonce = str(int(time.time() * 1000))
    data = urllib.parse.urlencode({"nonce": nonce, "pair": "XBTUSD"})
    headers = {"API-Key": api_key, "API-Sign": kraken_sign(path, data, nonce, api_secret),
               "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}
    if post is None:
        req = urllib.request.Request("https://api.kraken.com" + path, data=data.encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=20) as r:
            body = json.load(r)
    else:
        body = post(path, data, headers)
    if body.get("error"):
        raise ValueError(f"Kraken: {'; '.join(body['error'])}")
    res = body["result"]
    taker = next(iter(res["fees"].values()))["fee"]
    maker = next(iter((res.get("fees_maker") or res["fees"]).values()))["fee"]
    return FeeSchedule(maker=Decimal(str(maker)) / 100, taker=Decimal(str(taker)) / 100)


_KRAKEN_KEYS: dict[str, str] = {}  # pair -> Kraken's pair id, looked up once per process


def kraken_minutes(pair: str, cursor: str, get_json=None) -> tuple[pd.DataFrame, str, bool]:
    """One page (up to 1,000 trades) of Kraken's full trade history after `cursor`, as 1-minute
    bars. Kraken's API keeps every trade since a pair listed, unlike its 720-candle OHLC endpoint,
    so this is how the store gets complete minute history. The cursor is Kraken's own `last`."""
    from sleeve_fund.data import KRAKEN_API, _get_json, kraken_pair_key
    from sleeve_fund.history import trades_to_minutes

    get_json = get_json or _get_json
    key = _KRAKEN_KEYS.get(pair) or _KRAKEN_KEYS.setdefault(pair, kraken_pair_key(pair, get_json))
    data = get_json(f"{KRAKEN_API}/Trades?pair={key}&since={cursor or 0}&count=1000")
    if data.get("error"):
        raise ValueError(f"Kraken: {'; '.join(data['error'])}")
    result = data["result"]
    rows = next((v for k, v in result.items() if k != "last"), [])
    nxt = str(result.get("last", cursor))
    if not rows:
        return pd.DataFrame(), cursor, True
    trades = pd.DataFrame({"price": [float(r[0]) for r in rows], "volume": [float(r[1]) for r in rows]},
                          index=pd.to_datetime([float(r[2]) for r in rows], unit="s", utc=True))
    return trades_to_minutes(trades), nxt, len(rows) < 1000


def _kraken_listed(pair: str) -> None:
    from sleeve_fund.data import kraken_pair_key

    kraken_pair_key(pair)


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
    # Fallback only, until an account is connected (then the account's own rates are fetched):
    # spot Tier 1 (under $2,501 30-day spot volume), read from the account's Kraken fee page on
    # 3 Oct 2026. Tier 2 (0.30% maker, 0.60% taker) starts at $2,501.
    fees=FeeSchedule(maker=Decimal("0.0040"), taker=Decimal("0.0080")),
    fee_basis="Tier 1, 3 Oct 2026",
    daily_history=_kraken_daily,
    ohlc_history=_kraken_ohlc,
    list_instruments=kraken_instruments,
    asset_codes=kraken_asset_codes,
    data_client=_kraken_data_client,
    fetch_fees=kraken_account_fees,
    minute_loader=kraken_minutes,
    merge_minutes=True,
    # Kraken's Trades `since` takes Unix seconds (its `last` cursor comes back in nanoseconds).
    minute_cursor_at=lambda ts: str(int(ts.timestamp())),
    check_listed=_kraken_listed,
))
