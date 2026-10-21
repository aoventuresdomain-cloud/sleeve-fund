"""Venue profiles: everything that differs between venues, in one place.

The engine, strategies, runtime and research never name a venue. A sleeve names its venue, and
every mode (research, the backtest page, paper, later live) reads that venue's profile for its fee
schedule, price history, live market data, instrument naming and trading calendar. Adding a venue
means registering a profile here, not changing the engine.
"""

from __future__ import annotations

import json
import math
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Callable

import pandas as pd
from nautilus_trader.model import Venue

from sleeve_fund.instruments import FeeSchedule, pct, perpetual, spot_pair
from sleeve_fund.margin import Tier

# FEE-LIVE: the paper venue's published perp VIP 0 (the PM's pasted schedule, 15 Oct 2026; research/venue-data): the one
# place a perp fee rate is written. What the paper venue charges until its account's own rate is fetched (venue_terms),
# what research charges at least, the simulated perp's rates and the shadow outcomes' fee.
PAPER_PERP_FEES = FeeSchedule(maker=Decimal("0.0002"), taker=Decimal("0.00055"))


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
    # (pair, start ms, end ms) -> the venue's trades in [start, end) as (ns, price), oldest first: a replayed stop's
    # trigger minute (GAP-DOWN); None where the venue serves no trade history
    trade_history: Callable[[str, int, int], list] | None = None
    # (pair, cursor) -> (1-minute bars by open time, next cursor, caught_up), for the history store
    minute_loader: Callable[[str, str], tuple] | None = None
    merge_minutes: bool = False  # the loader's pages split minutes (bars built from trades)
    # time -> the loader's cursor for history starting then, so a backfill need not start at listing
    minute_cursor_at: Callable[[pd.Timestamp], str] | None = None
    # (pair, cursor) -> (volume and taker_buy by open time, next cursor, caught_up), for the store's taker-buy
    # volume beside its minutes (R-22); None where the venue's candles carry none
    taker_loader: Callable[[str, str], tuple] | None = None
    # (pair, start ms) -> the venue's settled funding records as it sends them, for the read-only spot check (R-02)
    funding_raw: Callable[[str, int], list] | None = None
    # pair -> None, raising ValueError when the venue doesn't list it (public data, no key)
    check_listed: Callable[[str], None] | None = None
    request_interval: float = 1.0  # seconds between loader requests, within the venue's rate limit
    calendar: str = "24/7"
    # Half the bid-ask spread a backtest charges on orders that take liquidity, until a paper sleeve
    # on this venue has measured the instrument's own (sleeve_fund.spreads). Deliberately cautious.
    assumed_half_spread: float = 0.0005
    # The venue lists linear perpetuals only (no spot): every strategy on it trades the "perp" market, under
    # the venue's own instrument symbol (`symbol`), with its real funding (`funding_loader`).
    perpetual: bool = False
    # pair -> the venue's instrument symbol, as its data client names the instrument (BTC/USDT -> BTCUSDT-PERP)
    symbol: Callable[[str], str] | None = None
    # (pair, start in ms) -> up to a page of settled funding, [(time in ms, rate)], oldest first
    funding_loader: Callable[[str, int], list] | None = None
    funding_hours: tuple[int, ...] = (0, 8, 16)  # UTC hours the venue settles funding at
    # pair -> the instrument's cap on |funding rate| per settlement as the venue publishes it, as a static value: a kept
    # rate beyond it is alerted but still charged as it is (Advisor, 6 Oct 2026). The cap that applied at each
    # settlement, kept from the venue, replaces this with DA-11.
    published_funding_caps: dict[str, float] = field(default_factory=dict)
    # CCY-8DP (Advisor 13 Oct 16:15 and 16:25 UK): account name -> why it may run a Strategy whose quote currency the
    # engine holds to fewer than the 8 places fills are booked to. Only named test accounts, never Strategy evidence;
    # every other account on the venue, and every venue without an entry, is refused (supervisor.check_quote_places).
    sub_8dp_test_accounts: dict[str, str] = field(default_factory=dict)
    # series -> (pair, start in ms) -> up to a page of the venue's snapshots, [(time in ms, *values)], oldest first,
    # one per `stats_minutes`: open interest and the positioning ratios (sleeve_fund.open_interest.SERIES)
    stats_loaders: dict[str, Callable[[str, int], list]] = field(default_factory=dict)
    stats_minutes: int = 5
    # pair -> {"price_precision", "size_precision", "min_quantity", "min_notional"}, the venue's contract limits
    contract: Callable[[str], dict] | None = None
    # Instruments the history store always keeps for this venue, before any strategy trades them
    core_pairs: tuple[str, ...] = ()
    # Instruments a form suggests on this venue (any other can still be typed); its core list when empty
    instrument_hints: tuple[str, ...] = ()
    # How far ahead a good-till-date order must expire for the venue to take it; None: it takes no good-till-date
    # orders (ORPHAN-GUARD og1 then sends good-till-cancelled and journals that only the backstops cover it)
    gtd_min_lead: timedelta | None = None
    # Whether margin can be added to an open isolated perp position (P2-8, Advisor ruling B's top-up). Off on every
    # venue until one is checked: a two-leg Strategy's perp leg near liquidation then closes the pair instead.
    allows_margin_top_up: bool = False
    # A market data hub (sleeve_fund.hub) feeds paper here: every bar is built from its minutes, so the venue's
    # own candles (EXTERNAL bar specs) aren't offered until slower bars are built from minutes (v2 P1-4)
    hub: bool = False
    # Where the venue's trading day starts, in minutes after 00:00 UTC. A strategy's slower candles (v2 P1-4) align
    # to it, so a daily candle closes there. 00:00 UTC on every venue so far (Independent Quant Advisor, 5 Oct); the
    # history store's resample assumes it, so a venue set otherwise can't warm slower candles from the store.
    daily_anchor_minutes: int = 0
    # VS-1 (A12; Advisor 15 Oct): pair -> the venue's published maintenance tiers (margin.Tier: notional ceiling, rate,
    # maintenance amount), and where and when they were read. A perp on a pair with no table is refused, never
    # liquidated at an assumed rate (markets.maintenance_for). Kept from the venue by DA-11 before G2.
    maintenance_tiers: dict[str, tuple] = field(default_factory=dict)
    maintenance_basis: str | None = None
    maintenance_read_on: date | None = None  # when the tables were read: the start check notes them past 30 days
    # FEE-LIVE: where the venue's public margin tiers are read ({symbol} filled in) and its account fee-rate path
    risk_limit_url: str = ""
    fee_rate_path: str = ""
    maintenance_limits: dict[str, float] = field(default_factory=dict)  # pair -> the venue's last published ceiling
    # B-36b: pair -> the contract's price tick, kept beside its tiers (read through markets.price_tick), and how the
    # venue's live tick is read for the start check (check_price_ticks)
    price_ticks: dict[str, str] = field(default_factory=dict)
    live_tick: Callable[[str], Decimal] | None = None
    # B-44: pair -> (quantity step, minimum order quantity), kept from the same listing as the tick, and how the venue's
    # live lot size is read (paper_venue_lot). Sizing rounds down to the step and refuses below the minimum.
    lot_sizes: dict[str, tuple[str, str]] = field(default_factory=dict)
    live_lot: Callable[[str], tuple[Decimal, Decimal]] | None = None
    # B44-STEP: pair -> why its new entries are refused (a listed step the engine can't round to), set by contract_lot
    lot_refusals: dict[str, str] = field(default_factory=dict)
    # FEE-LIVE: the published range a fetched fee rate must fall in ({"maker": (lowest, VIP 0), "taker": ...}) and
    # where it was read; a profile without one has no fetched rates (venue_terms refuses to read them)
    fee_bounds: dict[str, tuple] | None = None
    fee_bounds_basis: str = ""

    @property
    def venue(self) -> Venue:
        return Venue(self.name)

    def funding_cap(self, pair: str) -> float:
        """The cap on |funding rate| per settlement for the instrument: a rate beyond it is refused, never charged
        (QA P1-O17). The sanity default until each instrument's published cap is kept (DA-11)."""
        from sleeve_fund.funding import CAP

        return CAP

    @property
    def hints(self) -> tuple[str, ...]:
        return self.instrument_hints or self.core_pairs

    def symbol_of(self, pair: str) -> str:
        """The venue's own symbol for a BASE/QUOTE pair: the instrument id paper subscribes to."""
        return self.symbol(pair) if self.symbol is not None else pair

    def instrument(self, base: str, quote: str, price_precision: int = 2, size_precision: int = 8,
                   fees: FeeSchedule | None = None):
        """An instrument at this venue, carrying the given fees (sleeve_fund.fees.resolve: the connected
        account's) or, without them, the venue's published schedule: a spot pair, or on a perpetual venue
        its perpetual, with the venue's own symbol and contract limits where it can be asked."""
        if not self.perpetual:
            return spot_pair(base, quote, fees=fees or self.fees, venue=self.venue, price_precision=price_precision,
                             size_precision=size_precision)
        limits: dict = {}
        if self.contract is not None:
            try:
                limits = self.contract(f"{base}/{quote}")
            except Exception as exc:  # noqa: BLE001 - unreachable venue: the precisions given, cautious limits
                print(f"{self.label}: contract limits for {base}/{quote} unavailable ({exc!r})", file=sys.stderr)
        tick = self.contract_tick(f"{base}/{quote}")
        if tick is not None:  # B-43: the one tick source, over the contract limits' and the default's
            limits["price_precision"] = max(0, -tick.as_tuple().exponent)
        lot = self.contract_lot(f"{base}/{quote}")
        if lot is not None:  # B-44: the step and minimum from the same listing, over the contract limits'
            step, least = lot
            limits.update(size_precision=max(0, -step.as_tuple().exponent), min_quantity=float(least))
        return perpetual(base, quote, fees=fees or self.fees, venue=self.venue, symbol=self.symbol_of(f"{base}/{quote}"),
                         price_precision=limits.get("price_precision", price_precision),
                         size_precision=limits.get("size_precision", min(size_precision, 3)),
                         min_quantity=limits.get("min_quantity"), min_notional=limits.get("min_notional", 5.0),
                         size_increment=lot[0] if lot is not None else None)

    def contract_lot(self, pair: str) -> tuple[Decimal, Decimal] | None:
        """B-44: the (quantity step, minimum order quantity) an instrument on `pair` is built with: the venue's live
        listing, else the kept one, said on stderr either way they part (as contract_tick): a failed read takes the kept
        lot; a live lot that differs from the kept one wins. None when it can't be read and none is kept: the contract
        limits then stand, as before B-44."""
        kept = self.lot_sizes.get(pair)
        kept = tuple(Decimal(x).normalize() for x in kept) if kept else None
        if self.live_lot is None:
            return kept
        try:
            live = tuple(Decimal(str(x)).normalize() for x in self.live_lot(pair))
        except LotStepUnusable as exc:  # Advisor B44-STEP: an error at start, entries refused, exits run
            self.lot_refusals[pair] = (f"{self.symbol_of(pair)} lists a quantity step of {exc.step}, which the engine "
                                       f"can't round to; it is built on "
                                       + (f"the kept step {kept[0]}" if kept else "the contract's step")
                                       + ", and new entries are refused until a step it can round to is in place; "
                                       "exits and stops still run")
            print(f"{self.label}: {self.lot_refusals[pair]}", file=sys.stderr)
            return kept
        except Exception as exc:  # noqa: BLE001 - an unreachable venue: the kept lot, said
            if kept is not None:
                print(f"{self.label}: live lot size for {pair} unavailable ({exc!r}); the kept step {kept[0]} and "
                      f"minimum {kept[1]} are used", file=sys.stderr)
            return kept
        self.lot_refusals.pop(pair, None)
        if kept is not None and live != kept:
            print(f"{self.label}: the live lot size (step {live[0]}, minimum {live[1]}) for {pair} differs from the kept "
                  f"(step {kept[0]}, minimum {kept[1]}); the live one is used", file=sys.stderr)
        return live

    def contract_tick(self, pair: str) -> Decimal | None:
        """B-43: the price tick an instrument on `pair` is built with: the venue's live tick, else the kept one (B-36b),
        said on stderr either way they part (Advisor): a failed read takes the kept tick, never 0.01; a live tick that
        differs from the kept one wins (the start check refuses that Strategy until it is checked in). None when the
        venue reads no tick and keeps none."""
        kept = self.price_ticks.get(pair)
        kept = Decimal(kept).normalize() if kept else None
        if self.live_tick is None:
            return kept
        try:
            live = Decimal(str(self.live_tick(pair))).normalize()
        except Exception as exc:  # noqa: BLE001 - an unreachable venue: the kept tick, said
            if kept is not None:
                print(f"{self.label}: live price tick for {pair} unavailable ({exc!r}); the kept price tick {kept} is "
                      "used", file=sys.stderr)
            return kept
        if kept is not None and live != kept:
            print(f"{self.label}: the live price tick {live} for {pair} differs from the kept {kept}; the live one is "
                  "used", file=sys.stderr)
        return live

    def key_env(self, account: str) -> tuple[str, str]:
        """The two server environment variables holding a live account's key and secret on this venue."""
        suffix = account.upper().replace("-", "_")
        return f"{self.name}_API_KEY__{suffix}", f"{self.name}_API_SECRET__{suffix}"

    def fee_text(self) -> str:
        return f"{self.label}: {pct(self.fees.maker)} maker, {pct(self.fees.taker)} taker ({self.fee_basis})"


VENUES: dict[str, VenueProfile] = {}
# The venue of a strategy saved before strategies carried their own (all of them Kraken's then). Kept so those
# journals still price on the venue they traded on; every strategy since names its venue.
DEFAULT_VENUE = "KRAKEN"
# The venue Research and the strategy sprint start a study on: the paper test venue (PM, 5 Oct 2026).
RESEARCH_VENUE = "BINANCE"


# orders.venue and accounts.venue hold a venue's name in 16 characters (store.py); a longer one would fail every
# order insert on Postgres (CR F-P24-a), so it is refused when the venue is registered.
VENUE_NAME_MAX = 16


def register(profile: VenueProfile) -> VenueProfile:
    if len(profile.name) > VENUE_NAME_MAX:
        raise ValueError(f"venue name {profile.name!r} is over {VENUE_NAME_MAX} characters, the width orders store")
    VENUES[profile.name] = profile
    return profile


def base_venue(name: str) -> str:
    """The venue a portfolio run's venue clone stands for (research.portfolio.clone_venue: VENUE -> VENUE_P1): a
    clone trades with its venue's fees, spread, perpetual terms and day start. Any other name comes back as given."""
    return re.sub(r"_P\d+$", "", str(name))


def venue(name: str | None = None) -> VenueProfile:
    name = base_venue(name or DEFAULT_VENUE).upper()
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
    # FEE-LIVE (HoE): no venue falls back silently. No fee endpoint is read here, so the published Tier 1 is written
    # as a labelled row at each hourly attempt (venue_terms.fetch_due), checked against zero up to Tier 1.
    fee_bounds={"maker": (Decimal("0"), Decimal("0.0040")), "taker": (Decimal("0"), Decimal("0.0080"))},
    fee_bounds_basis="spot maker and taker, zero up to the published Tier 1, 3 Oct 2026",
    daily_history=_kraken_daily,
    sub_8dp_test_accounts={"paper": "the hub-fed USD test Strategies and the full-stack check"},
    ohlc_history=_kraken_ohlc,
    list_instruments=kraken_instruments,
    asset_codes=kraken_asset_codes,
    data_client=_kraken_data_client,
    fetch_fees=kraken_account_fees,
    minute_loader=kraken_minutes,
    merge_minutes=True,
    core_pairs=("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "SUI/USD"),
    instrument_hints=("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "SUI/USD", "ADA/USD", "DOGE/USD", "BTC/GBP",
                      "ETH/GBP"),
    # Kraken's Trades `since` takes Unix seconds (its `last` cursor comes back in nanoseconds).
    minute_cursor_at=lambda ts: str(int(ts.timestamp())),
    check_listed=_kraken_listed,
))


# --- Binance USD-M perpetuals ------------------------------------------------------------------
# Research and paper only, on public market data with no account and no key: Binance is closed to new UK
# users, so live trading here is a G2 decision after tax and legal advice (PM, 5 Oct 2026).

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
_BINANCE_INFO: dict = {}  # exchangeInfo, fetched once per process


def binance_symbol(pair: str) -> str:
    """BTC/USDT -> BTCUSDT, Binance's own symbol for the USD-M perpetual."""
    base, quote = pair.upper().split("/")
    return f"{base}{quote}"


def _binance_info(get_json=None) -> dict:
    from sleeve_fund.data import _get_json

    if get_json is not None:
        return get_json(f"{BINANCE_FAPI}/exchangeInfo")
    if not _BINANCE_INFO:
        _BINANCE_INFO.update(_get_json(f"{BINANCE_FAPI}/exchangeInfo"))
    return _BINANCE_INFO


def _binance_listing(pair: str, get_json=None) -> dict:
    sym = binance_symbol(pair)
    for info in _binance_info(get_json).get("symbols", []):
        if info.get("symbol") == sym and info.get("contractType") == "PERPETUAL":
            if info.get("status") != "TRADING":
                raise ValueError(f"Binance lists {pair} but it is not trading ({info.get('status')})")
            return info
    raise ValueError(f"Binance does not list a {pair} USD-M perpetual")


def binance_instruments(get_json=None) -> list[str]:
    """Every USD-M perpetual Binance is trading, as BASE/QUOTE."""
    return sorted(f"{i['baseAsset']}/{i['quoteAsset']}" for i in _binance_info(get_json).get("symbols", [])
                  if i.get("contractType") == "PERPETUAL" and i.get("status") == "TRADING")


def binance_contract(pair: str, get_json=None) -> dict:
    """The perpetual's price and size steps, smallest order and smallest notional, from exchangeInfo."""
    info = _binance_listing(pair, get_json)
    f = {x["filterType"]: x for x in info.get("filters", [])}

    def decimals(step: str) -> int:
        d = Decimal(step).normalize()
        return max(0, -d.as_tuple().exponent)

    out = {"price_precision": decimals(f["PRICE_FILTER"]["tickSize"]) if "PRICE_FILTER" in f else info["pricePrecision"],
           "size_precision": decimals(f["LOT_SIZE"]["stepSize"]) if "LOT_SIZE" in f else info["quantityPrecision"]}
    if "LOT_SIZE" in f:
        out["min_quantity"] = float(f["LOT_SIZE"]["minQty"])
    if "MIN_NOTIONAL" in f:
        out["min_notional"] = float(f["MIN_NOTIONAL"].get("notional", f["MIN_NOTIONAL"].get("minNotional", 5)))
    return out


# B-43 (Advisor 20 Oct): the paper venue's own public instrument listing, the venue its tiers (risk_limit_url) and fees
# come from and paper and the demo copy price against: the source of the live price tick, never another venue's.
PAPER_INSTRUMENTS_URL = "https://api.bybit.com/v5/market/instruments-info?category=linear&symbol={symbol}"


def paper_venue_listing(pair: str, get_json=None) -> dict:
    """The paper venue's listing row for the linear perpetual on `pair` (B-43): the one read its price tick and lot size
    come from. Raises ValueError when the venue refuses the request, doesn't list the pair, or lists it not trading."""
    from sleeve_fund.data import _get_json

    sym = binance_symbol(pair)  # BTC/USDT -> BTCUSDT, the same form on both venues
    answer = (get_json or _get_json)(PAPER_INSTRUMENTS_URL.format(symbol=sym))
    if answer.get("retCode") != 0:
        raise ValueError(f"the venue refused the listing for {pair} ({answer.get('retMsg')})")
    rows = [r for r in (answer.get("result") or {}).get("list", []) if r.get("symbol") == sym]
    if not rows:
        raise ValueError(f"the venue does not list a {pair} perpetual")
    if rows[0].get("status") != "Trading":
        raise ValueError(f"the venue lists {pair} but it is not trading ({rows[0].get('status')})")
    return rows[0]


class LotStepUnusable(ValueError):
    """A listed quantity step the engine can't round to (not a power of ten): Advisor B44-STEP."""

    def __init__(self, step: Decimal):
        self.step = step
        super().__init__(f"the venue's lot step {step} is not a power of ten, which the engine can't round to")


def paper_venue_lot(pair: str, get_json=None) -> tuple[Decimal, Decimal]:
    """B-44: the paper venue's live (quantity step, minimum order quantity) for `pair`, exactly as its listing's
    lotSizeFilter gives them. Raises ValueError when either is missing or not above 0, the minimum isn't a whole
    number of steps, or the step isn't a power of ten: the engine rounds a quantity by its step's exponent (quantize),
    so 0.007 would stay 0.007 on a 0.005 step and the venue would refuse it (QD). The kept lot then stands, said."""
    f = paper_venue_listing(pair, get_json).get("lotSizeFilter") or {}
    step, least = f.get("qtyStep"), f.get("minOrderQty")
    if not step or not least:
        raise ValueError(f"the venue lists no lot size for {pair}")
    step, least = Decimal(step).normalize(), Decimal(least).normalize()
    if step <= 0 or least <= 0 or least % step:
        raise ValueError(f"the venue's lot size for {pair} can't be used: step {step}, minimum {least}")
    if step != Decimal(1).scaleb(step.adjusted()):
        raise LotStepUnusable(step)
    return step, least


def paper_venue_tick(pair: str, get_json=None) -> Decimal:
    """The paper venue's live price tick for `pair`, exactly as its listing's priceFilter gives it (B-43)."""
    tick = (paper_venue_listing(pair, get_json).get("priceFilter") or {}).get("tickSize")
    if not tick:
        raise ValueError(f"the venue lists no price tick for {pair}")
    return Decimal(tick).normalize()


def _binance_klines(pair: str, interval: str, start_ms: int | None = None, limit: int = 1500,
                    get_json=None, taker: bool = False) -> pd.DataFrame:
    """Candles by OPEN time, the forming one included. taker (R-22): a taker_buy column too, the base volume bought
    by orders that took liquidity (the kline's taker-buy base volume)."""
    import urllib.parse

    from sleeve_fund.data import _get_json

    q = {"symbol": binance_symbol(pair), "interval": interval, "limit": limit}
    if start_ms is not None:
        q["startTime"] = start_ms
    rows = (get_json or _get_json)(f"{BINANCE_FAPI}/klines?" + urllib.parse.urlencode(q))
    if isinstance(rows, dict):  # an error body: {"code": ..., "msg": ...}
        raise ValueError(f"Binance: {rows.get('msg', rows)}")
    cols = ["open", "high", "low", "close", "volume"] + (["taker_buy"] if taker else [])
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame([r[:6] + ([r[9]] if taker else []) for r in rows], columns=["t", *cols])
    out = df[cols].astype(float)
    out.index = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
    out.index.name = "timestamp"
    return out


_BINANCE_INTERVALS = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h",
                      480: "8h", 720: "12h", 1440: "1d", 4320: "3d", 10080: "1w"}


def binance_ohlc(pair: str, minutes: int, get_json=None) -> pd.DataFrame:
    """Recent candles for charts and warm-up top-ups, by OPEN time, the forming one kept."""
    if minutes not in _BINANCE_INTERVALS:
        raise ValueError(f"Binance has no {minutes}-minute candles; use one of {sorted(_BINANCE_INTERVALS)}")
    return _binance_klines(pair, _BINANCE_INTERVALS[minutes], get_json=get_json)


def binance_daily(pair: str, get_json=None) -> pd.DataFrame:
    """Daily candles since listing (pages of 1,500), closed bars only, indexed by CLOSE time."""
    from sleeve_fund.data import validate_ohlcv

    parts, start = [], 0
    while True:
        page = _binance_klines(pair, "1d", start_ms=start, get_json=get_json)
        if page.empty:
            break
        parts.append(page)
        if len(page) < 1500:
            break
        start = int(page.index[-1].timestamp() * 1000) + 1
    if not parts:
        raise ValueError(f"no daily history for {pair}")
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].iloc[:-1]  # the newest day is still forming
    df.index = df.index + pd.Timedelta("1D")
    return validate_ohlcv(df)


def binance_minutes(pair: str, cursor: str, get_json=None) -> tuple[pd.DataFrame, str, bool]:
    """One page (1,500) of 1-minute candles from `cursor` (an open time in ms; empty starts at listing). Binance
    keeps every minute since a perpetual listed. Caught up once a page reaches the forming minute."""
    bars = _binance_klines(pair, "1m", start_ms=int(cursor) if cursor else 0, get_json=get_json)
    if bars.empty:
        return bars, cursor, True
    nxt = str(int(bars.index[-1].timestamp() * 1000))  # resume on the last minute: it may still have been forming
    return bars, nxt, len(bars) < 1500


def binance_taker_minutes(pair: str, cursor: str, get_json=None) -> tuple[pd.DataFrame, str, bool]:
    """binance_minutes with each minute's taker-buy base volume (R-22): one page of volume and taker_buy by open
    time, the next cursor, and whether it caught up."""
    bars = _binance_klines(pair, "1m", start_ms=int(cursor) if cursor else 0, get_json=get_json, taker=True)
    if bars.empty:
        return bars, cursor, True
    return bars[["volume", "taker_buy"]], str(int(bars.index[-1].timestamp() * 1000)), len(bars) < 1500


TRADES_TIMEOUT = 5  # seconds a request for a replayed stop's minute of trades may take (GAP-DOWN)
TRADES_MAX_PAGES = 10  # pages of 1,000 read for one minute before the replay books its extreme instead


def binance_trades(pair: str, start_ms: int, end_ms: int, get_json=None) -> list[tuple[int, float]]:
    """The perpetual's trades in [start_ms, end_ms) as (ns, price), in the venue's order, from its aggregated trades
    (one row per taker order and price): a replayed stop's trigger minute (GAP-DOWN). One request serves up to 1,000;
    a busier minute is read on from the last row's id."""
    import json
    import urllib.parse
    import urllib.request

    def fetch(url: str):  # public market data, no key; short, as the replay waits on it
        with urllib.request.urlopen(url, timeout=TRADES_TIMEOUT) as r:
            return json.load(r)

    out: list[tuple[int, float]] = []
    q: dict = {"symbol": binance_symbol(pair), "startTime": int(start_ms), "endTime": int(end_ms) - 1, "limit": 1000}
    for _ in range(TRADES_MAX_PAGES):
        rows = (get_json or fetch)(f"{BINANCE_FAPI}/aggTrades?" + urllib.parse.urlencode(q))
        if isinstance(rows, dict):  # an error body: {"code": ..., "msg": ...}
            raise ValueError(f"Binance: {rows.get('msg', rows)}")
        out += [(int(r["T"]) * 1_000_000, float(r["p"])) for r in rows if int(r["T"]) < end_ms]
        if len(rows) < 1000 or int(rows[-1]["T"]) >= end_ms:
            return out
        q = {"symbol": q["symbol"], "fromId": int(rows[-1]["a"]) + 1, "limit": 1000}
    raise ValueError(f"Binance: more than {TRADES_MAX_PAGES * 1000} trades in the minute; read no further")


def binance_funding(pair: str, start_ms: int, get_json=None) -> list[tuple[int, float]]:
    """Up to 1,000 settled funding rates from `start_ms`, [(settlement time in ms, rate)], oldest first. A
    positive rate is paid by longs to shorts, on the position's value at the settlement."""
    import urllib.parse

    from sleeve_fund.data import _get_json

    q = {"symbol": binance_symbol(pair), "startTime": start_ms, "limit": 1000}
    rows = (get_json or _get_json)(f"{BINANCE_FAPI}/fundingRate?" + urllib.parse.urlencode(q))
    if isinstance(rows, dict):
        raise ValueError(f"Binance: {rows.get('msg', rows)}")
    return [(int(r["fundingTime"]), _number_or_none(r.get("fundingRate"))) for r in rows]  # a null: refused by funding


def binance_funding_raw(pair: str, start_ms: int, get_json=None) -> list[dict]:
    """The venue's settled funding records from `start_ms` exactly as it sends them (rates as its own strings), up to
    3: the read-only spot check of the stored rates against the venue (R-02, Advisor) compares with these, not with
    the loader's parsed figures, so a parse or scale slip in the loader would show."""
    import urllib.parse

    from sleeve_fund.data import _get_json

    q = {"symbol": binance_symbol(pair), "startTime": start_ms, "limit": 3}
    rows = (get_json or _get_json)(f"{BINANCE_FAPI}/fundingRate?" + urllib.parse.urlencode(q))
    if isinstance(rows, dict):
        raise ValueError(f"Binance: {rows.get('msg', rows)}")
    return rows


BINANCE_FUTURES_DATA = "https://fapi.binance.com/futures/data"
BINANCE_OI_DAYS = 30  # Binance keeps open interest history for the latest 30 days only


def _binance_stats(path: str, fields: tuple[str, ...], pair: str, start_ms: int, get_json=None,
                   now_ms: int | None = None) -> list[tuple]:
    """Up to 500 five-minute snapshots of one of Binance's futures statistics from `start_ms`, [(time in ms,
    *fields)], oldest first, each stamped at the end of its completed period. Binance keeps only the latest 30 days,
    so an earlier start is moved up to the oldest it still has."""
    import time
    import urllib.parse

    from sleeve_fund.data import _get_json

    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    oldest = now_ms - BINANCE_OI_DAYS * 86_400_000 + 300_000  # a period inside the window, or Binance refuses it
    q = {"symbol": binance_symbol(pair), "period": "5m", "startTime": max(start_ms, oldest), "limit": 500}
    rows = (get_json or _get_json)(f"{BINANCE_FUTURES_DATA}/{path}?" + urllib.parse.urlencode(q))
    if isinstance(rows, dict):
        raise ValueError(f"Binance: {rows.get('msg', rows)}")
    # A null or junk value stays None, and open_interest.refresh refuses that row alone (QA P1-O13): a float() that
    # raised here would lose the whole page.
    return [(int(r["timestamp"]), *(_number_or_none(r.get(f)) for f in fields)) for r in rows]


def _number_or_none(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def binance_open_interest(pair: str, start_ms: int, get_json=None, now_ms: int | None = None) -> list[tuple]:
    """[(time in ms, contracts, notional in the quote currency)]: the open interest at each period's end."""
    return _binance_stats("openInterestHist", ("sumOpenInterest", "sumOpenInterestValue"), pair, start_ms,
                          get_json, now_ms)


_RATIO = ("longShortRatio", "longAccount", "shortAccount")
BINANCE_STATS = {
    "open_interest": binance_open_interest,
    # unused and unvalidated (HoE, 19:07): kept only because Binance drops them after 30 days
    "long_short_global": lambda pair, start, **kw: _binance_stats("globalLongShortAccountRatio", _RATIO, pair, start, **kw),
    "long_short_top": lambda pair, start, **kw: _binance_stats("topLongShortPositionRatio", _RATIO, pair, start, **kw),
}


def _binance_data_client() -> tuple:
    from nautilus_trader.adapters.binance import (
        BinanceDataClientConfig,
        BinanceDataClientFactory,
        BinanceEnvironment,
        BinanceProductType,
    )

    # Binance's production USD-M futures feed. No api_key/api_secret: public market data only.
    return BinanceDataClientFactory(), BinanceDataClientConfig(product_type=BinanceProductType.USD_M,
                                                               environment=BinanceEnvironment.LIVE)


# USD-M futures take a GTD order only if it expires more than 600 s ahead (the published goodTillDate rule).
# ORPHAN-GUARD og1 clamps a short bar's expiry past it; the testnet drill confirms the figure (HoQA, 13 Oct).
# VS-1 tables: the paper venue's published maintenance tiers for its linear BTCUSDT, ETHUSDT, SOLUSDT and XRPUSDT, as
# margin.Tier(notional ceiling, rate, deduction): maintenance = notional x rate - deduction. Tier 1's blank deduction
# is 0; the last ceiling is unbounded (the venue refuses a position past its last, kept as maintenance_limits). All four
# read in one pass on 21 Oct 2026 (PAPER-TIERS-3; BTC's unchanged from DA's hand-checked 15 Oct table) and checked
# against the replies themselves (tests/data/venue_tables, tests/test_paper_tiers_3.py) and continuous at every ceiling.
# Refreshed by DA-11 before G2; re-read and re-check when the venue announces a change.
BTCUSDT_TIERS = (
    Tier(300_000, 0.0033, 0),
    Tier(2_000_000, 0.005, 510),
    Tier(2_600_000, 0.0056, 1_710),
    Tier(3_200_000, 0.0063, 3_530),
    Tier(3_800_000, 0.0067, 4_810),
    Tier(4_400_000, 0.0071, 6_330),
    Tier(5_000_000, 0.0077, 8_970),
    Tier(5_600_000, 0.0091, 15_970),
    Tier(8_500_000, 0.01, 21_010),
    Tier(10_000_000, 0.013, 46_510),
    Tier(14_000_000, 0.015, 66_510),
    Tier(20_000_000, 0.016, 80_510),
    Tier(28_000_000, 0.02, 160_510),
    Tier(38_000_000, 0.025, 300_510),
    Tier(50_000_000, 0.029, 452_510),
    Tier(55_000_000, 0.03, 502_510),
    Tier(60_000_000, 0.035, 777_510),
    Tier(65_000_000, 0.036, 837_510),
    Tier(70_000_000, 0.038, 967_510),
    Tier(75_000_000, 0.04, 1_107_510),
    Tier(80_000_000, 0.045, 1_482_510),
    Tier(85_000_000, 0.05, 1_882_510),
    Tier(90_000_000, 0.055, 2_307_510),
    Tier(95_000_000, 0.06, 2_757_510),
    Tier(100_000_000, 0.065, 3_232_510),
    Tier(105_000_000, 0.07, 3_732_510),
    Tier(110_000_000, 0.085, 5_307_510),
    Tier(115_000_000, 0.095, 6_407_510),
    Tier(120_000_000, 0.1, 6_982_510),
    Tier(150_000_000, 0.12, 9_382_510),
    Tier(200_000_000, 0.14, 12_382_510),
    Tier(250_000_000, 0.18, 20_382_510),
    Tier(400_000_000, 0.3, 50_382_510),
    Tier(800_000_000, 0.42, 98_382_510),
    Tier(math.inf, 0.6, 242_382_510),
)
ETHUSDT_TIERS = (
    Tier(300_000, 0.0033, 0),
    Tier(900_000, 0.005, 510),
    Tier(1_200_000, 0.0056, 1_050),
    Tier(1_500_000, 0.0063, 1_890),
    Tier(1_800_000, 0.0067, 2_490),
    Tier(2_400_000, 0.0071, 3_210),
    Tier(3_200_000, 0.0077, 4_650),
    Tier(4_000_000, 0.0091, 9_130),
    Tier(5_000_000, 0.01, 12_730),
    Tier(8_000_000, 0.012, 22_730),
    Tier(12_000_000, 0.013, 30_730),
    Tier(16_000_000, 0.014, 42_730),
    Tier(22_000_000, 0.015, 58_730),
    Tier(28_000_000, 0.02, 168_730),
    Tier(32_000_000, 0.025, 308_730),
    Tier(34_000_000, 0.03, 468_730),
    Tier(36_000_000, 0.033, 570_730),
    Tier(38_000_000, 0.035, 642_730),
    Tier(40_000_000, 0.038, 756_730),
    Tier(42_000_000, 0.039, 796_730),
    Tier(44_000_000, 0.042, 922_730),
    Tier(46_000_000, 0.045, 1_054_730),
    Tier(48_000_000, 0.05, 1_284_730),
    Tier(50_000_000, 0.055, 1_524_730),
    Tier(52_000_000, 0.06, 1_774_730),
    Tier(54_000_000, 0.068, 2_190_730),
    Tier(58_000_000, 0.075, 2_568_730),
    Tier(62_000_000, 0.09, 3_438_730),
    Tier(66_000_000, 0.1, 4_058_730),
    Tier(72_000_000, 0.12, 5_378_730),
    Tier(80_000_000, 0.14, 6_818_730),
    Tier(100_000_000, 0.18, 10_018_730),
    Tier(200_000_000, 0.3, 22_018_730),
    Tier(600_000_000, 0.42, 46_018_730),
    Tier(math.inf, 0.6, 154_018_730),
)
SOLUSDT_TIERS = (
    Tier(50_000, 0.005, 0),
    Tier(100_000, 0.0056, 30),
    Tier(250_000, 0.0063, 100),
    Tier(400_000, 0.0067, 200),
    Tier(550_000, 0.0071, 360),
    Tier(700_000, 0.0077, 690),
    Tier(850_000, 0.0091, 1_670),
    Tier(1_000_000, 0.01, 2_435),
    Tier(2_000_000, 0.0111, 3_535),
    Tier(3_000_000, 0.0125, 6_335),
    Tier(4_000_000, 0.0143, 11_735),
    Tier(6_000_000, 0.0167, 21_335),
    Tier(8_000_000, 0.02, 41_135),
    Tier(10_000_000, 0.025, 81_135),
    Tier(12_000_000, 0.0263, 94_135),
    Tier(14_000_000, 0.0278, 112_135),
    Tier(16_000_000, 0.0294, 134_535),
    Tier(18_000_000, 0.0313, 164_935),
    Tier(20_000_000, 0.0333, 200_935),
    Tier(22_000_000, 0.0357, 248_935),
    Tier(24_000_000, 0.0385, 310_535),
    Tier(26_000_000, 0.0417, 387_335),
    Tier(28_000_000, 0.0455, 486_135),
    Tier(30_000_000, 0.05, 612_135),
    Tier(35_000_000, 0.0556, 780_135),
    Tier(40_000_000, 0.0625, 1_021_635),
    Tier(45_000_000, 0.0714, 1_377_635),
    Tier(50_000_000, 0.0833, 1_913_135),
    Tier(60_000_000, 0.1, 2_748_135),
    Tier(80_000_000, 0.125, 4_248_135),
    Tier(100_000_000, 0.1667, 7_584_135),
    Tier(150_000_000, 0.25, 15_914_135),
    Tier(200_000_000, 0.3497, 30_869_135),
    Tier(250_000_000, 0.3906, 39_049_135),
    Tier(math.inf, 0.5, 66_399_135),
)
XRPUSDT_TIERS = (
    Tier(50_000, 0.005, 0),
    Tier(100_000, 0.0056, 30),
    Tier(250_000, 0.0063, 100),
    Tier(400_000, 0.0067, 200),
    Tier(550_000, 0.0071, 360),
    Tier(700_000, 0.0077, 690),
    Tier(850_000, 0.0091, 1_670),
    Tier(1_000_000, 0.01, 2_435),
    Tier(2_000_000, 0.0111, 3_535),
    Tier(3_000_000, 0.0125, 6_335),
    Tier(4_000_000, 0.0143, 11_735),
    Tier(6_000_000, 0.0167, 21_335),
    Tier(8_000_000, 0.02, 41_135),
    Tier(10_000_000, 0.025, 81_135),
    Tier(12_000_000, 0.0263, 94_135),
    Tier(14_000_000, 0.0278, 112_135),
    Tier(16_000_000, 0.0294, 134_535),
    Tier(18_000_000, 0.0313, 164_935),
    Tier(20_000_000, 0.0333, 200_935),
    Tier(22_000_000, 0.0357, 248_935),
    Tier(24_000_000, 0.0385, 310_535),
    Tier(26_000_000, 0.0417, 387_335),
    Tier(28_000_000, 0.0455, 486_135),
    Tier(30_000_000, 0.05, 612_135),
    Tier(35_000_000, 0.0556, 780_135),
    Tier(40_000_000, 0.0625, 1_021_635),
    Tier(45_000_000, 0.0714, 1_377_635),
    Tier(50_000_000, 0.0833, 1_913_135),
    Tier(60_000_000, 0.1, 2_748_135),
    Tier(80_000_000, 0.125, 4_248_135),
    Tier(100_000_000, 0.1667, 7_584_135),
    Tier(150_000_000, 0.25, 15_914_135),
    Tier(200_000_000, 0.3497, 30_869_135),
    Tier(250_000_000, 0.3906, 39_049_135),
    Tier(math.inf, 0.5, 66_399_135),
)
PAPER_VENUE_TIERS_BASIS = ("paper venue risk-limit and instruments-info for BTCUSDT, ETHUSDT, SOLUSDT and XRPUSDT, "
                           "read 21 Oct 2026 21:30 UTC, research/venue-data/paper-venue-tables-2026-10-21.md")


BINANCE_GTD_MIN_LEAD = timedelta(seconds=600)

BINANCE = register(VenueProfile(
    name="BINANCE",
    label="Binance USD-M perpetuals",
    # FEE-LIVE: the paper venue's published perp VIP 0 (the PM's pasted schedule, 15 Oct 2026; research/venue-data),
    # what a new account pays, what research charges at least, and what a demo account with no fee endpoint is
    # charged on paper. Running strategies are charged the account's own rate per instrument (venue_terms).
    fees=PAPER_PERP_FEES,
    fee_basis="VIP 0, published schedule, 15 Oct 2026",
    # A fetched rate outside [the lowest published (Supreme VIP), VIP 0] is missing, never a wider range (Advisor (c)).
    fee_bounds={"maker": (Decimal("0"), PAPER_PERP_FEES.maker), "taker": (Decimal("0.0003"), PAPER_PERP_FEES.taker)},
    fee_bounds_basis="perp maker and taker, Supreme VIP to VIP 0, the PM's pasted schedule of 15 Oct 2026 "
                     "(research/venue-data/paper-venue-fee-schedule-2026-10-15.md), DA hand-check",
    daily_history=binance_daily,
    ohlc_history=binance_ohlc,
    trade_history=binance_trades,
    list_instruments=binance_instruments,
    data_client=_binance_data_client,
    minute_loader=binance_minutes,
    minute_cursor_at=lambda ts: str(int(ts.timestamp() * 1000)),
    taker_loader=binance_taker_minutes,
    funding_raw=binance_funding_raw,
    check_listed=lambda pair: (_binance_listing(pair), None)[1],
    request_interval=0.5,  # a 1,500-candle page weighs 10 of Binance's 2,400 a minute
    # BTC and ETH perpetuals trade a cent or less apart; 0.01% half spread is cautious for them, light for small ones.
    assumed_half_spread=0.0001,
    perpetual=True,
    gtd_min_lead=BINANCE_GTD_MIN_LEAD,
    symbol=lambda pair: f"{binance_symbol(pair)}-PERP",
    funding_loader=binance_funding,
    stats_loaders=BINANCE_STATS,
    contract=binance_contract,
    # 75% of the first-tier maintenance margin rate (0.4%), Binance's rule for its cap; confirmed against fundingInfo
    # once caps are kept from the venue (DA-11).
    published_funding_caps={"BTC/USDT": 0.003},
    # VS-1: a perp on a pair with no table kept here (SUI/USDT) is refused, in research too, until its table is checked in.
    maintenance_tiers={"BTC/USDT": BTCUSDT_TIERS, "ETH/USDT": ETHUSDT_TIERS, "SOL/USDT": SOLUSDT_TIERS,
                       "XRP/USDT": XRPUSDT_TIERS},
    maintenance_basis=PAPER_VENUE_TIERS_BASIS,
    maintenance_read_on=date(2026, 10, 21),
    # The PM's paper venue's public V5 risk-limit page (public market data, read-only, no key) and its signed
    # read-only fee-rate call (FEE-LIVE).
    risk_limit_url="https://api.bybit.com/v5/market/risk-limit?category=linear&symbol={symbol}",
    fee_rate_path="/v5/account/fee-rate",
    # past the last ceiling the venue takes no position; nor do we
    maintenance_limits={"BTC/USDT": 1_200_000_000, "ETH/USDT": 1_000_000_000, "SOL/USDT": 300_000_000,
                        "XRP/USDT": 300_000_000},
    # B-36b: each pair's tick, kept with its tiers from the same read
    price_ticks={"BTC/USDT": "0.1", "ETH/USDT": "0.01", "SOL/USDT": "0.01", "XRP/USDT": "0.0001"},
    live_tick=paper_venue_tick,
    # B-44: (quantity step, minimum order quantity) from the same read as the tiers; the live read on the server wins
    # and says so if it differs
    lot_sizes={"BTC/USDT": ("0.001", "0.001"), "ETH/USDT": ("0.01", "0.01"), "SOL/USDT": ("0.1", "0.1"),
               "XRP/USDT": ("0.1", "0.1")},
    live_lot=paper_venue_lot,
    core_pairs=("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "SUI/USDT"),
    hub=True,
))
