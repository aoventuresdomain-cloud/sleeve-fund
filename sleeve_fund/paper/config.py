"""Sleeve configuration for paper trading. Sleeves are data, not code."""

from __future__ import annotations

import importlib
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from sleeve_fund import markets
from sleeve_fund.instruments import FeeSchedule
from sleeve_fund.strategies import REGISTRY
from sleeve_fund.venues import DEFAULT_VENUE, venue as venue_profile

# Any spot pair (SUI/USD, XRP/GBP, ...). Whether the venue lists it is checked when the sleeve starts.
PAIR_RE = re.compile(r"^[A-Z0-9]{1,12}/[A-Z0-9]{2,6}$")

ALLOWED_BAR_SPECS = {
    # Built locally from the venue's trades; bars close on time, no venue buffering.
    "1-MINUTE-LAST-INTERNAL",
    "5-MINUTE-LAST-INTERNAL",
    "15-MINUTE-LAST-INTERNAL",
    "1-HOUR-LAST-INTERNAL",
    # The venue's own OHLC; needed for daily strategies so warm-up can come from REST history.
    "1-HOUR-LAST-EXTERNAL",
    "4-HOUR-LAST-EXTERNAL",
    "1-DAY-LAST-EXTERNAL",
}


# Most warm-up bars a sleeve loads at start: about five weeks of 1-minute bars from the store.
MAX_WARMUP_BARS = 50_000
# What a venue returns in one request: the most history venue candles (EXTERNAL bars) can warm up on. Bars built
# from trades (INTERNAL) load from the history store, up to MAX_WARMUP_BARS.
VENUE_WARMUP_BARS = 720


def auto_warmup(strategy: str, params: dict, bar_spec: str) -> int:
    """Bars to load at start so every indicator the model and its exits use is settled on the first live bar,
    as the model itself says (warmup_needed), within what can load for this kind of bar."""
    from sleeve_fund.data import spec_minutes
    from sleeve_fund.strategies.base import exit_warmup

    cls = REGISTRY[strategy][0]
    defaults = dict(getattr(importlib.import_module(cls.__module__), "SPEC").default_params or {})
    p = {**defaults, **params}
    cap = MAX_WARMUP_BARS if bar_spec.endswith("INTERNAL") else VENUE_WARMUP_BARS
    return min(cap, max(cls.warmup_needed(p, spec_minutes(bar_spec)), exit_warmup(p)))


def check_hub_bar_spec(venue: str, bar_spec: str) -> None:
    """Raises ValueError for the venue's own candles on a venue a market data hub feeds (VenueProfile.hub): there
    every bar is built from the hub's minutes, and slower ones only from P1-4 (QA P1-C7). Checked where a
    strategy is created, so it is refused there rather than failing to start."""
    profile = venue_profile(venue)
    if profile.hub and not bar_spec.endswith("-INTERNAL"):
        size = "-".join(bar_spec.split("-")[:2]).lower()  # 1-DAY-LAST-EXTERNAL -> 1-day
        # No venue name: this is shown on the dashboard when a strategy is created.
        raise ValueError(f"bar_spec: strategies on this market decide on bars built from the market data hub's "
                         f"minutes, so the venue's own {size} candles aren't available "
                         "there; choose 1, 5 or 15-minute or 1-hour bars")


@dataclass(frozen=True)
class SleeveConfig:
    name: str
    strategy: str
    instrument: str  # e.g. "BTC/USD"
    bar_spec: str  # e.g. "1-DAY-LAST-EXTERNAL"
    starting_balance: float
    params: dict = field(default_factory=dict)
    max_notional: float | None = None
    warmup_bars: int = 0
    risk_profile: str = "balanced"
    venue: str = DEFAULT_VENUE
    # The schedule to charge, from sleeve_fund.fees.resolve (the connected account's rates).
    # None: the venue's published schedule. There is deliberately no per-sleeve fee setting.
    fee_schedule: FeeSchedule | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "venue", venue_profile(self.venue).name)  # raises on an unknown venue
        if self.strategy not in REGISTRY:
            raise ValueError(f"unknown strategy {self.strategy!r}; known: {sorted(REGISTRY)}")
        if self.bar_spec not in ALLOWED_BAR_SPECS:
            raise ValueError(f"bar_spec {self.bar_spec!r} not in {sorted(ALLOWED_BAR_SPECS)}")
        # Any spot pair works; the venue rejects pairs it doesn't list at start-up.
        object.__setattr__(self, "instrument", self.instrument.strip().upper())  # frozen dataclass
        if not PAIR_RE.match(self.instrument):
            raise ValueError(f"instrument must look like BASE/QUOTE (e.g. SUI/USD), got {self.instrument!r}")
        markets.check_venue(self.params, self.venue)  # a perpetual venue has no spot
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        if self.max_notional is not None and self.max_notional <= 0:
            raise ValueError("max_notional must be positive")
        if not 0 <= self.warmup_bars <= MAX_WARMUP_BARS:
            raise ValueError(f"warmup_bars must be between 0 and {MAX_WARMUP_BARS}")
        # Never fewer bars than the model's indicators need to settle (PM, 5 Oct 2026); more can be asked for.
        object.__setattr__(self, "warmup_bars", max(self.warmup_bars, auto_warmup(self.strategy, self.params,
                                                                                  self.bar_spec)))
        from sleeve_fund.risk import profile

        profile(self.risk_profile)  # raises on unknown

    @property
    def base(self) -> str:
        return self.instrument.split("/")[0]

    @property
    def quote(self) -> str:
        return self.instrument.split("/")[1]

    @property
    def fees(self) -> FeeSchedule:
        # A perpetual pays its market's schedule (sleeve_fund.markets); spot pays the venue's.
        return markets.fees_for(self.params, self.fee_schedule or venue_profile(self.venue).fees, self.venue)

    @property
    def instrument_id(self) -> str:
        return f"{venue_profile(self.venue).symbol_of(self.instrument)}.{self.venue}"

    @property
    def bar_type(self) -> str:
        return f"{self.instrument_id}-{self.bar_spec}"


def load_sleeve(path: str | Path) -> SleeveConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    sleeve = raw.get("sleeve", {})
    if "fees" in raw:
        raise ValueError(f"{path}: fees come from the venue profile (sleeve_fund/venues.py); remove [fees]")
    check_hub_bar_spec(sleeve.get("venue", DEFAULT_VENUE), sleeve["bar_spec"])
    return SleeveConfig(
        name=sleeve["name"],
        strategy=sleeve["strategy"],
        instrument=sleeve["instrument"],
        bar_spec=sleeve["bar_spec"],
        starting_balance=float(sleeve["starting_balance"]),
        params=dict(raw.get("params", {})),
        max_notional=sleeve.get("max_notional"),
        warmup_bars=int(sleeve.get("warmup_bars", 0)),
        risk_profile=sleeve.get("risk_profile", "balanced"),
        venue=sleeve.get("venue", DEFAULT_VENUE),
    )


def from_store(sleeve, fee_schedule: FeeSchedule | None = None) -> SleeveConfig:
    """SleeveConfig from a database row (sleeve_fund.store.Sleeve)."""
    params = dict(sleeve.params)
    max_notional = params.pop("max_notional", None)
    return SleeveConfig(
        name=sleeve.name,
        strategy=sleeve.strategy,
        instrument=sleeve.instrument,
        bar_spec=sleeve.bar_spec,
        starting_balance=sleeve.starting_balance,
        params=params,
        max_notional=max_notional,
        warmup_bars=sleeve.warmup_bars,
        risk_profile=sleeve.risk_profile,
        venue=getattr(sleeve, "venue", None) or DEFAULT_VENUE,
        fee_schedule=fee_schedule,
    )


def to_store_kwargs(cfg: SleeveConfig) -> dict:
    params = dict(cfg.params)
    if cfg.max_notional is not None:
        params["max_notional"] = cfg.max_notional
    return dict(
        name=cfg.name, strategy=cfg.strategy, instrument=cfg.instrument, bar_spec=cfg.bar_spec,
        starting_balance=cfg.starting_balance, params=params, risk_profile=cfg.risk_profile,
        warmup_bars=cfg.warmup_bars, venue=None if cfg.venue == DEFAULT_VENUE else cfg.venue,
    )
