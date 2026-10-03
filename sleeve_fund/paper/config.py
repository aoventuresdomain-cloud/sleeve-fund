"""Sleeve configuration for paper trading. Sleeves are data, not code."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from sleeve_fund.instruments import KRAKEN_UK_ENTRY, FeeSchedule
from sleeve_fund.strategies import REGISTRY

ALLOWED_BAR_SPECS = {
    # Built locally from Kraken trades; bars close on time, no venue buffering.
    "1-MINUTE-LAST-INTERNAL",
    "5-MINUTE-LAST-INTERNAL",
    "15-MINUTE-LAST-INTERNAL",
    "1-HOUR-LAST-INTERNAL",
    # Kraken's own OHLC; needed for daily strategies so warm-up can come from REST history.
    "1-HOUR-LAST-EXTERNAL",
    "4-HOUR-LAST-EXTERNAL",
    "1-DAY-LAST-EXTERNAL",
}


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
    fees: FeeSchedule = KRAKEN_UK_ENTRY

    def __post_init__(self) -> None:
        if self.strategy not in REGISTRY:
            raise ValueError(f"unknown strategy {self.strategy!r}; known: {sorted(REGISTRY)}")
        if self.bar_spec not in ALLOWED_BAR_SPECS:
            raise ValueError(f"bar_spec {self.bar_spec!r} not in {sorted(ALLOWED_BAR_SPECS)}")
        base, sep, quote = self.instrument.partition("/")
        if not (sep and base and quote):
            raise ValueError(f"instrument must look like BASE/QUOTE, got {self.instrument!r}")
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        if self.max_notional is not None and self.max_notional <= 0:
            raise ValueError("max_notional must be positive")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")

    @property
    def base(self) -> str:
        return self.instrument.split("/")[0]

    @property
    def quote(self) -> str:
        return self.instrument.split("/")[1]

    @property
    def instrument_id(self) -> str:
        return f"{self.instrument}.KRAKEN"

    @property
    def bar_type(self) -> str:
        return f"{self.instrument_id}-{self.bar_spec}"


def load_sleeve(path: str | Path) -> SleeveConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    sleeve = raw.get("sleeve", {})
    fees = raw.get("fees")
    return SleeveConfig(
        name=sleeve["name"],
        strategy=sleeve["strategy"],
        instrument=sleeve["instrument"],
        bar_spec=sleeve["bar_spec"],
        starting_balance=float(sleeve["starting_balance"]),
        params=dict(raw.get("params", {})),
        max_notional=sleeve.get("max_notional"),
        warmup_bars=int(sleeve.get("warmup_bars", 0)),
        fees=FeeSchedule(Decimal(str(fees["maker"])), Decimal(str(fees["taker"]))) if fees else KRAKEN_UK_ENTRY,
    )
