"""Which fee schedule a backtest, paper sleeve or report uses, and why.

The fees come from the exchange itself: the supervisor reads each connected live account's fee
schedule from its venue (query-only key, on the server only) and records it. Every mode then
uses the most recent schedule fetched for the sleeve's venue. Until an account on that venue is
connected, the venue profile's published schedule is used, and every result says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sleeve_fund.instruments import FeeSchedule
from sleeve_fund.venues import venue as venue_profile


@dataclass(frozen=True)
class FeeQuote:
    fees: FeeSchedule
    source: str  # "account" or "published"
    account: str | None
    fetched_at: datetime | None
    venue_label: str
    basis: str = ""  # published rates: which tier, and where and when they were read

    @property
    def text(self) -> str:
        rates = f"{float(self.fees.maker):.2%} maker, {float(self.fees.taker):.2%} taker"
        if self.source == "account":
            return (f"{self.venue_label}: {rates}, from account {self.account}, "
                    f"fetched {self.fetched_at:%d %b %Y %H:%M} UTC")
        basis = f" ({self.basis})" if self.basis else ""
        return f"{self.venue_label}: {rates}, published schedule{basis}; no account on this venue is connected yet"


def resolve(venue: str | None = None, store=None) -> FeeQuote:
    """The fee schedule to use for `venue` now: the latest one fetched from a connected account,
    else the venue profile's published schedule."""
    profile = venue_profile(venue)
    row = None
    if store is not None:
        try:
            row = store.latest_fees(profile.name)
        except Exception:  # noqa: BLE001 - no table yet (an old database) or no database: published fees
            row = None
    if row:
        fees = FeeSchedule(Decimal(str(row["maker"])), Decimal(str(row["taker"])))
        return FeeQuote(fees, "account", row["account"], row["fetched_at"], profile.label)
    return FeeQuote(profile.fees, "published", None, None, profile.label, profile.fee_basis)
