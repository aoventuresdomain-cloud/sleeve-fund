"""Which fee schedule a backtest, paper sleeve or report uses, and why.

The fees come from the exchange itself: the supervisor reads each connected live account's fee
schedule from its venue (query-only key, on the server only) and records it. Every mode then
uses the most recent schedule fetched for the sleeve's venue. Until an account on that venue is
connected, the venue profile's published schedule is used, and every result says so.

FEE-LIVE (15 Oct 2026): on a venue with published fee bounds (the paper perp venue) the rate is the instrument's own,
from venue_terms: the account's fetched rate, or on paper the published VIP 0 labelled "published, not account". It
never falls back to a schedule silently; without an instrument (research, the venue list) it is the published VIP 0,
which research never charges below.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sleeve_fund.instruments import FeeSchedule, pct
from sleeve_fund.venues import venue as venue_profile


@dataclass(frozen=True)
class FeeQuote:
    fees: FeeSchedule
    source: str  # "account" or "published"
    account: str | None
    fetched_at: datetime | None
    venue_label: str
    basis: str = ""  # published rates: which tier, and where and when they were read
    words: str = ""  # FEE-LIVE: the instrument's own line (venue_terms), said in place of the venue-wide one

    @property
    def text(self) -> str:
        if self.words:
            return self.words
        rates = f"{pct(self.fees.maker)} maker, {pct(self.fees.taker)} taker"
        if self.source == "account":
            # "Venue fees", never the venue's name: this text reaches activity and the backtest (QA U8 ruling).
            return (f"Venue fees: {rates}, from account {self.account}, "
                    f"fetched {self.fetched_at:%d %b %Y %H:%M} UTC")
        basis = f" ({self.basis})" if self.basis else ""
        return f"Venue fees: {rates}, published schedule{basis}; no account on this venue is connected yet"

    @property
    def short(self) -> str:
        """One line for a result's header: "0.40/0.80% fees (Tier 1, 3 Oct 2026)"."""
        rates = f"{pct(self.fees.maker)[:-1]}/{pct(self.fees.taker)} fees"
        if self.source == "account":
            return f"{rates} (account {self.account}, {self.fetched_at.day} {self.fetched_at:%b %Y})"
        return f"{rates} ({self.basis or 'published'})"


def taker_of(store, sleeve) -> Decimal:
    """The taker rate a stored strategy's exits are charged at: its market's schedule, from the venue's fee source
    (resolve). What the loss at its stop counts (open_risk.position_risk, FEE-LIVE (5))."""
    from sleeve_fund import markets

    return markets.fees_for(sleeve.params, resolve(sleeve.venue, store).fees, sleeve.venue).taker


def resolve(venue: str | None = None, store=None, instrument: str | None = None, now: datetime | None = None,
            live: bool = False) -> FeeQuote:
    """The fee schedule to use for `venue` now: the latest one fetched from a connected account,
    else the venue profile's published schedule. On a FEE-LIVE venue (published fee bounds) with an instrument, the
    instrument's last-known-good rate (venue_terms.term); one never fetched is charged at the published VIP 0, its
    entries refused (venue_terms.entry_refusal) while its exits run."""
    profile = venue_profile(venue)
    if profile.fee_bounds:
        return _live_quote(profile, store, instrument, now, live)
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


def _live_quote(profile, store, instrument: str | None, now: datetime | None, live: bool) -> FeeQuote:
    from datetime import timezone

    from sleeve_fund import venue_terms

    if store is None or instrument is None:
        return FeeQuote(profile.fees, "published", None, None, profile.label, profile.fee_basis)
    t = venue_terms.term(store, profile, "fee", instrument, now or datetime.now(timezone.utc), live)
    rates = f"{float(profile.fees.maker):.3%} maker, {float(profile.fees.taker):.3%} taker"
    if t.value is None:
        return FeeQuote(profile.fees, "published", None, None, profile.label, profile.fee_basis,
                        words=f"Venue fees: {rates}, the published VIP 0, until {t.words}")
    maker, taker = t.value
    published = t.source == venue_terms.PUBLISHED_FEES
    return FeeQuote(FeeSchedule(maker, taker), "published" if published else "account", t.account, t.fetched_at,
                    profile.label, profile.fee_basis if published else "",
                    words=f"Venue fees: {float(maker):.3%} maker, {float(taker):.3%} taker; {t.words}")
