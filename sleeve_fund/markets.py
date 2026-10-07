"""Markets a strategy can trade an instrument on: spot (the default, long or flat) or a perpetual future
(long or short, on margin). The engine stays one engine; the market sets the account it trades in, the
costs it pays and the perp's own cash flows (funding) and limits (liquidation).

A perp here is simulated on the venue's live spot prices: research and paper only, with no venue account
(the venue is chosen at G2). Its costs are a low-fee perp venue's published taker and maker rates; the
"perp-venue-fees" market charges the venue's own spot schedule instead, as the stress case.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sleeve_fund.instruments import FeeSchedule

SPOT = "spot"
PERP = "perp"
PERP_VENUE_FEES = "perp-venue-fees"
MARKETS = (SPOT, PERP, PERP_VENUE_FEES)
# The simulated venue's leverage for a perp account: above every risk profile's cap (sleeve_fund.risk), so
# our own leverage and liquidation guards, not the simulated venue's margin check, decide.
VENUE_LEVERAGE = 10


@dataclass(frozen=True)
class PerpTerms:
    """What holding a linear (quote-settled) perpetual costs and risks."""

    label: str
    fees: FeeSchedule | None  # None: the venue's own schedule (the stress case)
    half_spread: float | None  # None: the venue's assumption
    funding_rate: float  # per funding interval, as a share of the position's value: longs pay, shorts receive
    funding_hours: tuple[int, ...]  # UTC hours funding is exchanged at
    maintenance_margin: float  # share of the position's value the account must keep, or it is liquidated
    # The venue whose settled funding rates are charged (sleeve_fund.funding); None charges funding_rate. With a
    # venue, funding_rate is only the fallback for a settlement the venue's records don't have.
    funding_venue: str | None = None


# Published rates of the large perp venues' entry tiers (0.02% maker, 0.05% taker), a tight BTC book
# (0.01% half spread), funding at the usual baseline of 0.01% every 8 hours (about 11% a year, paid by
# longs) and a 0.5% maintenance margin, as the long/short verdict (4 Oct 2026) assumed.
LOW_FEE_PERP = PerpTerms("Low-fee perpetual (simulated)", FeeSchedule(Decimal("0.0002"), Decimal("0.0005")),
                         0.0001, 0.0001, (0, 8, 16), 0.005)
VENUE_FEE_PERP = PerpTerms("Perpetual at the venue's spot fees (stress)", None, None, 0.0001, (0, 8, 16), 0.005)


def market_of(params: dict | None) -> str:
    m = (params or {}).get("market") or SPOT
    if m not in MARKETS:
        raise ValueError(f"unknown market {m!r}; choose one of {', '.join(MARKETS)}")
    return m


def is_perp(params: dict | None) -> bool:
    return market_of(params) != SPOT


def terms(params: dict | None, venue: str | None = None) -> PerpTerms | None:
    """What the strategy's market costs and risks. On a venue that lists perpetuals (sleeve_fund.venues), the
    perp market is that venue's own: its fees, spread and settled funding. Elsewhere a perp is simulated on
    the venue's spot prices at a low-fee perp venue's published rates."""
    m = market_of(params)
    if m == SPOT:
        return None
    if venue is not None:
        from sleeve_fund.venues import venue as venue_profile

        profile = venue_profile(venue)
        if profile.perpetual:
            if m != PERP:
                raise ValueError("this venue trades its own perpetuals: choose the perp market")
            return native_terms(profile)
    return LOW_FEE_PERP if m == PERP else VENUE_FEE_PERP


def native_terms(profile) -> PerpTerms:
    """A perpetual venue's own terms: its fee schedule and spread (None: the venue's), the funding it settles,
    and a 0.5% maintenance margin, cautious for BTC and ETH (0.4% at the smallest tier), light for small
    instruments (sleeve_fund.risk keeps the stop well inside liquidation either way)."""
    return PerpTerms(profile.label, None, None, LOW_FEE_PERP.funding_rate, profile.funding_hours,
                     LOW_FEE_PERP.maintenance_margin, funding_venue=profile.name)


def check_venue(params: dict | None, venue: str) -> None:
    """Raises ValueError when the market can't trade on the venue: a perpetual venue has no spot."""
    from sleeve_fund.venues import venue as venue_profile

    profile = venue_profile(venue)
    if profile.perpetual and market_of(params) != PERP:
        raise ValueError("this venue lists perpetuals only: set market = \"perp\"")


def fees_for(params: dict | None, venue_fees: FeeSchedule, venue: str | None = None) -> FeeSchedule:
    """The fee schedule a strategy pays: its market's, or the venue's."""
    t = terms(params, venue)
    return t.fees if t is not None and t.fees is not None else venue_fees


def half_spread_for(params: dict | None, venue_half_spread: float, venue: str | None = None) -> float:
    t = terms(params, venue)
    return t.half_spread if t is not None and t.half_spread is not None else venue_half_spread


def funding_times(after: datetime, until: datetime, hours: tuple[int, ...]) -> list[datetime]:
    """The funding times in (after, until], oldest first."""
    out, day = [], datetime(after.year, after.month, after.day, tzinfo=timezone.utc)
    while day <= until:
        for h in hours:
            t = day + timedelta(hours=h)
            if after < t <= until:
                out.append(t)
        day += timedelta(days=1)
    return out


def funding_interval(hours: tuple[int, ...]) -> timedelta:
    """The time between settlements on a schedule of UTC hours: the shortest step, wrapping past midnight."""
    h = sorted(hours)
    return timedelta(hours=min((b - a) % 24 or 24 for a, b in zip(h, h[1:] + h[:1])))


def baseline_rate(terms: PerpTerms) -> float:
    """The fixed rate charged for a settlement whose rate the venue's records lack: 0.01% is the 8-hour figure, so
    it is scaled to the schedule's interval, a 1-hour settlement paying an eighth of it (Advisor, 6 Oct 2026)."""
    return abs(terms.funding_rate) * (funding_interval(terms.funding_hours) / timedelta(hours=8))


def settlement_times(after: datetime, until: datetime, hours: tuple[int, ...], settled=None,
                     published: timedelta | None = None) -> list[datetime]:
    """The funding settlements in (after, until], oldest first, to the minute. With the venue's settled rates
    (`settled`, indexed by settlement time), its own times: a symbol moved from 8-hourly to 4- or 1-hourly
    settlements pays every one (QA P1-O1). A gap between two records wider than the interval before it is missing
    settlements at that interval (charged the baseline, QA P1-O17), unless the venue lengthened its interval: the
    step after the gap is as wide (or, at the newest record, the venue's `published` interval is). Past the newest
    record, the venue's latest interval carries on from it (paper, before the venue publishes the next rate);
    before the first record, and with no records, the venue profile's fixed `hours`."""
    if settled is None or len(settled) == 0:
        return funding_times(after, until, hours)
    idx = settled.index.round("min")
    first, last = idx[0].to_pydatetime(), idx[-1].to_pydatetime()
    out = funding_times(after, min(until, first - timedelta(seconds=1)), hours) if after < first else []
    out += [t.to_pydatetime() for t in idx[(idx > after) & (idx <= until)]]
    out += _missing_between(idx.as_unit("ns"), after, until, hours, published)
    if until > last:
        step = latest_interval(settled)
        if step is None:  # a single record (a new listing): the fixed hours after it
            out += funding_times(max(after, last), until, hours)
        else:
            t = last + step
            while t <= until:
                if t > after:
                    out.append(t)
                t += step
    return sorted(set(out))


def _missing_between(idx, after: datetime, until: datetime, hours: tuple[int, ...],
                     published: timedelta | None) -> list[datetime]:
    """The settlements in (after, until] missing from a gap between the venue's records (settlement_times): a gap
    wider than the profile's interval, or one wider than the interval before it that the venue then kept (a lost
    record after a move to shorter settlements), is filled at the shorter of the two. A gap no wider than the
    profile's interval that ends the records is the venue back on (or moved towards) the schedule, not a loss; a
    lengthening is only read as one where the venue publishes it (`published`, the newest gap; DA-11), so until
    then a longer interval reads as missing settlements, charged the baseline (the adverse side)."""
    out: list[datetime] = []
    fixed = funding_interval(hours)
    lo = max(int(idx.searchsorted(after, side="right")), 1)  # the first record past `after`, the gap before it
    hi = min(int(idx.searchsorted(until, side="left")) + 1, len(idx))  # up to the first record at or past `until`
    for i in range(lo, hi):
        a, b = idx[i - 1].to_pydatetime(), idx[i].to_pydatetime()
        before = (a - idx[i - 2].to_pydatetime()) if i >= 2 else fixed
        step = min(fixed, before) if before > timedelta(0) else fixed
        gap = b - a
        later = (idx[i + 1].to_pydatetime() - b) if i + 1 < len(idx) else None
        if gap <= step or (gap <= fixed and (later is None or later >= gap)):
            continue  # no gap, or the venue moved back towards the schedule
        if later is None and published is not None and gap <= published:
            continue  # the venue's published interval is this wide: lengthened, not lost
        t = a + step
        while t < b:
            if after < t <= until:
                out.append(t)
            t += step
    return out


def settlement_wait(ts: datetime, settled, wait: timedelta) -> timedelta:
    """How long paper waits after settlement `ts` for the venue's record before charging the baseline rate:
    `wait`; for a settlement foreseen past its newest record, one of the venue's latest intervals more, plus `wait`
    again for the store's refresh to bring in the next record. If the venue has lengthened its interval, the newer
    record that skips the foreseen time lands within that, and the time is then no settlement at all
    (settlement_times), so it is never charged (no phantom baseline charge)."""
    step = latest_interval(settled)
    if step is None or ts <= settled.index[-1].round("min").to_pydatetime():
        return wait
    return 2 * wait + step


def latest_interval(settled) -> timedelta | None:
    """The venue's settlement interval as its two newest records show it, or None with fewer than two."""
    if settled is None or len(settled) < 2:
        return None
    idx = settled.index.round("min")
    step = (idx[-1] - idx[-2]).to_pytimedelta()
    return step if step > timedelta(0) else None


def isolated_margin(qty: float, entry: float, leverage: float, balance: float | None = None) -> float:
    """The margin an isolated perpetual position puts up: its notional at entry over the leverage it is
    opened at (the risk profile's cap), never more than the balance there is to put up. The rest of the
    strategy's equity is not at risk to the venue's liquidation. The one margin figure for paper,
    backtest, the dashboard and the demo copy (set to isolated at the same leverage)."""
    margin = abs(qty) * entry / max(leverage, 1e-9)
    return min(margin, max(balance, 0.0)) if balance is not None else margin


def gap_loss_cap(qty: float, entry: float, leverage: float, balance: float, taker: float, liq: float | None) -> float:
    """The most an isolated perpetual position can lose however far the price gaps (Independent Quant Advisor, QA
    P1-D3): its whole isolated margin, plus the taker fee on the close at its liquidation price. The engine books a
    gap past the bankruptcy price at this (the insurance fund takes the rest), and the Risk page's stress rows use it."""
    return isolated_margin(qty, entry, leverage, balance) + taker * abs(qty) * (liq or 0.0)


def isolated_liquidation(cash: float, qty: float, entry: float, leverage: float, maintenance: float) -> float | None:
    """The liquidation price of a position of qty opened at entry on isolated margin at this leverage.
    cash is the strategy's spot-style cash (its balance less qty x entry). None when flat, or when no
    positive price liquidates it (a long at 1x or less is fully paid for)."""
    if qty == 0 or entry <= 0:
        return None
    margin = isolated_margin(qty, entry, leverage, cash + qty * entry)
    return liquidation_price(margin - qty * entry, qty, maintenance)


def liquidation_price(cash: float, qty: float, maintenance: float) -> float | None:
    """The price at which a position's equity (cash + qty x price) falls to the maintenance margin on
    its value, cash being the margin backing it less qty x entry (isolated_liquidation). None when flat,
    or when no positive price liquidates it (a long fully paid for in cash)."""
    if qty == 0:
        return None
    # cash + qty * p = maintenance * |qty| * p  =>  p = cash / (maintenance * |qty| - qty)
    denom = maintenance * abs(qty) - qty
    if denom == 0:
        return None
    p = cash / denom
    return p if p > 0 else None
