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
from sleeve_fund.margin import isolated_liquidation, isolated_margin, liquidation_price  # noqa: F401 (re-exported)

SPOT = "spot"
PERP = "perp"
PERP_VENUE_FEES = "perp-venue-fees"
MARKETS = (SPOT, PERP, PERP_VENUE_FEES)
# Research only, never offered as a choice: the simulated perp with funding at FUNDING_STRESS_RATE, the study's
# funding-stress line (Advisor, 7 Oct 2026: the 0.01% baseline is light in strong uptrends).
PERP_FUNDING_STRESS = "perp-funding-stress"
FUNDING_STRESS_RATE = 0.0003
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
    if m not in MARKETS and m != PERP_FUNDING_STRESS:
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
    if m == PERP_FUNDING_STRESS:
        from dataclasses import replace

        return replace(LOW_FEE_PERP, label=f"{LOW_FEE_PERP.label}, funding at {FUNDING_STRESS_RATE:.2%}",
                       funding_rate=FUNDING_STRESS_RATE)
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
                     published: timedelta | None = None, snapped_idx=None) -> list[datetime]:
    """The funding settlements in (after, until], oldest first, to the minute. With the venue's settled rates
    (`settled`, indexed by settlement time), its own times: a symbol moved from 8-hourly to 4- or 1-hourly
    settlements pays every one (QA P1-O1). A gap between two records wider than the interval before it is missing
    settlements at that interval (charged the baseline, QA P1-O17), unless the venue lengthened its interval: the
    step after the gap is as wide (or, at the newest record, the venue's `published` interval is). Past the newest
    record, the venue's published interval (or its latest step, counting provisionally missing settlements, no wider
    than the profile's) carries on from it
    (paper, before the venue publishes the next rate); before the first record, and with no records, the venue
    profile's fixed `hours`. `snapped_idx` is `snapped(settled)` computed once by a caller whose records are fixed
    (a backtest calls this every hour: re-snapping thousands of records each time made it many times slower, QA
    P1-O17a-17)."""
    if settled is None or len(settled) == 0:
        return funding_times(after, until, hours)
    idx = snapped(settled) if snapped_idx is None else snapped_idx
    first, last = idx[0].to_pydatetime(), idx[-1].to_pydatetime()
    out = funding_times(after, min(until, first - timedelta(seconds=1)), hours) if after < first else []
    out += [t.to_pydatetime() for t in idx[(idx > after) & (idx <= until)]]
    out += _missing_between(idx.as_unit("ns"), after, until, hours, published)
    if until > last:
        # The venue's published interval where it gives one; else its latest step, no wider than the profile's: a
        # wider one is lost records (filled above), and a lengthening is honoured only where published (DA-11).
        step = published or _latest_step(idx, hours)
        if step is not None and not published:
            step = min(step, funding_interval(hours))
        if step is None:  # a single record (a new listing): the fixed hours after it
            out += funding_times(max(after, last), until, hours)
        else:
            t = last + step
            while t <= until:
                if t > after:
                    out.append(t)
                t += step
    return sorted(set(out))


# A record within this either side of a settlement is that settlement's rate, published late or early: one window
# for the gap reader (snapped) and the engine's match (funding.MATCH is this), so one rate is never charged twice
# (Advisor, 7 Oct 2026, QA P1-O17a-14).
SNAP_WINDOW = timedelta(minutes=1)
_ONE_MINUTE = SNAP_WINDOW


def snapped(settled):
    """The venue's records' settlement times: each to the minute, and one within a minute either side of the hour
    is that hour's settlement, published late or early (Advisor, 7 Oct 2026, QA P1-O17a-14), so a record at 08:01
    is the 08:00 settlement, read as no gap and charged once. One to one: of two records snapping to one hour the
    nearer takes it, the other keeps its own minute. The records keep their own stamps (funding.snap_note)."""
    import pandas as pd

    idx = settled.index.round("min")
    hour = idx.round("h")
    near = abs(idx - hour) <= pd.Timedelta(_ONE_MINUTE)
    out, taken = list(idx), {}
    for i in (j for j in range(len(idx)) if near[j]):
        best = taken.get(hour[i])
        if best is None or abs(idx[i] - hour[i]) < abs(idx[best] - hour[i]):
            taken[hour[i]] = i
    for h, i in taken.items():
        out[i] = h
    return pd.DatetimeIndex(out).sort_values()


def _missing_between(idx, after: datetime, until: datetime, hours: tuple[int, ...],
                     published: timedelta | None) -> list[datetime]:
    """The settlements in (after, until] missing from a gap between the venue's records (settlement_times): a gap
    wider than the profile's interval, or one wider than the interval before it that the venue then kept (a lost
    record after a move to shorter settlements), is filled at the shorter of the two. A gap no wider than the
    profile's interval is the venue back on (or moved towards) the schedule, not a loss, only once the step after
    it is as wide; ending the records, it is provisionally missing (Advisor, 7 Oct 04:31); a
    lengthening is only read as one where the venue publishes it (`published`, the newest gap; DA-11), so until
    then a longer interval reads as missing settlements, charged the baseline (the adverse side)."""
    out: list[datetime] = []
    fixed = funding_interval(hours)
    lo = max(int(idx.searchsorted(after, side="right")), 1)  # the first record past `after`, the gap before it
    hi = min(int(idx.searchsorted(until, side="left")) + 1, len(idx))  # up to the first record at or past `until`
    for i in range(lo, hi):
        a, b = idx[i - 1].to_pydatetime(), idx[i].to_pydatetime()
        before = _whole_hours(a - idx[i - 2].to_pydatetime()) if i >= 2 else fixed
        step = min(fixed, before) if before > timedelta(0) else fixed
        gap = b - a
        later = (idx[i + 1].to_pydatetime() - b) if i + 1 < len(idx) else None
        if gap <= step or (gap <= fixed and later is not None and later >= gap):
            continue  # no gap, or the venue moved back towards the schedule, as the step after it shows
        # With no later record yet, a gap wider than the step before it is provisionally missing (the adverse
        # default): reversed by its own correction if the next step shows the move back (Advisor, 7 Oct 04:31).
        # TODO(DA-11): `published` is the venue's interval now, so it exempts only the newest gap; once a later record
        # lands, a lengthened gap reads as missing again. Settle with each instrument's interval history (CR minor 2).
        if later is None and published is not None and gap <= published:
            continue  # the venue's published interval is this wide: lengthened, not lost
        t = a + step
        while t < b - _ONE_MINUTE:  # a record within a minute is that settlement's own, charged once (QA P1-O17a-14)
            if after < t <= until:
                out.append(t)
            t += step
    return out


def settlement_wait(ts: datetime, settled, wait: timedelta, hours: tuple[int, ...] | None = None) -> timedelta:
    """How long paper waits after settlement `ts` for the venue's record before charging the baseline rate: `wait`,
    whether the time is recorded or foreseen past the newest record, so a missing rate is alerted, charged and blocks
    entries 15 minutes after it was due (Advisor, 7 Oct 2026, QA P1-O17a-13). A foreseen time the venue's next record
    shows was no settlement (its interval lengthened) has its baseline reversed by its own journaled correction
    (LongFlatStrategy._reverse_unsettled), never by waiting a whole interval first."""
    return wait


def _latest_step(idx, hours: tuple[int, ...]) -> timedelta | None:
    """The step to the newest record, counting the settlements missing before it (Advisor, 7 Oct 04:31): after
    12:00 on 4 h and a lost 16:00, the newest gap 12:00 -> 20:00 is a 4 h step, not 8 h."""
    if len(idx) < 2:
        return None
    last = idx[-1].to_pydatetime()
    filled = _missing_between(idx.as_unit("ns"), idx[-2].to_pydatetime(), last, hours, None)
    step = last - (filled[-1] if filled else idx[-2].to_pydatetime())
    return _whole_hours(step) if step > timedelta(0) else None


def _whole_hours(step: timedelta) -> timedelta:
    """A step between records as the venue's interval, in whole hours (at least one): a record stamped off its hour
    outside the snap window (08:02) leaves steps of 7 h 58 min and 8 h 2 min, still the 8-hourly schedule, never
    a 2-minute one (QA P1-O17a-14, outside-the-window pin)."""
    return max(timedelta(hours=1), timedelta(hours=round(step / timedelta(hours=1))))


def latest_interval(settled) -> timedelta | None:
    """The venue's settlement interval as its two newest records show it, or None with fewer than two."""
    if settled is None or len(settled) < 2:
        return None
    idx = snapped(settled)
    step = (idx[-1] - idx[-2]).to_pytimedelta()
    return step if step > timedelta(0) else None


def gap_loss_cap(qty: float, entry: float, leverage: float, balance: float, taker: float, liq: float | None) -> float:
    """The most an isolated perpetual position can lose however far the price gaps (Independent Quant Advisor, QA
    P1-D3): its whole isolated margin, plus the taker fee on the close at its liquidation price. The engine books a
    gap past the bankruptcy price at this (the insurance fund takes the rest), and the Risk page's stress rows use it."""
    return isolated_margin(qty, entry, leverage, balance) + taker * abs(qty) * (liq or 0.0)


@dataclass(frozen=True)
class LiquidationBooking:
    """What an isolated-margin liquidation books (GAP-LIQ-CAP, Independent Quant Advisor 6 Oct 23:42 and 7 Oct 00:19).

    fill_px: the bankruptcy price, where the fill is booked, gapped or not: there the price move loses exactly the
    posted margin. loss: X, that margin plus the entry and liquidation fees, the one figure the strategy books (the
    same X as the RAL halt's). market_px: where the market actually closed it. insurance: how far the market went past
    bankruptcy, in money, which the venue's insurance fund covers; forfeited: the margin the venue kept when it closed
    short of bankruptcy. Both are journal diagnostics only, never in any P&L line, equity mark or trip."""

    fill_px: float
    loss: float
    market_px: float
    insurance: float
    forfeited: float


def bankruptcy_price(qty: float, entry: float, leverage: float, balance: float | None = None) -> float:
    """The price at which an isolated position of signed qty has lost exactly its posted margin (never below 0)."""
    return max(entry - isolated_margin(qty, entry, leverage, balance) / qty, 0.0)


def liquidation_booking(qty: float, entry: float, exit_px: float, leverage: float, entry_fee: float,
                        liquidation_fee: float, balance: float | None = None) -> LiquidationBooking:
    """The booking for a position of signed qty at average entry `entry`, liquidated by the market at an average
    exit_px, on isolated margin at this leverage (balance: what there was to put up, as isolated_margin). The fees are
    the money charged (entry: on the whole liquidated qty, adds included; liquidation: qty x the liquidation trigger
    price x the rate, never on the booked fill (Advisor 7 Oct 00:19))."""
    if qty == 0 or entry <= 0 or exit_px <= 0 or leverage <= 0:
        raise ValueError("a liquidation needs a position, positive prices and a positive leverage")
    if entry_fee < 0 or liquidation_fee < 0:
        raise ValueError("fees are money charged, never negative")
    margin = isolated_margin(qty, entry, leverage, balance)
    bankrupt = bankruptcy_price(qty, entry, leverage, balance)
    past = -qty * (exit_px - bankrupt)  # > 0: the market went past bankruptcy; < 0: it closed short of it
    return LiquidationBooking(fill_px=bankrupt, loss=margin + entry_fee + liquidation_fee, market_px=exit_px,
                              insurance=max(past, 0.0), forfeited=max(-past, 0.0))
