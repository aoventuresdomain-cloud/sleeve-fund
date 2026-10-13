"""P2-5 regime, the data side (DA design data-architect/p2-5-data-design.md): which contracts made up the universe on a
day, and the universe's efficiency ratio (ER) with its low / mid / high state. Pure functions on daily data, so the
nightly job, a backtest and a test all compute the same rows from the same inputs.

Point in time throughout: a day's row reads only daily closes up to its `data_through`, which is before the day itself
(the lag), and the tercile cuts read only rows dated before it. The weight a state maps to is not here: QD's weighting
reads the state, and an "unknown" state is weight 1.0 there (Advisor spec 4.6).

Settings the Advisor fixes (N, lag, universe hysteresis, minimum listing age, the "unknown" rule) are arguments, never
defaults, so nothing runs on a guess."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from statistics import median

STATES = ("low", "mid", "high", "unknown")
LOOKBACKS = (20, 30)  # the spec's 20-day lookback and its one allowed variant
NOTIONAL_DAYS = 30  # the universe ranks by 30-day quote notional
CUT_DAYS = 365  # the tercile cuts come from the trailing year of the series
HYSTERESIS = Decimal("0.03")  # about ±0.03 around each cut


@dataclass(frozen=True)
class Listing:
    """One contract as `instruments` holds it: listed_at is its first trading day, delisted_at its last (None:
    still listed)."""
    symbol: str
    underlying: str
    listed_at: date
    delisted_at: date | None = None


@dataclass(frozen=True)
class Member:
    symbol: str
    underlying: str
    rank: int
    notional: Decimal


@dataclass(frozen=True)
class RegimeRow:
    date: date
    lookback: int
    universe_date: date
    data_through: date
    er: Decimal | None
    state: str
    lo: Decimal | None
    hi: Decimal | None
    members: int  # how many members had every close in the window


def notional(day_notional: dict[date, Decimal], through: date, days: int = NOTIONAL_DAYS) -> Decimal | None:
    """The quote notional traded over the `days` days ending at `through`, or None when any of them is missing (a
    contract is never ranked on part of a window)."""
    total = Decimal(0)
    for i in range(days):
        d = through - timedelta(days=i)
        if d not in day_notional:
            return None
        total += day_notional[d]
    return total


def universe(day: date, listings: list[Listing], day_notional: dict[str, dict[date, Decimal]], *, n: int,
             keep_within: int, min_age_days: int, lag_days: int, previous: frozenset[str] = frozenset()) -> list[Member]:
    """The universe for `day`: the top `n` by 30-day quote notional through `day - lag_days`, one contract per
    underlying (its highest-notional one), among contracts listed at least `min_age_days` before then and not delisted
    by then. Hysteresis: a member of `previous` stays while it ranks within `keep_within` (>= n); the places left go
    to the best of the rest. Ranked by notional, ties by symbol, so the order never depends on input order."""
    if not 0 < n <= keep_within:
        raise ValueError(f"need 0 < n <= keep_within, got n={n}, keep_within={keep_within}")
    if lag_days < 1:
        raise ValueError("the lag is at least one day: a day's universe never reads that day's own close")
    through = day - timedelta(days=lag_days)
    best: dict[str, tuple[Decimal, str]] = {}
    for x in listings:
        if x.listed_at > through - timedelta(days=min_age_days):
            continue
        if x.delisted_at is not None and x.delisted_at < through:
            continue
        v = notional(day_notional.get(x.symbol, {}), through)
        if v is None:
            continue
        held = best.get(x.underlying)
        if held is None or (-v, x.symbol) < (-held[0], held[1]):  # the higher notional; ties to the lower symbol
            best[x.underlying] = (v, x.symbol)
    ranked = sorted(((v, s, u) for u, (v, s) in best.items()), key=lambda r: (-r[0], r[1]))
    rank = {s: i + 1 for i, (_, s, _) in enumerate(ranked)}
    kept = [r for r in ranked if r[1] in previous and rank[r[1]] <= keep_within][:n]
    chosen = {r[1] for r in kept}
    for r in ranked:
        if len(chosen) >= n:
            break
        chosen.add(r[1])
    return [Member(s, u, rank[s], v) for v, s, u in ranked if s in chosen]


def efficiency_ratio(closes: list[Decimal]) -> Decimal:
    """Kaufman's ER over the closes given (lookback + 1 of them): |last - first| over the sum of |day-to-day change|;
    0 with no movement at all, as the indicator block reads."""
    path = sum((abs(b - a) for a, b in zip(closes, closes[1:])), Decimal(0))
    return Decimal(0) if path == 0 else abs(closes[-1] - closes[0]) / path


def universe_er(members: list[Member], closes: dict[str, dict[date, Decimal]], through: date,
                lookback: int) -> tuple[Decimal | None, int]:
    """The median ER across the members that have every daily close in the window ending at `through`, and how many
    did. A member with a missing close is left out, never filled in."""
    if lookback not in LOOKBACKS:
        raise ValueError(f"the lookback is one of {LOOKBACKS}, got {lookback}")
    ers = []
    for m in members:
        series = closes.get(m.symbol, {})
        window = [series.get(through - timedelta(days=i)) for i in range(lookback, -1, -1)]
        if all(c is not None for c in window):
            ers.append(efficiency_ratio(window))
    return (median(sorted(ers)) if ers else None), len(ers)


def cuts(history: list[Decimal]) -> tuple[Decimal, Decimal]:
    """The terciles of the ER values given (the trailing year, oldest first): the 1/3 and 2/3 points, linearly
    interpolated between order statistics."""
    xs = sorted(history)

    def q(p: Decimal) -> Decimal:
        pos = p * (len(xs) - 1)
        i = int(pos)
        frac = pos - i
        return xs[i] if i + 1 >= len(xs) else xs[i] + (xs[i + 1] - xs[i]) * frac

    return q(Decimal(1) / 3), q(Decimal(2) / 3)


def state_of(er: Decimal, lo: Decimal, hi: Decimal, previous: str | None, band: Decimal = HYSTERESIS) -> str:
    """low below lo, high above hi, mid between; with hysteresis, the state moves off `previous` only once the ER is
    `band` past the cut it crosses (from unknown or None there is nothing to hold, so the plain state)."""
    plain = "low" if er < lo else "high" if er > hi else "mid"
    if previous not in ("low", "mid", "high") or plain == previous:
        return plain
    if previous == "low":
        return "high" if er > hi + band else "mid" if er > lo + band else "low"
    if previous == "high":
        return "low" if er < lo - band else "mid" if er < hi - band else "high"
    return "low" if er < lo - band else "high" if er > hi + band else "mid"  # from mid


def series(days: list[date], listings: list[Listing], day_notional: dict[str, dict[date, Decimal]],
           closes: dict[str, dict[date, Decimal]], *, lookback: int, n: int, keep_within: int, min_age_days: int,
           lag_days: int, min_members: int, min_history: int) -> tuple[list[RegimeRow], dict[date, list[Member]]]:
    """The regime rows for `days` (oldest first, consecutive) and the universe each used. A day is "unknown" when
    fewer than `min_members` members had a full window, or fewer than `min_history` known ER values precede it in the
    trailing year (no cuts yet). Each day reads only closes up to day - lag_days and the rows before it."""
    rows: list[RegimeRow] = []
    universes: dict[date, list[Member]] = {}
    previous: frozenset[str] = frozenset()
    last_state: str | None = None
    for day in days:
        through = day - timedelta(days=lag_days)
        members = universe(day, listings, day_notional, n=n, keep_within=keep_within, min_age_days=min_age_days,
                           lag_days=lag_days, previous=previous)
        universes[day] = members
        previous = frozenset(m.symbol for m in members)
        er, counted = universe_er(members, closes, through, lookback)
        # every day with a valid ER counts toward the cuts, including the early days that were unknown only for want
        # of a year of history
        year = [r.er for r in rows if r.er is not None and r.members >= min_members
                and r.date >= day - timedelta(days=CUT_DAYS)]
        lo = hi = None
        if er is None or counted < min_members:
            er, state = (er if counted else None), "unknown"
        elif len(year) < min_history:
            state = "unknown"
        else:
            lo, hi = cuts(year)
            state = state_of(er, lo, hi, last_state)
        if state != "unknown":
            last_state = state
        rows.append(RegimeRow(day, lookback, day, through, er, state, lo, hi, counted))
    return rows, universes
