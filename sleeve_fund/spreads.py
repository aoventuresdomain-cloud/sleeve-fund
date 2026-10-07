"""Which bid-ask spread a backtest charges, and why.

Backtest bars carry trade prices, so a market order there fills at the bar's price, while a real
one pays the ask to buy and takes the bid to sell. Backtests therefore charge half the spread on
every order that takes liquidity. Paper sleeves subscribe to the venue's quotes, fill on the bid
or ask themselves, and record each instrument's typical spread; backtests use the latest of those
measurements, else the venue profile's cautious assumption, and every result says which.

The measurements are a point-in-time series (SPREAD-PIT, Independent Quant Advisor 6 Oct 23:42): each is in force
from its measured_at (the end of its sampling window) until the next, and a backtest or paper reads the value in force
at the minute it fills, never a later measurement (SpreadSeries).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sleeve_fund.venues import venue as venue_profile


@dataclass(frozen=True)
class SpreadQuote:
    half_spread: float  # a fraction of the mid price
    source: str  # "measured" or "assumed"
    samples: int
    measured_at: datetime | None

    @property
    def text(self) -> str:
        full = f"{2 * self.half_spread:.3%} bid-ask spread"
        if self.source == "measured":
            return f"{full}, the median of {self.samples:,} live quotes on {self.measured_at:%d %b %Y}"
        return f"{full}, assumed; no paper strategy has measured this instrument yet"

    @property
    def short(self) -> str:
        """One line for a result's header: "0.10% spread (assumed)"."""
        how = f"measured {self.measured_at.day} {self.measured_at:%b %Y}" if self.source == "measured" else "assumed"
        return f"{2 * self.half_spread:.2%} spread ({how})"


def resolve(venue: str | None, instrument: str, store=None, at: datetime | None = None,
            strict: bool = False) -> SpreadQuote:
    """The spread to charge for `instrument` (e.g. "BTC/USD") on `venue` now, or with `at` the one in force then (the
    latest measurement from at or before it, else the venue's assumption: SPREAD-PIT). strict: a failed read raises
    instead of assuming."""
    profile = venue_profile(venue)
    row = None
    if store is not None:
        try:
            row = store.latest_spread(profile.name, instrument, at=at)
        except Exception:  # noqa: BLE001 - an old database without the table: assume
            if strict:
                raise
            row = None
    if row:
        return SpreadQuote(row["half_spread"], "measured", row["samples"], row["measured_at"])
    return SpreadQuote(profile.assumed_half_spread, "assumed", 0, None)


@dataclass(frozen=True)
class SpreadSeries:
    """The half spread in force at each time: `points` are (effective-from ns, SpreadQuote), oldest first, each in
    force from its time until the next; before the first, `assumed` (the venue's cautious assumption) is."""

    points: tuple = ()
    assumed: SpreadQuote = SpreadQuote(0.0, "assumed", 0, None)
    _froms: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_froms", tuple(t for t, _ in self.points))

    def quote_at(self, ts_ns) -> SpreadQuote:
        """The quote in force at a time: UNIX ns, or an aware datetime (naive is taken as UTC)."""
        if isinstance(ts_ns, datetime):
            ts_ns = _ns(ts_ns if ts_ns.tzinfo else ts_ns.replace(tzinfo=timezone.utc))
        i = bisect_right(self._froms, ts_ns)
        return self.points[i - 1][1] if i else self.assumed

    def at_many(self, ts_ns):
        """The half spread in force at each of an array of UNIX ns times, as a numpy array."""
        import numpy as np

        values = np.array([self.assumed.half_spread] + [q.half_spread for _, q in self.points])
        return values[np.searchsorted(np.array(self._froms, dtype=np.int64), np.asarray(ts_ns, dtype=np.int64),
                                      side="right")]

    def at(self, ts_ns) -> float:
        return self.quote_at(ts_ns).half_spread

    def peak(self, start_ns: int, end_ns: int) -> float:
        """The widest half spread in force at any time in [start, end]."""
        return max(q.half_spread for _, q in self.used(start_ns, end_ns))

    def used(self, start_ns: int, end_ns: int) -> list[tuple[int, SpreadQuote]]:
        """The quotes in force over [start, end] as (in force from ns, quote), oldest first: the one at start, then
        each that took over."""
        return [(start_ns, self.quote_at(start_ns))] + [(t, q) for t, q in self.points if start_ns < t <= end_ns]

    def report(self, start_ns: int, end_ns: int) -> dict:
        """Which values a run over [start, end] charged, measured or assumed, and from when: the assumption until
        the first measurement (when the run began before it), how many measurements took over, and their range."""
        used = self.used(start_ns, end_ns)
        measured = [(t, q) for t, q in used if q.source == "measured"]
        first = used[0][1]
        return {
            "assumed": first.half_spread if first.source != "measured" else None,
            "assumed_until": (_iso(measured[0][0]) if measured else _iso(end_ns)) if first.source != "measured" else None,
            "measured_from": _iso(measured[0][0]) if measured else None,
            "measurements": len(measured),
            "low": min(q.half_spread for _, q in used),
            "high": max(q.half_spread for _, q in used),
            "at_start": first.half_spread,
            "at_end": used[-1][1].half_spread,
        }

    def text(self, start_ns: int, end_ns: int) -> str:
        """One line for a result: which spread it charged and from when."""
        r = self.report(start_ns, end_ns)
        if not r["measurements"]:
            how = "as given" if self.assumed.source == "given" else "assumed; no measurement was in force"
            return f"{2 * r['assumed']:.3%} bid-ask spread throughout, {how}"
        words = (f"{2 * r['low']:.3%} to {2 * r['high']:.3%}" if r["low"] != r["high"] else f"{2 * r['low']:.3%}")
        if r["assumed"] is not None:
            return (f"{2 * r['assumed']:.3%} bid-ask spread assumed until {r['assumed_until']}, then the "
                    f"{r['measurements']:,} measurements in force at each time ({words} over the whole run)")
        return f"the {r['measurements']:,} measured bid-ask spreads in force at each time ({words})"

    def plus(self, extra: float) -> SpreadSeries:
        """Every value widened by `extra` (the cost ladder's slippage step)."""
        def wider(q):
            return SpreadQuote(q.half_spread + extra, q.source, q.samples, q.measured_at)
        return SpreadSeries(tuple((t, wider(q)) for t, q in self.points), wider(self.assumed))

    @classmethod
    def constant(cls, half_spread: float, source: str = "assumed") -> SpreadSeries:
        return cls((), SpreadQuote(half_spread, source, 0, None))


def series(venue: str | None, instrument: str, store=None, strict: bool = False) -> SpreadSeries:
    """Every measurement of `instrument` on `venue`, as the series a backtest or paper reads (the venue's assumption
    before the first). Two strategies measuring one instrument at the same time: the later row wins, as in
    resolve. strict: a failed read raises (paper's hourly reload keeps the series it has) instead of assuming."""
    profile = venue_profile(venue)
    assumed = SpreadQuote(profile.assumed_half_spread, "assumed", 0, None)
    rows = []
    if store is not None:
        try:
            rows = store.spread_series(profile.name, instrument)
        except Exception:  # noqa: BLE001 - an old database without the table: assume
            if strict:
                raise
            rows = []
    points: dict[int, SpreadQuote] = {}
    for r in rows:  # oldest first, ties by id: a later row at the same time replaces the earlier
        at = r["measured_at"] if r["measured_at"].tzinfo else r["measured_at"].replace(tzinfo=timezone.utc)
        points[_ns(at)] = SpreadQuote(r["half_spread"], "measured", r["samples"], at)
    return SpreadSeries(tuple(sorted(points.items(), key=lambda p: p[0])), assumed)


def _iso(ns: int) -> str:
    return f"{datetime.fromtimestamp(ns // 1_000_000_000, tz=timezone.utc):%Y-%m-%d %H:%M} UTC"


def _ns(at: datetime) -> int:
    """A datetime as UNIX ns, exact to the µs (a float timestamp loses the last digits)."""
    delta = at - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000
