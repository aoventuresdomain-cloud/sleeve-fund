"""Hub bar parity: the history store's 1-minute bars (as the market data hub wrote them) against the venue's own
1-minute candles over the same window, per instrument. Replaces the shadow run for P1-1 (v2 spec).

    python scripts/hub_parity.py --venue <venue> --hours 24 [--pair <BASE/QUOTE> ...] [--out report.md]

Reports, per instrument: minutes on both sides, minutes only on one side, minutes whose OHLC differ by more than
the tolerance or whose volume differs by more than 0.1%, the largest differences, the minutes the hub refilled
from the venue's REST API, any refill that disagreed with a stored bar, the hub bars the venue's candle replaced
(HistoryStore.provenance), and the
late-trade rate when the hub supplies its counts. Reads only; nothing is written to the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from sleeve_fund.history import OHLCV, HistoryStore

VOLUME_TOLERANCE = 0.001  # relative: venues revise volume by tiny amounts as late trades settle
# Late trades (reaching the hub after their minute was built) above this share of an instrument's trades over the
# window: the Independent Quant Advisor's trigger to widen the hub's grace (P1-1-DELAY, 17:05 UK).
LATE_ALERT_RATE = 0.0005


@dataclass
class Parity:
    pair: str
    start: pd.Timestamp
    end: pd.Timestamp
    both: int = 0
    venue_only: list[pd.Timestamp] = field(default_factory=list)  # the store is missing these minutes
    store_only: list[pd.Timestamp] = field(default_factory=list)  # the venue has no candle for these
    price_diffs: int = 0
    volume_diffs: int = 0
    worst_price: float = 0.0  # the largest OHLC difference, in price units
    worst_volume: float = 0.0  # the largest relative volume difference
    refilled: int = 0
    conflicts: int = 0
    # hub live bars the venue's candle replaced (P1-1-CANON): the hub's own differences, once the store holds the
    # venue's record. The P1-1-DELAY acceptance reads this, since the store itself then matches the venue.
    replaced: int = 0
    late_trades: tuple[int, int] | None = None  # (late, total) from the hub, when it reports them

    @property
    def ok(self) -> bool:
        return not (self.venue_only or self.store_only or self.price_diffs or self.volume_diffs)


def venue_minutes(loader, pair: str, start: pd.Timestamp, end: pd.Timestamp, cursor_at, max_pages: int = 100) -> pd.DataFrame:
    """The venue's closed 1-minute candles with open times in [start, end), from its history loader."""
    cursor, parts = cursor_at(start), []
    for _ in range(max_pages):
        bars, nxt, caught_up = loader(pair, cursor)
        if not bars.empty:
            parts.append(bars)
        if caught_up or bars.empty or bars.index[-1] >= end or nxt == cursor:
            break
        cursor = nxt
    if not parts:
        return pd.DataFrame(columns=OHLCV)
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()[OHLCV].astype(float)
    return df[(df.index >= start) & (df.index < end)]


def store_minutes(store: HistoryStore, venue: str, pair: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """The store's 1-minute bars with open times in [start, end). read() stamps bars at their close."""
    try:
        df = store.read(venue, pair, 1, start=start, end=end)
    except KeyError:
        return pd.DataFrame(columns=OHLCV)
    df = df.set_axis(df.index - pd.Timedelta("1min"))
    return df[(df.index >= start) & (df.index < end)]


def compare(pair: str, ours: pd.DataFrame, theirs: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
            price_tolerance: float = 0.0, provenance: list[dict] | None = None,
            late_trades: tuple[int, int] | None = None) -> Parity:
    p = Parity(pair, start, end, late_trades=late_trades)
    common = ours.index.intersection(theirs.index)
    p.both = len(common)
    p.venue_only = list(theirs.index.difference(ours.index))
    p.store_only = list(ours.index.difference(theirs.index))
    if len(common):
        a, b = ours.loc[common], theirs.loc[common]
        price = (a[["open", "high", "low", "close"]] - b[["open", "high", "low", "close"]]).abs().max(axis=1)
        vol = (a["volume"] - b["volume"]).abs() / b["volume"].abs().clip(lower=1e-12)
        vol[(a["volume"] == 0) & (b["volume"] == 0)] = 0.0
        p.price_diffs = int((price > price_tolerance).sum())
        p.volume_diffs = int((vol > VOLUME_TOLERANCE).sum())
        p.worst_price = float(price.max())
        p.worst_volume = float(vol.max())
    for rec in provenance or []:
        if rec["kind"] == "refill" and start <= pd.Timestamp(rec["first"]) < end:
            p.refilled += int(rec["minutes"])
        elif rec["kind"] == "conflict" and start <= pd.Timestamp(rec["minute"]) < end:
            p.conflicts += 1
        elif rec["kind"] == "replaced" and start <= pd.Timestamp(rec["minute"]) < end:
            p.replaced += 1
    return p


def _runs(minutes: list[pd.Timestamp]) -> str:
    if not minutes:
        return "none"
    out, first, prev = [], minutes[0], minutes[0]
    for m in minutes[1:] + [None]:
        if m is not None and m - prev == pd.Timedelta("1min"):
            prev = m
            continue
        out.append(f"{first:%d %b %H:%M}" + (f" to {prev:%H:%M}" if prev != first else ""))
        if m is not None:
            first = prev = m
    shown = ", ".join(out[:8])
    return shown + (f" and {len(out) - 8} more runs" if len(out) > 8 else "")


def markdown(venue: str, results: list[Parity]) -> str:
    start, end = results[0].start, results[0].end
    lines = [f"# Hub bar parity: {venue}", "",
             f"Window {start:%d %b %Y %H:%M} to {end:%d %b %Y %H:%M} UTC ({(end - start) / pd.Timedelta('1h'):.0f} hours). "
             "The store's 1-minute bars as the hub wrote them, against the venue's own 1-minute candles.", "",
             "| Instrument | Result | Both | Missing in store | Store only | OHLC diffs | Volume diffs | Worst OHLC diff | "
             "Refilled | Refill conflicts | Hub bars replaced by venue | Late trades |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in results:
        late = "not reported" if p.late_trades is None else (
            f"{p.late_trades[0]} of {p.late_trades[1]} ({p.late_trades[0] / max(p.late_trades[1], 1):.3%})")
        lines.append(f"| {p.pair} | {'match' if p.ok else 'DIFFERS'} | {p.both} | {len(p.venue_only)} | "
                     f"{len(p.store_only)} | {p.price_diffs} | {p.volume_diffs} | {p.worst_price:g} | {p.refilled} | "
                     f"{p.conflicts} | {p.replaced} | {late} |")
    for p in results:
        if p.late_trades is not None and p.late_trades[0] > LATE_ALERT_RATE * p.late_trades[1]:
            lines += ["", (f"**{p.pair}**: LATE TRADES ABOVE {LATE_ALERT_RATE:.2%} "
                           f"({p.late_trades[0] / p.late_trades[1]:.3%}): the trigger to widen the hub's grace.")]
    for p in results:
        if p.venue_only or p.store_only:
            lines += ["", f"**{p.pair}**: missing in store: {_runs(p.venue_only)}. Store only: {_runs(p.store_only)}."]
    return "\n".join(lines) + "\n"


def late_counts(store, venue_name: str, path=None) -> dict | None:
    """The hub's late-trade counts, {pair: (late, total)}: from `path` if given, else from the file the hub keeps
    beside the store (<store root>/hub-late-<VENUE>.json), so the report shows the rate without being asked
    (spec P1-1). None when neither exists."""
    import json
    from pathlib import Path

    path = Path(path) if path is not None else store.root / f"hub-late-{venue_name}.json"
    if not path.exists():
        return None
    return {k: tuple(v) for k, v in json.loads(path.read_text()).items()}


def run(store: HistoryStore, profile, pairs: list[str], start: pd.Timestamp, end: pd.Timestamp,
        price_tolerance: float = 0.0, late: dict | None = None) -> list[Parity]:
    if profile.minute_loader is None or profile.minute_cursor_at is None:
        raise ValueError(f"{profile.label} has no 1-minute history loader")
    out = []
    for pair in pairs:
        theirs = venue_minutes(profile.minute_loader, pair, start, end, profile.minute_cursor_at)
        ours = store_minutes(store, profile.name, pair, start, end)
        out.append(compare(pair, ours, theirs, start, end, price_tolerance, store.provenance(profile.name, pair),
                           (late or {}).get(pair)))
    return out


def window(hours: float, now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The last `hours` of closed minutes: the current minute is still forming, so it is left out."""
    end = (now or pd.Timestamp.now(tz="UTC")).floor("1min")
    return end - pd.Timedelta(hours=hours), end

