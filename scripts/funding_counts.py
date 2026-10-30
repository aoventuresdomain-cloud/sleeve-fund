"""The funding table of a pre-registration (family A, section 3; Advisor check 7): for each pair, the settlements due
over a window, how many the store kept from the venue, how many a signal would have read at a stand-in rate, and how
many are missing.

    python scripts/funding_counts.py --venue BINANCE --pairs BTC/USDT,ETH/USDT,SOL/USDT --start 2024-01-01 --end 2026-10-30

Reads only the stored settled rates (data/history/<venue>/<pair>/funding.json, or HISTORY_DIR). The funding blocks
read those rates and nothing else, so a stand-in (baseline) rate never reaches a signal: a missing settlement gives
the block no value and opens nothing (FundingDecile). The baseline-filled column is therefore 0 by construction; the
costs side charges the baseline for each missing settlement, which is the missing column. A settlement is due on the
venue's schedule as the kept rates show it: every kept one, every one inside a hole (funding.gaps), and every one
between the window's edges and the first or last kept rate. An interval change the rates alone can't tell from a hole
(funding.interval_changes) is listed as a possible hole, not counted."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from sleeve_fund import funding


def counts(venue: str, pair: str, start: pd.Timestamp, end: pd.Timestamp, root=None) -> dict:
    """{due, venue_sourced, baseline_filled, missing, possible_holes} for settlements on [start, end + 1 day)."""
    t = funding.rates(venue, pair, root).index
    if not len(t):
        return {"due": None, "venue_sourced": 0, "baseline_filled": 0, "missing": None, "possible_holes": []}
    until = end + pd.Timedelta(days=1)
    kept = t[(t >= start) & (t < until)]
    steps = list(t[1:] - t[:-1])
    pos = {x: i for i, x in enumerate(t)}

    def step_at(i: int) -> pd.Timedelta:
        """The schedule's interval around kept rate i: the shortest of the few intervals either side (as gaps reads
        it), so a venue's change from 8 hours to 4 is followed rather than one interval assumed throughout."""
        near = steps[max(i - 3, 0):i + 4]
        return min(near) if near else pd.Timedelta(hours=8)

    def inside(first: pd.Timestamp, last: pd.Timestamp, step: pd.Timedelta) -> int:
        """Settlements on the schedule first, first + step, ... up to last (both included), inside the window."""
        if last < first:
            return 0
        return sum(1 for k in range(int((last - first) / step) + 1) if start <= first + k * step < until)

    missing = 0
    for a, b in funding.gaps(venue, pair, root):
        step = step_at(pos[a])
        missing += inside(a + step, b - step / 2, step)  # every slot short of the next kept rate
    first, last = step_at(0), step_at(len(t) - 1)
    missing += inside(t[0] - first * int((t[0] - start) / first), t[0] - first / 2, first)  # before the first kept
    missing += inside(t[-1] + last, t[-1] + last * int((until - t[-1]) / last), last)  # after the last kept
    possible = [(a, b) for a, b in funding.interval_changes(venue, pair, root) if b > start and a < until]
    return {"due": len(kept) + missing, "venue_sourced": len(kept), "baseline_filled": 0, "missing": missing,
            "possible_holes": possible}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--venue", required=True)
    ap.add_argument("--pairs", required=True, help="comma-separated, e.g. BTC/USDT,ETH/USDT,SOL/USDT")
    ap.add_argument("--start", required=True, help="first UTC day, e.g. 2024-01-01")
    ap.add_argument("--end", required=True, help="last UTC day, included, e.g. 2026-10-30")
    ap.add_argument("--root", default=None, help="history folder (default HISTORY_DIR or data/history)")
    a = ap.parse_args(argv)
    start, end = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC")
    if end < start:
        print("the window runs backwards")
        return 2
    print(f"Funding settlements {a.start} to {a.end} ({a.venue.upper()}), stored rates only\n")
    print("| Asset | Settlements due | Venue-sourced | Baseline-filled | Missing |\n|---|---|---|---|---|")
    notes, empty = [], False
    for pair in a.pairs.split(","):
        c = counts(a.venue, pair.strip(), start, end, a.root)
        asset = pair.strip().split("/")[0].upper()
        if c["due"] is None:
            empty = True
            print(f"| {asset} | no rates stored | 0 | 0 | unknown |")
            continue
        print(f"| {asset} | {c['due']} | {c['venue_sourced']} | {c['baseline_filled']} | {c['missing']} |")
        notes += [f"- {asset}: interval change {x:%Y-%m-%d %H:%M} to {y:%Y-%m-%d %H:%M} UTC is a possible hole, not "
                  "counted" for x, y in c["possible_holes"]]
    print("\nBaseline-filled is in the signal: the funding blocks read stored rates only, so it is 0 by "
          "construction. The costs side charges the baseline for each missing settlement.")
    if notes:
        print("\n" + "\n".join(notes))
    return 2 if empty else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
