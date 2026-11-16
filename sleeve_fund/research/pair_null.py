"""Family C's random null (pre-registration, section 6, rev 2; Advisor 16 Nov): random long/short assignment.

A rank book holds one asset long and one short. The claim is selection, not timing, so the draws keep the Strategy's
change weeks (every Monday any of its legs changed) and at each one re-draw the ordered (long, short) pair uniformly
from the 6 a basket of three allows, holding it to the next change week. A week the block couldn't rank is a HOLD for
the Strategy, so it is never a change week, and the draws skip exactly the same weeks.

Both sides are priced by the same model, so the comparison is like with like: a leg opens at the open of the first
execution bar opening at or after its Monday close, at the leg notional the Strategy sized at that change week, and
pays half the spread and the taker fee on each side, plus funding at each settlement it is held over; a leg kept
across a change week is not re-traded; a leg still open at the window's end is marked at the last close, unpaid.
Legs carried into the window are valued from the first change week's open, for the Strategy and every draw alike.
Pre-fee is the same with the taker fee at 0 (spread and funding still charged, as round-0 kill 1 reads "before
fees"). The model's Strategy P&L is reconciled against the engine's book over the same span (RECONCILE_PP)."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sleeve_fund.research.random_entry import BAR, DRAWS, RECONCILE_PP

PRE_FEE_KILL = 90.0  # round-0 kill 2: below this percentile before fees
WEEK = pd.Timedelta(days=7)


@dataclass
class PairAsset:
    """One asset as the null prices it: its execution bars (stamped at their close, `bar_ns` long), its taker fee and
    half spread per side, and its funding settlements (times ns, rates, baseline flags; None: no funding)."""

    name: str
    close_ns: np.ndarray
    opens: np.ndarray
    closes: np.ndarray
    bar_ns: int
    taker: float
    half: float
    settle_ns: np.ndarray | None = None
    settle_rates: np.ndarray | None = None
    settle_baseline: np.ndarray | None = None

    def open_at(self, t_ns: int) -> float:
        """The open of the first bar opening at or after t (never inside the decision candle)."""
        i = int(np.searchsorted(self.close_ns, t_ns + self.bar_ns, side="left"))
        if i >= len(self.close_ns):
            raise ValueError(f"{self.name}: no execution bar opens at or after {pd.Timestamp(t_ns, tz='UTC')}")
        return float(self.opens[i])

    def close_at(self, t_ns: int) -> float:
        """The last close at or before t."""
        i = int(np.searchsorted(self.close_ns, t_ns, side="right")) - 1
        if i < 0:
            raise ValueError(f"{self.name}: no execution bar closes by {pd.Timestamp(t_ns, tz='UTC')}")
        return float(self.closes[i])

    def funding(self, side: int, after_ns: int, upto_ns: int) -> float:
        """Funding per unit held over settlements in (after, upto], as the engine books it: a stored rate x mark a long
        pays and a short receives; a baseline one (no stored rate) both sides pay."""
        if self.settle_ns is None or not len(self.settle_ns):
            return 0.0
        lo = int(np.searchsorted(self.settle_ns, after_ns, side="right"))
        hi = int(np.searchsorted(self.settle_ns, upto_ns, side="right"))
        base = self.settle_baseline if self.settle_baseline is not None else np.zeros(len(self.settle_ns), dtype=bool)
        return float(sum(-(1 if base[i] else side) * self.settle_rates[i] * self.close_at(int(self.settle_ns[i]))
                         for i in range(lo, hi)))

    def baseline_in(self, after_ns: int, upto_ns: int) -> int:
        if self.settle_baseline is None or not len(self.settle_ns):
            return 0
        lo = int(np.searchsorted(self.settle_ns, after_ns, side="right"))
        hi = int(np.searchsorted(self.settle_ns, upto_ns, side="right"))
        return int(np.asarray(self.settle_baseline[lo:hi], dtype=bool).sum())


@dataclass
class PairNull:
    """The comparison: the Strategy's P&L over the fund's capital after fees and before fees, the share of draws it
    beats on each, the draws' medians, the legs opened by the Strategy and by the mean draw, and what was charged."""

    draws: int
    seed: int
    change_weeks: int
    strategy_net: float
    strategy_pre: float
    net_percentile: float
    pre_percentile: float
    median_net: float
    median_pre: float
    strategy_legs: int
    mean_draw_legs: float
    reconciled: tuple[float, float] | None = None  # (the model's Strategy P&L, the engine book's move), over capital
    baseline_settlements: int = 0
    skipped_weeks: int = 0  # unranked Mondays inside the span: a HOLD for the Strategy and every draw
    why_not: str = ""

    @property
    def unreconciled(self) -> bool:
        return self.reconciled is not None and not abs(self.reconciled[0] - self.reconciled[1]) <= RECONCILE_PP

    @property
    def judged(self) -> bool:
        return not self.why_not and not self.unreconciled and self.change_weeks > 0

    @property
    def beats(self) -> bool:
        """The bar for the four: 95% of draws after real fees."""
        return self.judged and self.net_percentile >= BAR

    @property
    def killed(self) -> bool:
        """Round-0 kill 2: below 90% of draws before fees."""
        return self.judged and self.pre_percentile < PRE_FEE_KILL


ORDERINGS = tuple(itertools.permutations(range(3), 2))  # the 6 ordered (long, short) pairs of three assets


def _runs(states: list[tuple]) -> list[tuple[int, int, int, int]]:
    """Each leg as (asset, side, first state, end state exclusive): a run of states holding the same asset on the
    same side, so a leg kept across a change week is one leg, not re-traded."""
    out = []
    for slot, side in ((0, 1), (1, -1)):
        start = None
        for i in range(len(states) + 1):
            cur = states[i][slot] if i < len(states) else None
            if start is not None and cur != states[start][slot]:
                out.append((states[start][slot], side, start, i))
                start = None
            if start is None and cur is not None:
                start = i
    return out


class _Pricer:
    """Leg P&L under the model, cached by (asset, side, first state, end state, fee on)."""

    def __init__(self, assets: list[PairAsset], times: list[int], notional: np.ndarray):
        self.assets, self.times, self.notional, self.cache = assets, times, notional, {}

    def leg(self, asset: int, side: int, a: int, b: int, fees: bool) -> float:
        key = (asset, side, a, b, fees)
        if key in self.cache:
            return self.cache[key]
        x, t = self.assets[asset], self.times
        k = len(t) - 1  # t[k] is the span's end
        carried, entry_t = a == 0, t[max(a - 1, 0)]
        size = self.notional[max(a - 1, 0)]
        entry = x.open_at(entry_t) * (1 if carried else 1 + side * x.half)
        qty = size / entry
        exit_t = t[b - 1] if b - 1 < k else t[k]
        at_end = b - 1 >= k
        out_px = x.close_at(exit_t) if at_end else x.open_at(exit_t) * (1 - side * x.half)
        fee = x.taker if fees else 0.0
        cost = (0.0 if carried else fee * qty * entry) + (0.0 if at_end else fee * qty * out_px)
        funding = qty * x.funding(side, entry_t, exit_t)
        self.cache[key] = side * qty * (out_px - entry) - cost + funding
        return self.cache[key]

    def book(self, states: list[tuple], fees: bool) -> tuple[float, int]:
        runs = _runs(states)
        return (sum(self.leg(a, s, i, j, fees) for a, s, i, j in runs),
                sum(1 for _, _, i, _ in runs if i > 0))


def pair_null(assets: list[PairAsset], change_ns: list[int], states: list[tuple], end_ns: int, notional, capital: float,
              draws: int = DRAWS, seed: int = 0, book_move: float | None = None, skipped: int = 0) -> PairNull:
    """The Strategy's books against `draws` random assignments. change_ns: its change weeks in order; states: k + 1
    (long index, short index) pairs, states[0] carried into the span and states[i] held from change_ns[i - 1];
    notional: the leg notional the Strategy sized at each change week; book_move: the engine book's P&L over the span,
    for the reconcile."""
    if draws < DRAWS:
        raise ValueError(f"refused: the random check draws at least {DRAWS:,} times, not {draws:,}")
    if len(assets) != 3:
        raise ValueError(f"the random long/short assignment draws from three assets' 6 orderings, not {len(assets)}")
    k = len(change_ns)
    if len(states) != k + 1 or len(notional) < k:
        raise ValueError("a state before the first change week and one per change week, with a notional each")
    if k == 0:
        return PairNull(draws, seed, 0, 0.0, 0.0, math.nan, math.nan, 0.0, 0.0, 0, 0.0, why_not="no change weeks",
                        skipped_weeks=skipped)
    if list(change_ns) != sorted(set(change_ns)) or end_ns <= change_ns[-1]:
        raise ValueError("change weeks are distinct, in order, and before the span's end")
    pricer = _Pricer(assets, [*map(int, change_ns), int(end_ns)], np.asarray(notional, dtype=float))
    net, legs = pricer.book(states, True)
    pre, _ = pricer.book(states, False)
    rng = np.random.default_rng(seed)
    nets, pres, draw_legs = np.empty(draws), np.empty(draws), np.empty(draws)
    for d in range(draws):
        drawn = [states[0], *(ORDERINGS[i] for i in rng.integers(len(ORDERINGS), size=k))]
        nets[d], draw_legs[d] = pricer.book(drawn, True)
        pres[d], _ = pricer.book(drawn, False)
    first = int(change_ns[0])
    baseline = sum(x.baseline_in(first, int(end_ns)) for x in assets)
    return PairNull(draws, seed, k, net / capital, pre / capital, float((nets < net).mean() * 100),
                    float((pres < pre).mean() * 100), float(np.median(nets)) / capital,
                    float(np.median(pres)) / capital, legs, float(draw_legs.mean()),
                    reconciled=None if book_move is None else (net / capital, book_move / capital),
                    baseline_settlements=baseline, skipped_weeks=skipped)


def week_of(ts: pd.Timestamp) -> pd.Timestamp:
    """The Monday 00:00 UTC a fill belongs to: a rank book's fills follow its Monday close by minutes."""
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.normalize() - pd.Timedelta(days=ts.weekday())


def strategy_states(trips: dict[int, list[dict]], start: pd.Timestamp, end: pd.Timestamp) -> tuple[list[int], list]:
    """The Strategy's change weeks inside [start, end) and its states, from each asset's trips ({asset index: trips
    with opened, closed (None while open) and side}): states[0] is what it held going into the first change week."""
    events = set()
    for rows in trips.values():
        for t in rows:
            for key in ("opened", "closed"):
                if t.get(key) is not None:
                    w = week_of(t[key])
                    if start <= w < end:
                        events.add(w)
    weeks = sorted(events)

    def held(at: pd.Timestamp, before: bool) -> tuple:
        long = short = None
        for asset, rows in trips.items():
            for t in rows:
                o = week_of(t["opened"])
                c = week_of(t["closed"]) if t.get("closed") is not None else None
                on = (o < at and (c is None or c >= at)) if before else (o <= at and (c is None or c > at))
                if on:
                    if int(t["side"]) > 0:
                        long = asset
                    else:
                        short = asset
        return long, short

    if not weeks:
        return [], [held(start, True)]
    return [w.value for w in weeks], [held(weeks[0], True), *(held(w, False) for w in weeks)]


def null_words(r: PairNull | None) -> str:
    """The comparison in words, its verdict first."""
    if r is None:
        return "N/A: not a rank book"
    if r.why_not:
        return f"NOT JUDGED: {r.why_not}"
    trips, book = r.reconciled or (r.strategy_net, r.strategy_net)
    out = (f"after real fees the book's {r.strategy_net:+.1%} of the fund's capital beats {r.net_percentile:.0f}% of "
           f"{r.draws:,} random long/short assignments (median {r.median_net:+.1%}; bar {BAR:.0f}%); before fees "
           f"{r.strategy_pre:+.1%} beats {r.pre_percentile:.0f}% (median {r.median_pre:+.1%}; below "
           f"{PRE_FEE_KILL:.0f}% is round-0 kill 2); {r.change_weeks} change weeks kept fixed, each re-drawing the "
           f"ordered pair from 6; legs opened: the book {r.strategy_legs}, the mean draw {r.mean_draw_legs:.1f}; "
           f"{r.skipped_weeks} unranked Mondays held by both; seed {r.seed}; the model's book adds to {trips:+.1%} "
           f"against the engine's {book:+.1%}")
    if r.baseline_settlements:
        out += f"; {r.baseline_settlements} funding settlements at the baseline rate, not stored"
    if r.unreconciled:
        return f"NOT JUDGED: {out} (more than {RECONCILE_PP * 100:.1f} points apart)"
    verdict = "ABOVE THE BAR" if r.beats else "BELOW THE BAR"  # the list allows no PASS wording
    return f"{verdict}{' (round-0 kill 2)' if r.killed else ''}: {out}"
