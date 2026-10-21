"""Random-entry benchmark (v2 P1-7, C3): did the strategy's timing add anything, or would being in the market
for the same trades at random times have done as well?

For each walk-forward test window, the strategy's own out-of-sample trades keep their number, sides and holding
periods; only their entry times are drawn at random, without overlap, inside the same window. Every draw pays
the same cost per side. The strategy's net return is then a percentile of the draws' (Independent Quant
Advisor, 5 Oct 2026): judged on return, pooled across windows, one-sided, with a bar at the 95th percentile.
The Sharpe percentile is shown beside it and flagged when the two disagree by more than 30 points. When the
strategy is in the market most of the time the draws have nowhere else to go and sit almost on top of it, so
the benchmark is called weak: not judged, neither a pass nor a fail (Advisor, 19:19).

The random-side test (C3b) is its twin for strategies that can go short: the trades keep their entry times and
holding periods, and only long or short is drawn at random. It asks whether the strategy picks the direction, as
the random-entry test asks whether it picks the moment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sleeve_fund.markets import LOW_FEE_PERP, liquidation_price

DRAWS = 1000
BAR = 95.0  # percentile of the random draws the strategy's net return must reach
WEAK_EXPOSURE = 0.6  # in the market more than this share of the test bars: the benchmark is weak
DISAGREE = 30.0  # percentile points between the return and Sharpe readings worth flagging
# How a strength-sized variant's draws take their sizes (Advisor G5-DW5 Q1), named in its words.
PERMUTED = "sizes drawn from the variant's own, permuted within each window"
REPLACED = "sizes drawn from the variant's own with replacement (a window's trade count differed)"
# RANDOM-ENTRY (HoE 14:08 UK 21 Oct; Advisor): the strategy's side is its realised trips, which must add up to what its
# equity did over them within this much, absolute, or the comparison is NOT JUDGED.
RECONCILE_PP = 0.005
NOT_JUDGED = "NOT JUDGED"
# STOP-FILL gap (Advisor ruling (A), HoE 21 Oct): the two sides book stops differently until STOP-FILL unifies them.
STOP_FILL_LINE = ("random draws book stops at the level; the Strategy books the bar's low less 0.05%; the bias is "
                  "against the Strategy")
MARGINAL = "marginal FAIL, within fill-model gap; re-judge after STOP-FILL unification"
# Draws the engine would have held back are priced flat, which pulls the draws towards zero; above this share of the
# draw trips the comparison is NOT JUDGED (Advisor, PR E round 1; re-drawing them can come later).
HELD_BACK_MAX = 0.02


@dataclass(frozen=True)
class Trade:
    entry: int  # bar positions in the close series: bought (or sold short) at entry's close, out at exit's
    exit: int
    side: int  # 1 long, -1 short
    # A perpetual's leverage on isolated margin: a draw through its liquidation price on any close it holds is
    # liquidated as the engine books one, losing exactly its margin, 1 / leverage, plus fees (GAP-LIQ-CAP,
    # Independent Quant Advisor 7 Oct 00:19 (4)). None: unlevered, the close-to-close move whatever it is.
    leverage: float | None = None
    # ABSORB-GAP (Advisor 15 Oct ~02:55 UK, option A): the gap past bankruptcy the venue absorbed on this trade's
    # liquidation, over its entry notional (the base of its return): charged to the strategy's trade, never a draw's.
    charge: float = 0.0
    # G5 research side (Advisor G5-DW5): a strength-sized variant's size factor f on this trade, its return on equity f
    # times the trade's; 1 for a flat version (and then the benchmark is exactly as before). Only trades taken carry
    # one, so it is above 0.
    size: float = 1.0


@dataclass
class RandomEntryResult:
    return_percentile: float
    sharpe_percentile: float
    strategy_return: float
    median_random_return: float
    trades: int
    exposure: float  # share of the test bars the strategy held a position
    draws: int
    weak: bool
    disagree: bool
    sizing: str = ""  # a strength-sized variant's: how its draws took their sizes (PERMUTED or REPLACED)
    # RANDOM-ENTRY: (the counted trips' realised returns compounded on equity, what the equity curve did over the
    # same trips); None for the close-priced comparison of a model with no rule definition to fill stops from.
    reconciled: tuple | None = None
    held_back: int = 0  # draws the engine would have held back (no level yet, or on the wrong side), priced flat
    held_back_share: float = 0.0  # of the draws' trips
    stopped_share: float = 0.0  # the draws' trips that filled at a level inside a candle
    why_not: str = ""  # why the comparison can't be judged (RANDOM-ENTRY: exits the draws can't fill)
    strategy_stops: int = 0  # the strategy's counted trips its stop closed
    draw_stops: float = 0.0  # the draws' trips that stopped out, per draw
    # The STOP-FILL gap: the strategy's stop exits re-priced at their levels, as the draws' are; what that adds to its
    # total return, and the percentile its return would then reach (None: the close-priced comparison).
    gap_return: float = 0.0
    gap_percentile: float | None = None

    @property
    def too_many_held_back(self) -> bool:
        return self.held_back_share > HELD_BACK_MAX

    @property
    def marginal(self) -> bool:
        """A FAIL that the STOP-FILL gap added back would turn into a PASS: still a FAIL, never promoted."""
        return self.verdict == "FAIL" and self.gap_percentile is not None and self.gap_percentile >= BAR

    @property
    def unreconciled(self) -> bool:
        return self.reconciled is not None and not abs(self.reconciled[0] - self.reconciled[1]) <= RECONCILE_PP

    @property
    def verdict(self) -> str:
        if self.trades == 0:
            return "N/A"
        if self.unreconciled or self.why_not or self.too_many_held_back:
            return NOT_JUDGED
        if self.weak:
            return "WEAK"
        return "PASS" if self.return_percentile >= BAR else "FAIL"

    @property
    def words(self) -> str:
        if self.trades == 0:
            return "no out-of-sample trades to compare"
        out = (f"net return {self.strategy_return:+.1%} beats {self.return_percentile:.0f}% of {self.draws} random-entry "
               f"runs with the same {self.trades} trades and holding periods (median {self.median_random_return:+.1%}; "
               f"bar: {BAR:.0f}%)")
        if self.weak:
            out += (f"; in the market {self.exposure:.0%} of the time, so random entries can hardly differ from it: "
                    "a weak test")
        if self.disagree:
            out += f"; on Sharpe it beats {self.sharpe_percentile:.0f}%, far from the return reading"
        if self.sizing:
            out += f"; {self.sizing}"
        if self.why_not:
            return out + f"; not judged: {self.why_not}"
        if self.reconciled is None:
            out += ("; both sides priced on the decision candles' closes (no rule definition to fill stops from), "
                    "so it compares timing, not fills")
        else:
            trips, equity = self.reconciled
            out += (f"; the strategy's side is its realised trips (fees, fills and funding), which compound to "
                    f"{trips:+.1%} on equity against {equity:+.1%} on its equity curve over the same trips"
                    f"{' (more than 0.5 points apart: not judged)' if self.unreconciled else ''}; each draw holds "
                    f"for one of its holding periods at most, its own stop and trail filled on the candles' lows and "
                    f"highs as the engine fills them ({self.stopped_share:.0%} of the draws' trips stopped out), "
                    f"funding charged at the stored settlement times")
            if self.held_back:
                out += (f"; {self.held_back} draw trips ({self.held_back_share:.1%}) priced flat, as the engine would "
                        f"have held them back"
                        + (f"; not judged: too many draws held back (above {HELD_BACK_MAX:.0%})"
                           if self.too_many_held_back else ""))
            out += (f"; {STOP_FILL_LINE}: {self.strategy_stops} stop exits in the Strategy's trips, "
                    f"{self.draw_stops:.1f} per draw; at the level its return would be {self.gap_return * 100:+.2f} "
                    f"points higher")
            if self.gap_percentile is not None:
                out += f", {self.gap_percentile - self.return_percentile:+.0f} percentile points"
            if self.marginal:
                out = f"{MARGINAL}: {out}"
        return out


@dataclass
class RandomSideResult:
    return_percentile: float
    strategy_return: float
    median_random_return: float
    trades: int
    draws: int

    @property
    def verdict(self) -> str:
        if self.trades == 0:
            return "N/A"
        return "PASS" if self.return_percentile >= BAR else "FAIL"

    @property
    def words(self) -> str:
        if self.trades == 0:
            return "no out-of-sample trades to compare"
        return (f"net return {self.strategy_return:+.1%} beats {self.return_percentile:.0f}% of {self.draws} runs with "
                f"the same {self.trades} entries and holding periods but long or short drawn at random (median "
                f"{self.median_random_return:+.1%}; bar: {BAR:.0f}%)")


def _trade_returns(closes: np.ndarray, entries: np.ndarray, holds: np.ndarray, sides: np.ndarray,
                   cost, levs: np.ndarray | None = None) -> np.ndarray:
    gross = closes[entries + holds] / closes[entries] - 1
    out = sides * gross - _round_trip(cost, entries, holds)
    if levs is None:
        return out
    for k in np.flatnonzero(~np.isnan(levs)):
        trigger = _liquidated_at(closes, int(entries[k]), int(holds[k]), int(sides[k]), float(levs[k]))
        if trigger is not None:
            # Booked at the bankruptcy price: the margin and no more, the liquidation fee on the trigger price.
            c_in, c_out = _side_costs(cost, int(entries[k]), int(entries[k] + holds[k]))
            out[k] = -1 / levs[k] - c_in - c_out * trigger
    return out


def _round_trip(cost, entries: np.ndarray, holds: np.ndarray):
    """Both sides' cost: one figure per side, or one per bar (the spread in force at each bar, SPREAD-PIT), charged
    at the entry's bar and the exit's."""
    if np.ndim(cost) == 0:
        return 2 * cost
    return cost[entries] + cost[entries + holds]


def _side_costs(cost, entry: int, exit_: int) -> tuple[float, float]:
    """One trade's cost per side at its entry's bar and its exit's: the one figure, or the bars' own (SPREAD-PIT)."""
    if np.ndim(cost) == 0:
        return float(cost), float(cost)
    return float(cost[entry]), float(cost[exit_])


def _liquidated_at(closes: np.ndarray, entry: int, hold: int, side: int, lev: float) -> float | None:
    """The liquidation price, as a multiple of the entry, if a close the trade held reached it, else None."""
    trigger = liquidation_price(1 / lev - side, side, LOW_FEE_PERP.maintenance_margin)
    if trigger is None:
        return None
    path = closes[entry + 1:entry + hold + 1] / closes[entry]
    hit = path <= trigger if side > 0 else path >= trigger
    return trigger if hit.any() else None


def _levs(trades, default: float | None) -> np.ndarray | None:
    levs = np.array([t.leverage if t.leverage is not None else default if default is not None else np.nan
                     for t in trades], dtype=float)
    return None if np.isnan(levs).all() else levs


def _sharpe(r: np.ndarray) -> float:
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def _compound(r: np.ndarray) -> float:
    return float(np.prod(1 + r) - 1)


def _draw_sizes(rng: np.random.Generator, sizes: np.ndarray, n: int) -> tuple[np.ndarray, str]:
    """A draw's n trade sizes from a window's own: a permutation, so the window's total exposure is the variant's;
    with replacement only when the counts differ (Advisor G5-DW5 Q1 guard)."""
    if len(sizes) == n:
        return rng.permutation(sizes), PERMUTED
    return rng.choice(sizes, size=n, replace=True), REPLACED


def _random_entries(rng: np.random.Generator, start: int, end: int, holds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Entry bars for `holds` placed in random order, without overlap, between bar `start` and bar `end`
    (an exit may land on `end`). Returns (order, entries): entries[k] is the entry of holds[order[k]]."""
    order = rng.permutation(len(holds))
    free = (end - start) - int(holds.sum())
    if free < 0:
        raise ValueError("the trades hold for longer than their window: they must overlap")
    # A random way to share the free bars among the n + 1 gaps around the trades: trade k waits cuts[k] free
    # bars in all before it starts.
    cuts = np.sort(rng.integers(0, free + 1, size=len(holds)))
    entries = start + cuts + np.r_[0, holds[order].cumsum()[:-1]]
    return order, entries


def _by_window(trades: list[Trade], windows: list[tuple[int, int]]):
    """Each window's trades as (start, end, entries, holds, sides), skipping windows without one, plus the
    windows' bars in all and the bars the trades held."""
    per_window = []
    test_bars = held = 0
    for start, end in windows:
        inside = sorted((t for t in trades if start <= t.entry and t.exit <= end), key=lambda t: t.entry)
        if any(b.entry < a.exit for a, b in zip(inside, inside[1:])):
            raise ValueError("out-of-sample trades overlap; the benchmark needs one position at a time")
        test_bars += end - start + 1  # inclusive: a window from bar 0 to bar 9 is 10 bars (QA P1-R3)
        if not inside:
            continue
        holds = np.array([t.exit - t.entry for t in inside])
        if (holds < 1).any():
            raise ValueError("a trade must hold for at least one bar")
        if any(not t.size > 0 for t in inside):
            raise ValueError("a trade's size factor is above 0: an entry skipped below the floor is no trade")
        held += int(holds.sum())
        per_window.append((start, end, np.array([t.entry for t in inside]), holds, np.array([t.side for t in inside]),
                           inside))
    return per_window, test_bars, held


def random_side(closes, trades: list[Trade], windows: list[tuple[int, int]], cost_per_side,
                draws: int = DRAWS, seed: int = 0, leverage: float | None = None) -> RandomSideResult:
    """As random_entry, but the entries stay put and each trade's side is drawn at random, long or short. A
    perpetual's draws are liquidated as the engine books one, as random_entry's (Trade.leverage)."""
    c = np.asarray(closes, dtype=float)
    rng = np.random.default_rng(seed)
    per_window, _, _ = _by_window(trades, windows)
    if not per_window:
        return RandomSideResult(math.nan, 0.0, 0.0, 0, draws)
    entries = np.concatenate([w[2] for w in per_window])
    holds = np.concatenate([w[3] for w in per_window])
    sides = np.concatenate([w[4] for w in per_window])
    levs = _levs([t for w in per_window for t in w[5]], leverage)
    charges = np.array([t.charge for w in per_window for t in w[5]])
    actual = _compound(_trade_returns(c, entries, holds, sides, cost_per_side, levs) - charges)
    long_ = _trade_returns(c, entries, holds, np.ones_like(sides), cost_per_side, levs)
    short = _trade_returns(c, entries, holds, -np.ones_like(sides), cost_per_side, levs)
    drawn = rng.choice(np.array([-1, 1]), size=(draws, len(holds)))
    rets = np.prod(1 + np.where(drawn > 0, long_, short), axis=1) - 1
    return RandomSideResult(return_percentile=float((rets < actual).mean() * 100), strategy_return=actual,
                            median_random_return=float(np.median(rets)), trades=len(holds), draws=draws)


def random_entry(closes, trades: list[Trade], windows: list[tuple[int, int]], cost_per_side,
                 draws: int = DRAWS, seed: int = 0, in_market: float | None = None,
                 leverage: float | None = None) -> RandomEntryResult:
    """closes: the bar closes the strategy traded on. trades: its out-of-sample round trips. windows: each
    walk-forward test window as (first bar, last bar), inclusive. cost_per_side: fee plus half the spread, as a
    fraction, charged on entry and exit alike: one figure, or one per bar of closes (the spread in force then). in_market: the share of the windows' bars the strategy held any
    position, trades carried in and still open at the end included, though those stay out of the comparison
    (Independent Quant Advisor, 6 Oct 2026); without it, the bars the given trades held. leverage: for trades that
    don't carry their own, a perpetual's: the strategy's trades and every draw are liquidated as the engine books
    one (Trade.leverage)."""
    c = np.asarray(closes, dtype=float)
    rng = np.random.default_rng(seed)
    per_window, test_bars, held = _by_window(trades, windows)
    n = sum(len(w[3]) for w in per_window)
    if n == 0:
        return RandomEntryResult(math.nan, math.nan, 0.0, 0.0, 0, 0.0, draws, False, False)
    levs = [_levs(w[5], leverage) for w in per_window]
    # G5: a strength-sized variant's trades at their own sizes, and each draw at a draw of those sizes; a flat
    # version (every size 1) takes no draw of sizes, so its benchmark is exactly as before.
    sizes = [np.array([t.size for t in w[5]], dtype=float) for w in per_window]
    sized = any((f != 1.0).any() for f in sizes)
    actual = np.concatenate([(_trade_returns(c, e, h, s, cost_per_side, lv) - np.array([t.charge for t in inside]))
                             * (f if sized else 1)
                             for (_, _, e, h, s, inside), lv, f in zip(per_window, levs, sizes)])
    rets, sharpes = np.empty(draws), np.empty(draws)
    how = set()
    for d in range(draws):
        parts = []
        for (start, end, _, holds, sides, _), lv, f in zip(per_window, levs, sizes):
            order, entries = _random_entries(rng, start, end, holds)
            r = _trade_returns(c, entries, holds[order], sides[order], cost_per_side, None if lv is None else lv[order])
            if sized:
                drawn, method = _draw_sizes(rng, f, len(holds))
                how.add(method)
                r = r * drawn
            parts.append(r)
        r = np.concatenate(parts)
        rets[d], sharpes[d] = _compound(r), _sharpe(r)
    strategy_return, strategy_sharpe = _compound(actual), _sharpe(actual)
    ret_pct = float((rets < strategy_return).mean() * 100)
    sharpe_pct = float((sharpes < strategy_sharpe).mean() * 100)
    exposure = in_market if in_market is not None else held / test_bars if test_bars else 0.0
    return RandomEntryResult(
        return_percentile=ret_pct, sharpe_percentile=sharpe_pct, strategy_return=strategy_return,
        median_random_return=float(np.median(rets)), trades=n, exposure=exposure, draws=draws,
        weak=exposure > WEAK_EXPOSURE, disagree=abs(ret_pct - sharpe_pct) > DISAGREE,
        sizing=(REPLACED if REPLACED in how else PERMUTED) if sized else "",
    )


def _filled_windows(trades: list[Trade], windows: list[tuple[int, int]]):
    """As _by_window, for filled draws: a trip may open and close on one candle (hold 0), as the engine's can."""
    per_window, test_bars, held = [], 0, 0
    for start, end in windows:
        inside = [k for k, t in enumerate(trades) if start <= t.entry and t.exit <= end]
        test_bars += end - start + 1
        if not inside:
            continue
        holds = np.array([trades[k].exit - trades[k].entry for k in inside])
        if (holds < 0).any():
            raise ValueError("a trade can't close before it opens")
        held += int(holds.sum())
        per_window.append((start, end, holds, np.array([trades[k].side for k in inside]), inside))
    return per_window, test_bars, held


def random_entry_filled(trades: list[Trade], realised: np.ndarray, windows: list[tuple[int, int]], pricer,
                        reconciled: tuple, draws: int = DRAWS, seed: int = 0, in_market: float | None = None,
                        leverage: float | None = None, gaps: np.ndarray | None = None,
                        stops: np.ndarray | None = None) -> RandomEntryResult:
    """RANDOM-ENTRY: the strategy's realised trip returns (realised[k] for trades[k]: pnl over entry notional, after
    fees, fills and funding, less any ABSORB-GAP charge) against draws that keep its trips' count, sides and holding
    periods (as caps), placed at random without overlap in each window and priced by `pricer` (fills.Pricer) as the
    engine fills them. reconciled: (the counted trips compounded on equity, the equity curve over them). gaps[k]:
    what filling trades[k]'s stop exit at its level adds to its return (0 for a trip no stop closed); stops[k]: its
    stop closed it."""
    rng = np.random.default_rng(seed)
    per_window, test_bars, held = _filled_windows(trades, windows)
    n = sum(len(w[2]) for w in per_window)
    if n == 0:
        return RandomEntryResult(math.nan, math.nan, 0.0, 0.0, 0, 0.0, draws, False, False, reconciled=reconciled)
    sizes = [np.array([trades[k].size for k in w[4]], dtype=float) for w in per_window]
    sized = any((f != 1.0).any() for f in sizes)
    levs = [_levs([trades[k] for k in w[4]], leverage) for w in per_window]
    actual = np.concatenate([np.asarray(realised, dtype=float)[w[4]] * (f if sized else 1)
                             for w, f in zip(per_window, sizes)])
    ks = [k for w in per_window for k in w[4]]
    gap = np.zeros(len(realised)) if gaps is None else np.asarray(gaps, dtype=float)
    gapped = actual + np.concatenate([gap[w[4]] * (f if sized else 1) for w, f in zip(per_window, sizes)])
    strategy_stops = 0 if stops is None else int(np.asarray(stops, dtype=bool)[ks].sum())
    rets, sharpes = np.empty(draws), np.empty(draws)
    how, held_back, stopped, priced = set(), 0, 0, 0
    for d in range(draws):
        parts = []
        for (start, end, holds, sides, _), lv, f in zip(per_window, levs, sizes):
            order, entries = _random_entries(rng, start, end, holds)
            fills = pricer.draws(entries, holds[order], sides[order], None if lv is None else lv[order])
            held_back, stopped, priced = held_back + fills.held_back, stopped + fills.stopped, priced + len(entries)
            r = fills.returns
            if sized:
                drawn, method = _draw_sizes(rng, f, len(holds))
                how.add(method)
                r = r * drawn
            parts.append(r)
        r = np.concatenate(parts)
        rets[d], sharpes[d] = _compound(r), _sharpe(r)
    strategy_return, strategy_sharpe = _compound(actual), _sharpe(actual)
    ret_pct = float((rets < strategy_return).mean() * 100)
    sharpe_pct = float((sharpes < strategy_sharpe).mean() * 100)
    exposure = in_market if in_market is not None else held / test_bars if test_bars else 0.0
    return RandomEntryResult(
        return_percentile=ret_pct, sharpe_percentile=sharpe_pct, strategy_return=strategy_return,
        median_random_return=float(np.median(rets)), trades=n, exposure=exposure, draws=draws,
        weak=exposure > WEAK_EXPOSURE, disagree=abs(ret_pct - sharpe_pct) > DISAGREE,
        sizing=(REPLACED if REPLACED in how else PERMUTED) if sized else "", reconciled=reconciled,
        held_back=held_back, held_back_share=held_back / priced if priced else 0.0,
        stopped_share=stopped / priced if priced else 0.0,
        strategy_stops=strategy_stops, draw_stops=stopped / draws, gap_return=_compound(gapped) - strategy_return,
        gap_percentile=float((rets < _compound(gapped)).mean() * 100),
    )


def random_side_filled(trades: list[Trade], realised: np.ndarray, windows: list[tuple[int, int]], pricer,
                       draws: int = DRAWS, seed: int = 0, leverage: float | None = None) -> RandomSideResult:
    """As random_side, filled: each trip keeps its entry and holding period (as a cap), long or short drawn at
    random, each side priced by `pricer` with that side's own exits; the strategy's side is its realised trips."""
    rng = np.random.default_rng(seed)
    per_window, _, _ = _filled_windows(trades, windows)
    if not per_window:
        return RandomSideResult(math.nan, 0.0, 0.0, 0, draws)
    ks = [k for w in per_window for k in w[4]]
    entries = np.array([trades[k].entry for k in ks])
    holds = np.concatenate([w[2] for w in per_window])
    levs = _levs([trades[k] for k in ks], leverage)
    actual = _compound(np.asarray(realised, dtype=float)[ks])
    long_ = pricer.draws(entries, holds, np.ones(len(ks), dtype=int), levs).returns
    short = pricer.draws(entries, holds, -np.ones(len(ks), dtype=int), levs).returns
    drawn = rng.choice(np.array([-1, 1]), size=(draws, len(ks)))
    rets = np.prod(1 + np.where(drawn > 0, long_, short), axis=1) - 1
    return RandomSideResult(return_percentile=float((rets < actual).mean() * 100), strategy_return=actual,
                            median_random_return=float(np.median(rets)), trades=len(ks), draws=draws)
