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


@dataclass(frozen=True)
class Trade:
    entry: int  # bar positions in the close series: bought (or sold short) at entry's close, out at exit's
    exit: int
    side: int  # 1 long, -1 short
    # A perpetual's leverage on isolated margin: a draw through its liquidation price on any close it holds is
    # liquidated as the engine books one, losing exactly its margin, 1 / leverage, plus fees (GAP-LIQ-CAP,
    # Independent Quant Advisor 7 Oct 00:19 (4)). None: unlevered, the close-to-close move whatever it is.
    leverage: float | None = None


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

    @property
    def verdict(self) -> str:
        if self.trades == 0:
            return "N/A"
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
    actual = _compound(_trade_returns(c, entries, holds, sides, cost_per_side, levs))
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
    actual = np.concatenate([_trade_returns(c, e, h, s, cost_per_side, lv)
                             for (_, _, e, h, s, _), lv in zip(per_window, levs)])
    rets, sharpes = np.empty(draws), np.empty(draws)
    for d in range(draws):
        parts = []
        for (start, end, _, holds, sides, _), lv in zip(per_window, levs):
            order, entries = _random_entries(rng, start, end, holds)
            parts.append(_trade_returns(c, entries, holds[order], sides[order], cost_per_side,
                                        None if lv is None else lv[order]))
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
    )
