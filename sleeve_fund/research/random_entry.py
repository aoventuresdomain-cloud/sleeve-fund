"""Random-entry benchmark (v2 P1-7, C3): did the strategy's timing add anything, or would being in the market
for the same trades at random times have done as well?

For each walk-forward test window, the strategy's own out-of-sample trades keep their number, sides and holding
periods; only their entry times are drawn at random, without overlap, inside the same window. Every draw pays
the same cost per side. The strategy's net return is then a percentile of the draws' (Independent Quant
Advisor, 5 Oct 2026): judged on return, pooled across windows, one-sided, with a bar at the 95th percentile.
The Sharpe percentile is shown beside it and flagged when the two disagree by more than 30 points. When the
strategy is in the market most of the time the draws have nowhere else to go and sit almost on top of it, so
the benchmark is called weak rather than passed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DRAWS = 1000
BAR = 95.0  # percentile of the random draws the strategy's net return must reach
WEAK_EXPOSURE = 0.6  # in the market more than this share of the test bars: the benchmark is weak
DISAGREE = 30.0  # percentile points between the return and Sharpe readings worth flagging


@dataclass(frozen=True)
class Trade:
    entry: int  # bar positions in the close series: bought (or sold short) at entry's close, out at exit's
    exit: int
    side: int  # 1 long, -1 short


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


def _trade_returns(closes: np.ndarray, entries: np.ndarray, holds: np.ndarray, sides: np.ndarray,
                   cost: float) -> np.ndarray:
    gross = closes[entries + holds] / closes[entries] - 1
    return sides * gross - 2 * cost


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


def random_entry(closes, trades: list[Trade], windows: list[tuple[int, int]], cost_per_side: float,
                 draws: int = DRAWS, seed: int = 0) -> RandomEntryResult:
    """closes: the bar closes the strategy traded on. trades: its out-of-sample round trips. windows: each
    walk-forward test window as (first bar, last bar), inclusive. cost_per_side: fee plus half the spread, as a
    fraction, charged on entry and exit alike."""
    c = np.asarray(closes, dtype=float)
    rng = np.random.default_rng(seed)
    per_window = []
    test_bars = held = 0
    for start, end in windows:
        inside = sorted((t for t in trades if start <= t.entry and t.exit <= end), key=lambda t: t.entry)
        if any(b.entry < a.exit for a, b in zip(inside, inside[1:])):
            raise ValueError("out-of-sample trades overlap; the benchmark needs one position at a time")
        test_bars += end - start
        if not inside:
            continue
        holds = np.array([t.exit - t.entry for t in inside])
        if (holds < 1).any():
            raise ValueError("a trade must hold for at least one bar")
        held += int(holds.sum())
        per_window.append((start, end, np.array([t.entry for t in inside]), holds, np.array([t.side for t in inside])))
    n = sum(len(w[3]) for w in per_window)
    if n == 0:
        return RandomEntryResult(math.nan, math.nan, 0.0, 0.0, 0, 0.0, draws, False, False)
    actual = np.concatenate([_trade_returns(c, e, h, s, cost_per_side) for _, _, e, h, s in per_window])
    rets, sharpes = np.empty(draws), np.empty(draws)
    for d in range(draws):
        parts = []
        for start, end, _, holds, sides in per_window:
            order, entries = _random_entries(rng, start, end, holds)
            parts.append(_trade_returns(c, entries, holds[order], sides[order], cost_per_side))
        r = np.concatenate(parts)
        rets[d], sharpes[d] = _compound(r), _sharpe(r)
    strategy_return, strategy_sharpe = _compound(actual), _sharpe(actual)
    ret_pct = float((rets < strategy_return).mean() * 100)
    sharpe_pct = float((sharpes < strategy_sharpe).mean() * 100)
    exposure = held / test_bars if test_bars else 0.0
    return RandomEntryResult(
        return_percentile=ret_pct, sharpe_percentile=sharpe_pct, strategy_return=strategy_return,
        median_random_return=float(np.median(rets)), trades=n, exposure=exposure, draws=draws,
        weak=exposure > WEAK_EXPOSURE, disagree=abs(ret_pct - sharpe_pct) > DISAGREE,
    )
