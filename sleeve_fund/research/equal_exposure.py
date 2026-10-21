"""R-07 (Advisor ruling, 21 Oct 2026 ~10:20 UK): a volatility-sized variant (V3, V4) judged against its parent at the
same average gross exposure. Sizing to a target volatility only ever shrinks an entry, so the variant holds less than
its parent and raw return alone misleads.

k is the variant's average gross exposure over the parent's, over the same out-of-sample test bars (time-weighted: the
mean over every test bar of |position notional / equity|, a flat bar counting 0, as strength_report reads it), overall
and per test year. The parent's daily returns scaled by k are the parent held at k times its size: its fees, spread and
funding are all in proportion to notional, so it is costed at k x notional. The variant's CAGR, drawdown and
random-entry percentile are judged against that scaled parent; Sharpe, which scaling leaves as it is, against the
parent directly.

The verdict (Advisor ruling, 21 Oct 2026 ~10:46 UK, fixed before any run): the variant is better than its parent only
when ALL of (i) its Sharpe beats the unscaled parent's, and a paired stationary bootstrap of the two daily return
series (mean block SHARPE_BLOCK_DAYS days, SHARPE_RESAMPLES resamples, seed SHARPE_SEED) has it higher in at least
SHARPE_SHARE of resamples; (ii) its largest drawdown is no worse than the scaled parent's; (iii) it still passes Edge
found on its own terms: at or above the 95th percentile of random entries, and profitable at twice the fees. CAGR is
shown and does not count. No study's own G1 verdict changes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sleeve_fund.research.metrics import summary

SHARPE_BLOCK_DAYS = 20
SHARPE_RESAMPLES = 1000
SHARPE_SEED = 0
SHARPE_SHARE = 0.90


@dataclass(frozen=True)
class EqualExposure:
    k: float
    k_by_year: dict
    variant_exposure: float
    parent_exposure: float
    scaled_exposure: float  # the scaled parent's: k x the parent's, so the variant's
    variant: dict  # summary() of the variant's out-of-sample daily returns
    scaled: dict  # summary() of the parent's scaled by k
    parent_sharpe: float
    variant_percentile: float  # random entry: the share of draws the net return beats
    scaled_percentile: float
    sharpe_share: float = math.nan  # the share of paired resamples the variant's Sharpe is the higher in
    own_edge: bool = False  # (iii): at or above the 95th percentile of random entries, and profitable at 2x fees

    @property
    def sharpe_better(self) -> bool:
        return (math.isfinite(self.variant["sharpe"]) and self.variant["sharpe"] > self.parent_sharpe
                and self.sharpe_share >= SHARPE_SHARE)

    @property
    def drawdown_better(self) -> bool:
        return self.variant["max_drawdown"] >= self.scaled["max_drawdown"]

    @property
    def better(self) -> bool:
        return math.isfinite(self.k) and self.sharpe_better and self.drawdown_better and self.own_edge


def _sharpe(r: np.ndarray) -> np.ndarray:
    sd = r.std(axis=-1, ddof=1)
    return np.where(sd > 0, r.mean(axis=-1) / np.where(sd > 0, sd, 1), 0.0)


def stationary_indices(n: int, block: float, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """(resamples, n) day indices: each day starts a new block with chance 1/block (so blocks average `block` days),
    else follows the day before, wrapping round."""
    idx = np.empty((resamples, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, resamples)
    fresh = rng.random((resamples, n)) < 1 / block
    starts = rng.integers(0, n, (resamples, n))
    for t in range(1, n):
        idx[:, t] = np.where(fresh[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def paired_sharpe_share(variant: pd.Series, parent: pd.Series, block: int = SHARPE_BLOCK_DAYS,
                        resamples: int = SHARPE_RESAMPLES, seed: int = SHARPE_SEED) -> float:
    """The share of paired stationary-bootstrap resamples (Politis and Romano: blocks of geometric length, mean
    `block`, wrapping round) of the days both series have in which the variant's Sharpe is higher than the parent's.
    Paired: each resample takes the same days from both, as they share their signals."""
    both = pd.concat([variant, parent], axis=1, join="inner").dropna().to_numpy(dtype=float)
    n = len(both)
    if n < 2:
        return math.nan
    idx = stationary_indices(n, block, resamples, np.random.default_rng(seed))
    v, p = both[:, 0][idx], both[:, 1][idx]
    return float((_sharpe(v) > _sharpe(p)).mean())


def exposure_ratio(variant: pd.Series, parent: pd.Series) -> tuple[float, dict]:
    """k = mean |variant exposure| / mean |parent exposure| over the same test bars, and the same per calendar year of
    those bars (nan where the parent held nothing)."""
    if not variant.index.equals(parent.index):
        raise ValueError("the variant's and the parent's exposure must cover the same test bars")
    v, p = variant.abs(), parent.abs()

    def ratio(a: pd.Series, b: pd.Series) -> float:
        return float(a.mean() / b.mean()) if len(b) and b.mean() > 0 else math.nan

    years = sorted(set(v.index.year))
    return ratio(v, p), {y: ratio(v[v.index.year == y], p[p.index.year == y]) for y in years}


def compare(variant_exposure: pd.Series, parent_exposure: pd.Series, variant_returns: pd.Series,
            parent_returns: pd.Series, variant_percentile: float, scaled_percentile: float,
            own_edge: bool = False) -> EqualExposure:
    """variant_returns, parent_returns: their out-of-sample daily net returns; the percentiles: the variant's random
    entry, and the parent's with every trip sized k (the caller's, as only it has the trips); own_edge: (iii)."""
    k, by_year = exposure_ratio(variant_exposure, parent_exposure)
    scale = k if math.isfinite(k) else 0.0
    return EqualExposure(
        k=k, k_by_year=by_year, variant_exposure=float(variant_exposure.abs().mean()),
        parent_exposure=float(parent_exposure.abs().mean()),
        scaled_exposure=float((parent_exposure.abs() * scale).mean()),
        variant=summary(variant_returns), scaled=summary(parent_returns * scale),
        parent_sharpe=summary(parent_returns)["sharpe"], variant_percentile=variant_percentile,
        scaled_percentile=scaled_percentile, sharpe_share=paired_sharpe_share(variant_returns, parent_returns),
        own_edge=own_edge)


def _better(a: float, b: float) -> str:
    if not (math.isfinite(a) and math.isfinite(b)):
        return "n/a"
    return "yes" if a > b else "no"


def section(e: EqualExposure | None) -> list[str]:
    """The tear sheet's section for a volatility-sized variant; none for any other study."""
    if e is None:
        return []
    out = ["## Sizing to a target volatility, against the parent at the same exposure", ""]
    if not math.isfinite(e.k):
        return out + ["The parent held no position in the out-of-sample windows, so there is no exposure to match: "
                      "no comparison.", ""]
    years = ", ".join(f"{y} {k:.2f}" if math.isfinite(k) else f"{y} n/a" for y, k in e.k_by_year.items())
    v, s = e.variant, e.scaled

    def pct(x: float) -> str:
        return f"{x:+.2%}" if math.isfinite(x) else "n/a"

    def share(x: float) -> str:
        return f"{x:.0f}%" if math.isfinite(x) else "n/a"

    def mark(ok: bool) -> str:
        return "✓" if ok else "✗"

    return out + [
        f"**Variant better than parent: {'YES' if e.better else 'NO'}** (Sharpe {mark(e.sharpe_better)} "
        f"p={e.sharpe_share:.2f}, drawdown {mark(e.drawdown_better)}, own edge {mark(e.own_edge)}). Better needs all "
        f"three: a higher Sharpe than the parent's, higher in at least {SHARPE_SHARE:.0%} of {SHARPE_RESAMPLES:,} "
        f"paired resamples (stationary bootstrap, mean block {SHARPE_BLOCK_DAYS} days); a largest drawdown no worse "
        "than the scaled parent's; and its own edge (95th percentile of random entries, profitable at twice the fees). "
        "CAGR is shown and does not count.",
        "",
        f"k = {e.k:.3f}: the variant's average gross exposure ({e.variant_exposure:.1%}) over the parent's "
        f"({e.parent_exposure:.1%}), out of sample. By test year: {years}. The parent's returns are scaled by k, "
        f"costed at k x its notional (fees, spread and funding in proportion), so it holds {e.scaled_exposure:.1%} "
        "on average, the variant's.",
        "",
        "| | Parent scaled by k | Variant | Variant better |",
        "|---|---|---|---|",
        f"| CAGR (shown, does not count) | {pct(s['cagr'])} | {pct(v['cagr'])} | {_better(v['cagr'], s['cagr'])} |",
        f"| Largest drawdown | {pct(s['max_drawdown'])} | {pct(v['max_drawdown'])} | "
        f"{'yes' if e.drawdown_better else 'no'} |",
        f"| Random-entry percentile | {share(e.scaled_percentile)} | {share(e.variant_percentile)} | "
        f"{_better(e.variant_percentile, e.scaled_percentile)} |",
        f"| Sharpe (against the parent unscaled) | {e.parent_sharpe:.2f} | {v['sharpe']:.2f} | "
        f"{_better(v['sharpe'], e.parent_sharpe)} |",
        "",
    ]
