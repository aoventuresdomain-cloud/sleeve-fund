"""R4: the probability of backtest overfitting, by combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de
Prado and Zhu). A G1 check (Independent Quant Advisor 7 Oct 14:05, 18:20 Q2 and 18:25 Q7 UK; PE1 plan section 3).

The study's sensitivity grid gives one daily net return series per configuration tried, discarded ones included, all
over the same research-period days. The days are cut into BLOCKS contiguous blocks; for each of the C(16, 8) = 12,870
ways to take half of them as in-sample, the configuration with the best in-sample Sharpe is ranked among all of them on
the other half. PBO is the share of splits in which that pick ranks at or below the median out of sample: how often
choosing the best in-sample picks one that is no better than average out of sample.

Fail above 0.5, warn from 0.25 to 0.5, pass below 0.25. Fewer than MIN_VARIANTS configurations shows "too few
variants"; too few days for the blocks shows "n/a" with the reason. Ablation variants are never fed in (Advisor 18:25
Q8). Every split is computed, so the same returns always give the same value."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

BLOCKS = 16  # C(16, 8) = 12,870 splits
PBO_CHECK = "Probability of backtest overfitting"
MIN_VARIANTS = 10  # Advisor 14:05, confirmed 18:25 Q7
MIN_DAYS_PER_BLOCK = 2  # a Sharpe needs a spread
FAIL_ABOVE = 0.5
WARN_FROM = 0.25
TOO_FEW = "too few variants"
NA = "n/a"


@dataclass(frozen=True)
class PBOResult:
    value: float | None
    verdict: str  # "PASS", "WARN", "FAIL", TOO_FEW or NA
    n_variants: int
    n_combos: int = 0
    words: str = ""


def verdict_of(value: float) -> str:
    """0.5 itself warns (only above fails); 0.25 itself warns (only below passes)."""
    return "FAIL" if value > FAIL_ABOVE else "WARN" if value >= WARN_FROM else "PASS"


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis, 1 the lowest; ties share the mean of their places."""
    order = np.argsort(x, axis=-1, kind="stable")
    ranked = np.take_along_axis(x, order, axis=-1)
    out = np.empty_like(x, dtype=float)
    for idx in np.ndindex(x.shape[:-1]):
        row, pos = ranked[idx], np.arange(1, x.shape[-1] + 1, dtype=float)
        start = 0
        while start < len(row):
            end = start + 1
            while end < len(row) and row[end] == row[start]:
                end += 1
            pos[start:end] = (start + 1 + end) / 2
            start = end
        out[idx][order[idx]] = pos
    return out


def pbo(returns: pd.DataFrame, blocks: int = BLOCKS) -> PBOResult:
    """returns: one column per configuration tried, one row per day, daily net returns. Rows with a gap in any
    column are dropped, so every configuration is judged on the same days."""
    data = returns.dropna(how="any")
    n = data.shape[1]
    if n < MIN_VARIANTS:
        return PBOResult(None, TOO_FEW, n, words=f"{n} configuration{'s' if n != 1 else ''} tried, fewer than the "
                         f"{MIN_VARIANTS} it needs")
    days = len(data)
    if days < blocks * MIN_DAYS_PER_BLOCK:
        return PBOResult(None, NA, n, words=f"{days} days of returns, fewer than the {blocks * MIN_DAYS_PER_BLOCK} "
                         f"{blocks} blocks need")
    x = data.to_numpy(dtype=float)
    parts = np.array_split(np.arange(days), blocks)  # contiguous, sizes differing by at most a day
    count = np.array([len(p) for p in parts], dtype=float)
    means = np.array([x[p].mean(axis=0) for p in parts])  # blocks x variants
    within = np.array([((x[p] - x[p].mean(axis=0)) ** 2).sum(axis=0) for p in parts])
    combos = list(itertools.combinations(range(blocks), blocks // 2))
    mask = np.zeros((len(combos), blocks), dtype=bool)
    for i, c in enumerate(combos):
        mask[i, list(c)] = True

    def sharpe(m: np.ndarray) -> np.ndarray:
        """Each half's Sharpe from its blocks' counts, means and squared deviations, combined without the
        cancellation a raw sum of squares suffers (Chan et al.)."""
        w = m * count  # splits x blocks: each block's days where it is in the half
        k = w.sum(axis=1)[:, None]
        mean = (w @ means) / k
        spread = m.astype(float) @ within + np.einsum("sb,sbv->sv", w, (means[None, :, :] - mean[:, None, :]) ** 2)
        sd = np.sqrt(spread / (k - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 1e-15, mean / sd, 0.0)  # a flat series has no Sharpe: scored 0

    is_sharpe, oos_sharpe = sharpe(mask), sharpe(~mask)
    best = np.argmax(is_sharpe, axis=1)  # the first of tied bests, in column order
    rank = _ranks(oos_sharpe)[np.arange(len(combos)), best]
    omega = rank / (n + 1)
    logit = np.log(omega / (1 - omega))
    value = float(np.mean(logit <= 0))
    verdict = verdict_of(value)
    return PBOResult(value, verdict, n, len(combos),
                     words=f"{value:.0%} of {len(combos):,} half-and-half splits of {days} days: the best of {n} "
                           f"configurations in-sample ranked at or below the median out of sample (fail above "
                           f"{FAIL_ABOVE:.0%}, warn from {WARN_FROM:.0%})")


def g1_row(result: PBOResult | None) -> tuple[str, str, str]:
    """The tear sheet's G1 row. FAIL fails G1; WARN shows amber and doesn't; too few variants or n/a is shown, not a
    verdict."""
    if result is None:
        return PBO_CHECK, "N/A", "not computed for this study"
    if result.value is None:
        return PBO_CHECK, "N/A", f"{result.verdict}: {result.words}"
    return PBO_CHECK, result.verdict, result.words

