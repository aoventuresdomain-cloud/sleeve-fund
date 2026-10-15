"""R5: minimum track record length (Bailey and Lopez de Prado, 2012). How many days of daily returns a measured Sharpe
needs before it is above zero with 95% confidence, fat tails counted through the returns' skew and kurtosis
(Independent Quant Advisor, 6 Oct 2026 14:05).

Reported at G1 (shown, not a test). Gated at G2: no scaling past the minimum size until the paper or live track is at
least this long (v2/before-g2.md); may_scale() is that gate's pure core."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import pandas as pd

CONFIDENCE = 0.95
MIN_DAYS = 30  # fewer days give no usable skew or kurtosis: not computed
CHECK = "Minimum track record (shown, not a test; gates G2)"


@dataclass(frozen=True)
class TrackRecord:
    days_needed: float | None  # None: not computed (too few days) or never (no edge to prove)
    days_have: int
    sharpe: float | None  # daily, unannualised
    words: str

    @property
    def met(self) -> bool:
        return self.days_needed is not None and self.days_have >= self.days_needed


def min_trl(returns: pd.Series, confidence: float = CONFIDENCE, benchmark: float = 0.0) -> TrackRecord:
    """returns: daily net returns. benchmark: the daily Sharpe to beat (0: any edge at all).
    MinTRL = 1 + (1 - skew x SR + (kurtosis - 1) / 4 x SR^2) x (z / (SR - benchmark))^2, kurtosis not excess."""
    r = returns.dropna()
    n = len(r)
    if n < MIN_DAYS or r.std(ddof=1) == 0:
        return TrackRecord(None, n, None, f"not computed: {n} days of returns, fewer than the {MIN_DAYS} it needs"
                           if n < MIN_DAYS else "not computed: the returns never move")
    sr = float(r.mean() / r.std(ddof=1))
    if sr <= benchmark:
        return TrackRecord(None, n, sr, f"never: the Sharpe measured over {n} days shows no edge to prove "
                           "(at or below zero), so no track record would confirm one")
    skew, kurt = float(r.skew()), float(r.kurt()) + 3
    z = NormalDist().inv_cdf(confidence)
    need = 1 + max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12) * (z / (sr - benchmark)) ** 2
    rec = TrackRecord(need, n, sr, "")
    met = f"the {n} days measured are enough" if rec.met else f"the {n} days measured are not enough yet"
    return TrackRecord(need, n, sr, f"{math.ceil(need):,} days (about {need / 30.44:.1f} months) of daily returns for "
                       f"{confidence:.0%} confidence the Sharpe is above zero, fat tails counted (skew {skew:+.2f}, "
                       f"kurtosis {kurt:.1f}); {met}. At G2 the strategy stays at minimum size until its paper or "
                       "live track is this long")


def g1_row(returns: pd.Series) -> tuple[str, str, str]:
    """The tear sheet's row, on the out-of-sample daily returns: shown, never a G1 pass or fail."""
    return CHECK, "INFO", min_trl(returns).words


def may_scale(track: pd.Series, confidence: float = CONFIDENCE) -> tuple[bool, str]:
    """The G2 gate: may a strategy scale past its minimum size? Only once its own paper or live track (daily net
    returns) is at least its minimum track record length, measured on that track."""
    rec = min_trl(track, confidence)
    if rec.days_needed is None:
        return False, f"stays at minimum size: track record {rec.words}"
    if not rec.met:
        return False, (f"stays at minimum size: {rec.days_have} of the {math.ceil(rec.days_needed):,} days its "
                       "track needs")
    return True, f"may scale: its {rec.days_have}-day track is at least the {math.ceil(rec.days_needed):,} days needed"
