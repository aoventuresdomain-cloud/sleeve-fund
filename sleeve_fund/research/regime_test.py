"""P2-5 acceptance test for regime weighting (Advisor spec quant-review/specs-p2-3-p2-8-p2-5.md section 4, MUST): the
weighted run against two nulls, the unweighted run, double fees and a minimum of entries per state. A fail means the
weight is 1.0 everywhere; nothing here re-tunes a mapping to rescue it (a new mapping is a new counted idea).

Primary null: the regime state series circularly shifted by a random offset of at least MIN_SHIFT_DAYS per draw, which
keeps how long regimes last and how often each weight is used but breaks their timing. Secondary null: each entry's
weight drawn independently with the frequencies the regime run used. DRAWS seeded draws of each."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DRAWS = 1000
MIN_SHIFT_DAYS = 60
PCTL = 95.0
MAX_DD_WORSE = 0.10  # relative: the weighted run's max drawdown at most 10% deeper than the unweighted run's
MIN_ENTRIES_PER_STATE = 30
STATES = ("low", "mid", "high")
NOT_JUDGED, PASS, FAIL = "NOT JUDGED", "PASS", "FAIL"


def shifted_states(states: pd.Series, rng: np.random.Generator, min_shift: int = MIN_SHIFT_DAYS) -> pd.Series:
    """The primary null's state series: `states` (one per day, in date order) rolled by an offset in
    [min_shift, len - min_shift], wrapping round, on the same dates."""
    n = len(states)
    if n < 2 * min_shift + 1:
        raise ValueError(f"a shifted null needs more than {2 * min_shift} days of regime states, got {n}")
    k = int(rng.integers(min_shift, n - min_shift + 1))
    return pd.Series(np.roll(states.to_numpy(), k), index=states.index)


def iid_weights(used: Sequence, rng: np.random.Generator) -> list:
    """The secondary null's weights: one per entry, drawn independently with the frequencies of `used` (the weights
    the regime run gave its entries)."""
    values, counts = np.unique(np.asarray([float(w) for w in used]), return_counts=True)
    return list(rng.choice(values, size=len(used), p=counts / counts.sum()))


def daily_sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe of daily net returns (365 days: the venue trades every day)."""
    r = returns.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(365)) if len(r) > 1 and sd > 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    """Deepest fall of the compounded curve from its high-water mark, as a positive fraction."""
    curve = (1 + returns.fillna(0)).cumprod()
    return float((1 - curve / curve.cummax()).max()) if len(curve) else 0.0


@dataclass(frozen=True)
class Verdict:
    result: str  # PASS, FAIL or NOT JUDGED
    sharpe: float
    unweighted_sharpe: float
    primary_p95: float
    secondary_p95: float
    drawdown: float
    unweighted_drawdown: float
    double_fees: bool
    entries: dict
    reasons: tuple[str, ...]


def judge(weighted: pd.Series, unweighted: pd.Series, primary: Sequence[float], secondary: Sequence[float],
          double_fees_pass: bool, entries_per_state: dict) -> Verdict:
    """Section 4.4 on the out-of-sample daily net returns: every condition must hold. Fewer than MIN_ENTRIES_PER_STATE
    entries in any state, or a null that isn't complete, is NOT JUDGED, never a pass. The holdout confirm is a
    separate single run after a pass."""
    sr, base = daily_sharpe(weighted), daily_sharpe(unweighted)
    p1 = float(np.percentile(primary, PCTL)) if len(primary) else float("nan")
    p2 = float(np.percentile(secondary, PCTL)) if len(secondary) else float("nan")
    dd, base_dd = max_drawdown(weighted), max_drawdown(unweighted)
    entries = {s: int(entries_per_state.get(s, 0)) for s in STATES}
    short = [s for s, n in entries.items() if n < MIN_ENTRIES_PER_STATE]
    args = (sr, base, p1, p2, dd, base_dd, double_fees_pass, entries)
    if short:
        why = f"fewer than {MIN_ENTRIES_PER_STATE} entries in {', '.join(short)}"
        return Verdict(NOT_JUDGED, *args, (why,))
    if len(primary) < DRAWS or len(secondary) < DRAWS or not np.isfinite([sr, base, p1, p2]).all():
        return Verdict(NOT_JUDGED, *args, (f"the nulls need {DRAWS} draws each and every Sharpe measured",))
    fails = []
    if not sr > p1:
        fails.append(f"Sharpe {sr:.2f} not above the shifted-regime null's 95th percentile {p1:.2f}")
    if not sr > p2:
        fails.append(f"Sharpe {sr:.2f} not above the random-weight null's 95th percentile {p2:.2f}")
    if sr < base:
        fails.append(f"Sharpe {sr:.2f} below unweighted {base:.2f}")
    if dd > base_dd * (1 + MAX_DD_WORSE):
        fails.append(f"max drawdown {dd:.1%} more than 10% worse than unweighted {base_dd:.1%}")
    if not double_fees_pass:
        fails.append("fails at double fees")
    return Verdict(FAIL if fails else PASS, *args, tuple(fails))
