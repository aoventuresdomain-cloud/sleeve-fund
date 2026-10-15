"""G5: sizing from signal strength, the engine hook (Independent Quant Advisor spec, 15 Oct 2026 ~09:25 UK,
quant-review/g5-signal-strength-spec.md). OFF by default: a Model with no `signal_strength` setting sizes exactly as
before.

A Model attaches a strength to an entry signal, in its values: "strength", a number from 0 to 1, or "votes", k agreeing
of n. Its configured size (the smallest of its own sizing limits: risk per trade, the risk profile's cap, a size share or
step) is scaled by f(strength), 0 <= f <= 1, so the configured size is the CEILING: strength never makes a position
bigger. Every other limit still applies after it, smallest wins (the leverage cap, the venue minimum, the ramp, then the
compliance tests at the gate). Three mappings only, no free-form curves:

- linear: f = s, with a floor; below the floor the entry is skipped and journaled.
- votes: f = k / n, with a minimum k; below it the entry is skipped.
- steps: at most 3 fixed bands, e.g. [0.33, 0.66, 1.0]: f is the highest band at or below s; below the lowest, skipped.

A strength that is missing, NaN or out of range skips the entry with an alert: it fails closed, never full size. Size is
set at entry only (adds are sized from the strength at the add); stops and take-profits are unchanged."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from decimal import Decimal

MAPPINGS = ("linear", "votes", "steps")
LIMIT = "signal strength"  # the sizing limit's name in the journal's "sized by"
MAX_BANDS = 3
# Rebalance trades (a band's top-ups) are sized outside the entry sizers, so strength would silently not reach them
# (CR F-G5-1): the two can't be combined yet.
NO_REBALANCE = ("sizing by signal strength works on entries; rebalancing to a target weight trades outside them, so "
                "the two can't be combined yet")
# A leg of a two-leg Strategy is sized by its Pair, never by its own entry sizers (QD F-G5-4).
NO_LEG = ("sizing by signal strength works on a Strategy's own entries; a leg of a two-leg Strategy is sized by its "
          "pair, so the two can't be combined")


def check(cfg: dict | None) -> dict | None:
    """A Model's signal_strength setting, validated: {"mapping": "linear", "floor": 0.3}, {"mapping": "votes",
    "min_k": 3} or {"mapping": "steps", "bands": [0.33, 0.66, 1.0]}. None: off."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict) or cfg.get("mapping") not in MAPPINGS:
        raise ValueError(f"signal_strength needs a mapping, one of {', '.join(MAPPINGS)}, not {cfg!r}")
    kind = cfg["mapping"]
    keys = {"linear": {"mapping", "floor"}, "votes": {"mapping", "min_k"}, "steps": {"mapping", "bands"}}[kind]
    if set(cfg) - keys:
        raise ValueError(f"signal_strength {kind} takes only {', '.join(sorted(keys - {'mapping'}))}, not "
                         f"{', '.join(sorted(set(cfg) - keys))}")
    if kind == "linear":
        floor = cfg.get("floor", 0.0)
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not 0 <= floor <= 1:
            raise ValueError(f"the linear floor is a number from 0 to 1, not {floor!r}")
        return {"mapping": kind, "floor": float(floor)}
    if kind == "votes":
        k = cfg.get("min_k", 1)
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"the votes minimum k is a whole number from 1, not {k!r}")
        return {"mapping": kind, "min_k": k}
    bands = cfg.get("bands")
    if (not isinstance(bands, (list, tuple)) or not 1 <= len(bands) <= MAX_BANDS
            or any(isinstance(b, bool) or not isinstance(b, (int, float)) or not 0 < b <= 1 for b in bands)
            or list(bands) != sorted(set(bands))):
        raise ValueError(f"steps takes 1 to {MAX_BANDS} rising bands above 0 and at most 1, not {bands!r}")
    return {"mapping": kind, "bands": [float(b) for b in bands]}


@dataclass(frozen=True)
class Strength:
    f: float | None  # the share of the configured size; None: the entry is skipped
    words: str  # in the PM's words, for the journal
    alert: bool = False  # a missing, NaN or out-of-range strength: skipped with an alert


def _number(x) -> float | None:
    """A real number (numpy's and Decimal included, QD NIT), finite; never a bool or text."""
    if isinstance(x, bool) or not isinstance(x, (numbers.Real, Decimal)):
        return None
    try:
        f = float(x)
    except (OverflowError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _whole(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def factor(cfg: dict, values: dict) -> Strength:
    """f(strength) for one entry, from the Model's values; fails closed."""
    kind = cfg["mapping"]
    if kind == "votes":
        votes = values.get("votes")
        ok = (isinstance(votes, (list, tuple)) and len(votes) == 2
              and all(_whole(v) for v in votes) and votes[1] > 0 and 0 <= votes[0] <= votes[1])
        if not ok:
            return Strength(None, f"entry skipped: the Model's vote {votes!r} isn't k of n agreeing", alert=True)
        k, n = int(votes[0]), int(votes[1])
        if k < cfg["min_k"]:
            return Strength(None, f"entry skipped: {k} of {n} agree, fewer than the {cfg['min_k']} it needs")
        return Strength(k / n, f"signal strength {k} of {n} (votes, at least {cfg['min_k']})")
    s = _number(values.get("strength"))
    if s is None or not 0 <= s <= 1:
        return Strength(None, f"entry skipped: the Model's signal strength {values.get('strength')!r} isn't a "
                              "number from 0 to 1", alert=True)
    if kind == "linear":
        if s == 0:  # nothing to size: no entry, journaled (never an alert: 0 is a valid strength)
            return Strength(None, "entry skipped: signal strength 0, so no position")
        if s < cfg["floor"]:
            return Strength(None, f"entry skipped: signal strength {s:.2f} is below its floor {cfg['floor']:.2f}")
        return Strength(s, f"signal strength {s:.2f} (linear, floor {cfg['floor']:.2f})")
    below = [b for b in cfg["bands"] if b <= s]
    if not below:
        return Strength(None, f"entry skipped: signal strength {s:.2f} is below its lowest band {cfg['bands'][0]:g}")
    return Strength(below[-1], f"signal strength {s:.2f} (steps {', '.join(f'{b:g}' for b in cfg['bands'])})")


def words(st: Strength, configured_pct: float) -> str:
    """'signal strength 0.62 (linear, floor 0.30) -> 62% of 6.0% = 3.72% of fund'."""
    return (f"{st.words} -> {st.f:.0%} of {configured_pct:.1%} = {st.f * configured_pct:.2%} of fund"
            if st.f is not None else st.words)
