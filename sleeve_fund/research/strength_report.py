"""G5 research side (Done-when 5; Advisor ruling G5-DW5, 15 Oct 2026 ~10:07 UK): a strength-sized variant side by side
with its flat version, on the same out-of-sample windows, on risk-adjusted terms. Sizing by strength lowers exposure,
so raw return alone misleads: the variant is kept only if it beats the flat version on Sharpe AND on return per unit of
average gross exposure, net of fees, out of sample. A tie, or no flat version to compare, keeps the flat version.

Average gross exposure is time-weighted: the mean over every test bar of |position notional / equity| marked to market,
a flat bar counting 0 (Advisor Q2). The paired difference (strength minus flat, day by day) is shown with a moving-block
bootstrap 90% interval, for information only (Advisor FOLLOW-UP): it never decides the verdict.

Report only. A flat study (no signal_strength) gets no section and is unchanged."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sleeve_fund.research.metrics import summary

KEPT, FLAT_STAYS = "kept", "flat stays"
BOOT_DRAWS = 2000
INTERVAL = 0.90


@dataclass(frozen=True)
class Stats:
    sharpe: float
    net_return: float
    exposure: float  # time-weighted mean gross exposure over the test bars, flat bars 0
    per_exposure: float  # net return per unit of average gross exposure; nan when never exposed
    trades: int
    breakeven: str


def is_sized(r) -> bool:
    """A study of a strength-sized variant: its settings carry signal_strength."""
    params = {**(getattr(r, "default_params", None) or {}), **(getattr(r, "chosen_params", None) or {})}
    return params.get("signal_strength") is not None


def stats(r) -> Stats:
    """The out-of-sample figures of one study (its folds' test windows)."""
    s = summary(r.oos_returns)
    bars = sum(f.window_bars for f in r.folds)
    exposure = sum(f.gross_exposure for f in r.folds) / bars if bars else 0.0
    net = s["total_return"]
    return Stats(sharpe=s["sharpe"], net_return=net, exposure=exposure,
                 per_exposure=net / exposure if exposure > 0 else math.nan,
                 trades=sum(f.test_trades for f in r.folds), breakeven=getattr(r, "breakeven", "") or "")


def paired_interval(flat: pd.Series, sized: pd.Series, seed: int = 0, draws: int = BOOT_DRAWS,
                    block: int | None = None) -> tuple[float, float, float]:
    """The mean daily difference, strength minus flat, on the days both have, and its moving-block bootstrap 90%
    interval (blocks of about the cube root of the days, at least 2, so runs of days keep their dependence)."""
    d = (sized - flat).dropna().to_numpy(dtype=float)
    n = len(d)
    if n == 0:
        return math.nan, math.nan, math.nan
    b = block or max(2, round(n ** (1 / 3)))
    b = min(b, n)
    rng = np.random.default_rng(seed)
    k = math.ceil(n / b)
    starts = rng.integers(0, n - b + 1, size=(draws, k))
    idx = (starts[:, :, None] + np.arange(b)).reshape(draws, -1)[:, :n]
    means = d[idx].mean(axis=1)
    tail = (1 - INTERVAL) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(d.mean()), float(lo), float(hi)


@dataclass(frozen=True)
class Comparison:
    verdict: str
    lines: list
    interval: tuple = (math.nan, math.nan)


def _beats(a: float, b: float) -> bool:
    return math.isfinite(a) and (not math.isfinite(b) or a > b)


def compare(flat, sized, seed: int = 0) -> Comparison:
    """The verdict and the table's lines. flat: the flat version's study on the same windows, or None."""
    s = stats(sized)
    if flat is None:
        why = "no flat version on the same windows to compare against, so the flat version stays"
        return Comparison(FLAT_STAYS, [why], (math.nan, math.nan))
    f = stats(flat)
    mean, lo, hi = paired_interval(flat.oos_returns, sized.oos_returns, seed=seed)
    kept = _beats(s.sharpe, f.sharpe) and _beats(s.per_exposure, f.per_exposure)
    verdict = KEPT if kept else FLAT_STAYS

    def pct(x: float) -> str:
        return f"{x:+.2%}" if math.isfinite(x) else "n/a"

    rows = [
        "| | Flat | Strength-sized |",
        "|---|---|---|",
        f"| Sharpe | {f.sharpe:.2f} | {s.sharpe:.2f} |",
        f"| Return per unit of exposure | {pct(f.per_exposure)} | {pct(s.per_exposure)} |",
        f"| Net return | {pct(f.net_return)} | {pct(s.net_return)} |",
        f"| Average gross exposure | {f.exposure:.1%} | {s.exposure:.1%} |",
        f"| Trades | {f.trades} | {s.trades} |",
        f"| Break-even fee | {f.breakeven or 'n/a'} | {s.breakeven or 'n/a'} |",
        (f"| Strength minus flat, mean daily return | | {mean:+.4%} (90% interval {lo:+.4%} to {hi:+.4%}; "
         "information only) |"),
    ]
    why = ("beats the flat version on Sharpe and on return per unit of exposure, net of fees, out of sample" if kept
           else "doesn't beat the flat version on both Sharpe and return per unit of exposure (a tie keeps flat)")
    return Comparison(verdict, rows + ["", f"Verdict: {verdict}: {why}"], (lo, hi))


def section(sized, flat) -> list[str]:
    """The tear sheet's section for a strength-sized study; none for a flat one."""
    if not is_sized(sized):
        return []
    c = compare(flat, sized)
    return ["## Sizing by signal strength, side by side with the flat version", "",
            *(c.lines if flat is not None else [f"Verdict: {c.verdict}: {c.lines[0]}"]), ""]
