"""Test 1 of the v4 testing order: does the day-type classifier mean anything?

At the decision time (session start + N minutes) each session is labelled Trend up, Trend down,
Balance or Unclear using only bars closed by then. We then measure what price did next:

- continuation: return from the decision close to the horizon, in the trend's direction, in
  daily-ATR units (should be clearly positive on Trend days),
- move: the absolute return over the same span (should be larger on Trend than Balance days),
- range: high-to-low after the decision, in ATR (should be smaller on Balance days),
- revisits: whether price touched VWAP again after the decision (should be likelier on Balance).

If Trend days don't continue more than Balance days, strategies 1 and 2 stop here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sleeve_fund.lab import blocks

HORIZONS = {"4h": 240, "8h": 480, "session": 24 * 60}


def session_outcomes(bars: pd.DataFrame, *, anchor: str = "utc", decide_at: int = 60, **thresholds) -> pd.DataFrame:
    """One row per session: label at the decision time and what happened next."""
    out = blocks.with_session_blocks(bars, anchor)
    out["label"] = blocks.classify(out, **thresholds)
    bar = blocks.bar_minutes(bars)
    rows = []
    for sid, s in out.groupby("session", sort=True):
        d = s[s["elapsed"] == decide_at]
        if d.empty or len(s) * bar < 23 * 60:  # incomplete session (gaps, or the data's edges)
            continue
        d = d.iloc[0]
        atr = d["atr_d"]
        if not atr or math.isnan(atr):
            continue
        after = s[s["elapsed"] > decide_at]
        row = {"session": sid, "label": d["label"], "atr": atr, "decide_close": d["close"]}
        for name, mins in HORIZONS.items():
            h = after[after["elapsed"] <= decide_at + mins]
            if h.empty:
                continue
            ret = (h["close"].iloc[-1] - d["close"]) / atr
            row[f"ret_{name}"] = ret
            row[f"range_{name}"] = (h["high"].max() - h["low"].min()) / atr
            row[f"revisit_{name}"] = bool(((h["low"] <= h["vwap"]) & (h["high"] >= h["vwap"])).any())
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        sign = df["label"].map({blocks.TREND_UP: 1, blocks.TREND_DOWN: -1}).fillna(0)
        for name in HORIZONS:
            if f"ret_{name}" in df:
                df[f"cont_{name}"] = df[f"ret_{name}"] * sign
    return df


def _mean_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se else float("nan")


def summarise(df: pd.DataFrame, horizon: str = "session", seed: int = 7) -> dict:
    """The verdict numbers for one instrument (or the pooled basket)."""
    rng = np.random.default_rng(seed)
    if df.empty:
        return {"sessions": 0}
    trend = df[df["label"].isin([blocks.TREND_UP, blocks.TREND_DOWN])]
    bal = df[df["label"] == blocks.BALANCE]
    allx = df
    r, c, rg, rv = f"ret_{horizon}", f"cont_{horizon}", f"range_{horizon}", f"revisit_{horizon}"
    cont = _mean_ci(trend[c].to_numpy(float), rng)
    out = {
        "sessions": len(df),
        "share": {k: round(float((df["label"] == k).mean()), 3) for k in
                  (blocks.TREND_UP, blocks.TREND_DOWN, blocks.BALANCE, blocks.UNCLEAR)},
        "trend_n": len(trend), "balance_n": len(bal),
        "trend_continuation_atr": cont,
        "trend_continuation_hit": float((trend[c] > 0).mean()) if len(trend) else float("nan"),
        "move_trend_atr": float(trend[r].abs().mean()) if len(trend) else float("nan"),
        "move_balance_atr": float(bal[r].abs().mean()) if len(bal) else float("nan"),
        "move_all_atr": float(allx[r].abs().mean()),
        "move_t_trend_vs_balance": _welch_t(trend[r].abs().to_numpy(float), bal[r].abs().to_numpy(float)),
        "range_trend_atr": float(trend[rg].mean()) if len(trend) else float("nan"),
        "range_balance_atr": float(bal[rg].mean()) if len(bal) else float("nan"),
        "revisit_trend": float(trend[rv].mean()) if len(trend) else float("nan"),
        "revisit_balance": float(bal[rv].mean()) if len(bal) else float("nan"),
    }
    lo = cont[1]
    out["verdict"] = verdict(out, lo)
    return out


def verdict(s: dict, cont_lo: float) -> str:
    """Plain reading of the numbers. Thresholds are deliberately simple and stated up front."""
    if s.get("trend_n", 0) < 30 or s.get("balance_n", 0) < 30:
        return "too few sessions to judge"
    trends_on = cont_lo > 0
    bigger = s["move_t_trend_vs_balance"] > 2
    if trends_on and bigger:
        return "works: trend days keep going and move more than balance days"
    if bigger and not trends_on:
        return "partial: trend days move more, but not reliably in the trend's direction"
    if trends_on:
        return "partial: trend days keep going, but don't move clearly more than balance days"
    return "fails: trend days don't behave differently enough"
