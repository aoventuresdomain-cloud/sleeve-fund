"""Research guardrails against overfitting (v2 P1-6, Independent Quant Advisor, adopted).

Two checks a study result must pass beyond the Sharpe test:
- at least MIN_OOS_TRADES round trips out of sample, since a Sharpe on fewer is mostly luck;
- the result holds at the settings next to the chosen ones, not only at one best value: every neighbour
  keeps a positive Sharpe and the median neighbour at least half the chosen Sharpe.
"""

from __future__ import annotations

import math

import pandas as pd

MIN_OOS_TRADES = 100  # the spec's "at least about 100 out-of-sample trades"
# Independent Quant Advisor (5 Oct 2026): every neighbour must keep a positive Sharpe, and the median neighbour
# at least this share of the chosen settings' Sharpe. A cliff next to the chosen value says it was picked from
# noise; one weak neighbour among several strong ones does not. Sharpe is net of fees and half the spread.
NEIGHBOUR_SHARE = 0.5


def _setting(v) -> str:
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def neighbours(sensitivity: pd.DataFrame, centre: dict, params: list[str]) -> pd.DataFrame:
    """The grid points one step from `centre` along exactly one setting, every other setting equal. A step is
    to the next value tried for that setting, above or below, so the check follows the grid's own spacing."""
    if sensitivity.empty or not params or any(p not in sensitivity.columns or p not in centre for p in params):
        return sensitivity.iloc[0:0]
    keep = []
    for p in params:
        others = [q for q in params if q != p]
        line = sensitivity
        for q in others:
            line = line[line[q] == centre[q]]
        values = sorted(line[p].unique())
        if centre[p] not in values:
            continue
        i = values.index(centre[p])
        near = [values[j] for j in (i - 1, i + 1) if 0 <= j < len(values)]
        keep.append(line[line[p].isin(near)])
    return pd.concat(keep) if keep else sensitivity.iloc[0:0]


def nearby_settings(sensitivity: pd.DataFrame, centre: dict, params: list[str]) -> tuple[str, str]:
    """PASS, FAIL or N/A for "holds at nearby settings", and the evidence in words."""
    row = sensitivity
    for p in params:
        if p not in sensitivity.columns or p not in centre:
            return "N/A", "the chosen settings are not on the grid tried"
        row = row[row[p] == centre[p]]
    if row.empty:
        return "N/A", "the chosen settings are not on the grid tried"
    near = neighbours(sensitivity, centre, params)
    if near.empty:
        return "N/A", "no nearby settings were tried"
    sharpe = float(row["sharpe"].iloc[0])
    bar = NEIGHBOUR_SHARE * sharpe
    if not math.isfinite(sharpe) or sharpe <= 0:
        return "FAIL", f"the chosen settings' Sharpe is {sharpe:.2f}, so there is nothing to hold"
    sharpes = near["sharpe"].astype(float).fillna(-math.inf)  # a neighbour with no Sharpe is a loser
    losing = int((sharpes <= 0).sum())
    median = float(sharpes.median())
    words = (f"{len(near) - losing} of {len(near)} nearby settings keep a positive Sharpe; the median one "
             f"{median:.2f} against a bar of {bar:.2f} ({NEIGHBOUR_SHARE:.0%} of the chosen {sharpe:.2f})")
    if losing or median < bar:
        worst = near.loc[sharpes.idxmin()]
        words += "; weakest " + ", ".join(f"{p} {_setting(worst[p])}" for p in params) + f" at {worst['sharpe']:.2f}"
    return ("PASS" if not losing and median >= bar else "FAIL"), words
