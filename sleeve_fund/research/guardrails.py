"""Research guardrails against overfitting (v2 P1-6, Independent Quant Advisor, adopted).

Two checks a study result must pass beyond the Sharpe test:
- at least MIN_OOS_TRADES round trips out of sample, since a Sharpe on fewer is mostly luck;
- the result holds at the settings next to the chosen ones, not only at one best value: every neighbour
  keeps a positive Sharpe and the median neighbour at least half the chosen Sharpe. Chosen means what
  tuning picked, in each walk-forward fold and over the whole period, never the defaults (P1-G2).
"""

from __future__ import annotations

import math

import pandas as pd

# The G1 rules a sheet was judged under, written on it. A pass under any other rules no longer counts: the
# pipeline reads it as "old bar" until the study is re-run (QA F3). Change it whenever G1 gets stricter.
G1_RULES = "2026-10-06"
MIN_OOS_TRADES = 100  # the spec's "at least about 100 out-of-sample trades"
# Independent Quant Advisor (5 Oct 2026): every neighbour must keep a positive Sharpe, and the median neighbour
# at least this share of the chosen settings' Sharpe. A cliff next to the chosen value says it was picked from
# noise; one weak neighbour among several strong ones does not. Sharpe is net of fees and half the spread.
NEIGHBOUR_SHARE = 0.5


def _setting(v) -> str:
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _order(v) -> tuple:
    """Sort key for a setting's values: a grid can mix words, numbers and None without the sort raising
    (QA F5). Missing first, then numbers, then words."""
    if _missing(v):
        return (0, 0.0, "")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (1, float(v), "")
    return (2, 0.0, str(v))


def _equal(a, b) -> bool:
    return (_missing(a) and _missing(b)) or (not _missing(a) and not _missing(b) and _order(a) == _order(b))


def _at(frame: pd.DataFrame, p: str, v) -> pd.DataFrame:
    """The rows whose setting `p` is `v`; a None default matches a missing value, which == never does (QA F5)."""
    return frame[frame[p].map(lambda x: _equal(x, v))]


def _values(frame: pd.DataFrame, p: str) -> list:
    out = []
    for v in sorted(frame[p].tolist(), key=_order):
        if not any(_equal(v, w) for w in out):
            out.append(v)
    return out


def tunable(sensitivity: pd.DataFrame, params: list[str]) -> list[str]:
    """The settings the grid actually varies: the ones a choice was made on."""
    return [p for p in params if p in sensitivity.columns and len(_values(sensitivity, p)) > 1]


def _line(sensitivity: pd.DataFrame, centre: dict, params: list[str], p: str) -> pd.DataFrame:
    line = sensitivity
    for q in params:
        if q != p:
            line = _at(line, q, centre[q])
    return line


def neighbours(sensitivity: pd.DataFrame, centre: dict, params: list[str]) -> pd.DataFrame:
    """The grid points one step from `centre` along exactly one setting, every other setting equal. A step is
    to the next value tried for that setting, above or below, so the check follows the grid's own spacing."""
    if sensitivity.empty or not params or any(p not in sensitivity.columns or p not in centre for p in params):
        return sensitivity.iloc[0:0]
    keep = []
    for p in params:
        line = _line(sensitivity, centre, params, p)
        values = _values(line, p)
        i = next((j for j, v in enumerate(values) if _equal(v, centre[p])), None)
        if i is None:
            continue
        near = [values[j] for j in (i - 1, i + 1) if 0 <= j < len(values)]
        keep.append(line[line[p].map(lambda x: any(_equal(x, v) for v in near))])
    return pd.concat(keep) if keep else sensitivity.iloc[0:0]


def _trades(row) -> str:
    n = row.get("round_trips")
    return "" if n is None or _missing(n) else f", {int(n)} trade{'' if int(n) == 1 else 's'}"


def nearby_settings(sensitivity: pd.DataFrame, centre: dict, params: list[str]) -> tuple[str, str]:
    """PASS or FAIL for "holds at nearby settings", and the evidence in words."""
    verdict, words, _ = nearby_scored(sensitivity, centre, params)
    return verdict, words


def nearby_scored(sensitivity: pd.DataFrame, centre: dict, params: list[str]) -> tuple[str, str, float]:
    """nearby_settings plus how close it came: the median neighbour over its bar, -inf when it couldn't be
    checked, so the worst of several checks can be reported. Independent Quant Advisor (6 Oct 2026):
    - with nothing tunable it passes, flagged (P1-G3);
    - a tunable setting without a chosen value, or chosen settings off the grid, fail: the check can't be
      made, and N/A let an untested choice through (P1-G3);
    - a choice at the edge of the grid fails: it has a neighbour on one side only, and the best value may
      lie beyond the grid (P1-G4)."""
    tuned = tunable(sensitivity, params)
    if not tuned:
        return "PASS", "flag: the grid varies no setting, so there was no choice to hold", math.inf
    unset = [p for p in tuned if p not in centre]
    if unset:
        return "FAIL", f"no chosen value for {', '.join(unset)}, so nearness can't be checked", -math.inf
    row = sensitivity
    for p in tuned:
        row = _at(row, p, centre[p])
    where = ", ".join(f"{p} {_setting(centre[p])}" for p in tuned)
    if row.empty:
        return "FAIL", f"the chosen settings ({where}) are not on the grid tried, so nearness can't be checked", -math.inf
    edges, lonely = [], []
    for p in tuned:
        values = _values(_line(sensitivity, centre, tuned, p), p)
        if len(values) < 2:
            lonely.append(p)  # tuned elsewhere on the grid, but nothing tried next to this choice (QA P1-G5)
        elif _equal(centre[p], values[0]) or _equal(centre[p], values[-1]):
            edges.append(f"{p} {_setting(centre[p])}")
    if lonely:
        return "FAIL", f"no nearby value tried for {', '.join(lonely)}, so its nearness can't be checked", -math.inf
    if edges:
        return "FAIL", f"chosen at grid edge ({', '.join(edges)}), extend the grid", -math.inf
    near = neighbours(sensitivity, centre, tuned)
    if near.empty:
        return "FAIL", "no nearby settings were tried, so nearness can't be checked", -math.inf
    sharpe = float(row["sharpe"].iloc[0])
    bar = NEIGHBOUR_SHARE * sharpe
    if not math.isfinite(sharpe) or sharpe <= 0:
        return "FAIL", f"the chosen settings' Sharpe is {sharpe:.2f}, so there is nothing to hold", -math.inf
    sharpes = near["sharpe"].astype(float).fillna(-math.inf)  # a neighbour with no Sharpe is a loser
    losing = int((sharpes <= 0).sum())
    median = float(sharpes.median())
    words = (f"{len(near) - losing} of {len(near)} nearby settings keep a positive Sharpe; the median one "
             f"{median:.2f} against a bar of {bar:.2f} ({NEIGHBOUR_SHARE:.0%} of the chosen {sharpe:.2f})")
    if "round_trips" in near.columns:
        idle = int((near["round_trips"].fillna(0) == 0).sum())
        if idle:
            words += f"; {idle} of them made no trades"
    if losing or median < bar:
        worst = near.loc[sharpes.idxmin()]
        if isinstance(worst, pd.DataFrame):
            worst = worst.iloc[0]
        words += ("; weakest " + ", ".join(f"{p} {_setting(worst[p])}" for p in tuned)
                  + f" at {worst['sharpe']:.2f}{_trades(worst)}")
    return ("PASS" if not losing and median >= bar else "FAIL"), words, median / bar
