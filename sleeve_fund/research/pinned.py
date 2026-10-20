"""B-37 (Advisor: PIN IT): a study list's own dates. Each list under configs/studies/ pins the days its studies read:
the research period (data start to research end), the holdout after it, and the walk-forward's train/test splits.
A study run from the list reads those days and nothing else, whatever day it runs on: never a bar after the pinned
end, and the same list run on two different days gives the same results. A list that pins no dates is refused.

The list's block, in markdown (UTC calendar days; a day holds the bars that close after its 00:00 and by the next):

    ## Pinned dates
    - **Data start:** 2019-10-20
    - **Research end:** 2025-10-19
    - **Holdout:** 2025-10-20 to 2026-10-19
    | Fold | Train start | Test start | Test end |
    | 1 | 2019-10-20 | 2021-10-19 | 2022-10-18 |
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

_DAY = r"(\d{4}-\d{2}-\d{2})"
_START = re.compile(rf"^-\s*\*\*Data start:\*\*\s*{_DAY}\s*$", re.M)
_END = re.compile(rf"^-\s*\*\*Research end:\*\*\s*{_DAY}\s*$", re.M)
_HOLDOUT = re.compile(rf"^-\s*\*\*Holdout:\*\*\s*{_DAY}\s+to\s+{_DAY}\s*$", re.M)
_FOLD = re.compile(rf"^\|\s*(\d+)\s*\|\s*{_DAY}\s*\|\s*{_DAY}\s*\|\s*{_DAY}\s*\|\s*$", re.M)
MISSING = ("the study list pins no dates: add its ## Pinned dates block (data start, research end, holdout, and each "
           "walk-forward fold's train start, test start and test end), so every run reads the same days whatever the "
           "day it runs")


@dataclass(frozen=True)
class Fold:
    train_start: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class Pinned:
    data_start: date
    research_end: date
    holdout_start: date
    holdout_end: date
    folds: tuple[Fold, ...]

    def __post_init__(self) -> None:
        if not self.data_start <= self.research_end:
            raise ValueError(f"pinned dates: the data start {self.data_start} is after the research end "
                             f"{self.research_end}")
        if self.holdout_start != self.research_end + timedelta(days=1) or self.holdout_end < self.holdout_start:
            raise ValueError(f"pinned dates: the holdout {self.holdout_start} to {self.holdout_end} must start the day "
                             f"after the research end {self.research_end}")
        if not self.folds:
            raise ValueError("pinned dates: no walk-forward folds")

    @property
    def holdout_days(self) -> int:
        return (self.holdout_end - self.holdout_start).days + 1

    @property
    def end(self) -> date:
        """The last day any run may read: the holdout's, opened or not."""
        return self.holdout_end


def _day(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise ValueError(f"pinned dates: {s!r} is not a calendar day") from None


def parse(text: str) -> Pinned:
    """The list's pinned dates; ValueError in words when the list pins none, or pins them inconsistently."""
    start, end, holdout = _START.findall(text), _END.findall(text), _HOLDOUT.findall(text)
    folds = _FOLD.findall(text)
    if not (start or end or holdout or folds):
        raise ValueError(MISSING)
    if len(start) != 1 or len(end) != 1 or len(holdout) != 1:
        raise ValueError("pinned dates: the list must pin exactly one data start, one research end and one holdout")
    numbers = [int(n) for n, *_ in folds]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError("pinned dates: the folds must be numbered 1, 2, 3... in order")
    return Pinned(_day(start[0]), _day(end[0]), _day(holdout[0][0]), _day(holdout[0][1]),
                  tuple(Fold(_day(a), _day(b), _day(c)) for _, a, b, c in folds))


def day_of(ts: pd.Timestamp, bar: pd.Timedelta) -> date:
    """The UTC day a bar belongs to: the day its open falls on (a daily bar closing at 00:00 is the day before's)."""
    return (ts - bar).tz_convert("UTC").date() if ts.tzinfo else (ts - bar).date()


def upto(prices: pd.DataFrame | None, last: date) -> pd.DataFrame | None:
    """Bars closing by the end of day `last`: nothing a pinned study may read lies after it."""
    if prices is None:
        return None
    return prices[prices.index <= pd.Timestamp(last + timedelta(days=1), tz="UTC")]


def cut(prices: pd.DataFrame, pinned: Pinned) -> pd.DataFrame:
    """The bars a pinned study reads: the data start to the holdout end, never one after. Refused when the history
    doesn't cover them all: a short history would move the holdout."""
    if prices.empty:
        raise ValueError("pinned dates: there is no history to read")
    bar = pd.Timedelta(minutes=_minutes(prices))
    have = day_of(prices.index[0], bar), day_of(prices.index[-1], bar)
    if have[0] > pinned.data_start or have[1] < pinned.end:
        raise ValueError(f"pinned dates: the history covers {have[0]} to {have[1]}, not the pinned {pinned.data_start} "
                         f"to {pinned.end}")
    kept = upto(prices, pinned.end)
    return kept[kept.index >= pd.Timestamp(pinned.data_start, tz="UTC") + bar]


def _minutes(prices: pd.DataFrame) -> int:
    from sleeve_fund.research.study import bar_minutes_of

    return bar_minutes_of(prices)


def check_folds(pinned: Pinned, research: pd.DataFrame, train_bars: int, test_bars: int, bar: pd.Timedelta) -> None:
    """The walk-forward the study is about to run is the pinned one, fold for fold; ValueError naming the first
    difference otherwise."""
    got, start = [], 0
    while start + train_bars + test_bars <= len(research):
        idx = research.index
        got.append(Fold(day_of(idx[start], bar), day_of(idx[start + train_bars], bar),
                        day_of(idx[start + train_bars + test_bars - 1], bar)))
        start += test_bars
    if day_of(research.index[-1], bar) != pinned.research_end:
        raise ValueError(f"pinned dates: the research period ends {day_of(research.index[-1], bar)}, not the pinned "
                         f"{pinned.research_end}")
    if tuple(got) != pinned.folds:
        first = next((i for i, (a, b) in enumerate(zip(got, pinned.folds)) if a != b), min(len(got), len(pinned.folds)))
        mine = got[first] if first < len(got) else "none"
        theirs = pinned.folds[first] if first < len(pinned.folds) else "none"
        raise ValueError(f"pinned dates: walk-forward fold {first + 1} would be {mine}, but the list pins {theirs}")
