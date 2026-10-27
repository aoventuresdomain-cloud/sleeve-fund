"""STUDY-ASSET-LIST (PM 21 Oct; Deploy 3): a study request's asset list, run as one basket.

A StudyRequest with pairs=("BTC/USDT", "ETH/USDT", ...) runs its one definition over every listed asset as a B-38
basket (research.basket): one fund run of the whole list, judged on the pooled book, its per-asset lines diagnostic
only, and one try in the trials register for the fixed list and settings. The days are the single study's: the
list's pinned dates when it has them, the same walk-forward test windows (or, continuous, one window after the
definition's warm-up to the research end), and the holdout locked unless the request opens it."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from sleeve_fund.research import basket as bk

DAY = pd.Timedelta(days=1)


def dataset_name(venue: str, pairs, minutes: int = 1440) -> str:
    """The basket's dataset, as the trials register and its sheet name it: its assets in list order, and its bar
    length when not daily (as a single study's)."""
    quote = pairs[0].split("/")[1]
    return (f"{venue}-basket-{'-'.join(p.split('/')[0] for p in pairs)}-{quote}-store".lower()
            + (f"-{minutes}m" if minutes != 1440 else ""))


def _day(ts: pd.Timestamp, bar: pd.Timedelta = DAY) -> date:
    """The UTC day a bar of length `bar` closing at `ts` holds (a bar closing at 00:00 is the day before's)."""
    return (ts - bar).date()


def common_clock(prices: list[pd.DataFrame], names: list[str]) -> list[pd.DataFrame]:
    """Every asset's bars over the days all of them have: from the latest first bar to the earliest last one. A gap
    inside that span is left for basket.check_assets to refuse, naming the asset."""
    first = max(p.index[0] for p in prices)
    last = min(p.index[-1] for p in prices)
    if first >= last:
        raise ValueError(f"the listed assets' histories don't overlap: {', '.join(names)}")
    return [p[(p.index >= first) & (p.index <= last)] for p in prices]


def windows(clock: pd.DatetimeIndex, holdout_days: int, train_days: int, test_days: int,
            warmup_bars: int | None = None, minutes: int = 1440) -> tuple[tuple, int, date, tuple | None]:
    """The study's days on bars of `minutes` (daily, or 4-hour: each day `1440 // minutes` bars): (test windows as
    (first day, last day), warm-up bars before the first, the research end, the holdout as (first day, last day) or
    None). Walk-forward: run_study's test windows, each train_days after the last's start, the training bars warming
    the first; continuous (warmup_bars): one window from the bar after the warm-up to the research end."""
    per_day, bar = 1440 // minutes, pd.Timedelta(minutes=minutes)
    holdout, train, test_n = holdout_days * per_day, train_days * per_day, test_days * per_day
    research = clock[:-holdout] if holdout else clock
    if len(clock) < holdout + train + test_n:
        raise ValueError(f"{len(clock) // per_day:,} days the listed assets share is too short for holdout "
                         f"{holdout_days} + train {train_days} + test {test_days} days")
    held = (_day(clock[-holdout], bar), _day(clock[-1], bar)) if holdout else None
    if warmup_bars is not None:
        if warmup_bars >= len(research):
            raise ValueError(f"the warm-up of {warmup_bars:,} bars leaves nothing to score")
        return ((_day(research[warmup_bars], bar), _day(research[-1], bar)),), warmup_bars, \
            _day(research[-1], bar), held
    out, start = [], 0
    while start + train + test_n <= len(research):
        test = research[start + train: start + train + test_n]
        out.append((_day(test[0], bar), _day(test[-1], bar)))
        start += test_n
    return tuple(out), train, _day(research[-1], bar), held


def _minutes(history, venue: str, pair: str, first: date, last: date) -> pd.DataFrame:
    """The stored 1-minute bars closing on the days `first` to `last`, or a refusal naming the asset when they don't
    cover them."""
    lo, hi = pd.Timestamp(first, tz="UTC"), pd.Timestamp(last + timedelta(days=1), tz="UTC")
    minute = history.read(venue, pair, 1, start=lo, end=hi)
    if not len(minute) or minute.index[0] > lo + pd.Timedelta(hours=1) or minute.index[-1] != hi:
        raise ValueError(f"{pair}: the stored 1-minute bars don't cover {first} to {last}: no 1-minute run")
    return minute


def run_store_basket(req, store=None, out_dir: Path | None = None, history=None) -> Path:
    """The request's asset list as one basket study (research.basket.basket_study) on the venue's stored history;
    writes its sheet and returns its path. Needs the journal: the basket's try is counted in its trials register."""
    from sleeve_fund.fees import resolve as resolve_fees
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.instruments import history_price_decimals
    from sleeve_fund.research import pinned as pinning
    from sleeve_fund.research.holdout import HoldoutLocks
    from sleeve_fund.research.run import TEARSHEETS, spec_of, study_market
    from sleeve_fund.research.study import CONTINUOUS, continuous_warmup
    from sleeve_fund.research.trials import TrialsRegister
    from sleeve_fund.venues import venue as venue_profile

    if store is None:
        raise ValueError("a basket study needs the journal: its one try is counted in the trials register")
    spec = spec_of(req.strategy)
    profile = venue_profile(req.venue)
    history = history or HistoryStore()
    fees = resolve_fees(profile.name, store)
    bases = [p.split("/")[0] for p in req.pairs]
    daily, instruments = [], []
    for pair in req.pairs:
        try:
            prices = history.read(profile.name, pair, req.minutes)
        except KeyError:
            raise ValueError(f"no stored history for {pair} on this venue; ask for it under Stored history on the "
                             "Research page") from None
        if prices.empty:
            raise ValueError(f"the stored history for {pair} has no {'daily' if req.minutes == 1440 else '4-hour'} "
                             "bars yet")
        if req.pinned is not None:
            try:
                prices = pinning.cut(prices, req.pinned)
            except ValueError as exc:
                raise ValueError(f"{pair}: {exc}") from None
        floor = fees
        if profile.fee_bounds:  # FEE-LIVE: never below the published VIP 0, as the single study pays
            from sleeve_fund import venue_terms

            got = venue_terms.research_fees(store, profile, pair)
            floor = replace(fees, fees=got.fees, words=got.text)
        base, quote = pair.split("/")
        instruments.append(profile.instrument(base, quote, fees=floor.fees,
                                              price_precision=history_price_decimals(prices["close"])))
        daily.append(prices)
    daily = common_clock(daily, bases)
    params = {**(req.params or spec.default_params or {}), **req.exits(), **study_market(req)}  # every leg's market
    warmup = continuous_warmup(req.strategy, params, req.minutes) if req.score_window == CONTINUOUS else None
    if req.pinned is not None and req.holdout_days != req.pinned.holdout_days:
        raise ValueError(f"pinned dates: the holdout is {req.pinned.holdout_days} days, not {req.holdout_days}")
    tests, warm, research_end, holdout = windows(daily[0].index, req.holdout_days, req.train_days, req.test_days,
                                                 warmup, req.minutes)
    if req.pinned is not None and research_end != req.pinned.research_end:
        raise ValueError(f"pinned dates: the research period the listed assets share ends {research_end}, not the "
                         f"pinned {req.pinned.research_end}")
    assets = []
    for pair, instrument, prices in zip(req.pairs, instruments, daily):
        if req.minutes == 1440:
            hourly = history.read(profile.name, pair, 60, start=prices.index[0] - DAY, end=prices.index[-1])
            assets.append(bk.BasketAsset(instrument, prices, hourly))
            continue
        # A 4-hour list fills each asset on 1-minute bars on every day a run trades (the test days, and the holdout
        # when it is opened), or is refused naming the asset (PM 23 Oct: no quiet fall back to coarser fills). Only
        # those days are read: three assets' whole pinned windows of minutes went past the study container's
        # memory (27 Oct), and a run's warm-up reads its decision bars only.
        minute = _minutes(history, profile.name, pair, tests[0][0], holdout[1] if req.use_holdout else research_end)
        assets.append(bk.BasketAsset(instrument, prices, minute, bar_minutes=req.minutes, exec_minutes=1))
    label = req.label or spec.name
    universe = f"{datetime.now(timezone.utc):%Y-%m-%d}" if req.pinned is None else str(req.pinned.data_start)
    basket = bk.Basket(label, req.strategy, params, tuple(assets), universe)
    dataset = dataset_name(profile.name, req.pairs, req.minutes)
    # Advisor 27 Oct: the unguarded run is the try (counted once, as any basket); a size scale's run is never one
    study = bk.basket_study(basket, tests, research_end, register=TrialsRegister(store), dataset=dataset,
                            warm_days=warm, filled=True, profile=bk.PORTFOLIO if req.fund_guards else bk.UNGUARDED,
                            scale=req.size_scale, counted=req.size_scale is None)
    opened = None
    if req.use_holdout:
        opened = bk.open_holdout(study, holdout, HoldoutLocks(store), warm_days=warm)
    stem = f"{label}_{dataset}_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out = (out_dir or TEARSHEETS) / f"{stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while out.exists():  # two runs in the same second
        n += 1
        out = out.with_name(f"{stem}-{n}.md")
    out.write_text(bk.render(study, opened), encoding="utf-8")
    return out
