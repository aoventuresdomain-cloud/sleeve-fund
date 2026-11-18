"""SIGNAL-COUNT (HoE 18 Nov, for the Advisor's batch pre-checks): before a batch is dispatched, how often each row's
definition enters on each asset over the TRAIN window only, so the at-least-60-expected-trades and at-most-half-from-
one-asset pre-checks are read before any try is spent.

It never reads a bar on or after the first test day (nor the holdout): each asset's stored bars are cut to the pinned
data start to the day before the first fold's test start, and each asset runs alone on them, unguarded, at the venue's
published research fees and no spread. Nothing is journaled, registered or written: no try, no sheet.

The expected count over the test window is the train count pro-rated by days, counting only the days after the
definition's own warm-up (the engine decides nothing before its blocks have settled, so a long warm-up would otherwise
understate the rate). It is an estimate: a strategy whose entries depend on the regime can trade more or less often
after the train year. Prints counts, sizes and dates only, never a return or a fill price (public log)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

MIN_EXPECTED = 60  # the Advisor's pre-check: at least 60 expected trades over the test window, all assets together
MAX_ASSET_SHARE = 0.5  # and at most half of them from one asset
FIRST_SHOWN = 3  # the first entries printed with their times (the Advisor's spot check of when a rule fires)


@dataclass
class AssetCount:
    entries: list = field(default_factory=list)  # (decision close, side) of each entry
    sizes: list = field(default_factory=list)  # each entry's size against today's (vol_factor): 1.0 when not cut
    capped: int = 0  # entries the volatility target didn't cut: sized at the 1x cap or another limit
    rule_exits: int = 0  # exits by the definition's exit rule
    rule_exits_next_day: int = 0  # of those, on a later UTC day than their entry (a day-anchored block reset between)
    vol_skipped: int = 0  # entries skipped for want of the daily closes to read the volatility on


@dataclass(frozen=True)
class SignalCount:
    train_first: date  # the first train day read
    train_last: date  # the last, the day before the first test day
    decided_first: date | None  # the first day the definition could decide, after its warm-up; None: never
    test_days: int
    assets: dict  # pair -> AssetCount
    vol_target: dict | None = None  # the run's vol_target_size, when it sizes to one

    @property
    def train_days(self) -> int:
        return (self.train_last - self.train_first).days + 1

    @property
    def decided_days(self) -> int:
        return (self.train_last - self.decided_first).days + 1 if self.decided_first else 0

    @property
    def entries(self) -> dict:
        return {p: len(a.entries) for p, a in self.assets.items()}

    @property
    def total(self) -> int:
        return sum(self.entries.values())

    @property
    def expected(self) -> float:
        """The train count pro-rated to the test window's days, over the days the definition could decide on."""
        return self.total * self.test_days / self.decided_days if self.decided_days else 0.0

    def share(self, pair: str) -> float:
        return self.entries[pair] / self.total if self.total else 0.0

    @property
    def top(self) -> tuple[str, float]:
        """The asset with the most entries and its share of them (0 when there are none)."""
        pair = max(self.entries, key=self.entries.get)
        return pair, self.share(pair)

    def lines(self) -> list[str]:
        def name(pair):
            return pair.split("/")[0]

        per = ", ".join(f"{name(p)} {n:,}" for p, n in self.entries.items())
        shares = ", ".join(f"{name(p)} {self.share(p):.0%}" for p in self.assets)
        pair, share = self.top
        enough = self.expected >= MIN_EXPECTED
        spread = share <= MAX_ASSET_SHARE
        warm = (f"decided from {self.decided_first} after its warm-up, {self.decided_days} days"
                if self.decided_first else "its warm-up outlasts the train window: it never decided")
        out = [(f"train window {self.train_first} to {self.train_last} ({self.train_days} days), {warm}; "
                f"entries: {per}; {self.total:,} in all"),
               f"expected over the {self.test_days}-day test window, pro-rated: {self.expected:,.1f}",
               f"pre-check at least {MIN_EXPECTED} expected trades: {'PASS' if enough else 'FAIL'}",
               (f"pre-check at most {MAX_ASSET_SHARE:.0%} from one asset: {'PASS' if spread else 'FAIL'} "
                f"(top {name(pair)}; shares {shares})")]
        first = sorted((ts, name(p), side) for p, a in self.assets.items() for ts, side in a.entries)[:FIRST_SHOWN]
        if first:
            out.append("first entries: " + ", ".join(f"{ts:%Y-%m-%d %H:%M} UTC {n} {s}" for ts, n, s in first))
        rule = sum(a.rule_exits for a in self.assets.values())
        later = sum(a.rule_exits_next_day for a in self.assets.values())
        out.append(f"exit-rule exits on a later UTC day than their entry: {later:,} of {rule:,}"
                   + (f" ({later / rule:.0%})" if rule else ""))
        if self.vol_target:
            sizes = ", ".join(
                f"{name(p)} median {statistics.median(a.sizes):.2f}x, capped {a.capped / len(a.sizes):.0%}"
                if a.sizes else f"{name(p)} no entries" for p, a in self.assets.items())
            skipped = sum(a.vol_skipped for a in self.assets.values())
            out.append(f"sizing to {self.vol_target['annual']:.0%} a year over {self.vol_target['lookback_days']} days "
                       f"(size against the uncut size; capped = the target didn't cut it): {sizes}"
                       + (f"; {skipped:,} entries skipped for want of the daily closes" if skipped else ""))
        return out


def train_window(pinned) -> tuple[date, date]:
    """The first and last day the count reads: the pinned data start to the day before the earliest test start."""
    first_test = min(f.test_start for f in pinned.folds)
    return pinned.data_start, first_test - timedelta(days=1)


def days_tested(pinned) -> int:
    return sum((f.test_end - f.test_start).days + 1 for f in pinned.folds)


def train_bars(prices: pd.DataFrame, pinned) -> pd.DataFrame:
    """The stored bars whose candle falls in the train window, none on or after the first test day."""
    from sleeve_fund.research.pinned import _minutes, day_of

    if prices.empty:
        raise ValueError("there is no history to read")
    first, last = train_window(pinned)
    bar = pd.Timedelta(minutes=_minutes(prices))
    have = day_of(prices.index[0], bar), day_of(prices.index[-1], bar)
    if have[0] > first or have[1] < last:
        raise ValueError(f"the history covers {have[0]} to {have[1]}, not the train window {first} to {last}")
    days = pd.Index([day_of(ts, bar) for ts in prices.index])
    return prices[(days >= first) & (days <= last)]


def _decided_at(decision: dict) -> pd.Timestamp | None:
    """The decision candle's close: the latest candle the signal read."""
    ends = [pd.Timestamp(span[1]) for span in ((decision.get("signal") or {}).get("bars") or {}).values()]
    return max(ends) if ends else None


def _utc_day(ts: pd.Timestamp) -> date:
    return (ts - pd.Timedelta(1, "ns")).tz_convert("UTC").date()  # a candle closing at 00:00 ends the old day


def tally(decisions: dict, vol_on: bool, decided_first: date | None = None) -> AssetCount:
    """One asset's entries, sizes and exit-rule exits from a backtest's decisions, in the order they were made. Only
    entries on or after decided_first count (CR F1 on #368): a rule whose own blocks settled before the definition's
    longest warm-up can enter earlier, and the count is pro-rated over the days from decided_first."""
    from sleeve_fund.strategies.rules import VOL_SIZE_LIMIT

    out, entered = AssetCount(), None
    for d in decisions.values():
        sig = d.get("signal") or {}
        if d.get("intent") == "entry":
            entered = pd.Timestamp(sig["bar"]["close_ts"])
            if decided_first is not None and _utc_day(entered) < decided_first:
                entered = None  # before the counted days: neither it nor its exit is counted
                continue
            out.entries.append((entered, sig.get("side") or str(sig.get("rule", "long")).split(".")[0]))
            if vol_on:
                out.sizes.append(float(sig.get("vol_factor", 1.0)))
                out.capped += sig.get("sized_by") != VOL_SIZE_LIMIT
        elif d.get("intent") == "exit" and str(sig.get("rule", "")).endswith(".exit") and entered is not None:
            out.rule_exits += 1
            at = _decided_at(d)
            out.rule_exits_next_day += at is not None and _utc_day(at) != _utc_day(entered)
    return out


def count_signals(req, history=None, store=None, backtest=None) -> SignalCount:
    """Each asset of the request run alone over the train window; its entries counted."""
    from sleeve_fund.fees import resolve as resolve_fees
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.instruments import history_price_decimals
    from sleeve_fund.research.pinned import day_of
    from sleeve_fund.research.run import spec_of, study_market
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.venues import venue as venue_profile

    if req.pinned is None:
        raise ValueError("a signal count reads the list's pinned dates, and this list pins none")
    if backtest is None:
        from sleeve_fund.research.runner import run_backtest as backtest
    profile = venue_profile(req.venue)
    history = history or HistoryStore()
    params = {**(req.params or spec_of(req.strategy).default_params or {}), **req.exits(), **study_market(req)}
    vol = params.get("vol_target_size")
    settle = REGISTRY[req.strategy][0].warmup_needed(params, req.minutes)  # as the engine holds its decisions back
    first, last = train_window(req.pinned)
    runs, decided = {}, None
    for pair in req.pairs or [req.pair]:
        try:
            prices = train_bars(history.read(profile.name, pair, req.minutes), req.pinned)
        except KeyError:
            raise ValueError(f"no stored history for {pair} on this venue") from None
        except ValueError as exc:
            raise ValueError(f"{pair}: {exc}") from None
        if settle < len(prices):
            at = day_of(prices.index[settle], pd.Timedelta(minutes=req.minutes))
            decided = at if decided is None else max(decided, at)
        base, quote = pair.split("/")
        fees = resolve_fees(profile.name, store, instrument=pair)
        instrument = profile.instrument(base, quote, fees=fees.fees,
                                        price_precision=history_price_decimals(prices["close"]))
        runs[pair] = backtest(req.strategy, prices, instrument, params=params, bar_minutes=req.minutes, half_spread=0)
    assets = {}
    for pair, res in runs.items():  # counted from the one first day every asset could decide on
        assets[pair] = tally(res.decisions, vol is not None, decided) if decided else AssetCount()
        assets[pair].vol_skipped = int((getattr(res, "rule_counts", None) or {}).get("vol_skipped", 0))
    return SignalCount(train_first=first, train_last=last, decided_first=decided, test_days=days_tested(req.pinned),
                       assets=assets, vol_target=vol)
