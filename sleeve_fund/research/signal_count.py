"""SIGNAL-COUNT (HoE 18 Nov, for the Advisor's batch pre-checks): before a batch is dispatched, how often each row's
definition enters on each asset over the TRAIN window only, so the at-least-60-expected-trades and at-most-half-from-
one-asset pre-checks are read before any try is spent.

It never reads a bar on or after the first test day (nor the holdout): each asset's stored bars are cut to the pinned
data start to the day before the first fold's test start, and each asset runs alone on them, unguarded, at no fee,
charged the half-spread its studies charge (the measured series, the venue's assumption before its first measurement)
and its funding. An entry and its exit don't depend on the fee; its size against the uncut size does only through the
fee term in the leverage cap, within about 0.1% of the size (CR F1 on #372). The journal is read for the spreads only:
nothing is journaled, registered or written, no try, no sheet.

The expected count over the test window is the train count pro-rated by days, counting only the days after the
definition's own warm-up (the engine decides nothing before its blocks have settled, so a long warm-up would otherwise
understate the rate). It is an estimate: a strategy whose entries depend on the regime can trade more or less often
after the train year.

EDGE BEFORE FEES (HoE 18 Nov; the Advisor's rulings at 16:41 and 17:10): the same run's closed trades, each one's P&L
at no fee, after the half-spread and funding (the sheets' "before fees" line), over its entry notional in bps, per
asset and pooled, from the trades decided on or after the definition's first decided day (a trade still open at the
train window's end is left out: the engine closes nothing at the end of a run). Two pre-checks read them: the pooled
mean at least 25 bps a trade, and a positive mean on at least 2 assets. Prints counts, sizes, dates and the checks' verdicts
only, never a return, an edge figure or a fill price (public log): the figures stay in the result (gross_bps,
pooled_bps) for a private channel."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import UTC, date, timedelta

import pandas as pd

MIN_EXPECTED = 60  # the Advisor's pre-check: at least 60 expected trades over the test window, all assets together
MAX_ASSET_SHARE = 0.5  # and at most half of them from one asset
FIRST_SHOWN = 3  # the first entries printed with their times (the Advisor's spot check of when a rule fires)
MIN_GROSS_BPS = 25.0  # the Advisor's pre-check (17:10): at least 25 bps a trade before fees, pooled
MIN_POSITIVE_ASSETS = 2  # and a positive gross edge on at least 2 of the assets


@dataclass
class AssetCount:
    entries: list = field(default_factory=list)  # (decision close, side) of each entry
    sizes: list = field(default_factory=list)  # each entry's size against today's (vol_factor): 1.0 when not cut
    capped: int = 0  # entries the volatility target didn't cut: sized at the 1x cap or another limit
    rule_exits: int = 0  # exits by the definition's exit rule
    rule_exits_next_day: int = 0  # of those, on a later UTC day than their entry (a day-anchored block reset between)
    vol_skipped: int = 0  # entries skipped for want of the daily closes to read the volatility on
    gross_bps: list = field(default_factory=list)  # each closed trade's P&L before fees, bps of its entry notional
    spread: str = ""  # "measured" or "assumed": the half-spread its run was charged


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

    def gross_bps(self, pair: str) -> float | None:
        """The asset's mean gross edge a closed trade, in bps of its entry notional; None with no closed trade."""
        xs = self.assets[pair].gross_bps
        return sum(xs) / len(xs) if xs else None

    @property
    def gross_trades(self) -> int:
        return sum(len(a.gross_bps) for a in self.assets.values())

    @property
    def pooled_bps(self) -> float | None:
        """Every asset's closed trades together: their mean gross edge, in bps of entry notional."""
        xs = [x for a in self.assets.values() for x in a.gross_bps]
        return sum(xs) / len(xs) if xs else None

    @property
    def positive_assets(self) -> list[str]:
        return [p for p in self.assets if (self.gross_bps(p) or 0.0) > 0]

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
        trades = ", ".join(f"{name(p)} {len(a.gross_bps):,}" for p, a in self.assets.items())
        spreads = ", ".join(f"{name(p)} {a.spread or 'none'}" for p, a in self.assets.items())
        pooled = self.pooled_bps
        positive = self.positive_assets
        out += [(f"edge before fees: closed trades decided from the decided day, those open at the train window's end "
                 f"left out; each one's P&L at 0 fees, after the half-spread the studies charge ({spreads}) and "
                 f"funding, over its entry notional (figures not printed: public log): {trades}; "
                 f"{self.gross_trades:,} in all"),
                (f"pre-check pooled edge before fees at least {MIN_GROSS_BPS:g} bps a trade: "
                 f"{'PASS' if pooled is not None and pooled >= MIN_GROSS_BPS else 'FAIL'}"
                 + ("" if pooled is not None else " (no closed trade)")),
                (f"pre-check edge before fees positive on at least {MIN_POSITIVE_ASSETS} of {len(self.assets)} assets: "
                 f"{'PASS' if len(positive) >= MIN_POSITIVE_ASSETS else 'FAIL'} "
                 f"(positive: {', '.join(name(p) for p in positive) or 'none'})")]
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


def gross_bps(fills, decided_first: date | None, decisions: dict | None = None,
              funding: list | None = None) -> list[float]:
    """Each closed trade's P&L over its entry notional in bps, for the trades decided on or after decided_first. The
    run is at no fee and its fills carry the half-spread; its funding payments are booked to the trades here (the
    fills carry none), so this is the edge before fees. A trade's day is its entry decision's close, as tally reads
    an entry's (CR F2 on #372), else its first fill's day."""
    from sleeve_fund.research.metrics import fills_to_rows, trades

    if fills is None or len(fills) == 0 or decided_first is None:
        return []
    out = []
    for t in trades(fills_to_rows(fills), shorts=True, funding=funding or None):
        bar = (((decisions or {}).get(str(t.get("entry_order"))) or {}).get("signal") or {}).get("bar") or {}
        if bar.get("close_ts") is not None:
            day = _utc_day(pd.Timestamp(bar["close_ts"]))
        else:
            opened = pd.Timestamp(t["opened"])
            day = (opened.tz_localize("UTC") if opened.tzinfo is None else opened).tz_convert("UTC").date()
        if t["closed"] is not None and t["cost"] and day >= decided_first:
            out.append(t["pnl"] / t["cost"] * 1e4)
    return out


def spread_words(spread, first: date, last: date) -> str:
    """Which half-spread the train window was charged: "measured", "assumed" (the venue's assumption, every
    measurement after the window or none) or "assumed, then measured" (the first measurement inside it)."""
    from datetime import datetime

    def ns(d: date) -> int:
        return int(pd.Timestamp(datetime(d.year, d.month, d.day, tzinfo=UTC)).value)

    froms = [at for at, _ in spread.points]
    if not froms or froms[0] >= ns(last + timedelta(days=1)):
        return "assumed"
    return "measured" if froms[0] <= ns(first) else "assumed, then measured"


def count_signals(req, history=None, backtest=None, store=None) -> SignalCount:
    """Each asset of the request run alone over the train window, at no fee and the half-spread its studies charge;
    its entries and its trades' edge before fees counted. store: the journal, read for the measured spreads only
    (None: the venue's assumption)."""
    from decimal import Decimal

    from sleeve_fund import spreads
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.instruments import FeeSchedule, history_price_decimals
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
    runs, decided, charged = {}, None, {}
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
        instrument = profile.instrument(base, quote, fees=FeeSchedule(maker=Decimal(0), taker=Decimal(0)),
                                        price_precision=history_price_decimals(prices["close"]))
        spread = spreads.series(profile.name, pair, store)  # as run_store_study charges it
        charged[pair] = spread_words(spread, first, last)
        runs[pair] = backtest(req.strategy, prices, instrument, params=params, bar_minutes=req.minutes,
                              half_spread=spread)
    assets = {}
    for pair, res in runs.items():  # counted from the one first day every asset could decide on
        assets[pair] = tally(res.decisions, vol is not None, decided) if decided else AssetCount()
        assets[pair].vol_skipped = int((getattr(res, "rule_counts", None) or {}).get("vol_skipped", 0))
        assets[pair].gross_bps = gross_bps(getattr(res, "fills", None), decided, res.decisions,
                                           getattr(res, "funding", None))
        assets[pair].spread = charged[pair]
    return SignalCount(train_first=first, train_last=last, decided_first=decided, test_days=days_tested(req.pinned),
                       assets=assets, vol_target=vol)
