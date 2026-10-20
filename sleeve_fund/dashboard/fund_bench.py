"""The fund's benchmark (Advisor B34-BENCHMARK, 20 Oct): one buy-and-hold of the benchmark asset on the fund's capital,
counted once on the book's line and never written or summed per Strategy.

Its value is E0 x (1 - fee) x P / P0, E0 the fund's marked equity at the period's start (book_marks), with one taker
entry fee, so its first point is above 0 whenever E0 is. A deposit buys units at the flow's time, net of the same fee,
and a withdrawal sells units at that price, so a flow is never a return on either side. On a perpetual the 1x long
pays or receives each settled funding rate on what it holds; perpetual closes without funding are never used. Worked
out on read from what the fund already writes (book_marks, fund_flows) and the hub's stored closes: no table of its
own, so nothing to keep in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

BENCH_PAIR = "BTC/USDT"


@dataclass(frozen=True)
class FundBench:
    e0: float  # the fund's marked equity at the period's start: the return base
    values: pd.Series  # the hold's value at each point, oldest first
    source: str  # the footnote: which closes, and whether funding is in
    fee: float
    as_of: pd.Timestamp  # the last point's time: the latest stored close, not the last daily close


def hold_values(prices: pd.Series, e0: float, fee: float, t0, flows=(), funding: pd.Series | None = None,
                points=None) -> pd.Series:
    """The hold's value at each of `points` (default: every price after t0). `prices` are closes indexed by UTC time;
    P0 is the last at or before t0. `flows`: (time, amount) with withdrawals negative. `funding`: settled rates by
    settlement time, charged to the long as units x price x rate (a positive rate pays)."""
    prices = prices.sort_index()
    upto = prices[prices.index <= t0]
    if not len(upto):
        raise ValueError(f"no benchmark price at or before {t0}")
    p0 = float(upto.iloc[-1])
    units, cash = e0 * (1 - fee) / p0, 0.0

    def price_at(t) -> float:
        known = prices[prices.index <= t]
        return float(known.iloc[-1]) if len(known) else p0

    events = [(pd.Timestamp(t), 0, float(a)) for t, a in flows if pd.Timestamp(t) > t0]
    if funding is not None:
        events += [(t, 1, float(r)) for t, r in funding.items() if t > t0]
    events.sort(key=lambda e: (e[0], e[1]))
    points = prices.index[prices.index > t0] if points is None else pd.DatetimeIndex(points)
    out, i = [], 0
    for t in points:
        while i < len(events) and events[i][0] <= t:
            at, kind, x = events[i]
            p = price_at(at)
            if kind == 0:
                units += x * (1 - fee) / p if x > 0 else x / p
            else:
                cash -= units * p * x
            i += 1
        out.append(units * price_at(t) + cash)
    return pd.Series(out, index=points, dtype=float)


def chained_return(values: pd.Series, base: float, flows=()) -> float | None:
    """The return over `values` from `base`, its value just before the first point, with each flow taken out on the
    step it lands in (DAY-RET's rule): the product of (V - F) / V_prev. Without flows it is V_last / base - 1. None
    when the base, or any step's, is 0 or below: nothing to measure from."""
    if base <= 0:
        return None
    prev_t, prev, growth = None, base, 1.0
    flows = [(pd.Timestamp(t), float(a)) for t, a in flows]
    for t, v in values.items():
        f = sum(a for ft, a in flows if (prev_t is None or ft > prev_t) and ft <= t)
        if prev <= 0:
            return None
        growth *= (float(v) - f) / prev
        prev_t, prev = t, float(v)
    return growth - 1


def fund_benchmark(store, start: datetime | None = None, end: datetime | None = None, hist=None,
                   now: datetime | None = None, fee: float | None = None) -> FundBench | None:
    """The fund's benchmark over (start, end], from the bridge on (supervisor.bridge_start); None before the bridge
    (each Strategy carries its own capital and its own benchmark then) or with no mark or no stored closes. Points: the
    daily closes, plus the latest stored close as the last, so the line is never up to a day behind."""
    from sleeve_fund import funding as fundingm
    from sleeve_fund.fees import resolve
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.supervisor import bridge_start
    from sleeve_fund.venues import VENUES

    bridge = bridge_start(store)
    if bridge is None:
        return None
    t0 = pd.Timestamp(max(start, bridge) if start is not None else bridge)
    e0 = store.book_equity_at_or_before(t0.to_pydatetime())
    if e0 is None:  # the window opens before the first mark: from the first one
        first = store.first_book_mark()
        if first is None:
            return None
        t0, e0 = pd.Timestamp(first["ts"]), first["equity"]
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0
    hist = hist or HistoryStore()
    venue = next((v for v in VENUES.values() if v.hub), None)
    if venue is None or hist.coverage(venue.name, BENCH_PAIR) is None:
        return None
    minutes = hist.read(venue.name, BENCH_PAIR, minutes=1, start=t0 - pd.Timedelta(days=2), end=end)["close"]
    if not len(minutes) or minutes.index[0] > t0:
        return None
    perp = bool(venue.perpetual)
    rates = fundingm.rates(venue.name, BENCH_PAIR, hist.root) if perp else None
    # the account's taker fee for the instrument, where a Strategy's start reads it (paper.node: fees.resolve)
    fee = float(resolve(venue.name, store, BENCH_PAIR, now=now).fees.taker) if fee is None else fee
    flows = [(pd.Timestamp(f["applied_at"]), float(-f["amount"] if f["kind"] == "withdraw" else f["amount"]))
             for f in store.fund_flows() if f["applied_at"] is not None and pd.Timestamp(f["applied_at"]) > t0
             and (end is None or pd.Timestamp(f["applied_at"]) <= pd.Timestamp(end))]
    daily = minutes[minutes.index > t0].resample("1D").last().dropna()
    points = sorted(set(daily.index.map(lambda d: minutes.index[minutes.index <= d + pd.Timedelta(days=1)][-1]))
                    | {minutes.index[-1]})
    points = [p for p in points if p > t0]
    values = hold_values(minutes, float(e0), fee, t0, flows, rates, points)
    source = (f"{BENCH_PAIR} perpetual closes with settled funding on the 1x long" if perp
              else f"{BENCH_PAIR} spot closes")
    return FundBench(float(e0), values, source, fee, minutes.index[-1])
