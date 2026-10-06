"""The interim open-risk limit for the hand-coded models on a perpetual (Independent Quant Advisor, 6 Oct; phase 2's
portfolio limits, P2-2, absorb it). A new entry is refused when the book's open risk, counting the entry, would pass
5% of the book.

- A position with a stop risks what it would lose if the stop filled now: |mark - stop| x qty, from the current mark,
  not its entry. A stop already in profit risks nothing.
- A position with no placed stop counts at notional x max(10%, 3 x the daily Wilder ATR(14) as a share of price). Full
  margin would assume a 100% move at 1x; the 1x cap on such a position (strategies.check_perp_stop) bounds the tail.
- The book is the current equity of every paper strategy on the same account that isn't archived, read from the
  journal, and its open risk is theirs plus the entry's.

Paper gates on it. A single-strategy backtest doesn't: its book would be the strategy's own equity, so research results
would depend on how the book is made up. It counts how often the limit would have bound instead (BacktestResult).
Spot positions aren't counted: the limit is for perpetuals."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd

LIMIT = 0.05  # of the book
STOPLESS_FLOOR = 0.10  # the smallest move a position with no placed stop counts at
STOPLESS_ATRS = 3.0  # daily ATRs a position with no placed stop counts at, when that is more
ATR_DAYS = 14


def position_risk(qty: float, mark: float, stop: float | None = None, atr_pct: float | None = None) -> float:
    """What a position (signed qty) risks at `mark`: to its stop price when it has one, else at the stopless move."""
    if not qty or mark <= 0:
        return 0.0
    if stop is not None:
        return max(0.0, (mark - stop) * qty)  # a short (qty < 0) loses as the price rises to its stop
    if atr_pct is None or not math.isfinite(atr_pct):
        raise ValueError("the daily ATR isn't known, so a position with no stop can't be measured")
    return abs(qty) * mark * stopless_move(atr_pct)


def stopless_move(atr_pct: float) -> float:
    return max(STOPLESS_FLOOR, STOPLESS_ATRS * atr_pct)


def wilder_atr_pct(daily: pd.DataFrame) -> pd.Series:
    """The Wilder ATR(14) of daily bars (high, low, close) as a share of each day's close, known from that day's
    close on: the first 14 true ranges averaged, then (previous x 13 + true range) / 14. NaN before day 14."""
    high, low, close = (daily[c].astype(float).to_numpy() for c in ("high", "low", "close"))
    out = [math.nan] * len(close)
    atr = None
    trs = []
    for i in range(len(close)):
        tr = high[i] - low[i] if i == 0 else max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        if atr is None:
            trs.append(tr)
            if len(trs) == ATR_DAYS:
                atr = sum(trs) / ATR_DAYS
        else:
            atr = (atr * (ATR_DAYS - 1) + tr) / ATR_DAYS
        if atr is not None and close[i] > 0:
            out[i] = atr / close[i]
    return pd.Series(out, index=daily.index, dtype=float)


def daily_atr_lookup(prices: pd.DataFrame) -> dict[int, float]:
    """Backtest: from bars stamped at their close, the daily ATR share known at each UTC day's start, as
    {day start in ns: share}. Only whole days the bars have closed count."""
    if prices is None or not len(prices) or not {"high", "low", "close"} <= set(prices.columns):
        return {}
    # A bar stamped at its close belongs to the day it closes in, less a nanosecond: the bar closing at 00:00 is the day before's.
    days = prices[["high", "low", "close"]].groupby((prices.index - pd.Timedelta(1, "ns")).floor("D")).agg(
        {"high": "max", "low": "min", "close": "last"})
    atr = wilder_atr_pct(days).dropna()
    return {int((day + pd.Timedelta(days=1)).value): float(v) for day, v in atr.items()}


def check_entry(book: float, open_risk: float, entry_risk: float) -> str | None:
    """Why an entry is refused, or None when the book's open risk with it stays within the limit."""
    total = open_risk + entry_risk
    if book <= 0 or total > LIMIT * book:
        return (f"open risk would be {total:,.2f} with this entry ({entry_risk:,.2f}), over {LIMIT:.0%} of the book "
                f"({LIMIT * max(book, 0.0):,.2f})")
    return None


_ATR_CACHE: dict[tuple[str, str, str], float | None] = {}


def history_atr_pct(venue: str, pair: str, now: datetime, history=None) -> float | None:
    """Paper: the daily ATR share for a venue's pair from the history store, as of the last whole day, cached per day.
    None when the store can't give 14 whole days."""
    key = (venue, pair, now.strftime("%Y-%m-%d"))
    if key not in _ATR_CACHE:
        from sleeve_fund.history import HistoryStore

        try:
            daily = (history or HistoryStore()).read(venue, pair, 1440, start=now - timedelta(days=ATR_DAYS * 3))
            series = wilder_atr_pct(daily[daily.index <= pd.Timestamp(now)].dropna(subset=["close"])).dropna()
            _ATR_CACHE[key] = float(series.iloc[-1]) if len(series) else None
        except (KeyError, OSError, ValueError):
            _ATR_CACHE[key] = None
        if len(_ATR_CACHE) > 256:
            _ATR_CACHE.pop(next(iter(_ATR_CACHE)))
    return _ATR_CACHE[key]


def _journal_stop(store, sleeve, qty: float) -> float | None:
    """The stop price working for a strategy's open position, from the plan its latest entry journaled."""
    book = store.journal_book(sleeve.name, sleeve.starting_balance)
    entry_px = book["entry_px"]
    if not entry_px:
        return None
    side = 1 if qty > 0 else -1
    entry = next((o for o in store.orders(sleeve.name, limit=200)
                  if o.get("intent") == "entry" and o.get("side") == ("BUY" if side > 0 else "SELL")), None)
    if entry is None:
        return None
    plan = store.exit_plan(sleeve.name, entry["order_id"])
    frac = plan["stop_frac"] if plan is not None else (entry.get("signal") or {}).get("stop_frac")
    return None if frac is None else entry_px * (1 - side * frac)


def account_book(store, sleeve_name: str, own_equity: float, atr_pct) -> tuple[float, float]:
    """Paper: (the book, the open risk of the other perp strategies on the account). `atr_pct(sleeve)` gives a
    strategy's daily ATR share. Raises ValueError when a stopless position can't be measured."""
    from sleeve_fund import markets

    account = store.account_of(sleeve_name)
    archived = store.archived()
    book, risk = own_equity, 0.0
    for s in store.sleeves():
        if s.name == sleeve_name or s.name in archived or store.account_of(s.name) != account:
            continue
        last = store.last_equity(s.name)
        if last is None:
            book += s.starting_balance
            continue
        book += last["equity"]
        if not last["qty"] or not markets.is_perp(s.params):
            continue
        stop = _journal_stop(store, s, last["qty"])
        risk += position_risk(last["qty"], last["price"], stop, None if stop is not None else atr_pct(s))
    return book, risk
