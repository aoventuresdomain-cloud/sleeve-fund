"""The interim open-risk limit for the hand-coded models on a perpetual (Independent Quant Advisor, 6 Oct; phase 2's
portfolio limits, P2-2, absorb it). A new entry is refused when the book's open risk, counting the entry, would pass
5% of the book.

- A position with a stop risks what it would lose if the stop filled now: |mark - stop| x qty, from the current mark,
  not its entry. One the price has already gone through, still open, counts as stopless (and alerts).
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
ATR_WINDOW = ATR_DAYS * 3  # whole days the daily ATR is measured over, in paper and backtest alike


def position_risk(qty: float, mark: float, stop: float | None = None, atr_pct: float | None = None) -> float:
    """What a position (signed qty) risks at `mark`: to its stop price when it has one, else at the stopless move."""
    if not qty or mark <= 0:
        return 0.0
    if stop is not None and (mark - stop) * qty > 0:
        return (mark - stop) * qty  # a short (qty < 0) loses as the price rises to its stop
    # No stop, or one the price has gone through with the position still open (its exit not filled yet): it counts
    # at the stopless measure, never 0 (Independent Quant Advisor, QA P1-S9).
    if atr_pct is None or not math.isfinite(atr_pct):
        raise ValueError("the daily ATR isn't known, so a position with no stop can't be measured")
    return abs(qty) * mark * stopless_move(atr_pct)


def stopless_move(atr_pct: float) -> float:
    return max(STOPLESS_FLOOR, STOPLESS_ATRS * atr_pct)


def daily_atr_pct(daily: pd.DataFrame) -> float | None:
    """The Wilder ATR(14) of the last ATR_WINDOW whole daily bars (high, low, close), as a share of the last close: the
    indicator library's Atr (QA P1-S5), over the same window in paper and backtest. None with under 14 days."""
    from sleeve_fund.strategies.indicators.averages import Atr

    days = daily.dropna(subset=["high", "low", "close"]).tail(ATR_WINDOW)
    if len(days) < ATR_DAYS or float(days["close"].iloc[-1]) <= 0:
        return None
    atr = Atr(ATR_DAYS)
    for h, lo, c in days[["high", "low", "close"]].itertuples(index=False):
        atr.update_raw(h, lo, c)
    # The library calls it settled after ten lengths (for a signal); the risk measure reads the Wilder average from
    # day 14 on, over the same window in paper and backtest, so both see the same number.
    return atr._outputs()["value"] / float(days["close"].iloc[-1])


def daily_atr_lookup(prices: pd.DataFrame) -> dict[int, float]:
    """Backtest: from bars stamped at their close, the daily ATR share known at each UTC day's start (the whole days
    before it, as paper reads them), as {day start in ns: share}."""
    if prices is None or not len(prices) or not {"high", "low", "close"} <= set(prices.columns):
        return {}
    # A bar stamped at its close belongs to the day it closes in, less a nanosecond: the bar closing at 00:00 is the day before's.
    days = prices[["high", "low", "close"]].groupby((prices.index - pd.Timedelta(1, "ns")).floor("D")).agg(
        {"high": "max", "low": "min", "close": "last"})
    out = {}
    for i in range(ATR_DAYS - 1, len(days)):
        v = daily_atr_pct(days.iloc[max(0, i + 1 - ATR_WINDOW):i + 1])
        if v is not None:
            out[int((days.index[i] + pd.Timedelta(days=1)).value)] = v
    return out


def check_entry(book: float, open_risk: float, entry_risk: float) -> str | None:
    """Why an entry is refused, or None when the book's open risk with it stays within the limit."""
    total = open_risk + entry_risk
    if book <= 0 or total > LIMIT * book:
        return (f"open risk would be {total:,.2f} with this entry ({entry_risk:,.2f}), over {LIMIT:.0%} of the book "
                f"({LIMIT * max(book, 0.0):,.2f})")
    return None


_ATR_CACHE: dict[tuple[str, str, str], float] = {}


def history_atr_pct(venue: str, pair: str, now: datetime, history=None) -> float | None:
    """Paper: the daily ATR share for a venue's pair from the history store, as of the last whole day, cached per day
    once read. None when the store can't give 14 whole days; that isn't cached, so the next entry asks again (QA
    P1-S3)."""
    key = (venue, pair, now.strftime("%Y-%m-%d"))
    if key in _ATR_CACHE:
        return _ATR_CACHE[key]
    from sleeve_fund.history import HistoryStore

    try:
        daily = (history or HistoryStore()).read(venue, pair, 1440, start=now - timedelta(days=ATR_WINDOW + 2))
        value = daily_atr_pct(daily[daily.index <= pd.Timestamp(now)])
    except (KeyError, OSError, ValueError):
        return None
    if value is not None:
        if len(_ATR_CACHE) > 256:
            _ATR_CACHE.pop(next(iter(_ATR_CACHE)))
        _ATR_CACHE[key] = value
    return value


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


def gapped(qty: float, mark: float, stop: float | None) -> bool:
    """A position whose stop the price has already gone through while it is still open."""
    return bool(qty) and stop is not None and (mark - stop) * qty <= 0


def book_open_risk(store, atr_pct, account: str | None = None) -> list[dict]:
    """Read-only, for the dashboard's Open risk (FE v2, signature agreed with PE2 7 Oct): what the limit counts for
    each open perp position, by the same rules as account_book, so the tile equals the gate. One row per strategy
    that isn't archived, is a perpetual and holds a position, in store.sleeves() order, on `account` (None: every
    account): {sleeve, risk, basis, atr_pct}. basis is "stop" (measured to its journaled stop), "stopless" (no
    placed stop, a close-checked trail included) or "gapped" (the price has gone through its stop, still open).
    `atr_pct(sleeve)` is asked only for stopless and gapped rows; risk is None when it gives nothing. Never raises."""
    from sleeve_fund import markets

    archived = store.archived()
    rows = []
    for s in store.sleeves():
        if s.name in archived or (account is not None and store.account_of(s.name) != account):
            continue
        last = store.last_equity(s.name)
        if last is None or not last["qty"] or not markets.is_perp(s.params):
            continue
        qty, mark = last["qty"], last["price"]
        stop = _journal_stop(store, s, qty)
        basis = "gapped" if gapped(qty, mark, stop) else ("stop" if stop is not None else "stopless")
        atr = None
        if basis != "stop":
            atr = atr_pct(s)
            if atr is not None and not math.isfinite(atr):
                atr = None
        try:
            risk = position_risk(qty, mark, stop if basis == "stop" else None, atr)
        except ValueError:  # stopless or gapped with no daily ATR: can't be measured
            risk = None
        rows.append({"sleeve": s.name, "risk": risk, "basis": basis, "atr_pct": atr})
    return rows


def account_book(store, sleeve_name: str, own_equity: float, atr_pct) -> tuple[float, float, list[str]]:
    """Paper: (the book, the open risk of the other perp strategies on the account, those of them whose stop the
    price has gone through while still open). `atr_pct(sleeve)` gives a strategy's daily ATR share. Raises
    ValueError when a stopless position can't be measured."""
    from sleeve_fund import markets

    account = store.account_of(sleeve_name)
    archived = store.archived()
    book, risk, through = own_equity, 0.0, []
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
        if gapped(last["qty"], last["price"], stop):
            through.append(s.name)
        measured = stop is not None and s.name not in through
        risk += position_risk(last["qty"], last["price"], stop, None if measured else atr_pct(s))
    return book, risk, through
