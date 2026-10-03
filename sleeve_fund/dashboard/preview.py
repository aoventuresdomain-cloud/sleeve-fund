"""Quick look-back for the new-sleeve form: how would these settings have traded recently?

Runs the real strategy through the same backtest engine and fees as research, on
the venue's daily candles (from its venue profile). It is in-sample and unvalidated, so the
form says so; G1 is still decided by the research loop.
"""

from __future__ import annotations

import threading
import time

import pandas as pd

from sleeve_fund.research.metrics import fills_to_rows, returns_from_equity, summary, trade_stats, trades
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.venues import venue as venue_profile

CACHE_SECONDS = 6 * 3600  # daily candles: refetching more often adds nothing
_history: dict[tuple[str, str], tuple[float, object]] = {}
_lock = threading.Lock()


def history(pair: str, fetch=None, venue: str | None = None):
    """Daily candles for a pair: the full stored history when the history store has it, otherwise
    the venue's recent candles. Cached so the form doesn't reload on every change."""
    from sleeve_fund.history import HistoryStore

    profile = venue_profile(venue)
    key = (profile.name, pair)
    with _lock:
        hit = _history.get(key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
    df = None
    if fetch is None:
        store = HistoryStore()
        cov = store.coverage(profile.name, pair)
        # Only a series that has caught up to now: a backfill still in 2017 would hide recent years.
        if cov is not None and pd.Timestamp.now(tz="UTC") - cov.last < pd.Timedelta("2D"):
            df = store.read(profile.name, pair, 1440)
    if df is None:
        df = (fetch or profile.daily_history)(pair)
    with _lock:
        _history[key] = (time.time(), df)
    return df


def _finite(v):
    """JSON has no NaN or infinity; the form shows None as n/a."""
    return None if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))) else v


def _precision(price: float) -> int:
    return 2 if price >= 100 else 4 if price >= 1 else 6


def benchmark(prices: pd.DataFrame, starting: float, taker_fee: float, cap: float = 1.0) -> pd.Series:
    """Buy and hold at the same exposure the sleeve is allowed: `cap` of capital bought on day one
    (after the taker fee), the rest left in cash. Comparing a 33%-capped strategy with a fully
    invested benchmark would credit the strategy for simply holding less."""
    return starting * (1 - cap) + starting * cap * (1 - taker_fee) * prices["close"] / prices["close"].iloc[0]


def run(strategy: str, pair: str, params: dict, starting: float = 10_000.0, fetch=None, days: int | None = None,
        detail: bool = False, cap: float | None = None, venue: str | None = None, fee_quote=None) -> dict:
    """Backtest these settings on the venue's daily history. days trims to the most recent N days;
    detail adds every trade with its journaled reason, drawdown and fill markers (the backtest page).
    cap is the risk profile's largest position as a share of equity, applied exactly as paper does,
    and the buy-and-hold benchmark is held at the same exposure."""
    from sleeve_fund.fees import resolve

    profile = venue_profile(venue)
    quote_fees = fee_quote or resolve(profile.name)
    prices = history(pair, fetch, profile.name)
    if days:
        prices = prices.iloc[-days:]
    if len(prices) < 60:
        raise ValueError(f"only {len(prices)} days of {profile.label} history for {pair}; need at least 60")
    base, quote = pair.split("/")
    inst = profile.instrument(base, quote, price_precision=_precision(float(prices["close"].median())),
                              fees=quote_fees.fees)
    if cap is not None:
        params = {**params, "position_cap_pct": cap}
    res = run_backtest(strategy, prices, inst, params=params, starting_capital=starting)
    bench = benchmark(prices, starting, float(inst.taker_fee), cap if cap is not None else 1.0)
    s, b = summary(returns_from_equity(res.equity)), summary(returns_from_equity(bench))
    rows = fills_to_rows(res.fills)
    trips = trades(rows)
    stats = trade_stats(trips)
    step = 1 if detail else max(1, len(res.equity) // 400)
    out = {
        "pair": pair,
        "from": prices.index[0].strftime("%d %b %Y"),
        "to": prices.index[-1].strftime("%d %b %Y"),
        "days": len(prices),
        "t": [t.isoformat() for t in res.equity.index[::step]],
        "equity": [round(v, 2) for v in res.equity.iloc[::step]],
        "benchmark": [round(v, 2) for v in bench.iloc[::step]],
        "strategy": {k: s[k] for k in ("total_return", "cagr", "sharpe", "max_drawdown", "volatility")},
        "hold": {k: b[k] for k in ("total_return", "cagr", "sharpe", "max_drawdown", "volatility")},
        "trades": {k: _finite(stats[k]) for k in ("trades", "win_rate", "expectancy", "profit_factor")},
        "fees": round(res.fees_paid, 2),
        "exposure": round(float(res.exposure.mean()), 4),
        "cap": cap,
        "fee_schedule": {"maker": float(inst.maker_fee), "taker": float(inst.taker_fee), "text": quote_fees.text,
                         "source": quote_fees.source},
    }
    if detail:
        from sleeve_fund.dashboard import trading

        peak = res.equity.cummax()
        out["drawdown"] = [round(float(v), 6) for v in (1 - res.equity / peak)]
        out["fills"] = [{"t": r["ts"].isoformat(), "side": r["side"], "price": r["price"]} for r in rows]
        out["trips"] = trading.trips(list(reversed(rows)), [], res.decisions)
        out["stats"] = {k: _finite(v) for k, v in stats.items()}
        out["strategy"].update(sortino=s["sortino"], calmar=s["calmar"])
        out["hold"].update(sortino=b["sortino"], calmar=b["calmar"])
        out["start"] = round(float(starting), 2)
        from sleeve_fund.dashboard import charts

        # Daily candles are stamped at their close; the chart wants open times. Fills land on the
        # bar they decided on, which closed at the fill time.
        opened = prices.set_axis(prices.index - pd.Timedelta("1D"))
        out["price"] = charts.payload(opened, 1440, rows, res.decisions, [], "venue", shift_bars=1)
    return out
