"""Quick look-back for the new-sleeve form: how would these settings have traded recently?

Runs the real strategy through the same backtest engine and fees as research, on
Kraken's last ~2 years of daily candles. It is in-sample and unvalidated, so the
form says so; G1 is still decided by the research loop.
"""

from __future__ import annotations

import threading
import time

import pandas as pd

from sleeve_fund.data import fetch_kraken_daily
from sleeve_fund.instruments import spot_pair
from sleeve_fund.research.metrics import fills_to_rows, returns_from_equity, summary, trade_stats, trades
from sleeve_fund.research.runner import run_backtest

CACHE_SECONDS = 6 * 3600  # daily candles: refetching more often adds nothing
_history: dict[str, tuple[float, object]] = {}
_lock = threading.Lock()


def history(pair: str, fetch=None):
    """Daily candles for a pair, cached so the form doesn't call Kraken on every change."""
    with _lock:
        hit = _history.get(pair)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
    df = (fetch or fetch_kraken_daily)(pair)
    with _lock:
        _history[pair] = (time.time(), df)
    return df


def _finite(v):
    """JSON has no NaN or infinity; the form shows None as n/a."""
    return None if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))) else v


def _precision(price: float) -> int:
    return 2 if price >= 100 else 4 if price >= 1 else 6


def run(strategy: str, pair: str, params: dict, starting: float = 10_000.0, fetch=None, days: int | None = None,
        detail: bool = False) -> dict:
    """Backtest these settings on Kraken daily history. days trims to the most recent N days;
    detail adds every trade with its journaled reason, drawdown and fill markers (the backtest page)."""
    prices = history(pair, fetch)
    if days:
        prices = prices.iloc[-days:]
    if len(prices) < 60:
        raise ValueError(f"only {len(prices)} days of Kraken history for {pair}; need at least 60")
    base, quote = pair.split("/")
    inst = spot_pair(base, quote, price_precision=_precision(float(prices["close"].median())))
    res = run_backtest(strategy, prices, inst, params=params, starting_capital=starting)
    bench = starting * (1 - float(inst.taker_fee)) * prices["close"] / prices["close"].iloc[0]
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
        out["price"] = charts.payload(opened, 1440, rows, res.decisions, [], "kraken", shift_bars=1)
    return out
