"""Quant Researcher re-score lab (20 Oct 2026): run one rule-builder definition on BTC perpetual history through the
real engine (sleeve_fund.research.runner.run_backtest) and score it on the fee ladder, with break-even fee, a
random-entry benchmark and buy-and-hold. Development years only: the history it is given stops before the locked
holdout, so the holdout is never read. A screen, not a G1 study: one pre-registered setting per definition.

    python research/qr-lab/rescore.py --minutes-csv btc_1m.csv.gz --definition configs/definitions/qr01_trend_ema_long.toml \
        --bar 240 --exec 15 --out results/qr01_trend_ema_long.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from sleeve_fund.instruments import FeeSchedule
from sleeve_fund.research.random_entry import Trade, random_entry
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.definitions import check_definition, definition_hash, load_definition_file, to_params
from sleeve_fund.venues import venue

VENUE_TAKER = 0.00055  # paper venue VIP 0 perpetual taker (research/venue-data/paper-venue-fee-schedule-2026-10-15.md)
HALF_SPREAD = 0.0002  # 2 bp a side assumed slippage
RUNGS = (0.0, 0.0002, 0.00055, 0.0011)  # fee per side: zero, maker-like, venue taker, double venue taker
FUNDING_8H = 0.0001  # approximate funding: longs pay, shorts receive 0.01% per 8h held (the base rate)


def candles(minutes: pd.DataFrame, bar: int) -> pd.DataFrame:
    if bar == 1:
        return minutes
    return minutes.resample(f"{bar}min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def round_trips(fills: pd.DataFrame) -> list[dict]:
    """Fills to round trips on a one-position-at-a-time strategy: entry and exit average prices, times and side."""
    out, pos, cur = [], 0.0, None
    for _, f in fills.sort_values("ts_last").iterrows():
        q = float(f["filled_qty"]) * (1 if str(f["side"]).endswith("BUY") else -1)
        px = float(f["avg_px"])
        if pos == 0:
            cur = {"side": 1 if q > 0 else -1, "entry_ts": f["ts_last"], "eq": 0.0, "ev": 0.0, "xq": 0.0, "xv": 0.0}
        if (q > 0) == (cur["side"] > 0):
            cur["eq"] += abs(q); cur["ev"] += abs(q) * px
        else:
            cur["xq"] += abs(q); cur["xv"] += abs(q) * px
        pos += q
        if abs(pos) < 1e-9:
            pos = 0.0
            entry, exit_ = cur["ev"] / cur["eq"], cur["xv"] / cur["xq"]
            out.append({"side": cur["side"], "entry_ts": cur["entry_ts"], "exit_ts": f["ts_last"],
                        "entry": entry, "exit": exit_, "gross": cur["side"] * (exit_ / entry - 1)})
            cur = None
    return out


def compound(rets) -> float:
    r = np.asarray(rets, dtype=float)
    return float(np.prod(1 + r) - 1) if len(r) else 0.0


def break_even(gross: np.ndarray) -> float | None:
    """The fee per side at which the compounded net return is zero (net = gross - 2 x fee per trade)."""
    if not len(gross) or compound(gross) <= 0:
        return None
    lo, hi = 0.0, 0.05
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if compound(gross - 2 * mid) > 0 else (lo, mid)
    return lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes-csv", required=True)
    ap.add_argument("--definition", required=True)
    ap.add_argument("--bar", type=int, required=True)
    ap.add_argument("--exec", dest="exec_minutes", type=int, default=0)
    ap.add_argument("--end", default="2025-10-19", help="last development day; the holdout starts after it")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    m = pd.read_csv(a.minutes_csv, index_col=0, parse_dates=True)
    m.index = pd.to_datetime(m.index, utc=True)
    m = m[m.index <= pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)]
    prices = candles(m, a.bar)
    exec_prices = candles(m, a.exec_minutes) if a.exec_minutes and a.exec_minutes < a.bar else None

    defn = load_definition_file(a.definition)
    checked = check_definition(defn, bar_minutes=a.bar)
    params = {**to_params(defn), "market": "perp", "allow_short": "short" in defn}
    inst = venue("BINANCE").instrument("BTC", "USDT")
    kw = dict(bar_minutes=a.bar, half_spread=HALF_SPREAD)
    if exec_prices is not None:
        kw.update(exec_prices=exec_prices, exec_minutes=a.exec_minutes)

    zero = run_backtest("rules", prices, inst, params=params, fees=FeeSchedule(Decimal("0"), Decimal("0")), **kw)
    real = run_backtest("rules", prices, inst, params=params,
                        fees=FeeSchedule(Decimal(str(VENUE_TAKER)), Decimal("0.0002")), **kw)

    trips = round_trips(zero.fills) if len(zero.fills) else []
    gross = np.array([t["gross"] for t in trips])
    hours = np.array([(t["exit_ts"] - t["entry_ts"]) / pd.Timedelta(hours=1) for t in trips])
    sides = np.array([t["side"] for t in trips])
    funding = -sides * FUNDING_8H * hours / 8 if len(trips) else np.array([])
    years = (prices.index[-1] - prices.index[0]) / pd.Timedelta(days=365.25)

    ladder = {f"{f:.4%}": compound(gross - 2 * f) for f in RUNGS}
    ladder_funding = compound(gross - 2 * VENUE_TAKER + funding) if len(trips) else 0.0

    eq = real.equity.dropna()
    daily = eq.resample("1D").last().dropna().pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * math.sqrt(365)) if daily.std() > 0 else None
    mdd = float((eq / eq.cummax() - 1).min()) if len(eq) else None
    exposure = float((real.exposure.abs() > 1e-9).mean()) if len(real.exposure) else 0.0

    closes = prices["close"].to_numpy()
    idx = prices.index
    re_trades = []
    for t in trips:
        e = int(idx.searchsorted(t["entry_ts"])); x = int(idx.searchsorted(t["exit_ts"]))
        if x > e and x < len(idx):
            re_trades.append(Trade(entry=e, exit=x, side=int(t["side"])))
    first = re_trades[0].entry if re_trades else 0
    rnd = None
    if len(re_trades) >= 5:
        r = random_entry(closes, re_trades, [(first, len(closes) - 1)], VENUE_TAKER + HALF_SPREAD,
                         in_market=exposure)
        rnd = {"return_percentile": r.return_percentile, "sharpe_percentile": r.sharpe_percentile,
               "strategy_return": r.strategy_return, "median_random_return": r.median_random_return,
               "exposure": r.exposure}
    bh = float(closes[-1] / closes[first] - 1) if len(closes) else None

    per_year = {}
    for t, g in zip(trips, gross):
        y = str(pd.Timestamp(t["exit_ts"]).year)
        per_year.setdefault(y, []).append(g - 2 * VENUE_TAKER)
    per_year = {y: compound(v) for y, v in sorted(per_year.items())}

    out = {
        "definition": Path(a.definition).stem, "hash": definition_hash(defn), "bar_minutes": a.bar,
        "exec_minutes": a.exec_minutes or None, "from": str(prices.index[0]), "to": str(prices.index[-1]),
        "years": years, "warmup_bars": checked.warmup.get(None), "trades": len(trips),
        "trades_per_year": len(trips) / years if years else None,
        "win_rate": float((gross > 0).mean()) if len(gross) else None,
        "avg_gross_bp": float(gross.mean() * 1e4) if len(gross) else None,
        "avg_hold_hours": float(hours.mean()) if len(hours) else None,
        "net_return_ladder": ladder, "net_return_venue_with_funding_est": ladder_funding,
        "break_even_fee_per_side": break_even(gross),
        "engine_net_return_venue_fee": float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) else None,
        "engine_fees_paid": real.fees_paid, "sharpe_venue": sharpe, "max_drawdown_venue": mdd,
        "in_market": exposure, "random_entry": rnd, "buy_and_hold_same_window": bh,
        "net_by_exit_year_venue": per_year, "handler_errors": real.handler_errors[:3],
        "seconds": time.time() - t0,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("definition", "trades", "break_even_fee_per_side", "seconds")}, default=str))


if __name__ == "__main__":
    main()
