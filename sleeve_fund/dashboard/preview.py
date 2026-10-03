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


# Maker orders are matched against short bars from the history store. Past one year the page
# uses 5-minute bars so a long backtest stays quick; an order then counts as filled only once a
# whole 5-minute bar inside its wait has traded through it.
MINUTE_MATCH_DAYS = 365


def execution_history(pair: str, venue: str, start, end, minutes: int):
    """Short bars from the history store to match resting orders against, or None when the store
    hasn't caught up for this instrument (the daily candles then came from the venue)."""
    from sleeve_fund.history import HistoryStore

    store = HistoryStore()
    cov = store.coverage(venue, pair)
    if cov is None or pd.Timestamp.now(tz="UTC") - cov.last >= pd.Timedelta("2D"):
        return None
    df = store.read(venue, pair, minutes, start=start, end=end)
    return df if len(df) else None


def _execution(res, wait, matched_on) -> dict:
    filled = res.fills[res.fills["filled_qty"].astype(float) > 0] if not res.fills.empty else res.fills
    sides = list(filled["liquidity_side"]) if not filled.empty else []
    out = {"maker": bool(wait), "wait": wait, "matched_on": matched_on,
           "maker_orders": sides.count("MAKER"), "orders": len(sides)}
    if wait and matched_on is None:
        out["note"] = ("There is no minute-by-minute history for this instrument here yet, so every maker order is "
                       "assumed to miss and is charged the taker fee.")
    elif wait:
        out["note"] = (f"Maker orders were matched against {matched_on} bars: they count as filled only where the "
                       "price traded through them, never on a touch.")
    return out


def _risk(events: list[dict], risk_profile: str | None) -> dict:
    """What the runtime's risk guard did, in words the page shows."""
    halts = [e for e in events if e["kind"] == "risk_halt"]
    pauses = [e for e in events if e["kind"] == "risk_pause"]
    out = {"profile": risk_profile, "halted": None, "pauses": len(pauses),
           "events": [{"t": e["ts"].strftime("%d %b %Y"), "kind": e["kind"], "message": e["message"]} for e in events]}
    notes = []
    if halts:
        h = halts[0]
        out["halted"] = h["ts"].strftime("%d %b %Y")
        reason = h["message"].split(";")[0]
        notes.append(f"The risk guard halted this strategy on {out['halted']} ({reason}). In paper it would stay "
                     "flat until you resume it, so from then on this backtest holds cash.")
    if pauses:
        notes.append(f"It paused for a day {len(pauses)} time{'s' if len(pauses) != 1 else ''} after a daily loss "
                     "past the profile's limit, flattening each time, as paper would.")
    out["note"] = " ".join(notes)
    return out


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
        detail: bool = False, cap: float | None = None, venue: str | None = None, fee_quote=None,
        risk_profile: str | None = None) -> dict:
    """Backtest these settings on the venue's daily history. days trims to the most recent N days;
    detail adds every trade with its journaled reason, drawdown and fill markers (the backtest page).
    cap is the risk profile's largest position as a share of equity, applied exactly as paper does,
    and the buy-and-hold benchmark is held at the same exposure. risk_profile runs the paper runtime
    itself (cap, drawdown halt, daily-loss pause, journal) and sets cap from the profile."""
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
    if risk_profile is not None:
        from sleeve_fund.risk import profile as risk_profile_of

        cap = risk_profile_of(risk_profile).max_position_pct
    elif cap is not None:
        params = {**params, "position_cap_pct": cap}
    wait = params.get("maker_wait_minutes")
    exec_prices, matched_on = None, None
    if wait and fetch is None:
        step = 1 if len(prices) <= MINUTE_MATCH_DAYS else 5
        exec_prices = execution_history(pair, profile.name, prices.index[0] - pd.Timedelta("1D"), prices.index[-1], step)
        matched_on = None if exec_prices is None else f"{step}-minute"
    res = run_backtest(strategy, prices, inst, params=params, starting_capital=starting, exec_prices=exec_prices,
                       exec_minutes=5 if matched_on == "5-minute" else 1, risk_profile=risk_profile)
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
        "execution": _execution(res, wait, matched_on),
        "risk": _risk(res.risk_events, risk_profile),
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
