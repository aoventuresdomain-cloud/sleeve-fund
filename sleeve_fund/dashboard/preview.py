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


def history(pair: str, fetch=None, venue: str | None = None, minutes: int = 1440, days: int | None = None):
    """Bars of `minutes` for a pair, stamped at their close. Daily: the full stored history when the
    history store has it, otherwise the venue's recent candles. Shorter bars come only from the
    history store, and only once it has caught up to now. Cached so the form doesn't reload on every
    change."""
    from sleeve_fund.history import HistoryStore

    profile = venue_profile(venue)
    if minutes != 1440:
        return _intraday(pair, profile, minutes, days)
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


def _intraday(pair: str, profile, minutes: int, days: int | None) -> pd.DataFrame:
    from sleeve_fund.history import HistoryStore

    store = HistoryStore()
    cov = store.coverage(profile.name, pair)
    if cov is None or pd.Timestamp.now(tz="UTC") - cov.last >= pd.Timedelta("2D"):
        raise ValueError(f"interval: {profile.label}'s minute-by-minute history for {pair} hasn't finished "
                         "loading here yet, so only daily bars can be backtested for now")
    days = min(days or MAX_DAYS[minutes], MAX_DAYS[minutes]) if minutes in MAX_DAYS else days
    start = cov.last - pd.Timedelta(days=days) if days else None
    key = (profile.name, pair, minutes, days)
    with _lock:
        hit = _history.get(key)
        if hit and time.time() - hit[0] < INTRADAY_CACHE_SECONDS:
            return hit[1]
    df = store.read(profile.name, pair, minutes, start=start)
    with _lock:
        _history[key] = (time.time(), df)
    return df


# Short bars mean a lot of rows: a year of 1-minute bars is 525,600, and the engine holds every bar
# in memory. A year at 1 minute takes about 20 s and 0.7 GB (measured 4 Oct 2026, with the backtest
# journal in memory); the server has 4 GB shared with paper trading, so longer runs wait for the
# engine to be fed in chunks.
MAX_DAYS = {1: 365, 5: 365 * 3}
INTRADAY_CACHE_SECONDS = 15 * 60
CHART_CANDLES = 720  # what the price chart shows at most


def _daily(series: pd.Series) -> pd.Series:
    """Daily closes of an intraday series, so returns, Sharpe and drawdowns are annualised as daily."""
    return series.resample("1D", closed="right", label="right").last().dropna()


def _chart_minutes(minutes: int, span: pd.Timedelta) -> int:
    """The finest candle, no finer than the strategy's bars, that fits the whole period on the chart."""
    for m in (minutes, 5, 15, 60, 240, 1440):
        if m >= minutes and span / pd.Timedelta(minutes=m) <= CHART_CANDLES:
            return m
    return 1440


def _coarsen(prices: pd.DataFrame, minutes: int) -> pd.DataFrame:
    g = prices.resample(f"{minutes}min", closed="right", label="right", origin="epoch")
    return g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


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


def _data_note(prices: pd.DataFrame, minutes: int) -> dict:
    """Long stretches with no trades in the bars tested: kept flat, as the venue reported, but said."""
    from sleeve_fund.history import quiet_runs

    q = quiet_runs(prices, minutes)
    note = ""
    if q["count"]:
        hours = q["longest_minutes"] / 60
        note = (f"The price history has {q['count']} stretch{'es' if q['count'] != 1 else ''} of an hour or more "
                f"with no trades (the longest, {hours:.0f} hour{'s' if round(hours) != 1 else ''}, ended "
                f"{q['longest_end']:%d %b %Y %H:%M} UTC). They are held flat at the last price, as the venue "
                "reported no trades; a long one may be a venue outage.")
    return {**q, "longest_end": q["longest_end"].isoformat() if q["longest_end"] is not None else None, "note": note}


def run(strategy: str, pair: str, params: dict, starting: float = 10_000.0, fetch=None, days: int | None = None,
        detail: bool = False, cap: float | None = None, venue: str | None = None, fee_quote=None,
        risk_profile: str | None = None, spread_quote=None, minutes: int = 1440, progress=None,
        keep: dict | None = None) -> dict:
    """Backtest these settings on the venue's history, deciding on bars of `minutes` (daily by default). days trims to the most recent N days;
    detail adds every trade with its journaled reason, drawdown and fill markers (the backtest page).
    cap is the risk profile's largest position as a share of equity, applied exactly as paper does,
    and the buy-and-hold benchmark is held at the same exposure. risk_profile runs the paper runtime
    itself (cap, drawdown halt, daily-loss pause, journal) and sets cap from the profile. spread_quote
    (sleeve_fund.spreads.resolve) is the spread charged on orders that take liquidity; without one,
    the venue's assumption. progress(fraction done) is called as the run goes; keep, when given,
    receives the run's journal under "journal" (with a risk profile), for Store.save_backtest."""
    from sleeve_fund import spreads
    from sleeve_fund.fees import resolve

    profile = venue_profile(venue)
    quote_fees = fee_quote or resolve(profile.name)
    prices = history(pair, fetch, profile.name, minutes=minutes, days=days)
    if days and len(prices):
        prices = prices[prices.index > prices.index[-1] - pd.Timedelta(days=days)]
    if len(prices) < 60:
        raise ValueError(f"only {len(prices)} bars of {profile.label} history for {pair}; need at least 60")
    spread = spread_quote or spreads.resolve(profile.name, pair)
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
    span = prices.index[-1] - prices.index[0]
    if wait and fetch is None:
        step = 1 if span <= pd.Timedelta(days=MINUTE_MATCH_DAYS) or minutes < 5 else 5
        if step < minutes:
            exec_prices = execution_history(pair, profile.name, prices.index[0] - pd.Timedelta(minutes=minutes),
                                            prices.index[-1], step)
            matched_on = None if exec_prices is None else f"{step}-minute"
    tick = None
    if progress is not None:
        t0, t1 = prices.index[0].to_pydatetime(), prices.index[-1].to_pydatetime()
        whole = max((t1 - t0).total_seconds(), 1.0)

        def tick(now):
            progress(min(1.0, max(0.0, (now - t0).total_seconds() / whole)))

    res = run_backtest(strategy, prices, inst, params=params, starting_capital=starting, exec_prices=exec_prices,
                       exec_minutes=5 if matched_on == "5-minute" else 1, risk_profile=risk_profile,
                       half_spread=spread.half_spread, bar_minutes=minutes, progress=tick)
    if keep is not None:
        keep["journal"] = res.journal
    bench = benchmark(prices, starting, float(inst.taker_fee), cap if cap is not None else 1.0)
    if minutes < 1440:  # judge returns day by day, whatever the bar length, so Sharpe is annualised right
        equity, bench = _daily(res.equity), _daily(bench)
    else:
        equity = res.equity
    s, b = summary(returns_from_equity(equity)), summary(returns_from_equity(bench))
    rows = fills_to_rows(res.fills)
    trips = trades(rows)
    stats = trade_stats(trips)
    step = 1 if detail else max(1, len(equity) // 400)
    out = {
        "pair": pair,
        "from": prices.index[0].strftime("%d %b %Y"),
        "to": prices.index[-1].strftime("%d %b %Y"),
        "days": max(1, round((span + pd.Timedelta(minutes=minutes)) / pd.Timedelta("1D"))),  # each bar covers its own time
        "bars": len(prices),
        "minutes": minutes,
        "t": [t.isoformat() for t in equity.index[::step]],
        "equity": [round(v, 2) for v in equity.iloc[::step]],
        "benchmark": [round(v, 2) for v in bench.iloc[::step]],
        "strategy": {k: s[k] for k in ("total_return", "cagr", "sharpe", "max_drawdown", "volatility")},
        "hold": {k: b[k] for k in ("total_return", "cagr", "sharpe", "max_drawdown", "volatility")},
        "trades": {k: _finite(stats[k]) for k in ("trades", "win_rate", "expectancy", "profit_factor")},
        "fees": round(res.fees_paid, 2),
        "exposure": round(float(res.exposure.mean()), 4),
        "cap": cap,
        "data": _data_note(prices, minutes),
        "execution": _execution(res, wait, matched_on),
        "risk": _risk(res.risk_events, risk_profile),
        "spread": {"half": spread.half_spread, "paid": round(res.spread_paid, 2), "text": spread.text,
                   "source": spread.source},
        "fee_schedule": {"maker": float(inst.maker_fee), "taker": float(inst.taker_fee), "text": quote_fees.text,
                         "source": quote_fees.source},
    }
    if detail:
        from sleeve_fund.dashboard import trading

        peak = equity.cummax()
        out["drawdown"] = [round(float(v), 6) for v in (1 - equity / peak)]
        out["fills"] = [{"t": r["ts"].isoformat(), "side": r["side"], "price": r["price"]} for r in rows]
        out["trips"] = trading.trips(list(reversed(rows)), [], res.decisions)
        out["stats"] = {k: _finite(v) for k, v in stats.items()}
        out["strategy"].update(sortino=s["sortino"], calmar=s["calmar"])
        out["hold"].update(sortino=b["sortino"], calmar=b["calmar"])
        out["start"] = round(float(starting), 2)
        from sleeve_fund.dashboard import charts

        # Bars are stamped at their close; the chart wants open times. Fills land on the bar they
        # decided on, which closed at the fill time. Short bars are drawn coarser so the chart holds
        # the whole period.
        shown = _chart_minutes(minutes, span)
        candles = prices if shown == minutes else _coarsen(prices, shown)
        opened = candles.set_axis(candles.index - pd.Timedelta(minutes=shown))
        out["price"] = charts.payload(opened, shown, rows, res.decisions, [], "venue", shift_bars=1)
        out["chart_minutes"] = shown
        out["chart_label"] = "daily candles" if shown == 1440 else (
            f"{shown // 60}-hour candles" if shown % 60 == 0 else f"{shown}-minute candles")
    return out
