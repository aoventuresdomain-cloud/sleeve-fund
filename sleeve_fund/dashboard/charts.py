"""Price charts: the venue's candles with each fill marked, the reason behind it, and the open
position's entry, stop and target as lines. Drawn by TradingView Lightweight Charts."""

from __future__ import annotations

import threading
import time

import pandas as pd

from sleeve_fund.dashboard import trading
from sleeve_fund.venues import venue as venue_profile

# Every candle length the venue's OHLC endpoint serves, 1 minute to 1 week.
INTERVALS = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}
FORCED = {"stop_loss": "Stop", "take_profit": "Target", "risk_halt": "Halt", "risk_pause": "Pause", "pm_flatten": "Flatten",
          "liquidation": "Liquidated", "liquidation_cut": "Liq. cut"}
_cache: dict[tuple[str, str, int], tuple[float, pd.DataFrame]] = {}
_lock = threading.Lock()


def default_interval(bar_spec: str) -> str:
    """The chart interval closest to how often the sleeve decides."""
    step, unit = bar_spec.split("-")[:2]
    minutes = int(step) * {"MINUTE": 1, "HOUR": 60, "DAY": 1440}.get(unit, 1440)
    return min(INTERVALS, key=lambda k: abs(INTERVALS[k] - minutes))


def candles(pair: str, minutes: int, fetch=None, venue: str | None = None) -> pd.DataFrame:
    """The venue's candles, cached briefly: a minute for intraday charts, an hour for daily ones."""
    profile = venue_profile(venue)
    if fetch is None and profile.ohlc_history is None:
        raise OSError(f"{profile.label} has no candle source")
    ttl = 3600 if minutes >= 1440 else 60
    key = (profile.name, pair, minutes)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    df = (fetch or profile.ohlc_history)(pair, minutes)
    with _lock:
        _cache[key] = (time.time(), df)
    return df


_listed: dict[str, tuple[float, list[str]]] = {}


def instruments(get_json=None, venue: str | None = None) -> list[str]:
    """Every BASE/QUOTE pair the venue lists, for the chart's instrument dropdown. Cached for a day;
    raises OSError/ValueError when the venue can't be reached, and the caller falls back to a short list."""
    profile = venue_profile(venue)
    if profile.list_instruments is None:
        raise OSError(f"{profile.label} has no instrument list")
    with _lock:
        hit = _listed.get(profile.name)
        if hit and time.time() - hit[0] < 86400:
            return hit[1]
    pairs = profile.list_instruments(get_json)
    with _lock:
        _listed[profile.name] = (time.time(), pairs)
    return pairs


def from_marks(marks: list[dict], minutes: int) -> pd.DataFrame:
    """Candles built from the sleeve's own price marks, for when the venue can't be reached."""
    if not marks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    s = pd.Series([m["price"] for m in marks], index=pd.DatetimeIndex([m["ts"] for m in marks]))
    s = s[s > 0].sort_index()
    ohlc = s.resample(f"{minutes}min").ohlc().dropna()
    ohlc["volume"] = 0.0
    return ohlc


def _secs(ts) -> int:
    return int(pd.Timestamp(ts).timestamp())


def payload(df: pd.DataFrame, minutes: int, fills: list[dict], orders: dict[str, dict], lines: list[dict],
            source: str, shift_bars: int = 0, limit: int | None = 720) -> dict:
    """Everything the chart draws. fills are oldest first. shift_bars moves a marker filled exactly at a
    bar's close back that many bars, since that fill belongs to the candle that just ended; a fill inside
    a bar (a maker order or a stop matched minute by minute) stays on the candle it happened in. limit:
    the most recent candles kept (a live screen); None keeps them all (a backtest's whole period)."""
    if limit is not None:
        df = df.iloc[-limit:]
    out_candles = [{"time": _secs(t), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
                   for t, r in df.iterrows()]
    volume = [{"time": _secs(t), "value": float(r.volume)} for t, r in df.iterrows()] if "volume" in df else []
    start = df.index[0] if len(df) else None
    bucket = pd.Timedelta(minutes=minutes)
    markers, notes = [], {}
    for i, f in enumerate(fills):
        ts = pd.Timestamp(f["ts"])
        if start is not None and ts < start:
            continue  # older than the chart's first candle
        t = ts.floor(f"{minutes}min")
        if t == ts:  # filled exactly at a bar's close: it belongs to the candle that just ended
            t -= shift_bars * bucket
        o = orders.get(f.get("order_id") or "")
        mid = f"m{i}"
        buy = f["side"] == "BUY"
        kind = trading.INTENTS.get(o["intent"], "") if o else ""
        # Plain arrows for signal trades; a short word only where the exit was forced, as on a trading chart.
        markers.append({"time": _secs(t), "position": "belowBar" if buy else "aboveBar",
                        "shape": "arrowUp" if buy else "arrowDown", "id": mid,
                        "text": FORCED.get(o["intent"], "") if o else ""})
        notes[mid] = {"side": "Buy" if buy else "Sell", "qty": f["qty"], "price": f["price"], "fee": f.get("fee", 0.0),
                      "ts": ts.strftime("%d %b %Y %H:%M UTC"), "intent": kind or ("Entry" if buy else "Exit"),
                      "reason": o["reason"] if o else None, "signal": trading.signal_items(o["signal"]) if o else []}
    markers.sort(key=lambda m: m["time"])
    return {"interval": minutes, "source": source, "candles": out_candles, "volume": volume, "markers": markers,
            "notes": notes, "lines": lines}


def position_lines(position: dict | None) -> list[dict]:
    if not position:
        return []
    out = [{"price": position["entry_px"], "title": "Entry", "kind": "entry"}]
    if position.get("stop_px"):
        out.append({"price": position["stop_px"], "title": "Stop", "kind": "stop"})
    if position.get("target_px"):
        out.append({"price": position["target_px"], "title": "Target", "kind": "target"})
    return out
