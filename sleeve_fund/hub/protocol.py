"""The market data hub's stream (v2 P1-1): one JSON object per line over TCP.

A client opens with {"v": 1, "sub": ["BTCUSDT-PERP.BINANCE", ...]} and the hub answers {"t": "hello", "v": 1,
"pending": [ids not relayed yet], "instruments": [each relayed one's definition, Instrument.to_dict()]}, or
{"t": "error", ...} and closes. Then, for the instruments subscribed:
- {"t": "trade", "id", "px", "qty", "side", "tid", "ts", "recv"}: a trade print;
- {"t": "quote", "id", "bid", "ask", "bid_qty", "ask_qty", "ts", "recv"}: the best bid and ask;
- {"t": "bar", "id", "o", "h", "l", "c", "v", "ts", "recv", "refilled"}: a closed 1-minute bar stamped at its
  close (ts), built once by the hub from the trades it relays; refilled = fetched from the venue's REST candles
  after a gap rather than built live;
- {"t": "gap", "id", "since", "until"}: minutes the hub could not build live (a dropped connection), being
  refilled;
and to every client, every HEARTBEAT_SECONDS, {"t": "hb", "ts", "venue_up"}: so a client tells a quiet market
from a dead hub. Prices and sizes travel as the venue's own decimal strings, so nothing is rounded on the way.
Timestamps are UNIX nanoseconds: ts the venue's (or the bar's close), recv when the hub received it."""

from __future__ import annotations

import json

VERSION = 1
HEARTBEAT_SECONDS = 5.0
MAX_LINE = 64 * 1024  # a subscription line longer than this is refused


def encode(msg: dict) -> bytes:
    return (json.dumps(msg, separators=(",", ":"), default=str) + "\n").encode()


def decode(line: bytes) -> dict:
    msg = json.loads(line)
    if not isinstance(msg, dict) or "t" not in msg and "sub" not in msg:
        raise ValueError(f"not a hub message: {line[:80]!r}")
    return msg


def subscription(instrument_ids: list[str]) -> dict:
    return {"v": VERSION, "sub": sorted(set(instrument_ids))}


def check_subscription(msg: dict) -> list[str]:
    """The instrument ids a client's opening line asks for. Raises ValueError for another version or no list."""
    if msg.get("v") != VERSION:
        raise ValueError(f"hub stream version {msg.get('v')!r} not served; this hub speaks {VERSION}")
    sub = msg.get("sub")
    if not isinstance(sub, list) or not sub or not all(isinstance(s, str) and s for s in sub):
        raise ValueError("the opening line must list the instrument ids to subscribe to")
    return sub


def trade(tick, recv_ns: int) -> dict:
    return {"t": "trade", "id": str(tick.instrument_id), "px": str(tick.price), "qty": str(tick.size),
            "side": str(tick.aggressor_side), "tid": str(tick.trade_id), "ts": int(tick.ts_event), "recv": recv_ns}


def quote(tick, recv_ns: int) -> dict:
    return {"t": "quote", "id": str(tick.instrument_id), "bid": str(tick.bid_price), "ask": str(tick.ask_price),
            "bid_qty": str(tick.bid_size), "ask_qty": str(tick.ask_size), "ts": int(tick.ts_event), "recv": recv_ns}


def bar(instrument_id: str, o, h, l, c, v, close_ns: int, recv_ns: int, refilled: bool = False) -> dict:  # noqa: E741
    return {"t": "bar", "id": instrument_id, "o": str(o), "h": str(h), "l": str(l), "c": str(c), "v": str(v),
            "ts": int(close_ns), "recv": recv_ns, "refilled": refilled}


def bar_from_nautilus(b, recv_ns: int) -> dict:
    return bar(str(b.bar_type.instrument_id), b.open, b.high, b.low, b.close, b.volume, b.ts_event, recv_ns)
