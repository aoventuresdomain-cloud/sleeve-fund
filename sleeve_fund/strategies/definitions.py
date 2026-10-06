"""Strategy definitions for the rule builder (v2 P1-5): which indicator blocks a strategy reads and its rules for
entering and leaving long and short, as data rather than code. Definitions are TOML files in configs/definitions/,
copied into a strategy's params with their version and content hash when it is created (to_params), so a strategy
never changes because a file did. Design: /mnt/project-files/sleeve-fund/v2/p1-5-rule-builder-design.md.

    version = 1
    reason = "why it should work, written before it is tested"
    [blocks.rsi]   kind = "rsi", period = 14
    [blocks.rsi15] kind = "rsi", period = 14, timeframe = "15m"   # read only once its candle has closed (P1-4)
    [blocks.avg]   kind = "sma", period = 3, input = "rsi"        # fed the RSI once the RSI has settled
    [long]  entry = {left = "rsi", op = "crosses_above", right = 30}, exit = {left = "rsi", op = ">=", right = 55}
    [exits] time_stop = {minutes = 720}, stop = {level = "dc.lower"}

A condition is {left, op, right}. An operand is a number, a price field (open, high, low, close, volume), a block
("rsi", or "bb.upper" for one of several outputs; its value as set at the last close), {prev = x} (x on the decision
candle before), {add|sub|mul|div = [a, b]}, {time = "hour"|"weekday", tz = "Europe/London"} (at the candle's close,
Monday 0; UTC unless tz names a zone, whose summer time it follows), or {max_since_entry = x} / {min_since_entry = x}
(since the leg's entry candle, that candle included). Conditions combine with {all = [...]},
{any = [...]}, {only_when = gate, then = rule}, {event = rule, confirm = rule, within = N} (the event, confirmed on
its candle or one of the next N) and {setup = rule, trigger = rule, expire_after = {bars, timeframe}, cancel = rule}
(the setup is checked as its slower candle closes and arms the trigger until N candles of that timeframe have
passed, a trigger on the last of their closes included; the cancel holding on a later close, or on the same one as a
trigger, disarms it; a fresh setup re-arms it from that close).

Exits: time_stop = {minutes = N} or {bars = N} is wall-clock, ending the leg on the first candle at or after the
deadline, so missing candles never lengthen a hold; {bars = N, count = "bars"} counts the candles that came, as
rsi_cross does. stop = {level} is set when the position opens; trail = {level, ratchet} follows its level and with
ratchet never loosens; exit_at_level = {level, rest = "previous_close"} follows its level both ways. Each rests in
the next candle at the level as set at the close before it, never at the candle's own (Independent Quant Advisor,
6 Oct 16:40 and 16:43).

check_definition() refuses a definition with the reason; compile() turns a checked one into the rules the `rules`
model evaluates (sleeve_fund.strategies.rules)."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import re
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sleeve_fund.strategies.indicators import BLOCKS, Donchian
from sleeve_fund.strategies.indicators._common import warmup
from sleeve_fund.strategies.timeframes import DAY_MINUTES, MINUTE_NS, span

VERSION = 1
FIELDS = ("open", "high", "low", "close", "volume")
COMPARE = {">": lambda a, b: a > b, "<": lambda a, b: a < b, ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b}
CROSSES = ("crosses_above", "crosses_below")
ARITHMETIC = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
              "div": lambda a, b: a / b if b else None}
TIMES = ("hour", "weekday")
SINCE_ENTRY = {"max_since_entry": max, "min_since_entry": min}
TOP_KEYS = ("version", "name", "reason", "blocks", "long", "short", "exits", "costs")
LEVEL_EXITS = ("stop", "trail", "exit_at_level")
MAX_BREAKOUT_SLIPPAGE_BP = 500

class ClosedChannel(Donchian):
    """A Donchian channel as the rule builder defines one (Advisor, 6 Oct 17:07): the n most recent closed bars,
    the bar just closed included. At bar i's close upper is the highest high of bars i-n+1..i; a level resting
    inside the next bar is therefore that bar's n predecessors, and a breakout compares the close with the
    previous value ({"prev": "dc.upper"}). The library block, as the donchian model trades it, leaves bar i out."""

    def update_raw(self, high: float, low: float, close: float) -> None:
        hi, lo = (float(close), float(close)) if self.source == "close" else (float(high), float(low))
        while self._highs and self._highs[-1][1] <= hi:
            self._highs.pop()
        self._highs.append((self.count, hi))
        while self._lows and self._lows[-1][1] >= lo:
            self._lows.pop()
        self._lows.append((self.count, lo))
        self.count += 1
        oldest = self.count - self.period
        while self._highs[0][0] < oldest:
            self._highs.popleft()
        while self._lows[0][0] < oldest:
            self._lows.popleft()
        upper, lower = self._highs[0][1], self._lows[0][1]
        self._value = (upper + lower) / 2
        self._vals = {"upper": upper, "lower": lower, "mid": self._value}

    @property
    def initialized(self) -> bool:
        return self.count >= self.period

    @warmup
    def warmup_bars(cls, s) -> int:
        return s["period"]


# The blocks a definition builds, by name: the library's, with channels counted as the rule builder defines them.
BUILDER_BLOCKS = {**BLOCKS, "donchian": ClosedChannel}


# What the builder offers for each block, by its name in BLOCKS. ATR is Wilder's; the simple average of true ranges
# that the hand-coded models size their stops on is offered beside it (Independent Quant Advisor, 6 Oct 15:35).
CATALOGUE = {
    "sma": "SMA", "ema": "EMA", "wma": "WMA", "vwap": "VWAP", "rsi": "RSI", "bollinger": "Bollinger bands",
    "atr": "ATR", "atr_sma": "ATR (simple average)", "relative_volume": "Relative volume",
    "efficiency_ratio": "Efficiency ratio", "donchian": "Donchian channel", "rsi_divergence": "RSI divergence",
    "stochastic": "Stochastic", "keltner": "Keltner channel",
}
_ID = re.compile(r"^[a-z_][a-z0-9_]*$")
_TIMEFRAME = re.compile(r"^(\d+)(m|h|d)$")


def timeframe_minutes(tf) -> int:
    m = _TIMEFRAME.match(str(tf).strip().lower())
    if not m or int(m[1]) <= 0:
        raise ValueError(f"timeframe {tf!r}: write it as a number of minutes, hours or days, e.g. 15m, 4h or 1d")
    return int(m[1]) * {"m": 1, "h": 60, "d": DAY_MINUTES}[m[2]]


def timeframe_label(minutes: int) -> str:
    """240 -> 4h, as a definition writes it."""
    return (f"{minutes // DAY_MINUTES}d" if minutes % DAY_MINUTES == 0 else
            f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes}m")


def _canonical(x):
    """The definition as hashed: tables in key order (TOML tables are unordered maps), every number as a float
    (30 and 30.0 are the same setting)."""
    if isinstance(x, dict):
        out = {str(k): _canonical(x[k]) for k in sorted(x)}
        for k in ("all", "any"):  # one variant whatever order its members are written in (Advisor 6 Oct, 5.9)
            if isinstance(out.get(k), list):
                members = {json.dumps(m, sort_keys=True): m for m in out[k]}
                out[k] = [members[t] for t in sorted(members)]
        return out
    if isinstance(x, (list, tuple)):
        return [_canonical(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return float(x)
    raise ValueError(f"a definition holds numbers, text, true/false, lists and tables, not {type(x).__name__}")


def definition_hash(defn: dict, venue: str | None = None) -> str:
    """The definition's content hash: the trials register's key for a variant. Its name and reason are left out,
    so rewording the reason is not a new variant; any change to its blocks, rules, exits or costs is, and so is
    running it on a venue whose day (and so its slower candles) starts away from 00:00 UTC (Advisor 6 Oct, 4.4)."""
    body = {k: v for k, v in defn.items() if k not in ("name", "reason")}
    if venue is not None:
        from sleeve_fund.venues import VENUES

        profile = VENUES.get(str(venue))
        anchor = profile.daily_anchor_minutes if profile is not None else 0
        if anchor:  # a venue anchored at 00:00 UTC leaves the hash as it is without one
            body["daily_anchor_minutes"] = anchor
    text = json.dumps(_canonical(body), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_definition_file(path) -> dict:
    with open(Path(path), "rb") as f:
        return tomllib.load(f)


def to_params(defn: dict, venue: str | None = None) -> dict:
    """A `rules` strategy's params for this definition, as copied in when the strategy is created: the definition
    itself with its version and content hash (on `venue`, when known), which the model checks again when it
    starts."""
    checked = check_definition(defn)
    return {"definition": copy.deepcopy(defn), "definition_version": checked.version,
            "definition_hash": definition_hash(defn, venue)}


@dataclass(frozen=True)
class Checked:
    """A definition that passed check_definition. warmup: decision-candle (None) or slower-timeframe minutes to
    the candles of that size its blocks need before every one reads as settled. notes: what to know before
    running it (a warm-up longer than the venue's candles can give)."""

    definition: dict
    content_hash: str
    version: int
    blocks: dict  # block id -> {"kind", "settings", "timeframe" (None or minutes), "input"}
    order: tuple  # block ids, each after the block it is fed from
    warmup: dict
    notes: tuple = ()

    @property
    def slower(self) -> dict[int, int]:
        """{minutes: candles of warm-up} for each slower timeframe the definition reads."""
        return {tf: n for tf, n in self.warmup.items() if tf is not None}


def _block_outputs(kind: str) -> tuple:
    return BUILDER_BLOCKS[kind].OUTPUTS


def _single_input(kind: str) -> bool:
    params = [p for p in inspect.signature(BUILDER_BLOCKS[kind].update_raw).parameters.values()
              if p.name != "self" and p.default is p.empty]
    return len(params) == 1


def check_definition(defn: dict, bar_spec: str | None = None, bar_minutes: int | None = None) -> Checked:
    """Check a definition (a dict as TOML parses it) and return it Checked, or raise ValueError with the reason.
    bar_spec or bar_minutes: the candle the strategy decides on, when known, so its slower timeframes and warm-up
    are checked against it."""
    from sleeve_fund.paper.config import MAX_WARMUP_BARS, VENUE_WARMUP_BARS

    if not isinstance(defn, dict):
        raise ValueError("a definition is a table of settings")
    unknown = set(defn) - set(TOP_KEYS)
    if unknown:
        raise ValueError(f"a definition has no {', '.join(sorted(unknown))}; it takes {', '.join(TOP_KEYS)}")
    if defn.get("version") != VERSION:
        raise ValueError(f"definition version must be {VERSION}, got {defn.get('version')!r}")
    if not isinstance(defn.get("reason"), str) or not defn["reason"].strip():
        raise ValueError("a definition carries its reason, written before it is tested")
    if bar_minutes is None and bar_spec:
        from sleeve_fund.data import spec_minutes

        bar_minutes = spec_minutes(bar_spec)
    blocks = _check_blocks(defn.get("blocks") or {}, bar_minutes)
    order = _feed_order(blocks)
    uses_prev = []
    for side in ("long", "short"):
        if side in defn:
            _check_side(defn[side], side, blocks, bar_minutes, uses_prev)
    if "long" not in defn and "short" not in defn:
        raise ValueError("a definition needs a long or a short side, each with its entry rule")
    _check_exits(defn.get("exits") or {}, blocks, bar_minutes, uses_prev,
                 sides=[side for side in ("long", "short") if side in defn])
    _check_costs(defn.get("costs") or {}, defn)
    warmup: dict = {None: 0}
    longest: dict = {}
    for bid in order:
        b = blocks[bid]
        own = BUILDER_BLOCKS[b["kind"]].warmup_bars(**b["settings"])
        src = b["input"]
        b["warmup"] = own + (blocks[src[1]]["warmup"] if src and src[0] == "block" else 0)
        if b["warmup"] > warmup.get(b["timeframe"], 0):
            warmup[b["timeframe"]], longest[b["timeframe"]] = b["warmup"], bid
    if uses_prev:
        warmup[None] += 1  # a cross or a previous value reads the candle before the first settled one
    notes = []
    for tf, n in warmup.items():
        size = f"{span(tf)} candles" if tf is not None else "decision candles"
        if n > MAX_WARMUP_BARS:
            raise ValueError(f"block {longest.get(tf, '?')} needs {n:,} {size} of warm-up, more than the "
                             f"{MAX_WARMUP_BARS:,} that can ever load (MAX_WARMUP_BARS), so it would never settle")
    if bar_spec and not bar_spec.endswith("INTERNAL") and warmup[None] > VENUE_WARMUP_BARS:
        notes.append(f"Short warm-up: block {longest.get(None, '?')} needs {warmup[None]:,} candles and the venue's "
                     f"own candles give {VENUE_WARMUP_BARS}, so it waits for live candles before it decides")
    return Checked(definition=copy.deepcopy(defn), content_hash=definition_hash(defn), version=VERSION,
                   blocks=blocks, order=tuple(order), warmup=warmup, notes=tuple(notes))


def _check_blocks(spec: dict, bar_minutes: int | None) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("blocks is a table of block name to its settings")
    out = {}
    for bid, b in spec.items():
        if not _ID.match(str(bid)) or bid in FIELDS:
            raise ValueError(f"block name {bid!r}: use lower-case letters, digits and _, and not a price field")
        if not isinstance(b, dict) or "kind" not in b:
            raise ValueError(f"block {bid} needs a kind, e.g. kind = \"rsi\"")
        kind = b["kind"]
        if kind not in BLOCKS:
            raise ValueError(f"block {bid}: no indicator block called {kind!r}; known: {', '.join(sorted(BLOCKS))}")
        settings = {k: v for k, v in b.items() if k not in ("kind", "timeframe", "input")}
        try:
            settings = {k: v for k, v in BUILDER_BLOCKS[kind].check_settings(settings).items() if v is not None}
        except ValueError as exc:
            raise ValueError(f"block {bid} ({kind}): {exc}") from None
        tf = _timeframe(b["timeframe"], bar_minutes, f"block {bid}") if "timeframe" in b else None
        out[bid] = {"kind": kind, "settings": settings, "timeframe": tf, "input": b.get("input")}
    for bid, b in out.items():
        src = b["input"]
        if src is None:
            continue
        if not isinstance(src, str):
            raise ValueError(f"block {bid}: input is a price field or another block, e.g. input = \"rsi\"")
        if not _single_input(b["kind"]):
            raise ValueError(f"block {bid}: a {b['kind']} block reads several prices, so it can't be fed another "
                             "block or a single field")
        if src in FIELDS:
            b["input"] = ("field", src)
            continue
        ref, _, output = src.partition(".")
        if ref not in out:
            raise ValueError(f"block {bid}: input {src!r} is not a block in this definition or a price field")
        if out[ref]["timeframe"] != b["timeframe"]:
            raise ValueError(f"block {bid}: input {ref} is on another timeframe; chained blocks share one")
        outputs = _block_outputs(out[ref]["kind"])
        if output and output not in outputs:
            raise ValueError(f"block {bid}: {ref} has no output {output!r}; it has {', '.join(outputs)}")
        b["input"] = ("block", ref, output or None)
    return out


def _timeframe(tf, bar_minutes: int | None, where: str) -> int | None:
    minutes = timeframe_minutes(tf)
    if DAY_MINUTES % minutes:
        raise ValueError(f"{where}: {minutes}-minute candles don't divide a day (e.g. 15m, 1h, 4h or 1d)")
    if bar_minutes is None:
        return minutes
    if minutes == bar_minutes:
        return None
    if minutes < bar_minutes or minutes % bar_minutes:
        raise ValueError(f"{where}: {span(minutes)} candles can't be built from the {span(bar_minutes)} candles "
                         "the strategy decides on: a timeframe is a whole multiple of them")
    return minutes


def _feed_order(blocks: dict) -> list:
    """Block ids, each after the block that feeds it. A loop is refused naming every block in it."""
    order, state = [], {}

    def visit(bid, path):
        if state.get(bid) == "done":
            return
        if state.get(bid) == "open":
            loop = path[path.index(bid):]
            raise ValueError(f"blocks {', '.join(loop)} feed each other in a loop")
        state[bid] = "open"
        src = blocks[bid]["input"]
        if src and src[0] == "block":
            visit(src[1], path + [src[1]])
        state[bid] = "done"
        order.append(bid)

    for bid in blocks:
        visit(bid, [bid])
    return order


def _check_side(side: dict, name: str, blocks: dict, bar_minutes, uses_prev: list) -> None:
    if not isinstance(side, dict) or "entry" not in side:
        raise ValueError(f"the {name} side needs its entry rule")
    unknown = set(side) - {"entry", "exit", "breakout"}
    if unknown:
        raise ValueError(f"the {name} side has no {', '.join(sorted(unknown))}; it takes entry, exit, breakout")
    if not isinstance(side.get("breakout", False), bool):
        raise ValueError(f"{name}.breakout is true or false")
    _check_rule(side["entry"], f"{name}.entry", blocks, bar_minutes, uses_prev, entry=True)
    if "exit" in side:
        _check_rule(side["exit"], f"{name}.exit", blocks, bar_minutes, uses_prev)


def _check_rule(rule, where: str, blocks: dict, bar_minutes, uses_prev: list, entry: bool = False) -> None:
    if not isinstance(rule, dict):
        raise ValueError(f"{where}: a rule is a condition {{left, op, right}} or all/any/only_when/event/setup")
    keys = set(rule)
    if keys == {"left", "op", "right"}:
        if rule["op"] not in COMPARE and rule["op"] not in CROSSES:
            raise ValueError(f"{where}: no comparison {rule['op']!r}; use {', '.join([*COMPARE, *CROSSES])}")
        if rule["op"] in CROSSES:
            uses_prev.append(where)
        _check_operand(rule["left"], where, blocks, uses_prev)
        _check_operand(rule["right"], where, blocks, uses_prev)
    elif keys in ({"all"}, {"any"}):
        items = rule["all"] if "all" in rule else rule["any"]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{where}: all/any takes a list of rules")
        for i, r in enumerate(items):
            _check_rule(r, f"{where}[{i}]", blocks, bar_minutes, uses_prev, entry)
    elif keys == {"only_when", "then"}:
        _check_rule(rule["only_when"], f"{where}.only_when", blocks, bar_minutes, uses_prev)
        _check_rule(rule["then"], f"{where}.then", blocks, bar_minutes, uses_prev, entry)
    elif keys == {"event", "confirm", "within"}:
        if isinstance(rule["within"], bool) or not isinstance(rule["within"], int) or not 1 <= rule["within"] <= 1000:
            raise ValueError(f"{where}: within is a whole number of candles, 1 to 1,000")
        _check_rule(rule["event"], f"{where}.event", blocks, bar_minutes, uses_prev)
        _check_rule(rule["confirm"], f"{where}.confirm", blocks, bar_minutes, uses_prev)
    elif keys == {"first_touch"}:
        ft = rule["first_touch"]
        if not isinstance(ft, dict) or set(ft) != {"reach", "before"}:
            raise ValueError(f"{where}: first_touch is {{reach, before}}: the price reaches one level before the other "
                             "inside the candle, e.g. {reach = {add = [\"lvl\", \"atr\"]}, before = {sub = [\"lvl\", "
                             "\"atr\"]}}")
        uses_prev.append(where)  # its levels are read at the candle before's close
        if _canonical(ft["reach"]) == _canonical(ft["before"]):
            raise ValueError(f"{where}: first_touch's reach and before are the same level, so neither can come first")
        _check_operand(ft["reach"], f"{where}.first_touch.reach", blocks, uses_prev)
        _check_operand(ft["before"], f"{where}.first_touch.before", blocks, uses_prev)
    elif {"setup", "trigger", "expire_after"} <= keys <= {"setup", "trigger", "expire_after", "cancel"}:
        if not entry:
            raise ValueError(f"{where}: a setup and trigger arms an entry, so it belongs in an entry rule")
        ex = rule["expire_after"]
        if not isinstance(ex, dict) or set(ex) != {"bars", "timeframe"}:
            raise ValueError(f"{where}: expire_after is {{bars, timeframe}}, e.g. four 15m candles")
        if isinstance(ex["bars"], bool) or not isinstance(ex["bars"], int) or not 1 <= ex["bars"] <= 10_000:
            raise ValueError(f"{where}: expire_after.bars is a whole number of candles, 1 to 10,000")
        _timeframe(ex["timeframe"], bar_minutes, f"{where}.expire_after")
        _check_rule(rule["setup"], f"{where}.setup", blocks, bar_minutes, uses_prev)
        _check_rule(rule["trigger"], f"{where}.trigger", blocks, bar_minutes, uses_prev)
        if "cancel" in rule:
            _check_rule(rule["cancel"], f"{where}.cancel", blocks, bar_minutes, uses_prev)
    else:
        raise ValueError(f"{where}: not a rule: {', '.join(sorted(map(str, keys)))}")


def _check_operand(x, where: str, blocks: dict, uses_prev: list) -> None:
    if isinstance(x, bool):
        raise ValueError(f"{where}: {x!r} is not a number or a value")
    if isinstance(x, (int, float)):
        if not math.isfinite(x):
            raise ValueError(f"{where}: {x!r} is not a finite number")
        return
    if isinstance(x, str):
        _reference(x, where, blocks)
        return
    if isinstance(x, dict) and "time" in x and set(x) <= {"time", "tz"}:
        if x["time"] not in TIMES:
            raise ValueError(f"{where}: time is one of {', '.join(TIMES)}, got {x['time']!r}")
        if "tz" in x:
            _zone(x["tz"], where)
        return
    if isinstance(x, dict) and len(x) == 1:
        (key, arg), = x.items()
        if key == "prev":
            uses_prev.append(where)
            _check_operand(arg, where, blocks, uses_prev)
            return
        if key in ARITHMETIC:
            if not isinstance(arg, list) or len(arg) != 2:
                raise ValueError(f"{where}: {key} takes two operands, e.g. {{{key} = [\"close\", 2]}}")
            for a in arg:
                _check_operand(a, where, blocks, uses_prev)
            return
        if key in SINCE_ENTRY:
            _check_operand(arg, where, blocks, uses_prev)
            return
    raise ValueError(f"{where}: not a value: {x!r}")


def _zone(tz, where: str):
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        return ZoneInfo(str(tz))
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"{where}: no timezone called {tz!r}; use a name such as Europe/London or UTC") from None


def _reference(ref: str, where: str, blocks: dict) -> tuple:
    if ref in FIELDS:
        return ("field", ref)
    bid, _, output = ref.partition(".")
    if bid not in blocks:
        raise ValueError(f"{where}: unknown reference {ref!r}: not a block in this definition or a price field")
    outputs = _block_outputs(blocks[bid]["kind"])
    if output and output not in outputs:
        raise ValueError(f"{where}: {bid} has no output {output!r}; it has {', '.join(outputs)}")
    return ("block", bid, output or None)


def _never_stop_side(level, side: str, blocks: dict) -> bool:
    """Whether a level is on the target side of the decision close on every candle, so as a `side` position's stop
    it would hold back every entry (Code Reviewer on a69618a): the close itself, the candle's high for a long or low
    for a short, or this timeframe's channel upper for a long or lower for a short (it takes in the candle just
    closed, ClosedChannel). Any other level can sit on either side, and an entry waits while it is wrong
    (Rules._plan_exits)."""
    if not isinstance(level, str):
        return False
    if level in FIELDS:
        return level in ("close", "high" if side == "long" else "low")
    bid, _, output = level.partition(".")
    b = blocks.get(bid)
    return (b is not None and b["kind"] == "donchian" and b["timeframe"] is None
            and output == ("upper" if side == "long" else "lower"))


def _check_exits(exits: dict, blocks: dict, bar_minutes, uses_prev: list, sides=("long", "short")) -> None:
    if not isinstance(exits, dict):
        raise ValueError("exits is a table: time_stop, stop, trail, exit_at_level")
    unknown = set(exits) - {"time_stop", *LEVEL_EXITS}
    if unknown:
        raise ValueError(f"exits has no {', '.join(sorted(unknown))}; it takes time_stop, {', '.join(LEVEL_EXITS)}")
    if "time_stop" in exits:
        ts = exits["time_stop"]
        form = isinstance(ts, dict) and (set(ts) == {"minutes"} or set(ts) in ({"bars"}, {"bars", "count"}))
        n = (ts.get("minutes", ts.get("bars")) if form else None)
        if not form or isinstance(n, bool) or not isinstance(n, int) or n < 1 or ts.get("count", "bars") != "bars":
            raise ValueError("exits.time_stop is {minutes = N} (wall-clock), {bars = N} (that many candles of time) "
                             "or {bars = N, count = \"bars\"} (the candles that came), N a whole number, at least 1")
    for kind in LEVEL_EXITS:
        if kind not in exits:
            continue
        e = exits[kind]
        allowed = {"level", "ratchet"} if kind == "trail" else {"level", "rest"} if kind == "exit_at_level" else {"level"}
        if not isinstance(e, dict) or "level" not in e or not set(e) <= allowed:
            raise ValueError(f"exits.{kind} is {{{', '.join(sorted(allowed))}}}")
        if e.get("rest", "previous_close") != "previous_close":
            raise ValueError(f"exits.{kind}.rest: it rests at the level set at the previous close "
                             "(\"previous_close\"); a candle's own level isn't known until it closes, so it can't rest "
                             f"on it ({e['rest']!r}) without reading ahead")
        if not isinstance(e.get("ratchet", True), bool):
            raise ValueError("exits.trail.ratchet is true or false")
        _check_operand(e["level"], f"exits.{kind}", blocks, uses_prev)
        for side in sides:
            if _never_stop_side(e["level"], side, blocks):
                raise ValueError(f"exits.{kind}: {e['level']} is never {'below' if side == 'long' else 'above'} the "
                                 f"close, so as the {side} side's stop it would hold back every {side} entry; a level "
                                 "exit works only on the stop side")


def _check_costs(costs: dict, defn: dict) -> None:
    if not isinstance(costs, dict) or not set(costs) <= {"breakout_slippage_bp"}:
        raise ValueError("costs takes breakout_slippage_bp")
    bp = costs.get("breakout_slippage_bp", 0)
    if isinstance(bp, bool) or not isinstance(bp, (int, float)) or not 0 <= bp <= MAX_BREAKOUT_SLIPPAGE_BP:
        raise ValueError(f"breakout_slippage_bp is in basis points, 0 to {MAX_BREAKOUT_SLIPPAGE_BP}")
    if bp and not any((defn.get(s) or {}).get("breakout") for s in ("long", "short")):
        raise ValueError("breakout_slippage_bp is charged on a side marked breakout = true, and none is")


# --- evaluating a checked definition -------------------------------------------------------------------------------

def uses_first_touch(defn) -> bool:
    """Whether a definition has a first_touch rule anywhere, which needs the 1-minute bars inside each candle."""
    if isinstance(defn, dict):
        return "first_touch" in defn or any(uses_first_touch(v) for v in defn.values())
    if isinstance(defn, list):
        return any(uses_first_touch(v) for v in defn)
    return False


@dataclass
class Env:
    """What the rules read on one decision candle."""

    ohlcv: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)
    ts: int = 0  # ns, the candle's close
    blocks: dict = field(default_factory=dict)  # block id -> block
    prev: dict = field(default_factory=dict)  # operand key -> its value on the decision candle before
    since: dict = field(default_factory=dict)  # operand key -> its max or min since the leg's entry candle
    closed: set = field(default_factory=set)  # slower timeframes (minutes) whose candle closed on this one
    bar: int = 0  # decision candles seen
    # The candle's 1-minute bars, (close ns, open, high, low, close), oldest first, for first_touch: None when no
    # minutes are to hand; minutes_due is how many the candle has.
    minutes: list | None = None
    minutes_due: int = 1
    note: object = None  # note(kind, message): the strategy's own notes, for what a rule couldn't judge


def block_value(block, output: str | None):
    """A block's output as set at the last close, None until it reads as settled (QA P1-A5: never `initialized`)."""
    if not block.settled:
        return None
    v = block.value if output is None else block.values.get(output)
    return None if v is None or not math.isfinite(v) else float(v)


class Compiled:
    """A checked definition ready to evaluate: its blocks, its rules as nodes, and the operands whose previous or
    since-entry values it keeps."""

    def __init__(self, checked: Checked) -> None:
        self.checked = checked
        self.blocks = {bid: BUILDER_BLOCKS[b["kind"]](**b["settings"]) for bid, b in checked.blocks.items()}
        self.tracked: dict[str, object] = {}  # key -> fn, recorded after every candle for prev and crosses
        self.since: dict[str, tuple] = {}  # key -> (fn, max or min)
        self.setups: list[Setup] = []
        self.touches: list[FirstTouch] = []
        defn = checked.definition
        self.sides = {}
        for sign, name in ((1, "long"), (-1, "short")):
            if name in defn:
                s = defn[name]
                self.sides[sign] = {"entry": self.rule(s["entry"]),
                                    "exit": self.rule(s["exit"], exit_rule=True) if "exit" in s else None,
                                    "breakout": bool(s.get("breakout", False))}
        exits = defn.get("exits") or {}
        ts = exits.get("time_stop") or {}
        # ("minutes", n) wall-clock, ("bars", n) n candles of wall-clock time, ("count", n) the candles that came
        self.time_stop = (("minutes", ts["minutes"]) if "minutes" in ts else ("count", ts["bars"]) if "count" in ts
                          else ("bars", ts["bars"]) if ts else None)
        self.levels = {k: self.operand(exits[k]["level"])[0] for k in LEVEL_EXITS if k in exits}  # level fns
        self.ratchet = bool((exits.get("trail") or {}).get("ratchet", True))
        self.slippage_bp = float((defn.get("costs") or {}).get("breakout_slippage_bp", 0) or 0)

    # operands: (fn(env) -> float | None, key, slowest timeframe read)
    def operand(self, x):
        if isinstance(x, (int, float)):
            v = float(x)
            return (lambda env: v), json.dumps(v), None
        if isinstance(x, str):
            ref = _reference(x, "", self.checked.blocks)
            if ref[0] == "field":
                i = FIELDS.index(ref[1])
                return (lambda env: env.ohlcv[i]), x, None
            block, output = self.blocks[ref[1]], ref[2]
            return (lambda env: block_value(block, output)), x, self.checked.blocks[ref[1]]["timeframe"]
        if "time" in x:
            what, zone = x["time"], _zone(x["tz"], "") if "tz" in x else None

            def clock(env):
                if zone is None:
                    days, rest = divmod(env.ts // MINUTE_NS, DAY_MINUTES)
                    return float(rest // 60) if what == "hour" else float((days + 3) % 7)  # 1 Jan 1970: a Thursday
                local = datetime.fromtimestamp(env.ts / 1e9, tz=zone)
                return float(local.hour if what == "hour" else local.weekday())

            return clock, f"time({what}{',' + x['tz'] if zone else ''})", None
        (key, arg), = x.items()
        if key == "prev":
            fn, inner, tf = self.operand(arg)
            self.tracked[inner] = fn
            return (lambda env: env.prev.get(inner)), f"prev({inner})", tf
        if key in ARITHMETIC:
            (fa, ka, ta), (fb, kb, tb) = self.operand(arg[0]), self.operand(arg[1])
            op = ARITHMETIC[key]

            def arith(env):
                a, b = fa(env), fb(env)
                if a is None or b is None:
                    return None
                v = op(a, b)
                return None if v is None or not math.isfinite(v) else v

            return arith, f"{key}({ka},{kb})", _slowest(ta, tb)
        fn, inner, tf = self.operand(arg)
        k = f"{key}({inner})"
        self.since[k] = (fn, SINCE_ENTRY[key])
        return (lambda env: env.since.get(k)), k, tf

    def rule(self, r, exit_rule: bool = False) -> "Node":
        """A rule as a node. exit_rule: it ends a leg, so an ambiguous first_touch in it resolves to true."""
        if "op" in r:
            (fl, kl, tl), (fr, kr, tr) = self.operand(r["left"]), self.operand(r["right"])
            if r["op"] in CROSSES:
                self.tracked[kl], self.tracked[kr] = fl, fr
            return Cond(r, fl, fr, kl, kr, _slowest(tl, tr))
        if "all" in r or "any" in r:
            items = [self.rule(x, exit_rule) for x in (r["all"] if "all" in r else r["any"])]
            return Group(items, "all" in r)
        if "only_when" in r:
            return Group([self.rule(r["only_when"], exit_rule), self.rule(r["then"], exit_rule)], True)
        if "event" in r:
            return Confirm(self.rule(r["event"], exit_rule), self.rule(r["confirm"], exit_rule), r["within"])
        if "first_touch" in r:
            (fx, kx, tx), (fy, ky, ty) = self.operand(r["first_touch"]["reach"]), self.operand(r["first_touch"]["before"])
            fc, kc, _ = self.operand("close")
            self.tracked.update({kx: fx, ky: fy, kc: fc})  # the levels and the reference, as the candle before closed
            node = FirstTouch(kx, ky, kc, _slowest(tx, ty), exit_rule)
            self.touches.append(node)
            return node
        setup = Setup(self.rule(r["setup"]), self.rule(r["trigger"]),
                      self.rule(r["cancel"]) if "cancel" in r else None, r["expire_after"]["bars"],
                      timeframe_minutes(r["expire_after"]["timeframe"]))
        self.setups.append(setup)
        return setup

    def record(self, env: Env) -> None:
        """After a candle is decided: the values `prev` reads on the next one."""
        env.prev = {k: fn(env) for k, fn in self.tracked.items()}

    def start_leg(self, env: Env) -> None:
        env.since = {k: fn(env) for k, (fn, _) in self.since.items()}

    def carry_leg(self, env: Env) -> None:
        for k, (fn, pick) in self.since.items():
            v, was = fn(env), env.since.get(k)
            env.since[k] = v if was is None else (was if v is None else pick(was, v))


def _slowest(a, b):
    if a is None:
        return b
    return a if b is None else max(a, b)


class Node:
    timeframe = None

    def tick(self, env: Env) -> None:
        """Every candle, before any rule is read: the state a setup or a confirmation keeps."""

    def test(self, env: Env) -> bool:
        raise NotImplementedError

    def consume(self) -> None:
        """This rule just opened a leg: a setup or an event it used is spent."""

    def leaves(self) -> list:
        return [self]


class Cond(Node):
    def __init__(self, r: dict, fl, fr, kl: str, kr: str, timeframe) -> None:
        self.op, self.fl, self.fr, self.kl, self.kr, self.timeframe = r["op"], fl, fr, kl, kr, timeframe
        self.text = f"{kl} {r['op'].replace('_', ' ')} {kr}"

    def values(self, env: Env) -> tuple:
        return self.fl(env), self.fr(env)

    def test(self, env: Env) -> bool:
        a, b = self.values(env)
        if a is None or b is None:
            return False
        if self.op in COMPARE:
            return COMPARE[self.op](a, b)
        pa, pb = env.prev.get(self.kl), env.prev.get(self.kr)
        if pa is None or pb is None:
            return False
        return pa < pb and a >= b if self.op == "crosses_above" else pa > pb and a <= b


class Group(Node):
    def __init__(self, items: list, every: bool) -> None:
        self.items, self.every = items, every
        self.timeframe = None
        for n in items:
            self.timeframe = _slowest(self.timeframe, n.timeframe)

    def tick(self, env: Env) -> None:
        for n in self.items:
            n.tick(env)

    def test(self, env: Env) -> bool:
        return all(n.test(env) for n in self.items) if self.every else any(n.test(env) for n in self.items)

    def consume(self) -> None:
        for n in self.items:
            n.consume()

    def leaves(self) -> list:
        return [x for n in self.items for x in n.leaves()]


class Confirm(Node):
    """An event that counts once a confirmation holds on its candle or one of the next `within`."""

    def __init__(self, event: Node, confirm: Node, within: int) -> None:
        self.event, self.confirm, self.within = event, confirm, within
        self.timeframe = _slowest(event.timeframe, confirm.timeframe)
        self.seen: int | None = None  # the decision candle the event last held on

    def tick(self, env: Env) -> None:
        self.event.tick(env)
        self.confirm.tick(env)
        if self.event.test(env):
            self.seen = env.bar

    def test(self, env: Env) -> bool:
        return self.seen is not None and env.bar - self.seen <= self.within and self.confirm.test(env)

    def consume(self) -> None:
        self.seen = None

    def leaves(self) -> list:
        return self.event.leaves() + self.confirm.leaves()


class Setup(Node):
    """Setup then trigger: the setup is read as its slower candle closes and arms the trigger until `bars`
    candles of `minutes` have passed since it armed (a trigger on the last of their closes still counts), the
    cancel holds on a later close, or the trigger opens a leg. A later close where the setup holds again arms it
    afresh from that close."""

    def __init__(self, setup: Node, trigger: Node, cancel: Node | None, bars: int, minutes: int) -> None:
        self.setup, self.trigger, self.cancel = setup, trigger, cancel
        self.span_ns = bars * minutes * MINUTE_NS
        self.timeframe = _slowest(setup.timeframe, trigger.timeframe)
        self.armed_at: int | None = None
        self.until: int | None = None

    def _closed(self, node: Node, env: Env) -> bool:
        return node.timeframe is None or node.timeframe in env.closed

    def tick(self, env: Env) -> None:
        for n in (self.setup, self.trigger, self.cancel):
            if n is not None:
                n.tick(env)
        if self.armed_at is not None and env.ts > self.until:
            self.armed_at = self.until = None
        if self.armed_at is not None and self.cancel is not None and self._closed(self.cancel, env) \
                and self.cancel.test(env):
            self.armed_at = self.until = None
        if self._closed(self.setup, env) and self.setup.test(env):
            self.armed_at, self.until = env.ts, env.ts + self.span_ns

    def test(self, env: Env) -> bool:
        return self.armed_at is not None and self.trigger.test(env)

    def consume(self) -> None:
        self.armed_at = self.until = None

    def leaves(self) -> list:
        return self.setup.leaves() + self.trigger.leaves() + (self.cancel.leaves() if self.cancel else [])


@dataclass(frozen=True)
class Touch:
    """How one candle reached two levels. holds: `reach` came first. reach_at, before_at: the close (ns) of the first
    minute that reached each, when minutes were read. by: "range" when the candle's own high and low settled it (at
    most one level inside it), "minutes" when its minutes did. same_minute: both first reached in one minute.
    unknown: why the order couldn't be told ("missing": a minute before the first reach isn't to hand; "inconsistent":
    the range says a level was reached and no minute did). An ambiguous candle (same_minute or unknown) resolves to
    the safe side of where the rule is used (FirstTouch)."""

    holds: bool
    reach_at: int | None = None
    before_at: int | None = None
    by: str = "range"
    same_minute: bool = False
    unknown: str | None = None
    reached: bool = False  # either level was reached in the candle

    @property
    def ambiguous(self) -> bool:
        return self.same_minute or self.unknown is not None


def _reached(level: float, high: float, low: float, ref: float) -> bool:
    """A level at or above `ref` (the candle before's close) is reached by a high, one below it by a low."""
    return high >= level if level >= ref else low <= level


def first_touch(minutes, reach: float, before: float, ref: float, *, high: float | None = None,
                low: float | None = None, start: int | None = None, step: int = 0) -> Touch:
    """Whether the price reached `reach` before `before` (Independent Quant Advisor, 6 Oct ~22:07).

    With the candle's high and low: if at most one level lies inside its range, that settles it without minutes.
    Otherwise (or without them) its minutes, (close ns, open, high, low, close) oldest first, are walked: the first
    that reaches a level decides; one reaching both is a same-minute case. With `start` (the candle before's close,
    ns) and `step` (minutes in the candle), a minute missing before the first reach makes it unknown, and so does a
    full set of minutes reaching neither level the range says was reached."""
    if high is not None and low is not None:
        x, y = _reached(reach, high, low, ref), _reached(before, high, low, ref)
        if not (x and y):
            return Touch(x, reached=x or y)
    have = {m[0]: m for m in minutes}
    due = [start + k * MINUTE_NS for k in range(1, step + 1)] if start is not None else sorted(have)
    for ts in due:
        m = have.get(ts)
        if m is None:
            return Touch(False, by="minutes", unknown="missing", reached=True)
        x, y = _reached(reach, m[2], m[3], ref), _reached(before, m[2], m[3], ref)
        if x and y:
            return Touch(False, ts, ts, by="minutes", same_minute=True, reached=True)
        if y:
            return Touch(False, None, ts, by="minutes", reached=True)
        if x:
            return Touch(True, ts, None, by="minutes", reached=True)
    if high is None:
        return Touch(False, by="minutes")
    return Touch(False, by="minutes", unknown="inconsistent", reached=True)


class FirstTouch(Node):
    """first_touch {reach, before}: within this candle the price reached `reach` before `before`, both levels as they
    stood at the candle before's close, so nothing from the path being judged sets them. An ambiguous candle (both
    reached in one minute, or the order unknown) resolves to the safe side (Advisor ~22:07): false where the rule
    opens or arms an entry, true in an exit rule, so it never blocks an exit. `flip` resolves the other way, for the
    G1 check on the worse of the two. Counts what it judged, for the report."""

    def __init__(self, kx: str, ky: str, kc: str, timeframe, exit_rule: bool = False) -> None:
        self.kx, self.ky, self.kc, self.timeframe, self.exit_rule = kx, ky, kc, timeframe, exit_rule
        self.text = f"first_touch(reach {kx}, before {ky})"
        self.flip = False
        self.judged_ts: int | None = None
        self.last: Touch | None = None
        self.result = False
        self.stats = {"candles": 0, "held": 0, "either_reached": 0, "by_minutes": 0, "same_minute": 0,
                      "unknown_missing": 0, "unknown_inconsistent": 0}

    def test(self, env: Env) -> bool:
        if self.judged_ts != env.ts:
            self.judged_ts, self.last = env.ts, self._judge(env)
            t, st = self.last, self.stats
            self.result = (self.exit_rule != self.flip) if t.ambiguous else t.holds
            st["candles"] += 1
            st["held"] += self.result
            st["either_reached"] += t.reached
            st["by_minutes"] += t.by == "minutes"
            st["same_minute"] += t.same_minute
            st["unknown_missing"] += t.unknown == "missing"
            st["unknown_inconsistent"] += t.unknown == "inconsistent"
            if t.unknown and env.note is not None:
                why = ("a minute before the first reach isn't to hand" if t.unknown == "missing"
                       else "its minutes reach neither level, though its high and low do")
                env.note("first_touch_unknown", f"{self.text}: both levels are inside the candle to {_iso(env.ts)} "
                         f"and the order can't be told ({why}); taken as {str(self.result).lower()}, the safe side")
        return self.result

    def _judge(self, env: Env) -> Touch:
        x, y, ref = env.prev.get(self.kx), env.prev.get(self.ky), env.prev.get(self.kc)
        if x is None or y is None or ref is None:
            return Touch(False)  # a level not settled at the candle before: nothing to judge
        _, high, low, _, _ = env.ohlcv
        return first_touch(env.minutes or (), x, y, ref, high=high, low=low,
                           start=env.ts - env.minutes_due * MINUTE_NS, step=env.minutes_due)

    def lineage(self) -> dict:
        t = self.last
        return {"held": self.result, "by": t.by, "reach_at": _iso(t.reach_at) if t.reach_at else None,
                "before_at": _iso(t.before_at) if t.before_at else None, "same_minute": t.same_minute,
                "unknown": t.unknown}


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()
