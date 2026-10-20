"""B-16: a rule-builder definition's rules in plain English, for the Strategy page. Written from the definition as
check_definition() reads it and Compiled builds it (sleeve_fund.strategies.definitions), never by hand, so the words
can't drift from what the engine trades. Every rule and operand form the checker accepts has its words here; one it
doesn't know raises ValueError, which the page shows in place of the sentence (tests/test_b16_strategy_words.py
renders every file in configs/definitions/ and every form the checker accepts).

Display only: nothing here changes a definition, its hash or how it trades."""

from __future__ import annotations

from sleeve_fund.strategies.definitions import (
    ARITHMETIC,
    CATALOGUE,
    COMPARE,
    CROSSES,
    FIELDS,
    LEVEL_EXITS,
    SINCE_ENTRY,
    Checked,
    side_exits,
    timeframe_label,
    timeframe_minutes,
)
from sleeve_fund.strategies.timeframes import DAY_MINUTES

OPS = {">": "is above", "<": "is below", ">=": "is at or above", "<=": "is at or below",
       "crosses_above": "crosses above", "crosses_below": "crosses below"}
assert set(OPS) == set(COMPARE) | set(CROSSES)
SIDES = {"long": ("Buy", "Sell"), "short": ("Sell short", "Buy back")}
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# What a moving average is called, by kind.
AVERAGES = {"sma": "average", "ema": "EMA", "wma": "weighted average"}


def num(x) -> str:
    """2.0 -> 2; 1.5 -> 1.5; 1200 -> 1,200."""
    x = float(x)
    return f"{x:,.0f}" if x == int(x) else f"{x:,g}"


def candles(n: int, minutes: int | None) -> str:
    """'10 candles', '10 daily candles', '7 days' for a period on a block's own candle."""
    n = int(n)
    if minutes == DAY_MINUTES:
        return f"{n} day{'s' if n != 1 else ''}"
    if minutes:
        return f"{n} candle{'s' if n != 1 else ''} of {timeframe_label(minutes)}"
    return f"{n} candle{'s' if n != 1 else ''}"


def _per(n: int, minutes: int | None) -> str:
    """The adjective form: '200-day', '5-candle', '20-candle (1h)'."""
    if minutes == DAY_MINUTES:
        return f"{int(n)}-day"
    return f"{int(n)}-candle" + (f" ({timeframe_label(minutes)})" if minutes else "")


def _on(minutes: int | None) -> str:
    return f" on {timeframe_label(minutes)}" if minutes and minutes != DAY_MINUTES else (" daily" if minutes else "")


def _unit(minutes: int | None) -> str:
    """The candle as an adjective: 'one-hour', 'four-hour', '15-minute', 'daily'."""
    if minutes == DAY_MINUTES:
        return "daily"
    if not minutes:
        return ""
    if minutes % 60 == 0:
        h = minutes // 60
        return {1: "one-hour", 2: "two-hour", 4: "four-hour"}.get(h, f"{h}-hour")
    return f"{minutes}-minute"


def _next_candles(n: int, minutes: int | None) -> str:
    """'the next 10 one-hour candles', 'the next one-hour candle'."""
    n = int(n)
    unit = f"{_unit(minutes)} " if minutes else ""
    return f"the next {n} {unit}candles" if n != 1 else f"the next {unit}candle"


PAST = {"is above": "was above", "is below": "was below", "is at or above": "was at or above",
        "is at or below": "was at or below", "crosses above": "crossed above", "crosses below": "crossed below"}
# Comparisons of relative volume read as "volume is at least 1.5 × its 20-candle average".
VOLUME = {">=": "is at least", ">": "is more than", "<=": "is at most", "<": "is less than"}
# Kinds whose words carry their candle ("RSI(14) on 15m"), so two on different candles need it said.
LABELLED = ("rsi", "atr", "atr_sma", "efficiency_ratio", "bollinger", "keltner", "stochastic", "vwap")


def _is_prev(x) -> bool:
    return isinstance(x, dict) and len(x) == 1 and "prev" in x


def _all_prev(x) -> bool:
    """A value made only of 'the candle before' readings (and plain numbers), at least one of them."""
    if _is_prev(x):
        return True
    if isinstance(x, dict) and len(x) == 1 and next(iter(x)) in ARITHMETIC:
        parts = next(iter(x.values()))
        return any(_all_prev(v) for v in parts) and all(_all_prev(v) or isinstance(v, (int, float)) for v in parts)
    return False


def _unprev(x):
    if _is_prev(x):
        return x["prev"]
    if isinstance(x, dict) and len(x) == 1 and next(iter(x)) in ARITHMETIC:
        (k, parts), = x.items()
        return {k: [_unprev(v) for v in parts]}
    return x


class Words:
    """One checked definition, put into words."""

    def __init__(self, checked: Checked, decision_minutes: int | None = None) -> None:
        self.checked = checked
        self.defn = checked.definition
        self.blocks = checked.blocks  # id -> {"kind", "settings", "timeframe" (minutes or None), "input"}
        # The candle it decides on, when the file names one: only a daily one changes the words ("200-day").
        self.daily = decision_minutes == DAY_MINUTES
        self.decision = decision_minutes
        # A kind used on more than one candle is said with its candle everywhere ("RSI(14) on 15m"), so a reader
        # can't take two of them for the same one.
        candles_of: dict[str, set] = {}
        for b in self.blocks.values():
            candles_of.setdefault(b["kind"], set()).add(b["timeframe"] or decision_minutes)
        self.mixed = {k for k, v in candles_of.items() if len(v) > 1 and k in LABELLED}

    # --- values ----------------------------------------------------------------------------------------------
    def block(self, bid: str, output: str | None) -> str:
        b = self.blocks[bid]
        kind, s, tf = b["kind"], b["settings"], b["timeframe"] or (DAY_MINUTES if self.daily else None)
        lab = b["timeframe"] or (self.decision if kind in self.mixed else None)  # the candle to say, if any
        fed = ""
        if b["input"] and b["input"][0] == "block":
            src = self.block(b["input"][1], b["input"][2])
            fed = f" of {src}"
        elif b["input"] and b["input"][0] == "field" and b["input"][1] != "close":
            fed = f" of the {b['input'][1]}"
        p = s.get("period")
        if kind in AVERAGES:
            return f"the {_per(p, tf)} {AVERAGES[kind]}{fed}"
        if kind in ("rsi", "atr", "atr_sma", "efficiency_ratio"):
            name = {"rsi": "RSI", "atr": "ATR", "atr_sma": "ATR", "efficiency_ratio": "efficiency ratio"}[kind]
            simple = ", simple average" if kind == "atr_sma" else ""
            return f"{name}({int(p)}{simple}){fed}{_on(lab)}"
        if kind == "relative_volume":
            return f"volume as a multiple of its {_per(p, tf)} average"
        if kind == "bollinger":
            what = {"mid": "the middle Bollinger band", "upper": "the upper Bollinger band",
                    "lower": "the lower Bollinger band", "width": "the Bollinger band width",
                    "pct_b": "the close's place in the Bollinger bands (%b)"}[output or "mid"]
            return f"{what} ({int(p)}, {num(s.get('k', 2.0))}){fed}{_on(lab)}"
        if kind == "keltner":
            what = {"mid": "the middle", "upper": "the upper", "lower": "the lower"}[output or "mid"]
            return f"{what} Keltner channel ({int(p)}, {num(s.get('k', 2.0))} × ATR({int(s.get('atr_period', 10))})){_on(lab)}"
        if kind == "donchian":
            hi, lo = ("close", "close") if s.get("source") == "close" else ("high", "low")
            last = f"the last {candles(p, tf)}"
            return {"upper": f"the highest {hi} of {last}", "lower": f"the lowest {lo} of {last}",
                    "mid": f"the middle of {last}' range"}[output or "mid"]
        if kind in ("rolling_max", "rolling_min"):
            word = "highest" if kind == "rolling_max" else "lowest"
            return f"the {word} value{fed or ' of the close'} over the last {candles(p, tf)}"
        if kind == "prior_period":
            return f"the previous {s.get('period', 'week')}'s {output or 'close'}"
        if kind == "session_levels":
            return f"the {s.get('session', 'asia').capitalize()} session's {output or 'high'}"
        if kind == "stochastic":
            return f"stochastic %{(output or 'k').upper()}({int(p)}){_on(lab)}"
        if kind == "rsi_divergence":
            return f"a {output or 'bullish'} RSI divergence"
        if kind == "vwap":
            return f"VWAP ({s.get('anchor', 'day')}){_on(lab)}"
        return f"{CATALOGUE.get(kind, kind)}{fed}{_on(lab)}"

    def value(self, x) -> str:
        if isinstance(x, bool):
            raise ValueError(f"not a value: {x!r}")
        if isinstance(x, (int, float)):
            return num(x)
        if isinstance(x, str):
            if x in FIELDS:
                return f"the {x}"
            bid, _, output = x.partition(".")
            if bid not in self.blocks:
                raise ValueError(f"unknown reference {x!r}")
            return self.block(bid, output or None)
        if not isinstance(x, dict):
            raise ValueError(f"not a value: {x!r}")
        if "time" in x:
            zone = " London time" if x.get("tz") == "Europe/London" else (f" {x['tz']} time" if x.get("tz") else " UTC")
            return {"hour": f"the hour ({zone.strip()})", "weekday": f"the weekday ({zone.strip()})",
                    "day_of_month": "the day of the month", "days_to_month_end": "the days left in the month"}[x["time"]]
        if len(x) != 1:
            raise ValueError(f"not a value: {x!r}")
        (key, arg), = x.items()
        if key == "prev":
            if isinstance(arg, str) and arg in FIELDS:
                return f"the previous candle's {arg}"
            return f"{self.value(arg)} as of the candle before"
        if key in ARITHMETIC:
            a, b = (self._term(v) for v in arg)
            if key == "mul" and isinstance(arg[0], (int, float)) and not isinstance(arg[0], bool):
                return f"{a} × {b}"
            return {"add": f"{a} plus {b}", "sub": f"{a} minus {b}", "mul": f"{a} × {b}", "div": f"{a} ÷ {b}"}[key]
        if key in SINCE_ENTRY:
            word = "highest" if key == "max_since_entry" else "lowest"
            return f"the {word} {self.value(arg).removeprefix('the ')} since the entry candle"
        raise ValueError(f"not a value: {x!r}")

    def _term(self, x) -> str:
        """An arithmetic operand: a nested sum or difference in brackets, so 'a minus (b plus c)' reads right."""
        words = self.value(x)
        if isinstance(x, dict) and len(x) == 1 and next(iter(x)) in ("add", "sub"):
            return f"({words})"
        return words

    # --- rules -----------------------------------------------------------------------------------------------
    def _hour_cond(self, r) -> tuple[str, int, str] | None:
        """(op, hour, zone) for a test of the clock hour against a whole number, else None."""
        left, right = r.get("left"), r.get("right")
        if (isinstance(left, dict) and left.get("time") == "hour" and isinstance(right, (int, float))
                and not isinstance(right, bool) and r.get("op") in COMPARE):
            return r["op"], int(right), self.value(left).split("(", 1)[1].rstrip(")")
        return None

    def _hours(self, items: list, joint: str) -> str | None:
        """Two tests of the same clock hour, as one phrase: 'closes at or after 21:00 or before 06:00 London time'
        (either one), 'closes between 08:00 and 21:00 London time' (both)."""
        if len(items) != 2 or not all(isinstance(i, dict) and set(i) == {"left", "op", "right"} for i in items):
            return None
        hs = [self._hour_cond(i) for i in items]
        if None in hs or hs[0][2] != hs[1][2]:
            return None
        ge = next((h for h in hs if h[0] == ">="), None)
        lt = next((h for h in hs if h[0] == "<"), None)
        if not ge or not lt:
            return None
        zone = ge[2]
        if joint == "any":
            return f"the candle closes at or after {ge[1]:02d}:00 or before {lt[1]:02d}:00 {zone}"
        return f"the candle closes between {ge[1]:02d}:00 and {lt[1]:02d}:00 {zone}"

    def cond(self, r: dict) -> str:
        left, right, op = r["left"], r["right"], r["op"]
        if (h := self._hour_cond(r)) is not None:
            _, n, zone = h
            return {">=": f"the candle closes at or after {n:02d}:00 {zone}", "<": f"the candle closes before {n:02d}:00 {zone}",
                    ">": f"the candle closes after {n:02d}:59 {zone}",
                    "<=": f"the candle closes at or before {n:02d}:59 {zone}"}[op]
        if isinstance(left, dict) and left.get("time") == "weekday" and isinstance(right, (int, float)):
            return f"the weekday {OPS[op].replace('above', 'after').replace('below', 'before')} {WEEKDAYS[int(right)]}"
        if (isinstance(left, str) and isinstance(right, (int, float)) and not isinstance(right, bool)
                and op in VOLUME and self.blocks.get(left.partition(".")[0], {}).get("kind") == "relative_volume"):
            b = self.blocks[left.partition(".")[0]]
            tf = b["timeframe"] or (DAY_MINUTES if self.daily else None)
            return f"volume {VOLUME[op]} {num(right)} × its {_per(b['settings'].get('period'), tf)} average"
        return f"{self.value(left)} {OPS[op]} {self.value(right)}"

    def _prev_cond(self, r) -> str | None:
        """A comparison made wholly on the candle before, without saying so each time ('the close was above ...'),
        for the caller to say 'at the candle before' once; None for any other rule."""
        if not (isinstance(r, dict) and set(r) == {"left", "op", "right"} and r["op"] in COMPARE):
            return None
        if not (_all_prev(r["left"]) and (_all_prev(r["right"]) or isinstance(r["right"], (int, float)))):
            return None
        return f"{self.value(_unprev(r['left']))} {PAST[OPS[r['op']]]} {self.value(_unprev(r['right']))}"

    def rule(self, r, nested: bool = False) -> str:
        if not isinstance(r, dict):
            raise ValueError(f"not a rule: {r!r}")
        keys = set(r)
        if keys == {"left", "op", "right"}:
            return self.cond(r)
        if keys in ({"all"}, {"any"}):
            joint = " and " if "all" in r else " or "
            items = r["all"] if "all" in r else r["any"]
            if (hours := self._hours(items, "all" if "all" in r else "any")) is not None:
                return hours
            parts, before, sep = [], [], joint
            for x in items:
                if "all" in r and (t := self._prev_cond(x)) is not None:
                    before.append(t)
                    if len(before) == 1:
                        parts.append(None)  # where the candle-before group goes
                else:
                    parts.append(self.rule(x, nested=True))
            if before:
                at = f"at the candle before, {' and '.join(before)}"
                parts = [at if p is None else p for p in parts]
                sep = "; and " if len(parts) > 1 else joint
            text = sep.join(parts)
            return f"({text})" if nested and len(parts) > 1 else text
        if keys == {"only_when", "then"}:
            return f"{self.rule(r['then'], True)}, only while {self.rule(r['only_when'], True)}"
        if keys == {"event", "confirm", "within"}:
            n = int(r["within"])
            return (f"{self.rule(r['event'], True)}, confirmed when {self.rule(r['confirm'], True)} on that candle or "
                    f"one of the next {n}")
        if keys == {"first_touch"}:
            ft = r["first_touch"]
            return (f"inside this candle, the price reaches {self.value(ft['reach'])} before "
                    f"{self.value(ft['before'])} (levels as set at the candle before)")
        if {"setup", "trigger", "expire_after"} <= keys <= {"setup", "trigger", "expire_after", "cancel"}:
            ex = r["expire_after"]
            tf = timeframe_minutes(ex["timeframe"])
            text = (f"{self.rule(r['setup'])} (checked as each {timeframe_label(tf)} candle closes), then "
                    f"within {_next_candles(ex['bars'], tf)} {self.rule(r['trigger'], 'cancel' in r)}")
            if "cancel" in r:
                text += f", unless {self.rule(r['cancel'], True)} first"
            return text
        raise ValueError(f"not a rule: {', '.join(sorted(map(str, keys)))}")

    # --- exits -----------------------------------------------------------------------------------------------
    @staticmethod
    def time_stop(ts: dict | None) -> str | None:
        if not ts:
            return None
        if "minutes" in ts:
            m = int(ts["minutes"])
            if m % DAY_MINUTES == 0:
                n = m // DAY_MINUTES
                return f"{n} day{'s' if n != 1 else ''} have passed"
            if m % 60 == 0:
                return f"{m // 60} hour{'s' if m != 60 else ''} have passed"
            return f"{m} minute{'s' if m != 1 else ''} have passed"
        n = int(ts["bars"])
        if ts.get("count") == "bars":
            return f"{n} candle{'s' if n != 1 else ''} have closed"
        return f"{n} candle{'s' if n != 1 else ''} have passed"

    def side(self, name: str) -> dict:
        """One side's sentences: entry, exit (rule, time stop), and its level exits."""
        s = self.defn[name]
        enter, leave = SIDES[name]
        exits = side_exits(self.defn.get("exits"), name)
        entry = f"{enter} when {self.rule(s['entry'])}".replace("when at the candle before,", "when, at the candle before,", 1)
        if s.get("breakout"):
            entry += " (a breakout, so the fill allows for extra slippage)"
        outs = [self.rule(s["exit"])] if "exit" in s else []
        if (ts := self.time_stop(exits.get("time_stop"))) is not None:
            outs.append(ts)
        out = f"{leave} when {' or '.join(outs)}" if outs else f"{leave} only on the exits below"
        if s.get("reverse"):
            out += f", and reverse into the {'short' if name == 'long' else 'long'} on that candle"
        levels = []
        if "stop" in exits:
            levels.append(("Stop", f"{self.value(exits['stop']['level'])}, set as the trade opens"))
        if "trail" in exits:
            t = exits["trail"]
            how = "moves only in the trade's favour" if t.get("ratchet", True) else "moves both ways"
            levels.append(("Trailing stop", f"{self.value(t['level'])}, as set at each close; it {how}"))
        if "exit_at_level" in exits:
            levels.append(("Exit at", f"{self.value(exits['exit_at_level']['level'])}, as set at the close before"))
        return {"side": name, "entry": entry, "exit": out, "levels": levels,
                "stopless": not any(k in exits for k in LEVEL_EXITS)}


def rules(checked: Checked, decision_minutes: int | None = None) -> list[dict]:
    """Each side's rules as sentences (long first). ValueError for a form this module has no words for."""
    w = Words(checked, decision_minutes)
    return [w.side(name) for name in ("long", "short") if name in checked.definition]


def _settings_words(kind: str, s: dict) -> str:
    """A block's settings other than its period, in words ('1.5 × ATR(20)', '2 standard deviations')."""
    s = dict(s)
    out = []
    if kind == "keltner":
        k, atr = s.pop("k", None), s.pop("atr_period", None)
        if k is not None or atr is not None:
            out.append(f"{num(2.0 if k is None else k)} × ATR({num(10 if atr is None else atr)})")
    if kind == "bollinger" and "k" in s:
        k = s.pop("k")
        out.append(f"{num(k)} standard deviation{'s' if float(k) != 1 else ''}")
    if kind == "donchian" and "source" in s:
        out.append({"high_low": "highs and lows", "close": "closes"}.get(s.pop("source"), "highs and lows"))
    out += [f"{k.replace('_', ' ')} {num(v) if isinstance(v, (int, float)) else v}" for k, v in s.items()]
    return ", ".join(out)


def indicators(checked: Checked, decision: str | None = None) -> list[dict]:
    """The blocks table: name, kind, period (or settings), candle (the decision candle's, as the file names it, for
    a block on it), input."""
    out = []
    for bid in checked.order:
        b = checked.blocks[bid]
        s = dict(b["settings"])
        period = s.pop("period", None)
        rest = _settings_words(b["kind"], s)
        src = b["input"]
        fed = ("" if src is None else src[1] if src[0] == "field" else src[1] + (f".{src[2]}" if src[2] else ""))
        out.append({"name": bid, "kind": CATALOGUE.get(b["kind"], b["kind"]),
                    "period": num(period) if period is not None else "", "settings": rest,
                    "candle": timeframe_label(b["timeframe"]) if b["timeframe"] else (decision or "decision candle"),
                    "input": fed or "price"})
    return out
