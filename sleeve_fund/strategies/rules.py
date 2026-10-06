"""The rule builder's model (v2 P1-5): one strategy class that trades a definition (sleeve_fund.strategies.definitions)
instead of hand-written code. A new idea is a new definition, not a new model.

It decides nothing until every block it reads has settled (QA P1-A5). Then, as RSI bands and RSI cross do, it is on
one leg at a time: on a leg it counts the candle and checks the leg's exit rule and time stop; flat, or on the
candle a leg ended, it checks the long entry and then the short one. A short leg on spot is held flat.

The definition's level exits work on the position as today's stop does: `stop` is the level when the position opens,
`trail` follows its level and with `ratchet` never loosens, `exit_at_level` follows its level both ways. The tightest
rests at the venue in a backtest and is watched on every trade in paper, moved after each close; a close already
past it exits at market on that close. Every order carries the DA-4 lineage payload: the definition's hash, the rule
that fired, the block values, the candles they were read from and where those came from."""

from __future__ import annotations

from datetime import datetime, timezone

from nautilus_trader.model import Bar

from sleeve_fund.data import bar_minutes
from sleeve_fund.strategies.base import MAX_STOP, MIN_STOP, IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.definitions import (CATALOGUE, LEVEL_EXITS, Checked, Compiled, Env, block_value,
                                                check_definition, definition_hash, load_definition_file,
                                                timeframe_label, to_params)
from sleeve_fund.strategies.timeframes import MINUTE_NS

_hash_of = definition_hash  # RulesConfig takes a parameter of that name

SPEC = IdeaSpec(
    listed=False,
    summary="Trades its definition: the blocks and rules in its settings.",
    name="rules",
    family="rule-builder",
    idea="A strategy written as a definition: indicator blocks and the rules that enter and leave long and short.",
    rules=(
        "Waits until every block has settled. On a leg: its exit rule, its time stop and the definition's level "
        "exits end it. Flat: the long entry, then the short entry, opens a leg. A short leg is held flat on spot. "
        "All of its capital in or out."
    ),
    data_needs="1-minute OHLCV",
    default_params={},
    known_weaknesses="As good as its definition. Each definition carries its reason, written before it is tested.",
)

__all__ = ["CATALOGUE", "SPEC", "Rules", "RulesConfig", "check_definition", "definition_hash", "load_definition_file",
           "to_params"]


class RulesConfig(LongFlatConfig):
    def __init__(self, *, definition: dict | None = None, definition_version: int | None = None,
                 definition_hash: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if not definition:
            raise ValueError("a rules strategy needs its definition (a file in configs/definitions/)")
        checked = check_definition(definition, bar_minutes=bar_minutes(self.bar_type))
        content_hash = _hash_of(definition, self.instrument_id.venue.value)
        if definition_hash is not None and definition_hash != content_hash:
            raise ValueError(f"the definition's content hash is {content_hash}, not the {definition_hash} it was "
                             "created with: it, or its venue's day start, changed after it was copied in")
        if definition_version is not None and definition_version != checked.version:
            raise ValueError(f"definition version {definition_version} isn't one this engine runs ({checked.version})")
        levels = [k for k in LEVEL_EXITS if k in (definition.get("exits") or {})]
        if levels and (self.stop_loss or self.stop_atr or self.stop_swing_bars):
            raise ValueError("choose the definition's stop or the strategy's stop-loss setting, not both")
        self.definition = definition
        self.definition_hash = content_hash
        self.checked: Checked = checked


class _Feed:
    """One timeframe's blocks, fed in order: a block chained to another gets that block's value once it has
    settled, or the price field it names; the rest take the candle."""

    def __init__(self, compiled: Compiled, ids: list) -> None:
        checked = compiled.checked.blocks
        self.nodes = [(compiled.blocks[i], checked[i]["input"]) for i in ids]
        self.blocks = compiled.blocks
        self.warmup_bars = max((checked[i]["warmup"] for i in ids), default=0)

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        fields = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
        for block, src in self.nodes:
            if src is None:
                block.update_ohlcv(open_, high, low, close, volume, ts_ns)
            elif src[0] == "field":
                block.update_raw(fields[src[1]])
            else:
                x = block_value(self.blocks[src[1]], src[2])
                if x is not None:
                    block.update_raw(x)


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


class Rules(LongFlatStrategy):
    def __init__(self, config: RulesConfig) -> None:
        super().__init__(config)
        self.c = config
        self.rules = Compiled(config.checked)
        self.env = Env(blocks=self.rules.blocks)
        by_tf: dict = {}
        for bid in config.checked.order:
            by_tf.setdefault(config.checked.blocks[bid]["timeframe"], []).append(bid)
        self._feed = _Feed(self.rules, by_tf.pop(None, []))
        self._slow = {tf: self.slower(tf, _Feed(self.rules, ids)) for tf, ids in sorted(by_tf.items())}
        self._counts = dict.fromkeys(self._slow, 0)
        self._leg = 0  # the leg the rules are on: +1 long, -1 short, 0 flat
        self._held = 0  # decision candles the leg has run
        self._leg_ts: int | None = None  # ns: the close of the candle the leg opened on, for a wall-clock time stop
        self._held_before: dict[int, bool] = {}  # whether each side's entry rule held on the candle before
        self._levels: dict | None = None  # the open position's level exits, by kind, set when it opened
        self._why: tuple[str, dict] | None = None  # why the leg changed on this candle, with the lineage payload
        # first_touch: (after ns, until ns) -> the 1-minute bars closing in (after, until], as (close ns, open, high,
        # low, close): the backtest's own minutes (research.runner), the hub's in paper (paper.node)
        self.minute_source = None

    @classmethod
    def warmup_needed(cls, params: dict, bar_minutes: int) -> int:
        if not params.get("definition"):
            return 0
        checked = check_definition(params["definition"], bar_minutes=bar_minutes)
        need = checked.warmup[None]
        for tf, candles in checked.slower.items():  # a backtest builds its slower candles from these too
            # One more than the blocks take: the warm-up rarely starts on a slower candle's open, and the part
            # candle it starts in is dropped (SlowerCandles), at any venue anchor.
            need = max(need, (candles + 1) * tf // max(bar_minutes, 1))
        return need

    @classmethod
    def slower_needs(cls, params: dict) -> dict[int, int]:
        return check_definition(params["definition"]).slower if params.get("definition") else {}

    def resume_leg(self, side: int, held: int) -> None:
        # After a restart: the leg, and its time stop from the journal's entry candle. Counting back the candles
        # held from the next decision would come out late after missing candles, lengthening the hold (5.2); that
        # is only the fallback when the journal has no entry for the position.
        self._leg, self._held, self._leg_ts = side, held, self._resume_entry_ns

    def _unsettled(self) -> bool:
        return False  # every value its rules read has settled (block_value), or the rule doesn't hold

    def update_indicators(self, bar: Bar) -> None:
        env = self.env
        if env.bar:
            if self.rules.slippage_bp:  # a breakout candle is the first on which its entry rule holds
                self._held_before = {sign: s["entry"].test(env) for sign, s in self.rules.sides.items()}
            self.rules.record(env)  # the candle before's values, for prev and crosses, before this one is fed
        o, h, lo, c, v = (bar.open.as_double(), bar.high.as_double(), bar.low.as_double(), bar.close.as_double(),
                          bar.volume.as_double())
        self._feed.update_ohlcv(o, h, lo, c, v, bar.ts_event)
        env.ohlcv, env.ts, env.bar = (o, h, lo, c, v), int(bar.ts_event), env.bar + 1
        if self.rules.touches:
            env.minutes, env.minutes_due = self._minutes_of(bar), bar_minutes(self._cfg.bar_type)
        env.closed = {tf for tf, s in self._slow.items() if s.count != self._counts[tf]}
        self._counts = {tf: s.count for tf, s in self._slow.items()}
        for side in self.rules.sides.values():  # setups arm and confirmations count on every candle
            side["entry"].tick(env)
            if side["exit"] is not None:
                side["exit"].tick(env)

    def _minutes_of(self, bar: Bar) -> list | None:
        """The candle's own 1-minute bars, oldest first, for first_touch: never one closing after it (look-ahead) or
        at or before the candle before's close. None, or fewer than the candle has, when they aren't to hand: the
        path is then unknown and the rule doesn't hold on it."""
        step, end = bar_minutes(self._cfg.bar_type), int(bar.ts_event)
        if step == 1:
            return [(end, bar.open.as_double(), bar.high.as_double(), bar.low.as_double(), bar.close.as_double())]
        start = end - step * MINUTE_NS
        got = self.minute_source(start, end) if self.minute_source is not None else None
        got = sorted({m[0]: m for m in got or () if start < m[0] <= end}.values())
        if len(got) < step:
            self._note("first_touch_unknown", f"First-touch not judged on the candle to {_iso(end)}: "
                       f"{step - len(got)} of its {step} minutes aren't to hand, so the rule doesn't hold on it",
                       level="info")
        else:
            self._noted.discard("first_touch_unknown")
        return got

    def first_touch_stats(self) -> dict:
        """For the report, per first_touch rule: candles judged, held, either level reached, both first reached in one
        minute (the adverse level taken as first), and not judged for minutes missing."""
        return {n.text: dict(n.stats) for n in self.rules.touches}

    # --- the leg ----------------------------------------------------------------------------------------------

    def want_side(self, bar: Bar) -> int | None:
        env, word = self.env, {1: "long", -1: "short"}
        fired, ended = [], 0
        if self._leg:
            step = bar_minutes(self._cfg.bar_type) * MINUTE_NS
            if self._leg_ts is None:
                self._leg_ts = env.ts - (self._held + 1) * step
            self._held += 1
            self.rules.carry_leg(env)
            side, name = self.rules.sides[self._leg], word[self._leg]
            past = self._move_levels()
            if side["exit"] is not None and side["exit"].test(env):
                fired.append((f"{name}.exit", f"the {name} exit rule holds"))
            elif self._time_up(step):
                fired.append((f"{name}.time_stop", f"{self._held_words(step)} without the exit: the {name} ends on "
                                                   "its time stop"))
            elif past:
                fired.append((f"{name}.{past[0]}", f"the close {env.ohlcv[3]:,.6g} is past the {past[0].replace('_', ' ')} "
                                                   f"at {past[1]:,.6g}"))
            if not fired:
                self._why = None  # nothing fired: if the position still has to follow the leg, explain() says so
                return self._leg
            why = fired[0][0].split(".")[1]
            if why != "exit" and not (why == "time_stop" and self.rules.time_stop[0] == "count"):
                # A time stop or a level ends the hold: the same side opens again only on a later candle, or the
                # position would simply be held on. Counting candles keeps rsi_cross's way, which re-enters at once.
                ended = self._leg
            self._leg = 0
        for sign in (1, -1):
            side = self.rules.sides.get(sign)
            if side is not None and sign != ended and side["entry"].test(env):
                armed = [s.armed_at for s in self.rules.setups if s.armed_at is not None]
                side["entry"].consume()
                self._leg, self._held, self._leg_ts = sign, 0, env.ts
                self.rules.start_leg(env)
                fired.append((f"{word[sign]}.entry", f"the {word[sign]} entry rule holds"))
                values = self._lineage(fired, armed)
                if side["breakout"] and self.rules.slippage_bp and not self._held_before.get(sign, False):
                    # On the candle the breakout first holds; charged by a backtest only (_submit), never on exits.
                    values["breakout_slippage_bp"] = self.rules.slippage_bp
                self._why = ("; ".join(w for _, w in fired), values)
                return self._leg
        self._why = ("; ".join(w for _, w in fired), self._lineage(fired)) if fired else None
        return self._leg

    def _time_up(self, step: int) -> bool:
        """The time stop: wall-clock by default, so missing candles never lengthen a hold; it ends the leg on the
        first candle at or after the deadline. A definition can count the candles that came instead, as rsi_cross
        does (Independent Quant Advisor, 6 Oct, 5.2)."""
        if self.rules.time_stop is None:
            return False
        kind, n = self.rules.time_stop
        if kind == "count":
            return self._held >= n
        return self.env.ts >= self._leg_ts + (n * MINUTE_NS if kind == "minutes" else n * step)

    def _held_words(self, step: int) -> str:
        kind, n = self.rules.time_stop
        if kind == "count":
            return f"{self._held} candles"
        return f"{(self.env.ts - self._leg_ts) // MINUTE_NS:,} minutes"

    def want_long(self, bar: Bar) -> bool | None:
        side = self.want_side(bar)
        return None if side is None else side == 1  # spot: the short leg is held flat

    def explain(self, bar: Bar, target) -> tuple[str, dict]:
        if self._why is None:  # the position catches up with a leg already on (an entry held back, say), or flat
            name = {1: "long", -1: "short"}.get(self._leg)
            rule = (f"{name}.entry", f"the {name} leg is on: its entry rule held on an earlier candle") if name else (
                "flat", "no leg is on")
            self._why = (rule[1], self._lineage([rule]))
        reason, values = self._why
        return reason[:1].upper() + reason[1:], dict(values)

    def _lineage(self, fired: list, armed: list | None = None) -> dict:
        """The DA-4 signal payload, v1."""
        blocks = {}
        for bid, b in self.rules.blocks.items():
            outputs = type(b).OUTPUTS
            for out in ((None,) if outputs == ("value",) else outputs):
                v = block_value(b, out)
                if v is not None:
                    blocks[bid if out is None else f"{bid}.{out}"] = v
        step = bar_minutes(self._cfg.bar_type)
        warmup = self.c.checked.warmup

        def span_of(last_open: int, minutes: int, n: int) -> list:
            return [_iso(last_open - (max(n, 1) - 1) * minutes * MINUTE_NS), _iso(last_open)]

        bars = {timeframe_label(step): span_of(self.env.ts - step * MINUTE_NS, step, warmup[None])}
        for tf, s in self._slow.items():
            if s.last is not None:
                bars[timeframe_label(tf)] = span_of(s.last.end - tf * MINUTE_NS, tf, warmup.get(tf, 1))
        source = "hub" if self.hub_fed else "store" if self._backtest else "venue_rest"
        payload = {"v": 1, "definition": self.c.definition_hash, "rule": ", ".join(r for r, _ in fired),
                   "blocks": blocks, "bars": bars, "data": {"source": source, "refilled_in_range": None}}  # unknown until provenance is read (DA-4)
        if armed:
            payload["armed_at"] = _iso(max(armed))
        touched = {n.text: n.lineage() for n in self.rules.touches if n.judged_ts == self.env.ts}
        if touched:
            payload["first_touch"] = touched
        return payload

    # --- level exits ------------------------------------------------------------------------------------------

    @property
    def _has_exits(self) -> bool:
        return bool(self.rules.levels) or super()._has_exits

    def _stop_cfg(self) -> dict:
        if not self.rules.levels:
            return super()._stop_cfg()
        exits = self.c.definition.get("exits") or {}
        return {"definition_exits": {k: exits[k] for k in LEVEL_EXITS if k in exits}}

    def _level_values(self) -> dict | None:
        out = {k: fn(self.env) for k, fn in self.rules.levels.items()}
        return None if any(v is None for v in out.values()) else out

    def _tightest(self, levels: dict, side: int) -> float:
        return max(levels.values()) if side > 0 else min(levels.values())

    def _plan_exits(self, close: float, side: int = 1):
        if not self.rules.levels:
            return super()._plan_exits(close, side)
        levels = self._level_values()
        if levels is None or close <= 0:
            return None
        level = self._tightest(levels, side)
        stop = side * (1 - level / close)
        if stop <= 0:
            # A warning naming the definition: a level that sits on the target side every time (a long's exit at an
            # average above the price, say) would otherwise leave a definition that never trades (Code Reviewer).
            self._note("stop_level_wrong_side", f"Entry held back: the stop level {level:,.6g} is "
                       f"{'at or above' if side > 0 else 'at or below'} the close {close:,.6g}, so it can't be a "
                       f"stop (definition {self.c.definition_hash[:12]}); a level exit only works on the stop side")
            return None
        self._noted.discard("stop_level_wrong_side")
        basis = ", ".join(f"{k.replace('_', ' ')} at {v:,.6g}" for k, v in levels.items())
        clamped = min(max(stop, MIN_STOP), MAX_STOP)
        if clamped != stop:
            basis += f", held to {clamped:.1%}"
            levels = {k: close * (1 - side * clamped) if v == level else v for k, v in levels.items()}
        self._levels = levels
        return clamped, self._target_for(clamped, side), basis

    def _move_levels(self) -> tuple | None:
        """Each close while a position is open: `trail` follows its level (never loosening with `ratchet`),
        `exit_at_level` follows its own, `stop` stays where it opened; the tightest becomes the position's stop
        from the next candle. Returns (kind, level) when this close is already past it."""
        if self._entry_px is None or not self.rules.levels:
            return None
        side = self._entry_side or 1
        now = self._level_values() or {}
        if self._levels is None:  # after a restart: from the stop the journal kept
            if self._stop_frac is None:
                return None
            kept = self._entry_px * (1 - side * self._stop_frac)
            self._levels = {k: kept for k in self.rules.levels}
        for kind in ("trail", "exit_at_level"):
            if kind in now:
                was = self._levels.get(kind)
                ratchet = kind == "trail" and self.rules.ratchet and was is not None
                self._levels[kind] = (max(was, now[kind]) if side > 0 else min(was, now[kind])) if ratchet else now[kind]
        level = self._tightest(self._levels, side)
        close = self.env.ohlcv[3]
        if (close - level) * side <= 0:
            kind = next(k for k, v in self._levels.items() if v == level)
            return kind, level
        frac = side * (1 - level / self._entry_px)
        if frac != self._stop_frac:
            self._stop_frac = frac
            if self._backtest:
                self._rest_exits()  # moves the resting stop
        return None

    def _rest_exits(self) -> None:
        if self._levels is not None and self._entry_px:  # the levels, not the decision close's share of them
            self._stop_frac = (self._entry_side or 1) * (1 - self._tightest(self._levels, self._entry_side or 1)
                                                         / self._entry_px)
        super()._rest_exits()
