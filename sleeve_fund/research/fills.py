"""RANDOM-ENTRY (HoE 14:08 and 14:43 UK 21 Oct; Advisor): a trip priced as the engine fills it, for random-entry draws.

A draw enters at a decision candle's close, as the strategy's market entries do, and holds for one of the strategy's
own holding periods at most. While it holds, the definition's own level exits (stop, trail, exit_at_level) rest in
each next candle at the level set at the close before it, and fill by the engine's own function, never a copy
(Advisor, binding for PR E): strategies.base.replay_missed over the candle's execution bars (the candle itself
without them), the stop first, then a perpetual's liquidation, each booked as it books them. A close beyond a level
ends the trip at that close, as Rules does. Funding is charged at the stored settlement times the trip holds through.
A draw is priced on its entry notional, as the strategy's realised trip returns are (pnl over entry notional).

Exits it does not model are refused in words (unmodelled), never priced on closes: a take_profit (TP-EXIT, added once
built), a level that reads a slower timeframe's block, or one that reads a since-entry value through prev.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sleeve_fund.instruments import taker_slippage
from sleeve_fund.markets import LOW_FEE_PERP, liquidation_price
from sleeve_fund.strategies.base import MAX_STOP, MIN_STOP, replay_missed
from sleeve_fund.strategies.definitions import (ARITHMETIC, FIELDS, LEVEL_EXITS, SINCE_ENTRY, Compiled, Env,
                                                check_definition, side_exits)

MODELLED_EXITS = frozenset({"time_stop", *LEVEL_EXITS})  # what a draw's exits may hold; the time stop is the hold cap
NAMES = {1: "long", -1: "short"}


def _walk(x):
    """Every operand node of a level expression, depth first."""
    yield x
    if isinstance(x, dict):
        (key, arg), = x.items() if "time" not in x else (("time", None),)
        if key in ARITHMETIC:
            for a in arg:
                yield from _walk(a)
        elif key != "time":
            yield from _walk(arg)


def _has_since(x) -> bool:
    return any(isinstance(n, dict) and next(iter(n)) in SINCE_ENTRY for n in _walk(x))


def _levels_of(definition: dict, side: int) -> tuple[dict, bool]:
    """A side's level exits as {kind: level expression}, and whether its trail ratchets."""
    exits = side_exits(definition.get("exits"), NAMES[side])
    return ({k: exits[k]["level"] for k in LEVEL_EXITS if k in exits},
            bool((exits.get("trail") or {}).get("ratchet", True)))


def unmodelled(definition: dict, bar_minutes: int) -> str | None:
    """Why a draw can't be priced as the engine fills this definition's exits, or None."""
    checked = check_definition(definition, bar_minutes=bar_minutes)
    defn = checked.definition
    for name in ("long", "short"):
        if name not in defn:
            continue
        extra = set(side_exits(defn.get("exits"), name)) - MODELLED_EXITS
        if extra:
            return f"its {name} side has a {', '.join(sorted(extra))} exit, which the draws don't model yet"
        levels, _ = _levels_of(defn, 1 if name == "long" else -1)
        for level in levels.values():
            for n in _walk(level):
                if isinstance(n, str) and n not in FIELDS:
                    bid = n.partition(".")[0]
                    if checked.blocks[bid]["timeframe"] is not None:
                        return f"its {name} exit level reads {bid}, a slower timeframe's block"
                if isinstance(n, dict) and "prev" in n and _has_since(n["prev"]):
                    return f"its {name} exit level reads a since-entry value through prev"
    return None


def _nan(v) -> float:
    return np.nan if v is None else float(v)


def _arith(key: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == "add":
            out = a + b
        elif key == "sub":
            out = a - b
        elif key == "mul":
            out = a * b
        else:
            out = np.where(b != 0, a / np.where(b != 0, b, 1.0), np.nan)
    return np.where(np.isfinite(out), out, np.nan)


@dataclass
class DrawFills:
    """The draws' trips, priced. held_back: draws whose entry the engine would have held back (a level not yet
    known, or on the wrong side of the close), priced flat; stopped: those that filled at a level inside a candle."""

    returns: np.ndarray
    held_back: int
    stopped: int


class Pricer:
    """A definition's level exits over `prices` (decision candles indexed by close time), for pricing draws.

    cost_per_side: the fee and half spread a market fill pays, one figure or one per candle (SPREAD-PIT); half_spread:
    its half spread, the same way. A market entry or a close exit pays cost_per_side, as the engine's do; a stop's exit
    pays the taker's slippage in place of the half spread, max(half spread, 0.05%), as the engine's stop does (D13;
    Advisor, PR E round 1).
    exec_prices: the execution bars a level is matched on (the candle itself without them). funding:
    (settlement times ns, rates, baseline mask), each settlement's rate as the engine charges it, a missing or
    simulated one at the baseline both sides pay; None: spot."""

    def __init__(self, definition: dict, prices: pd.DataFrame, bar_minutes: int, cost_per_side,
                 exec_prices: pd.DataFrame | None = None, funding: tuple | None = None, half_spread=0.0) -> None:
        from sleeve_fund.strategies.rules import _Feed

        why = unmodelled(definition, bar_minutes)
        if why:
            raise ValueError(f"random-entry draws can't be filled as the engine fills this definition: {why}")
        checked = check_definition(definition, bar_minutes=bar_minutes)
        self.compiled = Compiled(checked)
        defn = checked.definition
        self.sides = {s: _levels_of(defn, s) for s in (1, -1) if NAMES[s] in defn}
        idx = prices.index if prices.index.tz is not None else prices.index.tz_localize("UTC")
        self.ts = idx.as_unit("ns").asi8
        self.o, self.h, self.lo, self.c, v = (prices[k].to_numpy(dtype=float)
                                              for k in ("open", "high", "low", "close", "volume"))
        n = len(self.c)
        self.cost = np.broadcast_to(np.asarray(cost_per_side, dtype=float), n)
        self.half = np.broadcast_to(np.asarray(half_spread, dtype=float), n)
        if exec_prices is not None and len(exec_prices):
            xi = exec_prices.index if exec_prices.index.tz is not None else exec_prices.index.tz_localize("UTC")
            self.x_ts = xi.as_unit("ns").asi8
            self.x = tuple(exec_prices[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
            self.x_at = np.searchsorted(self.x_ts, self.ts, side="right")  # exec bars closing in (ts[i-1], ts[i]]
        else:
            self.x = None
        if funding is not None:
            self.f_times = np.asarray(funding[0], dtype=np.int64)
            self.f_rates = np.asarray(funding[1], dtype=float)
            self.f_base = np.asarray(funding[2], dtype=bool)
        else:
            self.f_times = None
        # One pass over the candles: every since-free operand of every level, at each close.
        self._full: dict[str, np.ndarray] = {}
        leaves: dict[str, object] = {}
        for levels, _ in self.sides.values():
            for level in levels.values():
                for node in _walk(level):
                    if self._is_leaf(node):
                        leaves.setdefault(json.dumps(node, sort_keys=True), node)
        fns = {k: self.compiled.operand(node)[0] for k, node in leaves.items()}
        decision = [b for b in checked.order if checked.blocks[b]["timeframe"] is None]
        feed = _Feed(self.compiled, decision)
        env = Env(blocks=self.compiled.blocks)
        values = {k: np.empty(n) for k in fns}
        for i in range(n):
            feed.update_ohlcv(self.o[i], self.h[i], self.lo[i], self.c[i], v[i], int(self.ts[i]))
            env.ohlcv, env.ts = (self.o[i], self.h[i], self.lo[i], self.c[i], v[i]), int(self.ts[i])
            for k, fn in fns.items():
                values[k][i] = _nan(fn(env))
        self._leaf = values

    def at_cost(self, cost_per_side, half_spread=None) -> "Pricer":
        """This pricer with another fee and half spread per side (a cost-ladder rung), its level pass shared."""
        other = copy.copy(self)
        other.cost = np.broadcast_to(np.asarray(cost_per_side, dtype=float), len(self.c))
        if half_spread is not None:
            other.half = np.broadcast_to(np.asarray(half_spread, dtype=float), len(self.c))
        return other

    def _stop_cost(self, j: int) -> float:
        """A stop exit's cost in candle j: the fee, and the taker's slippage in place of the half spread."""
        return float(self.cost[j] - self.half[j] + float(taker_slippage(self.half[j])))

    @staticmethod
    def _is_leaf(node) -> bool:
        return not isinstance(node, dict) or "time" in node

    def _series(self, node, lo: int, hi: int) -> np.ndarray:
        """node's value at closes lo..hi (inclusive) of a leg entered at lo's close: a since-entry value runs from
        the entry candle, as Rules.start_leg and carry_leg keep it."""
        if self._is_leaf(node):
            if isinstance(node, (int, float)):
                return np.full(hi - lo + 1, float(node))
            return self._leaf[json.dumps(node, sort_keys=True)][lo:hi + 1]
        (key, arg), = node.items()
        if key in ARITHMETIC:
            return _arith(key, self._series(arg[0], lo, hi), self._series(arg[1], lo, hi))
        if key == "prev":  # since-free (unmodelled refuses the rest): the value at the close before
            inner = self._series(arg, max(lo - 1, 0), hi)
            return inner[:-1] if lo > 0 else np.r_[np.nan, inner[:-1]]
        inner = self._series(arg, lo, hi)  # max_since_entry / min_since_entry, skipping candles without a value
        acc = np.fmax.accumulate if SINCE_ENTRY[key] is max else np.fmin.accumulate
        return acc(inner)

    def _exit_in(self, i: int, venue: dict, side: int):
        """The engine's exit inside candle i for levels resting in `venue` ({"stop_loss", "liquidation"}), as
        replay_missed books it over the candle's execution bars, or None. Only a candle whose own low (a long; high,
        a short) reached a level is replayed: none of its bars can reach one it didn't."""
        reach = self.lo[i] if side > 0 else self.h[i]
        if not any(v is not None and side * (reach - v) <= 0 for v in venue.values()):
            return None
        if self.x is not None and self.x_at[i] > (self.x_at[i - 1] if i else 0):
            k = slice(self.x_at[i - 1] if i else 0, self.x_at[i])
            x_o, x_h, x_lo = self.x
            minutes = zip(self.x_ts[k], x_o[k], x_h[k], x_lo[k])
        else:
            minutes = [(self.ts[i], self.o[i], self.h[i], self.lo[i])]
        return replay_missed(minutes, side, venue, {}, None)

    def _funding(self, e: int, x: int, side: int) -> float:
        """The settlements a trip held from candle e's close to candle x's, over its entry notional: a long pays a
        positive rate, a short receives it, and a missing (baseline) rate is paid either way."""
        if self.f_times is None or x <= e:
            return 0.0
        a, b = np.searchsorted(self.f_times, [self.ts[e], self.ts[x]], side="right")
        rates, base = self.f_rates[a:b], self.f_base[a:b]
        return float(np.where(base, np.abs(rates), side * rates).sum())

    def trip(self, e: int, hold: int, side: int, leverage: float | None = None) -> tuple[float, str]:
        """One draw entered at candle e's close and held `hold` candles at most: its return on entry notional, and
        how it ended ("held back", "stopped", "liquidated", "closed"): every exit records its kind."""
        levels, ratchet = self.sides.get(side, ({}, True))
        end = min(e + hold, len(self.c) - 1)
        epx, x, px, how = self.c[e], end, self.c[end], "closed"
        liq = None
        if leverage is not None:
            trigger = liquidation_price(1 / leverage - side, side, LOW_FEE_PERP.maintenance_margin)
            liq = None if trigger is None else epx * trigger
        now: dict = {}
        if levels:
            paths = {k: self._series(level, e, end) for k, level in levels.items()}
            now = {k: p[0] for k, p in paths.items()}
            if any(np.isnan(v) for v in now.values()):
                return 0.0, "held back"
            bind = max(now.values()) if side > 0 else min(now.values())
            frac = 1 - bind / epx if side > 0 else bind / epx - 1
            if frac <= 0:
                return 0.0, "held back"
            clamped = min(max(frac, MIN_STOP), MAX_STOP)
            if clamped != frac:
                at = epx * (1 - clamped) if side > 0 else epx * (1 + clamped)
                now = {k: at if v == bind else v for k, v in now.items()}
        for j in range(e + 1, end + 1):
            rest = (max(now.values()) if side > 0 else min(now.values())) if now else None
            hit = self._exit_in(j, {"stop_loss": rest, "liquidation": liq}, side)
            if hit is not None:
                intent, booked = hit[0], hit[1]
                if intent == "liquidation":
                    # Booked at the bankruptcy price, as random_entry books one: the margin, and the fees.
                    return -1 / leverage - self.cost[e] - self.cost[j] * (liq / epx), "liquidated"
                x, px, how = j, booked, "stopped"
                break
            if now:
                for kind in ("trail", "exit_at_level"):
                    if kind in paths and not np.isnan(paths[kind][j - e]):
                        v = paths[kind][j - e]
                        if kind == "trail" and ratchet:
                            v = max(now[kind], v) if side > 0 else min(now[kind], v)
                        now[kind] = v
                beyond = max(now.values()) if side > 0 else min(now.values())
                if (self.c[j] <= beyond) if side > 0 else (self.c[j] >= beyond):
                    x, px = j, self.c[j]
                    break
        out = self._stop_cost(x) if how == "stopped" else self.cost[x]
        return float(side * (px / epx - 1) - self.cost[e] - out - self._funding(e, x, side)), how

    def draws(self, entries: np.ndarray, holds: np.ndarray, sides: np.ndarray, levs: np.ndarray | None = None
              ) -> DrawFills:
        out = np.empty(len(entries))
        held_back = stopped = 0
        for k in range(len(entries)):
            lev = None if levs is None or np.isnan(levs[k]) else float(levs[k])
            out[k], how = self.trip(int(entries[k]), int(holds[k]), int(sides[k]), lev)
            held_back += how == "held back"
            stopped += how == "stopped"
        return DrawFills(out, held_back, stopped)
