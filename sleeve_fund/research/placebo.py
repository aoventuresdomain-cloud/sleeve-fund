"""R-24 (Advisor ruling, 21 Oct 2026 10:49 UK): did a filter on a parent's entries add anything, or would blocking the
same share of the parent's entries at random have done as well?

A filtered variant (V1, V2: no new long in the funding rate's top decile; D: taker flow) is its parent with
conditions added to the entry (filter_of). A signal is a candle on which the parent's entry rule holds, read on every
scored candle whatever the position. The filter's mask is the scored candles its added conditions fail on (for V1 and
V2, decile 10 or no value), read on every candle whatever the parent's entry. Each placebo draw (Advisor 11:28 UK)
shifts that whole mask round the scored window by one seeded offset of at least SHIFT_MIN_DAYS either way, so the
blocked share and the lengths of the filter's runs are kept, and blocks the signals under the shifted mask: the
parent's own rules run, no entry opening on a blocked candle, so the next candle can open, as under the filter. The
share each draw blocks varies a little; the sheet gives its mean and range.

The draws run on a fast path: the definition's own compiled blocks and rules, read on bar closes, as the Rules
strategy reads them. A stop or level resting from the close before ends a trip inside the candle its low (a short's
high) reaches it, filled as the engine fills a resting stop: at the first execution bar to trade through it (its
open when it opened through, else its low), with the engine's stop slippage; with 1-minute execution bars that is the
level, or the open where the candle gapped through it. Rule exits are at the close. Each trip pays the cost per side
random entry charges and a perpetual's stored settled funding while it is open, at the engine's invested share,
compounded. The variant's fast-path return is judged against the draws: its percentile, with the bar at 95; a tie
with a draw is not beating it.

The fast path is calibrated once per variant against the full engine: the parent, the variant and draws 0 to 2
(CALIBRATION_DRAWS), each run both ways. Each must give the same trips (entry and exit candles, side) and a net return
within CALIBRATION_PP of the engine's; any that doesn't makes the row NOT JUDGED. The tolerance is never widened. A
definition the fast path doesn't model (setups, confirmations, first_touch, slower candles, breakout slippage, a
short or reversing leg), or a variant that isn't its parent with entry conditions added (not a filter), is NOT
JUDGED too."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sleeve_fund.research.random_entry import BAR, DRAWS
from sleeve_fund.strategies.base import MAX_STOP, MIN_STOP
from sleeve_fund.strategies.definitions import (Compiled, Confirm, Env, FirstTouch, FundingDecile, Group, Setup,
                                                check_definition)
from sleeve_fund.strategies.timeframes import MINUTE_NS

CALIBRATION_DRAWS = 3  # the gated draws re-run on the full engine, after the parent and the variant
CALIBRATION_PP = 0.005  # absolute: the fast path's net return within half a percentage point of the engine's
NOT_JUDGED = "NOT JUDGED"


@dataclass(frozen=True)
class FastRun:
    trips: list  # (entry candle, exit candle, side), positions in the bars; only trips closed by the last bar
    returns: np.ndarray  # each trip's net return, in order
    stopped: list  # per trip: it ended inside its exit candle (a level reached), else at that candle's close
    holds: np.ndarray  # bool per bar: the entry rule held at that close (read whatever the position)
    held_back: int = 0  # entries the stop level held back (it sat on the wrong side of the close)
    open_return: float = 0.0  # a trip still open at the last bar, marked at its close (entry cost and funding only)

    @property
    def net(self) -> float:
        """The window's net return, compounded, a trip still open marked at the last close as the engine marks it."""
        return float(np.prod(1 + self.returns) * (1 + self.open_return) - 1)

    def matches(self, index: pd.DatetimeIndex, full: list) -> bool:
        """Whether the engine's trips, (opened, closed, side) fill times, are these: each opened by a fill at or after
        its entry candle's close (before the next), and closed so too, or inside its exit candle for a level."""
        if len(full) != len(self.trips):
            return False
        idx = index.as_unit("ns").asi8

        def at_close(t) -> int:
            return int(np.searchsorted(idx, pd.Timestamp(t).value, side="right")) - 1

        def inside(t) -> int:
            return int(np.searchsorted(idx, pd.Timestamp(t).value, side="left"))

        return all(at_close(o) == e and (inside(c) if s else at_close(c)) == x and int(side) == sd
                   for (e, x, sd), s, (o, c, side) in zip(self.trips, self.stopped, full))


@dataclass(frozen=True)
class Calibration:
    run: str  # "parent", "variant", "draw 0" ...
    fast_trips: int
    full_trips: int
    same_trips: bool
    fast_return: float
    full_return: float

    @property
    def gap(self) -> float:
        return abs(self.fast_return - self.full_return)

    @property
    def ok(self) -> bool:
        return self.same_trips and math.isfinite(self.gap) and self.gap <= CALIBRATION_PP


@dataclass
class PlaceboResult:
    signals: int = 0  # candles the parent's entry held on, in the scored bars
    blocked: int = 0  # of those, the filter blocked
    masked: int = 0  # scored candles the filter's own conditions fail on, whatever the parent's entry: its mask
    variant_return: float = math.nan  # fast path
    full_return: float = math.nan  # the variant's on the full engine, for reference
    draws: int = 0
    returns: np.ndarray = field(default_factory=lambda: np.empty(0))
    draw_blocked: np.ndarray = field(default_factory=lambda: np.empty(0))  # signals each draw's shifted mask blocked
    calibration: list = field(default_factory=list)
    why_not: str = ""  # set: NOT JUDGED, and why
    size: float = 1.0  # the share of equity a trip holds, from the variant run's own trips (its instrument and market)
    where: str = ""  # that run's instrument and market, as the sheet names them
    stop_words: str = ""  # how the engine (and so the fast path) fills a resting stop, on this study's execution bars

    @property
    def share(self) -> float:
        return self.blocked / self.signals if self.signals else math.nan

    @property
    def percentile(self) -> float:
        return float((self.returns < self.variant_return).mean() * 100) if len(self.returns) else math.nan

    @property
    def judged(self) -> bool:
        return not self.why_not and all(c.ok for c in self.calibration) and len(self.returns) > 0

    @property
    def passed(self) -> bool:
        return self.judged and self.percentile >= BAR


def unsupported(definition: dict, bar_minutes: int) -> str | None:
    """Why the fast path can't run this definition as the engine does, or None."""
    checked = check_definition(definition, bar_minutes=bar_minutes)
    c = Compiled(checked)
    if any(b["timeframe"] is not None for b in checked.blocks.values()):
        return "it reads slower candles"
    if c.setups or c.touches:
        return "it has a setup or a first_touch rule"
    if c.slippage_bp:
        return "it charges breakout slippage"
    if set(c.sides) != {1}:
        return "it has a short leg"

    def walk(n) -> bool:
        return isinstance(n, (Confirm, Setup, FirstTouch)) or (isinstance(n, Group) and any(walk(x) for x in n.items))

    side = c.sides[1]
    if walk(side["entry"]) or (side["exit"] is not None and walk(side["exit"])):
        return "it has a confirmation"
    return None


def _time_up(kind_n, ts: int, leg_ts: int, held: int, step: int) -> bool:
    if kind_n is None:
        return False
    kind, n = kind_n
    if kind == "count":
        return held >= n
    return ts >= leg_ts + (n * MINUTE_NS if kind == "minutes" else n * step)


def fast_path(definition: dict, prices: pd.DataFrame, bar_minutes: int, trade_from: int, cost_per_side,
              funding: tuple | None = None, blocked=frozenset(), exec_prices: pd.DataFrame | None = None,
              stop_slippage=0.0, size: float = 1.0) -> FastRun:
    """The definition's long leg on `prices` (bars indexed by close time), trading from bar position `trade_from`
    (before it the rules are read, never traded, as a study's warm-up is); no entry opens on a bar position in
    `blocked`. funding: (settlement times ns, rates, interval ns) as the runner hands a funding block, read by such a
    block and charged to a perpetual's trips (long pays a positive rate); None: spot, nothing charged.

    A level reached inside a candle fills as the engine fills a resting stop: at the first execution bar inside it
    (exec_prices; the candle itself without them) to trade through the level, at its open when it opened through,
    else at its low, worse by stop_slippage (the engine's: the larger of half the spread and 0.05%, less the half
    spread cost_per_side already charges). size: the share of equity a trip holds (the engine's invested share)."""
    from sleeve_fund.strategies.rules import _Feed

    checked = check_definition(definition, bar_minutes=bar_minutes)
    rules = Compiled(checked)
    for b in rules.blocks.values():
        if isinstance(b, FundingDecile):
            b.source = funding
    env = Env(blocks=rules.blocks, note=lambda *a: None, journal=None)
    feed = _Feed(rules, list(checked.order))
    side = rules.sides[1]
    exits = rules.exits[1]
    step = bar_minutes * MINUTE_NS
    ts = prices.index.as_unit("ns").asi8 if prices.index.tz is not None else prices.index.tz_localize("UTC").as_unit("ns").asi8
    o, h, lo, c, v = (prices[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    cost = np.broadcast_to(np.asarray(cost_per_side, dtype=float), len(c))
    slip = np.broadcast_to(np.asarray(stop_slippage, dtype=float), len(c))
    f_times, f_rates = (np.asarray(funding[0]), np.asarray(funding[1], dtype=float)) if funding else (None, None)
    if exec_prices is not None and len(exec_prices):
        xi = exec_prices.index
        x_ts = (xi if xi.tz is not None else xi.tz_localize("UTC")).as_unit("ns").asi8
        x_o, x_lo = exec_prices["open"].to_numpy(dtype=float), exec_prices["low"].to_numpy(dtype=float)
        x_at = np.searchsorted(x_ts, ts, side="right")  # exec bars closing in (ts[i-1], ts[i]]: x_at[i-1]:x_at[i]
    else:
        x_ts = None

    def through(i: int, stop: float) -> float:
        """Where a resting stop at `stop` fills inside candle i, whose low reached it."""
        if x_ts is not None:
            for k in range(x_at[i - 1] if i else 0, x_at[i]):
                if x_lo[k] <= stop:
                    return (x_o[k] if x_o[k] <= stop else x_lo[k]) * (1 - slip[i])
        return (o[i] if o[i] <= stop else lo[i]) * (1 - slip[i])

    holds = np.zeros(len(c), dtype=bool)
    trips, rets, stopped, held_back = [], [], [], 0
    leg = held = 0
    leg_ts = None
    pos = None  # (entry bar, entry price, levels dict)

    def trip_return(i: int, px: float, exit_cost: float) -> float:
        e, epx, _ = pos
        r = px / epx - 1 - cost[e] - exit_cost
        if f_times is not None:
            a, b = np.searchsorted(f_times, [ts[e], ts[i]], side="right")
            r -= float(f_rates[a:b].sum())
        return size * r

    def close_trip(i: int, px: float, level: bool = False) -> None:
        nonlocal pos
        e, r = pos[0], trip_return(i, px, cost[i])
        trips.append((e, i, 1))
        rets.append(r)
        stopped.append(level)
        pos = None

    for i in range(len(c)):
        if pos is not None and pos[2]:  # a level resting from the close before, reached inside this candle
            stop = max(pos[2].values())
            if lo[i] <= stop:
                close_trip(i, through(i, stop), True)
                leg, held, leg_ts = 0, 0, None  # exit_leg_closed: the entry rules are read afresh at this close
        if env.bar:
            rules.record(env)
        feed.update_ohlcv(o[i], h[i], lo[i], c[i], v[i], int(ts[i]))
        env.ohlcv, env.ts, env.bar, env.closed = (o[i], h[i], lo[i], c[i], v[i]), int(ts[i]), env.bar + 1, set()
        side["entry"].tick(env)
        if side["exit"] is not None:
            side["exit"].tick(env)
        holds[i] = side["entry"].test(env)
        ended = 0
        if leg:
            if leg_ts is None:
                leg_ts = env.ts - (held + 1) * step
            held += 1
            rules.carry_leg(env)
            past = False
            if pos is not None and exits["levels"]:
                now = {k: fn(env) for k, fn in exits["levels"].items()}
                for kind in ("trail", "exit_at_level"):
                    if kind in now and now[kind] is not None:
                        was = pos[2].get(kind)
                        ratchet = kind == "trail" and exits["ratchet"] and was is not None
                        pos[2][kind] = max(was, now[kind]) if ratchet else now[kind]
                past = c[i] - max(pos[2].values()) <= 0
            if (side["exit"] is not None and side["exit"].test(env)) or _time_up(exits["time_stop"], env.ts, leg_ts,
                                                                                held, step) or past:
                ended, leg = 1, 0
        if not leg and not ended and holds[i] and i not in blocked:
            side["entry"].consume()
            leg, held, leg_ts = 1, 0, env.ts
            rules.start_leg(env)
        if i < trade_from:
            continue
        if pos is not None and not leg:
            close_trip(i, c[i])
        if leg and pos is None:
            levels = {k: fn(env) for k, fn in exits["levels"].items()}
            if any(x is None for x in levels.values()):
                held_back += 1
                continue
            if levels:
                level = max(levels.values())
                frac = 1 - level / c[i]
                if frac <= 0:
                    held_back += 1
                    continue
                clamped = min(max(frac, MIN_STOP), MAX_STOP)
                if clamped != frac:
                    levels = {k: c[i] * (1 - clamped) if x == level else x for k, x in levels.items()}
            pos = (i, c[i], levels)
    still = trip_return(len(c) - 1, c[-1], 0.0) if pos is not None else 0.0
    return FastRun(trips, np.array(rets, dtype=float), stopped, holds, held_back, still)


SHIFT_MIN_DAYS = 30  # a draw shifts the filter's mask by at least this much either way round the window


def filter_of(parent: dict, variant: dict) -> dict | None:
    """The definition whose entry is the variant's added conditions alone, when the variant is its parent with
    conditions added to the long entry (an `all` holding the parent's entry, or each of its own `all` items), and
    otherwise the same; None when it isn't such a filter."""
    pe, ve = parent["long"]["entry"], variant["long"]["entry"]
    if not isinstance(ve, dict) or "all" not in ve:
        return None
    items = list(ve["all"])
    own = list(pe["all"]) if isinstance(pe, dict) and "all" in pe and pe not in items else [pe]
    if any(x not in items for x in own):
        return None
    extra = [x for x in items if x not in own]
    if not extra:
        return None
    rest = {k: v for k, v in variant.items() if k not in ("long", "blocks", "name", "title", "reason")}
    if rest != {k: v for k, v in parent.items() if k not in ("long", "blocks", "name", "title", "reason")} or \
            {k: v for k, v in variant["long"].items() if k != "entry"} != \
            {k: v for k, v in parent["long"].items() if k != "entry"} or \
            any(variant.get("blocks", {}).get(k) != b for k, b in parent.get("blocks", {}).items()):
        return None
    out = copy.deepcopy(variant)
    out["long"]["entry"] = extra[0] if len(extra) == 1 else {"all": extra}
    return out


def shift_offsets(scored: int, bar_minutes: int, draws: int, seed: int) -> list[int]:
    """Each draw's shift of the mask, in candles: at least SHIFT_MIN_DAYS either way round the scored window, draw d
    from its own seeded stream (seed, d)."""
    k = -(-SHIFT_MIN_DAYS * 1440 // bar_minutes)
    if scored < 2 * k + 1:
        raise ValueError(f"the scored window is under {2 * SHIFT_MIN_DAYS} days: no shift of at least "
                         f"{SHIFT_MIN_DAYS} days either way")
    return [int(np.random.default_rng([seed, d]).integers(k, scored - k + 1)) for d in range(draws)]


def shifted_blocks(mask: np.ndarray, holds: np.ndarray, trade_from: int, offset: int) -> frozenset:
    """The bar positions a draw blocks: the filter's mask over the scored candles, shifted round them by `offset`,
    where the parent's entry holds."""
    scored = np.zeros(len(mask), dtype=bool)
    scored[trade_from:] = np.roll(mask[trade_from:], offset)
    return frozenset(int(i) for i in np.flatnonzero(scored & holds))


def placebo(parent: dict, variant: dict, prices: pd.DataFrame, bar_minutes: int, trade_from: int, cost_per_side,
            funding: tuple | None = None, full=None, draws: int = DRAWS, seed: int = 0, **fast) -> PlaceboResult:
    """The variant against `draws` placebo draws of its parent. full(definition, blocked bar positions) -> (trips,
    net return): the same window on the full engine, its trips as (opened, closed, side) fill times, for the
    calibration; None: not calibrated, so NOT JUDGED. fast: fast_path's execution bars, stop slippage and size."""
    out = PlaceboResult()
    why = unsupported(parent, bar_minutes) or unsupported(variant, bar_minutes)
    if why:
        out.why_not = f"the fast path doesn't model this definition: {why}"
        return out
    only = filter_of(parent, variant)
    if only is None:
        out.why_not = ("the variant isn't its parent with conditions added to the entry, so it isn't a filter on the "
                       "parent's entries")
        return out
    p = fast_path(parent, prices, bar_minutes, trade_from, cost_per_side, funding, **fast)
    var = fast_path(variant, prices, bar_minutes, trade_from, cost_per_side, funding, **fast)
    mask = ~fast_path(only, prices, bar_minutes, trade_from, cost_per_side, funding, **fast).holds  # blocks, every candle
    scored = np.arange(len(prices)) >= trade_from
    out.signals, out.blocked = int((p.holds & scored).sum()), int((p.holds & mask & scored).sum())
    out.masked = int((mask & scored).sum())
    out.variant_return = var.net
    if (var.holds & ~p.holds & scored).any() or (var.holds != (p.holds & ~mask))[scored].any():
        out.why_not = "the variant opens on candles its parent doesn't: it isn't a filter on the parent's entries"
        return out
    if p.held_back or var.held_back:
        out.why_not = "an entry was held back (a stop level on the wrong side of the close), which the fast path skips"
        return out
    try:
        offsets = shift_offsets(int(scored.sum()), bar_minutes, draws, seed)
    except ValueError as exc:
        out.why_not = str(exc)
        return out
    blocks = [shifted_blocks(mask, p.holds, trade_from, o) for o in offsets]
    runs = [fast_path(parent, prices, bar_minutes, trade_from, cost_per_side, funding, b, **fast) for b in blocks]
    out.draws, out.returns = draws, np.array([r.net for r in runs])
    out.draw_blocked = np.array([len(b) for b in blocks])
    if full is None:
        out.why_not = "not calibrated against the full engine"
        return out
    for name, defn, run, blocked in [("parent", parent, p, frozenset()), ("variant", variant, var, frozenset())] + [
            (f"draw {d}", parent, runs[d], blocks[d]) for d in range(min(CALIBRATION_DRAWS, draws))]:
        trips, net = full(defn, blocked)
        out.calibration.append(Calibration(name, len(run.trips), len(trips), run.matches(prices.index, trips),
                                           run.net, float(net)))
        if name == "variant":
            out.full_return = float(net)
    if not all(c.ok for c in out.calibration):
        out.why_not = (f"the fast path didn't match the full engine (same trips, net return within "
                       f"{CALIBRATION_PP * 100:.1f} pp) on every calibration run")
    return out


def section(r: PlaceboResult | None) -> list[str]:
    """The tear sheet's section for a filtered variant; none for any other study."""
    if r is None:
        return []

    def pct(x: float) -> str:
        return f"{x:+.2%}" if math.isfinite(x) else "n/a"

    head = ["## The filter against a placebo (the same share of the parent's entries blocked at random)", ""]
    if r.why_not:
        verdict = f"**Placebo: {NOT_JUDGED}**: {r.why_not}."
    else:
        verdict = (f"**Placebo: {'PASS' if r.passed else 'FAIL'}**: the variant's return beats {r.percentile:.0f}% of "
                   f"{r.draws:,} draws (the bar is the {BAR:.0f}th percentile).")
    lines = head + [verdict, ""]
    if r.signals:
        drawn = r.draw_blocked / r.signals if len(r.draw_blocked) else np.empty(0)
        lines += [f"The parent's entry rule held on {r.signals:,} scored candles; the filter blocked {r.blocked:,} of "
                  f"them ({r.share:.1%}). Each draw shifts the filter's mask ({r.masked:,} candles) round the scored "
                  f"window by a seeded offset of at least {SHIFT_MIN_DAYS} days, keeping its share and its runs, and "
                  "runs the parent's rules with the signals under it blocked"
                  + (f": they blocked {drawn.mean():.1%} on average ({drawn.min():.1%} to {drawn.max():.1%})"
                     if len(drawn) else "")
                  + f". Fast path (bar closes): the variant {pct(r.variant_return)}, the full engine "
                  f"{pct(r.full_return)}"
                  + (f"; the draws' median {pct(float(np.median(r.returns)))}, their 95th percentile "
                     f"{pct(float(np.percentile(r.returns, BAR)))}." if len(r.returns) else "."), ""]
    if r.where or r.stop_words:
        lines += [f"Priced as the engine books it: each trip holds {r.size:.1%} of equity, the variant's own run's "
                  f"invested share on {r.where or 'its instrument'}. {r.stop_words} A trip still open at the "
                  "window's end is marked at its last close, as the engine marks it, for the variant and every "
                  "draw alike.".replace("  ", " "), ""]
    if r.calibration:
        lines += [f"Calibration: each run both ways, the same trips and a net return within {CALIBRATION_PP * 100:.1f} "
                  "pp of the full engine's, or the row is not judged.", "",
                  "| Run | Trips (fast / full) | Same trips | Net return (fast / full) | Gap | Matches |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {c.run} | {c.fast_trips} / {c.full_trips} | {'yes' if c.same_trips else 'no'} | "
                  f"{pct(c.fast_return)} / {pct(c.full_return)} | {c.gap * 100:.2f} pp | {'yes' if c.ok else 'no'} |"
                  for c in r.calibration]
        lines.append("")
    return lines
