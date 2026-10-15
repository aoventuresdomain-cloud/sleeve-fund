"""A10 (Independent Quant Advisor, 15 Oct 07:02 UK): a recursive indicator is checked at every warm-up by seeding it
twice. A Wilder or exponential average never forgets where it started, only discounts it, so one seeded N bars back
and one seeded 2N bars back read the same only once the start no longer shows. They are compared at the last loaded
bar against a tolerance per indicator: RSI 0.1 points; an EMA, and what is built on one, 1 bp of its value; Wilder's
ATR 0.5% of its value. Apart, both seeds move further back (2N and the most the store holds, up to CAP bars); still
apart, or the store too short, the Strategy opens nothing new until a pair carried on over live bars agrees. Window
indicators (a simple average, a channel) read the same from any start once their window is full, so need no check.

A unit is what is seeded: one indicator a model holds, or a rule-builder timeframe's chained blocks (anything with
update_ohlcv). Its recursive leaves are found by walking into the blocks it holds, so a Keltner channel is checked
on its EMA and its ATR, an RSI divergence on its RSI."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from nautilus_trader.indicators import ExponentialMovingAverage

from sleeve_fund.strategies.indicators._common import BlockBase, settle_bars
from sleeve_fund.strategies.indicators.averages import Atr, Ema, Vwap, Wma
from sleeve_fund.strategies.indicators.bands import Bollinger, Donchian
from sleeve_fund.strategies.indicators.classic import AtrSma, Rsi, Sma
from sleeve_fund.strategies.indicators.filters import EfficiencyRatio, RelativeVolume
from sleeve_fund.strategies.indicators.oscillators import Stochastic

CAP = 1_000  # the most bars a check seeds from (Advisor): 2N above it compares CAP/2 with CAP
POINTS, RELATIVE = "points", "relative"
TOLERANCE: dict[type, tuple[str, float]] = {
    Rsi: (POINTS, 0.1),
    Ema: (RELATIVE, 1e-4),
    ExponentialMovingAverage: (RELATIVE, 1e-4),
    Atr: (RELATIVE, 5e-3),
}
# Read the same from any start once their window is full: nothing to check.
WINDOW: tuple[type, ...] = (Sma, AtrSma, Wma, Vwap, Bollinger, Donchian, EfficiencyRatio, RelativeVolume, Stochastic)

Candle = tuple  # (open, high, low, close, volume, ts_ns)


def _engine(x) -> bool:
    """One of the engine's own indicators (compiled: no attributes to walk, no reset to copy from)."""
    return type(x).__module__.startswith("nautilus_trader.indicators")


def name(leaf) -> str:
    period = getattr(leaf, "period", None)
    kind = {Rsi: "RSI", Ema: "EMA", ExponentialMovingAverage: "EMA", Atr: "ATR"}.get(type(leaf), type(leaf).__name__)
    return f"{kind}({period})" if period is not None else kind


def _indicator(x) -> bool:
    return isinstance(x, BlockBase) or _engine(x) or (hasattr(x, "update_raw") and hasattr(x, "initialized"))


def _members(obj) -> list:
    """The indicators `obj` holds: a rule-builder feed's blocks in order, else the indicators among its attributes."""
    if hasattr(obj, "nodes"):
        return [b for b, _ in obj.nodes]
    if _engine(obj):
        return []
    return [v for v in vars(obj).values() if _indicator(v)]


def leaves(obj) -> list:
    """The recursive indicators inside `obj`, each once, in the order they are fed."""
    if type(obj) in TOLERANCE:
        return [obj]
    out = []
    for m in _members(obj):
        out += [x for x in leaves(m) if all(x is not y for y in out)]
    return out


def uncovered(obj) -> list[str]:
    """Indicators inside `obj` the check can't judge: neither a known recursive kind nor a window one, and holding
    no indicators of their own to judge it by. The start check names them (Advisor: MUST before paper)."""
    if type(obj) in TOLERANCE or isinstance(obj, WINDOW):
        return []
    members = _members(obj)
    if not members:
        return [name(obj)] if _indicator(obj) else []
    return [u for m in members for u in uncovered(m)]


def need(obj) -> int:
    """Bars the unit's recursive leaves take to settle: N, the nearer seed."""
    if hasattr(obj, "warmup_bars") and not _engine(obj):
        return int(obj.warmup_bars)
    return max((int(x.warmup_bars) if isinstance(x, BlockBase) else settle_bars(x.period) for x in leaves(obj)),
               default=0)


class Group:
    """Blocks fed the same candles side by side, as slower candles feed theirs: one unit to check."""

    def __init__(self, blocks, warmup_bars: int) -> None:
        self.nodes = [(b, None) for b in blocks]
        self.warmup_bars = warmup_bars

    def update_ohlcv(self, *candle) -> None:
        for b, _ in self.nodes:
            b.update_ohlcv(*candle)


def _reset(obj) -> None:
    if hasattr(obj, "reset"):
        obj.reset()  # a block's reset builds its own parts afresh
    else:
        for m in _members(obj):
            _reset(m)


def fresh(obj):
    """A copy of the unit with nothing fed."""
    if _engine(obj):
        return type(obj)(obj.period)
    out = copy.deepcopy(obj)
    _reset(out)
    return out


def feed(obj, c: Candle) -> None:
    if _engine(obj):
        obj.update_raw(c[3])
    else:
        obj.update_ohlcv(*c)


def read(leaf) -> float | None:
    """The leaf's value now, whether or not it has passed its own warm-up; None before its first one."""
    if isinstance(leaf, Rsi) or _engine(leaf):
        return leaf.value if leaf.initialized else None
    return leaf._value if leaf.count else None


@dataclass(frozen=True)
class Gap:
    label: str
    gap: float  # points, or a share of the value
    kind: str
    tolerance: float

    @property
    def within(self) -> bool:
        return self.gap <= self.tolerance

    def __str__(self) -> str:
        if self.kind == POINTS:
            return f"{self.gap:.2f} points apart (tolerance {self.tolerance:g})"
        return f"{self.gap * 100:.3f}% apart (tolerance {self.tolerance * 100:g}%)"


def worst(a, b) -> Gap | None:
    """The leaf furthest from its tolerance between two seedings of one unit; None when every one agrees. A leaf with
    no value yet in either is apart without bound."""
    out = None
    for x, y in zip(leaves(a), leaves(b), strict=True):
        kind, tol = TOLERANCE[type(x)]
        vx, vy = read(x), read(y)
        if vx is None or vy is None:
            g = math.inf
        elif kind == POINTS:
            g = abs(vx - vy)
        else:
            g = abs(vx - vy) / abs(vy) if vy else (0.0 if vx == vy else math.inf)
        gap = Gap(name(x), g, kind, tol)
        if not gap.within and (out is None or g / tol > out.gap / out.tolerance):
            out = gap
    return out


@dataclass
class Pair:
    """Two seedings of a unit carried on over live candles until they agree: `near` started later than `far`.
    Settled once they agree and `near` has had the unit's N candles, so two that started a bar apart can't agree
    by chance before either has settled."""

    near: object
    far: object
    fed: int  # candles `near` has had
    n: int
    last: Gap | None = None

    def update(self, c: Candle) -> bool:
        feed(self.near, c)
        feed(self.far, c)
        self.fed += 1
        return self.settled

    @property
    def settled(self) -> bool:
        self.last = worst(self.near, self.far)
        return self.last is None and self.fed >= self.n


def _seeded(obj, candles) -> object:
    out = fresh(obj)
    for c in candles:
        feed(out, c)
    return out


@dataclass
class Result:
    """One unit's check: settled, the two seeds compared (bars back), the bars the model itself should be fed so
    its own reading is the nearer agreeing seed's or better, the gap when apart, why when the store is too short,
    and the pair to carry on live while not settled."""

    label: str
    settled: bool
    seeds: tuple[int, int]
    feed_bars: int
    gap: Gap | None = None
    short: str | None = None
    pair: Pair | None = field(default=None, repr=False)

    def why(self) -> str:
        if self.short:
            return f"{self.label}: {self.short}"
        which = "" if self.gap.label == self.label else f"{self.gap.label} "
        return f"{self.label} seeded {self.seeds[0]} and {self.seeds[1]} bars back reads {which}{self.gap}"


def check(obj, candles: list, label: str, cap: int = CAP) -> Result:
    """Seed `obj` N and 2N candles back (N its settle, both within cap) and compare at the last candle; apart, move
    both back (2N and the most there is, up to cap) and compare again."""
    n = need(obj)
    if not leaves(obj) or n == 0:
        return Result(label, True, (0, 0), 0)
    n1 = min(n, cap // 2)
    n2 = min(2 * n1, cap)
    have = min(len(candles), cap)
    if have < n2:
        near_from = have // 2
        near = _seeded(obj, candles[len(candles) - near_from:] if near_from else [])
        pair = Pair(near, _seeded(obj, candles[len(candles) - have:]), near_from, n1)
        short = (f"the history store holds {have} bars, and the check seeds it {n1} and {n2} bars back, so it isn't "
                 "known to have settled")
        return Result(label, False, (n1, n2), have, short=short, pair=pair)
    near, far = _seeded(obj, candles[-n1:]), _seeded(obj, candles[-n2:])
    gap = worst(near, far)
    if gap is None:
        return Result(label, True, (n1, n2), n1)
    if have > n2:
        longest = _seeded(obj, candles[-have:])
        again = worst(far, longest)
        if again is None:
            return Result(label, True, (n2, have), n2)
        return Result(label, False, (n2, have), have, gap=again, pair=Pair(far, longest, n2, n1))
    return Result(label, False, (n1, n2), n2, gap=gap, pair=Pair(near, far, n1, n1))
