"""What every indicator block shares: its settings spec, warm-up, the None-until-ready outputs and peeking."""

from __future__ import annotations

import copy
import functools
import math
from dataclasses import dataclass

# Wilder's averages (RSI, ATR) and exponential ones (EMA) never forget their starting value, only discount it:
# after k bars it still weighs (1 - 1/n)^k for Wilder, (1 - 2/(n+1))^k for an EMA. Ten lengths take it below
# 0.01% for RSI(14), so a strategy started on that much history reads what a long-running one would (PM, 5 Oct
# 2026: "if bare minimum it requires a warm up of 140+ then it requires 140+"). A simple average needs only its
# own length.
SETTLE_LENGTHS = 10
# Longest lookback a definition may ask for: a 200-period filter on 4h candles built from minute bars is 48,000.
PERIOD_MAX = 50_000


def settle_bars(period: int) -> int:
    """Bars of history a Wilder or exponential average of `period` needs before it reads as a settled one."""
    return SETTLE_LENGTHS * int(period)


def whole(name: str, value, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value < least:
        raise ValueError(f"{name} must be a whole number, at least {least}, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class Setting:
    """One setting of a block, for definition schema checks and counting tunable settings (P1-6). `default` None
    means the setting is optional; `choices` lists the allowed values of a text setting."""

    name: str
    type: type
    default: object = None
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None

    def check(self, value):
        if self.type is int:
            value = whole(self.name, value, int(self.min) if self.min is not None else 0)
        elif self.type is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{self.name} must be a number, got {value!r}")
            value = float(value)
        elif not isinstance(value, self.type):
            raise ValueError(f"{self.name} must be {self.type.__name__}, got {value!r}")
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"{self.name} must be one of {', '.join(map(str, self.choices))}, got {value!r}")
        if self.min is not None and value < self.min:
            raise ValueError(f"{self.name} must be at least {self.min:g}, got {value!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{self.name} must be at most {self.max:g}, got {value!r}")
        return value


class _Warmup:
    """`warmup_bars` read on a block gives its own warm-up; called on the class with settings, e.g.
    Ema.warmup_bars(period=20), it gives the warm-up those settings would need, before any block exists."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __get__(self, obj, cls):
        if obj is None:
            return lambda **settings: self.fn(cls, cls.check_settings(settings))
        return self.fn(cls, obj.settings)


def warmup(fn):
    """Decorates `fn(cls, settings) -> bars` as a block's `warmup_bars`."""
    return _Warmup(fn)


def _finite_inputs(update_raw):
    """Wraps a block's update_raw: refuses a NaN or infinite input, which would otherwise poison every later
    value while the block still reads as ready, and counts the bars fed for `settled`."""

    @functools.wraps(update_raw)
    def checked(self, *args, **kwargs):
        for x in (*args, *kwargs.values()):  # keyword calls too (QA P1-A3)
            if x is not None and not math.isfinite(x):
                raise ValueError(f"{type(self).__name__} was fed {x!r}: every input must be a finite number")
        update_raw(self, *args, **kwargs)
        self._fed = getattr(self, "_fed", 0) + 1

    return checked


def _counted_reset(reset):
    @functools.wraps(reset)
    def fresh(self):
        reset(self)
        self._fed = 0

    return fresh


class BlockBase:
    """The interface every block keeps. A subclass lists SETTINGS (named as its attributes) and OUTPUTS, and
    implements `_outputs()` with every output by name; `values` hides them as None until it is initialized."""

    SETTINGS: tuple[Setting, ...] = ()
    OUTPUTS: tuple[str, ...] = ("value",)
    confirm_lag = 0  # bars after an event before the block can report it; swing-confirmed blocks set it
    _fed = 0

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "update_raw" in cls.__dict__:
            cls.update_raw = _finite_inputs(cls.__dict__["update_raw"])
        if "reset" in cls.__dict__:
            cls.reset = _counted_reset(cls.__dict__["reset"])

    @property
    def settled(self) -> bool:
        """Fed at least its warm-up: it now reads what a block running over all history would (QA, v2 P1-3, F1).
        Library blocks are not initialized before this; Sma, AtrSma and Rsi keep the earlier `initialized` the
        hand-coded models trade on, so a rule reads `settled`."""
        return self.initialized and self._fed >= self.warmup_bars

    @classmethod
    def check_settings(cls, settings: dict) -> dict:
        """Settings with defaults filled in, each checked against the spec; unknown names are refused."""
        known = {s.name: s for s in cls.SETTINGS}
        unknown = set(settings) - set(known)
        if unknown:
            raise ValueError(f"{cls.__name__} has no setting {', '.join(sorted(unknown))}; "
                             f"it takes {', '.join(known) or 'none'}")
        out = {}
        for name, spec in known.items():
            value = settings.get(name, spec.default)
            out[name] = None if value is None else spec.check(value)
        return out

    @property
    def settings(self) -> dict:
        return {s.name: getattr(self, s.name) for s in self.SETTINGS}

    def _outputs(self) -> dict:
        raise NotImplementedError

    @property
    def values(self) -> dict:
        if not self.initialized:
            return dict.fromkeys(self.OUTPUTS)
        return self._outputs()

    def handle_bar(self, bar) -> None:
        self.update_ohlcv(bar.open.as_double(), bar.high.as_double(), bar.low.as_double(), bar.close.as_double(),
                          bar.volume.as_double(), ts_ns=bar.ts_event)


class Block(BlockBase):
    """A library block whose primary output, `value`, is None until it is initialized."""

    _value = 0.0

    @property
    def value(self):
        return self._value if self.initialized else None


def peek(block, close: float):
    """The value `block` would read if `close` closed the next bar, worked out on a copy so the block itself
    is untouched: the forming candle's value for display, never for a decision. None until the copy has
    enough bars."""
    probe = copy.deepcopy(block)
    probe.update_raw(close)
    return probe.value if probe.initialized else None


def peek_values(block, close: float) -> dict:
    """Every output `block` would read if `close` closed the next bar; the block itself is untouched."""
    probe = copy.deepcopy(block)
    probe.update_raw(close)
    return probe.values
