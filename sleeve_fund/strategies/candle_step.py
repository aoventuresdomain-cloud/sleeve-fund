"""The candle-step test Strategy's plan (pe1-plans/candle-step-design.md; HoE QR-B10 re-cut; Advisor CANDLE-STEP; PM
22:38 UK "run as i specified"). A test of the plumbing, not an idea: it runs alone, on the PM's go only.

A cycle is 6 bars, keyed by the bar's close time (never by when the Strategy started), so a backtest, paper and a
restart land on the same (cycle, step):

    0 buy +Q | 1 sell half +Q/2 | 2 sell the rest and short -Q | 3 cover half -Q/2 | 4 close and reverse +Q | 5 wait +Q

Each step is an absolute target in units of Q, fixed at step 0 (Advisor: sized from fund equity at that mark, journaled
with the cycle, reused for the cycle; each open still gated fresh). From what is held, a step asks for:
- an open (steps 0, 2, 4) to Q on the step's side, a flip as two orders: the close of everything held (never gated),
  then the open; never one double-size order (Advisor MUST);
- a reduce (steps 1, 3) of what is held above Q/2, never an add, so after a trim at the gate it halves at most.
A stop or target inside a cycle makes the rest of it a no-op; the next cycle starts flat at step 0. A step acts once,
however often its bar is seen. Started mid-cycle, it waits for step 0: it never sizes from a guess. Once the PM's
run window (run_for_hours, no default) has passed, it closes what it holds and opens nothing.

This module is pure: no engine, no clock, no journal, so every path reads the same answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

CYCLE = 6
HALF = Decimal("0.5")


@dataclass(frozen=True)
class Step:
    index: int
    target: Decimal  # signed, in units of Q
    opens: bool  # whether it may open (a reduce step never does)


STEPS = (Step(0, Decimal(1), True), Step(1, HALF, False), Step(2, Decimal(-1), True), Step(3, -HALF, False),
         Step(4, Decimal(1), True), Step(5, Decimal(1), False))


def step_at(close: datetime, bar_minutes: int) -> tuple[int, int]:
    """(cycle, step) of the bar closing at `close`: bars since the epoch, in 6s. The same bar always gives the same pair,
    whatever second of it the close is read at."""
    if bar_minutes <= 0:
        raise ValueError(f"bar_minutes must be positive, got {bar_minutes}")
    close = close if close.tzinfo is not None else close.replace(tzinfo=UTC)
    bar = int(close.timestamp()) // (bar_minutes * 60)
    return divmod(bar, CYCLE)


def _side(x: Decimal) -> int:
    return (x > 0) - (x < 0)


@dataclass
class Plan:
    """One cycle's plan: its Q (None until step 0 sized it), the steps already acted on, and whether a stop or
    target closed the position inside it."""

    cycle: int
    q: Decimal | None
    started: datetime | None = None
    run_for_hours: float | None = None
    done: set[int] = field(default_factory=set)
    exited: bool = False

    def __post_init__(self):
        if self.q is not None and self.q <= 0:
            raise ValueError(f"Q must be positive, got {self.q}")
        if self.run_for_hours is not None and self.run_for_hours <= 0:
            raise ValueError(f"run_for_hours must be positive, got {self.run_for_hours}")

    def closed_by_exit(self) -> None:
        """A stop or target closed the position inside this cycle: the rest of it is a no-op."""
        self.exited = True

    def orders(self, step: int, held: Decimal) -> list[tuple[str, int, Decimal]]:
        """What step `step` asks for, given the signed quantity held: (kind, side, qty) in the order they go, kind
        "close", "reduce" (both never gated) or "open" (gated). Empty when it asks for nothing."""
        if not 0 <= step < CYCLE:
            raise ValueError(f"step {step} outside 0-{CYCLE - 1}")
        if self.q is None or self.exited:
            return []
        s = STEPS[step]
        want = s.target * self.q
        if not s.opens:
            # A reduce or a wait: only ever towards 0, and only by what is held beyond the target on its own side.
            if _side(held) != _side(want) or abs(held) <= abs(want):
                return []
            return [("reduce", -_side(held), abs(held) - abs(want))]
        if held == want:
            return []
        out = []
        if held != 0 and _side(held) != _side(want):
            out.append(("close", -_side(held), abs(held)))
            held = Decimal(0)
        if abs(held) < abs(want):
            out.append(("open", _side(want), abs(want) - abs(held)))
        return out

    def act(self, step: int, held: Decimal) -> list[tuple[str, int, Decimal]]:
        """orders(), once per step: a step already acted on asks for nothing (a duplicate bar, a restart)."""
        if step in self.done:
            return []
        self.done.add(step)
        return self.orders(step, held)

    def expired(self, now: datetime) -> bool:
        """Whether the PM's run window has passed. Never, when the PM set none."""
        if self.run_for_hours is None or self.started is None:
            return False
        return now >= self.started + timedelta(hours=self.run_for_hours)

    def after_expiry(self, held: Decimal) -> list[tuple[str, int, Decimal]]:
        """Past the run window: close what is held, open nothing."""
        return [] if held == 0 else [("close", -_side(held), abs(held))]


__all__ = ["CYCLE", "STEPS", "Plan", "Step", "step_at"]
