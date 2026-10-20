"""The candle-step test Strategy's plan (pe1-plans/candle-step-design.md; HoE QR-B10 re-cut; Advisor CANDLE-STEP; PM
22:38 UK "run as i specified"). A test of the plumbing, not an idea: it runs alone, on the PM's go only.

A cycle is 6 bars, keyed by the bar's close time (never by when the Strategy started), so a backtest, paper and a
restart land on the same (cycle, step):

    0 buy +Q | 1 sell half +Q/2 | 2 sell the rest and short -Q | 3 cover half -Q/2 | 4 close and reverse +Q | 5 wait +Q

Each step is an absolute target in units of Q, fixed at step 0 (Advisor: sized from fund equity at that mark, journaled
with the cycle, reused for the cycle; each open still gated fresh). From what is held, a step asks for:
- an open (steps 0, 2, 4) to Q on the step's side, a flip as two orders: the close of everything held (never gated),
  then the open; never one double-size order (Advisor MUST);
- a reduce (steps 1, 3) to its leg less half of it: the leg is what was held when that leg's open filled (step 0's
  long, step 2's short), so after a trim at the gate it covers half the size actually held, not down to Q/2 (Advisor
  B30-HELD-SIZE, 20 Oct; the PM's "cover half"). Still an absolute target, never a delta or an add: a step seen again
  sends nothing. Step 5 waits, reducing only what is held above Q.
A stop or target inside a cycle makes the rest of it a no-op; the next cycle starts flat at step 0. A step acts once,
however often its bar is seen. Started mid-cycle, it waits for step 0: it never sizes from a guess. Once the PM's
run window (run_for_hours, no default) has passed, it closes what it holds and opens nothing.

The plan (step_at, Plan) is pure: no engine, no clock, no journal, so every path reads the same answer. CandleStep is
the model the engine runs it through (LongFlatStrategy._on_bar_stepped): the same order path, gate and exits as every
Strategy, so it tests that path."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal

from nautilus_trader.model import Bar

from sleeve_fund.data import bar_minutes
from sleeve_fund.strategies.base import EXIT_LEGS, IdeaSpec, LongFlatConfig, LongFlatStrategy

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
    legs: dict[int, Decimal] = field(default_factory=dict)  # open step -> the size held once its open filled (B-30)

    def __post_init__(self):
        if self.q is not None and self.q <= 0:
            raise ValueError(f"Q must be positive, got {self.q}")
        if self.run_for_hours is not None and self.run_for_hours <= 0:
            raise ValueError(f"run_for_hours must be positive, got {self.run_for_hours}")

    def closed_by_exit(self) -> None:
        """A stop or target closed the position inside this cycle: the rest of it is a no-op."""
        self.exited = True

    def leg_filled(self, step: int, held: Decimal) -> None:
        """An open of step `step` filled, leaving `held` (signed): the size its half step halves."""
        if step < CYCLE and STEPS[step].opens and held != 0:
            self.legs[step] = abs(held)

    def orders(self, step: int, held: Decimal, lot: Decimal | None = None) -> list[tuple[str, int, Decimal]]:
        """What step `step` asks for, given the signed quantity held: (kind, side, qty) in the order they go, kind
        "close", "reduce" (both never gated) or "open" (gated). Empty when it asks for nothing. lot: the order step,
        which a half step's cover rounds down to (never over-closing)."""
        if not 0 <= step < CYCLE:
            raise ValueError(f"step {step} outside 0-{CYCLE - 1}")
        if self.q is None or self.exited:
            return []
        s = STEPS[step]
        want = s.target * self.q
        leg = self.legs.get(step - 1) if abs(s.target) == HALF else None
        if leg is not None:  # B30-HELD-SIZE: the leg less half of it, the half rounded down to the lot
            cover = leg / 2 if lot is None else (leg / 2).quantize(lot, rounding=ROUND_DOWN)
            want = _side(s.target) * (leg - cover)
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

    def act(self, step: int, held: Decimal, lot: Decimal | None = None) -> list[tuple[str, int, Decimal]]:
        """orders(), once per step: a step already acted on asks for nothing (a duplicate bar, a restart)."""
        if step in self.done:
            return []
        self.done.add(step)
        return self.orders(step, held, lot)

    def expired(self, now: datetime) -> bool:
        """Whether the PM's run window has passed. Never, when the PM set none."""
        if self.run_for_hours is None or self.started is None:
            return False
        return now >= self.started + timedelta(hours=self.run_for_hours)

    def after_expiry(self, held: Decimal) -> list[tuple[str, int, Decimal]]:
        """Past the run window: close what is held, open nothing."""
        return [] if held == 0 else [("close", -_side(held), abs(held))]


SPEC = IdeaSpec(
    summary="A test of the order path: a {size_share} share of fund equity bought, halved, reversed and closed on a "
            "fixed 6-candle cycle.",
    name="candle_step",
    family="test",
    idea="The PM's timed test: buy, sell half, sell the rest and short, cover half, close and reverse, wait, repeat, "
         "with a tight stop and target, to see every order respect its stop, its target and the fund's limits.",
    rules=(
        "A cycle of 6 candles keyed by the candle's close time: buy Q, sell half, sell the rest and short Q, cover "
        "half, close the short and buy Q, wait. Q is sized at the cycle's first candle from the fund's equity (the "
        "size share) and kept for the cycle. A flip is two orders: the close, then the open once it fills. A stop or "
        "target inside a cycle ends that cycle; the next one starts at its first candle."
    ),
    data_needs="1-minute OHLCV",
    default_risk_profile="balanced",
    default_params={"size_share": 0.6, "market": "perp", "allow_short": True, "keep_sub_cost_target": True},
    known_weaknesses="A test of the plumbing, not an idea: it pays a round trip's costs on every step and is expected "
                     "to lose. It runs alone, and only on the PM's go.",
    listed=False,
)

def D_(x) -> Decimal:
    return Decimal(str(x))


def _ts(t) -> datetime:
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


STEP_WORDS = ("buy Q", "sell half", "sell the rest and short Q", "cover half", "close the short and buy Q", "wait")


class CandleStepConfig(LongFlatConfig):
    def __init__(self, *, size_share: float = 0.6, run_for_hours: float | None = None, **kwargs) -> None:
        # PM 22:38 UK "run as i specified": its 0.1% target is placed though below the round trip's costs (QD's C8
        # flag, on for this test Strategy only; the PM's settings may still turn it off).
        kwargs.setdefault("keep_sub_cost_target", True)
        super().__init__(**kwargs)
        if not 0 < size_share <= 1:
            raise ValueError(f"size_share {size_share} outside (0, 1]: the share of fund equity Q is sized to, "
                             "e.g. 0.6 for 60%")
        if run_for_hours is not None and not run_for_hours > 0:
            raise ValueError(f"run_for_hours {run_for_hours} must be positive, or unset for no end")
        self.size_share, self.run_for_hours = size_share, run_for_hours


class CandleStep(LongFlatStrategy):
    STEPPED = True  # decided by step_orders (_on_bar_stepped), not want_side
    REENTER_AFTER_EXIT_LEG = True  # a stop or target ends the cycle (exit_leg_closed), never locks a side

    def __init__(self, config: CandleStepConfig) -> None:
        super().__init__(config)
        self.c = config
        self._plan: Plan | None = None
        self._started: datetime | None = None
        self._last_q: Decimal | None = None  # the last cycle's Q, carried while its long is held
        self._resumed: tuple[int, Decimal | None, bool] | None = None  # (cycle, Q, ended) from the journal
        self._resumed_legs: dict[int, Decimal] = {}  # that cycle's legs as their opens filled (Plan.legs)

    def resume_cycle(self) -> None:
        """After a restart: the cycle the journal's newest step order was in, its Q, and whether a stop or target has
        ended it, so the cycle carries on where it stood. Every step is an absolute target in Q, so a step decided after
        the restart sends only what is still missing, and one missed while down is made up by the next: it never
        doubles a position. With nothing in the journal it starts afresh at the next step 0."""
        if self.runtime is None or self.runtime.backtest:
            return
        rows = self.runtime.store.orders(self.runtime.name, limit=100)
        steps = [o for o in rows if "cycle" in (o["signal"] or {})]
        if not steps:
            return
        if (started := steps[0]["signal"].get("run_started")) is not None:
            self._started = _ts(datetime.fromisoformat(started))  # the run window counts from the first start
        cycle = steps[0]["signal"]["cycle"]
        mine = [o for o in steps if o["signal"]["cycle"] == cycle]
        q = next((D_(o["signal"]["q"]) for o in mine if o["signal"].get("q") is not None), None)
        if q is None:  # only the cycle's first open is journaled yet: what it asked for is Q
            q = next((D_(o["qty"]) for o in mine if o["intent"] == "entry" and o["signal"].get("step") == 0), None)
        first = min(_ts(o["ts"]) for o in mine)
        ended = any(o["intent"] in EXIT_LEGS and (o["filled_qty"] or 0) > 0 and _ts(o["ts"]) >= first for o in rows)
        self._resumed, self._last_q = (cycle, q, ended), q
        # Each leg as its open filled (B-30): what its entry orders filled, or a long carried in at step 0 (Q).
        self._resumed_legs = {}
        for k in (0, 2):
            got = sum((D_(o["filled_qty"]) for o in mine
                       if o["intent"] == "entry" and o["signal"].get("step") == k and (o["filled_qty"] or 0) > 0),
                      Decimal(0))
            if got > 0:
                self._resumed_legs[k] = got
        if 0 not in self._resumed_legs and q is not None and any(o["signal"].get("step") == 1 for o in mine):
            self._resumed_legs[0] = q  # step 0 carried the last cycle's long in without an order

    def _carried_into(self, cycle: int, held: Decimal) -> bool:
        """After a restart: the journal's newest cycle is the one before this and ended long Q, so this cycle's step 0
        carried that long in without an order (nothing for the journal to show) and its Q is the same. Without this a
        restart before the cycle's first order sat out the whole cycle (PE1 restart cell, drop-b-18)."""
        r = self._resumed
        return r is not None and r[0] == cycle - 1 and not r[2] and r[1] is not None and held == r[1]

    def want_side(self, bar: Bar) -> int | None:
        return None  # never decides by side: every order comes from the step

    def want_long(self, bar: Bar) -> bool | None:
        return None

    def exit_leg_closed(self) -> None:
        if self._plan is not None:
            self._plan.closed_by_exit()

    def step_orders(self, bar: Bar, held: Decimal) -> tuple[list[tuple[str, int, Decimal | None]], str, dict]:
        """The orders this candle's step asks for, given the signed quantity held, with the reason and values to
        journal. An open with qty None is the cycle's first: sized by the size share, and its size becomes Q
        (entry_sent). Past the run window it only closes."""
        close = datetime.fromtimestamp(int(bar.ts_event) / 1e9, tz=UTC)
        cycle, step = step_at(close, bar_minutes(self._cfg.bar_type))
        if self._started is None:
            self._started = close
        plan = self._plan
        if plan is None or plan.cycle != cycle:
            q, ended = None, False
            if self._resumed is not None and self._resumed[0] == cycle:
                _, q, ended = self._resumed  # a restart inside this cycle: carry on with its Q
            elif held > 0 and (step == 0 or self._carried_into(cycle, held)):
                # The last cycle's long, carried into this one's first candle: its Q carries too, topped up to if
                # short of it. Q is sized afresh only from flat, so a missed step never shrinks every later cycle.
                q = self._last_q or held
            plan = self._plan = Plan(cycle, q, self._started, self.c.run_for_hours, exited=ended)
            if self._resumed is not None and self._resumed[0] == cycle:
                plan.legs.update(self._resumed_legs)
            elif q is not None and held > 0:
                plan.leg_filled(0, held)  # the last cycle's long, carried in: step 0's leg as it stands
            if q is not None:
                self._last_q = q
        values = {"cycle": cycle, "step": step, "run_started": self._started.isoformat()}
        if plan.q is not None:
            values["q"] = float(plan.q)
        if plan.expired(close):
            return plan.after_expiry(held), "The run window is over: closing what is held", values
        reason = f"Candle step {step + 1} of {CYCLE} (cycle {cycle}): {STEP_WORDS[step]}"
        if step == 0 and plan.q is None and not plan.exited:
            if 0 in plan.done:
                return [], reason, values
            plan.done.add(0)
            values["size_share"] = self.c.size_share
            return ([("close", 1, -held)] if held < 0 else []) + [("open", 1, None)], reason, values
        return plan.act(step, held, self._lot() if getattr(self, "instrument", None) is not None else None), reason, values

    def on_order_filled(self, event) -> None:
        super().on_order_filled(event)
        # B-30: a step's open filled: record the leg its half step halves, the size held now (all fills so far)
        info = self.decisions.get(str(event.client_order_id)) or {}
        sig = info.get("signal") or {}
        if (self._plan is not None and info.get("intent") == "entry" and sig.get("cycle") == self._plan.cycle
                and sig.get("step") is not None):
            self._plan.leg_filled(int(sig["step"]), self._signed_qty())

    def entry_sent(self, qty: Decimal, values: dict) -> None:
        """The cycle's first open was sized: its quantity is Q for the rest of the cycle (Advisor CANDLE-STEP)."""
        if self._plan is not None and self._plan.q is None and values.get("cycle") == self._plan.cycle:
            self._plan.q = self._last_q = qty


__all__ = ["CYCLE", "STEPS", "SPEC", "CandleStep", "CandleStepConfig", "Plan", "Step", "step_at"]
