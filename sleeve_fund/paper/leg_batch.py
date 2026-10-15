"""A multi-asset Strategy's same-close ideas on paper (multi-asset spec §4, BRIDGE-3-QA 2; Advisor MA-SPEC): its legs
run in one process, and each leg's opening order on a close is held here until every leg has decided on that close, or
until close + WAIT_NS (ACT-DRIFT), whichever is first. The pass then gates them one at a time, strongest idea first and
then by the asset's place in the list, as the backtest's CloseBatch does: never by which feed's bar came first. A leg
that decides after its close's pass gates at once, after its group, and is counted late (the weekly replay report)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

WAIT_NS = 2_000_000_000  # close + 2 s (ACT-DRIFT)


@dataclass
class _Post:
    strength: float
    place: int
    gate: Callable[[], None]  # the leg's own paper gate and send (LongFlatStrategy._send_entry)


@dataclass
class LegBatch:
    names: tuple[str, ...]  # the legs, in the asset list's order
    pending: dict[int, list[_Post]] = field(default_factory=dict)
    decided: dict[int, set[str]] = field(default_factory=dict)
    done: list[int] = field(default_factory=list)  # closes already gated, oldest first (kept short)
    late: dict[str, int] = field(default_factory=dict)  # per leg: ideas gated after their group

    def post(self, clock, close_ns: int, name: str, strength: float, gate: Callable[[], None]) -> None:
        """Hold `name`'s opening order on the close `close_ns` for the pass; after that close's pass, gate it now."""
        if close_ns in self.done:
            self.late[name] = self.late.get(name, 0) + 1
            gate()
            return
        first = close_ns not in self.pending
        self.pending.setdefault(close_ns, []).append(_Post(strength, self.names.index(name), gate))
        if first:
            at = max(close_ns + WAIT_NS, clock.timestamp_ns() + 1)
            clock.set_time_alert_ns(f"legs-gate-{close_ns}", at, callback=lambda event, t=close_ns: self.run(t))

    def decide(self, close_ns: int, name: str) -> None:
        """`name` has decided on the close `close_ns` (with or without an idea): once every leg has, the pass runs."""
        if close_ns in self.done:
            return
        seen = self.decided.setdefault(close_ns, set())
        seen.add(name)
        if seen >= set(self.names) and close_ns in self.pending:
            self.run(close_ns)

    def run(self, close_ns: int) -> None:
        """The pass for the close: each held order gated in turn, strongest first, then by asset place."""
        if close_ns in self.done:
            return
        self.done = (self.done + [close_ns])[-64:]
        self.decided.pop(close_ns, None)
        for p in sorted(self.pending.pop(close_ns, []), key=lambda p: (-p.strength, p.place)):
            p.gate()
