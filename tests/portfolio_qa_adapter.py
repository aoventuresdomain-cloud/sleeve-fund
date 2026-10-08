"""QA Tester 2's assumed P2-2 interface (quant-review/p2-2-xfails, 6 Oct), mapped onto the built gate so QA's cells
run unchanged: their README says to adapt the names, never the assertions. Translation only: every decision is
sleeve_fund.portfolio.gate.check_order on a ledger fed from QA's state snapshot, every holding comes from
portfolio.holding_for, every halt and pause from mark_book, and Resume and cash flows from resume_after_halt and apply_flow (F208-1). QA's figures are floats; they become exact
Decimals here, at the edge (Decimal(str(x)), DA's rule 1).

What lives only here: QA's Gate object (paper's call sites hold the ledger and call check_order themselves), its
reduce-only path, which approves without asking because exits never go to the gate at all, and QA's resting order: an
approved entry is an order sent and live until its release (cancel confirmed, rejected or filled), so the sweep cancels
it and holds the reservation (Advisor 20:25 UK)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal

from sleeve_fund.open_risk import position_risk
from sleeve_fund.portfolio import book as _book
from sleeve_fund.portfolio import gate as _gate
from sleeve_fund.portfolio.holding import Position as _Position
from sleeve_fund.portfolio.holding import holding_for, underlying
from sleeve_fund.portfolio.limits import LIMITS
from sleeve_fund.portfolio.limits import Intent as _Intent
from sleeve_fund.risk import trading_day

RESERVATION_TTL_SECONDS = int(_gate.RESERVATION_TTL.total_seconds())
next_utc_midnight = _gate.next_utc_midnight
_RELEASE = {"cancelled": "cancel", "cancel": "cancel", "rejected": "reject", "reject": "reject", "filled": "fill",
            "fill": "fill"}


def D(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def book_equity(strategy_equities, unallocated) -> float:
    return float(_book.book_equity([D(e) for e in strategy_equities], D(unallocated)))


def backtest_book(strategy_equity, allocation_share):
    book, label = _book.backtest_book(D(strategy_equity), allocation_share)
    return float(book), label


@dataclass(frozen=True)
class Position:
    strategy_id: str
    instrument: str
    venue: str
    side: int
    qty: Decimal
    mark: float
    leverage: float | None
    stop_price: float | None
    atr_frac: float = 0.0

    def holding(self):
        p = _Position(self.strategy_id, underlying(self.instrument), self.side * D(self.qty),
                      None if self.stop_price is None else D(self.stop_price), self.leverage)
        return holding_for(p, D(self.mark), self.atr_frac)


@dataclass(frozen=True)
class PortfolioState:
    equity: float
    positions: list
    hwm: float
    day_start_equity: float
    mark_ts: datetime
    halt_reference: float | None = None


@dataclass(frozen=True)
class Intent:
    strategy_id: str
    instrument: str
    venue: str
    side: int
    qty: Decimal
    price: float
    leverage: float | None
    stop_frac: float | None
    atr_frac: float
    lot: Decimal
    min_qty: Decimal
    fee_rate: float = 0.0
    slippage: float = 0.0
    reduce_only: bool = False
    risk_budget: float = 0.0

    def core(self) -> _Intent:
        """Per unit: margin at the leverage (spot: the full price); risk from the expected fill to the planned stop,
        or the stopless measure, by open_risk.position_risk. Fees and slippage stay in sizing, not in the open-risk
        measure (HoE 20:12 UK; QA's cells all use none)."""
        px = D(self.price)
        stop = None if self.stop_frac is None else px * (1 - self.side * D(self.stop_frac))
        risk = position_risk(Decimal(self.side), px, stop, self.atr_frac)
        margin = px / D(self.leverage) if self.leverage else px
        return _Intent(underlying(self.instrument), self.side, D(self.qty), px, margin, risk, D(self.lot),
                       D(self.min_qty))


@dataclass(frozen=True)
class Decision:
    outcome: str
    approved_qty: Decimal
    requested_qty: Decimal
    limit_hit: str | None
    reason: str
    numbers: dict
    seq: int | None
    id: int | None
    event: dict | None


@dataclass
class _SnapshotLedger(_gate.MemoryLedger):
    """The in-memory ledger, its book and portfolio state read from QA's snapshot once per check, marked through
    mark_book so the halt and the pause are the gate's own. A failing snapshot fails inside check_order."""

    provider: object = None
    profile: object = None
    fresh: bool = False
    held_from: list = field(default_factory=list)

    def state(self):
        if not self.fresh:
            self.fresh = True
            snap = self.provider()
            self.equity, self.held = D(snap.equity), [p.holding() for p in snap.positions]
            prev = self.current
            self.current = _gate.PortfolioState(
                hwm=D(snap.hwm), day=trading_day(snap.mark_ts), day_start_equity=D(snap.day_start_equity),
                halt_reference=None if snap.halt_reference is None else D(snap.halt_reference),
                stale_told=prev.stale_told if prev.mark_ts == snap.mark_ts else None)
            _gate.mark_book(self, D(snap.equity), snap.mark_ts, self.profile)
        return self.current


class Gate:
    def __init__(self, profile, state_provider, clock, mode="paper"):
        self.profile, self.clock, self.mode = profile, clock, mode
        self.ledger = _SnapshotLedger(provider=state_provider, profile=profile)
        self.journal: list[Decision] = []

    def decide(self, intent: Intent) -> Decision:
        if intent.reduce_only:  # exits never go to the gate
            q = D(intent.qty)
            return Decision("approved", q, q, None, "Exit: never gated", {}, None, None, None)
        self.ledger.fresh = False
        alerts = len(self.ledger.alerts)
        core = intent.core()
        c = _gate.check_order(self.ledger, intent.strategy_id, core, self.profile, self.clock())
        if c.reservation is not None:  # QA's approved entry is an order sent and resting until it is released
            self.ledger.attach_order(c.reservation, f"Q-{c.reservation}")
        d = c.decision
        new = self.ledger.alerts[alerts:]
        k = d.limit_hit if d.limit_hit in LIMITS else "gross"
        per_unit = {"gross": core.price, "net_instrument": core.side * core.price, "margin": core.margin_per_unit,
                    "open_risk": core.risk_per_unit}[k]
        e = self.ledger.equity
        numbers = {}
        if d.figures:
            used = D(repr(d.figures[k]["before"])) * e
            numbers = {"limit": float(D(repr(d.figures[k]["cap"])) * e),
                       "post": float(abs(used + d.requested_qty * per_unit))}
        out = Decision(d.outcome, d.approved_qty, d.requested_qty, d.limit_hit, d.reason, numbers, c.seq,
                       c.reservation, {"kind": new[-1][0], "message": new[-1][1]} if new else None)
        self.journal.append(out)
        return out

    def release(self, decision: Decision, why: str) -> None:
        self.ledger.release(decision.id, _RELEASE[why])

    def sweep(self) -> list[dict]:
        alerts = len(self.ledger.alerts)
        resting = {r.order_id for r in self.ledger.reservations()}
        _gate.sweep(self.ledger, self.clock(), resting.__contains__, lambda order_id: None)
        return [{"kind": k, "message": m} for k, m, _ in self.ledger.alerts[alerts:]]


@dataclass(frozen=True)
class Verdict:
    halt: bool
    pause: bool
    flatten: bool
    reasons: list


def _ledger(state: PortfolioState) -> _gate.MemoryLedger:
    """A ledger holding QA's snapshot as the gate's last mark, so the book functions below are the product's own
    (gate.mark_book, resume_after_halt, apply_flow; F208-1), not copies of them."""
    led = _gate.MemoryLedger(equity=D(state.equity))
    led.current = _gate.PortfolioState(
        equity=D(state.equity), mark_ts=state.mark_ts, hwm=D(state.hwm), day=trading_day(state.mark_ts),
        day_start_equity=D(state.day_start_equity),
        halt_reference=None if state.halt_reference is None else D(state.halt_reference))
    return led


def _snapshot(state: PortfolioState, st: _gate.PortfolioState) -> PortfolioState:
    return replace(state, equity=float(st.equity), hwm=float(st.hwm), day_start_equity=float(st.day_start_equity),
                   halt_reference=None if st.halt_reference is None else float(st.halt_reference))


def evaluate(state: PortfolioState, profile, now: datetime) -> Verdict:
    led = _ledger(state)
    acted = _gate.mark_book(led, D(state.equity), now, profile)
    st = led.state()
    halt, pause = acted == "halt", acted == "pause"
    return Verdict(halt, pause, halt, [st.halted] if halt else [st.paused] if pause else [])


def resume(state: PortfolioState) -> PortfolioState:
    led = _ledger(state)
    return _snapshot(state, _gate.resume_after_halt(led))


def apply_cash_flow(state: PortfolioState, amount: float) -> PortfolioState:
    led = _ledger(state)
    _gate.apply_flow(led, D(amount))
    return _snapshot(state, led.state())
