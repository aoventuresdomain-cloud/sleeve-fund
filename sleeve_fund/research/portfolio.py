"""Portfolio backtest (v2 P2-7): several strategies in ONE NautilusTrader run, each on its own cloned simulated venue,
with every entry they want at a bar's close sized by the central sizing core and passed through the same portfolio
limits core as paper, in a fixed strategy order (Independent Quant Advisor, condition 2: no second copy of the limits).

The P2-7a spike (qd/p2-7a-spike-note.md) showed one engine does it:
- each strategy trades its own clone of the venue (same symbol, venue = the clone), with an independent account;
- an alert at bar close + 1 ns sees every strategy's intents for that close and handles them in one gate pass;
- a strategy may hold two clones (cash + margin) for two legs.

This module holds the venue clones, the per-close batch, the one fill-cost hook and the run's gate (PortfolioGate):
P2-2's check_order over a MemoryLedger, with each approved order's reservation held until its fill, reject or cancel."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from nautilus_trader.model import CryptoPerpetual, CurrencyPair, InstrumentId, Venue

from sleeve_fund.money import money, scale
from sleeve_fund.portfolio.gate import Checked, MemoryLedger, check_order, mark_book
from sleeve_fund.portfolio.holding import Position, holding_for, underlying
from sleeve_fund.portfolio.limits import Holding, Intent, rejected
from sleeve_fund.risk import PortfolioProfile

BATCH_DELAY_NS = 1  # the gate pass runs this long after a bar's close: after every venue's bar for that close
DAY_NS = 86_400_000_000_000
# ACT-DRIFT (Independent Quant Advisor, 7 Oct 20:05 UK): paper acts at the bar boundary + 2 s, the backtest at the
# close. No drift is modelled until the parity report shows mean adverse drift above 2.5 bp over 200 or more fills;
# then this fixed figure is set, and every portfolio fill pays it through fill_price(), the one hook.
ACT_DRIFT_BP = 0.0

_TYPES = {"CurrencyPair": CurrencyPair, "CryptoPerpetual": CryptoPerpetual}


def clone_venue(base: Venue | str, index: int) -> Venue:
    """The simulated venue clone for the index-th strategy (or leg) of a portfolio run: VENUE -> VENUE_P1.

    No hyphen: an account id is venue-number, split on its hyphen."""
    return Venue(f"{base}_P{index}")


def clone_instrument(instrument, venue: Venue):
    """The same instrument listed on `venue`: every field kept, the id's venue swapped, so a clone trades exactly as the
    single-strategy run's instrument does."""
    kind = type(instrument).__name__
    if kind not in _TYPES:
        raise TypeError(f"no clone for a {kind}")
    d = _TYPES[kind].to_dict(instrument)
    d["id"] = str(InstrumentId(instrument.id.symbol, venue))
    return _TYPES[kind].from_dict(d)


def fill_price(close: Decimal, side: int, half_spread: float, drift_bp: float | None = None) -> Decimal:
    """The expected fill of a market order at a bar's close: close x (1 ± (half spread + act drift)) for the side. The
    sizing core's price and the gate's Intent.price both come from here, and ACT-DRIFT changes only ACT_DRIFT_BP."""
    if side not in (1, -1):
        raise ValueError(f"side is 1 or -1, got {side}")
    bp = ACT_DRIFT_BP if drift_bp is None else drift_bp
    return money(close, "close") + scale(money(close, "close"), side * (half_spread + bp / 10_000))


@dataclass(frozen=True)
class Pending:
    """One strategy's opening intent (an entry, or an addition to a holding) at a close, waiting for the gate pass."""

    strategy: str
    intent: Any  # the gate's Intent (P2-2), or any payload the submit callback understands
    submit: Callable[[Any, Any], None]  # (intent, decision) -> sends what the gate approved; never called on a refusal
    refused: Callable[[Any, Any], None] | None = None  # (intent, decision) on a refusal, so the strategy can count it


# The gate pass: (strategy, intent, ts_ns) -> a decision with .approved_qty (P2-2's check_order over the MemoryLedger).
GateFn = Callable[[str, Any, int], Any]


@dataclass
class CloseBatch:
    """Every strategy's entry intents at one bar close, handled in ONE gate pass at close + BATCH_DELAY_NS, in the fixed
    strategy order given (the run's order, so the result never depends on which venue's bar arrived first)."""

    order: tuple[str, ...]
    gate: GateFn
    pending: dict[int, list[Pending]] = field(default_factory=dict)
    passes: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)  # (close, strategies seen) for the record

    def post(self, clock, ts: int, item: Pending) -> None:
        """Queue an intent for the close `ts`; the first one for that close arms the alert on `clock`."""
        if item.strategy not in self.order:
            raise ValueError(f"{item.strategy!r} is not in this run")
        if any(t == ts for t, _ in self.passes):
            # Fail loudly: re-arming would put an alert in the past and the intent would skip the close's one pass.
            raise ValueError(f"the close {ts} has already been gated")
        first = ts not in self.pending
        self.pending.setdefault(ts, []).append(item)
        if first:
            # set_time_alert_ns, never set_time_alert(datetime): a datetime keeps microseconds only, so the extra
            # nanosecond is lost and the alert lands on the close itself (P2-7a spike, trap 1).
            clock.set_time_alert_ns(
                f"portfolio-gate-{ts}", ts + BATCH_DELAY_NS, callback=lambda event, t=ts: self.run(t))

    def run(self, ts: int) -> list[tuple[Pending, Any]]:
        """The gate pass for the close `ts`: each intent in strategy order, then in the order it was posted."""
        items = self.pending.pop(ts, [])
        rank = {name: n for n, name in enumerate(self.order)}
        items = sorted(items, key=lambda p: rank[p.strategy])  # stable: a strategy's own intents keep their order
        self.passes.append((ts, tuple(dict.fromkeys(p.strategy for p in items))))
        out = []
        for p in items:
            decision = self.gate(p.strategy, p.intent, ts + BATCH_DELAY_NS)
            if getattr(decision, "approved_qty", 0) > 0:
                p.submit(p.intent, decision)
            elif p.refused is not None:
                p.refused(p.intent, decision)
            out.append((p, decision))
        return out


def gate_intent(d: dict) -> Intent:
    """P2-2's Intent for a strategy's opening order, from the hook's intent (LongFlatStrategy._send_entry): its side,
    qty, expected fill (`price`, from fill_price), instrument, venue step and minimum, and what one unit posts and
    risks. Those come from holding_for on one unit at the expected fill, so the gate counts the order exactly as it
    counts the position once held: margin = price / leverage on a perp (leverage None: spot, the full price); risk to
    the stop (stop_frac from the expected fill) or, with none, the stopless measure from the daily ATR (atr_pct).
    Raises when a figure can't be used (a float price, a stopless order with no ATR), and the gate fails closed."""
    side, price = d["side"], money(d["price"], "price")
    frac = d.get("stop_frac")
    stop = price - scale(price, side * frac) if frac else None
    name = underlying(d["instrument"])
    unit = holding_for(Position("", name, Decimal(side), stop, d.get("leverage")), price, d.get("atr_pct"))
    return Intent(name, side, money(d["qty"], "qty"), price, unit.margin, unit.risk, d["step"], d["min_qty"])


def _utc(ts_ns: int) -> datetime:
    """ts_ns as a datetime, rounded UP to the microsecond a datetime keeps, so a close + 1 ns stays after the close
    (a gate pass at 00:00 + 1 ns belongs to the new day, never the old one)."""
    us = -(-ts_ns // 1000)
    return datetime.fromtimestamp(us // 1_000_000, tz=timezone.utc).replace(microsecond=us % 1_000_000)


@dataclass(frozen=True)
class Gated:
    """One gate answer and its reservation's lifecycle: attach it to the order sent, reduce it as the order fills,
    release it when the order closes (fill, reject or cancel) or is never sent."""

    checked: Checked
    gate: "PortfolioGate"

    @property
    def decision(self):
        return self.checked.decision

    @property
    def approved_qty(self) -> Decimal:
        return self.checked.decision.approved_qty

    def attach(self, order_id: str) -> None:
        if self.checked.reservation is not None:
            self.gate.ledger.attach_order(self.checked.reservation, order_id)

    def filled(self, qty: Decimal) -> None:
        if self.checked.reservation is not None:
            self.gate.ledger.reduce(self.checked.reservation, money(qty, "qty"))

    def release(self, reason: str) -> None:
        if self.checked.reservation is not None:
            self.gate.ledger.release(self.checked.reservation, reason)


@dataclass
class PortfolioGate:
    """The GateFn of a portfolio run: P2-2's check_order over a MemoryLedger (the one in-memory Ledger; Advisor
    condition 2). book(ts_ns) gives the fund's marked equity and every strategy's held positions as Holdings
    (holding_for); it is read at every check, so an order that filled since the last one counts as held and its
    reservation has gone. Each check marks the book first (mark_book): the mark is never stale, and a drawdown halt or
    daily pause it sets blocks the entry. The runner also calls mark() at every close, so the high-water mark sees
    every close and not only those with an entry, and acts on its "halt" (flatten and halt every strategy)."""

    book: Callable[[int], tuple[Decimal, tuple[Holding, ...]]]
    profile: PortfolioProfile = field(default_factory=PortfolioProfile)
    ledger: MemoryLedger = field(default_factory=MemoryLedger)
    on_action: Callable[[int, str], None] | None = None  # (ts_ns, "halt" | "pause"), whichever mark sets it

    def mark(self, ts_ns: int) -> str | None:
        """Mark the fund at ts_ns, a close + BATCH_DELAY_NS: "halt" or "pause" when that mark has just set one
        (on_action hears it), else None. At 00:00 UTC the order is fixed (Advisor 8 Oct 03:25 UK): first the old
        day's final mark at the close itself, then the new day's daily start from it, then the Strategies' decisions,
        so no decision sees the old day's pause or a stale daily start, whichever alert at that instant runs first.
        By design the daily pause lifts at 00:00 UTC, when a daily Strategy decides, so it never blocks one."""
        equity, held = self.book(ts_ns)
        self.ledger.equity, self.ledger.held = money(equity, "equity"), list(held)
        close = ts_ns - BATCH_DELAY_NS
        marked = self.ledger.state().mark_ts
        acts = []
        if close % DAY_NS == 0 and (marked is None or marked < _utc(close)):
            acts.append((close, mark_book(self.ledger, self.ledger.equity, _utc(close), self.profile)))
        acts.append((ts_ns, mark_book(self.ledger, self.ledger.equity, _utc(ts_ns), self.profile)))
        for at, acted in acts:
            if acted is not None and self.on_action is not None:
                self.on_action(at + (BATCH_DELAY_NS if at == close else 0), acted)
        return next((a for _, a in reversed(acts) if a is not None), None)

    def __call__(self, strategy: str, intent: dict, ts_ns: int) -> Gated:
        now = _utc(ts_ns)
        try:
            self.mark(ts_ns)
            order = gate_intent(intent)
        except Exception as e:  # noqa: BLE001 - an order the gate can't read is never sent
            why = f"Entry rejected: the portfolio check couldn't run ({type(e).__name__}: {e}), so nothing is sent"
            self.ledger.alert("portfolio_check_failed", f"{strategy}: {why}", now)
            qty = intent.get("qty") if isinstance(intent, dict) else None
            qty = qty if isinstance(qty, Decimal) else Decimal(0)
            return Gated(Checked(rejected(qty, None, why), None, None), self)
        return Gated(check_order(self.ledger, strategy, order, self.profile, now), self)


def strategy_order(names: Iterable[str]) -> tuple[str, ...]:
    """The fixed order a run gates in: as given, each name once."""
    out = tuple(names)
    if len(set(out)) != len(out):
        raise ValueError(f"a strategy appears twice in the run: {out}")
    return out
