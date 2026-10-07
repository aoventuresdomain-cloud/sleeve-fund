"""Portfolio backtest (v2 P2-7): several strategies in ONE NautilusTrader run, each on its own cloned simulated venue,
with every entry they want at a bar's close sized by the central sizing core and passed through the same portfolio
limits core as paper, in a fixed strategy order (Independent Quant Advisor, condition 2: no second copy of the limits).

The P2-7a spike (qd/p2-7a-spike-note.md) showed one engine does it:
- each strategy trades its own clone of the venue (same symbol, venue = the clone), with an independent account;
- an alert at bar close + 1 ns sees every strategy's intents for that close and handles them in one gate pass;
- a strategy may hold two clones (cash + margin) for two legs.

This module holds the pieces that don't depend on the gate's core (P2-2), which plugs in through `GateFn` once it is on
main: the venue clones, the per-close batch and the one fill-cost hook."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from nautilus_trader.model import CryptoPerpetual, CurrencyPair, InstrumentId, Venue

from sleeve_fund.money import money, scale

BATCH_DELAY_NS = 1  # the gate pass runs this long after a bar's close: after every venue's bar for that close
# ACT-DRIFT (Independent Quant Advisor, 7 Oct 20:05 UK): paper acts at the bar boundary + 2 s, the backtest at the close.
# No drift is modelled until the parity report shows mean adverse drift above 2.5 bp over 200 or more fills; then this
# fixed figure is set, and every portfolio fill pays it through fill_price(), the one hook.
ACT_DRIFT_BP = 0.0

_TYPES = {"CurrencyPair": CurrencyPair, "CryptoPerpetual": CryptoPerpetual}


def clone_venue(base: Venue | str, index: int) -> Venue:
    """The simulated venue clone for the index-th strategy (or leg) of a portfolio run: KRAKEN -> KRAKEN_P1 (no hyphen: an account id is venue-number, split on its hyphen)."""
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
    """One strategy's entry intent at a close, waiting for the gate pass."""

    strategy: str
    intent: Any  # the gate's Intent (P2-2), or any payload the submit callback understands
    submit: Callable[[Any, Any], None]  # (intent, decision) -> sends what the gate approved; never called on a refusal


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
            clock.set_time_alert_ns(f"portfolio-gate-{ts}", ts + BATCH_DELAY_NS, callback=lambda event, t=ts: self.run(t))

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
            out.append((p, decision))
        return out


def strategy_order(names: Iterable[str]) -> tuple[str, ...]:
    """The fixed order a run gates in: as given, each name once."""
    out = tuple(names)
    if len(set(out)) != len(out):
        raise ValueError(f"a strategy appears twice in the run: {out}")
    return out
