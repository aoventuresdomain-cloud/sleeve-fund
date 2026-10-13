"""P2-5 regime weighting, the weight side (v2/phase2-spec.md "P2-5"; Advisor spec quant-review/specs-p2-3-p2-8-p2-5.md
section 4, 13 Oct 19:10 UK). A confidence weight on NEW entries only: it scales an entry's size, is fixed at entry,
never scales an exit and never closes a Strategy. It is a separate counted variant (PM 13 Oct 17:57 UK), never inside
G1: nothing applies it unless a run is given a Weighting.

The regime state per day comes from the Data Architect's regime_series (sleeve_fund.regime): a row dated D reads daily
closes through D-1 (the pre-registered 1-day lag, Advisor 19:20 UK), so an entry on UTC day D uses the row dated D and
sees only the prior close. A day with no row, or a row whose state is "unknown", weighs 1.0.

Pre-registered (changing any of it is a new counted mapping in the "regime" family, not a tweak):
- three steps, 0.25 / 0.6 / 1.0, the floor 0.25;
- direction per family: trend and breakout are favoured when ER is high, mean reversion when ER is low; the favoured
  state weighs 1.0, mid 0.6, the other 0.25;
- a family with no pre-registered direction is not weighted (1.0 everywhere)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import ROUND_DOWN, Decimal
from types import MappingProxyType

STATES = ("low", "mid", "high")
UNKNOWN = "unknown"
FLOOR, MID, FULL = Decimal("0.25"), Decimal("0.6"), Decimal("1.0")
DIRECTION = MappingProxyType({"trend": "high", "breakout": "high", "mean-reversion": "low"})


def weight(family: str, state: str | None) -> Decimal:
    """The weight of a new entry by a Strategy of `family` on a day in `state`."""
    favoured = DIRECTION.get(family)
    if favoured is None or state not in STATES:
        return FULL
    if state == favoured:
        return FULL
    return MID if state == "mid" else FLOOR


@dataclass
class Weighting:
    """The regime weight per gated order of a run. state_on(day) is the regime state of the row dated `day` (None for no
    row); families maps each Strategy to its family. The weight is fixed when a position is opened (`opens`: the
    Strategy was flat when it sent the order): an add or a rebalance that adds reuses it, and one whose opening entry
    wasn't seen here is weighed 1.0 (opened before weighting began)."""

    state_on: Callable[[date], str | None]
    families: Mapping[str, str]
    _held: dict[str, tuple[Decimal, str]] = field(default_factory=dict, init=False)

    def for_order(self, strategy: str, opens: bool, bar_ts: datetime) -> tuple[Decimal, str]:
        if not opens:
            return self._held.get(strategy, (FULL, UNKNOWN))
        state = self.state_on(bar_ts.date()) or UNKNOWN
        state = state if state in STATES else UNKNOWN
        out = (weight(self.families.get(strategy, ""), state), state)
        self._held[strategy] = out
        return out


def weighed(order, w: Decimal, state: str):
    """The gated order (gate_ledger.TaggedIntent) at weight w: its quantity scaled and rounded down to the venue's step,
    never up, and the weight and state it was sized under kept for the journal. An order the weight takes under the
    venue minimum is still checked, and the check refuses it, naming the minimum (portfolio.limits)."""
    qty = order.qty * w
    if order.step > 0:
        qty = (qty / order.step).to_integral_value(rounding=ROUND_DOWN) * order.step
    return replace(order, qty=qty, regime_weight=w, regime_state=state)


def from_rows(rows: Mapping[date, str]) -> Callable[[date], str | None]:
    """state_on from regime_series rows keyed by their date (one reference venue and lookback)."""
    return rows.get
