"""P2-7 paper replay parity: paper's gate decisions, as the journal holds them, lined up against the portfolio
backtest's over the same bars, and the fill-cost line that tells the ACT-DRIFT trigger when to fire.

Pure: no store, no engine. The caller reads the journal (DbLedger.replayable()) and runs run_portfolio over the paper
window, then hands both lists here. Decisions are keyed on (strategy, bar_ts), bar_ts being the deciding bar's close;
two decisions on one key pair in the order each side made them. Only 'submit' stage rows are compared: a backtest has
no fill-stage backstop, and 'error' rows never reach replayable(). A paper entry sent outside a bar decision (a
rebalance add, a resumed entry) carries the check time as its bar_ts (PE2, W2) and no signal.bar on its order row: the
caller names those intents (`unbarred`), and they are listed on their own line, never matched or mismatched. The
backtest's own such orders are marked in its decisions (bar_decided) and listed apart the same way."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from sleeve_fund.gate_ledger import Journaled
from sleeve_fund.research import portfolio

# ACT-DRIFT (Independent Quant Advisor, 7 Oct 20:05 UK): a drift is set only once the mean adverse drift is above
# 2.5 bp over 200 or more fills.
DRIFT_TRIGGER_BP = 2.5
DRIFT_MIN_FILLS = 200

FIELDS = ("outcome", "limit", "requested", "approved")  # compared in this order; the first is the likeliest cause


@dataclass(frozen=True)
class Decided:
    """One gate decision, from either side, in the terms both share."""

    strategy: str
    bar_ts: datetime
    outcome: str
    requested: Decimal
    approved: Decimal
    limit: str | None
    bar_decided: bool = True  # False: sent outside a bar decision, so bar_ts is only the check time


@dataclass(frozen=True)
class Mismatch:
    """One key where the two sides differ. what: 'paper only', 'backtest only', or the FIELDS that differ."""

    strategy: str
    bar_ts: datetime
    what: tuple[str, ...]
    paper: Decided | None
    backtest: Decided | None


@dataclass(frozen=True)
class Drift:
    """Mean adverse drift of real fills from the expected fill, in basis points, over n fills; act: the ACT-DRIFT
    trigger is met, so ACT_DRIFT_BP should be set (a decision for the HoE and the Advisor, never automatic)."""

    n: int
    mean_bp: float | None
    act: bool


@dataclass(frozen=True)
class ParityReport:
    paper: int
    backtest: int
    matched: int
    mismatches: list[Mismatch] = field(default_factory=list)
    unbarred: list[Decided] = field(default_factory=list)  # paper entries not decided on a bar: listed, not compared
    unbarred_backtest: list[Decided] = field(default_factory=list)  # the backtest's likewise

    @property
    def clean(self) -> bool:
        return not self.mismatches

    def counts(self) -> dict[str, int]:
        """Mismatches by kind: each field that differs counts once per key."""
        out: dict[str, int] = {}
        for m in self.mismatches:
            for w in m.what:
                out[w] = out.get(w, 0) + 1
        return out


def from_journal(rows: Iterable[Journaled], unbarred: Collection[str] = ()) -> list[Decided]:
    """Paper's side: the journal's replayable rows at stage 'submit'. unbarred: the intent_ids whose order row has no
    signal.bar (sent outside a bar decision)."""
    return [Decided(r.strategy, r.bar_ts, r.decision.outcome, r.decision.requested_qty, r.decision.approved_qty,
                    r.decision.limit_hit, r.intent_id not in unbarred) for r in rows if r.stage == "submit"]


def from_backtest(decisions: Iterable[dict]) -> list[Decided]:
    """The backtest's side: PortfolioResult.decisions."""
    return [Decided(d["strategy"], d["bar_ts"], d["outcome"], d["requested"], d["approved"], d["limit"],
                    d.get("bar_decided", True)) for d in decisions]


def _within(rows: list[Decided], start: datetime | None, end: datetime | None) -> list[Decided]:
    return [r for r in rows if (start is None or r.bar_ts >= start) and (end is None or r.bar_ts <= end)]


def _keyed(rows: list[Decided]) -> dict[tuple[str, datetime], list[Decided]]:
    out: dict[tuple[str, datetime], list[Decided]] = {}
    for r in rows:
        out.setdefault((r.strategy, r.bar_ts), []).append(r)
    return out


def compare(paper: list[Decided], backtest: list[Decided], start: datetime | None = None,
            end: datetime | None = None) -> ParityReport:
    """Every decision on either side between start and end (bar closes, inclusive), matched on (strategy, bar_ts).
    Quantities are compared exactly: both sides size on the same step, so any difference is a real one."""
    p, b = _within(paper, start, end), _within(backtest, start, end)
    unbarred, p = [r for r in p if not r.bar_decided], [r for r in p if r.bar_decided]
    unbarred_b, b = [r for r in b if not r.bar_decided], [r for r in b if r.bar_decided]
    pk, bk = _keyed(p), _keyed(b)
    matched, out = 0, []
    for key in sorted(pk.keys() | bk.keys()):
        ps, bs = pk.get(key, []), bk.get(key, [])
        for i in range(max(len(ps), len(bs))):
            x, y = (ps[i] if i < len(ps) else None), (bs[i] if i < len(bs) else None)
            if y is None:
                out.append(Mismatch(*key, ("paper only",), x, None))
            elif x is None:
                out.append(Mismatch(*key, ("backtest only",), None, y))
            else:
                what = tuple(f for f in FIELDS if getattr(x, f) != getattr(y, f))
                if what:
                    out.append(Mismatch(*key, what, x, y))
                else:
                    matched += 1
    return ParityReport(len(p), len(b), matched, out, unbarred, unbarred_b)


def adverse_bp(side: int, expected: Decimal, fill: Decimal) -> float:
    """How far a fill landed against the order from its expected fill, in bp: positive when it cost more (a buy
    above, a sell below)."""
    if side not in (1, -1):
        raise ValueError(f"side is 1 or -1, got {side}")
    if expected <= 0:
        raise ValueError(f"an expected fill is above 0, got {expected}")
    return float(side * (fill - expected) / expected * 10_000)


def drift(fills: Iterable[tuple[int, Decimal, Decimal]]) -> Drift:
    """fills: (side, expected fill, real fill) for each opening fill paper made. The expected fill is the price the
    gate was handed (fill_price at the bar's close, with the ACT_DRIFT_BP then in force taken back out, so the line
    measures drift before any correction)."""
    bps = [adverse_bp(s, e, f) for s, e, f in fills]
    if not bps:
        return Drift(0, None, False)
    mean = sum(bps) / len(bps)
    return Drift(len(bps), mean, len(bps) >= DRIFT_MIN_FILLS and mean > DRIFT_TRIGGER_BP)


def uncorrected(close: Decimal, side: int, half_spread: float) -> Decimal:
    """The expected fill with no act drift: what drift() measures against."""
    return portfolio.fill_price(close, side, half_spread, drift_bp=0.0)
