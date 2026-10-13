"""The portfolio gate on paper's entry path (v2 P2-1b W2; plan pe2-plans/p2-1b-paper-wiring.md). A strategy's opening
order is checked against the whole fund's book in the journal (gate_ledger.DbLedger) before it is sent, the same check
and the same Gated answer as a portfolio backtest's (research.portfolio.PortfolioGate). Paper differs in two ways:

- No mark per check (HoQA w2d). The supervisor marks the fund every poll (Supervisor.mark_portfolio); a check reads
  that mark, and one older than 60 s blocks the entry as stale, so a check never moves the high-water mark or the day.
- The reservation's life is the journal's. record_order(reservation=...) links it to the order row
  (gate_decisions.order_id) in the order's own transaction, and book_fill and update_order reduce or release it in
  theirs (Store._settle_reservation), so an order's end and its reservation's end commit together (HoQA w2b, w2c).

Exits, stops, take-profits, the PM's close, risk halts and liquidations never reach check_order (HoQA F208-1, w2e):
the gate answers None for them and they are sent as they are, even while the portfolio is halted."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select

from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.money import to_decimal
from sleeve_fund.portfolio.gate import Checked, check_order
from sleeve_fund.portfolio.holding import Position, holding_for, underlying
from sleeve_fund.portfolio.limits import Holding, rejected
from sleeve_fund.research.portfolio import Gated, _utc, tagged_intent
from sleeve_fund.risk import PORTFOLIO, PortfolioProfile
from sleeve_fund.store import OPEN_ORDER_STATUSES, Store, gate_decisions_t, orders_t, utcnow

# Orders that only ever reduce a position: never checked, never reserved (HoQA F208-1).
NEVER_GATED = frozenset({"exit", "stop_loss", "take_profit", "pm_flatten", "risk_halt", "risk_pause", "liquidation",
                         "liquidation_cut"})


def fund_holdings(store: Store, now: datetime | None = None, atr_pct=None) -> tuple[Holding, ...]:
    """Every strategy's open position as the book's Holdings: its journal quantity (fills, the source of truth) at its
    latest mark, measured to the stop its entry journaled, or with none by the stopless measure from the daily ATR
    (`atr_pct(sleeve)`, the history store's by default). Leverage is the strategy's profile cap on a perpetual, None
    on spot. Raises ValueError when a stopless position's ATR isn't known: the check then fails closed."""
    from sleeve_fund import markets, risk
    from sleeve_fund.open_risk import _journal_stop, history_atr_pct

    now = now or utcnow()
    if atr_pct is None:
        def atr_pct(s):
            return history_atr_pct(s.venue or "", s.instrument, now)
    out = []
    archived = store.archived()
    for s in store.sleeves():
        if s.name in archived:
            continue
        book = store.journal_book(s.name, s.starting_balance)
        qty = to_decimal(book["qty"], "qty")
        if qty == 0:
            continue
        last = store.last_equity(s.name)
        mark = to_decimal(last["price"] if last is not None and last["price"] else book["entry_px"], "mark")
        stop = _journal_stop(store, s, float(qty))
        stop = None if stop is None else to_decimal(stop, "stop")
        through = stop is not None and (mark - stop) * qty <= 0  # gapped: the stop no longer bounds the loss
        lev = risk.profile(s.risk_profile).max_leverage if markets.is_perp(s.params) else None
        pos = Position(s.name, underlying(s.instrument), qty, None if stop is None or through else stop, lev)
        out.append(holding_for(pos, mark, None if pos.stop is not None else atr_pct(s)))
    return tuple(out)


@dataclass
class PaperGate:
    """The paper GateFn: gate(strategy, intent, ts_ns) -> Gated, or None for an order that is never gated. intent is
    the engine's (LongFlatStrategy._send_entry; research.portfolio.gate_intent reads it), plus its own `intent_id`
    and `bar` when it has them. The book is the supervisor's last mark of the fund's equity and every strategy's
    position now (fund_holdings); a book never marked is refused before it is read (stale)."""

    store: Store
    profile: PortfolioProfile = PORTFOLIO
    ledger: DbLedger = field(init=False)

    def __post_init__(self):
        self.ledger = DbLedger(self.store, self._positions)

    def _positions(self) -> tuple[Decimal, tuple[Holding, ...]]:
        equity = self.ledger.state().equity
        if equity is None:
            raise RuntimeError("the fund's book has never been marked")
        return equity, fund_holdings(self.store)

    def __call__(self, strategy: str, intent: dict, ts_ns: int) -> Gated | None:
        if intent.get("reduce_only") or intent.get("what") in NEVER_GATED:
            return None
        now = _utc(ts_ns)
        try:
            tagged = tagged_intent(intent, now)  # the portfolio backtest builds it the same way (P2-7)
        except Exception as e:  # noqa: BLE001 - an order the gate can't read is never sent
            why = f"Entry rejected: the portfolio check couldn't run ({type(e).__name__}: {e}), so nothing is sent"
            self.ledger.alert("portfolio_check_failed", f"{strategy}: {why}", now)
            qty = intent.get("qty")
            return Gated(Checked(rejected(qty if isinstance(qty, Decimal) else Decimal(0), None, why), None, None),
                         self)
        return Gated(check_order(self.ledger, strategy, tagged, self.profile, now), self)


def paper_gate(store: Store) -> PaperGate:
    return PaperGate(store)


def resting_entries(store: Store, strategy: str, before: datetime) -> list[str]:
    """A strategy's gated entry orders still open (sent, resting or part filled) decided on a bar that closed before
    `before`: the ones its process cancels at its next bar close (Advisor 8 Oct 06:10 UK). Read by the decision's bar
    (gate_decisions.bar_ts), not the order row's time, which is the journal's clock and not the run's."""
    o, d = orders_t, gate_decisions_t
    q =(select(o.c.order_id).join(d, d.c.order_id == o.c.order_id)
         .where(o.c.sleeve == strategy, o.c.status.in_(OPEN_ORDER_STATUSES), d.c.bar_ts < before).order_by(o.c.id))
    with store.engine.connect() as c:
        return [r for (r,) in c.execute(q)]

