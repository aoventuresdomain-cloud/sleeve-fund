"""What turns a strategy into a sleeve: journal, PM controls and the risk guard.

A strategy gets a SleeveRuntime in paper and live; in backtest it gets none and
behaves exactly as before. The runtime is called from the strategy's own thread
(a timer and fill events), so there is no concurrency inside it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sleeve_fund import risk
from sleeve_fund.store import Store, utcnow

RECONCILE_EVERY = timedelta(hours=24)


class SleeveRuntime:
    def __init__(self, store: Store, sleeve_name: str, now=utcnow, tick_seconds: int = 30) -> None:
        self.tick_seconds = tick_seconds
        self.store = store
        # Clock source: wall clock in paper, the engine's simulated clock in backtest tests.
        self.now = now
        self.name = sleeve_name
        sleeve = store.sleeve(sleeve_name)
        self.profile = risk.profile(sleeve.risk_profile)
        self.starting_balance = sleeve.starting_balance
        self.status = sleeve.status
        self.paused_until = sleeve.paused_until
        self.peak = store.peak_equity(sleeve_name) or sleeve.starting_balance
        first = store.first_equity(sleeve_name)
        self.bench_base_price = first["price"] if first else None
        self.taker_fee = 0.008
        # The paper engine keeps fills in memory, so a restart rebuilds the book from the journal.
        self.book = store.journal_book(sleeve_name, sleeve.starting_balance)
        self.last_reconciled = None
        self._day = None
        self._day_open = None
        # Why the last tick asked for a flatten, as (intent, reason), so the sell order records it.
        self.flatten_why: tuple[str, str] | None = None

    # --- lifecycle ------------------------------------------------------------

    def on_start(self, taker_fee: float, now=None) -> None:
        self.taker_fee = taker_fee
        if now is not None:
            self.now = now
        if self.status in ("halted",):
            self.store.event(self.name, "warning", "restart", "restarted while halted; stays halted until resumed")
        elif self.status == "paused" and self.paused_until and self.paused_until > self.now():
            self.store.event(self.name, "info", "restart", "restarted while paused")
        else:
            self._set("running", "")
        if self.book["fills"]:
            self.store.event(self.name, "info", "restore",
                             f"book restored from {self.book['fills']} journal fills: "
                             f"cash {self.book['cash']:,.2f}, position {self.book['qty']:g}")
        self.store.event(self.name, "info", "start", f"sleeve started ({self.profile.name} risk profile)")

    def on_stop(self) -> None:
        if self.status in ("running", "starting"):
            self._set("stopped", "process stopped")

    # --- gates ----------------------------------------------------------------

    def can_open(self) -> bool:
        if self.status == "paused" and self.paused_until and self.paused_until <= self.now():
            self._set("running", "daily-loss pause expired")
            self.store.event(self.name, "info", "resume", "daily-loss pause expired; trading again")
        return self.status == "running"

    def position_budget(self, equity: float) -> float:
        return equity * self.profile.max_position_pct

    # --- periodic tick ----------------------------------------------------------

    def tick(self, *, equity: float, cash: float, qty: float, price: float) -> str | None:
        """Mark, guard, then apply PM commands. Returns "flatten" if the strategy must flatten now."""
        now = self.now()
        self.store.heartbeat(self.name)
        if price <= 0:
            return None
        if self.bench_base_price is None:
            self.bench_base_price = price
        # Buy and hold at the exposure this sleeve may take (its profile's position cap), the rest in
        # cash, so the comparison isn't flattered or punished by the cap itself. Same as the backtest page.
        cap = self.profile.max_position_pct
        benchmark = self.starting_balance * ((1 - cap) + cap * (1 - self.taker_fee) * price / self.bench_base_price)
        self.store.record_equity(self.name, equity=equity, cash=cash, qty=qty, price=price, benchmark=benchmark,
                                 ts=now)
        self.peak = max(self.peak, equity)
        if self._day != now.date():
            self._day, self._day_open = now.date(), equity

        flatten = False
        self.flatten_why = None
        if self.status == "running":
            breach = risk.check(self.profile, equity, self.peak, self._day_open)
            if breach and breach.action == "halt":
                self._set("halted", breach.reason)
                self.store.event(self.name, "error", "risk_halt", breach.reason + "; flattened, PM must resume")
                flatten, self.flatten_why = True, ("risk_halt", f"Risk halt: {breach.reason}")
            elif breach and breach.action == "pause_day":
                until = now + timedelta(hours=24)
                self._set("paused", breach.reason, until)
                self.store.event(self.name, "warning", "risk_pause", breach.reason + "; flattened for 24 hours")
                flatten, self.flatten_why = True, ("risk_pause", f"Daily-loss pause: {breach.reason}")

        for cmd in self.store.pending_commands(self.name):
            if cmd["command"] == "flatten":
                flatten = True
                self.flatten_why = self.flatten_why or ("pm_flatten", f"Flattened by PM: {cmd['reason']}")
                if self.status != "halted":  # never downgrade a halt
                    self._set("paused", f"flattened by PM: {cmd['reason']}")
            elif cmd["command"] == "pause":
                if self.status != "halted":
                    self._set("paused", f"paused by PM: {cmd['reason']}")
            elif cmd["command"] == "resume":
                if self.status == "halted":
                    self.peak = equity  # a resume after a halt resets the drawdown reference
                self._day_open = equity
                self._set("running", "")
            self.store.event(self.name, "info", f"pm_{cmd['command']}", cmd["reason"])
            self.store.mark_applied(cmd["id"])
        return "flatten" if flatten else None

    # --- reconciliation ----------------------------------------------------------

    def reconcile_due(self) -> bool:
        return self.last_reconciled is None or self.now() - self.last_reconciled >= RECONCILE_EVERY

    def reconcile(self, *, cash: float, qty: float, qty_tolerance: float = 1e-8) -> bool:
        """Check the engine's balances against the journal. On a mismatch, halt and alert.

        Never corrects either side: trading on a book we can't vouch for is the risk this
        guards against, so a person decides (restart to rebuild from the journal, or resume).
        """
        self.last_reconciled = self.now()
        book = self.store.journal_book(self.name, self.starting_balance)
        # Each fill can round the cash by up to a cent (notional and fee), so allow for that.
        cash_tol = 0.01 * (1 + 2 * book["fills"])
        d_cash, d_qty = cash - book["cash"], qty - book["qty"]
        detail = (f"engine cash {cash:,.2f} vs journal {book['cash']:,.2f}; "
                  f"engine position {qty:g} vs journal {book['qty']:g} ({book['fills']} fills)")
        if abs(d_cash) <= cash_tol and abs(d_qty) <= qty_tolerance:
            self.store.event(self.name, "info", "reconcile", "engine matches journal: " + detail)
            return True
        self._set("halted", "reconciliation mismatch")
        self.store.event(self.name, "error", "reconcile_mismatch",
                         detail + ". Halted, nothing traded or corrected. Restart the sleeve to rebuild "
                         "from the journal, or resume once you have checked.")
        return False

    # --- orders -------------------------------------------------------------------

    def on_order(self, *, order_id: str, side: str, qty: float, intent: str, reason: str, signal: dict) -> None:
        """Journal an order and why it was sent, before it goes to the venue."""
        self.store.record_order(self.name, order_id=order_id, side=side, qty=qty, intent=intent, reason=reason,
                                signal=signal, ts=self.now())

    def on_order_status(self, order_id: str, status: str, message: str = "") -> None:
        self.store.update_order(order_id, status=status, message=message)
        if status in ("rejected", "denied"):
            self.store.event(self.name, "warning", f"order_{status}", f"order {order_id} {status}: {message}")

    def on_fill(self, *, side: str, qty: float, price: float, fee: float, order_id: str, trade_id: str) -> None:
        self.store.record_fill(self.name, side=side, qty=qty, price=price, fee=fee, order_id=order_id,
                               trade_id=trade_id)
        self.store.update_order(order_id, fill_qty=qty, fill_px=price, fee=fee)
        self.store.event(self.name, "info", "fill", f"{side} {qty:g} @ {price:,.2f}, fee {fee:,.2f}")

    def _set(self, status: str, reason: str, paused_until: datetime | None = None) -> None:
        self.status, self.paused_until = status, paused_until
        self.store.set_status(self.name, status, reason, paused_until)
