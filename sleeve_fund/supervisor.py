"""Keeps each sleeve's paper process matching what the PM asked for.

One OS process per sleeve (a crash or leak hits one sleeve only). Every few
seconds the supervisor compares the database's desired_state with what is
running, starts or stops processes, restarts crashed ones with backoff, and
restarts any whose heartbeat goes stale.

    python -m sleeve_fund.supervisor clear configs/clear.toml
    python -m sleeve_fund.supervisor seed configs/sleeves/*.toml
    python -m sleeve_fund.supervisor run
"""

from __future__ import annotations

import argparse
import os
import tomllib
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from sleeve_fund import accounts, liquidation
from sleeve_fund.alerts import Forwarder
from sleeve_fund.exact import float_view
from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.portfolio.gate import PortfolioState, mark_book, sweep
from sleeve_fund.portfolio.limits import Book, Holding
from sleeve_fund.risk import PORTFOLIO
from sleeve_fund.paper.safety import credential_var
from sleeve_fund.paper.config import check_hub_bar_spec, load_sleeve, to_store_kwargs
from sleeve_fund.store import DUST_NOTIONAL, OPEN_ORDER_STATUSES, Sleeve, Store, is_dust, utcnow
from sleeve_fund.strategies import check_perp_sizing, check_perp_stop
from sleeve_fund.paper.runtime import entry_blocked, liquidation_head, said_since_last_fill
from sleeve_fund.strategies.base import EXITS_ONLY

POLL_SECONDS = 5
KEY_CHECK_EVERY = 12  # polls between key-presence checks: about a minute
FEE_CHECK_EVERY = 720  # polls between fee-schedule reads from connected accounts: about an hour
ALERT_EVERY = 12  # polls between alert sends and uptime pings: about a minute
CLEAR_EVERY = 12  # polls between retries of a clean slate waiting on a flatten: about a minute
HEARTBEAT_STALE = timedelta(minutes=3)
STARTUP_GRACE = timedelta(minutes=3)
MAX_BACKOFF = 300
# Flattens a reset or a clean slate queues for one strategy before it stops asking and says the PM must close
# it: a position below the venue's smallest order can't be closed by an order, and asking again every step
# only piled up commands (review round 13, m13-E1).
SYSTEM_FLATTENS = 3
# The exits-only reason of a stopped strategy that still holds a position (P1-U35): its process runs for its exits.
STOPPED_HOLDING = "it was stopped while it still holds a position"


def fund_equity(store: Store) -> Decimal:
    """The whole fund's marked equity: every strategy's, running or not, plus the cash no strategy holds (Advisor 3).
    It reads CASH-2's strategy_cash_pnl, the one cash reader; until that is on main it raises, so the portfolio's mark
    fails, its book goes stale after 60 s and no strategy opens anything (fail closed)."""
    raise NotImplementedError("the fund's equity reads CASH-2's strategy_cash_pnl, which isn't on main yet")


def fund_holdings(store: Store) -> tuple[Holding, ...]:
    """Every strategy's positions as the book's Holdings, for book_marks' figures. The paper gate's positions (P2-1b
    W2) build them; until then a fund that holds nothing has none, and one that holds anything raises."""
    if any(_holding(store, s) for s in store.sleeves()):
        raise NotImplementedError("the book's holdings come with the paper gate's positions (P2-1b W2)")
    return ()


def _holding(store: Store, s: Sleeve) -> bool:
    """Whether a strategy's journal holds a position an order can close (not dust)."""
    book = store.journal_book(s.name, s.starting_balance)
    return abs(book["qty"]) > 1e-12 and not is_dust(book)


def _status(st: PortfolioState, now: datetime) -> str:
    return "halted" if st.halted else "paused" if st.paused_until and now < st.paused_until else "ok"


@dataclass
class Proc:
    popen: subprocess.Popen | None = None
    started_at: datetime | None = None
    crashes: int = 0
    holds: bool | None = None  # a stopped strategy's journal holds a position (not dust); None: not read yet
    watched: bool = False  # the incident for a stopped strategy still holding is written (_watch_stopped_holder)
    next_start: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=utcnow().tzinfo))

    @property
    def alive(self) -> bool:
        return self.popen is not None and self.popen.poll() is None


def decide(sleeve: Sleeve, proc: Proc, now: datetime, holds: bool = False) -> str:
    """One of: start, stop, restart_stale, crashed, wait, none. A stopped strategy that still `holds` a position is
    wanted running too, for its exits only (P1-U35)."""
    want = sleeve.desired_state == "running" or holds
    if want and not proc.alive:
        if proc.popen is not None:
            return "crashed"
        return "start" if now >= proc.next_start else "wait"
    if not want and proc.alive:
        return "stop"
    if want and proc.alive and proc.started_at and now - proc.started_at > STARTUP_GRACE:
        if sleeve.heartbeat_at is None or now - sleeve.heartbeat_at > HEARTBEAT_STALE:
            return "restart_stale"
    return "none"


def check_funding_schedule(s: Sleeve) -> None:
    """Raises ValueError when a perp's newest stored settlement step is shorter than the schedule funding is charged
    on: the engine would skip settlements (funding.schedule_mismatch; Advisor, 6 Oct 2026), until DA-11."""
    from sleeve_fund import funding, markets
    from sleeve_fund.venues import DEFAULT_VENUE

    try:  # the venue as from_store reads it, without building the node's config (its warm-up reads the model)
        terms = markets.terms(s.params, getattr(s, "venue", None) or DEFAULT_VENUE)
    except ValueError:  # a market the venue doesn't list: the node's own start refuses it, with its reason
        return
    if terms is None or terms.funding_venue is None:
        return
    why = funding.schedule_mismatch(terms.funding_venue, s.instrument, terms.funding_hours, latest=True)
    if why:
        raise ValueError(why)


class Supervisor:
    def __init__(self, store: Store, python: str = sys.executable, clear_path: str | None = None) -> None:
        self.store = float_view(store)  # its figures are floats: the journal's exact ones (DA-9) read as floats
        self.python = python
        self.clear_path = clear_path  # a clean slate still waiting on a flatten finishes here, not on a redeploy
        self.procs: dict[str, Proc] = {}
        self._stopping = False
        # The portfolio gate on the exact journal; the supervisor never runs an entry check, so it reads no positions
        self.ledger = DbLedger(store, lambda: (_ for _ in ()).throw(RuntimeError("the supervisor runs no entry check")))
        self._equity_failing = False  # a failed equity read is said once per spell

    def _refused(self, name: str) -> bool:
        """A model that can't run on its market (check_perp_sizing) is not started: it is stopped, and says why,
        rather than started into a crash loop. A start that only sells a position it holds (a flatten waiting:
        the kill switch, a PM close) still goes ahead. So does one still holding a position (Independent Quant
        Advisor, NA-4, QA P1-S7): it is never left unwatched, but started for its exits only (_exits_only). One saved
        on the venue's own candles where the market data hub feeds the venue (check_hub_bar_spec, QA P1-C10) can't
        start at all, so it is refused even then."""
        s = self.store.sleeve(name)
        try:
            check_hub_bar_spec(s.venue, s.bar_spec)
        except ValueError as exc:
            qty = self.store.journal_book(name, s.starting_balance)["qty"]
            held = (f". It still holds a position ({qty:.12g}), which a flatten can't close while it can't start: "
                    "an engineer needs to close it" if abs(qty) > 1e-12 else "")
            self.store.set_desired_state(name, "stopped")
            self.store.set_status(name, "stopped", f"not started: {exc}{held}")
            self.store.event(name, "error", "start_refused", f"Not started: {exc}{held}")
            return True
        try:
            check_perp_sizing(s.strategy, s.params)
            check_perp_stop(s.strategy, s.params, s.risk_profile)
            check_funding_schedule(s)
        except ValueError as exc:
            if any(c["command"] == "flatten" for c in self.store.pending_commands(name)):
                self.store.event(name, "warning", "start_refused", f"Started only to sell its position: {exc}. The "
                                 "flatten pauses it, and it can't be started to trade")
                return False
            if self._exits_only(s, str(exc)):
                return False
            self.store.set_desired_state(name, "stopped")
            self.store.set_status(name, "stopped", f"not started: {exc}")
            self.store.event(name, "error", "start_refused", f"Not started: {exc}")
            return True
        return False

    def _exits_only(self, s: Sleeve, why: str) -> bool:
        """A refused start still holding a position (not dust) is started anyway, paused with the EXITS_ONLY
        reason: its exits run, a safety stop is set from the mark, no entry or add opens, and an incident is raised
        (LongFlatStrategy._safety_stop_on_restore). Any refusal can use it, so a position is never left without a
        process watching it. True when it applies."""
        book = self.store.journal_book(s.name, s.starting_balance)
        if abs(book["qty"]) <= 1e-12 or is_dust(book):
            return False
        reason = f"{EXITS_ONLY}: {why}"
        if s.status != "paused" or s.status_reason != reason:
            self.store.set_status(s.name, "paused", reason)
            self.store.event(s.name, "error", "start_refused", f"Not started to trade: {why}. It still holds a "
                             "position, so it is started for its exits only, with a safety stop")
            self.store.event(s.name, "error", "incident",
                             f"Incident, {s.name}: not started to trade ({why}), but it holds {book['qty']:.12g}, so it "
                             "runs for its exits only, with a safety stop from the current price and no new entries")
        return True

    def _holds(self, s: Sleeve, proc: Proc) -> bool:
        """Whether a stopped strategy still holds a position (not dust). Without a process its journal can't change,
        so that is read once per stop; while its exits-only process runs, every step, so it stops once flat."""
        if s.desired_state == "running":
            return False
        if proc.holds is None or proc.alive:
            book = self.store.journal_book(s.name, s.starting_balance)
            proc.holds = abs(book["qty"]) > 1e-12 and not is_dust(book)
        return proc.holds

    def _watch_stopped_holder(self, s: Sleeve, proc: Proc, starting: bool = False) -> None:
        """P1-U35 (Advisor 20:56, HoE): a stopped strategy still holding a position (the PM's Stop on a holder, or a
        fill that raced the stop) is never left unwatched. It runs for its exits only: its own stop, or a safety stop
        from the mark where it has none (_safety_stop_on_restore), and nothing opens, with an incident. A halt or a
        pause keeps its own status, which already holds every entry and is cleared only by its own action (HC)."""
        book = self.store.journal_book(s.name, s.starting_balance)
        proc.watched = True
        if s.status not in ("halted", "paused"):
            self.store.set_status(s.name, "paused", f"{EXITS_ONLY}: {STOPPED_HOLDING}")
        if starting:
            return  # the process it starts sets the safety stop and writes the incident, once per position
        head = f"Incident, {s.name}: stopped, but it still holds "
        if said_since_last_fill(self.store, s.name, head):
            return  # already said for this position: a deploy or a crash restarts it without a second incident
        self.store.event(s.name, "error", "incident",
                         f"{head}{book['qty']:.12g}, so it runs for its exits only: its stop (or a safety stop from the "
                         "current price) still closes it, and nothing new opens. Flatten closes it; once it is flat it "
                         "stops.")

    def _start(self, name: str, proc: Proc) -> None:
        s = self.store.sleeve(name)
        if s.desired_state != "running":  # a stopped strategy still holding: its exits only (P1-U35)
            self._watch_stopped_holder(s, proc, starting=True)
        elif self._refused(name):
            return
        # Paper processes never need a venue key, so they don't inherit one.
        env = {k: v for k, v in os.environ.items() if not credential_var(k)}
        reload = self.store.pending_reload(name)
        proc.popen = subprocess.Popen([self.python, "-m", "sleeve_fund.paper", "--db-sleeve", name], env=env)
        proc.started_at = utcnow()
        self.store.event(name, "info", "process_start", f"paper process started (pid {proc.popen.pid})")
        if reload:  # a fresh process reads the settings as they are now
            self.store.mark_applied(reload["id"])
            self.store.event(name, "info", "settings_applied", "restarted to trade under the changed settings")

    def _stop(self, name: str, proc: Proc, why: str) -> None:
        if proc.alive:
            proc.popen.send_signal(signal.SIGINT)
            try:
                proc.popen.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.popen.kill()
                proc.popen.wait()
        proc.popen, proc.holds, proc.watched = None, None, False
        self.store.event(name, "info", "process_stop", why)

    def reset_pending(self) -> None:
        """Carry each PM reset forward (5 Oct 2026): a strategy still holding is flattened first (a PM flatten,
        which pauses it; started if stopped so the flatten can trade); once flat its process is stopped, the run
        so far is put away under its own name (Store.split_run), and the strategy starts again at its starting
        capital if it was running. A strategy copied to a demo account by quantity then has its demo copy resynced (flat, on
        the paper margin terms) before it trades again."""
        for req in self.store.pending_resets():
            name = req["sleeve"]
            if liquidation_head(self.store, name) is not None:
                # Asked for before the liquidation landed (its flatten maybe parked on the liquidating tick):
                # carried out now, it would put the liquidation away unanswered (U27, QA P1-U33, P1-D24). It is
                # closed unrun, its flatten and start taken back, and the halt waits for Reset after liquidation.
                self.store.refuse_reset(req, liquidation.REFUSAL)
                self._undo_reset_start(req)
                continue
            s = self.store.sleeve(name)
            pending = self.store.pending_commands(name)
            book = self.store.journal_book(name, s.starting_balance)
            qty = book["qty"]
            dust = is_dust(book)
            if abs(qty) > 1e-12 and not dust:
                why = f"Reset strategy: {req['reason']}"
                if not any(c["command"] == "flatten" for c in pending) and _flatten_again(self.store, name, why,
                                                                                         req["created_at"], qty):
                    self.store.command(name, "flatten", why, actor=req["actor"], holds_through_reset=False)
                    if s.desired_state != "running":
                        self.store.set_desired_state(name, "running")
                continue
            if dust:
                self.store.event(name, "warning", "reset_dust",
                                 f"The {qty:.12g} still held is worth under {DUST_NOTIONAL:g}, below any venue's smallest "
                                 "order, so no flatten can close it: the reset treats it as flat and it stays with the "
                                 "run put away")
            self._stop(name, self.procs.setdefault(name, Proc()), "reset by PM")
            self.store.drop_pending(name, "lapsed: the strategy was reset")
            run = self.store.split_run(req, dust_ok=dust)
            self.store.set_desired_state(name, "running" if req["restart"] else "stopped")
            self.store.decide("system", "reset", f"Started afresh at {s.starting_balance:,.0f}; the run before is "
                              f"kept as {run} under Previous book", name)
            self.store.event(name, "info", "reset", f"Reset by the PM ({req['reason']}): the run so far is kept as "
                             f"{run}; starting again at {s.starting_balance:,.0f}")
            if s.params.get("demo_mirror"):
                self.store.queue_resync(name, f"Reset strategy: {req['reason']}")

    def _undo_reset_start(self, req: dict) -> None:
        """A reset refused after its first pass has already queued its flatten and, for a strategy the PM had
        stopped, started it so the flatten could trade. Take both back: the flatten lapses, and the Stop the
        PM chose stands (Code review on #167)."""
        name = req["sleeve"]
        for cmd in self.store.pending_commands(name):
            if cmd["command"] == "flatten" and cmd["reason"].startswith("Reset strategy:"):
                self.store.mark_applied(cmd["id"])
                self.store.decide("system", "drop flatten", "lapsed: the reset it was for was not carried out "
                                  f"({cmd['reason']})", name)
        if not req["restart"] and self.store.sleeve(name).desired_state == "running":
            self.store.set_desired_state(name, "stopped")

    def step(self) -> None:
        self.reset_pending()
        now = utcnow()
        for sleeve in self.store.sleeves():
            proc = self.procs.setdefault(sleeve.name, Proc())
            holds = self._holds(sleeve, proc)
            action = decide(sleeve, proc, now, holds)
            exits_only = sleeve.status == "paused" and sleeve.status_reason == f"{EXITS_ONLY}: {STOPPED_HOLDING}"
            if action == "start":
                self._start(sleeve.name, proc)
            elif action == "crashed":
                code = proc.popen.returncode
                proc.popen = None
                proc.crashes += 1
                delay = min(MAX_BACKOFF, 10 * 2 ** (proc.crashes - 1))
                proc.next_start = now + timedelta(seconds=delay)
                if sleeve.status in ("paused", "halted"):
                    # A crash must not lift a pause or a halt: the status (and a pause's end time) stays, and
                    # the restarted process keeps it (review round 10, B10-3). The crash is in the events.
                    self.store.event(sleeve.name, "error", "process_crash",
                                     f"exit code {code}; restart in {delay}s, still {sleeve.status}")
                else:
                    self.store.set_status(sleeve.name, "error", f"process exited with code {code}; restarting in {delay}s")
                    self.store.event(sleeve.name, "error", "process_crash", f"exit code {code}; restart in {delay}s")
            elif action == "stop":
                self._stop(sleeve.name, proc, "stopped by PM")
                if not entry_blocked(self.store, sleeve.name, starting=True)[0]:  # a halt stays through a stop (HC)
                    self.store.set_status(sleeve.name, "stopped", "stopped by PM")
            elif action == "restart_stale":
                self.store.event(sleeve.name, "error", "heartbeat_stale", "no heartbeat for 3 minutes; restarting")
                self._stop(sleeve.name, proc, "restart after stale heartbeat")
                self._start(sleeve.name, proc)
            elif action == "none" and holds and sleeve.status not in ("halted", "paused"):
                # The PM's Stop on a strategy that still holds: restarted for its exits only (P1-U35)
                self._stop(sleeve.name, proc, "stopped by PM: restarted for its exits only, as it still holds a position")
                self._start(sleeve.name, proc)
            elif action == "none" and holds and not proc.watched:
                # Stopped while halted or paused and still holding: its process keeps running, as it is (P1-U35)
                self._watch_stopped_holder(sleeve, proc)
            elif action == "none" and exits_only and sleeve.desired_state == "running":
                # Started again by the PM after a stop that left it running for its exits only: it trades again
                self._stop(sleeve.name, proc, "started by PM: restarted to trade")
                self.store.set_status(sleeve.name, "stopped", "started by PM")
                self._start(sleeve.name, proc)
            elif action == "none" and proc.alive and self.store.pending_reload(sleeve.name):
                self._stop(sleeve.name, proc, "restart for changed settings")
                self._start(sleeve.name, proc)
            elif action == "none" and proc.alive and proc.crashes and now - proc.started_at > STARTUP_GRACE:
                proc.crashes = 0  # healthy again

    def mark_portfolio(self, now: datetime | None = None) -> None:
        """The portfolio's poll (v2 P2-2): mark the whole fund (portfolio.gate.mark_book), a book_marks row each minute
        and at every status change, act on a halt, and sweep orphaned reservations. The state row is written before
        the equity is read, so a read that fails leaves the gate in force and entries blocked once the mark is 60 s
        old. A pause needs nothing here: CHOKE reads it from the state (runtime.portfolio_block) until 00:00 UTC."""
        now = now or utcnow()
        led = self.ledger
        led.ensure_state()
        before = led.state()
        try:
            equity = fund_equity(self.ledger.store)
        except Exception as exc:  # noqa: BLE001 - no mark: the book goes stale and entries stop (fail closed)
            if not self._equity_failing:
                self.store.event(None, "error", "supervisor_error", f"the fund's equity couldn't be read, so the "
                                 f"portfolio isn't marked and entries stop once its mark is 60 s old: {exc!r}")
            self._equity_failing = True
        else:
            self._equity_failing = False
            acted = mark_book(led, equity, now, PORTFOLIO)
            st = led.state()
            last = led.last_book_mark()
            minute = now.replace(second=0, microsecond=0)
            if last is None or last.replace(second=0, microsecond=0) != minute or _status(before, now) != _status(st, now):
                try:
                    led.write_book_mark(now, Book(equity, fund_holdings(self.ledger.store)), st, PORTFOLIO.version)
                except NotImplementedError:
                    pass  # no holdings yet (P2-1b W2): the cache row waits; the state row above is the gate's
            if acted == "halt":
                self._halt_every_strategy(st.halted)
        sweep(led, now, self._order_live, self._cancel_order)

    def _halt_every_strategy(self, why: str) -> None:
        """A portfolio halt (15% under the reference): every strategy holding anything is flattened through its exit
        path, once per halt; CHOKE blocks every entry from the state row until the PM resumes the portfolio."""
        reason = f"Portfolio halted: {why}"
        for s in self.store.sleeves():
            if _holding(self.store, s):
                self.store.command(s.name, "flatten", reason, actor="supervisor")

    def _order_live(self, order_id: str) -> bool:
        """Whether an order may still fill, read through the ledger's own transaction: inside the sweep's lock a second
        connection would, on SQLite's one shared connection, roll the sweep's own writes back (HoQA, P2-1b cells)."""
        return self.ledger.order_status(order_id) in OPEN_ORDER_STATUSES

    def _cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("asking a strategy's process to cancel one order comes with the paper gate (P2-1b "
                                  "W2)")

    def check_keys(self) -> None:
        """Tell the dashboard which live accounts have their venue's key on this server (presence only)."""
        live = [a for a in self.store.accounts() if a["kind"] == "live"]
        if live:
            self.store.report_keys({a["name"]: accounts.key_present(a["name"], venue=a["venue"]) for a in live})

    def check_fees(self, environ=None) -> None:
        """Read each connected live account's fee schedule from its venue, so backtests and paper
        charge what the exchange actually charges that account. Query-only; the key never leaves
        this process."""
        from sleeve_fund.venues import venue as venue_profile

        for a in self.store.accounts():
            if a["kind"] != "live":
                continue
            creds = accounts.credentials(a["name"], a["venue"], environ)
            profile = venue_profile(a["venue"])
            if creds is None or profile.fetch_fees is None:
                continue
            try:
                fees = profile.fetch_fees(*creds)
                self.store.record_fees(profile.name, a["name"], float(fees.maker), float(fees.taker))
            except Exception as exc:  # noqa: BLE001 - keep the last good schedule; show why on the dashboard
                msg = str(exc).replace(creds[0], "***").replace(creds[1], "***")
                self.store.event(None, "warning", "fee_fetch_failed", f"{a['name']}: {msg}")

    def _send_alerts(self, alerts: Forwarder) -> None:
        try:
            alerts.step()
        except Exception as exc:  # noqa: BLE001 - the dashboard shows it; the next minute tries again
            self.store.event(None, "error", "supervisor_error", f"alerts: {exc!r}")

    def run(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stopping", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stopping", True))
        alerts = Forwarder(self.store)
        self.store.event(None, "info", "supervisor_start", "supervisor started")
        self.store.event(None, "info", "alerts_config", alerts.describe())
        loops, sender = 0, None
        while not self._stopping:
            try:
                if loops % ALERT_EVERY == 0 and (sender is None or not sender.is_alive()):
                    # Its own thread: a slow webhook or monitor never holds up supervising the sleeves.
                    sender = threading.Thread(target=self._send_alerts, args=(alerts,), name="alerts", daemon=True)
                    sender.start()
                if loops % KEY_CHECK_EVERY == 0:
                    self.check_keys()
                if loops % FEE_CHECK_EVERY == 0:
                    self.check_fees()
                if self.clear_path and loops % CLEAR_EVERY == 0 and os.path.exists(self.clear_path):
                    clear(self.store, self.clear_path)
                loops += 1
                self.step()
                self.mark_portfolio()
            except Exception as exc:  # keep supervising; the dashboard shows the error
                self.store.event(None, "error", "supervisor_error", repr(exc))
            time.sleep(POLL_SECONDS)
        for name, proc in self.procs.items():
            self._stop(name, proc, "supervisor shutting down")


def seed(store: Store, paths: list[str]) -> list[str]:
    """Insert sleeves from TOML files that aren't in the database yet. Never overwrites. A file with
    `start = false` under [sleeve] adds its strategy stopped, for the PM to start from the dashboard.
    One exception for a strategy already there: a file asking for the demo mirror turns it on when the strategy
    has never had that setting (the first perpetual strategies were added before the mirror could copy them,
    5 Oct 2026). It only tells the mirror to copy; the strategy itself is not restarted or changed."""
    existing = {s.name for s in store.sleeves()}
    added = []
    for path in paths:
        cfg = load_sleeve(path)
        if cfg.name in existing:
            if cfg.params.get("demo_mirror") and store.add_missing_param(cfg.name, "demo_mirror", True):
                store.decide("system", "mirror", f"demo mirror turned on from {path}", cfg.name)
            continue
        check_perp_sizing(cfg.strategy, cfg.params)
        # A stopless perp above 1x is seeded as written and refused when started (_refused), with the reason: a
        # shipped file must not stop the supervisor from coming up.
        with open(path, "rb") as fh:
            start = tomllib.load(fh).get("sleeve", {}).get("start", True)
        store.create_sleeve(**to_store_kwargs(cfg), desired_state="running" if start else "stopped")
        store.decide("system", "create", f"seeded from {path}" + ("" if start else " (stopped, to be started by the PM)"),
                     cfg.name)
        added.append(cfg.name)
    return added


def _flatten_again(store: Store, name: str, why: str, since, qty: float) -> bool:
    """Whether a reset or a clean slate may queue another flatten (`why`) for a strategy still holding qty: at
    most SYSTEM_FLATTENS since `since` (the reset's request, or the last clean slate finished, so an earlier slate
    with the same reason doesn't count). The last refusal says once what is left and that the PM must close it."""
    asked = sum(1 for d in store.decisions(name, limit=1_000, action="flatten", since=since)
                if d["reason"] == why.strip())
    if asked < SYSTEM_FLATTENS:
        return True
    if not any(e["kind"] == "flatten_gave_up" and e["message"].endswith(why)
               for e in store.events(name, limit=200)):
        store.event(name, "error", "flatten_gave_up",
                    f"still holding {qty:.12g} after {SYSTEM_FLATTENS} flattens, perhaps less than the venue's smallest "
                    f"order; the PM must close it before this can finish: {why}")
    return False


def clear(store: Store, path: str) -> list[str]:
    """Put away the strategies on the book, once per [[clear]] entry in the file: each is stopped and
    archived, and its journal stays as it is (nothing is deleted). An entry's optional `keep` list names
    strategies it leaves alone. An entry already applied is skipped, so this can run on every start;
    strategies added after it are never touched. A strategy still holding a position is not archived, since
    archiving would drop a position nobody then watches from the book (review round 11): it is flattened
    instead (a PM flatten, which also pauses it, so it opens nothing new), started if it was stopped so the
    flatten can trade, and the entry stays open to finish once it is flat: the supervisor retries it about
    every minute, and a retry touches only the strategies it was waiting on, never one added since."""
    with open(path, "rb") as fh:
        entries = tomllib.load(fh).get("clear", [])
    done = {d["reason"].split(":", 1)[0] for d in store.decisions(action="clear", limit=10_000)}
    cleared = []
    for entry in entries:
        key, reason = str(entry["id"]), str(entry["reason"]).strip()
        keep = set(entry.get("keep", []))
        if key in done:
            continue
        put_away = store.archived()
        waiting = {e["sleeve"] for e in store.events_of(("clear_held",), limit=10_000)
                   if e["message"].startswith(f"{reason}:")}
        holding = []
        for s in store.sleeves():
            if s.name in keep or (waiting and s.name not in waiting):
                continue
            qty = store.journal_book(s.name, s.starting_balance)["qty"]
            if s.name in put_away and abs(qty) <= 1e-12 and s.name not in waiting:
                continue  # put away flat by an earlier slate
            # One archived while still holding (the 4 Oct slate, before holders were flattened first) is
            # flattened like any other holder: until it is flat it stays in the book's figures.
            if abs(qty) > 1e-12:
                holding.append(s.name)
                why = f"{reason}: flattened so it can be archived"
                if (not any(c["command"] == "flatten" for c in store.pending_commands(s.name))
                        and _flatten_again(store, s.name, why, store.book_start(), qty)):
                    store.command(s.name, "flatten", why, actor="system", holds_through_reset=False)
                if s.desired_state != "running":
                    store.set_desired_state(s.name, "running")
                    store.decide("system", "start", f"{reason}: started only to flatten its position", s.name)
                store.event(s.name, "warning", "clear_held",
                            f"{reason}: not archived yet, it still holds {qty:.12g}; it is being flattened and "
                            "paused, and the next start archives it")
                continue
            if s.desired_state != "stopped":
                store.set_desired_state(s.name, "stopped")
                store.drop_pending(s.name, "lapsed: the strategy was stopped before it acted")
                store.decide("system", "stop", reason, s.name)
            store.archive(s.name)
            store.decide("system", "archive", reason, s.name)
            cleared.append(s.name)
        if holding:
            continue  # applied again on the next start, once those are flat
        store.decide("system", "clear", f"{key}: {reason} ({len(cleared)} put away)")
        store.event(None, "info", "book_cleared", f"{reason}: {', '.join(cleared) or 'nothing to put away'}")
        done.add(key)
    return cleared


def book_line(store: Store) -> str:
    """The current book in one line for the deploy log: each strategy with its starting balance and
    whether it has traded or been marked yet, so a fresh book can be checked without the dashboard."""
    archived = store.archived()
    parts = []
    for s in store.sleeves():
        if s.name in archived:
            qty = store.journal_book(s.name, s.starting_balance)["qty"]
            if abs(qty) > 1e-12:  # still counted in the book until it is flat
                parts.append(f"{s.name} {s.starting_balance:,.0f} (archived, still holding {qty:.12g}, {s.desired_state})")
            continue
        history = "has history" if store.fills(s.name, limit=1) or store.last_equity(s.name) else "no history"
        parts.append(f"{s.name} {s.starting_balance:,.0f} ({s.desired_state}, {history})")
    return "; ".join(parts) or "empty"


def book_figures(store: Store) -> str:
    """The book's headline figures as the Portfolio counts them (every strategy not in an earlier book):
    starting capital, equity, fees, open positions and the first mark, for the deploy log."""
    earlier = store.previous_book()
    current = [s for s in store.sleeves() if s.name not in earlier]
    start = sum(s.starting_balance for s in current)
    equity = sum(float((store.last_equity(s.name) or {"equity": s.starting_balance})["equity"]) for s in current)
    fees = sum(float(f["fee"]) for s in current for f in store.fills(s.name, limit=1_000_000))
    held = [s.name for s in current if abs(store.journal_book(s.name, s.starting_balance)["qty"]) > 1e-12]
    firsts = [m["ts"] for s in current if (m := store.first_equity(s.name))]
    return (f"from {start:,.2f}, equity {equity:,.2f}, fees {fees:,.2f}, "
            f"positions {', '.join(held) or 'none'}, first mark {min(firsts).isoformat() if firsts else 'none'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.supervisor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sd = sub.add_parser("seed", help="add sleeves from TOML files if missing")
    sd.add_argument("paths", nargs="+")
    cl = sub.add_parser("clear", help="stop and archive every strategy, once per entry in the file")
    cl.add_argument("path")
    sub.add_parser("book", help="print the book's strategies and headline figures")
    rn = sub.add_parser("run", help="supervise sleeve processes until stopped")
    rn.add_argument("--clear", help="clean slates file to retry while one waits on a flatten")
    args = ap.parse_args(argv)
    store = Store()
    if args.cmd == "seed":
        print("added:", seed(store, args.paths) or "nothing new")
    elif args.cmd == "clear":
        print("put away:", clear(store, args.path) or "nothing")
        print("book:", book_line(store))
    elif args.cmd == "book":
        print("book:", book_line(store))
        print("book figures:", book_figures(store))
    else:
        Supervisor(store, clear_path=args.clear).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
