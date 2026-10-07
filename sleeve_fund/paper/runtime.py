"""What turns a strategy into a sleeve: journal, PM controls and the risk guard.

A strategy gets a SleeveRuntime in paper and live, and backtests run with the same
runtime on an in-memory journal (SleeveRuntime.for_backtest), so sizing, halts, pauses
and the journal behave identically in every mode. Research runs without one. The runtime
is called from the strategy's own thread (a timer and fill events), so there is no
concurrency inside it.
"""

from __future__ import annotations

import math

from decimal import Decimal

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sleeve_fund import risk
from sleeve_fund.store import LIQUIDATION_RESET, OPEN_ORDER_STATUSES, RELOAD, Store, replay_book, utcnow

FLATTEN_RETRIES = 3  # times a flatten that did not close the position is sent again before the PM is asked
RECONCILE_EVERY = timedelta(hours=24)
# How often the typical spread is recorded from live quotes, and the fewest quotes worth a reading.
SPREAD_EVERY = timedelta(hours=1)
FEED_WRITE_EVERY = timedelta(seconds=3)  # the price feed age on the strategy page is at most this stale
SPREAD_MIN_SAMPLES = 100
# A liquidation order working this long with the PM's commands waiting behind it is stuck: an incident says so once
# (QA P1-D23). Paper fills a liquidation at once; at a venue one can hang.
LIQ_STUCK = timedelta(minutes=5)
LIQ_STUCK_HEAD = "A liquidation order is still working"


WIPED_OUT = "Position margin lost (liquidated)"  # how a liquidation's halt begins (LongFlatStrategy._margin_lost)
RESET_AFTER_LIQUIDATION = LIQUIDATION_RESET  # what a reset after liquidation journals (#164)


def liquidation_reason(head: str, covered: float) -> str:
    """A liquidation's halt once flat: its head, and what the venue's insurance fund covered, if anything."""
    return head + (f"; the venue's insurance fund covered the {covered:,.2f} shortfall" if covered > 0 else "")

def fold(text: str | None) -> str:
    """Text for a loose match: one plain space between words, in any case (a no-break space counts as one)."""
    return " ".join((text or "").split()).casefold()


LIQUIDATED_WORDS = (fold(WIPED_OUT), "wiped out", "position margin lost")  # a liquidation's halt, in any wording


class Why(str):
    """Why nothing opens, in words, with `code` saying which state it is, so callers key off the code and never the
    words: liquidated, halted, daily_pause, retired, hold (the engine's own: stale data, missing funding), paused or
    stopped. It compares and prints as its words."""
    code: str

    def __new__(cls, text: str, code: str) -> Why:
        why = super().__new__(cls, text)
        why.code = code
        return why


LIQUIDATED = Why("it was liquidated, and only a reset after liquidation clears that", "liquidated")


def clearing_action(sleeve, now: datetime | None = None) -> Why | None:
    """What alone clears the halt a strategy is in, in words, or None when it isn't halted or paused for the day
    (Independent Quant Advisor, 6 Oct 18:17, HC): a liquidation only a reset after liquidation; a drawdown halt only a
    resume; a daily-loss pause only the next 00:00 UTC roll. Stop, Start and restarts never do. Without `now` a daily
    pause counts until the runtime itself rolls it; with it, one whose roll has passed doesn't, so a strategy stopped
    across the roll can still be started (its first tick then lifts the pause)."""
    reason = sleeve.status_reason or ""
    if sleeve.status == "halted":
        if fold(reason).startswith(LIQUIDATED_WORDS):
            return LIQUIDATED
        return Why(f"it is halted ({reason or 'by its risk limits'}), and only a resume clears that", "halted")
    if sleeve.status == "paused" and sleeve.paused_until is not None and (now is None or sleeve.paused_until > now):
        return Why(f"it is paused for the day's loss until {sleeve.paused_until:%d %b %H:%M} UTC, and only the next "
                   "00:00 UTC roll clears that", "daily_pause")
    return None


# A block episode in the journal (Advisor 22:29): one alert when nothing may open any more, naming why, and one cleared
# event when it ends, with how many orders it refused. Each refused order is its own decision row.
BLOCK_STARTED, BLOCK_CLEARED, BLOCK_PREFIX = "entry_blocked", "entry_block_cleared", "Nothing opens: "
REFUSED = "entry_blocked"  # the decision log's action for an order the gate refused (QA's exposure-gate master)
RETIRED = Why("it is retired (archived), and only Restore clears that", "retired")


def blocked_state(sleeve, now: datetime | None = None, *, liquidated: str | None = None, archived: bool = False,
                  holds: dict[str, str] | None = None, starting: bool = False) -> tuple[bool, Why | None]:
    """CHOKE (HoE 6 Oct 20:52, Advisor): the one "nothing opens" gate, as (blocked, why in words). Blocked: liquidated
    until a reset after liquidation (`liquidated`, from the journal: liquidation_head), a drawdown halt, the daily-loss
    pause, then (not when `starting`: the supervisor's Stop and the dashboard's Start and Resume, which act on these
    themselves) a pause with no end time, stopped, or a hold the engine has raised (`holds`, e.g. no funding rate or
    stale data). The engine checks every order that would make the position bigger,
    at submit and at fill; the supervisor and the dashboard's Start and Resume read the same answer. Stops,
    reduce-only orders, the PM's close and a liquidation are never gated. The why carries a code (Why)."""
    if liquidated is not None:
        return True, LIQUIDATED
    why = clearing_action(sleeve, now)
    if why is not None:
        return True, why
    if archived:
        return True, RETIRED
    if starting:  # Start and Resume act on a stopped or paused strategy: what they can't clear is above
        return False, None
    for why in (holds or {}).values():  # the engine's own (stale data, missing funding), ahead of pause and stop
        return True, Why(why, "hold")
    if sleeve.status == "paused" and sleeve.paused_until is None:  # the PM's pause, a flatten, the kill switch, or
        # started for its exits only (a daily-loss pause past its roll is lifted by the runtime's next tick)
        return True, Why("it is paused", "paused")
    if "stopped" in (sleeve.status, getattr(sleeve, "desired_state", "running")):
        return True, Why("it is stopped, and only Start clears that", "stopped")
    return False, None


def liquidation_head(store, name: str) -> str | None:
    """The head of a strategy's liquidation halt while it is liquidated (Independent Quant Advisor, 6 Oct 17:57), else
    None: it stays halted through a resume, Stop/Start and restarts until the PM resets it after the liquidation. The
    one liquidated rule, for the engine, the gate and the dashboard (CR 7). The journal decides, not the latest halt's
    reason, which a later halt or stop can overwrite (QA on #164): liquidated is a liquidation event or order, or a
    liquidation's halt (in any case and spacing, P1-U28a), with no reset after liquidation since."""
    reset = store.last_event(name, (RESET_AFTER_LIQUIDATION,))
    since_id, since_ts = (reset["id"], reset["ts"]) if reset else (0, None)
    since = store.sleeve_events_since(name, ("liquidation", "risk_halt"), since_id)
    heads = [e["message"].split(";")[0] for e in since
             if e["kind"] == "risk_halt" and fold(e["message"]).startswith(fold(WIPED_OUT))]
    if heads:
        return heads[-1]
    if any(e["kind"] == "liquidation" or fold(e["message"]).startswith(LIQUIDATED_WORDS) for e in since):
        return WIPED_OUT  # liquidated without the ruled halt (an older wording, or a drawdown halt first)
    order = store.last_order(name, ("liquidation",))
    if order is None or (since_ts is not None and order["ts"] < since_ts):
        return None
    if since_ts is not None and order["ts"] == since_ts:
        # A liquidation order in the same instant as the reset: orders and events share no id, so the liquidation
        # wins the tie (P1-U28b) unless the reset answered a liquidation event of that same instant.
        answered = store.last_event(name, ("liquidation",))
        if answered is not None and answered["id"] < since_id and answered["ts"] == order["ts"]:
            return None
    return WIPED_OUT


def said_since_last_fill(store, name: str, head: str) -> bool:
    """Whether an incident starting with `head` was written since the position last changed (its last fill): one
    incident per position, read only back to that fill (CR 13)."""
    last = store.fills(name, limit=1)
    return any(e["message"].startswith(head) for e in
               store.sleeve_events_since(name, ("incident",), since=last[0]["ts"] if last else None))


def open_block(store, name: str, now: datetime | None = None) -> str | None:
    """Why nothing opens, as the engine last journaled it in a block episode it hadn't closed by `now`, else None."""
    last = store.last_event(name, (BLOCK_STARTED, BLOCK_CLEARED), before=now)
    return last["message"].removeprefix(BLOCK_PREFIX) if last and last["kind"] == BLOCK_STARTED else None


def entry_blocked(store, name: str, now: datetime | None = None, *, starting: bool = False) -> tuple[bool, str | None]:
    """The gate for a strategy as its journal has it (blocked_state), for the supervisor, the dashboard and QA's
    exposure-gate set, with no process needed: the engine's own holds (stale data, missing funding) are read from the
    block episode it left open."""
    held = None if starting else open_block(store, name, now)
    return blocked_state(store.sleeve(name), now, liquidated=liquidation_head(store, name),
                         archived=name in store.archived(), holds={"engine": held} if held else None,
                         starting=starting)


class SleeveRuntime:
    # True when replaying history: no live trade feed, so the strategy ticks once a bar and
    # rests its stop at the simulated venue instead of watching every trade.
    backtest = False
    # Backtests: called with the simulated time at every tick, so a long run can report how far it is.
    progress = None

    @classmethod
    def for_backtest(cls, *, strategy: str, instrument: str, bar_spec: str, starting_balance: float,
                     risk_profile: str, params: dict | None = None, bar_seconds: int = 86_400) -> "SleeveRuntime":
        """The paper runtime on an in-memory journal, ticking on the backtest's clock once a bar.
        Store.save_backtest copies the journal into the real one when the run is worth keeping."""
        from sleeve_fund.paper.journal import MemoryJournal

        store = MemoryJournal()
        store.create_sleeve(name="backtest", strategy=strategy, instrument=instrument, bar_spec=bar_spec,
                            starting_balance=starting_balance, risk_profile=risk_profile, params=params or {})
        rt = cls(store, "backtest", tick_seconds=bar_seconds)
        rt.backtest = True
        return rt

    def risk_events(self) -> list[dict]:
        """Halts and pauses, oldest first, for a backtest to show."""
        kinds = ("risk_halt", "risk_pause", "resume", "reconcile_mismatch", "liquidation", "liquidation_cut")
        return [e for e in reversed(self.store.events(self.name, limit=10_000)) if e["kind"] in kinds]

    def __init__(self, store: Store, sleeve_name: str, now=utcnow, tick_seconds: int = 30) -> None:
        self.tick_seconds = tick_seconds
        self.store = store
        # Clock source: wall clock in paper, the engine's simulated clock in backtest tests.
        self.now = now
        self.name = sleeve_name
        sleeve = store.sleeve(sleeve_name)
        self.profile = risk.profile(sleeve.risk_profile)
        self.cap = risk.position_cap(self.profile, sleeve.params)  # on a perp, the margin cap times the leverage cap
        self.starting_balance = sleeve.starting_balance
        self.status = sleeve.status
        self.paused_until = sleeve.paused_until
        self.peak = self._restored_peak(sleeve.starting_balance)
        self.liquidated = self._last_liquidation()  # its halt, when the last one was a liquidation
        self.wiped_out = self.liquidated is not None
        first = store.first_equity(sleeve_name)
        self.bench_base_price = first["price"] if first else None
        self.taker_fee = 0.008
        # The paper engine keeps fills in memory, so a restart rebuilds the book from the journal.
        self.book = store.journal_book(sleeve_name, sleeve.starting_balance)
        self.last_reconciled = None
        self._day = None
        self._day_open = None
        self._last_equity = None
        self._feed_written = None
        self._spreads: list[float] = []
        self._spread_since = None
        # Why the last tick asked for a flatten, as (intent, reason), so the sell order records it.
        self.flatten_why: tuple[str, str] | None = None
        # A flatten the last process sent but may not have seen filled, owed on the first tick (sanity S-3).
        self._owed_flatten: tuple[str, str] | None = None
        self._flatten_retries = 0
        # The smallest position the strategy can close (its lot or the venue's minimum, set at start): less
        # than this is dust a flatten can't sell, so it owes nothing (sanity, 4 Oct).
        self.close_floor = 0.0
        # Why the engine holds new entries back for now, by what raised it (e.g. "funding", "data"): CHOKE's
        # entry_blocked reads them. Each holder sets and clears its own key.
        self.holds: dict[str, str] = {}
        self._block = open_block(store, sleeve_name)  # the block episode open in the journal, by its reason
        # Why a fill that raced a halt, a liquidation or the daily pause is owed a flatten (the strategy sets it on
        # such a fill; Advisor 23:05, SG7): the block's own flatten covers what filled after it, through the exit path.
        self.raced: str | None = None
        self.liq_working_since: datetime | None = None  # when the liquidation order now working was first seen
        self._refused = 0

    def _last_liquidation(self) -> str | None:
        """The head of its liquidation halt while it is liquidated, else None (liquidation_head)."""
        return liquidation_head(self.store, self.name)

    def _restored_peak(self, starting_balance: float) -> float:
        """The drawdown reference after a (re)start: the highest mark since the PM last resumed from a
        halt, which reset it, else the highest mark ever. Without the reset a settings edit after such a
        resume re-halted and flattened at once (review round 9, M9-2)."""
        reset = self.store.last_event(self.name, ("drawdown_reset",))
        if reset is None:
            return self.store.peak_equity(self.name) or starting_balance
        # The reset's own mark was taken a moment before its event, on the tick that applied the resume.
        at = self.store.equity_at_or_before(self.name, reset["ts"])
        marks = [self.store.peak_equity(self.name, since=reset["ts"]), at["equity"] if at else None]
        return max((float(m) for m in marks if m is not None), default=starting_balance)

    def _restored_day_open(self, now: datetime, equity: float) -> float:
        """The daily-loss baseline after a (re)start: the equity at the PM's last resume today, which reset
        it (review round 9, M9-2), else the day's open (review round 8, B8-2), else this mark."""
        midnight = datetime.combine(risk.trading_day(now), datetime.min.time(), tzinfo=timezone.utc)
        resumed = self.store.last_event(self.name, ("pm_resume",))
        if resumed is not None and resumed["ts"] >= midnight:
            mark = self.store.equity_at_or_before(self.name, resumed["ts"])
            if mark is not None:
                return float(mark["equity"])
        restored = self.store.day_open_equity(self.name, midnight)
        return restored if restored is not None else equity

    # --- lifecycle ------------------------------------------------------------

    def on_start(self, taker_fee: float, now=None) -> None:
        self.taker_fee = taker_fee
        if now is not None:
            self.now = now
        if self.status in ("halted",):
            self.store.event(self.name, "warning", "restart", "restarted while halted; stays halted until resumed", ts=self.now())
        elif self.status == "paused" and self.paused_until is None:
            # The PM's pause, flatten or the book kill switch: no end time, so only a resume lifts it. Any
            # restart (a settings edit's reload, a stale heartbeat, a crash) keeps it (review round 10, B10-3).
            self.store.event(self.name, "info", "restart", "restarted while paused by the PM; stays paused until "
                             "resumed", ts=self.now())
        elif self.status == "paused" and self.paused_until > self.now():
            self.store.event(self.name, "info", "restart", "restarted while paused", ts=self.now())
        else:
            self._set("running", "")
        if not self.backtest:
            # Paper's venue is simulated in the process, so orders still working when it stopped went
            # with it; without this they would read "Working" for ever (review round 8, m8-9).
            for o in self.store.orders(self.name, statuses=OPEN_ORDER_STATUSES, limit=1000):
                self.store.update_order(o["order_id"], status="canceled",
                                        message="cancelled when the strategy restarted: paper's simulated venue "
                                                "went with the process that sent it")
        self._owed_flatten = self._interrupted_flatten()
        if self.book["fills"]:
            self.store.event(self.name, "info", "restore",
                             f"book restored from {self.book['fills']} journal fills: "
                             f"cash {self.book['cash']:,.2f}, position {self.book['qty']:g}", ts=self.now())
        self.store.event(self.name, "info", "start", f"strategy started ({self.profile.name} risk profile)", ts=self.now())

    def _interrupted_flatten(self) -> tuple[str, str] | None:
        """The flatten to send again after a restart, as (intent, reason), or None. A flatten is marked done
        when its sell is sent, not when it fills, so a process that stopped in between came back paused or
        halted and still holding, with nothing to sell it (sanity S-3). The latest PM flatten (the kill
        switch's too), risk halt or daily-loss pause that no resume has followed is owed again whenever a
        position is still held; a reconcile halt never flattens, and an expired daily-loss pause is over."""
        if self.backtest or self.status not in ("paused", "halted"):
            return None
        last = self.store.last_event(self.name, ("pm_flatten", "risk_halt", "risk_pause", "pm_resume"))
        if last is None or last["kind"] == "pm_resume":
            return None
        if last["kind"] == "risk_pause" and not (self.paused_until and self.paused_until > self.now()):
            return None
        label = {"pm_flatten": "Flattened by PM", "risk_halt": "Risk halt", "risk_pause": "Daily-loss pause"}
        fills = self.store.fills(self.name, limit=100_000)
        before = [f for f in fills if f["ts"] <= last["ts"]]
        if (last["kind"] != "pm_flatten" and len(before) < len(fills)
                and abs(replay_book(before[::-1], self.starting_balance)["qty"]) <= max(self.close_floor, 1e-12)):
            # Flat when it fired: what is held now filled after it (an entry that raced the cancel). The halt's or
            # the daily pause's own flatten covers it, through the exit path (Advisor 23:05, SG7).
            return last["kind"], f"Raced fill flattened by halt, after a restart ({label[last['kind']]}: {last['message']})"
        return last["kind"], f"{label[last['kind']]}, sent again after a restart: {last['message']}"

    def on_stop(self) -> None:
        if self.status in ("running", "starting"):
            self._set("stopped", "process stopped")

    # --- gates ----------------------------------------------------------------

    def entry_blocked(self) -> tuple[bool, str | None]:
        """CHOKE: whether nothing may open or add now, and why (blocked_state on this runtime's own state), keeping
        the journal's block episode in step. An expired daily-loss pause is rolled first; a backtest has no PM, so it
        is never stopped or retired."""
        self.can_open()
        sleeve = None if self.backtest else self.store.sleeve(self.name)
        state = SimpleNamespace(status=self.status, status_reason=sleeve.status_reason if sleeve else "",
                                paused_until=self.paused_until,
                                desired_state=sleeve.desired_state if sleeve else "running")
        blocked, why = blocked_state(state, self.now(),
                                     liquidated=(self.liquidated or WIPED_OUT) if self.wiped_out else None,
                                     archived=sleeve is not None and self.name in self.store.archived(),
                                     holds=self.holds)
        self._episode(why if blocked else None)
        return blocked, why

    def _episode(self, why: str | None) -> None:
        """Journal a block episode's start and end (Advisor 22:29): one alert when nothing may open any more, and one
        cleared event, with the orders it refused, when it ends or gives way to another reason."""
        if why == self._block:
            return
        if self._block is not None:
            self.store.event(self.name, "info", BLOCK_CLEARED,
                             f"Opening no longer held by this: {self._block}. Orders refused while it held: "
                             f"{self._refused}", ts=self.now())
        if why is not None:
            self.store.event(self.name, "warning", BLOCK_STARTED, BLOCK_PREFIX + why, ts=self.now())
        self._block, self._refused = why, 0

    def _liq_stuck(self, now: datetime, liquidating: bool) -> None:
        """The PM's commands wait while a liquidation order works (P1-U34). Should that order hang, they would wait
        unseen: past LIQ_STUCK, one incident names what waits (QA P1-D23). Stop is still taken by the dashboard."""
        if not liquidating:
            self.liq_working_since = None
            return
        self.liq_working_since = self.liq_working_since or now
        waiting = [c["command"] for c in self.store.pending_commands(self.name) if c["command"] != RELOAD]
        if waiting and now - self.liq_working_since >= LIQ_STUCK:
            mins = (now - self.liq_working_since).total_seconds() / 60
            self.incident_once(LIQ_STUCK_HEAD, f"{LIQ_STUCK_HEAD} after {mins:.0f} minutes, and the PM's "
                               f"{', '.join(waiting)} waits behind it: check the order at the venue. Stop is still "
                               "taken: it runs for its exits only")

    def incident_once(self, head: str, message: str) -> None:
        """An incident about the position held now, written once: a restart or a deploy doesn't repeat it while no
        fill has changed the position since (one incident per position, QA on P1-U35)."""
        if not said_since_last_fill(self.store, self.name, head):
            self.store.event(self.name, "error", "incident", message, ts=self.now())

    def last_watched_stop(self) -> float | None:
        """The level of the newest watched stop journaled for the position held now (since its last fill), open or
        cancelled by a restart, else None: a restart's safety stop never loosens it (QA SG4)."""
        last = self.store.fills(self.name, limit=1)
        since = last[0]["ts"] if last else None
        for o in self.store.orders(self.name, limit=200):  # newest first
            if since is not None and o["ts"] < since:
                return None
            if (o.get("signal") or {}).get("watched"):
                return float(o["signal"]["stop_px"])
        return None

    def position_incident(self) -> int | None:
        """The id of the latest incident about the position held now (written since its last fill), else None."""
        last = self.store.fills(self.name, limit=1)
        found = self.store.sleeve_events_since(self.name, ("incident",), since=last[0]["ts"] if last else None)
        return found[-1]["id"] if found else None

    def refused(self, why: str, what: str) -> None:
        """One decision row for an order the gate refused (Advisor 22:29); the episode's cleared event counts them."""
        self._refused += 1
        self.store.decide("system", REFUSED, f"Order not sent: {what}, and {why}", self.name)

    def can_open(self) -> bool:
        if self.wiped_out:
            return False  # liquidated: nothing opens until the PM resets it after the liquidation
        if self.status == "paused" and self.paused_until and self.paused_until <= self.now():
            self._set("running", "daily-loss pause expired")
            self.store.event(self.name, "info", "resume", "daily-loss pause expired; trading again", ts=self.now())
        return self.status == "running"

    def position_budget(self, equity: float) -> float:
        return equity * self.cap

    # --- periodic tick ----------------------------------------------------------

    def tick(self, *, equity: float, cash: float, qty: float, price: float,
             guard_equity: float | None = None, busy: bool = False, ruined: str | None = None,
             liquidating: bool = False) -> str | None:
        """Mark, guard, then apply PM commands. Returns "flatten" if the strategy must flatten now.
        guard_equity: the equity at the worst price since the last tick (a backtest's minute high or low on
        a perp), which the guard judges by when lower; the mark is still this tick's equity.
        busy: the strategy has an order working, so a flatten still owed waits for it rather than send another.
        ruined: why the strategy has nothing left (a gap past the bankruptcy price took its equity to zero): it
        halts, from running or paused, and flattens whatever is still open.
        liquidating: a liquidation order is working (the price went through the liquidation price): the PM's commands
        wait, so a resume queued while paused is never applied on the liquidating tick (QA P1-U34); the next tick has
        the liquidation's halt, which a resume doesn't clear."""
        now = self.now()
        self.store.heartbeat(self.name)
        if self.progress is not None:
            self.progress(now)
        if price <= 0:
            return None
        if self.bench_base_price is None:
            self.bench_base_price = price
        # Buy and hold at the exposure this sleeve may take (its position cap), the rest in cash, so the
        # comparison isn't flattered or punished by the cap itself; never above 1x, as a perp's cap can pass 1x
        # and a levered hold can't be liquidated. The backtest page's definition, so a saved
        # run's screen and its result agree (round 12, M12-U1).
        cap = min(self.cap, 1.0)
        benchmark = self.starting_balance * ((1 - cap) + cap * (1 - self.taker_fee) * price / self.bench_base_price)
        self.store.record_equity(self.name, equity=equity, cash=cash, qty=qty, price=price, benchmark=benchmark,
                                 ts=now)
        self.peak = max(self.peak, equity)
        if self._day != risk.trading_day(now):
            # The day opens at the equity last marked at or before midnight: the same thing in paper, which
            # marks every few seconds, and in a backtest, which marks once a bar. The first tick after a
            # (re)start reads it from the journal, so a restart mid-day, such as the reload a settings
            # edit makes, can't lift the daily-loss pause by resetting the baseline (review round 8, B8-2).
            if self._day is None and self._last_equity is None:
                self._day_open = self._restored_day_open(now, equity)
            else:
                self._day_open = self._last_equity if self._last_equity is not None else equity
            self._day = risk.trading_day(now)
        self._last_equity = equity
        if not self.backtest and self.status == "paused" and self.paused_until is not None and self.paused_until <= now:
            # Paper: the 00:00 roll lifts a daily pause on the tick, not at the next entry check, so the strategy page
            # reads it at once (QA SG14). A backtest has no page, and lifts it at its next entry check, where its
            # resting risk stops are placed again.
            self.can_open()

        flatten = False
        self.flatten_why = None
        if ruined and (self.status != "halted" or self.liquidated is None):  # a liquidation outranks a drawdown halt
            self.wiped_out, self.liquidated = True, ruined.split(";")[0]
            self._set("halted", ruined)
            held = abs(qty) >= max(self.close_floor, 1e-12)  # still open: past its liquidation price
            self.store.event(self.name, "error", "risk_halt",
                             ruined + ("; flattened" if held else "") + "; PM must reset it after the liquidation",
                             ts=self.now())
            if held:
                flatten, self.flatten_why = True, ("risk_halt", f"Risk halt: {ruined}")
        elif self.status == "running":
            judged = min(equity, guard_equity) if guard_equity is not None else equity
            breach = risk.check(self.profile, judged, self.peak, self._day_open)
            if breach and breach.action == "halt":
                self._set("halted", breach.reason)
                self.store.event(self.name, "error", "risk_halt", breach.reason + "; flattened, PM must resume", ts=self.now())
                flatten, self.flatten_why = True, ("risk_halt", f"Risk halt: {breach.reason}")
            elif breach and breach.action == "pause_day":
                # It clears at the next 00:00 UTC roll and nothing else (Independent Quant Advisor, 6 Oct 18:17, HC).
                until = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
                self._set("paused", breach.reason, until)
                self.store.event(self.name, "warning", "risk_pause", breach.reason + "; flattened until the next 00:00 UTC", ts=self.now())
                flatten, self.flatten_why = True, ("risk_pause", f"Daily-loss pause: {breach.reason}")
        elif (self.status == "paused" and abs(qty) >= max(self.close_floor, 1e-12) and self._owed_flatten is None
              and not busy):
            # Status governs entries only (Advisor 23:05, SG5): a holder that is paused (by the PM, for its exits
            # only, or after a flatten) keeps the drawdown halt and the daily-loss flatten. The halt outranks the
            # pause; a daily-loss breach flattens and is journaled but keeps the pause it was in, so a PM pause or an
            # exits-only start is never turned into one the 00:00 roll lifts. A daily pause already flattened.
            judged = min(equity, guard_equity) if guard_equity is not None else equity
            breach = risk.check(self.profile, judged, self.peak, self._day_open)
            if breach and breach.action == "halt":
                self._set("halted", breach.reason)
                self.store.event(self.name, "error", "risk_halt", breach.reason + "; flattened, PM must resume",
                                 ts=self.now())
                flatten, self.flatten_why = True, ("risk_halt", f"Risk halt: {breach.reason}")
            elif breach and breach.action == "pause_day" and self.paused_until is None:
                self.store.event(self.name, "warning", "risk_pause",
                                 f"{breach.reason}; flattened while paused, and it stays paused", ts=self.now())
                flatten, self.flatten_why = True, ("risk_pause", f"Daily-loss limit while paused: {breach.reason}")

        self._liq_stuck(now, liquidating)
        for cmd in [] if liquidating else self.store.pending_commands(self.name):
            if cmd["command"] == RELOAD:
                continue  # the supervisor's: it restarts this process under the new settings
            if cmd["command"] == "flatten":
                flatten = True
                self.flatten_why = self.flatten_why or ("pm_flatten", f"Flattened by PM: {cmd['reason']}")
                if self.status != "halted":  # never downgrade a halt
                    self._set("paused", f"flattened by PM: {cmd['reason']}")
            elif cmd["command"] == "pause":
                if self.status != "halted":
                    self._set("paused", f"paused by PM: {cmd['reason']}")
            elif cmd["command"] == "resume" and self.status == "running":
                # Already running: a resume would only reset the day's loss baseline (review round 10, m10-3).
                self.store.event(self.name, "info", "pm_resume_ignored",
                                 f"resume ignored, the strategy is already running: {cmd['reason']}", ts=self.now())
                self.store.mark_applied(cmd["id"])
                continue
            elif cmd["command"] == "resume" and self.status == "paused" and self.paused_until is not None:
                # A daily-loss pause clears only at the next 00:00 UTC roll, never by a resume (HC).
                self.store.event(self.name, "info", "pm_resume_ignored",
                                 f"resume ignored: the daily-loss pause ends at {self.paused_until:%d %b %H:%M} UTC, "
                                 f"the next 00:00 UTC roll, and nothing else clears it ({cmd['reason']})", ts=self.now())
                self.store.mark_applied(cmd["id"])
                continue
            elif cmd["command"] == "resume":
                # Both resets are journaled (this tick's mark, and the events below) so a restart keeps them.
                if self.wiped_out:
                    # A liquidation is not cleared by a resume (HoE and QA, 6 Oct; Advisor 18:17): it stays halted,
                    # not running for even a tick, drawdown still measured from the peak before the gap, and the
                    # halt is journaled again in its own words. Only a reset after liquidation starts it.
                    self.store.event(self.name, "info", "drawdown_kept",
                                     f"drawdown still measured from {self.peak:,.2f}: the halt was a liquidation, "
                                     "which a resume doesn't clear", ts=self.now())
                    why = liquidation_reason(self.liquidated or WIPED_OUT, self.store.insurance_total(self.name))
                    if self.status != "halted" or self.store.sleeve(self.name).status_reason != why:
                        self._set("halted", why)
                    self.store.event(self.name, "error", "risk_halt",
                                     f"{why}; a resume doesn't clear it, PM must reset it after the liquidation",
                                     ts=self.now())
                    self.store.event(self.name, "info", "pm_resume", cmd["reason"], ts=self.now())
                    self.store.mark_applied(cmd["id"])
                    continue
                elif self.status == "halted":
                    self.peak = equity  # a resume after a halt resets the drawdown reference
                    self.store.event(self.name, "info", "drawdown_reset",
                                     f"drawdown measured from {equity:,.2f}, the equity when the PM resumed "
                                     "after the halt", ts=self.now())
                self._day_open = equity
                self._set("running", "")
            self.store.event(self.name, "info", f"pm_{cmd['command']}", cmd["reason"], ts=self.now())
            self.store.mark_applied(cmd["id"])
        # A flatten is owed until the position is closed: one cut short by a restart (sanity S-3), or whose
        # order the venue rejected, is sent again, up to FLATTEN_RETRIES times, never while an order is
        # working, and never for dust or once the strategy is running again (a resume, an expired pause).
        if flatten:
            self._owed_flatten, self._flatten_retries = self.flatten_why, 0
        elif self._owed_flatten is not None:
            if self.status not in ("paused", "halted") or abs(qty) < max(self.close_floor, 1e-12):
                self._owed_flatten = None
            elif not busy and self._flatten_retries < FLATTEN_RETRIES:
                self._flatten_retries += 1
                flatten, self.flatten_why = True, self._owed_flatten
                self.store.event(self.name, "warning", "flatten_retry",
                                 f"still holding {qty:.12g} after a flatten that didn't complete; closing it again "
                                 f"(attempt {self._flatten_retries} of {FLATTEN_RETRIES}; {self._owed_flatten[1]})",
                                 ts=self.now())
            elif not busy and self._flatten_retries == FLATTEN_RETRIES:
                self._flatten_retries += 1
                self.store.event(self.name, "error", "flatten_failed",
                                 f"still holding {qty:.12g} after {FLATTEN_RETRIES} attempts to close it; the PM must "
                                 f"flatten it ({self._owed_flatten[1]})", ts=self.now())
        if not flatten and self.raced is not None:
            held = abs(qty) >= max(self.close_floor, 1e-12)
            flattening = self.status == "halted" or (self.status == "paused" and self.paused_until is not None)
            if held and flattening and not busy:
                kind = "risk_halt" if self.status == "halted" else "risk_pause"
                flatten, self.flatten_why = True, (kind, f"Raced fill flattened by halt: {self.raced}")
                self._owed_flatten, self._flatten_retries = self.flatten_why, 0
                self.store.event(self.name, "warning", "raced_fill_flattened",
                                 f"Raced fill flattened by halt: {qty:.12g} filled after it ({self.raced}); sold at "
                                 "market through the exit path", ts=self.now())
            if not (held and flattening and busy):
                self.raced = None
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
        d_cash = cash - book["cash"]
        # Compared in Decimal, each side as written, so float noise can't tip a one-lot gap into a halt.
        d_qty = float(Decimal(repr(float(qty))) - Decimal(repr(float(book["qty"]))))
        # Neither side holds more figures than a float does: at 1.6e8 units one float step is 3e-8, wider than two
        # lots of an 8-decimal instrument, so the tolerance is never finer than the float steps the journal's sum of
        # fills can drift by (review round 12, M12-E1: a 3e-8 gap on 159,660,965.631 halted a backtest).
        scale = max(abs(float(qty)), abs(float(book["qty"])))
        qty_tolerance = max(qty_tolerance, (2 + book["fills"]) * math.ulp(scale))
        detail = (f"engine cash {cash:,.2f} vs journal {book['cash']:,.2f}; "
                  f"engine position {qty:.12g} vs journal {book['qty']:.12g}"
                  # %g's 6 figures hid a 1e-7 gap on 9075.15 (review round 10, m1).
                  + (f", {d_qty:+.3g} apart" if abs(d_qty) > qty_tolerance else "") + f" ({book['fills']} fills)")
        if abs(d_cash) <= cash_tol and abs(d_qty) <= qty_tolerance:
            self.store.event(self.name, "info", "reconcile", "engine matches journal: " + detail, ts=self.now())
            return True
        self._set("halted", "reconciliation mismatch")
        self.store.event(self.name, "error", "reconcile_mismatch",
                         detail + ". Halted: no new trades, nothing flattened or corrected (a resting stop-loss stays). "
                         "Restart the strategy to rebuild "
                         "from the journal, or resume once you have checked.", ts=self.now())
        return False

    # --- quotes -------------------------------------------------------------------

    def market_seen(self) -> None:
        """A trade or quote arrived from the venue: kept for the price feed's age on the strategy page, at most
        every FEED_WRITE_EVERY so a busy feed doesn't write on every tick."""
        now = self.now()
        if self._feed_written is None or now - self._feed_written >= FEED_WRITE_EVERY:
            self.store.feed_seen(self.name, now)
            self._feed_written = now

    def on_quote(self, bid: float, ask: float, venue: str) -> None:
        """Sample the half spread; once an hour record its median, which backtests then charge."""
        mid = (bid + ask) / 2
        if mid <= 0 or ask < bid:
            return
        now = self.now()
        if self._spread_since is None:
            self._spread_since = now
        self._spreads.append((ask - bid) / 2 / mid)
        if now - self._spread_since >= SPREAD_EVERY:
            if len(self._spreads) >= SPREAD_MIN_SAMPLES:
                half = sorted(self._spreads)[len(self._spreads) // 2]
                if half < 0.05:  # a crossed or broken book is not a spread worth charging
                    instrument = self.store.sleeve(self.name).instrument
                    self.store.record_spread(venue, instrument, half, samples=len(self._spreads), ts=now)
            self._spreads, self._spread_since = [], now

    # --- orders -------------------------------------------------------------------

    def on_order(self, *, order_id: str, side: str, qty: float, intent: str, reason: str, signal: dict,
                 order_type: str = "MARKET") -> None:
        """Journal an order and why it was sent, before it goes to the venue."""
        self.store.record_order(self.name, order_id=order_id, side=side, qty=qty, intent=intent, reason=reason,
                                signal=signal, order_type=order_type, ts=self.now())

    def on_order_status(self, order_id: str, status: str, message: str = "") -> None:
        self.store.update_order(order_id, status=status, message=message)
        if status in ("rejected", "denied"):
            self.store.event(self.name, "warning", f"order_{status}", f"order {order_id} {status}: {message}", ts=self.now())

    def on_fill(self, *, side: str, qty: float, price: float, fee: float, order_id: str, trade_id: str,
                ts: datetime | None = None) -> None:
        """ts: when it filled, when that isn't now (a backtest's fill on a gap, at the bar's open)."""
        ts = ts or self.now()
        self.store.record_fill(self.name, side=side, qty=qty, price=price, fee=fee, order_id=order_id,
                               trade_id=trade_id, ts=ts)
        self.store.update_order(order_id, fill_qty=qty, fill_px=price, fee=fee)
        self.store.event(self.name, "info", "fill", f"{side} {qty:g} @ {price:,.2f}, fee {fee:,.2f}", ts=ts)

    # --- display -------------------------------------------------------------------

    def publish_signals(self, state: dict) -> None:
        """The model's conditions on the forming candle, for the strategy page's Signals tab (the strategy
        throttles these). Display only, and paper only: a backtest's runtime never writes them."""
        if not self.backtest:
            self.store.set_signal_state(self.name, state, ts=self.now())

    def _set(self, status: str, reason: str, paused_until: datetime | None = None) -> None:
        self.status, self.paused_until = status, paused_until
        self.store.set_status(self.name, status, reason, paused_until)
