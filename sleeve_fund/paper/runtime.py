"""What turns a strategy into a sleeve: journal, PM controls and the risk guard.

A strategy gets a SleeveRuntime in paper and live, and backtests run with the same
runtime on an in-memory journal (SleeveRuntime.for_backtest), so sizing, halts, pauses
and the journal behave identically in every mode. Research runs without one. The runtime
is called from the strategy's own thread (a timer and fill events), so there is no
concurrency inside it.
"""

from __future__ import annotations

import math
import re

from decimal import Decimal

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sleeve_fund import risk
from sleeve_fund.store import (BOOK_RESET, LIQUIDATION_RESET, OPEN_ORDER_STATUSES, RAL, RELOAD, WHY_STOP_FIELD, Store,
                               replay_book, utcnow)

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
EXITS_ONLY = "exits only"  # how the status of a strategy run for its exits only begins (P1-U35)


def liquidation_reason(head: str, covered: float) -> str:
    """A liquidation's halt once flat: its head, and what the venue's insurance fund covered, if anything."""
    return head + (f"; the venue's insurance fund covered the {covered:,.2f} shortfall" if covered > 0 else "")

def fold(text: str | None) -> str:
    """Text for a loose match: one plain space between words, in any case (a no-break space counts as one)."""
    return " ".join((text or "").split()).casefold()


LIQUIDATED_WORDS = (fold(WIPED_OUT), "wiped out", "position margin lost")  # a liquidation's halt, in any wording


# Reason codes (Advisor 7 Oct 00:20; QA pins them): stable, and listed in this order when several apply. Callers key
# off the codes, never the words. Each cause's text for the PM starts with its label, gives the figure where there is
# one, and says what alone clears it; no venue names.
CODES = ("liquidated", "drawdown_halt", "halted", "daily_pause", "retired", "winding_down", "stopped", "paused",
         "exits_only", "stale_data", "degraded_candle", "funding_missing")
LABELS = {"liquidated": "Liquidated", "drawdown_halt": "Drawdown halt", "halted": "Halted",
          "daily_pause": "Daily loss pause", "retired": "Retired", "winding_down": "Winding down", "stopped": "Stopped",
          "paused": "Paused", "exits_only": "Exits only", "stale_data": "Stale data",
          "degraded_candle": "Degraded candle", "funding_missing": "No funding rate"}
# The engine's holds (SleeveRuntime.holds) by key: a code, or an older key for one.
HOLD_CODES = {"data": "degraded_candle", "stale": "stale_data", "funding": "funding_missing"}
RESUMABLE = ("drawdown_halt", "halted")  # the halts the PM's Resume clears


class Why(str):
    """Why nothing opens, in words for the PM, with `code` (the first cause) and `codes` (every cause that applies, in
    CODES order). It compares and prints as its words."""
    code: str
    codes: tuple[str, ...]

    def __new__(cls, text: str, code: str, codes: tuple[str, ...] | None = None) -> Why:
        why = super().__new__(cls, text)
        why.code, why.codes = code, tuple(codes or (code,))
        return why

    @classmethod
    def of(cls, causes: list[Why]) -> Why | None:
        """Every cause in one, in CODES order, or None when there are none."""
        if not causes:
            return None
        causes = sorted(causes, key=lambda w: CODES.index(w.code))
        return cls(" ".join(causes), causes[0].code, tuple(w.code for w in causes))


def _cause(code: str, text: str) -> Why:
    return Why(f"{LABELS[code]}: {text}", code)


def liquidated_why(head: str | None = None) -> Why:
    """A liquidation, with its figures when the halt has them ("Position margin lost (liquidated): X, Y% ...")."""
    figures = (head or "").partition("): ")[2].strip()
    return _cause("liquidated", f"position margin lost{', ' + figures if figures else ''}. Only a reset after "
                  "liquidation clears it.")


LIQUIDATED = liquidated_why()


def clearing_action(sleeve, now: datetime | None = None, since: datetime | None = None) -> Why | None:
    """What alone clears the halt a strategy is in, in words, or None when it isn't halted or paused for the day
    (Independent Quant Advisor, 6 Oct 18:17, HC): a liquidation only a reset after liquidation; a drawdown halt only a
    resume; a daily-loss pause only the next 00:00 UTC roll. Stop, Start and restarts never do. Without `now` a daily
    pause counts until the runtime itself rolls it; with it, one whose roll has passed doesn't, so a strategy stopped
    across the roll can still be started (its first tick then lifts the pause). `since`: when the halt began."""
    reason = sleeve.status_reason or ""
    if sleeve.status == "halted":
        if fold(reason).startswith(LIQUIDATED_WORDS):
            return liquidated_why(reason.split(";")[0])
        at = f" since {since:%H:%M} UTC" if since else ""
        if (m := re.search(r"drawdown ([\d.]+%) hit the ([\d.]+%) limit", reason)):
            return _cause("drawdown_halt", f"down {m[1]} against a {m[2]} limit{at}. Only you can clear it, with Resume.")
        if "drawdown" in fold(reason):
            return _cause("drawdown_halt", f"{reason}{at}. Only you can clear it, with Resume.")
        return _cause("halted", f"{reason or 'by its risk limits'}{at}. Only you can clear it, with Resume.")
    if sleeve.status == "paused" and sleeve.paused_until is not None and (now is None or sleeve.paused_until > now):
        roll = f"It clears at the 00:00 UTC roll on {sleeve.paused_until:%d %b}."
        if (m := re.search(r"daily loss ([\d.]+%) hit the ([\d.]+%) limit", reason)):
            return _cause("daily_pause", f"down {m[1]} today against a {m[2]} limit. {roll}")
        return _cause("daily_pause", f"paused for the day's loss. {roll}")
    return None


# A block episode in the journal (Advisor 22:29): one alert when nothing may open any more, naming why, and one cleared
# event when it ends, with how many orders it refused. Each refused order is its own decision row.
BLOCK_STARTED, BLOCK_CLEARED, BLOCK_PREFIX = "entry_blocked", "entry_block_cleared", "Nothing opens. "
RACED_FILL = "raced_fill"  # an opening order that filled after the gate closed, with the ms after (P1-SG15)
ENTRY_CANCELLED = "resting_entry_cancelled"  # resting opening orders cancelled at a block, with the ms after it (P1-SG15)
# Inside an episode, which causes hold changed (one of several cleared, or another began): an info row, never an
# alert, so the journal's gate reads the engine's holds as they are now (Advisor 00:20 (b): events follow the
# transitions only).
BLOCK_CHANGED = "entry_block_changed"
REFUSED = "entry_blocked"  # the decision log's action for an order the gate refused (QA's exposure-gate master)
RETIRED = _cause("retired", "cannot be started. Only Restore brings it back.")
STOPPED = _cause("stopped", "only Start clears it.")


def _hold(key: str, text: str) -> Why:
    code = HOLD_CODES.get(key, key if key in LABELS else "stale_data")
    if key == "data" and "stale" in text:
        code = "stale_data"
    text = text.strip().rstrip(".")
    return Why(text + "." if text.startswith(LABELS[code] + ":") else f"{LABELS[code]}: {text}.", code)


def held_causes(text: str | None) -> dict[str, str]:
    """The engine's holds named in a journaled why (an open block episode), as holds: {code: its words}."""
    out = {}
    labels = "|".join(re.escape(LABELS[c]) for c in CODES)
    for part in re.split(rf"(?=\b(?:{labels}): )", text or ""):
        for code in ("stale_data", "degraded_candle", "funding_missing"):
            if part.startswith(LABELS[code] + ":"):
                out[code] = part.strip()
    return out


def block_codes(why: str | None) -> tuple[str, ...]:
    """The causes a why lists, in CODES order: its own codes, or read from its words (a why the journal holds)."""
    if why is None:
        return ()
    if isinstance(why, Why):
        return why.codes
    return tuple(c for c in CODES if re.search(rf"(?:^|\s){re.escape(LABELS[c])}: ", why))


def blocked_state(sleeve, now: datetime | None = None, *, liquidated: str | None = None, archived: bool = False,
                  holds: dict[str, str] | None = None, starting: bool = False,
                  since: datetime | None = None) -> tuple[bool, Why | None]:
    """CHOKE (HoE 6 Oct 20:52, Advisor): the one "nothing opens" gate, as (blocked, why). Blocked: liquidated until a
    reset after liquidation (`liquidated`, from the journal: liquidation_head), a halt, the daily-loss pause, retired,
    then (not when `starting`: the supervisor's Stop and the dashboard's Start and Resume, which act on these
    themselves) stopped, a pause with no end time, or a hold the engine has raised (`holds`: stale data, a degraded
    candle, no funding rate). The why lists every cause that applies (Advisor 00:20). The engine checks every order
    that would make the position bigger, at submit and at fill; the supervisor and the dashboard's Start and Resume
    read the same answer. Stops, reduce-only orders, the PM's close and a liquidation are never gated."""
    causes = []
    if liquidated is not None:
        causes.append(liquidated_why(liquidated))
    halt = clearing_action(sleeve, now, since)
    if halt is not None and not (liquidated is not None and halt.code == "liquidated"):
        causes.append(halt)
    if archived:
        causes.append(RETIRED)
    if not starting:  # Start and Resume act on a stopped or paused strategy: what they can't clear is above
        if "stopped" in (sleeve.status, getattr(sleeve, "desired_state", "running")):
            causes.append(STOPPED)
        if sleeve.status == "paused" and sleeve.paused_until is None:  # the PM's pause, a flatten, the kill switch,
            # or started for its exits only (a daily-loss pause past its roll is lifted by the runtime's next tick)
            reason = sleeve.status_reason or ""
            if reason.startswith(EXITS_ONLY):
                causes.append(_cause("exits_only", "it still holds a position it was stopped or refused with, so only "
                                     "its exits run. Flatten closes it; Start lets it trade again."))
            else:
                causes.append(_cause("paused", f"{reason or 'by you'}. Only Resume clears it."))
        causes += [_hold(k, v) for k, v in (holds or {}).items()]
    why = Why.of(causes)
    return why is not None, why


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


def liquidation_event(store, name: str) -> dict | None:
    """The liquidation a strategy is halted for while it is liquidated (liquidation_head), else None: its newest
    "liquidation" event since the last reset after liquidation, else its liquidation halt, else its liquidation order
    (as {"id": None, "ts": ...}). A reset after liquidation answers this one and no earlier one (P1-RAL)."""
    if liquidation_head(store, name) is None:
        return None
    reset = store.last_event(name, (RESET_AFTER_LIQUIDATION,))
    since = store.sleeve_events_since(name, ("liquidation", "risk_halt"), reset["id"] if reset else 0)
    for found in ([e for e in since if e["kind"] == "liquidation"],
                  [e for e in since if fold(e["message"]).startswith(fold(WIPED_OUT))
                   or fold(e["message"]).startswith(LIQUIDATED_WORDS)]):
        if found:
            return found[-1]
    order = store.last_order(name, ("liquidation",))
    return {"id": None, "ts": order["ts"]} if order is not None else None


def ral_refusal(store, name: str, incident: int | None, actor: str) -> str | None:
    """Why the PM's reset after liquidation (RAL) can't be taken, in words, else None (Advisor 17:57 and 18:17): only
    the PM sends it; only a liquidated strategy takes it; it names this liquidation's incident, which has its note on
    why the half-liquidation stop did not protect the position."""
    if actor != "PM":
        return "only the PM can reset a strategy after a liquidation"
    liq = liquidation_event(store, name)
    if liq is None:
        return ("it isn't halted after a liquidation, so there is nothing for a reset after liquidation to clear: "
                "its own action clears its halt or pause")
    if incident is None:
        return ("name the liquidation's incident: a reset after liquidation answers one incident, once its note says "
                f"{WHY_STOP_FIELD}")
    ev = store.event_by_id(incident)
    if ev is None or ev["kind"] != "incident" or ev["sleeve"] != name or ev["ts"] < liq["ts"]:
        return (f"#{incident} is not the incident of this strategy's liquidation of {liq['ts']:%d %b %H:%M} UTC: name "
                "that incident")
    if store.incident_note(incident) is None:
        return f"incident #{incident} has no note yet: write {WHY_STOP_FIELD}, and its author, first"
    return None


def ral_words(store, name: str, cmd: dict, liq: dict, old: float, equity: float) -> str:
    """What a reset after liquidation journals (Advisor 17:57, 18:17): the PM, the incident, the note's author, the old
    and new high-water marks and the equity before and after the liquidation. The engine's and the store's (a stopped
    strategy, which no process applies it for) say the same."""
    note = store.incident_note(cmd["incident"]) if cmd.get("incident") is not None else None
    before = store.equity_at_or_before(name, liq["ts"] - timedelta(seconds=1))
    return (f"Reset after liquidation by the PM, answering incident #{cmd['incident']} (note by "
            f"{note['author'] if note else 'nobody'}): high-water mark {old:,.2f} becomes {equity:,.2f}, "
            "the remaining equity, which is also the day's loss baseline. Equity "
            + (f"{before['equity']:,.2f}" if before else "unknown")
            + f" just before the liquidation, {equity:,.2f} after it. {cmd['reason']}")


def said_since_last_fill(store, name: str, head: str) -> bool:
    """Whether an incident starting with `head` was written since the position last changed (its last fill): one
    incident per position, read only back to that fill (CR 13)."""
    last = store.fills(name, limit=1)
    return any(e["message"].startswith(head) for e in
               store.sleeve_events_since(name, ("incident",), since=last[0]["ts"] if last else None))


def open_block(store, name: str, now: datetime | None = None) -> str | None:
    """Why nothing opens, as the engine last journaled it in a block episode it hadn't closed by `now`, else None."""
    last = store.last_event(name, (BLOCK_STARTED, BLOCK_CHANGED, BLOCK_CLEARED), before=now)
    return last["message"].removeprefix(BLOCK_PREFIX) if last and last["kind"] != BLOCK_CLEARED else None


def entry_blocked(store, name: str, now: datetime | None = None, *, starting: bool = False) -> tuple[bool, str | None]:
    """The gate for a strategy as its journal has it (blocked_state), for the supervisor, the dashboard and QA's
    exposure-gate set, with no process needed: the engine's own holds (stale data, missing funding) are read from the
    block episode it left open."""
    held = None if starting else held_causes(open_block(store, name, now))
    sleeve = store.sleeve(name)
    return blocked_state(sleeve, now, liquidated=liquidation_head(store, name), archived=name in store.archived(),
                         holds=held or None, starting=starting, since=_halted_since(store, name, sleeve))


def _halted_since(store, name: str, sleeve) -> datetime | None:
    """When the halt a strategy is in began (its risk_halt event), for the PM's words."""
    if sleeve is None or sleeve.status != "halted":
        return None
    last = store.last_event(name, ("risk_halt",))
    return last["ts"] if last else None


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
        # Paper: loads the instrument's spread measurements (sleeve_fund.spreads.series), reloaded each hour on_quote
        # closes a sampling window, so a new measurement (this strategy's or another's) is in force within the hour.
        self.spread_loader = None
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

    def last_fill_this_run(self) -> dict | None:
        """The newest fill a strategy may pick its cycle up from after a restart, else None: none once a reset after
        liquidation came after it, which starts a fresh run (Advisor 15:22 UK, RAL-ANCHOR), so the first entry after
        it follows the strategy's fresh-start rule rather than the liquidation's booked price. A reset after
        liquidation still waiting counts too: the process applies it on its first tick, after the strategy has
        started, and nothing trades before it does (the liquidation halt)."""
        fills = self.store.fills(self.name, limit=1)
        if not fills or any(c["command"] == RAL for c in self.store.pending_commands(self.name)):
            return None
        reset = self.store.last_event(self.name, (RESET_AFTER_LIQUIDATION,))
        return None if reset is not None and reset["ts"] >= fills[0]["ts"] else fills[0]

    def _last_liquidation(self) -> str | None:
        """The head of its liquidation halt while it is liquidated, else None (liquidation_head)."""
        return liquidation_head(self.store, self.name)

    def _restored_peak(self, starting_balance: float) -> float:
        """The drawdown reference after a (re)start: the highest mark since the PM last resumed from a
        halt, which reset it, else the highest mark ever. Without the reset a settings edit after such a
        resume re-halted and flattened at once (review round 9, M9-2)."""
        reset = self.store.last_event(self.name, ("drawdown_reset", RESET_AFTER_LIQUIDATION, BOOK_RESET))
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
        resumed = self.store.last_event(self.name, ("pm_resume", RESET_AFTER_LIQUIDATION, BOOK_RESET))
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
                                     holds=self.holds, since=_halted_since(self.store, self.name, state))
        self._episode(why if blocked else None)
        return blocked, why

    def block_began(self, why, now: datetime | None = None) -> datetime:
        """When the block a raced fill or a resting entry's cancel met began, for the ms they are journaled with (P1-SG15,
        Advisor 7 Oct 05:01, 05:47): a PM Stop's own acceptance (its decision), else the start of the block episode in
        the journal, else now. `now` to the microsecond, from the engine's clock (self.now() is to the second)."""
        now = now or self.now()
        if "stopped" in block_codes(why):
            stop = self.store.decisions(self.name, limit=1, action="stop")
            if stop and stop[0]["ts"] <= now:
                return stop[0]["ts"]
        began = self.store.last_event(self.name, (BLOCK_STARTED,), before=now)
        return began["ts"] if began else now

    def _episode(self, why: str | None) -> None:
        """Journal a block episode's start and end (Advisor 22:29): one alert when nothing may open any more, and one
        cleared event, with the orders it refused, when the last cause clears. Not when one of several clears or
        another joins, nor when a figure moves (stale data's age): those follow the transitions only (Advisor 00:20
        (b)); a change in which causes hold is an info row (BLOCK_CHANGED) the journal's gate reads."""
        if (why is None) == (self._block is None):
            if why is not None and block_codes(why) != block_codes(self._block):
                self.store.event(self.name, "info", BLOCK_CHANGED, BLOCK_PREFIX + why, ts=self.now())
            self._block = why
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

    def _reset_after_liquidation(self, cmd: dict, equity: float) -> None:
        """The PM's reset after liquidation (P1-RAL; Advisor 17:57, 18:17): a new high-water mark and day baseline at
        the remaining equity, the halt lifted, one liquidation_reset event naming the PM, the incident, the note's
        author, the old and new marks and the equity before and after the liquidation. The old mark stays in the
        journal; the book's own limits and figures are untouched. A reset asked for before the liquidation lapses,
        so it can't run without a fresh confirmation (#167 round). One already answered (a retry, a restart) is a no-op."""
        self.store.mark_applied(cmd["id"])
        liq = liquidation_event(self.store, self.name)
        if liq is None:
            self.store.event(self.name, "info", "ral_ignored",  # events.kind is String(32) (QA RAL-F1)
                             f"reset after liquidation ignored, nothing to reset: {cmd['reason']}", ts=self.now())
            return
        old = self.peak
        self.peak, self._day_open = equity, equity
        self.wiped_out, self.liquidated = False, None
        self._set("running", "")
        self.store.event(self.name, "info", RESET_AFTER_LIQUIDATION,
                         ral_words(self.store, self.name, cmd, liq, old, equity), ts=self.now())
        for request in self.store.pending_resets():
            if request["sleeve"] == self.name and request["created_at"] <= cmd["created_at"]:
                self.store.refuse_reset(request, "lapsed: asked before the liquidation, which the reset after "
                                        "liquidation answered")

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
        self.store.decide("system", REFUSED, f"Order not sent: {what}. {why}", self.name)

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
        self.store.heartbeat(self.name, now)
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
            elif cmd["command"] == RAL:
                self._reset_after_liquidation(cmd, equity)
                continue
            elif cmd["command"] == BOOK_RESET:
                # Store.book_reset journaled the clear and set the status; this process re-bases its own references
                # at its equity now. Never a liquidation, which only a reset after liquidation clears.
                if not self.wiped_out and (self.status == "halted" or (self.status == "paused" and self.paused_until)):
                    self.peak = self._day_open = equity
                    self._set("running", "")
                self.store.mark_applied(cmd["id"])
                continue
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

    def on_quote(self, bid: float, ask: float, venue: str):
        """Sample the half spread; once an hour record its median, which backtests then charge, and return the
        measurements reloaded (spread_loader) for the strategy to read from then on (SPREAD-PIT); else None."""
        mid = (bid + ask) / 2
        if mid <= 0 or ask < bid:
            return None
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
            if self.spread_loader is not None:
                try:
                    return self.spread_loader()
                except Exception as e:  # noqa: BLE001 - keep the series it has; the next hour tries again
                    self.store.event(self.name, "warning", "spreads_not_reloaded",
                                     f"the spread measurements could not be reloaded ({type(e).__name__}); "
                                     "the last ones loaded stay in force", ts=now)
        return None

    # --- orders -------------------------------------------------------------------

    def on_order(self, *, order_id: str, side: str, qty: float, intent: str, reason: str, signal: dict,
                 order_type: str = "MARKET", timing: dict | None = None) -> None:
        """Journal an order and why it was sent. An opening order's row is written before it goes to the venue;
        with paper's queued journal an exit's is queued and may land after it (paper.queued). timing: the
        decision's stamps (on_timing), journaled with the order; a backtest keeps none."""
        extra = {"timing": timing} if timing and not self.backtest else {}
        self.store.record_order(self.name, order_id=order_id, side=side, qty=qty, intent=intent, reason=reason,
                                signal=signal, order_type=order_type, ts=self.now(), **extra)

    def on_order_status(self, order_id: str, status: str, message: str = "") -> None:
        self.store.update_order(order_id, status=status, message=message)
        if status in ("rejected", "denied"):
            self.store.event(self.name, "warning", f"order_{status}", f"order {order_id} {status}: {message}", ts=self.now())

    def on_fill(self, *, side: str, qty: float, price: float, fee: float, order_id: str, trade_id: str,
                ts: datetime | None = None) -> None:
        """ts: when it filled, when that isn't now (a backtest's fill on a gap, at the bar's open)."""
        # One call, so a fill replayed after a reconnect or restart is booked once and moves its order once (DA-2).
        self.store.book_fill(self.name, side=side, qty=qty, price=price, fee=fee, order_id=order_id,
                             trade_id=trade_id, ts=ts or self.now())

    def on_timing(self, order_id: str, **stamps: int | None) -> None:
        """Paper and live (v2 P1-2): when the order's bar closed and arrived, the decision, the send, the venue's
        acceptance and each fill (UNIX ns), for close-to-fill times per strategy. A backtest's are its own
        replay clock, so it keeps none."""
        if not self.backtest:
            self.store.record_timing(self.name, order_id, **stamps)

    # --- display -------------------------------------------------------------------

    def publish_signals(self, state: dict) -> None:
        """The model's conditions on the forming candle, for the strategy page's Signals tab (the strategy
        throttles these). Display only, and paper only: a backtest's runtime never writes them."""
        if not self.backtest:
            self.store.set_signal_state(self.name, state, ts=self.now())

    def _set(self, status: str, reason: str, paused_until: datetime | None = None) -> None:
        self.status, self.paused_until = status, paused_until
        self.store.set_status(self.name, status, reason, paused_until)
