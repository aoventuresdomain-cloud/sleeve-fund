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
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sleeve_fund import accounts, lifecycle, liquidation, router
from sleeve_fund.alerts import Forwarder
from sleeve_fund.exact import float_view
from sleeve_fund.money import to_decimal
from sleeve_fund.gate_ledger import IDENTITY_HOLD_ACTION, IDENTITY_RESUME_ACTION, DbLedger
from sleeve_fund.portfolio.gate import EMPTY, PortfolioState, apply_flow, mark_book, sweep, unrestorable
from sleeve_fund.portfolio.limits import Book, Holding
from sleeve_fund.risk import PORTFOLIO, trading_day
from sleeve_fund.paper.safety import assert_portfolio_gate, credential_var
from sleeve_fund.paper.single_lock import EXIT_DUPLICATE
from sleeve_fund.paper.config import check_hub_bar_spec, load_sleeve, to_store_kwargs
from sleeve_fund.store import (DUST_NOTIONAL, OPEN_ORDER_STATUSES, TWO_LEG_FENCE, Sleeve, Store, check_own_capital,
                               is_backtest, is_dust, utcnow)
from sleeve_fund.strategies import check_perp_sizing, check_perp_stop
from sleeve_fund.paper.runtime import entry_blocked, liquidation_head, said_since_last_fill
from sleeve_fund.strategies.base import EXITS_ONLY

VENUE_EXITS = f"{EXITS_ONLY}: {router.VENUE_MOVED[router.EXITS_ONLY]}"  # F-WED1-3: the status reason of that start

POLL_SECONDS = 5
KEY_CHECK_EVERY = 12  # polls between key-presence checks: about a minute
FEE_CHECK_EVERY = 720  # polls between fee-schedule reads from connected accounts: about an hour
ALERT_EVERY = 12  # polls between alert sends and uptime pings: about a minute
CLEAR_EVERY = 12  # polls between retries of a clean slate waiting on a flatten: about a minute
TRUE_UP_EVERY = 12  # polls between true-ups for strategies with no process: about a minute
HEARTBEAT_STALE = timedelta(minutes=3)
STARTUP_GRACE = timedelta(minutes=3)
MOVE_ALERT_POLLS = 3  # a create or move still waiting after this many polls is alerted (Advisor ADV-ASSIGN)
HALT_RETELL = timedelta(minutes=1)  # a holder still holding under a portfolio halt is told to flatten again after this
MAX_BACKOFF = 300
# Flattens a reset or a clean slate queues for one strategy before it stops asking and says the PM must close
# it: a position below the venue's smallest order can't be closed by an order, and asking again every step
# only piled up commands (review round 13, m13-E1).
SYSTEM_FLATTENS = 3
# The exits-only reason of a stopped strategy that still holds a position (P1-U35): its process runs for its exits.
STOPPED_HOLDING = "it was stopped while it still holds a position"


CENT = Decimal("0.01")  # the identity's tolerance: rounding, never a real gap (RECON-B)


@dataclass(frozen=True)
class FundBook:
    """The fund-held book (Bridge Drop A): the fund's own cash put in (fund_flows since the re-base), what every
    Strategy's journal moved its cash by since then (from 0: no Strategy holds capital) and its positions at their last
    marks. total is the fund's equity."""
    since: datetime | None
    capital: Decimal
    pnl: Decimal
    positions: Decimal

    @property
    def total(self) -> Decimal:
        return self.capital + self.pnl + self.positions


def bridge_start(store: Store) -> datetime | None:
    """When the fund-held book began: the latest re-base deposit applied (CAP5-RESET), or None while every Strategy
    still carries its own capital (a journal from before the bridge, or before the PM's reset)."""
    reader = getattr(store, "fund_flows", None)
    if reader is None:
        return None
    at = [f["applied_at"] for f in reader() if f["rebase"] and f["applied_at"] is not None]
    return max(at) if at else None


def _in_book(store: Store, s) -> bool:
    from sleeve_fund.store import is_backtest

    return not is_backtest(s.name) and not outside_fund_book(store, s)


def _position_value(store: Store, s) -> Decimal:
    """A Strategy's journal position at its last mark; dust never marked at its entry price (F229-7). Raises when a
    position has no mark, so the fund isn't marked."""
    book = store.journal_book(s.name, s.starting_balance)
    qty = to_decimal(book["qty"], "qty")
    if qty == 0:
        return Decimal(0)
    last = store.last_equity(s.name)
    if last is not None and last["price"]:
        return qty * to_decimal(last["price"], "mark")
    if is_dust(book) and book["entry_px"]:
        return qty * to_decimal(book["entry_px"], "entry price")
    raise RuntimeError(f"{s.name} holds {qty} with no mark, so the fund can't be marked")


def fund_book(store: Store, since: datetime | None = None) -> FundBook:
    """The fund-held book (Bridge Drop A; DA spec 7): cash put in (applied flows from `since`, the re-base by default),
    plus every Strategy's cash P&L since then, put away or not (the cash is the fund's), plus its positions at the mark.
    The one book reader for the mark, sizing and the per-poll identity check."""
    since = since if since is not None else bridge_start(store)
    flows = [f for f in store.fund_flows() if f["applied_at"] is not None]
    # The capital starts at the re-base flow itself, by journal order, not its time: a flow applied at the same poll
    # but ahead of it in id order went into the book the re-base replaced (PE1's reset cell a3)
    base = max((f["id"] for f in flows if f["rebase"] and since is not None and f["applied_at"] <= since), default=None)
    capital = Decimal(0)
    for f in flows:
        if since is None or (f["id"] >= base if base is not None else f["applied_at"] >= since):
            amount = to_decimal(f["amount"], "amount")
            capital += -amount if f["kind"] == "withdraw" else amount
    pnl = positions = Decimal(0)
    for s in store.sleeves():
        if _in_book(store, s):
            pnl += to_decimal(store.strategy_cash_pnl(s.name, since=since), "cash")
            positions += _position_value(store, s)
    return FundBook(since, capital, pnl, positions)


def sizing_equity(store: Store, now: datetime | None = None) -> Decimal | None:
    """The fund equity a Strategy sizes off (Advisor Q5, CUT-A; QD's strategy.fund_equity): the book at its last
    mark less every open position's unrealised gain, so a gain not yet taken never sizes anything, while a loss
    counts in full. None when the book's mark is older than the gate's MARK_STALE (or never made): sizing then
    refuses the entry as "fund equity stale", as the gate would."""
    from sleeve_fund.portfolio.gate import MARK_STALE

    now = now or utcnow()
    st = DbLedger(store, lambda: ()).state()
    if st.equity is None or st.mark_ts is None or now - st.mark_ts > MARK_STALE:
        return None
    gains = Decimal(0)
    for s in store.sleeves():
        if not _in_book(store, s):
            continue
        book = store.journal_book(s.name, s.starting_balance)
        qty = to_decimal(book["qty"], "qty")
        last = store.last_equity(s.name) if qty != 0 else None
        if qty != 0 and last is not None and last["price"] and book["entry_px"]:
            gains += max(qty * (to_decimal(last["price"], "mark") - to_decimal(book["entry_px"], "entry")), Decimal(0))
    return st.equity - gains


def book_part(store: Store, s) -> Decimal:
    """What moves into or out of the book when Strategy `s` joins or leaves it: its equity while it carries its own
    capital; nothing once the fund holds the cash (Bridge Drop A), when a create, a move or a put-away only changes
    where it books."""
    return Decimal(0) if bridge_start(store) is not None else strategy_equity(store, s)


def identity_gap(store: Store) -> Decimal:
    """RECON-B (Advisor MUST): the fund's cash P&L read two ways, the journal's replay of every fill, funding and
    insurance row (journal_book) against the exact cash reader (strategy_cash_pnl), over every Strategy in the book.
    They are the same rows, so anything past a cent is a broken book: the poll then marks nothing and alerts, and
    entries stop once the mark is 60 s old."""
    gap = Decimal(0)
    for s in store.sleeves():
        if _in_book(store, s):
            replayed = to_decimal(store.journal_book(s.name, 0)["cash"], "cash")
            gap += replayed - to_decimal(store.strategy_cash_pnl(s.name), "cash")
    return gap


def identity_tolerance(store: Store) -> Decimal:
    """A gap within a cent is rounding, not a break. RECON-SCOPE allows max(0.01, 1e-6 x the book), but the journal is
    exact (DA-9) and HoQA's cells hold a 0.02 gap on a 100,000 book to be a break, so the cent stands (the stricter)."""
    return CENT


def positions_breaks(store: Store) -> list[str]:
    """RECON-SCOPE (Advisor, the positions leg): per Strategy in the book, the position its journal's fills come to up to
    its last mark against the live position that mark carries (the runtime's equity row). Apart by more than dust at
    the mark (DUST_NOTIONAL: below any venue's smallest order, so within a lot) is a break even when the cash agrees: a
    fill journaled twice, or one the journal lost. Returns each break in words, [] when every position ties."""
    from sqlalchemy import case, func, select

    from sleeve_fund.store import fills_t

    out = []
    for s in store.sleeves():
        if not _in_book(store, s):
            continue
        last = store.last_equity(s.name)
        if last is None:
            continue
        signed = case((fills_t.c.side == "BUY", fills_t.c.qty), else_=-fills_t.c.qty)
        q = select(func.coalesce(func.sum(signed), 0)).where(fills_t.c.sleeve == s.name, fills_t.c.ts <= last["ts"])
        with store.engine.connect() as c:
            replayed = to_decimal(c.execute(q).scalar(), "qty")
        live = to_decimal(last["qty"] or 0, "qty")
        price = to_decimal(last["price"] or 0, "mark")
        apart = abs(replayed - live)
        if apart > Decimal("1e-12") and (price <= 0 or apart * price >= Decimal(str(DUST_NOTIONAL))):
            out.append(f"{s.name}: the journal's fills come to {replayed.normalize():f} but the live position is "
                       f"{live.normalize():f}")
    return out


def insurance_breaks(store: Store, since: datetime | None = None) -> list[str]:
    """MUST-FUNDCASH-LOSS, TEC-INSURE-ZERO (HoE): once the fund holds the cash, a liquidation is booked at the bankruptcy
    price (GAP-LIQ-CAP), so no fund-held Strategy is ever credited by the insurance fund; a credit is a loss booked back
    as income, which the cash leg can't see (both its readers count insurance). Any credit from `since` (the bridge,
    or the PM's last Resume) on, per Strategy in the book, whatever fill it names. Returns each in words, [] for none."""
    from sqlalchemy import select

    from sleeve_fund.store import CLOSE_OUT, insurance_t

    names = [s.name for s in store.sleeves() if _in_book(store, s)]
    if not names or since is None:
        return []
    i = insurance_t.c
    # An old book close-out (RESET-NEG) is a transfer inside the book, not a credit from outside it (DA, cell 10); a
    # credit naming no fill (before 0013) still counts
    q = (select(i.sleeve, i.amount).where(i.sleeve.in_(names), i.ts >= since, i.order_id.is_distinct_from(CLOSE_OUT))
         .order_by(i.id))
    with store.engine.connect() as c:
        rows = list(c.execute(q))
    return [f"{r.sleeve}: an insurance credit of {to_decimal(r.amount, 'amount'):,.2f} booked inside the fund, which "
            "books every liquidation at the bankruptcy price" for r in rows]


def _last_resume(store: Store) -> datetime | None:
    rows = store.decisions(action=IDENTITY_RESUME_ACTION)
    return max((r["ts"] for r in rows), default=None) if rows else None


def empty_book_pockets(store: Store) -> list[str]:
    """What keeps the book from being empty, in the PM's words; empty is nothing left (CAP5-RESET: a re-base deposit
    lands only onto an empty book). Every Strategy whose equity still counts, one waiting to leave, anything that
    could trade or holds, any room held, and in a fund-held book its own cash and P&L."""
    led = DbLedger(store, lambda: ())
    out = []
    if (why := nothing_open(store, led)) is not None:
        out.append(why)
    out += [f"{n} waits to leave the book" for n in store.withdrawal_pending()]
    if bridge_start(store) is not None:
        book = fund_book(store)
        if abs(book.total) >= CENT:
            out.append(f"the fund's cash and P&L, {book.total:,.2f}")
        return out
    runs, gone = store.reset_runs(), store.withdrawn()
    for s in store.sleeves():
        if _in_book(store, s) and s.name not in gone:
            eq = strategy_equity(store, s, runs)
            if abs(eq) >= CENT:
                out.append(f"{s.name} {eq:,.2f}")
    return out


def resume_identity_hold(store: Store, actor: str, reason: str, now: datetime | None = None) -> str:
    """The PM's Resume of a book-check hold (RECON-B): refused while the book still doesn't add up, else journaled, and
    new entries may go out again. Returns the words journaled."""
    led = DbLedger(store, lambda: ())
    if not led.identity_hold():
        raise ValueError("the fund's book check isn't holding entries: nothing to resume")
    if not actor.strip() or not reason.strip():
        raise ValueError("a Resume names who and why")
    gap = identity_gap(store)
    if abs(gap) > identity_tolerance(store):
        raise ValueError(f"the book still doesn't add up by {gap:,.2f}: fix it before resuming")
    if apart := positions_breaks(store):
        raise ValueError(f"the positions still don't tie ({'; '.join(apart)}): fix them before resuming")
    words = f"book check resumed: the book adds up again (gap {gap:,.2f}); {reason.strip()}"
    store.decide(actor.strip(), IDENTITY_RESUME_ACTION, words, ts=now)
    led.alert("book_identity_resumed", f"The PM resumed new entries after the book check: {words}", now or utcnow())
    return words


def free_cash(store: Store, led) -> Decimal:
    """What the fund can pay out now (DA 7): the book less posted margin, less the loss at the stops, less every room
    an order holds."""
    book = fund_book(store).total
    held = fund_holdings(store)
    used = sum((h.margin + h.risk for h in held), Decimal(0))
    used += sum((r.holding.margin + r.holding.risk for r in led.reservations()), Decimal(0))
    return book - used


def fund_equity(store: Store) -> Decimal:
    """The whole fund's marked equity (Advisor 3): every strategy's, running or not, plus the cash no strategy holds.
    A strategy's is its starting capital, what the journal moved its cash by (CASH-2's strategy_cash_pnl, the one cash
    reader) and its journal position at its last mark. One the PM put away (archived, a clean slate) still counts in
    full: its cash is the fund's, unallocated, so archiving never moves the book. A run a reset put away counts only
    its cash movement: the strategy it came from starts again at the same capital, which it still counts (HoE 13 Oct).
    Raises when a position has no mark, so the portfolio isn't marked and entries stop once its mark is stale; dust
    with no mark is held at its entry price (F229-7)."""
    from sleeve_fund.store import is_backtest

    if bridge_start(store) is not None:  # Bridge Drop A: the fund holds the cash
        return fund_book(store).total
    runs, out = store.reset_runs(), store.withdrawn()
    total = Decimal(0)
    for s in store.sleeves():
        if is_backtest(s.name) or outside_fund_book(store, s):  # F229-6: a sub-8dp test Strategy isn't the fund's
            continue
        if s.name in out:  # F-C3B-A: put away, its equity taken out of the book with its flow
            continue
        total += strategy_equity(store, s, runs)
    return total


def nothing_open(store: Store, ledger, leaving: str | None = None) -> str | None:
    """Why the book isn't idle, in the PM's words, or None when nothing is open: no reservation held, no Strategy that
    could trade (could_trade), no position but dust. The one guard for an empty book (F282-3's no-op mark, F-C3B-A's
    emptying flow). leaving: the Strategy whose withdrawal empties it, skipped (F286-1): it is stopped in the same
    transaction, which this reads from outside, and found settled flat with no order open or room held before its
    equity was taken out (Store.withdrawal_blocker)."""
    if ledger.reservations():
        return "an order still holds room in the book"
    for s in store.sleeves():
        if s.name == leaving:
            continue
        book = store.journal_book(s.name, s.starting_balance)
        if could_trade(s):
            return f"Strategy {s.name} can still trade"
        if abs(book["qty"]) > 1e-12 and not is_dust(book):
            return f"Strategy {s.name} still holds {book['qty']}"
    return None


def _flowed(st: PortfolioState) -> tuple:
    """What a flow (portfolio.gate.apply_flow) moves: a change here between reading the book and marking it is one."""
    return st.equity, st.hwm, st.halt_reference, st.day_start_equity


def strategy_equity(store: Store, s, runs: dict | None = None) -> Decimal:
    """One Strategy's part of fund_equity, wherever it sits: what moves into or out of the book when it does
    (HALT-REF-1, Store._book_move)."""
    runs = store.reset_runs() if runs is None else runs
    cash = to_decimal(store.strategy_cash_pnl(s.name), "cash")
    if s.name in runs:
        return cash  # flat when put away (split_run): its capital is its successor's
    total = to_decimal(s.starting_balance, "starting balance") + cash
    book = store.journal_book(s.name, s.starting_balance)
    qty = to_decimal(book["qty"], "qty")
    if qty != 0:
        last = store.last_equity(s.name)
        if last is not None and last["price"]:
            total += qty * to_decimal(last["price"], "mark")
        elif is_dust(book) and book["entry_px"]:
            # F229-7: dust no venue takes an order for, never marked, is held at its entry price (under
            # DUST_NOTIONAL either way) rather than failing the whole fund closed over it
            total += qty * to_decimal(book["entry_px"], "entry price")
        else:
            raise RuntimeError(f"{s.name} holds {qty} with no mark, so the fund can't be marked")
    return total


def fund_holdings(store: Store, now: datetime | None = None) -> tuple[Holding, ...]:
    """Every strategy's positions as the book's Holdings, for book_marks' figures: the paper gate's own
    (paper.portfolio_gate.fund_holdings, P2-1b W2), so a book mark records the book the gate checks entries against
    (F-BM-1). Raises ValueError, as the gate's do, when a stopless position's daily ATR isn't known."""
    from sleeve_fund.paper import portfolio_gate

    return portfolio_gate.fund_holdings(store, now)


def _holding(store: Store, s: Sleeve) -> bool:
    """Whether a strategy's journal holds a position an order can close (not dust)."""
    book = store.journal_book(s.name, s.starting_balance)
    return abs(book["qty"]) > 1e-12 and not is_dust(book)


def _held_words(store: Store, name: str) -> str:
    """What the Strategy `name` belongs to holds, leg by leg when it has two (CR F283-2): "0.1", or "carry 0.1,
    carry~2 -0.1"."""
    legs = store.legs(name)
    qty = {leg: store.journal_book(leg, store.sleeve(leg).starting_balance)["qty"] for leg in legs}
    if len(legs) == 1:
        return f"{qty[legs[0]]:.12g}"
    return ", ".join(f"{leg} {q:.12g}" for leg, q in qty.items())


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
    want = (sleeve.desired_state == "running" or holds) and sleeve.lifecycle != lifecycle.RETIRED  # P2-6: never again
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


def check_venue_pin(store: Store, s: Sleeve) -> None:
    """Raises ValueError when the strategy's journal last traded on another venue than its pin (Store.venue_start):
    the supervisor's refusal then starts a holder for its exits only, on the venue its position is on, and refuses
    a flat one."""
    mode, traded = store.venue_start(s)
    if mode != router.NORMAL:
        print(f"{s.name}: venue {mode}: journal {traded}, pin {router.venue_for(s.venue)}", flush=True)
        raise ValueError(router.VENUE_MOVED[mode])


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


def true_up_stopped(store: Store, s: Sleeve, now: datetime | None = None) -> int:
    """CASH-4: a strategy with no process (stopped, archived, retired, or a reset's earlier run) can't notice the
    venue's rate arriving for a settlement it was charged the baseline for, so this trues it up from the stored rates:
    settled - baseline, on the baseline's position and price, booked now (Store.record_true_up). A running strategy
    does it itself; were both to, the journal books it once. Returns how many it booked."""
    import pandas as pd

    from sleeve_fund import funding, markets
    from sleeve_fund.venues import DEFAULT_VENUE

    owed = store.open_baselines(s.name)
    if not owed:
        return 0
    try:
        terms = markets.terms(s.params, getattr(s, "venue", None) or DEFAULT_VENUE)
    except ValueError:
        return 0
    if terms is None or terms.funding_venue is None:
        return 0
    series, cap = funding.rates(terms.funding_venue, s.instrument), funding.cap_of(terms.funding_venue, s.instrument)
    now, booked = now or utcnow(), 0
    for row in owed:
        ts = row["ts"] if row["ts"].tzinfo else row["ts"].replace(tzinfo=timezone.utc)
        rate = funding.rate_at(series, pd.Timestamp(ts), cap)
        if rate is None:
            continue
        settled = -float(row["qty"]) * float(row["price"]) * rate
        done, up = store.record_true_up(s.name, ts, rate=rate, amount=round(settled, 8), arrived_at=now)
        if done == "new":
            booked += 1
            store.event(s.name, "info", "funding_true_up",
                        f"Funding trued up: the venue's rate for the {ts:%d %b %Y %H:%M} UTC settlement arrived "
                        f"({rate:.4%}) while the strategy had no process, settling to {settled:+,.2f} against the "
                        f"baseline charged; {up:+,.2f} booked now", ts=now)
    return booked


RESTING_OFF = ("its entries would rest at the venue as post-only limits (maker-first execution), and paper sends no "
               "resting entry until the orphan guard is in, which cancels one its process can no longer watch: set "
               "its execution to market to run it")


# CCY-8DP (Advisor 13 Oct 16:15 and 16:25 UK): the journal books each fill's fee and cash leg at 8 dp, so a quote
# currency the engine holds to fewer places (USD: 2) would book what the venue never charged. Only the test accounts
# a venue profile names (VenueProfile.sub_8dp_test_accounts) may run on one, with a warning; every other account, a
# future venue included, is refused. Their results are never Strategy evidence.
FILL_PLACES = 8


def _test_account_reason(account: str, venue: str | None) -> str | None:
    """Why (account, venue) may run under 8 places, from the venue's profile; None when it may not (an unknown venue
    included)."""
    from sleeve_fund.venues import DEFAULT_VENUE, VENUES

    profile = VENUES.get((venue or DEFAULT_VENUE).upper())
    return None if profile is None else profile.sub_8dp_test_accounts.get(account)


def is_test_currency_account(account: str, venue: str | None) -> bool:
    """Whether (account, venue) is a named test account allowed under 8 places: its results are never Strategy
    evidence (CCY-8DP-UI keys the PM's views on this, not on a copy of the list)."""
    return _test_account_reason(account, venue) is not None


def _quote_places(s) -> tuple[str, int]:
    """The Strategy's quote currency and the places the engine holds it to (one it doesn't list it holds to 8)."""
    from nautilus_trader.model import Currency

    quote = s.instrument.split("/")[-1].split("-")[0].upper()
    return quote, Currency.from_str(quote).precision


def outside_fund_book(store: Store, s, account: str | None = None) -> bool:
    """F229-6 (HoE 13 Oct 18:12 UK): a Strategy whose quote currency is held under 8 places, on the test account its
    venue names (CCY-8DP), is outside the fund's book. Neither its equity (fund_equity) nor its position (both
    fund_holdings) counts, and its entries don't reserve against the book gate (LongFlatStrategy._paper_gated); its own
    per-Strategy checks still apply. The one predicate all of those read. account: as if it were on this account (a
    move about to be made), else the one it is on. A Strategy created on a marked book is outside it until the
    supervisor's poll lands its joining deposit (F-ASSIGN-STALE, Advisor ADV-ASSIGN); a waiting move leaves it on its
    old account until then, so on its old side."""
    if account is None and store.joining(s.name):
        return True
    return (_quote_places(s)[1] < FILL_PLACES
            and is_test_currency_account(account or store.account_of(s.name), getattr(s, "venue", None)))


def check_quote_places(s: Sleeve, account: str) -> str | None:
    """Raises ValueError when the Strategy's quote currency is held to fewer than 8 places on an account its venue
    doesn't name as a test account; returns the warning to log when it does; None when the currency holds 8. The
    words name no venue (they reach the PM)."""
    quote, places = _quote_places(s)
    if places >= FILL_PLACES:
        return None
    why = _test_account_reason(account, getattr(s, "venue", None))
    held = f"{places} places"
    if why is None:
        raise ValueError(f"its quote currency {quote} is held to {held}, and fills are booked to {FILL_PLACES}; only "
                         f"named test accounts may run on it, and account {account} is not one on this venue")
    return (f"Test account: its quote currency {quote} is held to {held}, under the {FILL_PLACES} fills are booked "
            f"to, which only {why} may do. Its results are not Strategy evidence")


def check_resting_entries(s: Sleeve) -> None:
    """RESTING-OFF (HoE, Advisor 13 Oct): raises ValueError for a Strategy whose entries would rest at the venue
    (maker_wait_minutes set), until ORPHAN-GUARD lands. Market entries are unaffected; a backtest never comes here."""
    if (s.params or {}).get("maker_wait_minutes"):
        raise ValueError(RESTING_OFF)


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
        self._holdings_failing = False  # so is a failed read of the book's holdings (F-BM-1)
        self._carried_told = False  # an emptied book's carried ratios that can't be restored, alerted once
        self._withdraw_told: dict[str, str] = {}  # a deferred withdrawal's refusal, said once while it lasts
        self._move_polls: dict[int, int] = {}  # a waiting move's id: the polls it has waited through (F-ASSIGN-STALE)
        self._move_told: dict[str, str] = {}  # a waiting move's refusal, said once while it lasts
        self._breach_told: set[str] = set()  # a withdrawn Strategy holding, alerted once per spell (F-C3B-A viii)
        self._flatten_told: dict[str, datetime] = {}  # when each holder was last told to flatten for a portfolio halt
        self._halt_tells: dict[str, int] = {}  # how often, this halt (F229-5: at most SYSTEM_FLATTENS)
        self._flow_told: dict[int, str] = {}  # a fund flow's wait, said once while it lasts
        self._dead_told: dict[str, str] = {}  # closing a stopped Strategy's orders failed: said once per failure

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
            held = (f". It still holds a position ({_held_words(self.store, name)}), which a flatten can't close while "
                    "it can't start: an engineer needs to close it" if self.store.holds_any(name) else "")
            self.store.set_desired_state(name, "stopped")
            self.store.set_status(name, "stopped", f"not started: {exc}{held}")
            self.store.event(name, "error", "start_refused", f"Not started: {exc}{held}")
            return True
        try:
            check_venue_pin(self.store, s)  # first: an off-pin holder's exits-only run reads this reason (CR F277-2)
            check_own_capital(self.store, s.name, s.starting_balance)  # F-ZS-1
            check_perp_sizing(s.strategy, s.params)
            check_perp_stop(s.strategy, s.params, s.risk_profile)
            check_funding_schedule(s)
            check_resting_entries(s)
            warn = check_quote_places(s, self.store.account_of(name))
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
        if warn:
            self.store.event(name, "warning", "test_currency", warn)
        return False

    def _exits_only(self, s: Sleeve, why: str) -> bool:
        """A refused start still holding a position (not dust) is started anyway, paused with the EXITS_ONLY
        reason: its exits run, a safety stop is set from the mark, no entry or add opens, and an incident is raised
        (LongFlatStrategy._safety_stop_on_restore). Any refusal can use it, so a position is never left without a
        process watching it. True when it applies."""
        if not self.store.holds_any(s.name):  # either leg of a two-leg Strategy (CR F283-2)
            return False
        held = _held_words(self.store, s.name)
        reason = f"{EXITS_ONLY}: {why}"
        if s.status != "paused" or s.status_reason != reason:
            self.store.set_status(s.name, "paused", reason)
            self.store.event(s.name, "error", "start_refused", f"Not started to trade: {why}. It still holds a "
                             "position, so it is started for its exits only, with a safety stop")
            self.store.event(s.name, "error", "incident",
                             f"Incident, {s.name}: not started to trade ({why}), but it holds {held}, so it "
                             "runs for its exits only, with a safety stop from the current price and no new entries")
        return True

    def _venue_exits_done(self, s: Sleeve, proc: Proc) -> None:
        """F-WED1-3 (HoE 14 Oct): a strategy running for its exits only on the venue its position is on stops once that
        position is closed, so exits-only never outlives it. Still on another venue than its pin, it is then refused
        as any flat one is (check_venue_pin) until a reset; its pin's venue again, it starts to trade."""
        mode, _ = self.store.venue_start(s)
        if mode == router.EXITS_ONLY:
            return
        self._stop(s.name, proc, "flat: its exits-only run on another venue is over")
        if mode == router.REFUSE:
            self.store.set_desired_state(s.name, "stopped")
            self.store.set_status(s.name, "stopped", f"not started: {router.VENUE_MOVED[mode]}")
            self.store.event(s.name, "info", "venue_exits_done", "Its position on the other venue is closed, so its "
                             f"exits-only run has stopped. Not started to trade: {router.VENUE_MOVED[mode]}")
        else:
            self.store.set_status(s.name, "stopped", "flat: its exits-only run is over")

    def _holds(self, s: Sleeve, proc: Proc) -> bool:
        """Whether a stopped strategy still holds a position (not dust). Without a process its journal can't change,
        so that is read once per stop; while its exits-only process runs, every step, so it stops once flat."""
        if s.desired_state == "running":
            return False
        if proc.holds is None or proc.alive:
            proc.holds = self.store.holds_any(s.name)  # either leg of a two-leg Strategy (CR F283-2)
        return proc.holds

    def _watch_stopped_holder(self, s: Sleeve, proc: Proc, starting: bool = False) -> None:
        """P1-U35 (Advisor 20:56, HoE): a stopped strategy still holding a position (the PM's Stop on a holder, or a
        fill that raced the stop) is never left unwatched. It runs for its exits only: its own stop, or a safety stop
        from the mark where it has none (_safety_stop_on_restore), and nothing opens, with an incident. A halt or a
        pause keeps its own status, which already holds every entry and is cleared only by its own action (HC)."""
        proc.watched = True
        if s.status not in ("halted", "paused"):
            self.store.set_status(s.name, "paused", f"{EXITS_ONLY}: {STOPPED_HOLDING}")
        if starting:
            return  # the process it starts sets the safety stop and writes the incident, once per position
        head = f"Incident, {s.name}: stopped, but it still holds "
        if said_since_last_fill(self.store, s.name, head):
            return  # already said for this position: a deploy or a crash restarts it without a second incident
        self.store.event(s.name, "error", "incident",
                         f"{head}{_held_words(self.store, s.name)}, so it runs for its exits only: its stop (or a safety stop from the "
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
        # A two-leg Strategy's second leg row (P2-8) is traded by its first leg's process, never by one of its own.
        for sleeve in self.store.sleeves(include_legs=False):
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
                if code == EXIT_DUPLICATE:
                    # SINGLE-LOCK: another live process holds the strategy's lock, so the strategy is still running there;
                    # not a crash of it. Its refusal is journalled (duplicate_refused); tried again after the backoff,
                    # and it takes over once that process is gone.
                    continue
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
            elif action == "none" and proc.alive and sleeve.status == "paused" and sleeve.status_reason == VENUE_EXITS:
                self._venue_exits_done(sleeve, proc)
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
        self.close_dead_orders()

    def close_dead_orders(self) -> None:
        """F-REFRESH-RES (HoE, DA): a Strategy put away whose process is gone can't fill an order its journal still
        shows open (the simulated venue went with the process), and that order's room would keep it, and the PM's
        refresh, waiting forever. Each one still in the book (not yet withdrawn) has those orders closed and the room
        released (Store.close_dead_orders), but only once its process is confirmed gone: none of ours alive, and its
        single-instance lock free, so no process started elsewhere holds it either. The lock is held while it closes,
        so none can start meanwhile. Flat ones only, paper only; a failure is said once, and never holds up the rest."""
        from sleeve_fund.paper.single_lock import instance_lock

        withdrawn = self.store.withdrawn()
        for name in self.store.archived():
            if name in withdrawn or is_backtest(name):
                continue
            runner = name.split("~", 1)[0]  # a leg is traded by its Strategy's own process
            if self.procs.get(runner, Proc()).alive or self.procs.get(name, Proc()).alive:
                continue
            try:
                # Only a flat one (DA): an archived holder runs for its exits, and its resting stop is an open order
                # in the journal; closed while its process is down between a crash and its restart, the restart
                # might not place it again. The refresh only ever reaches a put-away that is flat.
                if self.store.account_of(name) != "paper" or not self.store.lifecycle_flat(name)[1]:
                    continue
                lock = instance_lock(runner)
                if lock is None:  # a live process holds it: not confirmed gone
                    continue
                try:
                    self.store.close_dead_orders(name, "put away, and its process has stopped")
                finally:
                    lock.close()
            except Exception as exc:  # noqa: BLE001 - the next poll tries again; the dashboard shows why, once
                why = f"closing a stopped Strategy's orders: {exc!r}"
                if self._dead_told.get(name) != why:
                    self._dead_told[name] = why
                    self.store.event(name, "error", "supervisor_error", why)
            else:
                self._dead_told.pop(name, None)

    def true_up_stopped(self) -> None:
        """True up every strategy without a running process (true_up_stopped); one failing never holds up the rest."""
        for s in self.store.sleeves():
            if self.procs.get(s.name, Proc()).alive:
                continue
            try:
                true_up_stopped(self.store, s)
            except Exception as exc:  # noqa: BLE001 - the next minute tries again; the dashboard shows why
                self.store.event(s.name, "error", "supervisor_error", f"funding true-up: {exc!r}")

    def mark_portfolio(self, now: datetime | None = None) -> None:
        """The portfolio's poll (v2 P2-2): mark the whole fund (portfolio.gate.mark_book), a book_marks row each minute
        and at every status change, act on a halt, and sweep orphaned reservations. The state row is written before
        the equity is read, so a read that fails leaves the gate in force and entries blocked once the mark is 60 s
        old. A pause needs nothing here: CHOKE reads it from the state (runtime.portfolio_block) until 00:00 UTC."""
        now = now or utcnow()
        led = self.ledger
        led.ensure_state()
        before, behind = led.state(), False
        try:
            self._check_identity(now)
            equity = fund_equity(self.ledger.store)
        except Exception as exc:  # noqa: BLE001 - no mark: the book goes stale and entries stop (fail closed)
            if not self._equity_failing:
                self.store.event(None, "error", "supervisor_error", f"the fund's equity couldn't be read, so the "
                                 f"portfolio isn't marked and entries stop once its mark is 60 s old: {exc!r}")
            self._equity_failing = True
        else:
            self._equity_failing = False
            with led.lock():
                # HALT-REF-1: a capital move whose flow (Store._book_move) landed while the book was read left that read
                # behind it, a phantom gain or loss; this mark is skipped and the next one, seconds on, reads it all.
                st = led.state()
                behind = _flowed(st) != _flowed(before)
                # F282-3 (Advisor, 14 Oct 10:20 UK): an empty book never marked, with nothing open, has nothing to
                # mark. It writes, raises and clears nothing (a halt or pause in force stays for the PM), and entries
                # stay blocked as stale until the first capital's mark starts the book.
                behind = behind or (st.equity is None and st.hwm is None and abs(equity) < EMPTY
                                    and nothing_open(self.store, led) is None)
                # MUST-F286-H4: an emptied book's carried ratios wait for capital to restore them against, never ~0
                behind = behind or (st.equity is None and st.carried_hwm_drawdown is not None and equity < EMPTY)
                # Advisor MUST (14 Oct 14:45 UK): carried ratios that can't be restored fail closed, said once
                if not behind:
                    bad = unrestorable(st)
                    if bad is not None and not self._carried_told:
                        led.alert("portfolio_carried_invalid", f"Portfolio not marked: {bad}", now)
                    self._carried_told = behind = bad is not None
                acted = None if behind else mark_book(led, equity, now, PORTFOLIO)
        if not self._equity_failing and not behind:
            st = led.state()
            last = led.last_book_mark()
            minute = now.replace(second=0, microsecond=0)
            if last is None or last.replace(second=0, microsecond=0) != minute or _status(before, now) != _status(st, now):
                try:
                    held = fund_holdings(self.ledger.store, now)
                except Exception as exc:  # noqa: BLE001 - the cache row waits; the state row above is the gate's
                    # A warning, not an error: the row is a cache nothing trades on, and the gate's own check fails
                    # closed on the same read (a stopless position whose daily ATR isn't known yet, as on a fresh stack).
                    if not self._holdings_failing:
                        self.store.event(None, "warning", "book_mark_skipped", f"the book's holdings couldn't be read, "
                                         f"so the book's history (book_marks) isn't written until they can: {exc!r}")
                    self._holdings_failing = True
                else:
                    self._holdings_failing = False
                    led.write_book_mark(now, Book(equity, held), st, PORTFOLIO.version)
            if acted == "halt":
                self._flatten_told.clear()  # a new halt tells every holder at once
                self._halt_tells.clear()
            self._withdraw_flat(now)  # after the mark: the flatten's P&L is judged at the book's scale before it
            self._land_moves(now)  # after the mark, whatever it decided: a create or move judged at the fresh scale
        if not self._equity_failing:  # after what was put away left the book; an empty book's first deposit included
            self._apply_flows(now)
        self._alert_withdrawn_holders(now)
        self._count_waiting_moves(now)
        st = led.state()
        if st.halted:
            self._halt_every_strategy(st.halted, now)
        else:
            self._flatten_told.clear()
            self._halt_tells.clear()
        # No cancel yet (None): the owner cancels its own entries at its next bar close, and the supervisor's cancel of
        # a dead owner's orders is ORPHAN-GUARD og2 on the DA's 0015. Past the TTL, an unacked order's reservation is
        # kept and alerted on as not cancelled, never released blind.
        sweep(led, now, self._order_live, None, is_acked=self._order_acked)

    def _check_identity(self, now: datetime) -> None:
        """RECON-B (Advisor MUST, IDENT-STICKY): once the fund holds the cash, every poll checks cash + P&L + positions
        against the book read the other way (identity_gap). Past a cent, read twice so a fill landing between the reads
        isn't a break, it holds every new entry at once (identity_hold, journaled; the gate reads it), with one alert.
        The hold is latched: a clean poll doesn't lift it, only the PM's Resume (resume_identity_hold). The mark, exits,
        stops and close-now carry on."""
        store = self.ledger.store
        if bridge_start(store) is None or self.ledger.identity_hold():
            return
        tol = identity_tolerance(store)
        since = max((t for t in (bridge_start(store), _last_resume(store)) if t is not None), default=None)
        gap, apart = identity_gap(store), positions_breaks(store) + insurance_breaks(store, since)
        if abs(gap) <= tol and not apart:
            return
        # once more: a fill between the first reads moves both sides by the time of this one
        gap, apart = identity_gap(store), positions_breaks(store) + insurance_breaks(store, since)
        if abs(gap) <= tol and not apart:
            return
        why = "; ".join(([f"cash + Strategy P&L + positions differs from the journal by {gap:,.2f}"]
                         if abs(gap) > tol else []) + apart)
        store.decide("supervisor", IDENTITY_HOLD_ACTION, why, ts=now)
        self.ledger.alert("book_identity_broken", f"The fund's book doesn't add up: {why}. New entries are held until "
                          "the PM resumes; exits and stops still go out", now)

    def _apply_flows(self, now: datetime) -> None:
        """The fund's deposits and withdrawals (DA fund_flows), in journal order, after this poll's mark: an inflow
        never overtakes a pending outflow, so the first that must wait stops the rest (CAP5-RESET). A re-base deposit
        lands only onto an empty book (empty_book_pockets), starting it afresh at exactly the deposit with nothing
        carried; onto a book that isn't empty it is refused, naming what is left. A plain flow needs a fund-held book
        (the re-base first). A withdrawal waits while it is more than the free cash (free_cash)."""
        store, led = self.ledger.store, self.ledger
        if getattr(store, "pending_flows", None) is None:
            return
        for f in store.pending_flows():
            fid, amount = f["id"], to_decimal(f["amount"], "amount")
            try:
                if f["rebase"]:
                    left = empty_book_pockets(store)
                    if left:
                        store.refuse_flow(fid, now, "the book isn't empty, so it can't start afresh: " + "; ".join(left))
                        continue
                    with led.lock():
                        st = led.state()
                        led.set_state(replace(st, equity=amount, hwm=amount, halt_reference=None, mark_ts=now,
                                              day=trading_day(now), day_start_equity=amount, halted=None,
                                              paused_until=None, paused=None, carried_hwm_drawdown=None,
                                              carried_reference_drawdown=None, carried_day_return=None,
                                              carried_day=None))
                        store.settle_flow(fid, now, conn=led._local.conn)
                    store.event(None, "warning", "portfolio_rebased", f"The fund's book starts afresh at "
                                f"{amount:,.2f} (fund flow {fid}): nothing carried from before")
                elif bridge_start(store) is None:
                    store.refuse_flow(fid, now, "the fund holds no cash of its own until the PM's reset deposits it")
                    continue
                elif f["kind"] == "withdraw":
                    with led.lock():
                        free = free_cash(store, led)
                        if amount > free:
                            raise RuntimeError(f"it is more than the {free:,.2f} free now (the book less margin, "
                                               "the loss at the stops and the room orders hold)")
                        apply_flow(led, -amount, lambda: nothing_open(store, led), now)
                        store.settle_flow(fid, now, conn=led._local.conn)
                else:
                    with led.lock():
                        apply_flow(led, amount, None, now)
                        store.settle_flow(fid, now, conn=led._local.conn)
                self._flow_told.pop(fid, None)
            except Exception as exc:  # noqa: BLE001 - it waits, and every flow behind it, until the next poll
                if self._flow_told.get(fid) != str(exc):
                    store.event(None, "warning", "fund_flow_waits", f"The fund's {f['kind']} of {amount:,.2f} (flow "
                                f"{fid}) waits, and every flow after it: {exc}")
                    self._flow_told[fid] = str(exc)
                return

    def _withdraw_flat(self, now: datetime) -> None:
        """F-C3B-A (Advisor 14 Oct 10:45 UK): each Strategy put away and waiting (Store.withdrawal_pending: not flat
        then, or put away outside this poll, F286-CR1) whose withdrawal nothing blocks any more has its settled equity
        taken out of the book, with one journaled flow under the gate's lock (Store._put_away), the one place it leaves.
        Called after this poll's mark. A refusal (a book a day's pause keeps marked) is said once and tried again
        on the next poll; one Strategy's failure never stops the rest."""
        for name in self.store.withdrawal_pending():
            if self.store.withdrawal_blocker(name) is not None:
                continue
            try:
                with self.store._put_away(name, now, "flat after it was put away", polled=True):
                    pass
                self._withdraw_told.pop(name, None)
            except Exception as exc:  # noqa: BLE001 - kept in the book, tried again next poll
                if self._withdraw_told.get(name) != str(exc):
                    self.store.event(name, "warning", "withdrawal_waits", f"{name} is flat, and its equity "
                                     f"couldn't leave the fund's book yet: {exc}")
                    self._withdraw_told[name] = str(exc)

    def _land_moves(self, now: datetime) -> None:
        """F-ASSIGN-STALE (Advisor ADV-ASSIGN, 14 Oct 17:41 UK): every create and account move asked for since the last
        mark lands now, oldest first, after this poll's mark, whatever that mark decided (a Strategy joining a halted or
        paused book is halted or paused with it), with its flow at the fresh scale (Store.land_move). A refusal is
        said once and tried again next poll; one Strategy's failure never stops the rest."""
        for name, move in self.store.book_moves_pending().items():
            try:
                self.store.land_move(name, now)
                self._move_told.pop(name, None)
            except Exception as exc:  # noqa: BLE001 - it waits, entries refused, and is tried again next poll
                if self._move_told.get(name) != str(exc):
                    self.store.event(name, "warning", "book_move_waits", f"{name}'s move to account "
                                     f"{move['account']} couldn't land yet: {exc}")
                    self._move_told[name] = str(exc)

    def _count_waiting_moves(self, now: datetime) -> None:
        """Every poll, whether it marked or not: a create or move still waiting after MOVE_ALERT_POLLS polls raises an
        alert, once (Advisor ADV-ASSIGN), with the refusal it last met, if any."""
        waiting = self.store.book_moves_pending()
        for name, move in waiting.items():
            polls = self._move_polls[move["id"]] = self._move_polls.get(move["id"], 0) + 1
            if polls == MOVE_ALERT_POLLS:
                why = self._move_told.get(name) or "the book hasn't been marked since it was asked for"
                self.ledger.alert("book_move_stuck", f"{name}'s move to account {move['account']} has waited "
                                  f"{polls} polls without landing, and its entries stay refused: {why}", now)
        for gone in set(self._move_polls) - {m["id"] for m in waiting.values()}:
            self._move_polls.pop(gone, None)

    def _alert_withdrawn_holders(self, now: datetime) -> None:
        """A withdrawn Strategy holding a position is a broken invariant (Advisor 14 Oct 10:45 UK): its exposure
        would count against a book its equity left; dust no order can close is flat, as for the withdrawal. The gate's
        holdings refuse it (fund_holdings raises, so every
        entry fails closed); this raises the alert, once per spell."""
        out = self.store.withdrawn()
        # flat as a withdrawal judges it (Store.lifecycle_flat): dust left after one is no breach (Advisor 11:10 UK)
        breach = {s.name for s in self.store.sleeves() if s.name in out and not self.store.lifecycle_flat(s.name)[1]}
        for name in sorted(breach - self._breach_told):
            self.store.event(name, "error", "withdrawn_holds", f"{name}'s equity left the fund's book, yet it holds "
                             "a position: no Strategy may enter until it is resolved")
        self._breach_told = breach

    def _halt_every_strategy(self, why: str, now: datetime) -> None:
        """A portfolio halt (15% under the reference): every strategy holding anything is flattened through its exit
        path; CHOKE blocks every entry from the state row until the PM resumes the portfolio. Asked on every poll while
        the portfolio is halted, not only the one that halted it, so a holder a failed command skipped, or one still
        holding after its flatten, is told again: never while a flatten waits for it, and at most once a minute (CR
        F229-3). One strategy's failure never stops the rest."""
        reason = f"Portfolio halted: {why}"
        # A two-leg Strategy is told through its first leg's row, whose flatten reaches both legs, while either holds:
        # a flatten sent to the second leg's row alone has no process to act on it (CR F283-2).
        for s in self.store.sleeves(include_legs=False):
            told = self._flatten_told.get(s.name)
            if not self.store.holds_any(s.name):
                self._flatten_told.pop(s.name, None)
                self._halt_tells.pop(s.name, None)
                continue
            if told is not None and now - told < HALT_RETELL:
                continue
            if self._halt_tells.get(s.name, 0) >= SYSTEM_FLATTENS:
                # F229-5: still holding after that many flattens, perhaps less than the venue's smallest order. Said
                # once; the PM closes it. Its entries stay blocked by the halt meanwhile.
                if self._halt_tells[s.name] == SYSTEM_FLATTENS:
                    self.store.event(s.name, "error", "flatten_gave_up",
                                     f"still holding {_held_words(self.store, s.name)} after {SYSTEM_FLATTENS} flattens, perhaps less than the "
                                     f"venue's smallest order; the PM must close it: {reason}")
                    self._halt_tells[s.name] += 1
                continue
            try:
                if any(c["command"] == "flatten" for c in self.store.pending_commands(s.name)):
                    continue
                self.store.command(s.name, "flatten", reason, actor="supervisor")
                self._flatten_told[s.name] = now
                self._halt_tells[s.name] = self._halt_tells.get(s.name, 0) + 1
            except Exception as exc:  # noqa: BLE001 - asked again in a minute
                self._flatten_told[s.name] = now
                self.store.event(s.name, "error", "supervisor_error", "couldn't tell it to flatten for the portfolio "
                                 f"halt, asked again in a minute: {exc!r}")

    def _order_live(self, order_id: str) -> bool:
        """Whether an order may still fill, read through the ledger's own transaction: inside the sweep's lock a second
        connection would, on SQLite's one shared connection, roll the sweep's own writes back (HoQA, P2-1b cells)."""
        return self.ledger.order_status(order_id) in OPEN_ORDER_STATUSES

    def _order_acked(self, order_id: str) -> bool:
        """Whether the venue has acked the order (resting or part filled): its reservation then lives with it."""
        return self.ledger.order_status(order_id) in ("accepted", "partially_filled")

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
                if loops % TRUE_UP_EVERY == 0:
                    self.true_up_stopped()
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


def _flatten_again(store: Store, name: str, why: str, since, qty: float | str) -> bool:
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
                    f"still holding {qty if isinstance(qty, str) else f'{qty:.12g}'} after {SYSTEM_FLATTENS} flattens, "
                    "perhaps less than the venue's smallest "
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
        return clear_entries(store, tomllib.load(fh).get("clear", []))


def clear_entries(store: Store, entries: list[dict]) -> list[str]:
    """clear's work on its [[clear]] entries, read from a file or given by the PM's reset workflow (CAP5-RESET)."""
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
        two = [s.name for s in store.sleeves(include_legs=False)
               if s.name not in keep and s.name not in put_away and store.two_leg(s.name)]
        if two:  # refused whole, before any write; never raised, so the supervisor's poll goes on (TWO_LEG_FENCE)
            store.event(None, "error", "clear_refused", f"{reason}: a clean slate {TWO_LEG_FENCE} "
                        f"({', '.join(two)})")
            continue
        holding = []
        # A two-leg Strategy is cleared whole, through its first leg's row: its flatten and start reach both legs,
        # it holds while either leg does, and archive puts both away (CR F283-2).
        for s in store.sleeves(include_legs=False):
            if s.name in keep or (waiting and s.name not in waiting):
                continue
            legs = store.legs(s.name)
            held = any(abs(store.journal_book(n, store.sleeve(n).starting_balance)["qty"]) > 1e-12 for n in legs)
            qty = _held_words(store, s.name)
            if s.name in put_away and not held and s.name not in waiting:
                continue  # put away flat by an earlier slate
            # One archived while still holding (the 4 Oct slate, before holders were flattened first) is
            # flattened like any other holder: until it is flat it stays in the book's figures.
            if held:
                holding.append(s.name)
                why = f"{reason}: flattened so it can be archived"
                if (not any(c["command"] == "flatten" for c in store.pending_commands(s.name))
                        and _flatten_again(store, s.name, why, store.book_start(), qty)):
                    store.command(s.name, "flatten", why, actor="system", holds_through_reset=False)
                if s.desired_state != "running":
                    store.set_desired_state(s.name, "running")
                    store.decide("system", "start", f"{reason}: started only to flatten its position", s.name)
                store.event(s.name, "warning", "clear_held",
                            f"{reason}: not archived yet, it still holds {qty}; it is being flattened and "
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
        # F286-4 (HoQA): a slate on an empty book drops the drawdown its emptying carried, as a book reset does
        fresh = "; the drawdown the emptied book kept is dropped, so the next capital starts afresh" \
            if store.drop_carried() else ""
        store.decide("system", "clear", f"{key}: {reason} ({len(cleared)} put away{fresh})")
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
    starting capital, equity, fees, open positions and the first mark, for the deploy log. A Strategy outside the
    fund's book (F229-6, outside_fund_book) is left out, as fund_equity leaves it out, and named after the figures
    (F-HALTREF-4); so is one whose create waits for the supervisor's next mark (F-ASSIGN-STALE)."""
    earlier = store.previous_book()
    every = [s for s in store.sleeves() if s.name not in earlier]
    outside = [s.name for s in every if outside_fund_book(store, s)]
    current = [s for s in every if s.name not in outside]
    joining = [n for n in outside if store.joining(n)]  # created on a marked book, its deposit waits (F-ASSIGN-STALE)
    tests = [n for n in outside if n not in joining]
    start = sum(s.starting_balance for s in current)
    equity = sum(float((store.last_equity(s.name) or {"equity": s.starting_balance})["equity"]) for s in current)
    fees = sum(float(f["fee"]) for s in current for f in store.fills(s.name, limit=1_000_000))
    held = [s.name for s in current if abs(store.journal_book(s.name, s.starting_balance)["qty"]) > 1e-12]
    firsts = [m["ts"] for s in current if (m := store.first_equity(s.name))]
    return (f"from {start:,.2f}, equity {equity:,.2f}, fees {fees:,.2f}, "
            f"positions {', '.join(held) or 'none'}, first mark {min(firsts).isoformat() if firsts else 'none'}"
            + (f"; outside the book (a test account): {', '.join(tests)}" if tests else "")
            + (f"; joining the book at the supervisor's next mark: {', '.join(joining)}" if joining else ""))


REBASE_KEY = "HALT-REF-1"
DUST_SHARE = Decimal("0.001")  # the most dust the re-base takes as flat, a share of the book (Advisor, 09:55 UK)


def capital_in(store: Store) -> Decimal:
    """The capital put into the fund's book: every Strategy's starting balance that fund_equity counts (a reset's run
    counts only its cash, its successor the capital). What the book is worth with no gain or loss since, the
    flow-corrected reference the re-base measures the drawdown from (Advisor, 14 Oct 09:10 UK)."""
    runs, out = store.reset_runs(), store.withdrawn()
    return sum((to_decimal(s.starting_balance, "starting balance") for s in store.sleeves()
                if s.name not in runs and s.name not in out and not outside_fund_book(store, s)), Decimal(0))


def could_trade(s: Sleeve) -> bool:
    """What "running" means to the re-base: a Strategy that could open a position once the portfolio's halt clears. One
    enabled (desired running) can, unless its own status blocks every entry until the PM acts: halted, or paused by
    the PM (a pause with no end; a day's pause ends at 00:00 UTC). One stopped can't, once its process has gone."""
    if s.desired_state == "running":
        return not (s.status == "halted" or (s.status == "paused" and s.paused_until is None))
    return s.status in ("starting", "running")


def rebase_refusals(store: Store) -> list[str]:
    """Why the HALT-REF-1 re-base can't run now (Advisor conditions, 14 Oct): it has run already; a Strategy could
    trade (could_trade) or holds a position; or the dashboard's figures don't reconcile with the journal
    (start_check)."""
    from sleeve_fund.start_check import FAIL, check_books

    why = []
    if any(d["reason"].startswith(REBASE_KEY) for d in store.decisions(action="portfolio_rebase", limit=10_000)):
        why.append(f"the {REBASE_KEY} re-base has run already (decisions log): it runs once")
    for s in store.sleeves():
        if could_trade(s):
            why.append(f"{s.name} could trade once the halt clears ({s.desired_state}, {s.status}): pause it first")
        book = store.journal_book(s.name, s.starting_balance)
        if abs(book["qty"]) > 1e-12 and not is_dust(book):  # dust no venue takes an order for is flat (F229-7)
            why.append(f"{s.name} holds {book['qty']}: the book must be flat")
    why += [f"the books don't reconcile: {what}" for status, what in check_books(store).lines if status == FAIL]
    return why


def dust_holdings(store: Store) -> list[tuple[str, Decimal, Decimal]]:
    """Every dust position the re-base counts as flat (Advisor, 09:55 UK): (Strategy, qty, its value at the mark, or at
    its entry if never marked)."""
    held = []
    for s in store.sleeves():
        book = store.journal_book(s.name, s.starting_balance)
        if is_dust(book):
            last = store.last_equity(s.name)
            px = last["price"] if last is not None and last["price"] else book["entry_px"]
            qty = to_decimal(book["qty"], "qty")
            held.append((s.name, qty, qty * to_decimal(px, "mark")))
    return held


def _figure(v: Decimal | None) -> str:
    return "no mark" if v is None else f"{v:,.2f}"


def rebase_figures(store: Store) -> list[str]:
    """The re-base's figures line by line (F-HALTREF-5), printed by the dry run and the real run alike: every Strategy
    with whether the fund's book counts it and why, its starting balance, its journal's cash with the fills, fees,
    funding and insurance behind it, and what it adds to the true book and to the capital put in; each reset run's
    fills; the dust held against the cap. Reads only. The two totals are fund_equity's and capital_in's. A position
    with no mark can't be valued (strategy_equity raises): its row, the total and the dust cap say "no mark" and the
    rest still prints, so the dry run reaches its refusals (CR F285-1)."""
    from sleeve_fund.store import is_backtest

    runs = store.reset_runs()
    out = [("Strategy | book | starting balance | journal cash | fills | fees | funding | insurance | position | "
            "true book | capital")]
    equity_sum: Decimal | None = Decimal(0)
    capital_sum = Decimal(0)
    for s in store.sleeves():
        if is_backtest(s.name):
            continue
        fills = store.fills(s.name, limit=1_000_000)
        fees = sum((to_decimal(f["fee"], "fee") for f in fills), Decimal(0))
        cash = to_decimal(store.strategy_cash_pnl(s.name), "cash")
        qty = to_decimal(store.journal_book(s.name, s.starting_balance)["qty"], "qty")
        if outside_fund_book(store, s):
            where, equity, capital = "out: a test account (F229-6)", Decimal(0), Decimal(0)
        else:
            if s.name in runs:
                where, capital = f"in: a reset's run ({runs[s.name]:%d %b %H:%M} UTC), its cash only", Decimal(0)
            else:
                where, capital = "in", to_decimal(s.starting_balance, "starting balance")
            try:
                equity = strategy_equity(store, s, runs)
            except RuntimeError:  # F285-1: a position with no mark: the row says so and the rest still prints
                equity = None
        equity_sum = None if equity is None or equity_sum is None else equity_sum + equity
        capital_sum += capital
        out.append(f"{s.name} | {where} | {to_decimal(s.starting_balance, 'starting balance'):,.2f} | {cash:,.2f} | "
                   f"{len(fills)} | {fees:,.2f} | {store.funding_total(s.name):,.2f} | "
                   f"{store.insurance_total(s.name):,.2f} | {qty.normalize():f} | {_figure(equity)} | {capital:,.2f}")
    out.append(f"total | | | | | | | | | {_figure(equity_sum)} | {capital_sum:,.2f}")
    for name in runs:
        for f in sorted(store.fills(name, limit=1_000_000), key=lambda f: (f["ts"], f["id"])):
            out.append(f"  {name} fill {f['id']} {f['ts']:%d %b %H:%M:%S} {f['side']} {to_decimal(f['qty'], 'qty').normalize():f} "
                       f"at {to_decimal(f['price'], 'price'):,.2f}, fee {to_decimal(f['fee'], 'fee'):,.4f}")
    dust = dust_holdings(store)
    total = sum((abs(v) for _, _, v in dust), Decimal(0))
    try:
        book: Decimal | None = fund_equity(store)
    except RuntimeError:  # F285-1: as above, a position with no mark
        book = None
    share = f"{total / book:.4%}" if book is not None and book > 0 else "n/a"
    cap = f"{book * DUST_SHARE:,.2f}" if book is not None else "n/a: the book can't be valued"
    out.append(f"dust: {len(dust)} holding(s) worth {total:,.4f}, {share} of the book "
               f"(cap {DUST_SHARE:.1%}, {cap})"
               + "".join(f"; {n} {q.normalize():f} worth {v:,.4f}" for n, q, v in dust))
    return out


def rebase_halt_ref_1(store: Store, actor: str, commit: str, now: datetime | None = None
                      ) -> tuple[PortfolioState, PortfolioState, str]:
    """HALT-REF-1's one-off clean-up (Advisor, 14 Oct 09:00 and 09:10 UK): the paper portfolio's state missed every
    capital flow before they were wired (Strategies created after the first mark, test accounts leaving the book), so
    its HWM and halt reference sit far above the book and it halted. Run once, by an operator, from the supervisor's
    command line only (rebase.yml; no dashboard or API path); refused while a Strategy could trade (could_trade) or
    holds (dust aside), if the books don't reconcile, or if it has run before. It measures the book against the capital
    put in (capital_in): at the daily pause's 3% or more that is a real drawdown and it refuses, naming it; below, it sets the book, the HWM, the halt's reference and the
    day's start to the book now and clears the halt and the pause that stale reference set. This is the only place a
    halt is cleared other than the PM's Resume. One warning event and one decision row carry the old and new figures
    (the old HWM kept in the record), the drawdown and the commit, in the state row's transaction. Returns (before,
    after, the journaled words). Raises RuntimeError with every refusal."""
    from sqlalchemy import insert, select

    from sleeve_fund.store import decisions_t, events_t

    if not actor.strip() or not commit.strip():
        raise ValueError("an actor and the deployed commit are required")
    now = now or utcnow()
    led = DbLedger(store, lambda: (_ for _ in ()).throw(RuntimeError("a re-base runs no entry check")))
    # Everything is read under the gate's lock (CR R2, R3): no flow, mark or second run can land between the checks,
    # the book read and the write. Nothing trades meanwhile: every Strategy is flat and can't (the refusals).
    with led.lock():
        conn = led._local.conn
        refusals = rebase_refusals(store)
        if conn.execute(select(decisions_t.c.id).where(decisions_t.c.action == "portfolio_rebase",
                                                       decisions_t.c.reason.like(f"{REBASE_KEY}%"))).first():
            refusals.append(f"the {REBASE_KEY} re-base has run already (decisions log): it runs once")
        if refusals:
            raise RuntimeError(f"{REBASE_KEY} re-base refused: " + "; ".join(dict.fromkeys(refusals)))
        equity, capital = fund_equity(store), capital_in(store)
        drawdown = max(capital - equity, Decimal(0)) / capital if capital else Decimal(0)
        if drawdown >= Decimal(str(PORTFOLIO.daily_loss)):
            raise RuntimeError(f"{REBASE_KEY} re-base refused: the book {equity:,.2f} is {drawdown:.2%} under the "
                               f"{capital:,.2f} put in, a real drawdown at or over the {PORTFOLIO.daily_loss:.0%} "
                               "daily pause, which a re-base would hide")
        dust = dust_holdings(store)
        notional = sum((abs(v) for _, _, v in dust), Decimal(0))
        if equity > 0 and notional > equity * DUST_SHARE:
            raise RuntimeError(f"{REBASE_KEY} re-base refused: dust worth {notional:,.2f} across the book is "
                               f"{notional / equity:.3%} of it, over the {DUST_SHARE:.1%} the Advisor allows as flat")
        before = led.state()
        after = replace(before, equity=equity, mark_ts=now, hwm=equity, halt_reference=None, day=trading_day(now),
                        day_start_equity=equity, halted=None, paused_until=None, paused=None)
        led.set_state(after)

        def fig(x):
            return "none" if x is None else f"{x:,.2f}"

        words = (f"{REBASE_KEY}: portfolio re-based to the true book at commit {commit.strip()}: book {fig(before.equity)}"
                 f" -> {fig(equity)}, high-water mark {fig(before.hwm)} -> {fig(equity)} (the old one kept here), halt "
                 f"reference {fig(before.reference())} -> {fig(equity)}, day's start {fig(before.day_start_equity)} -> "
                 f"{fig(equity)}; {drawdown:.2%} drawdown against the {capital:,.2f} put in; dust counted as flat: "
                 f"{', '.join(f'{n} {q.normalize():f} worth {v:,.4f}' for n, q, v in dust) or 'none'}; "
                 f"{'the halt (' + before.halted + ') is cleared' if before.halted else 'no halt'}"
                 f"{', the day pause is cleared' if before.paused_until else ''}")
        conn.execute(insert(events_t).values(sleeve=None, ts=now, level="warning", kind="portfolio_rebased",
                                             message=words))
        conn.execute(insert(decisions_t).values(ts=now, actor=actor.strip(), action="portfolio_rebase", sleeve=None,
                                                reason=words))
    return before, after, words


def withdraw_archived(store: Store, actor: str, commit: str) -> list[str]:
    """F-C3B-A's one-off for the Strategies archived before it landed, which still count in the book: each is put on
    the supervisor's withdrawal (Store._put_away: WITHDRAWAL_DEFERRED), and its next poll takes each out with its own
    flow after its own mark, so the drawdown and the P&L don't move (Advisor 14 Oct 09:45 UK; F286-CR1). A reset's
    earlier run is not one: its cash stays in the book. Idempotent: one already out or already waiting is skipped.
    Refused, with nothing written, while any of them holds a position; dust no order can close is flat, as for the
    archive, and leaves with it (F286-2, Advisor D8b). Returns the names queued."""
    from sleeve_fund.store import is_backtest

    if not actor.strip() or not commit.strip():
        raise ValueError("an actor and the deployed commit are required")
    runs, out, waiting = store.reset_runs(), store.withdrawn(), set(store.withdrawal_pending())
    # One already waiting (put away after F-C3B-A, or queued by an earlier run) is the poll's to take out.
    due = [n for n in store.archived() if n not in runs and n not in out and not is_backtest(n) and n not in waiting]
    held = [f"{n} holds {float(qty):g}" for n in due for qty, flat in [store.lifecycle_flat(n)] if not flat]
    if held:
        raise RuntimeError("withdraw-archived refused, nothing written: " + "; ".join(held)
                           + " (flatten it first: a position has no settled equity to take out)")
    for n in due:
        with store._put_away(n, utcnow(), f"withdraw-archived by {actor.strip()} at {commit.strip()}"):
            pass
        store.decide(actor.strip(), "withdraw_archived", f"F-C3B-A: {n}'s equity taken out of the fund's book at "
                                                         f"commit {commit.strip()}", n)
    return due


REBASE_PAUSE = f"{REBASE_KEY}: paused for the portfolio re-base; resumed at the first test start"


def _rebase_command(store: Store, args) -> int:
    """The workflow's two runs (rebase.yml): a dry run prints the figures and any refusal and writes nothing. --yes
    first pauses, by a journaled PM pause command, each Strategy that could trade once the halt clears, waits for its
    process to apply it, then re-bases and shows the next mark. Refused for any other reason, it pauses nothing; a
    re-base refused after the pauses leaves them in force (the safe side), for the first test start to resume."""
    st = DbLedger(store, lambda: ()).state()
    print(f"now: book {st.equity}, high-water mark {st.hwm}, halt reference {st.reference()}, day's start "
          f"{st.day_start_equity}, halted {st.halted or 'no'}, paused {st.paused or 'no'}")
    refusals = rebase_refusals(store)
    to_pause = [s.name for s in store.sleeves() if could_trade(s)]
    others = [w for w in refusals if "could trade once the halt clears" not in w]  # what a pause can't answer
    if not others:
        print(f"true book (fund equity): {fund_equity(store)}; capital put in: {capital_in(store)}")
    for line in rebase_figures(store):
        print(line)
    for why in refusals:
        print(f"refused: {why}")
    if not args.yes:
        print("not written (a dry run)" + (f"; --yes pauses {', '.join(to_pause)} first" if to_pause else ""))
        return 1 if others else 0
    if others:
        print("not written; nothing paused")
        return 1
    for name in to_pause:
        store.command(name, "pause", REBASE_PAUSE, actor=args.actor)
        print(f"paused: {name}")
    deadline = time.monotonic() + args.wait
    while (left := [s.name for s in store.sleeves() if could_trade(s)]) and time.monotonic() < deadline:
        time.sleep(2)
    if left:
        print(f"not written: {', '.join(left)} didn't apply the pause within {args.wait:g} s; the pauses stay")
        return 1
    try:
        _, after, words = rebase_halt_ref_1(store, args.actor, args.commit)
    except RuntimeError as exc:
        print(f"{exc}; the pauses stay")
        return 1
    print(f"journaled: {words}")
    # The Advisor's last condition: the next mark's halt and pause checks are clean before any Strategy starts.
    Supervisor(store).mark_portfolio()
    st = DbLedger(store, lambda: ()).state()
    print(f"next mark: book {st.equity}, high-water mark {st.hwm}, halt reference {st.reference()}, day's start "
          f"{st.day_start_equity}, halted {st.halted or 'no'}, paused {st.paused or 'no'}")
    return 0 if not st.halted and not st.paused_until else 1


def _resume_identity_command(store: Store, args) -> int:
    """resume-identity: 0 once the hold is lifted (dry run: when there is one to lift), 1 when refused, naming why."""
    held = DbLedger(store, lambda: ()).identity_hold()
    print("book check hold:", held or "none")
    if args.dry_run:
        apart = positions_breaks(store)
        print(f"cash gap {identity_gap(store):,.2f}; positions: {'; '.join(apart) or 'all tie'}")
        return 0 if held else 1
    try:
        print(resume_identity_hold(store, args.actor, args.reason))
    except ValueError as exc:
        print(exc)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.supervisor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sd = sub.add_parser("seed", help="add sleeves from TOML files if missing")
    sd.add_argument("paths", nargs="+")
    cl = sub.add_parser("clear", help="stop and archive every strategy, once per entry in the file")
    cl.add_argument("path")
    sub.add_parser("book", help="print the book's strategies and headline figures")
    wa = sub.add_parser("withdraw-archived", help="F-C3B-A only: take each Strategy archived before it out of the book")
    wa.add_argument("--actor", required=True, help="who ran it, for the decision log")
    wa.add_argument("--commit", required=True, help="the deployed commit, for the journal")
    rb = sub.add_parser("rebase", help=f"{REBASE_KEY} only: re-base the paper portfolio's state to the true book, once")
    rb.add_argument("--actor", required=True, help="who ran it, for the decision log")
    rb.add_argument("--commit", required=True, help="the deployed commit, for the journal")
    rb.add_argument("--yes", action="store_true", help="pause what could trade, then write it; else a dry run")
    rb.add_argument("--wait", type=float, default=120.0, help="seconds to wait for the pauses to apply")
    rs = sub.add_parser("reset-book", help="CAP5-RESET only: the PM's clean slate of the paper book, then a fresh deposit")
    rs.add_argument("--actor", required=True, help="who ran it on the PM's word, for the decision log")
    rs.add_argument("--commit", required=True, help="the deployed commit, for the journal")
    rs.add_argument("--id", required=True, help="the reset's id, e.g. 2026-10-21-reset; a second run of it carries on")
    rs.add_argument("--deposit", default="100000", help="the fund's fresh capital, deposited once the book is empty")
    rs.add_argument("--dry-run", action="store_true", help="print the book, the steps and what would hold them")
    ri = sub.add_parser("resume-identity", help="RECON-B only: lift the book check's hold, on the PM's written word")
    ri.add_argument("--actor", required=True, help="who runs it on the PM's word, for the decision log")
    ri.add_argument("--reason", required=True, help="the PM's reason, quoted")
    ri.add_argument("--dry-run", action="store_true", help="print the hold and whether the book adds up; write nothing")
    rn = sub.add_parser("run", help="supervise sleeve processes until stopped")
    rn.add_argument("--clear", help="clean slates file to retry while one waits on a flatten")
    args = ap.parse_args(argv)
    if args.cmd == "reset-book":
        from sleeve_fund import reset_book

        return reset_book.main(args)
    store = Store(portfolio_gate=True)
    if args.cmd == "seed":
        print("added:", seed(store, args.paths) or "nothing new")
    elif args.cmd == "clear":
        print("put away:", clear(store, args.path) or "nothing")
        print("book:", book_line(store))
    elif args.cmd == "book":
        print("book:", book_line(store))
        print("book figures:", book_figures(store))
    elif args.cmd == "rebase":
        return _rebase_command(store, args)
    elif args.cmd == "resume-identity":
        return _resume_identity_command(store, args)
    elif args.cmd == "withdraw-archived":
        try:
            done = withdraw_archived(store, args.actor, args.commit)
        except RuntimeError as exc:
            print(exc)
            return 1
        print(f"queued for the supervisor's next mark: {', '.join(done) or 'none (nothing archived still counted)'}")
        if done:  # the book after it and the limits in use against it, as the dashboard shows them (F-C3B-A-UI)
            from sleeve_fund.dashboard.withdrawal import limits_in_use, words

            print(words(limits_in_use(store)))
    else:
        assert_portfolio_gate(store)
        Supervisor(store, clear_path=args.clear).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
