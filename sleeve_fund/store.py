"""The one database every part reads from: sleeves, equity marks, fills, events,
PM commands and the decision log.

Postgres in production (the same engine Supabase runs, so pointing DATABASE_URL at
a Supabase project is a config change). SQLite works for tests and quick local
runs. Tables are plain and typed so reports and BI tools can query them directly.
"""

from __future__ import annotations

from decimal import Decimal

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    func,
    insert,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from sleeve_fund.exact import EXACT
from sleeve_fund.money import stored, to_decimal

DEFAULT_URL = "sqlite:///data/sleeve_fund.db"

# The PM's reset after a liquidation (P1-RAL): ends a liquidation halt, from a new high-water mark at the remaining
# equity, once the liquidation's incident has its note (Independent Quant Advisor, 6 Oct 17:57 and 18:17).
RAL = "reset_after_liquidation"
# The book reset (Advisor 7 Oct 00:20 (a)): each strategy's own drawdown halt and daily-loss pause cleared, its
# references re-based at its equity now, journaled as an event of this kind; the command tells a running process.
BOOK_RESET = "book_reset"
COMMANDS = {"pause", "resume", "flatten", RAL, BOOK_RESET}
# Applied by the supervisor, not the strategy: restart the process so it trades under settings the PM
# changed (Store.change_settings).
RELOAD = "reload"
STATUSES = {"starting", "running", "paused", "halted", "stopped", "error"}
LEVELS = ("info", "warning", "error")

metadata = MetaData()
TS = DateTime(timezone=True)

sleeves_t = Table(
    "sleeves",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(64), unique=True, nullable=False),
    Column("strategy", String(64), nullable=False),
    Column("instrument", String(32), nullable=False),
    Column("bar_spec", String(64), nullable=False),
    Column("params", JSON, nullable=False, default=dict),
    Column("starting_balance", EXACT, nullable=False),
    Column("risk_profile", String(32), nullable=False, default="balanced"),
    Column("warmup_bars", Integer, nullable=False, default=0),
    Column("desired_state", String(16), nullable=False, default="running"),  # set by the PM
    Column("status", String(16), nullable=False, default="starting"),  # set by the sleeve
    Column("status_reason", Text, nullable=False, default=""),
    Column("paused_until", TS),
    Column("heartbeat_at", TS),
    Column("created_at", TS, nullable=False),
    Column("updated_at", TS, nullable=False),
)

equity_t = Table(
    "equity",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("ts", TS, nullable=False),
    Column("equity", EXACT, nullable=False),
    Column("cash", EXACT, nullable=False),
    Column("qty", EXACT, nullable=False),
    Column("price", EXACT, nullable=False),
    Column("benchmark", EXACT, nullable=False),
    Index("equity_sleeve_ts", "sleeve", "ts"),
)

fills_t = Table(
    "fills",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("ts", TS, nullable=False),
    Column("side", String(8), nullable=False),
    Column("qty", EXACT, nullable=False),
    Column("price", EXACT, nullable=False),
    Column("fee", EXACT, nullable=False),
    Column("order_id", String(64), nullable=False),
    Column("trade_id", String(64), nullable=False),
    Index("fills_sleeve_ts", "sleeve", "ts"),
    # DA-2: a fill is booked once. Keyed with its order: the paper venue's trade ids are deterministic per process
    # (Nautilus sandbox), so after a restart a new order's fill can carry an earlier trade id; the client order id
    # is unique (orders.order_id), so (order, trade) is not.
    Index("fills_sleeve_order_trade", "sleeve", "order_id", "trade_id", unique=True),
)

# A perpetual's funding, exchanged at each funding time while a position is held (sleeve_fund.markets):
# amount is what the strategy received (negative: paid), in the quote currency, booked to cash.
funding_t = Table(
    "funding",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("ts", TS, nullable=False),
    Column("qty", EXACT, nullable=False),
    Column("price", EXACT, nullable=False),
    Column("rate", Float, nullable=False),
    Column("amount", EXACT, nullable=False),
    # How the rate was set: "settled" (the venue's), "baseline" (missing, charged adversely) or "true_up" (a later
    # correction to the venue's rate), QA P1-O17.
    Column("kind", String(16), nullable=False, server_default="settled"),
    CheckConstraint("kind IN ('settled', 'baseline', 'true_up', 'reversal')", name="funding_kind"),
    Index("funding_sleeve_ts", "sleeve", "ts"),
)

# A perpetual's loss past the bankruptcy price, which the venue's insurance fund takes under isolated margin
# (a gap through the liquidation price): amount is what came back to the strategy's cash.
insurance_t = Table(
    "insurance",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("ts", TS, nullable=False),
    Column("price", EXACT, nullable=False),
    Column("amount", EXACT, nullable=False),
    Index("insurance_sleeve_ts", "sleeve", "ts"),
)

# The demo mirror (sleeve_fund.mirror): one row per journaled fill it copied, skipped or
# failed to copy, and a "start" row marking the fill it started after. The paper journal stays the record.
mirror_t = Table(
    "demo_mirror",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("fill_id", Integer, nullable=False),
    Column("ts", TS, nullable=False),
    Column("status", String(16), nullable=False),  # start, filled, skipped, error
    Column("instrument", String(32), nullable=False, server_default=""),
    Column("amount", EXACT, nullable=False, server_default="0"),  # signed, in the mirror venue's units
    Column("price", EXACT),
    Column("order_id", String(64), nullable=False, server_default=""),
    Column("message", Text, nullable=False, server_default=""),
    Index("demo_mirror_sleeve_fill", "sleeve", "fill_id"),
)

events_t = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64)),
    Column("ts", TS, nullable=False),
    Column("level", String(8), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("message", Text, nullable=False),
    Index("events_ts", "ts"),
)

commands_t = Table(
    "commands",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("command", String(32), nullable=False),
    Column("reason", Text, nullable=False),
    Column("created_at", TS, nullable=False),
    Column("applied_at", TS),
    # The liquidation incident (events.id) a reset after liquidation answers; one command per incident (P1-RAL).
    Column("incident", Integer, ForeignKey("events.id", ondelete="RESTRICT")),
    Index("commands_incident", "incident", unique=True, postgresql_where=text("incident IS NOT NULL"),
          sqlite_where=text("incident IS NOT NULL")),
)

decisions_t = Table(
    "decisions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("ts", TS, nullable=False),
    Column("actor", String(32), nullable=False),
    Column("action", String(32), nullable=False),
    Column("sleeve", String(64)),
    Column("reason", Text, nullable=False),
)
# PM acknowledgements of warning/error events (the alerts inbox). A separate table, so
# adding it is a plain CREATE TABLE on start-up rather than a change to an existing one.
acks_t = Table(
    "alert_acks",
    metadata,
    Column("event_id", Integer, ForeignKey("events.id"), primary_key=True),
    Column("ts", TS, nullable=False),
    Column("actor", String(32), nullable=False),
    Column("note", Text, nullable=False, default=""),
)
# Every order a sleeve sends, with why it was sent, written when the decision is made (not
# reconstructed later). Also a separate table, so it arrives as a plain CREATE TABLE.
orders_t = Table(
    "orders",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("order_id", String(64), nullable=False, unique=True),  # the client order id fills carry
    Column("ts", TS, nullable=False),  # decided and sent
    Column("updated_at", TS, nullable=False),
    Column("side", String(8), nullable=False),
    Column("order_type", String(16), nullable=False),
    Column("qty", EXACT, nullable=False),
    Column("status", String(16), nullable=False),  # one of ORDER_STATUSES
    Column("filled_qty", EXACT, nullable=False, default=0.0),
    Column("avg_px", EXACT),
    Column("fee", EXACT, nullable=False, default=0.0),
    Column("intent", String(16), nullable=False),  # one of INTENTS: what the order was for
    Column("reason", Text, nullable=False),  # plain English, e.g. "10-bar average crossed above 30-bar"
    Column("signal", JSON, nullable=False, default=dict),  # indicator values and price at the decision
    Column("message", Text, nullable=False, default=""),  # venue or risk-engine text on reject/cancel
    Index("orders_sleeve_ts", "sleeve", "ts"),
)
# When each paper or live order's decision bar closed and arrived, the decision, the send, the venue's acceptance
# and its fills, to the microsecond (v2 P1-2): close-to-fill per strategy. Our clock throughout, except venue_ts,
# the venue's own stamp on the last fill, which the clock check compares against. Backtests keep none.
order_timings_t = Table(
    "order_timings",
    metadata,
    Column("order_id", String(64), ForeignKey("orders.order_id"), primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("bar_close", TS),  # the decision bar's close; none when a tick or the risk guard decided
    Column("bar_recv", TS),  # when the node had that bar
    Column("decided", TS),
    Column("sent", TS),  # none for an order held by the strategy until it can fill (a paper post-only)
    Column("accepted", TS),
    Column("first_fill", TS),
    Column("last_fill", TS),
    Column("venue_ts", TS),
    Index("order_timings_sleeve_decided", "sleeve", "decided"),
)
# The stop and target an open position works to after the PM edited them, or after a restart set them
# again from the market, as shares of its entry price (the entry order's signal holds the plan it was
# entered with). The latest row per entry order is in force. A new table: CREATE TABLE.
exit_plans_t = Table(
    "exit_plans",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("entry_order", String(64), nullable=False),  # the entry's client order id
    Column("ts", TS, nullable=False),
    Column("kind", String(16), nullable=False),  # "edit": the PM changed the settings; "restart": set again
    Column("stop_frac", Float),
    Column("tp_frac", Float),
    Column("basis", Text, nullable=False, default=""),  # how the stop was set, in words
    Column("stop_cfg", JSON),  # the stop settings it was set from
    Column("risk_amount", EXACT),  # the trade's 1R from now on: the larger of the entry's and this plan's
    Column("planned_r", Float),
    Column("event_id", Integer),  # the newest settings-change event it applies
    Index("exit_plans_entry", "sleeve", "entry_order", "id"),
)
# Venue accounts (see sleeve_fund/accounts.py), which sleeve trades on which, and whether the
# supervisor can see a key for each live account. New tables, so they arrive as CREATE TABLE.
accounts_t = Table(
    "accounts",
    metadata,
    Column("name", String(41), primary_key=True),
    Column("kind", String(8), nullable=False),  # paper | live
    Column("venue", String(16), nullable=False, default=""),  # a live account's venue; "" for paper (any venue)
    Column("note", Text, nullable=False, default=""),
    Column("created_at", TS, nullable=False),
)
sleeve_accounts_t = Table(
    "sleeve_accounts",
    metadata,
    Column("sleeve", String(64), ForeignKey("sleeves.name"), primary_key=True),
    Column("account", String(41), ForeignKey("accounts.name"), nullable=False),
    Column("assigned_at", TS, nullable=False),
)
# The venue a strategy trades on, where it isn't the default (sleeve_fund.venues). A table of its own, so
# it arrives as CREATE TABLE: a strategy with no row here trades on the default venue, as every one did.
sleeve_venues_t = Table(
    "sleeve_venues",
    metadata,
    Column("sleeve", String(64), ForeignKey("sleeves.name"), primary_key=True),
    Column("venue", String(16), nullable=False),
)
account_keys_t = Table(
    "account_keys",
    metadata,
    Column("account", String(41), ForeignKey("accounts.name"), primary_key=True),
    Column("present", Integer, nullable=False),  # 1 if the supervisor sees both key and secret
    Column("checked_at", TS, nullable=False),
)
# Fee schedules read from each connected live account by the supervisor (query-only key, server
# only). Every fetch is kept, so a result can always say which schedule it used.
fee_schedules_t = Table(
    "fee_schedules",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("venue", String(16), nullable=False),
    Column("account", String(41), ForeignKey("accounts.name"), nullable=False),
    Column("maker", Float, nullable=False),
    Column("taker", Float, nullable=False),
    Column("fetched_at", TS, nullable=False),
)
# Typical half bid-ask spread per instrument, measured from a paper sleeve's live quotes. Backtests
# charge it on every order that takes liquidity, since their bars carry trade prices, not quotes.
spreads_t = Table(
    "spreads",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("venue", String(16), nullable=False),
    Column("instrument", String(32), nullable=False),
    Column("half_spread", Float, nullable=False),  # median (ask - bid) / 2 / mid
    Column("samples", Integer, nullable=False),
    Column("measured_at", TS, nullable=False),
)
# Instruments the PM asked the history collector to store, from the Research page, so a study can run
# on an instrument no strategy trades yet. A new table: CREATE TABLE.
history_requests_t = Table(
    "history_requests",
    metadata,
    Column("venue", String(16), primary_key=True),
    Column("instrument", String(32), primary_key=True),
    Column("since", TS, nullable=False),  # where the backfill starts
    Column("requested_at", TS, nullable=False),
)
# Accounts the PM has retired: no strategy can be moved onto one or started on one. A separate table,
# so it arrives as a plain CREATE TABLE; reinstating deletes the row.
account_retired_t = Table(
    "account_retired",
    metadata,
    Column("account", String(41), ForeignKey("accounts.name"), primary_key=True),
    Column("retired_at", TS, nullable=False),
)
# Stopped sleeves the PM has put away. Their history stays; they just leave the everyday lists.
sleeve_archive_t = Table(
    "sleeve_archive",
    metadata,
    Column("sleeve", String(64), ForeignKey("sleeves.name"), primary_key=True),
    Column("archived_at", TS, nullable=False),
)
# When each paper strategy last saw a trade or quote from its venue: the price feed's age on its page.
# Written at most every few seconds by the strategy's process. A new table: CREATE TABLE.
feed_seen_t = Table(
    "feed_seen",
    metadata,
    Column("sleeve", String(64), ForeignKey("sleeves.name"), primary_key=True),
    Column("seen_at", TS, nullable=False),
)
# A PM's "resync the demo copy" request, for the mirror container to act on (only it holds the demo keys):
# sleeve null means every mirrored strategy. done_at and result once it has. A new table: CREATE TABLE.
mirror_requests_t = Table(
    "mirror_requests",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=True),
    Column("reason", Text, nullable=False),
    Column("actor", String(64), nullable=False),
    Column("created_at", TS, nullable=False),
    Column("done_at", TS, nullable=True),
    Column("result", Text, nullable=False, default=""),
)
# A PM's "reset strategy" (5 Oct 2026): flatten, put the run so far away under its own name, and start again
# at the starting capital. run is the name the old run was put away under, once done; restart says whether
# the strategy was running when asked. A new table: CREATE TABLE.
resets_t = Table(
    "strategy_resets",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), nullable=False),
    Column("reason", Text, nullable=False),
    Column("actor", String(64), nullable=False),
    Column("restart", Integer, nullable=False, default=1),
    Column("created_at", TS, nullable=False),
    Column("done_at", TS, nullable=True),
    Column("run", String(64), nullable=False, default=""),
)
# A pause or halt in force when a reset was asked for, carried onto the fresh run so a reset never lifts
# the kill switch, a risk halt or the PM's pause (round 13, U13-4). A new table: CREATE TABLE.
reset_holds_t = Table(
    "strategy_reset_holds",
    metadata,
    Column("reset_id", Integer, primary_key=True),
    Column("status", String(16), nullable=False),
    Column("status_reason", Text, nullable=False, default=""),
    Column("paused_until", TS, nullable=True),
)
# Tables whose rows are a strategy's own run: moved to the run's name on a reset. sleeve_accounts and
# sleeve_venues are copied (both need them); feed_seen starts afresh; backtests are never strategies on a book.
# A saved backtest: one row per run, its result as the backtest page shows it, and its journal (orders,
# fills, equity marks and events) in the ordinary tables under a sleeve named BACKTEST_PREFIX + id, so
# the run opens in the same Orders, Trades and strategy screens as paper. A new table: CREATE TABLE.
backtests_t = Table(
    "backtests",
    metadata,
    Column("id", String(16), primary_key=True),
    Column("sleeve", String(64), nullable=False),
    Column("key", String(64), nullable=False),  # the settings that made it, hashed, to reuse a fresh run
    Column("title", Text, nullable=False),
    Column("query", Text, nullable=False),  # the backtest page's settings, as a query string
    Column("created_at", TS, nullable=False),
    # JSON text, not a JSON column: statistics such as a Sharpe ratio can be NaN, which Python's json
    # round-trips but Postgres's json type refuses.
    Column("result", Text, nullable=False),
    Index("backtests_key", "key", "created_at"),
)
# What each paper strategy's model reads on the forming candle: its long and short rules, met or not, for the
# strategy page's Signals tab. One row per strategy, replaced at most every few seconds by its process; display
# only. JSON text, as backtests.result. A new table: CREATE TABLE.
signal_state_t = Table(
    "signal_state",
    metadata,
    Column("sleeve", String(64), ForeignKey("sleeves.name"), primary_key=True),
    Column("ts", TS, nullable=False),
    Column("payload", Text, nullable=False),
)
# Every variant ever tested, for the research guardrails (v2 P1-6): the count that deflates a result's Sharpe.
# Append-only: nothing here is updated or deleted, and it has no strategy column, so a reset never touches it.
# One row per evaluation; a variant is one (definition_hash, code_version, dataset). A new table: CREATE TABLE.
trials_t = Table(
    "trials",
    metadata,
    Column("id", String(16), primary_key=True),
    Column("definition_hash", String(64), nullable=False),  # the whole definition, settings included
    Column("idea_hash", String(64), nullable=False),  # the definition with its tunable settings left out
    Column("code_version", String(40), nullable=False),  # the indicator library's version: changed code, new variant
    Column("definition_name", Text, nullable=False),
    Column("family", String(32), nullable=False),
    Column("settings", Text, nullable=False),  # JSON text, sorted keys; display only, nothing filters on it
    Column("dataset", String(128), nullable=False),
    Column("stage", String(16), nullable=False),
    Column("source", String(16), nullable=False),
    Column("sharpe", Float),  # annualised; NULL where it is undefined, never NaN
    Column("trades", Integer),
    Column("oos_trades", Integer),
    # Not a foreign key: saved backtests are pruned to the latest 50, and a trial outlives its run.
    Column("backtest_id", String(16)),
    Column("created_at", TS, nullable=False),
    # The bars the evaluation read. NULL is unknown (the old idea counter), which a holdout check counts as
    # overlapping: the safe side.
    Column("data_start", TS),
    Column("data_end", TS),
    Column("status", String(16), nullable=False, server_default="ok"),
    Column("error", Text),  # why a failed row's count failed; NULL on an ok row
    CheckConstraint("status IN ('ok', 'failed')", name="trials_status"),
    Index("trials_idea_hash", "idea_hash"),
    Index("trials_definition_dataset", "definition_hash", "dataset"),
)
# The held-back period each idea has opened (v2 P1-7, C4): one per idea and underlying, at any timeframe or
# venue, for ever. Append-only, and no strategy column, so a reset never touches it.
holdout_locks_t = Table(
    "holdout_locks",
    metadata,
    Column("id", String(16), primary_key=True),
    Column("idea_hash", String(64), nullable=False),
    Column("underlying", String(16), nullable=False),  # the base asset, upper-case: BTC covers every BTC pair
    Column("period_start", TS),  # NULL when imported from the idea counter, which kept no dates
    Column("period_end", TS),
    Column("opened_at", TS, nullable=False),
    # Not a foreign key, as trials.backtest_id: the opening is kept even when its trial row is not there.
    Column("trial_id", String(16)),
    Column("source", String(16), nullable=False),
    # claimed before the look, opened once it produced a result, crashed when the look failed after the claim:
    # spent either way, but a crash is not a failed result (Advisor and Head of Engineering, 6 Oct 2026).
    Column("status", String(16), nullable=False, server_default="opened"),
    UniqueConstraint("idea_hash", "underlying", name="holdout_locks_idea_underlying"),
)
HOLDOUT_SOURCES = ("study", "ledger_import")
HOLDOUT_STATUSES = ("claimed", "opened", "crashed")
# Stages a new trial records. Imported idea-counter rows keep the study's own stage names (sensitivity, wf_train...).
TRIAL_STAGES = ("in_sample", "out_of_sample", "holdout")
# "strategy": a paper strategy created, cloned or re-set: a variant chosen to run, counted though it has no
# Sharpe yet (QA P1-T1). "engineering": a run deliberately marked as a fixture, the only kind not counted, and
# never the default (Advisor, 6 Oct 2026).
TRIAL_SOURCES = ("study", "backtest", "strategy", "optimiser", "engineering", "ledger_import")
# "ok": a counted evaluation. "failed": a run whose count failed (QA P1-T8), recorded against its idea under its own
# source with its settings and the error, and no Sharpe, trades or dates (Data Architect, 6 Oct 2026).
TRIAL_STATUSES = ("ok", "failed")
# Events that say the strategy's own code raised: a handler, or the risk check's tick (see
# LongFlatStrategy._report).
ERROR_KINDS = ("handler_failed", "tick_failed")
# The event a PM's "Reset after liquidation" journals (item RAL): the one thing that ends a liquidation halt.
# The engine (#155) and the dashboard both read it from here.
LIQUIDATION_RESET = "liquidation_reset"
# The note on a liquidation's incident that a reset after liquidation needs (Advisor 18:17): an events row of its own,
# "Incident #<id> note by <author>: ...", so the incident it answers is in its words.
INCIDENT_NOTE = "incident_note"
WHY_STOP_FIELD = "why the half-liquidation stop did not protect the position"
# Backtest names can't collide with a strategy's: those are lower-case letters, digits and dashes.
BACKTEST_PREFIX = "bt:"
# "triggered": paper's watched stop (not an order at the venue) when it fired, its market stop-loss sent for it.
ORDER_STATUSES = ("submitted", "accepted", "partially_filled", "filled", "canceled", "rejected", "denied", "expired",
                  "triggered")
FINISHED_ORDER_STATUSES = ("filled", "canceled", "rejected", "denied", "expired", "triggered")
OPEN_ORDER_STATUSES = ("submitted", "accepted", "partially_filled")
INTENTS = ("entry", "exit", "stop_loss", "take_profit", "risk_halt", "risk_pause", "pm_flatten", "rebalance",
           "liquidation", "liquidation_cut")  # the venue would take it; cut back before it does (String(16))


def _check_rebook(order_id: str, row) -> None:
    """Refuse any re-booking but a filled stop-loss becoming a liquidation (Store/MemoryJournal.rebook_liquidation)."""
    if row is None:
        raise ValueError(f"no order {order_id!r} to re-book")
    if row["intent"] != "stop_loss" or not row["filled_qty"]:
        raise ValueError(f"order {order_id!r} can't be re-booked as a liquidation: only a filled stop-loss can "
                         f"(it is {row['intent']}, {row['filled_qty']:g} filled)")


def _rebook_words(order_id: str, reason: str) -> str:
    return f"Order {order_id} re-booked from stop_loss to liquidation: {reason}"


def exact_sum(a: float, b: float) -> float:
    """Two quantities added as the decimals they print as: 0.05 + 0.28253027 is 0.33253027, not the float
    sum 0.33253026999999996, so an order's filled quantity matches what was ordered."""
    return float(Decimal(repr(float(a))) + Decimal(repr(float(b))))


DUST = Decimal("1e-10")  # a position closer to flat than this is flat: the smallest lot is 1e-8


# A position worth less than this (in the quote currency) is below every venue's smallest order, so no flatten can
# close it: a reset treats it as flat (QA P1-D6). Venues' minimums are a few units (5 on the perpetuals).
DUST_NOTIONAL = 1.0


def is_dust(book: dict) -> bool:
    """A journal book (replay_book) holding a position too small for any venue to take an order for."""
    return 1e-12 < abs(book["qty"]) and abs(book["qty"]) * (book["entry_px"] or 0) < DUST_NOTIONAL


def replay_book(fills, starting_balance, funding=0, insurance=0) -> dict:
    """Cash, signed position and average entry from fills in time order, plus funding received and any
    shortfall the venue's insurance fund took.

    Spot-style cash: a buy costs qty * price + fee and a sell returns qty * price - fee, so a short holds
    the sale's proceeds as cash against a negative position and equity is cash + qty * price on either
    side. Fills that add to a position average its entry; fills that reduce it leave the entry alone; a
    fill that crosses through flat opens the remainder at its own price. The position is summed in
    Decimal from each fill as written: a float sum of many XRP-sized fills carries noise of a few 1e-12
    that could tip a one-lot difference over reconcile's tolerance. A long-only strategy's journal never
    goes negative; if it does, the negative stays visible so reconciliation catches it. entry_fees: the fees paid
    to open the position still held, pro-rated to what is left of it after a reduction.

    Exact (DA-9): every figure is a Decimal, each fill taken as written (a float as the decimal it prints as)."""
    D = to_decimal
    cash, qty, entry, n, fees = D(starting_balance, "starting balance"), D(0), None, 0, D(0)
    big, legs = 0.0, 0  # the largest fill and the fills since the position was last flat
    for f in fills:
        n += 1
        price, q, fee = D(f["price"], "price"), D(f["qty"], "qty"), D(f["fee"], "fee")
        sign = 1 if f["side"] == "BUY" else -1
        cash -= sign * q * price + fee
        new = qty + sign * q
        big, legs = max(big, abs(float(q))), legs + 1
        # Float residue from the fills' own arithmetic: below DUST, or for fills too large for a float to hold
        # every lot (1.6e8 units: one float step is 3e-8, wider than a 1e-8 lot) within a few float steps per
        # fill since flat. Left in, an exit back to flat read as 3e-8 held (review round 13, E13-2).
        if abs(new) < DUST or abs(new) <= (legs + 2) * Decimal(math.ulp(big)):
            new = D(0)
        if new == 0:
            legs = 0
            entry, fees = None, D(0)
        elif qty == 0 or (qty > 0) == (sign > 0):  # opening or adding
            entry = ((entry or 0) * abs(qty) + q * price) / abs(new)
            fees += fee
        elif (new > 0) != (qty > 0):  # through flat: what is left opened at this fill's price
            entry = price
            fees = fee * abs(new) / q
        else:  # reducing
            fees *= abs(new) / abs(qty)
        qty = new
    funding, insurance = D(funding, "funding"), D(insurance, "insurance")
    return {"cash": cash + funding + insurance, "qty": qty, "entry_px": entry, "fills": n,
            "funding": funding, "insurance": insurance, "entry_fees": fees}


def is_backtest(name: str | None) -> bool:
    return bool(name) and name.startswith(BACKTEST_PREFIX)


def _not_backtest(col):
    """Rows that belong to paper or live (or to no strategy), not to a saved backtest."""
    return or_(col.is_(None), ~col.like(BACKTEST_PREFIX + "%"))


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _from_ns(ns: int) -> datetime:
    """UNIX nanoseconds -> an aware UTC datetime to the microsecond (utcnow() drops them)."""
    return _EPOCH + timedelta(microseconds=ns // 1000)


def _aware(ts: datetime | None) -> datetime | None:
    # SQLite hands back naive datetimes; everything we store is UTC.
    if ts is not None and ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


@dataclass
class Sleeve:
    id: int
    name: str
    strategy: str
    instrument: str
    bar_spec: str
    params: dict
    starting_balance: float
    risk_profile: str
    warmup_bars: int
    desired_state: str
    status: str
    status_reason: str
    paused_until: datetime | None
    heartbeat_at: datetime | None
    created_at: datetime
    updated_at: datetime
    venue: str | None = None  # None: the default venue (sleeve_fund.venues.DEFAULT_VENUE)

    @classmethod
    def from_row(cls, row, venue: str | None = None) -> "Sleeve":
        d = dict(row._mapping)
        for k in ("paused_until", "heartbeat_at", "created_at", "updated_at"):
            d[k] = _aware(d[k])
        d["params"] = dict(d["params"] or {})
        # A configured balance, read by the engine's float figures: the journal (journal_book) takes it exactly.
        d["starting_balance"] = float(d["starting_balance"])
        return cls(**d, venue=venue)


def _rows(result) -> list[dict]:
    out = []
    for r in result:
        d = dict(r._mapping)
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = _aware(v)
        out.append(d)
    return out


def make_engine(url: str | None = None) -> Engine:
    url = url or os.environ.get("DATABASE_URL", DEFAULT_URL)
    if url.startswith("postgres://"):  # Supabase and Heroku style
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
        os.makedirs(os.path.dirname(url[len("sqlite:///"):]) or ".", exist_ok=True)
    engine = create_engine(url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=10000")
            cur.close()
    return engine



def check_trial(r: dict) -> None:
    """Raises ValueError for a trials row the register would refuse."""
    if r["source"] not in TRIAL_SOURCES:
        raise ValueError(f"a trial's source is one of {TRIAL_SOURCES}, got {r['source']!r}")
    if r.get("status", "ok") not in TRIAL_STATUSES:
        raise ValueError(f"a trial's status is one of {TRIAL_STATUSES}, got {r['status']!r}")
    if r["source"] != "ledger_import" and r["stage"] not in TRIAL_STAGES:
        raise ValueError(f"a trial's stage is one of {TRIAL_STAGES}, got {r['stage']!r}")
    sharpe = r.get("sharpe")
    if sharpe is not None and not math.isfinite(sharpe):
        raise ValueError("a trial's Sharpe is a finite number or None, never NaN or infinite")


def _put_trials(c, rows: list[dict]) -> int:
    """Insert trials on an open transaction, skipping ids already there. Returns how many were added."""
    have = {i for (i,) in c.execute(select(trials_t.c.id).where(trials_t.c.id.in_([r["id"] for r in rows])))}
    new = [{**r, "created_at": r.get("created_at") or utcnow(), "data_start": r.get("data_start"),
            "data_end": r.get("data_end"), "status": r.get("status", "ok"), "error": r.get("error")}
           for r in rows if r["id"] not in have]
    new = list({r["id"]: r for r in new}.values())  # the same row twice in one call counts once
    if new:
        c.execute(insert(trials_t), new)
    return len(new)


def _put_trial_in(c, row: dict) -> None:
    """Write a run's trial on the run's own save transaction. A SAVEPOINT holds the insert, so a row another
    process wrote between the check and the insert is skipped without aborting the save's transaction, which
    Postgres would otherwise do (Data Architect)."""
    try:
        with c.begin_nested():
            _put_trials(c, [row])
    except IntegrityError:
        if not c.execute(select(trials_t.c.id).where(trials_t.c.id == row["id"])).first():
            raise

LABEL_CLEARED = {"drawdown_halt": "drawdown halt", "daily_pause": "daily-loss pause"}


def _cleared_words(store: "Store", s: "Sleeve") -> str:
    """What a book reset clears for one strategy, with the old reference and level (Advisor 7 Oct 00:20)."""
    from sleeve_fund import risk

    profile = risk.profile(s.risk_profile)
    if s.status == "halted":
        peak = store.peak_equity(s.name) or s.starting_balance
        return (f"the drawdown halt ({s.status_reason}) is cleared: its reference, the high-water mark of {peak:,.2f} "
                f"against a {profile.max_drawdown:.0%} limit, is re-based with the book")
    return (f"the daily-loss pause ({s.status_reason}) is cleared: its reference, the day's opening equity, against a "
            f"{profile.daily_loss:.0%} limit, is re-based with the book")


class BookResetRefused(ValueError):
    """A book reset refused before it changed anything: `code` is stable, the text is the PM's."""

    def __init__(self, code: str, text: str) -> None:
        super().__init__(text)
        self.code = code


class Store:
    def __init__(self, url: str | None = None, engine: Engine | None = None) -> None:
        self.engine = engine or make_engine(url)
        # For another process to open the same journal; an in-memory database can't be shared.
        url = None if engine is not None else (url or os.environ.get("DATABASE_URL", DEFAULT_URL))
        self.url = None if url in (None, "sqlite://", "sqlite:///:memory:") else url
        metadata.create_all(self.engine)

    @classmethod
    def in_memory(cls) -> "Store":
        """A private, throwaway store (one connection shared, so every call sees the same tables):
        the journal a backtest's runtime writes to."""
        from sqlalchemy.pool import StaticPool

        return cls(engine=create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}))

    # --- sleeves -----------------------------------------------------------------

    def create_sleeve(
        self,
        *,
        name: str,
        strategy: str,
        instrument: str,
        bar_spec: str,
        starting_balance: float,
        params: dict | None = None,
        risk_profile: str = "balanced",
        warmup_bars: int = 0,
        desired_state: str = "running",
        venue: str | None = None,
        trial: dict | None = None,
    ) -> Sleeve:
        """trial: the strategy's row in the trials register, written in the same transaction, so a strategy is
        never made without being counted, nor counted without being made (QA P1-T8)."""
        if trial is not None:
            check_trial(trial)
        ts = utcnow()
        with self.engine.begin() as c:
            c.execute(insert(sleeves_t).values(
                name=name, strategy=strategy, instrument=instrument, bar_spec=bar_spec, params=params or {},
                starting_balance=starting_balance, risk_profile=risk_profile, warmup_bars=warmup_bars,
                desired_state=desired_state, status="starting" if desired_state == "running" else "stopped",
                status_reason="", created_at=ts, updated_at=ts,
            ))
            if venue:
                c.execute(insert(sleeve_venues_t).values(sleeve=name, venue=venue.upper()))
            if trial is not None:
                _put_trial_in(c, trial)
        return self.sleeve(name)

    def sleeve(self, name: str) -> Sleeve:
        with self.engine.connect() as c:
            row = c.execute(select(sleeves_t).where(sleeves_t.c.name == name)).first()
            venue = c.execute(select(sleeve_venues_t.c.venue).where(sleeve_venues_t.c.sleeve == name)).scalar()
        if row is None:
            raise KeyError(f"no strategy {name!r}")
        return Sleeve.from_row(row, venue)

    def sleeves(self, include_backtests: bool = False) -> list[Sleeve]:
        """Paper and live strategies. Saved backtests are left out unless asked for: they never run,
        count towards the book, or raise alerts."""
        q = select(sleeves_t).order_by(sleeves_t.c.created_at, sleeves_t.c.id)
        if not include_backtests:
            q = q.where(_not_backtest(sleeves_t.c.name))
        with self.engine.connect() as c:
            rows = c.execute(q).all()
            venues = dict(c.execute(select(sleeve_venues_t.c.sleeve, sleeve_venues_t.c.venue)).all())
        return [Sleeve.from_row(r, venues.get(r.name)) for r in rows]

    def _update_sleeve(self, name: str, **values) -> None:
        with self.engine.begin() as c:
            c.execute(update(sleeves_t).where(sleeves_t.c.name == name).values(updated_at=utcnow(), **values))

    def set_desired_state(self, name: str, state: str) -> None:
        if state not in {"running", "stopped"}:
            raise ValueError(f"bad desired state {state!r}")
        if is_backtest(name):
            raise ValueError("a saved backtest can't be started")
        self._update_sleeve(name, desired_state=state)

    def set_status(self, name: str, status: str, reason: str = "", paused_until: datetime | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"bad status {status!r}")
        self._update_sleeve(name, status=status, status_reason=reason, paused_until=paused_until)

    def heartbeat(self, name: str, at: datetime | None = None) -> None:
        """at: the strategy's clock (the wall clock in paper, the engine's in a replay); now by default."""
        with self.engine.begin() as c:
            c.execute(update(sleeves_t).where(sleeves_t.c.name == name).values(heartbeat_at=at or utcnow()))

    def feed_seen(self, name: str, at: datetime) -> None:
        """Paper only: the time of the latest trade or quote the strategy's venue sent."""
        with self.engine.begin() as c:
            c.execute(feed_seen_t.delete().where(feed_seen_t.c.sleeve == name))
            c.execute(insert(feed_seen_t).values(sleeve=name, seen_at=at))

    def last_feed(self, name: str) -> datetime | None:
        with self.engine.connect() as c:
            row = c.execute(select(feed_seen_t.c.seen_at).where(feed_seen_t.c.sleeve == name)).first()
        return _aware(row[0]) if row else None

    # --- journal -----------------------------------------------------------------

    def record_equity(self, sleeve: str, *, equity: float, cash: float, qty: float, price: float,
                      benchmark: float, ts: datetime | None = None) -> None:
        equity, cash, qty, price, benchmark = _exact(equity=equity, cash=cash, qty=qty, price=price,
                                                     benchmark=benchmark)
        with self.engine.begin() as c:
            c.execute(insert(equity_t).values(sleeve=sleeve, ts=ts or utcnow(), equity=equity, cash=cash,
                                              qty=qty, price=price, benchmark=benchmark))

    def record_fill(self, sleeve: str, *, side: str, qty: float, price: float, fee: float,
                    order_id: str, trade_id: str, ts: datetime | None = None) -> str:
        """Book a fill once (DA-2). Returns "new"; "same" for the same fill again (a replay after a reconnect or a
        restart: nothing changes); or "differs" when a fill with this (strategy, order, trade) is already booked
        with other values: the booked one is kept, nothing else is written, and an error event (alerts inbox) asks a
        person to check it."""
        booked = self._book(sleeve, side, qty, price, fee, order_id, trade_id, ts, with_order=False)
        if booked == "differs":
            self.event(sleeve, "error", "fill_conflict", _conflict_words(order_id, trade_id, side, qty, price, fee), ts=ts)
        return booked

    def book_fill(self, sleeve: str, *, side: str, qty: float, price: float, fee: float, order_id: str,
                  trade_id: str, ts: datetime | None = None) -> str:
        """A fill as the runtime books it (DA-2): the fill row and its order's filled quantity, price and fee in ONE
        transaction, so a replayed fill can never move the order twice, and the fill event. The same fill again
        changes nothing; a different fill under a booked (strategy, order, trade) is kept out and raised as an
        error event (alerts inbox) for a person to check. Returns as record_fill."""
        booked = self._book(sleeve, side, qty, price, fee, order_id, trade_id, ts, with_order=True)
        if booked == "new":
            self.event(sleeve, "info", "fill", f"{side} {qty:g} @ {price:,.2f}, fee {fee:,.2f}", ts=ts)
        elif booked == "differs":
            self.event(sleeve, "error", "fill_conflict", _conflict_words(order_id, trade_id, side, qty, price, fee), ts=ts)
        return booked

    def _book(self, sleeve, side, qty, price, fee, order_id, trade_id, ts, with_order: bool) -> str:
        qty, price, fee = _exact(qty=qty, price=price, fee=fee)
        key = (fills_t.c.sleeve == sleeve) & (fills_t.c.order_id == order_id) & (fills_t.c.trade_id == trade_id)
        for _ in range(2):  # a concurrent writer of the same fill: the second pass reads its row
            try:
                with self.engine.begin() as c:
                    row = c.execute(select(fills_t.c.side, fills_t.c.qty, fills_t.c.price, fills_t.c.fee)
                                    .where(key)).first()
                    if row is not None:
                        return "same" if _same_fill(row, side, qty, price, fee) else "differs"
                    c.execute(insert(fills_t).values(sleeve=sleeve, ts=ts or utcnow(), side=side, qty=qty,
                                                     price=price, fee=fee, order_id=order_id, trade_id=trade_id))
                    if with_order:
                        self._update_order(c, order_id, fill_qty=qty, fill_px=price, fee=fee)
                    return "new"
            except IntegrityError:
                continue
        raise RuntimeError(f"fill {order_id}/{trade_id} of {sleeve} could not be booked or read")

    def record_order(self, sleeve: str, *, order_id: str, side: str, qty: float, intent: str, reason: str,
                     signal: dict | None = None, order_type: str = "MARKET", ts: datetime | None = None,
                     timing: dict | None = None) -> None:
        """timing: the decision's stamps (bar_close, bar_recv, decided, as UNIX ns), written as the order's
        order_timings row in the same transaction, so that row never exists without its order (DA-8)."""
        if intent not in INTENTS:
            raise ValueError(f"bad intent {intent!r}")
        now = ts or utcnow()
        with self.engine.begin() as c:
            c.execute(insert(orders_t).values(sleeve=sleeve, order_id=order_id, ts=now, updated_at=now, side=side,
                                              order_type=order_type, qty=qty, status="submitted", filled_qty=0.0,
                                              fee=0.0, intent=intent, reason=reason, signal=signal or {},
                                              message=""))
            if timing:
                c.execute(insert(order_timings_t).values(order_id=order_id, sleeve=sleeve, **{
                    k: _from_ns(v) for k, v in timing.items() if v is not None}))

    def record_timing(self, sleeve: str, order_id: str, *, fill: int | None = None, **stamps: int | None) -> None:
        """An order's timing stamps (UNIX ns, kept to the microsecond), filled in as they come (the send, the
        acceptance, each fill) on the row record_order wrote with the decision; ignored for an order with no row
        (a risk stop, a restore: nothing decided them on a bar). A call with `decided` for an order with no row
        adds it. fill: our clock at a fill; the first is kept as first_fill, every one moves last_fill."""
        values = {k: _from_ns(v) for k, v in stamps.items() if v is not None}
        if fill is not None:
            values["last_fill"] = _from_ns(fill)
        with self.engine.begin() as c:
            row = c.execute(select(order_timings_t.c.first_fill).where(order_timings_t.c.order_id == order_id)).first()
            if row is None:
                if "decided" in values:
                    c.execute(insert(order_timings_t).values(order_id=order_id, sleeve=sleeve, **values))
            elif values:
                if fill is not None and row.first_fill is None:
                    values["first_fill"] = values["last_fill"]
                c.execute(update(order_timings_t).where(order_timings_t.c.order_id == order_id).values(**values))

    def timings(self, sleeve: str, limit: int = 500) -> list[dict]:
        """The strategy's latest order timings, newest decision first."""
        q = (select(order_timings_t).where(order_timings_t.c.sleeve == sleeve)
             .order_by(order_timings_t.c.decided.desc()).limit(limit))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def update_order(self, order_id: str, *, status: str | None = None, message: str | None = None,
                     fill_qty: float = 0.0, fill_px: float | None = None, fee: float = 0.0,
                     qty: float | None = None, intent: str | None = None) -> None:
        """Move an order on (accepted, cancelled, rejected...) or add a fill to it. Unknown ids are ignored:
        orders sent before this journal existed have no row. intent: what the order turned out to carry out (a
        backtest's resting stop booked as the target, when its bar opened through the target)."""
        if status is not None and status not in ORDER_STATUSES:
            raise ValueError(f"bad order status {status!r}")
        if intent is not None and intent not in INTENTS:
            raise ValueError(f"bad intent {intent!r}")
        with self.engine.begin() as c:
            self._update_order(c, order_id, status=status, message=message, fill_qty=fill_qty, fill_px=fill_px,
                               fee=fee, qty=qty, intent=intent)

    @staticmethod
    def _update_order(c, order_id: str, *, status: str | None = None, message: str | None = None,
                      fill_qty: float = 0.0, fill_px: float | None = None, fee: float = 0.0,
                      qty: float | None = None, intent: str | None = None) -> None:
        """update_order's change, inside the caller's transaction."""
        row = c.execute(select(orders_t).where(orders_t.c.order_id == order_id)).first()
        if row is None:
            return
        values = {"updated_at": utcnow()}
        if qty is not None:  # resized at the venue (a backtest's resting stop growing with its entry)
            values["qty"] = qty
        if fill_qty:
            fill_qty, fill_px, fee = _exact(fill_qty=fill_qty, fill_px=fill_px, fee=fee)
            filled = to_decimal(row.filled_qty) + fill_qty
            values["avg_px"] = ((to_decimal(row.avg_px or 0)) * to_decimal(row.filled_qty) + fill_qty * fill_px) / filled
            values["filled_qty"] = filled
            values["fee"] = to_decimal(row.fee) + fee
            values["status"] = "filled" if filled >= to_decimal(row.qty) - Decimal("1e-12") else "partially_filled"
        if status is not None and row.status not in FINISHED_ORDER_STATUSES:
            values["status"] = status  # a late "accepted" never reopens a finished order
        if message:
            values["message"] = message
        if intent is not None:
            values["intent"] = intent
        c.execute(update(orders_t).where(orders_t.c.order_id == order_id).values(**values))

    def rebook_liquidation(self, order_id: str, reason: str, signal: dict, ts: datetime | None = None) -> None:
        """GAP-LIQ (Advisor): a resting stop whose fill was at or past the liquidation price is re-booked as the
        liquidation it was. The only intent change the journal allows (Data Architect): stop_loss to liquidation, on
        an order that has filled; its fills stay as they are, and an order_rebooked event keeps the lineage."""
        with self.engine.begin() as c:
            row = c.execute(select(orders_t).where(orders_t.c.order_id == order_id)).first()
            _check_rebook(order_id, row and row._mapping)
            c.execute(update(orders_t).where(orders_t.c.order_id == order_id)
                      .values(intent="liquidation", reason=reason, signal=signal, updated_at=utcnow()))
        self.event(row.sleeve, "info", "order_rebooked", _rebook_words(order_id, reason), ts=ts)
    def merge_order_signal(self, order_id: str, values: dict) -> None:
        """Add to an order's signal what was known only once it filled (an entry's liquidation price). Unknown ids
        are ignored, as in update_order."""
        with self.engine.begin() as c:
            row = c.execute(select(orders_t.c.signal).where(orders_t.c.order_id == order_id).with_for_update()).first()
            if row is not None:
                c.execute(update(orders_t).where(orders_t.c.order_id == order_id)
                          .values(signal={**(row.signal or {}), **values}, updated_at=utcnow()))

    def orders(self, sleeve: str | None = None, statuses: tuple[str, ...] | None = None, limit: int = 500,
               intents: tuple[str, ...] | None = None) -> list[dict]:
        q = select(orders_t)
        q = q.where(orders_t.c.sleeve == sleeve) if sleeve else q.where(_not_backtest(orders_t.c.sleeve))
        if statuses:
            q = q.where(orders_t.c.status.in_(statuses))
        if intents:
            q = q.where(orders_t.c.intent.in_(intents))
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(orders_t.c.ts.desc(), orders_t.c.id.desc()).limit(limit)))

    def last_order(self, sleeve: str, intents: tuple[str, ...]) -> dict | None:
        """A sleeve's newest order with one of these intents, or None."""
        q = (select(orders_t).where(orders_t.c.sleeve == sleeve, orders_t.c.intent.in_(intents))
             .order_by(orders_t.c.ts.desc(), orders_t.c.id.desc()).limit(1))
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def order_counts(self, sleeve: str | None = None) -> dict[str, int]:
        q = select(orders_t.c.status, func.count()).group_by(orders_t.c.status)
        q = q.where(orders_t.c.sleeve == sleeve) if sleeve else q.where(_not_backtest(orders_t.c.sleeve))
        with self.engine.connect() as c:
            return {s: n for s, n in c.execute(q)}

    # --- accounts ------------------------------------------------------------------

    def _ensure_paper_account(self, c) -> None:
        from sleeve_fund.accounts import PAPER_NOTE, PAPER_NOTES_BEFORE

        row = c.execute(select(accounts_t).where(accounts_t.c.name == "paper")).first()
        if row is None:
            c.execute(insert(accounts_t).values(name="paper", kind="paper", venue="", note=PAPER_NOTE,
                                                created_at=utcnow()))
        elif row.venue or row.note in PAPER_NOTES_BEFORE:
            # Made when every account was on one venue: paper trades at each strategy's own venue. A note
            # the PM wrote is kept; only the old default one is replaced.
            note = PAPER_NOTE if row.note in PAPER_NOTES_BEFORE else row.note
            c.execute(update(accounts_t).where(accounts_t.c.name == "paper").values(venue="", note=note))

    def create_account(self, name: str, kind: str, note: str = "", venue: str | None = None) -> None:
        """A live account is on one venue, picked from the registered profiles; a paper one serves any."""
        from sleeve_fund.accounts import KINDS, NAME_RE
        from sleeve_fund.venues import VENUES

        if not NAME_RE.fullmatch(name):
            raise ValueError("account name: lower-case letters, digits and dashes, 2 to 41 characters")
        if kind not in KINDS:
            raise ValueError(f"account kind must be one of {KINDS}")
        if kind == "live" and (venue or "").upper() not in VENUES:
            raise ValueError(f"a live account needs its venue, one of {', '.join(p.label for p in VENUES.values())}")
        venue = (venue or "").lower() if kind == "live" else ""
        with self.engine.begin() as c:
            self._ensure_paper_account(c)
            if c.execute(select(accounts_t.c.name).where(accounts_t.c.name == name)).first():
                raise ValueError(f"an account called {name} already exists")
            c.execute(insert(accounts_t).values(name=name, kind=kind, venue=venue, note=note, created_at=utcnow()))

    def accounts(self) -> list[dict]:
        """Every account with its sleeves and, for live ones, whether the supervisor sees a key."""
        with self.engine.begin() as c:
            self._ensure_paper_account(c)
            rows = _rows(c.execute(select(accounts_t).order_by(accounts_t.c.created_at, accounts_t.c.name)))
            keys = {r["account"]: r for r in _rows(c.execute(select(account_keys_t)))}
            retired = {r["account"]: _aware(r["retired_at"]) for r in _rows(c.execute(select(account_retired_t)))}
            links = _rows(c.execute(select(sleeve_accounts_t)))
            names = [r[0] for r in c.execute(select(sleeves_t.c.name).where(_not_backtest(sleeves_t.c.name)))]
        assigned = {r["sleeve"]: r["account"] for r in links}
        for r in rows:
            r["sleeves"] = [n for n in names if assigned.get(n, "paper") == r["name"]]
            k = keys.get(r["name"])
            r["key_present"] = bool(k["present"]) if k else None  # None: the supervisor hasn't checked yet
            r["key_checked_at"] = k["checked_at"] if k else None
            r["retired_at"] = retired.get(r["name"])
        return rows

    def _account(self, name: str) -> dict:
        row = next((a for a in self.accounts() if a["name"] == name), None)
        if row is None:
            raise ValueError(f"no account called {name}")
        return row

    def set_account_note(self, name: str, note: str) -> None:
        self._account(name)
        with self.engine.begin() as c:
            c.execute(update(accounts_t).where(accounts_t.c.name == name).values(note=note.strip()[:200]))

    def retire_account(self, name: str) -> None:
        """Retire an account no running strategy uses. Its strategies stay assigned (their history
        names it), but none can start on it until it is reinstated or they move."""
        from sleeve_fund.accounts import PAPER

        a = self._account(name)
        if name == PAPER:
            raise ValueError("the shared paper account can't be retired")
        if a["retired_at"]:
            raise ValueError(f"{name} is already retired")
        running = [s.name for s in self.sleeves() if s.name in a["sleeves"] and s.desired_state == "running"]
        if running:
            raise ValueError(f"{', '.join(running)} still run{'s' if len(running) == 1 else ''} on {name}; "
                             "stop or move them first")
        # A stopped strategy can still hold a position, and on a retired account it could never start to
        # sell it (review round 8, M8-2). The journal's position decides, not the desired state.
        held = [s.name for s in self.sleeves() if s.name in a["sleeves"]
                and abs(self.journal_book(s.name, s.starting_balance)["qty"]) > 1e-12]
        if held:
            raise ValueError(f"{', '.join(held)} still hold{'s' if len(held) == 1 else ''} a position on {name}; "
                             "flatten first")
        with self.engine.begin() as c:
            c.execute(insert(account_retired_t).values(account=name, retired_at=utcnow()))

    def reinstate_account(self, name: str) -> None:
        if not self._account(name)["retired_at"]:
            raise ValueError(f"{name} isn't retired")
        with self.engine.begin() as c:
            c.execute(account_retired_t.delete().where(account_retired_t.c.account == name))

    def move_sleeve(self, sleeve: str, account: str, qty: float) -> str:
        """Move a strategy to another account; returns the account it left. Only while it is flat (qty is
        its position now, from the journal), so no position ever changes hands between accounts, whether
        the strategy is running or stopped (review round 8)."""
        self.sleeve(sleeve)
        if is_backtest(sleeve):
            raise ValueError("a saved backtest has no account")
        if abs(qty) > 1e-12:
            raise ValueError("flatten the strategy before moving it: a position can't change accounts")
        target = self._account(account)
        if target["retired_at"]:
            raise ValueError(f"{account} is retired")
        if target["kind"] == "live":  # the shell's live lock: nothing trades real money before G2
            raise ValueError("live accounts are locked until G2 is approved; choose a paper account")
        old = self.account_of(sleeve)
        if old == account:
            raise ValueError(f"{sleeve} is already on {account}")
        self.assign_account(sleeve, account)
        return old

    def account_of(self, sleeve: str) -> str:
        with self.engine.connect() as c:
            row = c.execute(select(sleeve_accounts_t.c.account).where(sleeve_accounts_t.c.sleeve == sleeve)).first()
        return row[0] if row else "paper"

    def assign_account(self, sleeve: str, account: str) -> None:
        with self.engine.begin() as c:
            self._ensure_paper_account(c)
            if c.execute(select(accounts_t.c.name).where(accounts_t.c.name == account)).first() is None:
                raise ValueError(f"no account called {account}")
            c.execute(sleeve_accounts_t.delete().where(sleeve_accounts_t.c.sleeve == sleeve))
            c.execute(insert(sleeve_accounts_t).values(sleeve=sleeve, account=account, assigned_at=utcnow()))

    def archived(self) -> dict[str, datetime]:
        with self.engine.connect() as c:
            return {r.sleeve: _aware(r.archived_at) for r in c.execute(select(sleeve_archive_t))}

    def book_start(self) -> datetime | None:
        """When the current book began: the latest clean slate (supervisor.clear), or None if never cleared."""
        with self.engine.connect() as c:
            row = c.execute(select(decisions_t.c.ts).where(decisions_t.c.action == "clear")
                            .order_by(decisions_t.c.ts.desc()).limit(1)).first()
        return _aware(row[0]) if row else None

    def previous_book(self) -> dict[str, datetime]:
        """Strategies put away by the latest clean slate, or before it, and not brought back since: an
        earlier book's. Their history stays and their pages still open; the book's figures leave them out.
        One still holding a position stays in the current book until it is flat: left out, its position
        counted in no exposure, risk figure or kill switch (5 Oct 2026, a clean slate that archived two
        strategies still long)."""
        runs = self.reset_runs()  # a reset's earlier run is an earlier book's too
        start = self.book_start()
        if start is None:
            return runs
        return {**runs, **{name: at for name, at in self.archived().items()
                           if at <= start and abs(self.journal_book(name, self.sleeve(name).starting_balance)["qty"]) <= 1e-12}}

    def archive(self, sleeve: str) -> None:
        """Put a stopped, flat sleeve away. Raises ValueError if it is running or still holds a position."""
        s = self.sleeve(sleeve)
        if s.desired_state != "stopped":
            raise ValueError("stop the strategy before archiving it")
        with self.engine.begin() as c:
            c.execute(sleeve_archive_t.delete().where(sleeve_archive_t.c.sleeve == sleeve))
            c.execute(insert(sleeve_archive_t).values(sleeve=sleeve, archived_at=utcnow()))

    def unarchive(self, sleeve: str) -> None:
        self.sleeve(sleeve)
        with self.engine.begin() as c:
            c.execute(sleeve_archive_t.delete().where(sleeve_archive_t.c.sleeve == sleeve))

    def report_keys(self, present: dict[str, bool]) -> None:
        """Supervisor only: whether each live account's key is on the server. Never the key itself."""
        now = utcnow()
        with self.engine.begin() as c:
            for name, ok in present.items():
                c.execute(account_keys_t.delete().where(account_keys_t.c.account == name))
                c.execute(insert(account_keys_t).values(account=name, present=int(bool(ok)), checked_at=now))

    def record_fees(self, venue: str, account: str, maker: float, taker: float) -> None:
        """Supervisor only: the fee schedule a live account's venue reports for it."""
        for label, v in (("maker", maker), ("taker", taker)):
            if not 0 <= v < 0.05:
                raise ValueError(f"{label} fee {v} outside [0, 5%)")
        with self.engine.begin() as c:
            c.execute(insert(fee_schedules_t).values(venue=venue.upper(), account=account, maker=float(maker),
                                                     taker=float(taker), fetched_at=utcnow()))

    def record_spread(self, venue: str, instrument: str, half_spread: float, samples: int,
                      ts: datetime | None = None) -> None:
        if not 0 <= half_spread < 0.05:
            raise ValueError(f"half spread {half_spread} outside [0, 5%)")
        with self.engine.begin() as c:
            c.execute(insert(spreads_t).values(venue=venue.upper(), instrument=instrument, half_spread=float(half_spread),
                                               samples=int(samples), measured_at=ts or utcnow()))

    def request_history(self, venue: str, instrument: str, since: datetime) -> bool:
        """Ask the history collector to store `instrument` from `since`. False if it was already asked."""
        with self.engine.begin() as c:
            key = (history_requests_t.c.venue == venue.upper()) & (history_requests_t.c.instrument == instrument)
            if c.execute(select(history_requests_t.c.venue).where(key)).first() is not None:
                return False
            c.execute(insert(history_requests_t).values(venue=venue.upper(), instrument=instrument, since=since,
                                                        requested_at=utcnow()))
        return True

    def history_requests(self, venue: str) -> list[dict]:
        q = (select(history_requests_t).where(history_requests_t.c.venue == venue.upper())
             .order_by(history_requests_t.c.requested_at))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def latest_spread(self, venue: str, instrument: str, at: datetime | None = None) -> dict | None:
        """The latest measurement, or with `at` the one in force then (measured_at, its effective-from, at or before
        it; equal times go to the later row: SPREAD-PIT)."""
        q = (select(spreads_t).where(spreads_t.c.venue == venue.upper(), spreads_t.c.instrument == instrument)
             .order_by(spreads_t.c.measured_at.desc(), spreads_t.c.id.desc()).limit(1))
        if at is not None:
            q = q.where(spreads_t.c.measured_at <= at)
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def spread_series(self, venue: str, instrument: str) -> list[dict]:
        """Every measurement of an instrument's spread, oldest first (ties by id), for SpreadSeries."""
        q = (select(spreads_t).where(spreads_t.c.venue == venue.upper(), spreads_t.c.instrument == instrument)
             .order_by(spreads_t.c.measured_at, spreads_t.c.id))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def latest_fees(self, venue: str, account: str | None = None) -> dict | None:
        """The most recent fetched schedule for a venue (or one account), or None."""
        q = select(fee_schedules_t).where(fee_schedules_t.c.venue == venue.upper())
        if account:
            q = q.where(fee_schedules_t.c.account == account)
        with self.engine.connect() as c:
            rows = _rows(c.execute(q.order_by(fee_schedules_t.c.fetched_at.desc(), fee_schedules_t.c.id.desc()).limit(1)))
        return rows[0] if rows else None

    def set_signal_state(self, sleeve: str, payload: dict, ts: datetime | None = None) -> None:
        """Paper only: the model's conditions on the forming candle, replacing the last ones (Signals tab)."""
        with self.engine.begin() as c:
            c.execute(signal_state_t.delete().where(signal_state_t.c.sleeve == sleeve))
            c.execute(insert(signal_state_t).values(sleeve=sleeve, ts=ts or utcnow(), payload=json.dumps(payload)))

    def signal_state(self, sleeve: str) -> dict | None:
        """The latest conditions the strategy's process wrote, as {"ts": ..., "payload": {...}}, or None."""
        with self.engine.connect() as c:
            row = c.execute(select(signal_state_t.c.ts, signal_state_t.c.payload)
                            .where(signal_state_t.c.sleeve == sleeve)).first()
        return None if row is None else {"ts": _aware(row[0]), "payload": json.loads(row[1])}

    def event(self, sleeve: str | None, level: str, kind: str, message: str, ts: datetime | None = None) -> None:
        if level not in LEVELS:
            raise ValueError(f"bad level {level!r}")
        with self.engine.begin() as c:
            c.execute(insert(events_t).values(sleeve=sleeve, ts=ts or utcnow(), level=level, kind=kind, message=message))

    def equity_series(self, sleeve: str, limit: int = 5000) -> list[dict]:
        q = select(equity_t).where(equity_t.c.sleeve == sleeve).order_by(equity_t.c.ts.desc(), equity_t.c.id.desc())
        with self.engine.connect() as c:
            rows = _rows(c.execute(q.limit(limit)))
        return rows[::-1]

    def equity_since(self, sleeve: str, ts: datetime) -> list[dict]:
        """Every mark at or after ts, oldest first."""
        q = (select(equity_t).where(equity_t.c.sleeve == sleeve, equity_t.c.ts >= ts)
             .order_by(equity_t.c.ts, equity_t.c.id))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def last_equity(self, sleeve: str) -> dict | None:
        rows = self.equity_series(sleeve, limit=1)
        return rows[0] if rows else None

    def first_equity(self, sleeve: str) -> dict | None:
        q = select(equity_t).where(equity_t.c.sleeve == sleeve).order_by(equity_t.c.ts, equity_t.c.id).limit(1)
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def peak_equity(self, sleeve: str, since: datetime | None = None) -> float | None:
        q = select(func.max(equity_t.c.equity)).where(equity_t.c.sleeve == sleeve)
        if since is not None:
            q = q.where(equity_t.c.ts >= since)
        with self.engine.connect() as c:
            v = c.execute(q).scalar()
        return float(v) if v is not None else None

    def max_drawdown(self, sleeve: str, start: float | None = None) -> float:
        """The deepest fall from a running peak over every mark kept, however many: the screens read only
        the latest marks, and a paper strategy marks every few seconds. start, when given, is a peak before
        the first mark (the starting balance), so a first mark that is already a loss counts (M12-F1)."""
        peak = func.max(equity_t.c.equity).over(order_by=(equity_t.c.ts, equity_t.c.id),
                                                rows=(None, 0)).label("peak")
        marks = select(equity_t.c.equity, peak).where(equity_t.c.sleeve == sleeve).subquery()
        q = select(func.max(1 - marks.c.equity / marks.c.peak), func.min(marks.c.equity)).where(marks.c.peak > 0)
        with self.engine.connect() as c:
            worst, low = c.execute(q).one()
        worst = float(worst or 0.0)
        # A running peak seeded with start: each mark's fall is the larger of the two, so the worst is too.
        return max(worst, 1 - float(low) / start) if start and low is not None else worst

    def day_open_equity(self, sleeve: str, day_start: datetime) -> float | None:
        """The equity the day opened at: the last mark at or before `day_start` (00:00 UTC), else the day's
        first mark. A restart reads it so the daily-loss guard keeps the day's real baseline."""
        before = (select(equity_t.c.equity).where(equity_t.c.sleeve == sleeve, equity_t.c.ts <= day_start)
                  .order_by(equity_t.c.ts.desc(), equity_t.c.id.desc()).limit(1))
        first = (select(equity_t.c.equity).where(equity_t.c.sleeve == sleeve, equity_t.c.ts > day_start)
                 .order_by(equity_t.c.ts, equity_t.c.id).limit(1))
        with self.engine.connect() as c:
            v = c.execute(before).scalar()
            if v is None:
                v = c.execute(first).scalar()
        return float(v) if v is not None else None

    def equity_at_or_before(self, sleeve: str, ts: datetime) -> dict | None:
        q = (select(equity_t).where(equity_t.c.sleeve == sleeve, equity_t.c.ts <= ts)
             .order_by(equity_t.c.ts.desc(), equity_t.c.id.desc()).limit(1))
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def fills(self, sleeve: str | None = None, limit: int = 200) -> list[dict]:
        q = select(fills_t)
        q = q.where(fills_t.c.sleeve == sleeve) if sleeve else q.where(_not_backtest(fills_t.c.sleeve))
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(fills_t.c.ts.desc(), fills_t.c.id.desc()).limit(limit)))

    def journal_book(self, sleeve: str, starting_balance: Decimal | float) -> dict:
        """Cash, signed position and average entry implied by the journal: the paper book's source of
        truth (replay_book), with the perp's funding and any insurance-fund cover booked to cash."""
        q = select(fills_t).where(fills_t.c.sleeve == sleeve).order_by(fills_t.c.ts, fills_t.c.id)
        with self.engine.connect() as c:
            fills = _rows(c.execute(q))
        return replay_book(fills, starting_balance, self.funding_total(sleeve), self.insurance_total(sleeve))

    def record_funding(self, sleeve: str, *, qty: float, price: float, rate: float, amount: float,
                       ts: datetime | None = None, kind: str = "settled") -> None:
        qty, price, amount = _exact(qty=qty, price=price, amount=amount)
        with self.engine.begin() as c:
            c.execute(funding_t.insert().values(sleeve=sleeve, ts=ts or utcnow(), qty=qty, price=price, rate=rate,
                                                amount=amount, kind=kind))

    def funding(self, sleeve: str, limit: int = 1000) -> list[dict]:
        q = select(funding_t).where(funding_t.c.sleeve == sleeve).order_by(funding_t.c.ts.desc()).limit(limit)
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def funding_total(self, sleeve: str, before: datetime | None = None) -> Decimal:
        """Funding booked to the strategy's cash, all of it or (before) only what settled before then."""
        q = select(func.sum(funding_t.c.amount)).where(funding_t.c.sleeve == sleeve)
        if before is not None:
            q = q.where(funding_t.c.ts < before)
        with self.engine.connect() as c:
            return to_decimal(c.execute(q).scalar() or 0)

    def record_insurance(self, sleeve: str, *, price: float, amount: float, ts: datetime | None = None) -> None:
        price, amount = _exact(price=price, amount=amount)
        with self.engine.begin() as c:
            c.execute(insurance_t.insert().values(sleeve=sleeve, ts=ts or utcnow(), price=price, amount=amount))

    def insurance(self, sleeve: str, limit: int = 1000) -> list[dict]:
        q = select(insurance_t).where(insurance_t.c.sleeve == sleeve).order_by(insurance_t.c.ts.desc()).limit(limit)
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def insurance_total(self, sleeve: str, before: datetime | None = None) -> Decimal:
        """What the venue's insurance fund covered, all of it or (before) only what it covered before then."""
        q = select(func.sum(insurance_t.c.amount)).where(insurance_t.c.sleeve == sleeve)
        if before is not None:
            q = q.where(insurance_t.c.ts < before)
        with self.engine.connect() as c:
            return to_decimal(c.execute(q).scalar() or 0)

    def fills_after(self, sleeve: str, fill_id: int, limit: int = 500) -> list[dict]:
        """A strategy's fills with ids above fill_id, oldest first."""
        q = (select(fills_t).where(fills_t.c.sleeve == sleeve, fills_t.c.id > fill_id)
             .order_by(fills_t.c.id).limit(limit))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def last_fill_id(self, sleeve: str) -> int:
        with self.engine.connect() as c:
            return int(c.execute(select(func.coalesce(func.max(fills_t.c.id), 0))
                                 .where(fills_t.c.sleeve == sleeve)).scalar() or 0)

    def record_mirror(self, sleeve: str, *, fill_id: int, status: str, instrument: str = "", amount: float = 0.0,
                      price: float | None = None, order_id: str = "", message: str = "",
                      ts: datetime | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(mirror_t.insert().values(sleeve=sleeve, fill_id=fill_id, ts=ts or utcnow(), status=status,
                                               instrument=instrument, amount=amount, price=price,
                                               order_id=order_id, message=message))

    def mirror_watermark(self, sleeve: str) -> int | None:
        """The last fill the mirror has dealt with for a strategy, or None if it has never run for it."""
        with self.engine.connect() as c:
            return c.execute(select(func.max(mirror_t.c.fill_id)).where(mirror_t.c.sleeve == sleeve)).scalar()

    def order_reason(self, order_id: str) -> str:
        with self.engine.connect() as c:
            return c.execute(select(orders_t.c.reason).where(orders_t.c.order_id == order_id)).scalar() or ""

    def request_reset(self, sleeve: str, reason: str, actor: str = "PM", *, book: bool = False) -> None:
        """Ask the supervisor to reset a strategy: it flattens it, puts the run so far away and restarts it at
        its starting capital (supervisor.Supervisor.reset_pending). A per-strategy reset keeps any halt or pause. A
        book reset (`book`) re-bases the high-water mark and the day's start with the book, so it clears the
        strategy's own drawdown halt and daily-loss pause, journaled once each with the old reference and level; never
        the PM's pause, a stop or a liquidation's incident (Independent Quant Advisor, 7 Oct 00:20)."""
        s = self.sleeve(sleeve)
        if is_backtest(sleeve):
            raise ValueError("a saved backtest can't be reset; run a new backtest instead")
        if sleeve in self.archived():
            raise ValueError("an archived strategy can't be reset; restore it first")
        if not reason.strip():
            raise ValueError("a reason is required")
        if self.pending_reset(sleeve):
            raise ValueError("a reset is already under way")
        from sleeve_fund.paper.runtime import liquidation_head  # the engine's liquidation rule

        if liquidation_head(self, sleeve) is not None:
            # Advisor 20:41 (U27): an ordinary reset can't clear a liquidation.
            raise ValueError("its position margin was lost (liquidated), so a reset can't clear it: use Reset after "
                             "liquidation, which asks for an incident note")
        with self.engine.begin() as c:
            rid = c.execute(insert(resets_t).values(sleeve=sleeve, reason=reason.strip(), actor=actor, created_at=utcnow(),
                                                    restart=int(s.desired_state == "running"), run="")).inserted_primary_key[0]
            cleared = book and (s.status == "halted" or (s.status == "paused" and s.paused_until is not None))
            # A pause leaves desired_state running, so restart alone would lift it: keep it (U13-4).
            hold = ({"status": s.status, "status_reason": s.status_reason or "", "paused_until": s.paused_until}
                    if s.status in ("paused", "halted") and not cleared else None)
            if not (hold and hold["status"] == "halted"):  # never downgrade a halt, as the paper process doesn't
                # P1-KR-2: a PM pause or flatten (the kill switch) pressed just before the reset, which the paper
                # process has not applied yet, is kept as the pause it would have set (paper.runtime): the reset
                # puts its pending commands away with the old run. The system's own flattens (a clean slate's) aren't.
                system = {r for (r,) in c.execute(select(decisions_t.c.reason).where(
                    decisions_t.c.sleeve == sleeve, decisions_t.c.actor == "system"))}
                asked = [r for r in c.execute(select(commands_t.c.command, commands_t.c.reason).where(
                    commands_t.c.sleeve == sleeve, commands_t.c.applied_at.is_(None),
                    commands_t.c.command.in_(("pause", "flatten"))).order_by(commands_t.c.id)) if r.reason not in system]
                if asked:
                    words = "flattened by PM" if asked[-1].command == "flatten" else "paused by PM"
                    hold = {"status": "paused", "status_reason": f"{words}: {asked[-1].reason}", "paused_until": None}
            if hold:
                c.execute(insert(reset_holds_t).values(reset_id=rid, **hold))
        if cleared:
            self.event(sleeve, "info", BOOK_RESET, f"Book reset: {_cleared_words(self, s)}")
        self.decide(actor, "reset", f"{'Book reset' if book else 'Reset strategy'}: {reason.strip()}", sleeve)

    def book_reset(self, reason: str, actor: str = "PM", now: datetime | None = None) -> list[str]:
        """The book reset (Independent Quant Advisor, 7 Oct 00:20 (a)): every strategy in the book is re-based at its
        equity now, so each one's own drawdown halt and daily-loss pause is cleared, once, journaled as a "book_reset"
        event with the old reference and the level. It never clears a liquidation (only a reset after liquidation
        does), a retirement, a stop or the PM's pause, and starts nothing: a stopped strategy stays stopped. Refused,
        changing nothing, unless every strategy is flat with no resting orders (BookResetRefused, with a code). One
        transaction: the clears, the new marks, the commands that tell running processes and the decision row land
        together or not at all. Returns the strategies cleared."""
        from sleeve_fund.paper.runtime import clearing_action, liquidation_head  # the engine's own rules

        if not reason.strip():
            raise ValueError("a reason is required")
        now = now or utcnow()
        archived = self.archived()
        book = [s for s in self.sleeves() if not is_backtest(s.name) and s.name not in archived]
        for s in book:
            held = self.journal_book(s.name, s.starting_balance)["qty"]
            if abs(held) > 1e-12:
                raise BookResetRefused("not_flat", f"{s.name} still holds a position ({held:g}), so the book can't be "
                                       "reset: close it (Flatten) first, then reset the book.")
            if (open_ := self.orders(s.name, statuses=OPEN_ORDER_STATUSES, limit=1)):
                raise BookResetRefused("resting_orders", f"{s.name} has a resting order ({open_[0]['intent']}), so the "
                                       "book can't be reset: cancel it (Flatten cancels it) first, then reset the book.")
        cleared = []
        for s in book:
            halt = clearing_action(s, now)
            if halt is None or halt.code not in ("drawdown_halt", "daily_pause"):
                continue
            if liquidation_head(self, s.name) is not None:
                continue  # a liquidation stays until its reset after liquidation (Advisor 17:57)
            mark = self.equity_at_or_before(s.name, now)
            level = float(mark["equity"]) if mark else float(self.journal_book(s.name, s.starting_balance)["cash"])
            if halt.code == "drawdown_halt":
                old = f"the high-water mark of {self.peak_equity(s.name) or s.starting_balance:,.2f}"
            else:
                midnight = datetime.combine(now.date(), datetime.min.time(), tzinfo=timezone.utc)
                day_open = self.day_open_equity(s.name, midnight)
                old = f"the day's opening equity of {day_open if day_open is not None else s.starting_balance:,.2f}"
            words = LABEL_CLEARED[halt.code]
            cleared.append((s, mark, level,
                            f"Book reset: the {words} ({s.status_reason}) is cleared. Its reference, {old}, is "
                            f"re-based at the equity now, {level:,.2f}. {reason.strip()}"))
        with self.engine.begin() as c:
            for s, mark, level, message in cleared:
                c.execute(insert(equity_t).values(sleeve=s.name, ts=now, equity=level, cash=level, qty=0.0,
                                                  price=float(mark["price"]) if mark else 0.0,
                                                  benchmark=float(mark["benchmark"]) if mark else level))
                c.execute(insert(events_t).values(sleeve=s.name, ts=now, level="info", kind=BOOK_RESET,
                                                  message=message))
                stopped = s.desired_state == "stopped"
                c.execute(update(sleeves_t).where(sleeves_t.c.name == s.name).values(
                    status="stopped" if stopped else "running", paused_until=None, updated_at=now,
                    status_reason="book reset: stays stopped until you press Start" if stopped else ""))
                if not stopped:  # its process re-bases its own references too
                    c.execute(insert(commands_t).values(sleeve=s.name, command=BOOK_RESET, reason=reason.strip(),
                                                        created_at=now))
            c.execute(insert(decisions_t).values(
                ts=now, actor=actor, action=BOOK_RESET, sleeve=None,
                reason=f"Book reset: {reason.strip()} (cleared: {', '.join(s.name for s, *_ in cleared) or 'none'})"))
        return [s.name for s, *_ in cleared]

    def pending_reset(self, sleeve: str | None = None) -> dict | None:
        rows = self.pending_resets()
        return next((r for r in rows if sleeve is None or r["sleeve"] == sleeve), None)

    def pending_resets(self) -> list[dict]:
        with self.engine.connect() as c:
            return _rows(c.execute(select(resets_t).where(resets_t.c.done_at.is_(None)).order_by(resets_t.c.id)))

    def reset_runs(self) -> dict[str, datetime]:
        """Runs put away by a reset: {run name: when}. They are an earlier book's, like previous_book's. A refused
        reset (refuse_reset) put nothing away."""
        with self.engine.connect() as c:
            return {r.run: _aware(r.done_at) for r in c.execute(
                select(resets_t).where(resets_t.c.done_at.is_not(None), resets_t.c.run != ""))}

    def refuse_reset(self, request: dict, why: str) -> None:
        """Close a reset the supervisor won't carry out: done, with no run put away, and journaled with why."""
        with self.engine.begin() as c:
            closed = c.execute(update(resets_t).where(resets_t.c.id == request["id"], resets_t.c.done_at.is_(None))
                               .values(done_at=utcnow(), run="")).rowcount
        if closed:  # journaled once, whoever asks twice
            self.decide("system", "reset_refused", f"Not reset: {why}", request["sleeve"])
            self.event(request["sleeve"], "warning", "reset_refused", f"Reset not carried out: {why}")

    def split_run(self, request: dict, now: datetime | None = None, dust_ok: bool = False) -> str:
        """Put a stopped, flat strategy's run so far away under a name of its own and start it afresh: its
        journal (fills, marks, orders, events, decisions, mirror record) moves to the run, archived, and the
        strategy keeps its name and settings with an empty journal, so it replays to its starting capital.
        Nothing is deleted. Returns the run's name. dust_ok: a position too small for any order (is_dust) goes with
        the run."""
        name, now = request["sleeve"], now or utcnow()
        s = self.sleeve(name)
        book = self.journal_book(name, s.starting_balance)
        if abs(book["qty"]) > 1e-12 and not (dust_ok and is_dust(book)):
            raise ValueError("a strategy still holding a position can't be reset; it is flattened first")
        run = f"{name[:46]}--{now:%Y%m%d%H%M%S}"
        moved = (decisions_t, events_t, commands_t, mirror_t, equity_t, exit_plans_t, fills_t, funding_t,
                 insurance_t, orders_t, order_timings_t, mirror_requests_t)
        with self.engine.begin() as c:
            # The reset row is locked before the hold is read, as Store.command locks it: a PM pause or flatten pressed
            # meanwhile is either in the hold read here or waits and stays a pending command of the fresh run.
            locked = c.execute(select(resets_t.c.restart).where(resets_t.c.id == request["id"]).with_for_update()).first()
            if locked is not None:  # a PM Stop since the request (Store.hold_on_reset) keeps the fresh run stopped
                request["restart"] = locked.restart
            row = dict(c.execute(select(sleeves_t).where(sleeves_t.c.name == name)).first()._mapping)
            row.pop("id")
            c.execute(insert(sleeves_t).values(**{**row, "name": run, "desired_state": "stopped", "status": "stopped",
                                                  "status_reason": f"run put away by a reset on {now:%d %b %Y %H:%M}",
                                                  "updated_at": now}))
            for t in moved:
                c.execute(update(t).where(t.c.sleeve == name).values(sleeve=run))
            for t in (sleeve_accounts_t, sleeve_venues_t):
                for r in c.execute(select(t).where(t.c.sleeve == name)).all():
                    c.execute(insert(t).values(**{**dict(r._mapping), "sleeve": run}))
            c.execute(feed_seen_t.delete().where(feed_seen_t.c.sleeve == name))
            # The old run's conditions on the Signals tab until the fresh process writes its own (m13-E2).
            c.execute(signal_state_t.delete().where(signal_state_t.c.sleeve == name))
            c.execute(insert(sleeve_archive_t).values(sleeve=run, archived_at=now))
            hold = c.execute(select(reset_holds_t).where(reset_holds_t.c.reset_id == request["id"])).first()
            # A strategy paused or halted before the reset starts afresh still paused or halted (U13-4): the paper
            # process keeps a status it starts with until the PM resumes it.
            kept = ({"status": hold.status, "status_reason": f"{hold.status_reason} (kept through a reset)".strip(),
                     "paused_until": hold.paused_until} if hold else
                    {"status": "stopped", "status_reason": "reset: starts afresh", "paused_until": None})
            c.execute(update(sleeves_t).where(sleeves_t.c.name == name).values(
                **kept, heartbeat_at=None, created_at=now, updated_at=now))
            c.execute(update(resets_t).where(resets_t.c.id == request["id"]).values(done_at=now, run=run))
        return run

    def request_resync(self, sleeve: str | None, reason: str, actor: str = "PM") -> None:
        """Ask the mirror to bring the demo account in line with the paper book now (sleeve None: all)."""
        if not reason.strip():
            raise ValueError("a reason is required")
        if sleeve is not None and not self.sleeve(sleeve).params.get("demo_mirror"):
            raise ValueError("this strategy isn't copied to a demo account, so there is nothing to resync")
        self.queue_resync(sleeve, reason, actor)
        self.decide(actor, "resync", f"Resync the demo copy: {reason.strip()}", sleeve)

    def queue_resync(self, sleeve: str | None, reason: str, actor: str = "system") -> None:
        with self.engine.begin() as c:
            c.execute(insert(mirror_requests_t).values(sleeve=sleeve, reason=reason.strip(), actor=actor,
                                                       created_at=utcnow(), result=""))

    def pending_resyncs(self) -> list[dict]:
        q = select(mirror_requests_t).where(mirror_requests_t.c.done_at.is_(None)).order_by(mirror_requests_t.c.id)
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def finish_resync(self, request_id: int, result: str) -> None:
        with self.engine.begin() as c:
            c.execute(update(mirror_requests_t).where(mirror_requests_t.c.id == request_id)
                      .values(done_at=utcnow(), result=result))

    def last_resync(self, sleeve: str) -> dict | None:
        """The latest resync request covering this strategy (its own or one for all)."""
        q = (select(mirror_requests_t).where(or_(mirror_requests_t.c.sleeve == sleeve,
                                                  mirror_requests_t.c.sleeve.is_(None)))
             .order_by(mirror_requests_t.c.id.desc()).limit(1))
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def mirror_rows(self, sleeve: str | None = None, limit: int = 200) -> list[dict]:
        q = select(mirror_t)
        if sleeve:
            q = q.where(mirror_t.c.sleeve == sleeve)
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(mirror_t.c.id.desc()).limit(limit)))

    def mirror_positions(self) -> dict[str, float]:
        """The net position the mirror has put on per mirror instrument: what its account should hold."""
        q = (select(mirror_t.c.instrument, func.sum(mirror_t.c.amount)).where(mirror_t.c.status == "filled")
             .group_by(mirror_t.c.instrument))
        with self.engine.connect() as c:
            return {i: float(a or 0.0) for i, a in c.execute(q)}

    def events(self, sleeve: str | None = None, limit: int = 100, min_level: str = "info") -> list[dict]:
        q = select(events_t).where(events_t.c.level.in_(LEVELS[LEVELS.index(min_level):]))
        q = q.where(events_t.c.sleeve == sleeve) if sleeve else q.where(_not_backtest(events_t.c.sleeve))
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(events_t.c.id.desc()).limit(limit)))

    def events_after(self, after_id: int, min_level: str = "warning", limit: int = 500) -> list[dict]:
        """Paper and live events newer than an id, oldest first: what the alert forwarder sends on."""
        q = (select(events_t).where(events_t.c.id > after_id, _not_backtest(events_t.c.sleeve),
                                    events_t.c.level.in_(LEVELS[LEVELS.index(min_level):]))
             .order_by(events_t.c.id).limit(limit))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def last_event_id(self) -> int:
        with self.engine.connect() as c:
            return int(c.execute(select(func.max(events_t.c.id))).scalar() or 0)

    def sleeve_events_since(self, sleeve: str, kinds: tuple[str, ...], after_id: int = 0,
                            since: datetime | None = None) -> list[dict]:
        """One sleeve's events of these kinds newer than an id (and at or after `since`), oldest first."""
        q = (select(events_t).where(events_t.c.sleeve == sleeve, events_t.c.kind.in_(kinds), events_t.c.id > after_id)
             .order_by(events_t.c.id))
        if since is not None:
            q = q.where(events_t.c.ts >= since)
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def set_exit_plan(self, sleeve: str, entry_order: str, *, kind: str, stop_frac: float | None,
                      tp_frac: float | None, basis: str = "", stop_cfg: dict | None = None,
                      risk_amount: float | None = None, planned_r: float | None = None, event_id: int | None = None,
                      ts: datetime | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(insert(exit_plans_t).values(
                sleeve=sleeve, entry_order=entry_order, ts=ts or utcnow(), kind=kind, stop_frac=stop_frac,
                tp_frac=tp_frac, basis=basis, stop_cfg=stop_cfg, risk_amount=risk_amount, planned_r=planned_r,
                event_id=event_id))

    def exit_plan(self, sleeve: str, entry_order: str) -> dict | None:
        """The plan in force for this entry since it was entered, or None if it still has its entry's."""
        q = (select(exit_plans_t).where(exit_plans_t.c.sleeve == sleeve, exit_plans_t.c.entry_order == entry_order)
             .order_by(exit_plans_t.c.id.desc()).limit(1))
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def exit_plans(self, sleeve: str) -> dict[str, dict]:
        """The plan in force for each of a sleeve's entries that has one, by entry order id."""
        q = select(exit_plans_t).where(exit_plans_t.c.sleeve == sleeve).order_by(exit_plans_t.c.id)
        with self.engine.connect() as c:
            return {r["entry_order"]: r for r in _rows(c.execute(q))}

    def last_event(self, sleeve: str, kinds: tuple[str, ...], before: datetime | None = None) -> dict | None:
        """The newest event of these kinds, or the newest at or before `before`."""
        q = select(events_t).where(events_t.c.sleeve == sleeve, events_t.c.kind.in_(kinds))
        if before is not None:
            q = q.where(events_t.c.ts <= before)
        q = q.order_by(events_t.c.id.desc()).limit(1)
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def alerts(self, limit: int = 50, include_acked: bool = False) -> list[dict]:
        """Warnings and errors, newest first, each with its acknowledgement (or None)."""
        q = (select(events_t, acks_t.c.ts.label("acked_at"), acks_t.c.actor.label("acked_by"),
                    acks_t.c.note.label("ack_note"))
             .select_from(events_t.outerjoin(acks_t, acks_t.c.event_id == events_t.c.id))
             .where(events_t.c.level.in_(("warning", "error")), _not_backtest(events_t.c.sleeve)))
        if not include_acked:
            q = q.where(acks_t.c.event_id.is_(None))
        with self.engine.connect() as c:
            rows = _rows(c.execute(q.order_by(events_t.c.id.desc()).limit(limit)))
        for r in rows:
            r["acked_at"] = _aware(r["acked_at"])
        return rows

    def open_alert_count(self) -> int:
        q = (select(func.count()).select_from(events_t.outerjoin(acks_t, acks_t.c.event_id == events_t.c.id))
             .where(events_t.c.level.in_(("warning", "error")), acks_t.c.event_id.is_(None),
                    _not_backtest(events_t.c.sleeve)))
        with self.engine.connect() as c:
            return c.execute(q).scalar() or 0

    def event_by_id(self, event_id: int) -> dict | None:
        with self.engine.connect() as c:
            rows = _rows(c.execute(select(events_t).where(events_t.c.id == event_id)))
        return rows[0] if rows else None

    def write_incident_note(self, incident_id: int, *, author: str, why_stop_did_not_protect: str) -> None:
        """The note on a liquidation's incident (Advisor 18:17): why the half-liquidation stop did not protect the
        position, and who wrote it. Both are required. A later note on the same incident replaces it."""
        why, author = (why_stop_did_not_protect or "").strip(), (author or "").strip()
        if not why:
            raise ValueError(f"the note needs {WHY_STOP_FIELD}")
        if not author:
            raise ValueError("the note needs its author")
        ev = self.event_by_id(incident_id)
        if ev is None or ev["kind"] != "incident" or ev["sleeve"] is None:
            raise ValueError(f"#{incident_id} is not a strategy's incident")
        self.event(ev["sleeve"], "info", INCIDENT_NOTE,
                   f"Incident #{incident_id} note by {author}: {WHY_STOP_FIELD}: {why}")

    def incident_note(self, incident_id: int) -> dict | None:
        """The latest note on an incident (write_incident_note), with its author, else None."""
        ev = self.event_by_id(incident_id)
        if ev is None or ev["sleeve"] is None:
            return None
        head = f"Incident #{incident_id} note by "
        notes = [e for e in self.sleeve_events_since(ev["sleeve"], (INCIDENT_NOTE,), after_id=incident_id)
                 if e["message"].startswith(head)]
        if not notes:
            return None
        note = notes[-1]
        note["author"] = note["message"][len(head):].split(f": {WHY_STOP_FIELD}: ", 1)[0]
        return note

    def incident_is_open(self, incident_id: int) -> bool:
        """An incident closes only once its note is written AND the PM has acknowledged it (Advisor 21:05, RAL 7/8):
        neither a reset after liquidation nor a book reset closes it."""
        with self.engine.connect() as c:
            acked = c.execute(select(acks_t.c.event_id).where(acks_t.c.event_id == incident_id)).first() is not None
        return not (acked and self.incident_note(incident_id) is not None)

    def ack(self, event_id: int, actor: str, note: str = "") -> None:
        with self.engine.begin() as c:
            ev = c.execute(select(events_t.c.level, events_t.c.sleeve).where(events_t.c.id == event_id)).first()
            if ev is None or ev.level == "info" or is_backtest(ev.sleeve):  # a saved backtest raises no alerts
                raise KeyError(f"no alert {event_id}")
            if c.execute(select(acks_t.c.event_id).where(acks_t.c.event_id == event_id)).first() is None:
                c.execute(insert(acks_t).values(event_id=event_id, ts=utcnow(), actor=actor, note=note.strip()))

    def events_of(self, kinds: tuple[str, ...], limit: int = 100) -> list[dict]:
        q = (select(events_t).where(events_t.c.kind.in_(kinds), _not_backtest(events_t.c.sleeve))
             .order_by(events_t.c.id.desc()).limit(limit))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def table_sizes(self) -> dict[str, int]:
        """Rows kept for paper and live; saved backtests' journals are counted once, as runs."""
        out = {}
        with self.engine.connect() as c:
            for t, col in ((sleeves_t, "name"), (equity_t, "sleeve"), (fills_t, "sleeve"), (orders_t, "sleeve"),
                           (events_t, "sleeve"), (commands_t, None), (decisions_t, None)):
                q = select(func.count()).select_from(t)
                if col:
                    q = q.where(_not_backtest(t.c[col]))
                out[t.name] = c.execute(q).scalar() or 0
            out["backtests"] = c.execute(select(func.count()).select_from(backtests_t)).scalar() or 0
        return out

    def database_bytes(self) -> int | None:
        """Size on disk where the engine can tell us; None otherwise."""
        try:
            with self.engine.connect() as c:
                if self.engine.dialect.name == "postgresql":
                    return int(c.exec_driver_sql("select pg_database_size(current_database())").scalar())
                if self.engine.dialect.name == "sqlite":
                    pages = c.exec_driver_sql("pragma page_count").scalar()
                    size = c.exec_driver_sql("pragma page_size").scalar()
                    return int(pages * size)
        except Exception:  # noqa: BLE001 - a missing figure on the ops page is not worth an error
            return None
        return None

    # --- saved backtests ------------------------------------------------------------

    def save_backtest(self, journal, *, run_id: str, key: str, title: str, query: str, result: dict,
                      bar_spec: str | None = None, trial: dict | None = None) -> str:
        """Copy a backtest's in-memory journal (sleeve_fund.paper.journal) into these tables under its
        own name, with the result the backtest page shows. Returns the backtest's strategy name.
        `bar_spec` is the interval the PM picked, where the run's own bars only match its length. `trial` is its
        row in the trials register, written in the same transaction (QA P1-T8)."""
        if trial is not None:
            check_trial(trial)
        name = BACKTEST_PREFIX + run_id
        src = journal.sleeve_row
        now = utcnow()
        with self.engine.begin() as c:
            c.execute(insert(sleeves_t).values(
                name=name, strategy=src.strategy, instrument=src.instrument, bar_spec=bar_spec or src.bar_spec,
                params=src.params, starting_balance=src.starting_balance, risk_profile=src.risk_profile,
                warmup_bars=0, desired_state="stopped", status="stopped", status_reason="backtest",
                paused_until=None, heartbeat_at=None, created_at=now, updated_at=now))
            marks = journal.marks_to_keep()
            if marks:
                c.execute(insert(equity_t), [dict(m, sleeve=name) for m in marks])
            # Client order ids repeat from run to run (and orders.order_id is unique), so each run's are
            # prefixed with its id, on the orders and on the fills that carry them.
            def oid(x: str) -> str:
                return f"{run_id}-{x}"[:64]

            if journal.fills_:
                c.execute(insert(fills_t), [{k: v for k, v in f.items() if k != "id"}
                                            | {"sleeve": name, "order_id": oid(f["order_id"])}
                                            for f in journal.fills_])
            if getattr(journal, "funding_", None):
                c.execute(insert(funding_t), [{k: v for k, v in f.items() if k != "id"} | {"sleeve": name}
                                              for f in journal.funding_])
            if getattr(journal, "insurance_", None):
                c.execute(insert(insurance_t), [{k: v for k, v in f.items() if k != "id"} | {"sleeve": name}
                                                for f in journal.insurance_])
            if journal.orders_:
                c.execute(insert(orders_t), [{k: v for k, v in o.items() if k != "id"}
                                             | {"sleeve": name, "order_id": oid(o["order_id"])}
                                             for o in journal.orders_.values()])
            if journal.events_:
                c.execute(insert(events_t), [{k: v for k, v in e.items() if k != "id"} | {"sleeve": name}
                                             for e in journal.events_])
            # A run whose strategy raised must say so wherever it is opened (review round 8, R8-9). A run
            # without a runtime journals no event of its own, so the result's words stand in for it.
            if result.get("errors") and not any(e.get("kind") in ERROR_KINDS for e in journal.events_):
                c.execute(insert(events_t).values(sleeve=name, ts=now, level="error", kind="handler_failed",
                                                  message=result["errors"]))
            # To the microsecond, so runs saved in the same second still sort (and prune) in order.
            c.execute(insert(backtests_t).values(id=run_id, sleeve=name, key=key, title=title, query=query,
                                                 created_at=datetime.now(timezone.utc), result=json.dumps(result)))
            if trial is not None:
                _put_trial_in(c, trial)
        return name

    def strategy_errors(self, sleeve: str, since_start: bool = False) -> int:
        """How many times the strategy's own code raised, as journaled (handler_failed events); with
        since_start, only since its process last started, so a fixed and restarted strategy reads clean."""
        q = select(func.count()).select_from(events_t).where(events_t.c.sleeve == sleeve,
                                                            events_t.c.kind.in_(ERROR_KINDS))
        start = self.last_event(sleeve, ("process_start",)) if since_start else None
        if start is not None:
            q = q.where(events_t.c.id > start["id"])
        with self.engine.connect() as c:
            return int(c.execute(q).scalar() or 0)

    def backtest(self, run_id: str) -> dict:
        with self.engine.connect() as c:
            rows = _rows(c.execute(select(backtests_t).where(backtests_t.c.id == run_id)))
        if not rows:
            raise KeyError(f"no backtest {run_id!r}")
        return rows[0] | {"result": json.loads(rows[0]["result"])}

    def fresh_backtest(self, key: str, since: datetime) -> dict | None:
        """The latest run of exactly these settings made since `since`, to show again rather than rerun."""
        q = (select(backtests_t.c.id).where(backtests_t.c.key == key, backtests_t.c.created_at >= since)
             .order_by(backtests_t.c.created_at.desc()).limit(1))
        with self.engine.connect() as c:
            row = c.execute(q).first()
        return self.backtest(row.id) if row else None

    def backtests(self, limit: int = 50) -> list[dict]:
        errored = (select(events_t.c.id).where(events_t.c.sleeve == backtests_t.c.sleeve,
                                               events_t.c.kind.in_(ERROR_KINDS)).exists())
        # A run the risk guard halted says so in the list, not only on its result page (round 10, m7).
        halted = (select(sleeves_t.c.name).where(sleeves_t.c.name == backtests_t.c.sleeve,
                                                 sleeves_t.c.status == "halted").exists())
        q = (select(backtests_t.c.id, backtests_t.c.sleeve, backtests_t.c.title, backtests_t.c.query,
                    backtests_t.c.created_at, errored.label("errored"), halted.label("halted"))
             .order_by(backtests_t.c.created_at.desc()).limit(limit))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    # --- trials register (research guardrails) ----------------------------------------

    def add_trials(self, rows: list[dict]) -> int:
        """Append trials; a row whose id is already there is skipped, so replaying the same rows is a no-op.
        Returns how many were added."""
        if not rows:
            return 0
        for r in rows:
            check_trial(r)
        try:
            return self._insert_trials(rows)
        except IntegrityError:
            # Another process added some of the same rows between the check and the insert (two start-ups
            # importing the idea counter at once, QA m3): add the rest one at a time, skipping those it has.
            added = 0
            for r in rows:
                try:
                    added += self._insert_trials([r])
                except IntegrityError:
                    pass
            return added

    def _insert_trials(self, rows: list[dict]) -> int:
        with self.engine.begin() as c:
            return _put_trials(c, rows)

    def trials(self, idea_hash: str | None = None) -> list[dict]:
        q = select(trials_t)
        if idea_hash is not None:
            q = q.where(trials_t.c.idea_hash == idea_hash)
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(trials_t.c.created_at, trials_t.c.id)))

    def add_holdout_lock(self, row: dict) -> bool:
        """Record a holdout opening. False, and nothing written, when this idea already holds a lock on this
        underlying: the first opening is the only one."""
        if row["source"] not in HOLDOUT_SOURCES:
            raise ValueError(f"a holdout lock's source is one of {HOLDOUT_SOURCES}, got {row['source']!r}")
        if row.get("status", "opened") not in HOLDOUT_STATUSES:
            raise ValueError(f"a holdout lock's status is one of {HOLDOUT_STATUSES}, got {row['status']!r}")
        row = {"period_start": None, "period_end": None, "trial_id": None, "status": "opened", **row,
               "underlying": row["underlying"].upper(), "opened_at": row.get("opened_at") or utcnow()}
        try:
            with self.engine.begin() as c:
                c.execute(insert(holdout_locks_t), [row])
        except IntegrityError:
            return False
        return True

    def holdout_locks(self, idea_hash: str | None = None) -> list[dict]:
        q = select(holdout_locks_t)
        if idea_hash is not None:
            q = q.where(holdout_locks_t.c.idea_hash == idea_hash)
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(holdout_locks_t.c.opened_at)))

    def settle_holdout_lock(self, lock_id: str, status: str, trial_id: str | None = None) -> None:
        """Record how a claimed look ended: opened, naming the trial it produced, or crashed. Only a claimed lock
        changes, so a settled one is never rewritten."""
        if status not in ("opened", "crashed"):
            raise ValueError(f"a claimed holdout settles as opened or crashed, got {status!r}")
        with self.engine.begin() as c:
            c.execute(update(holdout_locks_t).where(holdout_locks_t.c.id == lock_id,
                                                    holdout_locks_t.c.status == "claimed")
                      .values(status=status, trial_id=trial_id))

    def prune_backtests(self, keep: int = 50) -> int:
        """Delete all but the latest `keep` saved backtests and their journals. Returns how many went."""
        with self.engine.begin() as c:
            old = [r.sleeve for r in c.execute(select(backtests_t.c.sleeve)
                                               .order_by(backtests_t.c.created_at.desc()).offset(keep))]
            if not old:
                return 0
            # Acknowledgements are refused on backtest events, but one from before that rule would
            # otherwise block deleting its event (a foreign key) and with it every later prune.
            c.execute(delete(acks_t).where(acks_t.c.event_id.in_(select(events_t.c.id)
                                                                  .where(events_t.c.sleeve.in_(old)))))
            for t in (equity_t, fills_t, funding_t, insurance_t, orders_t, events_t, exit_plans_t, sleeve_venues_t):
                c.execute(delete(t).where(t.c.sleeve.in_(old)))
            c.execute(delete(backtests_t).where(backtests_t.c.sleeve.in_(old)))
            c.execute(delete(sleeves_t).where(sleeves_t.c.name.in_(old)))
        return len(old)

    # --- PM commands and decisions ----------------------------------------------

    def command(self, sleeve: str, command: str, reason: str, actor: str = "PM", holds_through_reset: bool = True,
                incident: int | None = None) -> None:
        """holds_through_reset: a pause or flatten asked for while a reset is under way is kept on the fresh run, as
        one in force before the reset is (m13-U5); the reset's own flatten passes False. `incident`: the liquidation
        event a reset after liquidation answers; a second command for the same incident raises IntegrityError
        (commands_incident), so a retry or a double click never resets twice."""
        if command not in COMMANDS:
            raise ValueError(f"bad command {command!r}")
        if not reason.strip():
            raise ValueError("every command needs a reason")
        if is_backtest(sleeve):
            raise ValueError("a saved backtest takes no commands")
        self.sleeve(sleeve)  # raises if unknown
        here = None  # a reset after liquidation applied here, as no process will apply it
        if command == RAL:
            from sleeve_fund.paper.runtime import ral_refusal  # the engine's liquidation rule (liquidation_head)

            if (why := ral_refusal(self, sleeve, incident, actor)) is not None:
                raise ValueError(why)
            reason = f"{reason.strip()} (incident #{incident})"
            here = self._ral_here(sleeve, reason, incident)
        elif incident is not None:
            raise ValueError("only a reset after liquidation names an incident")
        now = utcnow()
        try:
            with self.engine.begin() as c:
                cid = c.execute(insert(commands_t).values(sleeve=sleeve, command=command, reason=reason.strip(),
                                                          created_at=now, incident=incident)).inserted_primary_key[0]
                if holds_through_reset and command in ("pause", "flatten"):
                    self._hold_on_reset(c, sleeve, command, reason)
                if here:
                    self._apply_ral(c, sleeve, cid, here, now)
        except IntegrityError:
            if incident is None:
                raise
            raise ValueError(f"already reset for this liquidation: incident #{incident} has its reset after "
                             "liquidation") from None
        if here:
            self._lapse_resets_before(sleeve, now)
        self.decide(actor, command, reason, sleeve)

    def apply_waiting_ral(self, sleeve: str) -> bool:
        """A reset after liquidation still waiting when its strategy was stopped (QA RAL-F2: Stop after the RAL): no
        process will apply it once the strategy is stopped and flat, so it is applied here as Store.command applies
        one sent to a stopped strategy. True if one was applied; a process that got there first wins."""
        waiting = [c for c in self.pending_commands(sleeve) if c["command"] == RAL]
        if not waiting or (here := self._ral_here(sleeve, waiting[-1]["reason"], waiting[-1]["incident"])) is None:
            return False
        now = utcnow()
        with self.engine.begin() as c:
            if not self._apply_ral(c, sleeve, waiting[-1]["id"], here, now):
                return False
        self._lapse_resets_before(sleeve, now)
        return True

    def _apply_ral(self, c, sleeve: str, command_id: int, here: dict, now: datetime) -> bool:
        """Apply a reset after liquidation in the caller's transaction, as the engine's _reset_after_liquidation does:
        the command applied, the liquidation_reset event, a mark at the remaining equity (which a Start's restored
        high-water mark and day baseline read) and the strategy stopped, no longer halted. False (nothing written)
        when the command was applied already."""
        took = c.execute(update(commands_t).where(commands_t.c.id == command_id, commands_t.c.applied_at.is_(None))
                         .values(applied_at=now)).rowcount
        if not took:
            return False
        c.execute(insert(events_t).values(sleeve=sleeve, ts=now, level="info", kind=LIQUIDATION_RESET,
                                          message=here["words"]))
        if here["mark"] is not None:
            c.execute(insert(equity_t).values(sleeve=sleeve, ts=now, **here["mark"]))
        c.execute(update(sleeves_t).where(sleeves_t.c.name == sleeve).values(
            status="stopped", status_reason="stopped: reset after liquidation", paused_until=None))
        return True

    def _lapse_resets_before(self, sleeve: str, now: datetime) -> None:
        for request in self.pending_resets():  # as the engine does: a reset asked before the RAL lapses
            if request["sleeve"] == sleeve and request["created_at"] <= now:
                self.refuse_reset(request, "lapsed: asked before the liquidation, which the reset after "
                                  "liquidation answered")

    def _ral_here(self, sleeve: str, reason: str, incident: int) -> dict | None:
        """A reset after liquidation for a strategy no process will apply it for (CR on #193): stopped and flat, so
        the supervisor runs nothing, and Start is refused while it is liquidated. Returns what Store.command journals
        for it in the command's own transaction (the engine's words, and a mark at the remaining equity, so a later
        Start restores the new high-water mark and day baseline from it), else None for the process to apply."""
        from sleeve_fund.paper.runtime import SleeveRuntime, liquidation_event, ral_words

        s = self.sleeve(sleeve)
        book = self.journal_book(sleeve, s.starting_balance)
        liq = liquidation_event(self, sleeve)
        if s.desired_state == "running" or abs(book["qty"]) > 1e-12 or liq is None:
            return None  # liq None: nothing left to reset (a book reset ended it), so it isn't applied (QA D-1)
        equity = book["cash"]
        last = self.equity_at_or_before(sleeve, utcnow())
        cmd = {"incident": incident, "reason": reason}
        words = ral_words(self, sleeve, cmd, liq, SleeveRuntime(self, sleeve).peak, equity)
        mark = (None if last is None else
                {"equity": equity, "cash": equity, "qty": 0.0, "price": last["price"], "benchmark": last["benchmark"]})
        return {"words": words, "mark": mark}

    def hold_on_reset(self, sleeve: str, command: str, reason: str) -> bool:
        """A PM control on a strategy whose reset is under way, kept on the fresh run without a command for its
        process (Advisor, 7 Oct, P1-KR-1/3): a Pause or Flatten (the kill switch) whose sale the reset's own flatten
        already makes becomes the reset's hold; a Stop keeps the fresh run stopped. False when no reset is open."""
        with self.engine.begin() as c:
            return self._hold_on_reset(c, sleeve, command, reason)

    def _hold_on_reset(self, c, sleeve: str, command: str, reason: str) -> bool:
        # Locked as split_run locks it, so a press while the reset completes waits for it: it then finds the reset
        # done and acts on the fresh run as on any other, never on a run already put away.
        req = c.execute(select(resets_t.c.id).where(resets_t.c.sleeve == sleeve, resets_t.c.done_at.is_(None))
                        .with_for_update()).first()
        if req is None:
            return False
        if command == "stop":
            c.execute(update(resets_t).where(resets_t.c.id == req.id).values(restart=0))
            return True
        held = c.execute(select(reset_holds_t.c.status).where(reset_holds_t.c.reset_id == req.id)).scalar()
        if held != "halted":  # never downgrade a halt, as the paper process doesn't
            # The reset drops pending commands and starts the run afresh, so record the pause the paper process
            # would have set (paper.runtime) as the hold split_run carries over.
            words = "flattened by PM" if command == "flatten" else "paused by PM"
            c.execute(reset_holds_t.delete().where(reset_holds_t.c.reset_id == req.id))
            c.execute(insert(reset_holds_t).values(reset_id=req.id, status="paused",
                                                   status_reason=f"{words}: {reason.strip()}", paused_until=None))
        return True

    def add_missing_param(self, sleeve: str, key: str, value) -> bool:
        """Set one parameter a strategy has never had, without a restart (True if set). A key it already has,
        whatever its value, is the PM's or the file's and is left alone."""
        s = self.sleeve(sleeve)
        if key in s.params:
            return False
        self._update_sleeve(sleeve, params={**s.params, key: value})
        return True

    def change_settings(self, sleeve: str, *, risk_profile: str, params: dict, warmup_bars: int,
                        trial: dict | None = None) -> bool:
        """Save new risk settings. A running strategy is restarted by the supervisor to trade under
        them (True); a stopped one picks them up when it next starts (False). trial: the re-set strategy's row in
        the trials register, written in the same transaction as the settings (QA P1-T8)."""
        s = self.sleeve(sleeve)
        if is_backtest(sleeve):
            raise ValueError("a saved backtest's settings are what it tested; run a new backtest instead")
        if trial is not None:
            check_trial(trial)
        with self.engine.begin() as c:
            c.execute(update(sleeves_t).where(sleeves_t.c.name == sleeve).values(
                updated_at=utcnow(), risk_profile=risk_profile, params=params, warmup_bars=warmup_bars))
            if trial is not None:
                _put_trial_in(c, trial)
            if s.desired_state != "running":
                return False
            if not c.execute(select(commands_t.c.id).where(commands_t.c.sleeve == sleeve, commands_t.c.command == RELOAD,
                                                           commands_t.c.applied_at.is_(None))).first():
                c.execute(insert(commands_t).values(sleeve=sleeve, command=RELOAD, reason="settings changed",
                                                    created_at=utcnow()))
        return True

    def pending_reload(self, sleeve: str) -> dict | None:
        return next((c for c in self.pending_commands(sleeve) if c["command"] == RELOAD), None)

    def pending_commands(self, sleeve: str) -> list[dict]:
        q = (select(commands_t).where(commands_t.c.sleeve == sleeve, commands_t.c.applied_at.is_(None))
             .order_by(commands_t.c.id))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def drop_pending(self, sleeve: str, why: str) -> int:
        """Retire a strategy's waiting commands unapplied, each noted in the decision log."""
        # A reset after liquidation doesn't lapse: its incident can't be answered twice, so it is applied instead
        # (apply_waiting_ral, or the process still running for its exits) (QA RAL-F2).
        pending = [c for c in self.pending_commands(sleeve) if c["command"] != RAL]
        for cmd in pending:
            self.mark_applied(cmd["id"])
            if cmd["command"] != RELOAD:  # saved settings don't lapse: the next start trades under them
                self.decide("system", f"drop {cmd['command']}", f"{why} ({cmd['reason']})", sleeve)
        return len(pending)

    def mark_applied(self, command_id: int) -> None:
        with self.engine.begin() as c:
            c.execute(update(commands_t).where(commands_t.c.id == command_id).values(applied_at=utcnow()))

    def decide(self, actor: str, action: str, reason: str, sleeve: str | None = None,
               ts: datetime | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(insert(decisions_t).values(ts=ts or utcnow(), actor=actor, action=action, sleeve=sleeve,
                                                 reason=reason.strip()))

    def decisions(self, sleeve: str | None = None, limit: int = 200, action: str | None = None,
                  since: datetime | None = None, until: datetime | None = None) -> list[dict]:
        q = select(decisions_t)
        if sleeve:
            q = q.where(decisions_t.c.sleeve == sleeve)
        if action:
            q = q.where(decisions_t.c.action == action)
        if since:
            q = q.where(decisions_t.c.ts >= since)
        if until:
            q = q.where(decisions_t.c.ts < until)
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(decisions_t.c.id.desc()).limit(limit)))


def _exact(**values) -> list[Decimal]:
    """Each money or quantity figure as the exact Decimal the store writes (DA-9): NaN or infinity raises ValueError,
    anything not a number TypeError, before anything is written."""
    return [to_decimal(v, k) for k, v in values.items()]


def _same_fill(row, side: str, qty, price, fee) -> bool:
    """The booked fill and the offered one are the same fill: side equal, and each figure exactly equal as the
    journal stores it (DA-9: 18 places, half-even)."""
    return row.side == side and all(stored(a) == stored(b)
                                    for a, b in ((row.qty, qty), (row.price, price), (row.fee, fee)))


def _conflict_words(order_id: str, trade_id: str, side: str, qty: float, price: float, fee: float) -> str:
    return (f"A fill for order {order_id} (trade {trade_id}) arrived again with different figures: {side} {qty:g} @ "
            f"{price:,.2f}, fee {fee:,.2f}. The booked fill is kept and nothing moved; check the venue's record before "
            "trusting this strategy's book")
