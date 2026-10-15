"""The one database every part reads from: sleeves, equity marks, fills, events,
PM commands and the decision log.

Postgres in production (the same engine Supabase runs, so pointing DATABASE_URL at
a Supabase project is a config change). SQLite works for tests and quick local
runs. Tables are plain and typed so reports and BI tools can query them directly.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

import json
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DDL,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    Sequence,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    exists,
    false,
    func,
    case,
    inspect,
    insert,
    literal,
    and_,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import DropConstraint

from sleeve_fund import lifecycle
from sleeve_fund.exact import EXACT
from sleeve_fund.money import stored, to_decimal
from sleeve_fund import router
from sleeve_fund.router import venue_for

DEFAULT_URL = "sqlite:///data/sleeve_fund.db"

# The PM's reset after a liquidation (P1-RAL): ends a liquidation halt, from a new high-water mark at the remaining
# equity, once the liquidation's incident has its note (Independent Quant Advisor, 6 Oct 17:57 and 18:17).
RAL = "reset_after_liquidation"
# The book reset (Advisor 7 Oct 00:20 (a)): each strategy's own drawdown halt and daily-loss pause cleared, its
# references re-based at its equity now, journaled as an event of this kind; the command tells a running process.
BOOK_RESET = "book_reset"
COMMANDS = {"pause", "resume", "flatten", RAL, BOOK_RESET}
# The reason the supervisor gives the flatten it sends every holder under a portfolio halt (Supervisor.
# _halt_every_strategy). A command carries no actor, so the words a flatten leaves on a strategy come from this
# (F-HALTREF-2): "flattened by the portfolio halt", not "flattened by PM".
HALT_FLATTEN = "Portfolio halted: "
HALT_WORDS = "flattened by the portfolio halt"


def flattened_words(reason: str) -> str:
    """A strategy's status words for a flatten given with this reason."""
    if reason.startswith(HALT_FLATTEN):
        return f"{HALT_WORDS}: {reason[len(HALT_FLATTEN):]}"
    return f"flattened by PM: {reason}"


# Applied by the supervisor, not the strategy: restart the process so it trades under settings the PM
# changed (Store.change_settings).
RELOAD = "reload"
LIFECYCLE_ACTIONS = ("wind_down", "retire", "close_now")  # the PM's lifecycle moves (Store.set_lifecycle)
LIFECYCLE_EVENT = "lifecycle"  # one event per lifecycle move, with what moved and why
STATUSES = {"starting", "running", "paused", "halted", "stopped", "error"}
LEVELS = ("info", "warning", "error")

metadata = MetaData()
TS = DateTime(timezone=True)
LEG = "~2"  # a two-leg Strategy's second leg row: "<name>~2" (P2-8)
# Drop B (DA, 0028): a multi-asset Strategy's leg rows are "<name>~<ASSET>" (e.g. trend-x~ETH), many under one parent;
# "~2" still fits. The suffix is 1-16 upper-case letters or digits, and a parent holds no "~", so legs are one level
# deep. substr, not LIKE: a parent's own "_" or "%" would be a wildcard. "parent IS NOT NULL" is needed: with a NULL
# parent the second term is NULL, and a CHECK passes on NULL. Spelled per engine (Postgres regex, SQLite GLOB).
_LEG_HEAD = ("(parent IS NULL AND name NOT LIKE '%~%') "
             "OR (parent IS NOT NULL AND parent NOT LIKE '%~%' AND substr(name, 1, length(parent) + 1) = parent || '~' "
             "AND ")
_LEG_SUFFIX = "substr(name, length(parent) + 2)"
LEG_NAME_PG = _LEG_HEAD + f"{_LEG_SUFFIX} ~ '^[A-Z0-9]{{1,16}}$')"
LEG_NAME_SQLITE = _LEG_HEAD + (f"length({_LEG_SUFFIX}) BETWEEN 1 AND 16 AND {_LEG_SUFFIX} NOT GLOB '*[^A-Z0-9]*')")

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
    # P2-8 two legs (DA, 0022): a two-leg Strategy's second leg is its own row, "<name>~2", whose parent is the first.
    # Drop B (0028): a multi-asset Strategy's legs, "<name>~<ASSET>", many under one parent. The CHECK holds the shape
    # on both engines: "~" only as that suffix, one level.
    Column("parent", String(64), ForeignKey("sleeves.name", name="sleeves_parent_fk", ondelete="RESTRICT")),
    CheckConstraint(LEG_NAME_PG, name="sleeves_leg_name").ddl_if(dialect="postgresql"),
    CheckConstraint(LEG_NAME_SQLITE, name="sleeves_leg_name").ddl_if(dialect="sqlite"),
    # P2-6: the PM's lifecycle, kept apart from running/stopped/paused (DA). The row holds the current state only;
    # each move is journaled in events and decisions (Store.set_lifecycle, the one writer).
    Column("lifecycle", String(16), nullable=False, default="active", server_default="active"),
    Column("wind_down_started", TS),
    Column("wind_down_deadline", TS),
    CheckConstraint("lifecycle IN ('active', 'winding_down', 'retired')", name="sleeves_lifecycle"),
    CheckConstraint("lifecycle <> 'winding_down' OR (wind_down_started IS NOT NULL AND wind_down_deadline IS NOT NULL "
                    "AND wind_down_deadline >= wind_down_started)", name="sleeves_winding_down"),
)
# Drop B (DA, 0028): one row per (Strategy, asset). The Strategy is the parent's name on a leg row and the row's own
# name otherwise, so no multi-asset leg trades its parent's instrument or another leg's. A two-leg "~2" row is left out:
# a carry trades one symbol on two markets (spot and perp), which the market in params tells apart, not the row.
STRATEGY_ASSET_ROWS = "parent IS NULL OR name <> parent || '~2'"
Index("sleeves_strategy_asset", func.coalesce(sleeves_t.c.parent, sleeves_t.c.name), sleeves_t.c.instrument,
      unique=True, sqlite_where=text(STRATEGY_ASSET_ROWS), postgresql_where=text(STRATEGY_ASSET_ROWS))

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
    # Bridge A (DA, 0026): the instrument it was in, so a Strategy on several assets reads per asset. Filled from the
    # Strategy's own instrument at the database when the writer names none (INSTRUMENT_FILL triggers below).
    Column("instrument", String(32), nullable=False, server_default=""),
    Index("fills_sleeve_instrument_ts", "sleeve", "instrument", "ts"),
    # DA-2: a fill is booked once. Keyed with its order: the paper venue's trade ids are deterministic per process
    # (Nautilus sandbox), so after a restart a new order's fill can carry an earlier trade id; the client order id
    # is unique (orders.order_id), so (order, trade) is not.
    Index("fills_sleeve_order_trade", "sleeve", "order_id", "trade_id", unique=True),
)

FUNDING_CHARGE_KINDS = ("settled", "baseline")  # the kinds that charge a settlement: one of them per settlement
FUNDING_CHARGES = "kind IN ('settled', 'baseline')"

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
    # CASH-4: when a true-up moved cash, the real rate's arrival (ts stays the settlement's), so an as-of read before
    # it shows the baseline only (Advisor 8 Oct 04:10, ruling 2). Only a true-up has one; every other row moved cash at ts.
    Column("arrived_at", TS),
    # Bridge A (DA, 0026): the instrument it was in, so a Strategy on several assets reads per asset. Filled from the
    # Strategy's own instrument at the database when the writer names none (INSTRUMENT_FILL triggers below).
    Column("instrument", String(32), nullable=False, server_default=""),
    CheckConstraint("kind IN ('settled', 'baseline', 'true_up', 'reversal')", name="funding_kind"),
    CheckConstraint("(kind = 'true_up') = (arrived_at IS NOT NULL)", name="funding_true_up_arrived"),
    Index("funding_sleeve_ts", "sleeve", "ts"),
    # CASH-1: a settlement is charged once (settled or baseline, never both), and reversed or trued up at most once.
    # The key is the settlement time, so a second process or a restart that forgot it is refused by the database.
    Index("funding_once", "sleeve", "ts", "kind", unique=True),
    Index("funding_charged_once", "sleeve", "ts", unique=True, sqlite_where=text(FUNDING_CHARGES),
          postgresql_where=text(FUNDING_CHARGES)),
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
    # CASH-1: the fill that left the strategy flat below zero; one credit for it. Null on rows before 0013.
    Column("order_id", String(64)),
    Column("trade_id", String(64)),
    Index("insurance_sleeve_ts", "sleeve", "ts"),
    # Bridge A (DA, 0026): the instrument it was in, so a Strategy on several assets reads per asset. Filled from the
    # Strategy's own instrument at the database when the writer names none (INSTRUMENT_FILL triggers below).
    Column("instrument", String(32), nullable=False, server_default=""),
    Index("insurance_once", "sleeve", "order_id", "trade_id", unique=True, sqlite_where=text("order_id IS NOT NULL"),
          postgresql_where=text("order_id IS NOT NULL")),
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
    # The command a liquidation_reset carries out (F217-3): one reset per command, so a retry after a crash reads
    # whether its own reset was journaled from this key, never from the message's wording. NULL on every other row.
    Column("command_id", Integer, ForeignKey("commands.id", ondelete="RESTRICT", use_alter=True,
                                             name="events_command_id_fkey")),
    Index("events_ts", "ts"),
    Index("events_command_once", "command_id", unique=True, postgresql_where=text("command_id IS NOT NULL"),
          sqlite_where=text("command_id IS NOT NULL")),
)


def _unless_dropping_an_absent_fk(ddl, target, bind, **_kw) -> bool:
    """drop_all drops a use_alter FK by name before its table, and Postgres refuses a name that isn't there: a
    database built to a revision before 0014 has events without it. Skip that one drop then; adding it always runs."""
    if not isinstance(ddl, DropConstraint):
        return True
    return any(fk["name"] == target.name for fk in inspect(bind).get_foreign_keys(target.table.name))


next(c for c in events_t.foreign_key_constraints if c.name == "events_command_id_fkey").ddl_if(
    callable_=_unless_dropping_an_absent_fk)

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
    Column("venue", String(16), nullable=False),  # where it was sent: the strategy's one venue (router, P2-4)
    # P2-8 (DA, 0022): one id for every order of a two-leg trade, across both leg rows; none for a single leg.
    Column("parent_trade_id", String(64)),
    Index("orders_sleeve_ts", "sleeve", "ts"),
    # Bridge A (DA, 0026): the instrument it was in, so a Strategy on several assets reads per asset. Filled from the
    # Strategy's own instrument at the database when the writer names none (INSTRUMENT_FILL triggers below).
    Column("instrument", String(32), nullable=False, server_default=""),
    # Bridge A (DA, 0026): what the order risks at entry, in money: qty x distance to its stop with fees, or the charged
    # risk of a stopless one. Written by sizing; none on orders before it. R per trade = its P&L / this.
    Column("entry_risk", EXACT),
    Index("orders_parent_trade_id", "parent_trade_id"),
    CheckConstraint("venue = upper(venue)", name="orders_venue_upper"),
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
# F-ASSIGN-STALE (Advisor ADV-ASSIGN, 14 Oct 17:41 UK): a create or an account move asked for on a marked book waits
# here for the supervisor's poll, which lands it after its own mark with its flow, so P&L made since the last mark is
# judged at the book's scale first. At most one waiting row per Strategy (landed_at and superseded_at both NULL); a
# later request supersedes it. Rows are never deleted: each ends once, landed or superseded (DA, migration 0024).
book_moves_t = Table(
    "book_moves",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("account", String(41), ForeignKey("accounts.name"), nullable=False),  # where it lands
    Column("kind", String(8), nullable=False),  # create | move
    Column("requested_at", TS, nullable=False),
    Column("landed_at", TS),
    Column("superseded_at", TS),
    CheckConstraint("kind IN ('create', 'move')", name="book_moves_kind"),
    CheckConstraint("landed_at IS NULL OR superseded_at IS NULL", name="book_moves_one_end"),
    CheckConstraint("(landed_at IS NULL OR landed_at >= requested_at) "
                    "AND (superseded_at IS NULL OR superseded_at >= requested_at)", name="book_moves_order"),
    Index("ux_book_moves_waiting", "sleeve", unique=True,
          postgresql_where=text("landed_at IS NULL AND superseded_at IS NULL"),
          sqlite_where=text("landed_at IS NULL AND superseded_at IS NULL")),
)
# The venue a strategy trades on, where it isn't the default (sleeve_fund.venues). A table of its own, so
# it arrives as CREATE TABLE: a strategy with no row here trades on the default venue, as every one did.
sleeve_venues_t = Table(
    "sleeve_venues",
    metadata,
    Column("sleeve", String(64), ForeignKey("sleeves.name"), primary_key=True),
    Column("venue", String(16), nullable=False),
)
# F-WED1-3 (Advisor, HoE 14 Oct): the pin is insert-only, held by the database itself. A venue is never changed; a pin is
# deleted only with its saved backtest (prune_backtests); and a strategy that has an order or a fill can't be given one,
# so nothing moves it off the venue its position and working orders are on. A new venue is a new strategy. Migration
# 0020 writes the same triggers (spelled out there). They read orders, fills and backtests by name when they fire, so
# it doesn't matter which create_all makes first.
PIN_REFUSED = "a strategy''s venue is pinned: "  # quoted for SQL
SLEEVE_VENUES_PG = (
    "CREATE OR REPLACE FUNCTION sleeve_venue_pinned() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "IF TG_OP = 'UPDATE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    f"MESSAGE = '{PIN_REFUSED}' || OLD.sleeve || ' cannot change venue'; END IF; "
    "IF TG_OP = 'DELETE' THEN "
    "IF substr(OLD.sleeve, 1, 3) <> 'bt:' OR NOT EXISTS (SELECT 1 FROM backtests WHERE sleeve = OLD.sleeve) THEN RAISE EXCEPTION USING "
    f"ERRCODE = 'check_violation', MESSAGE = '{PIN_REFUSED}' || OLD.sleeve || "
    "' is not a saved backtest, so its pin stays'; END IF; RETURN OLD; END IF; "
    "IF EXISTS (SELECT 1 FROM orders WHERE sleeve = NEW.sleeve) OR EXISTS (SELECT 1 FROM fills WHERE sleeve = NEW.sleeve) "
    "THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    f"MESSAGE = '{PIN_REFUSED}' || NEW.sleeve || ' has traded, so it keeps its venue'; "
    "END IF; RETURN NEW; END $$",
    "CREATE TRIGGER sleeve_venue_pinned BEFORE INSERT OR UPDATE OR DELETE ON sleeve_venues FOR EACH ROW "
    "EXECUTE FUNCTION sleeve_venue_pinned()",
)
SLEEVE_VENUES_SQLITE = (
    "CREATE TRIGGER sleeve_venues_no_edit BEFORE UPDATE ON sleeve_venues BEGIN "
    f"SELECT RAISE(ABORT, '{PIN_REFUSED}a venue cannot change'); END",
    "CREATE TRIGGER sleeve_venues_no_delete BEFORE DELETE ON sleeve_venues "
    "WHEN substr(OLD.sleeve, 1, 3) <> 'bt:' OR NOT EXISTS (SELECT 1 FROM backtests WHERE sleeve = OLD.sleeve) BEGIN "
    f"SELECT RAISE(ABORT, '{PIN_REFUSED}only a saved backtest''s pin is deleted'); END",
    "CREATE TRIGGER sleeve_venues_no_late_pin BEFORE INSERT ON sleeve_venues "
    "WHEN EXISTS (SELECT 1 FROM orders WHERE sleeve = NEW.sleeve) OR EXISTS (SELECT 1 FROM fills WHERE sleeve = NEW.sleeve) "
    f"BEGIN SELECT RAISE(ABORT, '{PIN_REFUSED}a strategy that has traded keeps its venue'); END",
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
    # F-C3B-A (0023): the equity it took out of the fund's book when it was put away, with its flow; NULL for an archive
    # made before that (still counted until supervisor withdraw-archived), for a reset's run (its cash stays) and for one
    # archived while not flat, until the supervisor withdraws it once flat. NULL always means it still counts in the book.
    Column("withdrawn", EXACT),
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
    # What was evaluated (TRIAL_KINDS); source stays where the row came from (migration 0012, DA 8 Oct 2026).
    Column("kind", String(16), nullable=False, server_default="single"),
    CheckConstraint("status IN ('ok', 'failed')", name="trials_status"),
    CheckConstraint("kind IN ('single', 'ablation', 'portfolio_run')", name="trials_kind"),
    Index("trials_idea_hash", "idea_hash"),
    Index("trials_definition_dataset", "definition_hash", "dataset"),
)
# TRIALS-WORM (QD #238, HoE): the register is write-once. A row is never deleted, and an UPDATE may change only status
# and error (a failed count re-counted); anything else, created_at, definition_hash, settings or the data range above
# all, is refused by the database itself, so N, the earliest-row rule and the holdout check can't be edited after the
# fact. Migration 0018 writes the same triggers (spelled out there: a migration is frozen at what it ran).
TRIALS_FROZEN = ("id", "definition_hash", "idea_hash", "code_version", "definition_name", "family", "settings",
                 "dataset", "stage", "source", "sharpe", "trades", "oos_trades", "backtest_id", "created_at",
                 "data_start", "data_end", "kind")
TRIALS_WORM_PG = (
    "CREATE OR REPLACE FUNCTION trials_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "IF TG_OP = 'DELETE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    "MESSAGE = 'trials are write-once: ' || OLD.id || ' cannot be deleted'; END IF; "
    f"IF ({', '.join('NEW.' + c for c in TRIALS_FROZEN)}) IS DISTINCT FROM ({', '.join('OLD.' + c for c in TRIALS_FROZEN)}) "
    "THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    "MESSAGE = 'trials are write-once: only status and error of ' || OLD.id || ' may change'; END IF; "
    "RETURN NEW; END $$",
    "CREATE TRIGGER trials_write_once BEFORE UPDATE OR DELETE ON trials FOR EACH ROW EXECUTE FUNCTION trials_write_once()",
)
TRIALS_WORM_SQLITE = (
    "CREATE TRIGGER trials_no_delete BEFORE DELETE ON trials BEGIN "
    "SELECT RAISE(ABORT, 'trials are write-once: a row cannot be deleted'); END",
    "CREATE TRIGGER trials_no_edit BEFORE UPDATE ON trials WHEN "
    + " OR ".join(f"NEW.{c} IS NOT OLD.{c}" for c in TRIALS_FROZEN)
    + " BEGIN SELECT RAISE(ABORT, 'trials are write-once: only status and error may change'); END",
)
for _sql in TRIALS_WORM_PG:
    event.listen(trials_t, "after_create", DDL(_sql).execute_if(dialect="postgresql"))
for _sql in TRIALS_WORM_SQLITE:
    event.listen(trials_t, "after_create", DDL(_sql).execute_if(dialect="sqlite"))
for _sql in SLEEVE_VENUES_PG:
    event.listen(sleeve_venues_t, "after_create", DDL(_sql).execute_if(dialect="postgresql"))
for _sql in SLEEVE_VENUES_SQLITE:
    event.listen(sleeve_venues_t, "after_create", DDL(_sql).execute_if(dialect="sqlite"))

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

# The portfolio limits gate's tables (v2 P2-2; Data Architect's shapes, v2/p2-2-tables.md). Money and quantities are
# EXACT; limits and ratios keep their own scales. Paper only: a backtest keeps its decisions in its result.
LIMIT = Numeric(10, 4)
LIMIT_PLACES = Decimal("0.0001")
# A BIGINT key that still autoincrements on SQLite (which only does so for INTEGER PRIMARY KEY).
BIG_ID = BigInteger().with_variant(Integer, "sqlite")
# The PM's limits, one row per version, never updated: a change is a new version, the current one the highest.
portfolio_profile_t = Table(
    "portfolio_profile",
    metadata,
    Column("version", Integer, primary_key=True, autoincrement=False),
    Column("created_at", TS, nullable=False),
    Column("created_by", Text, nullable=False),  # "pm", or the migration's seed
    Column("gross_max", LIMIT, nullable=False),  # x book
    Column("net_underlying_max", LIMIT, nullable=False),  # x book, per underlying across venues and contract types
    Column("margin_max", LIMIT, nullable=False),  # share of book
    Column("open_risk_max", LIMIT, nullable=False),  # share of book
    Column("drawdown_halt", LIMIT, nullable=False),
    Column("daily_pause", LIMIT, nullable=False),
    Column("note", Text),
    CheckConstraint("gross_max > 0 AND net_underlying_max > 0 AND margin_max > 0 AND open_risk_max > 0 "
                    "AND daily_pause > 0 AND drawdown_halt > daily_pause", name="portfolio_profile_limits"),
)
# Version 1, the PM's accepted limits (6 Oct 2026, risk.PORTFOLIO), as migration 0010 seeds it: a store made by
# create_all holds it too, so no database ever has a gate without accepted limits.
PORTFOLIO_PROFILE_V1 = {
    "version": 1, "created_at": datetime(2026, 10, 6, 14, 17, tzinfo=timezone.utc), "created_by": "migration 0010",
    "gross_max": Decimal("1.5"), "net_underlying_max": Decimal("0.5"), "margin_max": Decimal("0.5"),
    "open_risk_max": Decimal("0.05"), "drawdown_halt": Decimal("0.15"), "daily_pause": Decimal("0.03"),
    "note": "the PM's accepted portfolio limits (6 Oct 2026)",
}
event.listen(portfolio_profile_t, "after_create",
             lambda target, conn, **kw: conn.execute(insert(target), [PORTFOLIO_PROFILE_V1]))
# The supervisor's one row: what the gate's entry block reads. Only the supervisor writes it, but for the PM's Resume.
PORTFOLIO_STATE_FIGURES = ("reference_equity", "hwm", "day_start_equity", "day_start", "book_equity")
# MUST-F286-H4 (DA, 14 Oct): the carried ratios, kept only across an unmarked book; the HWM's anchors the set.
PORTFOLIO_STATE_CARRIED_CHECKS = (
    ("unmarked", "marked_at IS NULL OR (carried_hwm_drawdown IS NULL AND carried_reference_drawdown IS NULL "
                 "AND carried_day_return IS NULL AND carried_day IS NULL)"),
    ("anchor", "carried_hwm_drawdown IS NOT NULL OR (carried_reference_drawdown IS NULL AND carried_day_return IS NULL)"),
    ("day", "(carried_day_return IS NULL) = (carried_day IS NULL)"),
    ("hwm_range", "carried_hwm_drawdown IS NULL OR (carried_hwm_drawdown >= 0 AND carried_hwm_drawdown < 1)"),
    ("reference_range", "carried_reference_drawdown IS NULL OR "
                        "(carried_reference_drawdown >= 0 AND carried_reference_drawdown < 1)"),
    ("day_range", "carried_day_return IS NULL OR carried_day_return > -1"),
)
portfolio_state_t = Table(
    "portfolio_state",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("status", String(16), nullable=False),
    Column("paused_until", TS),  # with 'paused': the next 00:00 UTC
    Column("halt_reason", Text),  # with 'halted'
    Column("pause_reason", Text),  # with paused_until (0011)
    # The book's figures, null only before the first mark (0011): all set or all null.
    Column("reference_equity", EXACT),  # the halt reference: re-based at the PM's Resume
    Column("hwm", EXACT),  # the true high-water mark; reset only by a book reset
    Column("day_start_equity", EXACT),
    Column("day_start", Date),  # the UTC day it belongs to
    Column("book_equity", EXACT),  # the last mark: the whole fund, unallocated cash included
    Column("marked_at", TS),  # older than 60 s: entries blocked as portfolio_state_stale
    Column("stale_told_at", TS),  # the mark a stale-book alert was raised for: once per spell, across processes (0011)
    # MUST-F286-H4 (0023): what a withdrawal that emptied the book carries to the next capital's first mark, taken
    # from the book before it; set only while unmarked, restored and cleared by that mark (portfolio.gate.mark_book).
    Column("carried_hwm_drawdown", EXACT),  # 1 - book / hwm
    Column("carried_reference_drawdown", EXACT),  # 1 - book / reference_equity; NULL when the reference was the HWM
    Column("carried_day_return", EXACT),  # book / day_start_equity - 1, for carried_day
    Column("carried_day", Date),  # the UTC day carried_day_return belongs to
    Column("profile_version", Integer, ForeignKey("portfolio_profile.version"), nullable=False),
    # Bridge A (DA, 0027): the compliance tests' version this row was checked under. Nullable: rows before it have the
    # profile only.
    Column("compliance_version", Integer, ForeignKey("compliance_versions.version")),
    Column("updated_at", TS, nullable=False),
    CheckConstraint("id = 1", name="portfolio_state_one_row"),
    CheckConstraint("status IN ('ok', 'paused', 'halted')", name="portfolio_state_status"),
    CheckConstraint("status <> 'paused' OR paused_until IS NOT NULL", name="portfolio_state_paused_until"),
    CheckConstraint("status <> 'halted' OR halt_reason IS NOT NULL", name="portfolio_state_halt_reason"),
    CheckConstraint("status <> 'paused' OR pause_reason IS NOT NULL", name="portfolio_state_pause_reason"),
    # a halt is never refused, even before the first mark (refusing it would fail open); a pause follows a mark
    # (DA 0012, CR F219-3)
    CheckConstraint("marked_at IS NOT NULL OR status <> 'paused'", name="portfolio_state_marked"),
    # all of the book's figures set, or none (before the first mark): one CHECK per figure (DA, 0011)
    *(CheckConstraint(f"({c} IS NULL) = (marked_at IS NULL)", name=f"portfolio_state_{c}_marked")
      for c in PORTFOLIO_STATE_FIGURES),
    *(CheckConstraint(sql, name=f"portfolio_state_carried_{name}") for name, sql in PORTFOLIO_STATE_CARRIED_CHECKS),
)
# The book's history: one mark a minute and one at every status change; a cache rebuilt from fills, which stay the
# source of truth. The 5 s mark lives in portfolio_state only.
book_marks_t = Table(
    "book_marks",
    metadata,
    Column("ts", TS, primary_key=True),
    Column("equity", EXACT, nullable=False),
    Column("gross", EXACT, nullable=False),  # sum of abs notional
    Column("net_max", EXACT, nullable=False),  # the largest abs net notional of any underlying
    Column("net_underlying", String(16)),  # which underlying that was
    Column("margin_used", EXACT, nullable=False),  # posted isolated margin; spot counts its full notional
    Column("open_risk", EXACT, nullable=False),
    Column("hwm", EXACT, nullable=False),
    Column("reference_equity", EXACT, nullable=False),
    Column("day_start_equity", EXACT, nullable=False),
    Column("status", String(16), nullable=False),
    Column("profile_version", Integer, ForeignKey("portfolio_profile.version"), nullable=False),
    # Bridge A (DA, 0027): the compliance tests' version this row was checked under. Nullable: rows before it have the
    # profile only.
    Column("compliance_version", Integer, ForeignKey("compliance_versions.version")),
)
# One row per check of an order that raises a position, at submit or as the backstop at fill. seq is the first-come
# order, taken inside the gate's lock (Postgres: the gate_decision_seq sequence; SQLite: max + 1 under the lock).
GATE_DECISION_SEQ = Sequence("gate_decision_seq", metadata=metadata)
GATE_OUTCOMES = ("approved", "trimmed", "rejected", "late", "portfolio_state_stale", "error")
GATE_LIMITS = ("gross", "net_instrument", "margin", "open_risk", "halt", "pause", "portfolio_state_stale",
               "lock_error", "below_min")
GATE_STAGES = ("submit", "fill")
GATE_RELEASES = ("fill", "reject", "cancel", "ttl")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


PAIR_STATES = ("flat", "legging", "hedged", "closing")
# The moves a pair makes (DA review 14 Oct 06:09 UK), from None (no row yet). legging -> flat: the first leg was never
# placed (Pair._send_first), or a restart found both legs flat (Pair.recover); hedged -> flat: that restart case too.
PAIR_MOVES = {None: ("flat",), "flat": ("legging",), "legging": ("hedged", "closing", "flat"),
              "hedged": ("closing", "flat"), "closing": ("flat",)}
PAIR_FIGURES = ("reference_equity", "hwm", "day_start_equity", "day_start")
PAIR_SHAPE = ("(state = 'flat' AND trade_id IS NULL AND deadline IS NULL) "
              "OR (state = 'hedged' AND trade_id IS NOT NULL AND deadline IS NULL) "
              "OR (state IN ('legging', 'closing') AND trade_id IS NOT NULL AND deadline IS NOT NULL)")
# P2-8 (Advisor M3, DA 14 Oct 05:02 UK): a two-leg Strategy's coordinator, one row per parent (the first leg's row; a
# leg never has one). Its state is a hint after a restart: the journal's fills by parent_trade_id are the truth. Its
# figures are the summed guard's (ruling E), split as portfolio_state's are: the halt reference re-based at a Resume, and
# the true high-water mark, only ever raised (reset by a book reset), with the UTC day's start.
pair_state_t = Table(
    "pair_state",
    metadata,
    Column("parent", String(64), ForeignKey("sleeves.name", name="pair_state_parent_fk", ondelete="RESTRICT"),
           primary_key=True),
    Column("state", String(8), nullable=False),
    Column("trade_id", String(64)),  # the trade's parent_trade_id, while one is open
    Column("deadline", TS),  # legging or closing: when the legged state is unwound by
    Column("reference_equity", EXACT),
    Column("hwm", EXACT),
    Column("day_start_equity", EXACT),
    Column("day_start", Date),
    Column("marked_at", TS),
    Column("updated_at", TS, nullable=False),
    CheckConstraint(_in("state", PAIR_STATES), name="pair_state_values"),
    CheckConstraint(PAIR_SHAPE, name="pair_state_shape"),
    *(CheckConstraint(f"({c} IS NULL) = (marked_at IS NULL)", name=f"pair_state_{c}_marked") for c in PAIR_FIGURES),
)

gate_decisions_t = Table(
    "gate_decisions",
    metadata,
    Column("id", BIG_ID, primary_key=True),
    Column("seq", BigInteger, nullable=False, unique=True),
    Column("sleeve_id", Integer, ForeignKey("sleeves.id", ondelete="RESTRICT"), nullable=False),  # outlives a strategy
    Column("bar_ts", TS, nullable=False),  # the decision's bar
    Column("intent_id", Text, nullable=False),  # the strategy's id for the intent
    Column("intent", String(16), nullable=False),  # entry, add, band add, rebalance increase, reversal open leg
    Column("underlying", String(16), nullable=False),  # what net is counted by, e.g. BTC
    Column("order_id", String(64), ForeignKey("orders.order_id")),  # set with the order row when sent
    Column("profile_version", Integer, ForeignKey("portfolio_profile.version"), nullable=False),
    # Bridge A (DA, 0027): the compliance tests' version this row was checked under. Nullable: rows before it have the
    # profile only.
    Column("compliance_version", Integer, ForeignKey("compliance_versions.version")),
    # the test that refused or trimmed it, in that version (a trigger checks it is one); limit_hit keeps the outcomes
    # that are not a test (lock_error, portfolio_state_stale, below_min)
    Column("test_hit", String(32)),
    Column("outcome", String(24), nullable=False),
    Column("limit_hit", String(24)),
    Column("requested_qty", EXACT, nullable=False),
    Column("approved_qty", EXACT, nullable=False),  # 0 when rejected; rounded down to the step
    Column("price", EXACT, nullable=False),  # the expected fill price used
    # the book before this entry, reservations included
    Column("book_equity", EXACT),
    Column("gross", EXACT),
    Column("net_underlying", EXACT),
    Column("margin_used", EXACT),
    Column("open_risk", EXACT),
    Column("regime_weight", Numeric(10, 6)),
    Column("regime_state", Text),
    Column("stage", String(8), nullable=False),
    Column("decided_at", TS, nullable=False),
    # Drop B (DA, 0028), written by the gate inside its lock; nullable, so rows before it stay as they are.
    Column("planned_stop", EXACT),  # the idea's stop and target prices: a refusal's shadow outcome replays them
    Column("planned_target", EXACT),
    # the refused or trimmed test's measure had the idea gone through at the requested size, as a share of
    # book_equity (T1-T3 loss if stops are hit, T4 gross or net); NULL for T5 and when no test bound it
    Column("would_be", EXACT),
    Column("side", SmallInteger),  # +1 long, -1 short; requested_qty stays unsigned
    CheckConstraint(_in("outcome", GATE_OUTCOMES), name="gate_decisions_outcome"),
    CheckConstraint(_in("limit_hit", GATE_LIMITS), name="gate_decisions_limit_hit"),
    CheckConstraint(_in("stage", GATE_STAGES), name="gate_decisions_stage"),
    CheckConstraint("approved_qty >= 0 AND approved_qty <= requested_qty", name="gate_decisions_approved_qty"),
    CheckConstraint("outcome <> 'rejected' OR approved_qty = 0", name="gate_decisions_rejected_none"),
    # One real decision per strategy, bar, intent and stage; an 'error' row is audit and never blocks the retry (0011).
    Index("gate_decisions_sleeve_bar_intent_stage", "sleeve_id", "bar_ts", "intent_id", "stage", unique=True,
          sqlite_where=text("outcome <> 'error'"), postgresql_where=text("outcome <> 'error'")),
    Index("gate_decisions_sleeve_decided_at", "sleeve_id", "decided_at"),
)
# Headroom held by resting entries: one per approved or trimmed decision, released in the same transaction as the
# fill, reject or cancel row. Never released blind while its order may still fill (Advisor 20:25 UK).
gate_reservations_t = Table(
    "gate_reservations",
    metadata,
    Column("decision_id", BIG_ID, ForeignKey("gate_decisions.id"), primary_key=True, autoincrement=False),
    Column("sleeve_id", Integer, ForeignKey("sleeves.id", ondelete="RESTRICT"), nullable=False),  # outlives a strategy
    Column("underlying", String(16), nullable=False),
    Column("remaining_qty", EXACT, nullable=False),  # unsigned, >= 0, side on the decision; reduced by partial fills
    Column("notional", EXACT, nullable=False),  # what it counts against each limit
    Column("margin", EXACT, nullable=False),
    Column("open_risk", EXACT, nullable=False),
    Column("created_at", TS, nullable=False),
    Column("cancel_sent_at", TS),  # the sweep's cancel of an order that outlived it; re-sent with an alert at 60 s
    Column("expires_at", TS, nullable=False),
    Column("released_at", TS),  # null while active
    Column("release_reason", String(8)),
    CheckConstraint(_in("release_reason", GATE_RELEASES), name="gate_reservations_release_reason"),
    CheckConstraint("(released_at IS NULL) = (release_reason IS NULL)", name="gate_reservations_released"),
    CheckConstraint("remaining_qty >= 0", name="gate_reservations_remaining_qty"),
    Index("gate_reservations_active", "underlying", sqlite_where=text("released_at IS NULL"),
          postgresql_where=text("released_at IS NULL")),
    Index("gate_reservations_active_sleeve", "sleeve_id", sqlite_where=text("released_at IS NULL"),
          postgresql_where=text("released_at IS NULL")),  # the sweep, and DA-6's reset guard
)
# Drop B (DA, 0028; Advisor MA-SPEC): what a trimmed or refused idea would have made, filled in after the trade it was
# part of (a trim) or a replay of its own stop, target and model exit at the asked size (a refusal, hypothetical). The
# idea itself (asked, approved, the test, the planned stop and target) is its gate_decisions row. One outcome per
# decision; write-once. A refusal's outcome never feeds the leg's real later entries.
SHADOW_KINDS = ("trim", "refusal")
SHADOW_METHODS = ("trade_scaled", "replay")  # trim: the real trade's per-unit P&L x (asked - approved); refusal: replay
shadow_outcomes_t = Table(
    "shadow_outcomes",
    metadata,
    Column("decision_id", BIG_ID, ForeignKey("gate_decisions.id", ondelete="RESTRICT"), primary_key=True,
           autoincrement=False),
    Column("kind", String(8), nullable=False),
    Column("pnl", EXACT, nullable=False),  # net of fees and slippage, in the fund's cash
    Column("hypothetical", Boolean, nullable=False),
    Column("method", String(16), nullable=False),
    Column("computed_at", TS, nullable=False),
    CheckConstraint(_in("kind", SHADOW_KINDS), name="shadow_outcomes_kind"),
    CheckConstraint(_in("method", SHADOW_METHODS), name="shadow_outcomes_method"),
    CheckConstraint("(kind = 'refusal' AND hypothetical AND method = 'replay') "
                    "OR (kind = 'trim' AND NOT hypothetical AND method = 'trade_scaled')", name="shadow_outcomes_shape"),
)
# Bridge A (DA, 0027; Advisor BRIDGE-3): the PM's compliance tests, versioned whole. One version is every test and the
# asset-group map together, so a decision that records its version re-checks exactly. Written only through
# Store.add_compliance_version (the PM's Portfolio Settings, or a migration's seed); never updated or deleted. The gate
# reads the latest version inside its lock, at evaluation, so an edit applies to the next order and never back.
COMPLIANCE_SCOPES = ("fund", "asset", "asset_type", "strategy")
COMPLIANCE_MEASURES = ("open_risk", "gross", "net", "margin", "day_loss", "drawdown", "strategy_risk", "contrib_dd")
# trim: to fit, refused below the venue minimum; pause_day: no entries until 00:00 UTC; pause_pm: the Strategy holds
# until the PM resumes it; halt: flatten everything and halt, only the PM restarts.
COMPLIANCE_ACTIONS = ("refuse", "trim", "pause_day", "pause_pm", "halt")
compliance_versions_t = Table(
    "compliance_versions",
    metadata,
    Column("version", Integer, primary_key=True, autoincrement=False),
    Column("created_at", TS, nullable=False),  # effective from: the next gate evaluation
    Column("created_by", Text, nullable=False),  # 'pm', or 'migration 0027' for the seed
    Column("note", Text),
    CheckConstraint("created_by <> 'pm' OR (note IS NOT NULL AND note <> '')", name="compliance_versions_pm_note"),
)
compliance_tests_t = Table(
    "compliance_tests",
    metadata,
    Column("version", Integer, ForeignKey("compliance_versions.version"), primary_key=True),
    Column("test", String(32), primary_key=True),  # a fixed key, e.g. asset_stop_loss (compliance_rules.KEYS)
    Column("label", Text, nullable=False),
    Column("scope", String(12), nullable=False),
    Column("measure", String(16), nullable=False),
    Column("limit_value", LIMIT, nullable=False),  # a share of fund equity; gross and net: times the book
    Column("action", String(12), nullable=False),
    Column("enabled", Boolean, nullable=False),
    CheckConstraint(_in("scope", COMPLIANCE_SCOPES), name="compliance_tests_scope"),
    CheckConstraint(_in("measure", COMPLIANCE_MEASURES), name="compliance_tests_measure"),
    CheckConstraint(_in("action", COMPLIANCE_ACTIONS), name="compliance_tests_action"),
    CheckConstraint("limit_value > 0", name="compliance_tests_limit"),
)
compliance_asset_types_t = Table(
    "compliance_asset_types",
    metadata,
    Column("version", Integer, ForeignKey("compliance_versions.version"), primary_key=True),
    Column("underlying", String(16), primary_key=True),
    Column("asset_type", String(32), nullable=False),  # the PM's group, e.g. large
    CheckConstraint("underlying = upper(underlying)", name="compliance_asset_types_upper"),
    CheckConstraint("asset_type <> ''", name="compliance_asset_types_named"),
)
COMPLIANCE_TABLES = ("compliance_versions", "compliance_tests", "compliance_asset_types")
COMPLIANCE_WORM_PG = (
    "CREATE OR REPLACE FUNCTION compliance_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    "MESSAGE = TG_TABLE_NAME || ' is write-once: a version is never edited or deleted, a change is a new version'; "
    "END $$",
)


def _compliance_worm_pg(table: str) -> str:
    return (f"CREATE TRIGGER {table}_write_once BEFORE UPDATE OR DELETE ON {table} FOR EACH ROW "
            "EXECUTE FUNCTION compliance_write_once()")


def _compliance_worm_sqlite(table: str) -> tuple[str, str]:
    return tuple(f"CREATE TRIGGER {table}_no_{op.lower()} BEFORE {op} ON {table} BEGIN SELECT RAISE(ABORT, "
                 f"'{table} is write-once: a version is never edited or deleted, a change is a new version'); END"
                 for op in ("UPDATE", "DELETE"))


# A decision's test_hit names a test of the version it records (so never without a version). A trigger, not a composite
# foreign key and a CHECK, so 0027 adds it without rebuilding gate_decisions on SQLite.
GATE_TEST_HIT_PG = (
    "CREATE OR REPLACE FUNCTION gate_test_hit_known() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "IF NEW.test_hit IS NOT NULL AND NOT EXISTS (SELECT 1 FROM compliance_tests WHERE version = NEW.compliance_version "
    "AND test = NEW.test_hit) THEN RAISE EXCEPTION USING ERRCODE = 'foreign_key_violation', "
    "MESSAGE = 'gate_decisions.test_hit ' || NEW.test_hit || ' is not a test of version ' "
    "|| COALESCE(NEW.compliance_version::text, 'none'); END IF; RETURN NEW; END $$",
    "CREATE TRIGGER gate_decisions_test_hit BEFORE INSERT OR UPDATE ON gate_decisions FOR EACH ROW "
    "EXECUTE FUNCTION gate_test_hit_known()",
)
GATE_TEST_HIT_SQLITE = tuple(
    f"CREATE TRIGGER gate_decisions_test_hit_{op.lower()} BEFORE {op} ON gate_decisions WHEN NEW.test_hit IS NOT NULL "
    "AND NOT EXISTS (SELECT 1 FROM compliance_tests WHERE version = NEW.compliance_version AND test = NEW.test_hit) "
    "BEGIN SELECT RAISE(ABORT, 'gate_decisions.test_hit is not a test of the version it records'); END"
    for op in ("INSERT", "UPDATE"))
for _t in (compliance_versions_t, compliance_tests_t, compliance_asset_types_t):
    for _sql in COMPLIANCE_WORM_PG:
        event.listen(_t, "after_create", DDL(_sql).execute_if(dialect="postgresql"))
    event.listen(_t, "after_create", DDL(_compliance_worm_pg(_t.name)).execute_if(dialect="postgresql"))
    for _sql in _compliance_worm_sqlite(_t.name):
        event.listen(_t, "after_create", DDL(_sql).execute_if(dialect="sqlite"))
for _sql in GATE_TEST_HIT_PG:
    event.listen(gate_decisions_t, "after_create", DDL(_sql).execute_if(dialect="postgresql"))
for _sql in GATE_TEST_HIT_SQLITE:
    event.listen(gate_decisions_t, "after_create", DDL(_sql).execute_if(dialect="sqlite"))

# Drop B (DA, 0028): the idea's columns on gate_decisions. Triggers, not CHECKs, so 0028 adds them without rebuilding
# gate_decisions on SQLite (as 0027's test_hit). would_be only with the test it measures.
GATE_IDEA_BAD = ("(NEW.side IS NOT NULL AND NEW.side NOT IN (-1, 1)) "
                 "OR (NEW.would_be IS NOT NULL AND NEW.test_hit IS NULL) "
                 "OR NEW.planned_stop <= 0 OR NEW.planned_target <= 0")
GATE_IDEA_PG = (
    "CREATE OR REPLACE FUNCTION gate_idea_shape() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    f"IF {GATE_IDEA_BAD} THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    "MESSAGE = 'gate_decisions: side is +1 or -1, would_be needs test_hit, a planned stop or target is above 0'; "
    "END IF; RETURN NEW; END $$",
    "CREATE TRIGGER gate_decisions_idea BEFORE INSERT OR UPDATE ON gate_decisions FOR EACH ROW "
    "EXECUTE FUNCTION gate_idea_shape()",
)
GATE_IDEA_SQLITE = tuple(
    f"CREATE TRIGGER gate_decisions_idea_{op.lower()} BEFORE {op} ON gate_decisions WHEN {GATE_IDEA_BAD} "
    "BEGIN SELECT RAISE(ABORT, 'gate_decisions: side is +1 or -1, would_be needs test_hit, a planned stop or target "
    "is above 0'); END"
    for op in ("INSERT", "UPDATE"))
# A shadow outcome is write-once, and its kind is its decision's outcome: a trim of a trimmed idea, a refusal of a
# rejected one.
SHADOW_WRONG_KIND = ("NOT EXISTS (SELECT 1 FROM gate_decisions WHERE id = NEW.decision_id AND outcome = "
                     "CASE NEW.kind WHEN 'trim' THEN 'trimmed' WHEN 'refusal' THEN 'rejected' END)")
SHADOW_PG = (
    "CREATE OR REPLACE FUNCTION shadow_outcome_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'shadow_outcomes is write-once'; END $$",
    "CREATE TRIGGER shadow_outcomes_write_once BEFORE UPDATE OR DELETE ON shadow_outcomes FOR EACH ROW "
    "EXECUTE FUNCTION shadow_outcome_write_once()",
    "CREATE OR REPLACE FUNCTION shadow_outcome_kind() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    f"IF {SHADOW_WRONG_KIND} THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    "MESSAGE = 'a shadow outcome is a trim of a trimmed decision or a refusal of a rejected one'; "
    "END IF; RETURN NEW; END $$",
    "CREATE TRIGGER shadow_outcomes_kind BEFORE INSERT ON shadow_outcomes FOR EACH ROW "
    "EXECUTE FUNCTION shadow_outcome_kind()",
)
SHADOW_SQLITE = (
    *(f"CREATE TRIGGER shadow_outcomes_no_{op.lower()} BEFORE {op} ON shadow_outcomes BEGIN "
      "SELECT RAISE(ABORT, 'shadow_outcomes is write-once'); END" for op in ("UPDATE", "DELETE")),
    f"CREATE TRIGGER shadow_outcomes_kind BEFORE INSERT ON shadow_outcomes WHEN {SHADOW_WRONG_KIND} BEGIN "
    "SELECT RAISE(ABORT, 'a shadow outcome is a trim of a trimmed decision or a refusal of a rejected one'); END",
)
for _t, _pg, _lite in ((gate_decisions_t, GATE_IDEA_PG, GATE_IDEA_SQLITE), (shadow_outcomes_t, SHADOW_PG, SHADOW_SQLITE)):
    for _sql in _pg:
        event.listen(_t, "after_create", DDL(_sql).execute_if(dialect="postgresql"))
    for _sql in _lite:
        event.listen(_t, "after_create", DDL(_sql).execute_if(dialect="sqlite"))

COMPLIANCE_V1_AT = datetime(2026, 10, 14, 21, 0, tzinfo=timezone.utc)
COMPLIANCE_V1_NOTE = "the PM's defaults (DEFAULTS-2, 14 Oct 2026); Advisor BRIDGE-3"


def _seed_compliance(target, conn, **kw) -> None:
    """Version 1 on a store made by create_all, as migration 0027 seeds it, so no database has a gate without tests.
    After every table is made (the map reads instruments), and only on an empty table."""
    from sleeve_fund import compliance_rules as rules

    if conn.execute(select(func.count()).select_from(compliance_versions_t)).scalar():
        return
    conn.execute(insert(compliance_versions_t).values(version=1, created_at=COMPLIANCE_V1_AT,
                                                      created_by="migration 0027", note=COMPLIANCE_V1_NOTE))
    conn.execute(insert(compliance_tests_t), [{"version": 1, "test": t["test"], "label": t["label"],
                                               "scope": t["scope"], "measure": t["measure"],
                                               "limit_value": t["limit"], "action": t["action"],
                                               "enabled": t["enabled"]} for t in rules.default_tests()])
    held = {u for (u,) in conn.execute(select(instruments_t.c.underlying).distinct())}
    groups = rules.DEFAULT_GROUPS | {u: rules.OTHER_GROUP for u in held if u not in rules.DEFAULT_GROUPS}
    conn.execute(insert(compliance_asset_types_t), [{"version": 1, "underlying": u, "asset_type": g}
                                                    for u, g in sorted(groups.items())])

# Bridge A (DA, 0025): the fund holds the capital, a Strategy none of its own. fund_flows is its only external capital,
# append-only: a row is never deleted or edited, only ended once at the poll (applied, or refused with the reason), in
# id order, so an inflow never overtakes a pending outflow (Advisor CAP5-RESET). rebase: the clean-slate deposit only,
# onto a book of exactly 0 (carry=False), checked in code.
FUND_FLOW_KINDS = ("deposit", "withdraw")
fund_flows_t = Table(
    "fund_flows",
    metadata,
    Column("id", BIG_ID, primary_key=True),
    Column("kind", String(8), nullable=False),
    Column("amount", EXACT, nullable=False),
    Column("rebase", Boolean, nullable=False, default=False, server_default=false()),
    Column("actor", String(32), nullable=False),
    Column("reason", Text, nullable=False),
    Column("requested_at", TS, nullable=False),
    Column("applied_at", TS),
    Column("refused_at", TS),
    Column("refuse_reason", Text),
    CheckConstraint(_in("kind", FUND_FLOW_KINDS), name="fund_flows_kind"),
    CheckConstraint("amount > 0", name="fund_flows_amount"),
    CheckConstraint("NOT rebase OR kind = 'deposit'", name="fund_flows_rebase"),
    CheckConstraint("actor <> ''", name="fund_flows_actor"),
    CheckConstraint("applied_at IS NULL OR refused_at IS NULL", name="fund_flows_one_end"),
    CheckConstraint("(refused_at IS NULL) = (refuse_reason IS NULL)", name="fund_flows_refuse_reason"),
    Index("fund_flows_pending", "id", sqlite_where=text("applied_at IS NULL AND refused_at IS NULL"),
          postgresql_where=text("applied_at IS NULL AND refused_at IS NULL")),
)
# What can be traded: static facts, one row per venue, symbol and contract. The asset type is not here: it is versioned
# with the compliance tests, so a decision's recorded version re-checks exactly (DA bridge spec 3).
CONTRACTS = ("spot", "perp")
instruments_t = Table(
    "instruments",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("venue", String(16), nullable=False),
    Column("symbol", String(32), nullable=False),  # as sleeves.instrument, e.g. BTC/USDT
    Column("contract", String(8), nullable=False),
    Column("underlying", String(16), nullable=False),  # what the gate nets by, e.g. BTC
    Column("created_at", TS, nullable=False),
    CheckConstraint("venue = upper(venue)", name="instruments_venue_upper"),
    CheckConstraint(_in("contract", CONTRACTS), name="instruments_contract"),
    UniqueConstraint("venue", "symbol", "contract", name="instruments_venue_symbol_contract"),
)
# The ledger is write-once at the database: no delete; an update may only end a pending row, once. Migration 0025
# writes the same triggers (spelled out there).
FUND_FLOWS_FROZEN = ("id", "kind", "amount", "rebase", "actor", "reason", "requested_at")
FUND_FLOWS_WORM_PG = (
    "CREATE OR REPLACE FUNCTION fund_flows_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "IF TG_OP = 'DELETE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    "MESSAGE = 'fund_flows is append-only: ' || OLD.id || ' cannot be deleted'; END IF; "
    f"IF ({', '.join('NEW.' + c for c in FUND_FLOWS_FROZEN)}) IS DISTINCT FROM "
    f"({', '.join('OLD.' + c for c in FUND_FLOWS_FROZEN)}) "
    "OR OLD.applied_at IS NOT NULL OR OLD.refused_at IS NOT NULL THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    "MESSAGE = 'fund_flows is append-only: ' || OLD.id || ' may only be ended once'; END IF; "
    "RETURN NEW; END $$",
    "CREATE TRIGGER fund_flows_write_once BEFORE UPDATE OR DELETE ON fund_flows FOR EACH ROW "
    "EXECUTE FUNCTION fund_flows_write_once()",
)
FUND_FLOWS_WORM_SQLITE = (
    "CREATE TRIGGER fund_flows_no_delete BEFORE DELETE ON fund_flows BEGIN "
    "SELECT RAISE(ABORT, 'fund_flows is append-only: a row cannot be deleted'); END",
    "CREATE TRIGGER fund_flows_no_edit BEFORE UPDATE ON fund_flows WHEN "
    + " OR ".join(f"NEW.{c} IS NOT OLD.{c}" for c in FUND_FLOWS_FROZEN)
    + " OR OLD.applied_at IS NOT NULL OR OLD.refused_at IS NOT NULL"
    + " BEGIN SELECT RAISE(ABORT, 'fund_flows is append-only: a row may only be ended once'); END",
)
for _sql in FUND_FLOWS_WORM_PG:
    event.listen(fund_flows_t, "after_create", DDL(_sql).execute_if(dialect="postgresql"))
for _sql in FUND_FLOWS_WORM_SQLITE:
    event.listen(fund_flows_t, "after_create", DDL(_sql).execute_if(dialect="sqlite"))
event.listen(metadata, "after_create", _seed_compliance)

# A row written without an instrument takes its Strategy's (bridge A, 0026): every writer today has one instrument per
# Strategy, so none of them changes, and Drop B's multi-asset writers name theirs. Postgres fills it before the insert;
# SQLite can't edit NEW, so the column defaults to '' and an AFTER INSERT trigger fills that row. Migration 0026 writes
# the same triggers (spelled out there).
INSTRUMENT_TABLES = ("orders", "fills", "funding", "insurance")
INSTRUMENT_FILL_PG = (
    "CREATE OR REPLACE FUNCTION fill_instrument() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "IF NEW.instrument IS NULL OR NEW.instrument = '' THEN "
    "NEW.instrument := COALESCE((SELECT instrument FROM sleeves WHERE name = NEW.sleeve), ''); END IF; "
    "RETURN NEW; END $$",
)


def _instrument_fill_pg(table: str) -> str:
    return (f"CREATE TRIGGER {table}_fill_instrument BEFORE INSERT ON {table} FOR EACH ROW "
            "EXECUTE FUNCTION fill_instrument()")


def _instrument_fill_sqlite(table: str) -> str:
    return (f"CREATE TRIGGER {table}_fill_instrument AFTER INSERT ON {table} WHEN NEW.instrument = '' BEGIN "
            f"UPDATE {table} SET instrument = COALESCE((SELECT instrument FROM sleeves WHERE name = NEW.sleeve), '') "
            f"WHERE id = NEW.id; END")


for _t in (orders_t, fills_t, funding_t, insurance_t):
    for _sql in INSTRUMENT_FILL_PG:
        event.listen(_t, "after_create", DDL(_sql).execute_if(dialect="postgresql"))
    event.listen(_t, "after_create", DDL(_instrument_fill_pg(_t.name)).execute_if(dialect="postgresql"))
    event.listen(_t, "after_create", DDL(_instrument_fill_sqlite(_t.name)).execute_if(dialect="sqlite"))

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
# What a row evaluated: one strategy's variant ("single", every row before 0012), an ablation of one (P2, Q5), or a
# multi-strategy portfolio run, whose member strategies' trial ids sit in its settings (QD M4). Whether ablations and
# portfolio runs count in N is the Advisor's call (research.trials._counted); the kind only records what was run.
TRIAL_KINDS = ("single", "ablation", "portfolio_run")
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
# A two-leg Strategy's leg (Advisor F283-O3, 14 Oct 13:15 UK): its incident also says why ruling B's close of the pair
# at half its entry distance to liquidation didn't prevent it; its reset after liquidation is refused until it does.
WHY_CLAMP_FIELD = "why closing the pair at half its distance to liquidation did not prevent it"
# The other leg's half of a two-leg Strategy's reset after liquidation: the summed figures re-based, nothing reset.
PAIR_REBASE = "pair re-base"
# Its journal kind on the other leg: never LIQUIDATION_RESET, which would answer that leg's own liquidation (each
# liquidated leg needs its own reset, Advisor F283-O3 (a)). It still starts a fresh run there (RAL-ANCHOR).
PAIR_REBASED = "pair_rebase"
# T fence (HoE + Advisor, 14 Oct 18:00 UK): F-C3B-A puts one row away, and how a two-leg Strategy's two legs leave the
# fund's book (one flow, both legs flat first) is not built yet. Archive, unarchive, retire, wind-down and a clean slate
# refuse one with these words; close-now, the PM's emergency exit, never does (it winds the pair down through its
# unwind and the retire waits for that design).
TWO_LEG_FENCE = "isn't built for a two-leg Strategy yet, so nothing was changed"
# Backtest names can't collide with a strategy's: those are lower-case letters, digits and dashes.
BACKTEST_PREFIX = "bt:"
# "triggered": paper's watched stop (not an order at the venue) when it fired, its market stop-loss sent for it.
# "unfilled_unknown" (F-RX-1, DA): the order-state check could not find the order at the venue and confirmed nothing
# filled. Finished, never open; a late fill still lands on it, and its gate room is released as a reject's.
ORDER_STATUSES = ("submitted", "accepted", "partially_filled", "filled", "canceled", "rejected", "denied", "expired",
                  "triggered", "unfilled_unknown")
FINISHED_ORDER_STATUSES = ("filled", "canceled", "rejected", "denied", "expired", "triggered", "unfilled_unknown")
OPEN_ORDER_STATUSES = ("submitted", "accepted", "partially_filled")
# F-C3B-A: a Strategy archived while not yet flat waits for its withdrawal; one with nothing to withdraw says so
WITHDRAWAL_DEFERRED, WITHDRAWAL_NONE = "withdrawal_deferred", "withdrawal_none"
PUT_AWAY_FUND_HELD = "put_away_fund_held"  # F-WN-1: put away once the fund holds the cash; withdrawn = 0
# RESET-NEG (Advisor, 15 Oct): the key of an old book close-out's cash rows (close_out_run). A transfer inside the book,
# never trading: the book's cash readers count it, the rows trades are paired with (insurance) leave it out.
CLOSE_OUT = "old-book-close-out"
NEXT_MARK = "flat, waiting for the supervisor's next mark"  # why a flat Strategy put away is deferred (F286-CR1)
# F-ASSIGN-STALE: a create or a move waiting for the poll (book_moves), and its landing
BOOK_MOVE_WAITING, BOOK_MOVE_LANDED = "book_move_waiting", "book_move_landed"
BOOK_MOVE_SUPERSEDED = "book_move_superseded"
GATED_INTENTS = ("entry", "rebalance")  # the orders the portfolio gate reserves for (engine OPENING_INTENTS)
INTENTS = ("entry", "exit", "stop_loss", "take_profit", "risk_halt", "risk_pause", "pm_flatten", "rebalance",
           "liquidation", "liquidation_cut", "wind_down", "wind_down_stop",  # the venue would take it; cut back before it does (String(16))
           "leg_unwind")  # P2-8: a two-leg Strategy's leg closed because its other leg couldn't be held
# F-RX-1: the orders that close or cut a position the Strategy holds (store.closing_orders). Not the venue's own
# liquidation, which the Strategy never sent.
CLOSING_INTENTS = ("exit", "stop_loss", "take_profit", "risk_halt", "risk_pause", "pm_flatten", "liquidation_cut",
                   "wind_down", "wind_down_stop", "leg_unwind")


def _traded_venue(c, sleeve: str) -> str | None:
    """F-WED1-3: the venue the strategy last traded on, as its journal holds it: that of the order behind its latest
    fill; None when it has no fills. Read on the caller's connection."""
    return c.execute(select(orders_t.c.venue).join(fills_t, (fills_t.c.order_id == orders_t.c.order_id)
                                                   & (fills_t.c.sleeve == orders_t.c.sleeve))
                     .where(fills_t.c.sleeve == sleeve).order_by(fills_t.c.ts.desc(), fills_t.c.id.desc())
                     .limit(1)).scalar()


def holding(qty) -> bool:
    """Whether a journal's net position is a position at all (F-WED1-3, DA): the one rule record_order's venue and
    the start's venue check share. Dust is a position here: its exit goes where it is."""
    return abs(qty) > 1e-12


def _position_venue(c, sleeve: str) -> str | None:
    """F-WED1-3: the venue the journal says the strategy's open position is on (_traded_venue) while it holds one;
    None when flat. The fills are summed here as Decimals, each read back exact (DA-9): a sum inside the database
    is in floats on SQLite (DA). Read on the caller's connection."""
    net = sum((to_decimal(q) if s == "BUY" else -to_decimal(q))
              for s, q in c.execute(select(fills_t.c.side, fills_t.c.qty).where(fills_t.c.sleeve == sleeve)))
    return _traded_venue(c, sleeve) if holding(net) else None


# The closes a fill at or past the liquidation price re-books as a liquidation (F283-P1, GAP-LIQ on every exit path):
# every intent but an opening one or a liquidation, exactly as the engine's _gap_liquidation relabels (T-FIX7).
REBOOKABLE = tuple(i for i in INTENTS if i not in GATED_INTENTS and i != "liquidation")


def _check_rebook(order_id: str, row) -> None:
    """Refuse any re-booking but a filled close becoming a liquidation (Store/MemoryJournal.rebook_liquidation)."""
    if row is None:
        raise ValueError(f"no order {order_id!r} to re-book")
    if row["intent"] not in REBOOKABLE or not row["filled_qty"]:
        raise ValueError(f"order {order_id!r} can't be re-booked as a liquidation: only a filled close can "
                         f"(it is {row['intent']}, {row['filled_qty']:g} filled)")


def _rebook_words(order_id: str, intent: str, reason: str) -> str:
    return f"Order {order_id} re-booked from {intent} to liquidation: {reason}"


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


def qty_words(qty: Decimal) -> str:
    """A quantity as words read it, the same on every engine: Postgres hands a NUMERIC back at the column's scale
    (1.000000000E-9), SQLite as 1E-9; plain digits either way, never an exponent (TEC-DUST-FMT)."""
    return format(qty.normalize(), "f")


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
        price, q, fee = D(f["price"], "price"), D(f["qty"], "qty"), per_fill(D(f["fee"], "fee"))
        sign = 1 if f["side"] == "BUY" else -1
        cash += fill_cash(f["side"], q, price, fee)
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


# A fill's cash leg and its fee each count at 8 dp, rounded half up, on their own: no carry, so a restart can't move
# one (Advisor 8 Oct 04:10 UK, ruling 1, CASH-3). Not to the cent: a per-fill cent is systematic on fixed-size fills.
FILL_DP = Decimal("1e-8")


def per_fill(amount: Decimal) -> Decimal:
    return amount.quantize(FILL_DP, rounding=ROUND_HALF_UP)


def fill_cash(side: str, qty, price, fee) -> Decimal:
    """What one fill moved the strategy's cash by, as replay_book books it: a buy pays qty * price and the fee, a sell
    receives qty * price less the fee, each at 8 dp half up (per_fill)."""
    D = to_decimal
    leg = per_fill(D(qty, "qty") * D(price, "price"))
    return (-leg if side == "BUY" else leg) - per_fill(D(fee, "fee"))


class _NotRetired(Exception):
    """complete_retire found it retired by another process: roll its put-away back and return False."""


def _outside_book(store: "Store", s, account: str | None = None) -> bool:
    """supervisor.outside_fund_book, the one predicate (imported late: the supervisor imports this module)."""
    from sleeve_fund.supervisor import outside_fund_book

    return outside_fund_book(store, s) if account is None else outside_fund_book(store, s, account)


def is_backtest(name: str | None) -> bool:
    return bool(name) and name.startswith(BACKTEST_PREFIX)


def _not_backtest(col):
    """Rows that belong to paper or live (or to no strategy), not to a saved backtest."""
    return or_(col.is_(None), ~col.like(BACKTEST_PREFIX + "%"))


def _insert_sleeve(c, ts: datetime, *, name: str, strategy: str, instrument: str, bar_spec: str, params: dict | None,
                   starting_balance, risk_profile: str, warmup_bars: int, desired_state: str, venue: str | None,
                   parent: str | None = None) -> None:
    c.execute(insert(sleeves_t).values(
        name=name, strategy=strategy, instrument=instrument, bar_spec=bar_spec, params=params or {},
        starting_balance=starting_balance, risk_profile=risk_profile, warmup_bars=warmup_bars,
        desired_state=desired_state, status="starting" if desired_state == "running" else "stopped",
        status_reason="", created_at=ts, updated_at=ts, parent=parent,
    ))
    if venue:
        c.execute(insert(sleeve_venues_t).values(sleeve=name, venue=venue.upper()))
    if not is_backtest(name):  # what it trades is in instruments from its first row (bridge A, 0025)
        from sleeve_fund.portfolio.holding import underlying
        from sleeve_fund.venues import DEFAULT_VENUE

        _ensure_instrument(c, (venue or DEFAULT_VENUE).upper(), instrument, _contract(params), underlying(instrument))


def _contract(params: dict | None) -> str:
    """The contract a Strategy trades: a perp when its params say a futures market, else spot (as 0025's seed)."""
    return "spot" if ((params or {}).get("market") or "spot") == "spot" else "perp"


def _ensure_instrument(c, venue: str, symbol: str, contract: str, underlying: str) -> int:
    """The instrument's id, its row written if missing; refused when the row there nets as another underlying."""
    if contract not in CONTRACTS:
        raise ValueError(f"a contract is {' or '.join(CONTRACTS)}, not {contract!r}")
    t, venue = instruments_t, venue.upper()
    row = c.execute(select(t.c.id, t.c.underlying)
                    .where(t.c.venue == venue, t.c.symbol == symbol, t.c.contract == contract)).first()
    if row is not None:
        if row.underlying != underlying:
            raise ValueError(f"{venue} {symbol} {contract} nets as {row.underlying}, not {underlying}")
        return row.id
    return c.execute(insert(t).values(venue=venue, symbol=symbol, contract=contract, underlying=underlying,
                                      created_at=utcnow())).inserted_primary_key[0]


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _from_ns(ns: int) -> datetime:
    """UNIX nanoseconds -> an aware UTC datetime to the microsecond (utcnow() drops them)."""
    return _EPOCH + timedelta(microseconds=ns // 1000)


def _money2(x) -> str:
    return "none" if x is None else f"{x:,.2f}"


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
    lifecycle: str = "active"  # P2-6: active, winding_down or retired (sleeve_fund.lifecycle)
    wind_down_started: datetime | None = None
    wind_down_deadline: datetime | None = None
    venue: str | None = None  # None: the default venue (sleeve_fund.venues.DEFAULT_VENUE)
    parent: str | None = None  # P2-8: on a two-leg Strategy's second leg row, the first leg's name

    @classmethod
    def from_row(cls, row, venue: str | None = None) -> "Sleeve":
        d = dict(row._mapping)
        for k in ("paused_until", "heartbeat_at", "created_at", "updated_at", "wind_down_started",
                  "wind_down_deadline"):
            d[k] = _aware(d[k])
        d["params"] = dict(d["params"] or {})
        # A configured balance, read by the engine's float figures: the journal (journal_book) takes it exactly.
        d["starting_balance"] = float(d["starting_balance"])
        return cls(**d, venue=venue)


def _end_move(c, row, **end: datetime) -> bool:
    """End a waiting book_moves row once (DA 0024 rule 1): landed_at or superseded_at, guarded on its still waiting, so
    a row two writers race for ends once; False when it had ended already. Never before it was asked for (CHECK
    book_moves_order): a poll whose clock reads behind the request's ends it at the request's time."""
    (col, ts), = end.items()
    req = _aware(row.requested_at)
    return bool(c.execute(update(book_moves_t).where(
        book_moves_t.c.id == row.id, book_moves_t.c.landed_at.is_(None), book_moves_t.c.superseded_at.is_(None))
        .values(**{col: max(_aware(ts), req)})).rowcount)


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
    if r.get("kind", "single") not in TRIAL_KINDS:
        raise ValueError(f"a trial's kind is one of {TRIAL_KINDS}, got {r['kind']!r}")
    if r["source"] != "ledger_import" and r["stage"] not in TRIAL_STAGES:
        raise ValueError(f"a trial's stage is one of {TRIAL_STAGES}, got {r['stage']!r}")
    sharpe = r.get("sharpe")
    if sharpe is not None and not math.isfinite(sharpe):
        raise ValueError("a trial's Sharpe is a finite number or None, never NaN or infinite")


def _put_trials(c, rows: list[dict]) -> int:
    """Insert trials on an open transaction, skipping ids already there. Returns how many were added."""
    have = {i for (i,) in c.execute(select(trials_t.c.id).where(trials_t.c.id.in_([r["id"] for r in rows])))}
    new = [{**r, "created_at": r.get("created_at") or utcnow(), "data_start": r.get("data_start"),
            "data_end": r.get("data_end"), "status": r.get("status", "ok"), "error": r.get("error"),
            "kind": r.get("kind", "single")}
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


def _drop_carried(c) -> bool:
    """Drops the ratios a withdrawal that emptied the book carried to the next capital's first mark (MUST-F286-H4),
    while the book is still empty: a book reset or a clean slate is a fresh start. True when there were any."""
    st = portfolio_state_t.c
    return bool(c.execute(update(portfolio_state_t)
                          .where(st.id == 1, st.marked_at.is_(None), st.carried_hwm_drawdown.is_not(None))
                          .values(carried_hwm_drawdown=None, carried_reference_drawdown=None,
                                  carried_day_return=None, carried_day=None)).rowcount)


def _fund_held(store) -> bool:
    """Whether the fund holds the cash (Bridge Drop A: the PM's re-base deposit applied), so a Strategy made, moved or
    put away moves no capital (supervisor.bridge_start)."""
    from sleeve_fund.supervisor import bridge_start

    return bridge_start(store) is not None


def _no_own_capital(store, name: str, starting_balance) -> None:
    """CUT-A (a) (HoE, Advisor): once the fund holds the cash, a Strategy is made with no capital of its own."""
    if not is_backtest(name) and to_decimal(starting_balance or 0) != 0 and _fund_held(store):
        raise ValueError(f"Strategy {name} can't be made with {to_decimal(starting_balance):,.2f}: the fund holds the "
                         "cash, so a Strategy has no capital of its own; make it with 0")


def check_own_capital(store, name: str, starting_balance) -> None:
    """F-ZS-1 (HoE, Drop A): until the fund holds the cash a Strategy is its own fund, so one with no balance of its own
    could only halt on its first bar. It is refused at start (the supervisor, and a --db-sleeve process started by
    hand) rather than at creation. A saved backtest is neither."""
    if is_backtest(name) or to_decimal(starting_balance or 0) > 0 or _fund_held(store):
        return
    raise ValueError(f"{name} has {to_decimal(starting_balance or 0):,.2f} of its own and the fund doesn't hold the "
                     "cash yet, so it would have nothing to trade with: until the fund holds the cash a Strategy "
                     "needs a starting balance above 0")


@dataclass(frozen=True)
class Compliance:
    """One version of the PM's compliance tests: the tests (dicts: test, label, scope, measure, limit, action,
    enabled) and the asset-group map (underlying -> group)."""
    version: int
    created_at: datetime
    created_by: str
    note: str | None
    tests: list
    asset_types: dict

    def test(self, key: str) -> dict | None:
        return next((t for t in self.tests if t["test"] == key), None)


def _compliance_changes(old: Compliance | None, rows: list[dict], groups: dict[str, str]) -> str:
    """What a new version changes, old -> new, for the decision log."""
    if old is None:
        return "first version"
    was = {t["test"]: t for t in old.tests}
    said = []
    for r in rows:
        o = was.get(r["test"])
        if o is None:
            said.append(f"{r['test']} added")
            continue
        if Decimal(o["limit"]) != r["limit_value"]:
            said.append(f"{r['test']} {o['limit']:f} -> {r['limit_value']:f}")
        if o["enabled"] != r["enabled"]:
            said.append(f"{r['test']} {'on' if r['enabled'] else 'off'}")
    for u in sorted(set(old.asset_types) | set(groups)):
        if old.asset_types.get(u) != groups.get(u):
            said.append(f"{u} group {old.asset_types.get(u) or 'none'} -> {groups.get(u) or 'none'}")
    return ", ".join(said) or "no change"


# DA-6 reset coverage: what a reset (Store.split_run) does with each table that names a Strategy (a sleeve, sleeve_id
# or parent column, or a foreign key to sleeves). Every such table is in exactly one list, and a cell
# (test_reset_coverage) fails when a new one is in none, so no new table silently mixes a put-away run with the fresh
# one. The journal moves to the run's name; the venue and account pins are copied to it (the fresh run keeps them);
# feed and signal state are the process's and go; everything else stays with the Strategy, for the reason given.
RESET_MOVES = (decisions_t, events_t, commands_t, mirror_t, equity_t, exit_plans_t, fills_t, funding_t, insurance_t,
               orders_t, order_timings_t, mirror_requests_t)
RESET_COPIES = (sleeve_accounts_t, sleeve_venues_t)
RESET_CLEARS = (feed_seen_t, signal_state_t)
RESET_STAYS = {
    sleeves_t: "the Strategy's own row; the run gets a new row (and a new id) made from it",
    sleeve_archive_t: "written for the run, never for the Strategy",
    resets_t: "the reset request itself, keyed to the Strategy that asked",
    backtests_t: "a saved backtest is never reset; it keeps its own name",
    book_moves_t: "the fund's record of the Strategy's capital landing in the book, not one run's",
    pair_state_t: "a two-leg Strategy's coordinator; a reset happens only when flat, and the state is a hint",
    gate_decisions_t: "keyed by the Strategy's id, which the reset keeps: the gate's record across runs",
    gate_reservations_t: ("keyed by the Strategy's id, kept as history; any still active is released as the run is "
                          "put away (its process is stopped first, so its order can't fill), so none from before a "
                          "reset counts against the tests after it"),
    # No Strategy column: each hangs off a row the reset moves or keeps by id (a move is an UPDATE of its sleeve, so
    # ids hold), and follows it
    acks_t: "an ack names its event by id: the event moves with the run, and the ack still points at it",
    shadow_outcomes_t: "one per gate decision, which stays with the Strategy (above); write-once, never moved",
}
STRATEGY_COLUMNS = frozenset({"sleeve", "sleeve_id", "parent", "strategy_id"})



LEG_IDENTITY = ("assets", "universe_date")  # create_legs: a multi-asset Strategy's identity, kept in every row's params


def model_params(params: dict | None) -> dict:
    """A row's settings as its model takes them: a multi-asset Strategy's asset list and universe_date (create_legs)
    are its identity, not the model's, and the model's config refuses them (CR A10-M2)."""
    return {k: v for k, v in (params or {}).items() if k not in LEG_IDENTITY}

class Store:
    def __init__(self, url: str | None = None, engine: Engine | None = None, portfolio_gate: bool = False) -> None:
        """portfolio_gate: the portfolio gate is in force on this journal, and with no state row its entries are
        blocked (runtime.portfolio_block). Paper and live open their journal with it, and their startup refuses one
        without it (paper.safety.assert_portfolio_gate); a research backtest's or a tool's journal doesn't."""
        self.engine = engine or make_engine(url)
        self.portfolio_gate = portfolio_gate
        # For another process to open the same journal; an in-memory database can't be shared.
        url = None if engine is not None else (url or os.environ.get("DATABASE_URL", DEFAULT_URL))
        self.url = None if url in (None, "sqlite://", "sqlite:///:memory:") else url
        metadata.create_all(self.engine)

    @classmethod
    def in_memory(cls, portfolio_gate: bool = False) -> "Store":
        """A private, throwaway store (one connection shared, so every call sees the same tables):
        the journal a backtest's runtime writes to."""
        from sqlalchemy.pool import StaticPool

        return cls(engine=create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}),
                   portfolio_gate=portfolio_gate)

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
        _no_own_capital(self, name, starting_balance)
        ts = utcnow()
        made = SimpleNamespace(name=name, instrument=instrument, venue=venue.upper() if venue else None)

        def into() -> Decimal:  # its capital comes into the fund's book: a deposit, not a gain (HALT-REF-1)
            in_book = not is_backtest(name) and not _outside_book(self, made)  # no account row yet: it starts on paper
            return to_decimal(starting_balance) if in_book and not _fund_held(self) else Decimal(0)

        def joins(c) -> None:
            # F-ASSIGN-STALE (Advisor ADV-ASSIGN): on a marked book its capital joins at the supervisor's next poll,
            # after that poll's mark; outside the book until then (supervisor.outside_fund_book), entries refused.
            self._ensure_paper_account(c)
            self._wait_move(c, name, "paper", "create", ts)

        defer = None if is_backtest(name) else joins
        with self._book_move(into, f"Strategy {name} created with {to_decimal(starting_balance):,.2f}", ts,
                             defer=defer) as c:
            _insert_sleeve(c, ts, name=name, strategy=strategy, instrument=instrument, bar_spec=bar_spec,
                           params=params, starting_balance=starting_balance, risk_profile=risk_profile,
                           warmup_bars=warmup_bars, desired_state=desired_state, venue=venue)
            if trial is not None:
                _put_trial_in(c, trial)
        return self.sleeve(name)

    def create_two_legs(self, *, name: str, strategy: str, bar_spec: str, starting_balance: float, first: dict,
                        second: dict, risk_profile: str = "balanced", warmup_bars: int = 0,
                        desired_state: str = "running", trial: dict | None = None) -> tuple[Sleeve, Sleeve]:
        """A two-leg Strategy (P2-8): the first leg's row `name` and the second's `name~2` under it, made together in
        one transaction (with its trials row when given), or neither. first and second: each leg's instrument,
        params and venue, its side (+1 long, -1 short) and its notional per unit of the first leg's (ratio, 1 for the
        first). The Strategy's balance is split by what each leg needs at its ratio (two_legs.split_capital), and
        each row's starting_balance is its share, so each leg's book reads from the journal like any other. Refused:
        a short leg that isn't a perp (Advisor, 13 Oct 23:55 UK: no spot borrowing) and a perp leg above 2x."""
        from sleeve_fund import two_legs

        first, second = two_legs.check_legs(first, second)
        _no_own_capital(self, name, starting_balance)
        # F-PAIR-IN-FUND: once the fund holds the cash both legs are made with 0, as any Strategy is (CUT-A (a)).
        shares = ((Decimal(0), Decimal(0)) if not is_backtest(name) and to_decimal(starting_balance or 0) == 0
                  and _fund_held(self) else two_legs.split_capital(starting_balance, (first, second)))
        if trial is not None:
            check_trial(trial)
        ts = utcnow()
        legs = ((name, first, shares[0], None), (name + LEG, second, shares[1], name))

        def in_book(leg_name: str, leg: dict) -> bool:
            return not is_backtest(leg_name) and not _outside_book(self, SimpleNamespace(
                name=leg_name, instrument=leg["instrument"], venue=(leg.get("venue") or "").upper() or None))

        def into() -> Decimal:  # each leg's share in the fund's book comes in as a deposit (HALT-REF-1)
            if _fund_held(self):
                return Decimal(0)
            return sum((to_decimal(share) for leg_name, leg, share, _ in legs if in_book(leg_name, leg)), Decimal(0))

        def joins(c) -> None:
            # F-ASSIGN-STALE: on a marked book both legs join at the supervisor's next poll, together (Store.land_move)
            self._ensure_paper_account(c)
            for leg_name, leg, _, _ in legs:
                if in_book(leg_name, leg):
                    self._wait_move(c, leg_name, "paper", "create", ts)

        with self._book_move(into, f"two-leg Strategy {name} created with {to_decimal(starting_balance):,.2f}", ts,
                             defer=None if is_backtest(name) else joins) as c:
            for leg_name, leg, share, parent in legs:
                _insert_sleeve(c, ts, name=leg_name, strategy=strategy, instrument=leg["instrument"],
                               bar_spec=bar_spec, params=two_legs.leg_params(leg), starting_balance=share,
                               risk_profile=risk_profile, warmup_bars=warmup_bars, desired_state=desired_state,
                               venue=leg.get("venue"), parent=parent)
            c.execute(insert(decisions_t).values(
                ts=ts, actor="system", action="two_legs", sleeve=name,
                reason=f"two legs made: {first['instrument']} gets {shares[0]} and {second['instrument']} "
                       f"{shares[1]} of the {starting_balance} balance"))
            if trial is not None:
                _put_trial_in(c, trial)
        return self.sleeve(name), self.sleeve(name + LEG)

    def create_legs(self, *, name: str, strategy: str, bar_spec: str, assets: list[str], params: dict | None,
                    starting_balance: float, universe_date: str, risk_profile: str = "balanced", warmup_bars: int = 0,
                    desired_state: str = "running", venue: str | None = None,
                    trial: dict | None = None) -> list[Sleeve]:
        """A multi-asset Strategy (multi-asset spec §1-2): the parent row `name` trades the first asset, and each other
        asset is a leg row "<name>~<UNDERLYING>" under it, every row with the one settings block, made together in one
        transaction (with its trials row when given), or none. The list and universe_date are kept on the parent's
        params: fixed for the Strategy's life (a new list is a new Strategy). Inside the fund only, every leg with 0
        (CUT-A (a)); legs are independent ideas, never a pair."""
        if is_backtest(name):
            raise ValueError("a multi-asset backtest is a portfolio run of its legs, not stored rows")
        if "~" in name:
            raise ValueError(f"a Strategy's name can't hold a \"~\": {name!r}")
        if not _fund_held(self):
            raise ValueError("a multi-asset Strategy runs inside the fund: make it once the fund holds the cash")
        if to_decimal(starting_balance or 0) != 0:
            _no_own_capital(self, name, starting_balance)
        assets = [str(a).strip().upper() for a in assets]
        if len(assets) < 2:
            raise ValueError("a multi-asset Strategy lists at least two assets; one asset is a single-asset Strategy")
        unders = [a.split("/")[0] for a in assets]
        if bad := [u for u in unders if not re.fullmatch(r"[A-Z0-9]{1,16}", u)]:  # 0028's sleeves_leg_name
            raise ValueError(f"an asset's name is 1-16 letters or digits: {bad}")
        if len(set(unders)) != len(unders):
            raise ValueError(f"an asset is listed twice: {assets}")
        from sleeve_fund.venues import DEFAULT_VENUE, venue as venue_profile

        venue_profile(venue)  # F-B3C-VENUE (DA): a registered venue, or ValueError (unknown venue)
        if bad := [a for a in assets if not re.fullmatch(r"[A-Z0-9]+/[A-Z0-9]+", a)]:  # F-B3C-VENUE

            raise ValueError(f"each asset is BASE/QUOTE on the Strategy's one venue ({(venue or DEFAULT_VENUE).upper()})"
                             f", with no venue of its own: {bad}")
        if len(quotes := {a.split("/")[1] for a in assets}) > 1:
            raise ValueError(f"a multi-asset Strategy's assets share one quote currency, its venue's account's: "
                             f"{', '.join(sorted(quotes))} in {assets}")
        if not universe_date:
            raise ValueError("a multi-asset Strategy names the date its list was chosen (universe_date)")
        params = {**(params or {}), "assets": assets, "universe_date": str(universe_date)}
        if trial is not None:
            check_trial(trial)
        ts = utcnow()
        rows = [(name, assets[0], None)] + [(f"{name}~{u}", a, name) for u, a in zip(unders[1:], assets[1:])]

        def joins(c) -> None:  # F-ASSIGN-STALE: every leg joins at the supervisor's next poll, together
            self._ensure_paper_account(c)
            for leg_name, _, _ in rows:
                self._wait_move(c, leg_name, "paper", "create", ts)

        with self._book_move(lambda: Decimal(0), f"multi-asset Strategy {name} created", ts, defer=joins) as c:
            for leg_name, instrument, parent in rows:
                _insert_sleeve(c, ts, name=leg_name, strategy=strategy, instrument=instrument, bar_spec=bar_spec,
                               params=params, starting_balance=0, risk_profile=risk_profile, warmup_bars=warmup_bars,
                               desired_state=desired_state, venue=venue, parent=parent)
            c.execute(insert(decisions_t).values(
                ts=ts, actor="system", action="multi_asset", sleeve=name,
                reason=f"asset legs made: {', '.join(assets)}, fixed on {universe_date}"))
            if trial is not None:
                _put_trial_in(c, trial)
        return [self.sleeve(n) for n, _, _ in rows]

    @contextmanager
    def _book_move(self, amount, why, ts: datetime | None = None, leaving: str | None = None, carry: bool = True,
                   defer=None, waits_out: list | None = None):
        """A capital move into (amount() > 0) or out of (< 0) the fund's book (HALT-REF-1): the caller's writes, made on
        the connection this yields, and the flow they make commit together under the portfolio gate's lock, so the
        book moves by it and its HWM, the halt's reference and the day's start scale with it (portfolio.gate.apply_flow),
        so the move is never read as a gain or a loss. The amount is read under the lock, before the writes, so two moves never both
        count the same Strategy, and a supervisor mark that read the book while a move landed is skipped
        (Supervisor.mark_portfolio). Journaled as a portfolio_flow event. With no state ever marked there is nothing to
        move: the first mark reads the book as it is. leaving: the Strategy put away by this move, which never holds an
        emptied book open itself (supervisor.nothing_open, F286-1). carry: a move that empties the book carries its
        drawdown and the day's change to the next capital's first mark (MUST-F286-H4); False for a clean slate's.
        defer: on a marked book, a move that would flow waits for the supervisor's poll instead (F-ASSIGN-STALE,
        Advisor ADV-ASSIGN: only the poll moves capital on a marked book): no flow now; defer(c) records the waiting
        move after the caller's writes, on the same connection. A move that flows nothing (the same side of the book)
        has no stale scale to fear, so it is made at once. waits_out: told (appended to) whether it waits, before the
        caller's writes, which are then the waiting move's own (defer) rather than the move's."""
        from sleeve_fund.portfolio.gate import apply_flow
        from sleeve_fund.supervisor import nothing_open

        led = self._gate_ledger()
        with led.lock():
            c = led._local.conn
            marked = c.execute(select(portfolio_state_t.c.book_equity).where(portfolio_state_t.c.id == 1)).scalar()
            amount = amount() if marked is not None else Decimal(0)
            waits = defer is not None and bool(amount)
            amount = Decimal(0) if waits else amount
            if waits_out is not None:
                waits_out.append(waits)
            yield c
            if waits:
                defer(c)
            if amount:
                before = led.state().reference()
                emptied = apply_flow(led, amount, lambda: nothing_open(self, led, leaving), ts, carry=carry)
                if emptied is not None:  # the old figures stay in the journal (Advisor 14 Oct 10:00 UK)
                    kept = led.state()
                    if kept.carried_hwm_drawdown is None:
                        then = "so the next mark starts it afresh"
                    else:  # MUST-F286-H4: what the next capital's first mark restores
                        day = ("" if kept.carried_day_return is None else
                               f" and the day's change of {kept.carried_day_return:.4%} on {kept.carried_day}")
                        ref = ("" if kept.carried_reference_drawdown is None else
                               f", {kept.carried_reference_drawdown:.4%} below the halt's reference")
                        then = (f"so the next capital's first mark keeps its drawdown of "
                                f"{kept.carried_hwm_drawdown:.4%}{ref}{day}")
                    tail = (f"which empties it, {then} (it was: high-water mark "
                            f"{_money2(emptied.hwm)}, the halt's reference {_money2(emptied.reference())}, the day's "
                            f"start {_money2(emptied.day_start_equity)}); a halt in force stays until the PM resumes")
                else:
                    tail = (f"and its high-water mark, the halt's reference and the day's start scale with it (the "
                            f"reference {before:,.2f} -> {led.state().reference():,.2f})")
                c.execute(insert(events_t).values(
                    sleeve=None, ts=ts or utcnow(), level="info", kind="portfolio_flow",
                    message=f"{why() if callable(why) else why}: the fund's book moves by {amount:+,.2f} {tail}, so "
                            "it isn't a gain or a loss"))

    def sleeve(self, name: str) -> Sleeve:
        with self.engine.connect() as c:
            row = c.execute(select(sleeves_t).where(sleeves_t.c.name == name)).first()
            venue = c.execute(select(sleeve_venues_t.c.venue).where(sleeve_venues_t.c.sleeve == name)).scalar()
        if row is None:
            raise KeyError(f"no strategy {name!r}")
        return Sleeve.from_row(row, venue)

    def sleeves(self, include_backtests: bool = False, include_legs: bool = True) -> list[Sleeve]:
        """Paper and live strategies. Saved backtests are left out unless asked for: they never run,
        count towards the book, or raise alerts. include_legs=False leaves out a two-leg Strategy's second leg rows
        (P2-8), for what lists or runs Strategies; the money paths keep them, as each leg holds its own book."""
        q = select(sleeves_t).order_by(sleeves_t.c.created_at, sleeves_t.c.id)
        if not include_backtests:
            q = q.where(_not_backtest(sleeves_t.c.name))
        if not include_legs:
            q = q.where(sleeves_t.c.parent.is_(None))
        with self.engine.connect() as c:
            rows = c.execute(q).all()
            venues = dict(c.execute(select(sleeve_venues_t.c.sleeve, sleeve_venues_t.c.venue)).all())
        return [Sleeve.from_row(r, venues.get(r.name)) for r in rows]

    def _update_sleeve(self, name: str, **values) -> None:
        if "starting_balance" in values:  # HALT-REF-1: capital into or out of the book only with its flow
            raise ValueError("a Strategy's starting balance is set when it is created and never edited")
        with self.engine.begin() as c:
            c.execute(update(sleeves_t).where(sleeves_t.c.name == name).values(updated_at=utcnow(), **values))

    def set_desired_state(self, name: str, state: str) -> None:
        if state not in {"running", "stopped"}:
            raise ValueError(f"bad desired state {state!r}")
        if is_backtest(name):
            raise ValueError("a saved backtest can't be started")
        if self.multi_asset(name) and self.sleeve(name).parent is not None:
            raise ValueError(f"{name} is an asset leg of {self.sleeve(name).parent}: start or stop the Strategy")
        names = [name] + ([leg] if (leg := self._leg_of(name)) is not None else self._asset_legs(name))
        with self.engine.begin() as c:  # a two-leg or multi-asset Strategy starts and stops as one (P2-8)
            c.execute(update(sleeves_t).where(sleeves_t.c.name.in_(names)).values(updated_at=utcnow(),
                                                                                  desired_state=state))

    def _leg_of(self, name: str, c=None) -> str | None:
        """A two-leg Strategy's second leg row, when `name` is its first leg's (P2-8); else None. A multi-asset
        Strategy's asset legs are not one: they are never a pair (_asset_legs)."""
        q = select(sleeves_t.c.name).where(sleeves_t.c.parent == name, sleeves_t.c.name == name + LEG)
        if c is not None:
            return c.execute(q).scalar()
        with self.engine.connect() as conn:
            return conn.execute(q).scalar()

    def legs(self, name: str) -> list[str]:
        """Every row of the Strategy `name` belongs to: [name] for one leg; a two-leg Strategy's first leg's row and
        its second's (P2-8), or a multi-asset Strategy's parent and its asset legs in asset order, from any of them."""
        row = self.sleeve(name)
        first = row.parent or name
        if (leg := self._leg_of(first)) is not None:
            return [first, leg]
        return [first] + self._asset_legs(first)

    def _asset_legs(self, parent: str, c=None) -> list[str]:
        """A multi-asset Strategy's asset leg rows ("<parent>~<ASSET>"), in its asset list's order; [] for any other."""
        q = (select(sleeves_t.c.name).where(sleeves_t.c.parent == parent, sleeves_t.c.name != parent + LEG)
             .order_by(sleeves_t.c.id))
        if c is not None:
            return list(c.execute(q).scalars())
        with self.engine.connect() as conn:
            return list(conn.execute(q).scalars())

    def two_leg(self, name: str) -> bool:
        """Whether `name` is either leg's row of a two-leg Strategy (P2-8): a hedged pair, never a multi-asset one."""
        row = self.sleeve(name)
        return name.endswith(LEG) and row.parent is not None or self._leg_of(name) is not None

    def multi_asset(self, name: str) -> bool:
        """Whether `name` is the parent or an asset leg of a multi-asset Strategy (multi-asset spec §1)."""
        row = self.sleeve(name)
        return (row.parent is not None and not name.endswith(LEG)) or bool(self._asset_legs(name))

    def holds_any(self, name: str) -> bool:
        """Whether any leg of the Strategy `name` belongs to holds a position an order can close (not dust): a
        two-leg Strategy holds while either leg does (CR F283-2)."""
        for leg in self.legs(name):
            book = self.journal_book(leg, self.sleeve(leg).starting_balance)
            if abs(book["qty"]) > 1e-12 and not is_dust(book):
                return True
        return False

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
                     timing: dict | None = None, reservation: int | None = None,
                     parent_trade_id: str | None = None) -> None:
        """timing: the decision's stamps (bar_close, bar_recv, decided, as UNIX ns), written as the order's
        order_timings row in the same transaction, so that row never exists without its order (DA-8).
        reservation: the portfolio gate's reservation this entry holds (its gate_decisions id), linked to the order
        (gate_decisions.order_id) in the same transaction, so the order's fills and end settle it (P2-1b W2).
        parent_trade_id: the two-leg trade this order belongs to, the same id on both legs' rows (P2-8)."""
        if intent not in INTENTS:
            raise ValueError(f"bad intent {intent!r}")
        now = ts or utcnow()
        with self.engine.begin() as c:
            # The venue is the router's, never the caller's (P2-4): read with the order, in its transaction.
            venue = venue_for(c.execute(select(sleeve_venues_t.c.venue).where(sleeve_venues_t.c.sleeve == sleeve))
                              .scalar())
            # F-WED1-3 (DA): an open position on another venue than the pin keeps its own. Its exits are journaled
            # there, where they go; an opening order is refused, behind the start's entry hold. Same transaction, so
            # a fill can't land between the two reads.
            held = _position_venue(c, sleeve)
            held = None if held is None else venue_for(held)  # a run's clone is its base venue, as router.start_mode
            if held is not None and held != venue:
                if intent in GATED_INTENTS:
                    raise ValueError(f"{sleeve}: an opening order is refused while its open position is on another "
                                     "venue than its pin")
                venue = held
            c.execute(insert(orders_t).values(sleeve=sleeve, order_id=order_id, ts=now, updated_at=now, side=side,
                                              order_type=order_type, qty=qty, status="submitted", filled_qty=0.0,
                                              fee=0.0, intent=intent, reason=reason, signal=signal or {},
                                              message="", venue=venue, parent_trade_id=parent_trade_id))
            if timing:
                c.execute(insert(order_timings_t).values(order_id=order_id, sleeve=sleeve, **{
                    k: _from_ns(v) for k, v in timing.items() if v is not None}))
            if reservation is not None:
                with self._gate_ledger().using(c):
                    self._gate_ledger().attach_order(reservation, order_id)

    def position_venue(self, sleeve: str) -> str | None:
        """The venue the strategy's open position is on, as the journal holds it (_position_venue); None when flat."""
        with self.engine.connect() as c:
            return _position_venue(c, sleeve)

    def traded_venue(self, sleeve: str) -> str | None:
        """The venue the strategy last traded on, as the journal holds it (_traded_venue); None with no fills."""
        with self.engine.connect() as c:
            return _traded_venue(c, sleeve)

    def venue_start(self, sleeve) -> tuple[str, str | None]:
        """How `sleeve` (a Sleeve) may start, by the venue its journal last traded on against its pin
        (router.start_mode), as (mode, that venue). Holding means a position no flatten leaves as dust (is_dust), as
        the supervisor's exits-only start reads it. One answer for the supervisor and the paper process (F-WED1-3)."""
        traded = self.traded_venue(sleeve.name)
        if traded is None:
            return router.NORMAL, None
        book = self.journal_book(sleeve.name, sleeve.starting_balance)
        held = holding(book["qty"]) and not is_dust(book)  # a dust holder off its pin is refused, as a flat one
        return router.start_mode(venue_for(sleeve.venue), traded, held), traded

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

    def _gate_ledger(self):
        """The portfolio gate's journal half, for settling a reservation inside an order's own transaction. One per
        store: DbLedger.using() is per thread. Imported here: gate_ledger imports this module."""
        led = self.__dict__.get("_settle_ledger")
        if led is None:
            from sleeve_fund.gate_ledger import DbLedger

            def no_book():
                raise RuntimeError("settling a reservation never reads the book")
            led = self.__dict__["_settle_ledger"] = DbLedger(self, no_book)
        return led

    def put_away_left_open(self) -> list[str]:
        """F-RRP-5 (CR; DA): the put-away Strategies still in the book (archived, not withdrawn) whose journal shows an
        order open or whose room is still held: the only ones Supervisor.close_dead_orders has anything to close for.
        One query, however many runs the resets have put away."""
        a, o, r, s = sleeve_archive_t, orders_t, gate_reservations_t, sleeves_t
        left_open = or_(exists().where(o.c.sleeve == a.c.sleeve, o.c.status.in_(OPEN_ORDER_STATUSES)),
                        exists().where(r.c.sleeve_id == s.c.id, r.c.released_at.is_(None)))
        with self.engine.connect() as c:
            return list(c.execute(select(a.c.sleeve).join(s, s.c.name == a.c.sleeve)
                                  .where(a.c.withdrawn.is_(None), left_open).order_by(a.c.sleeve)).scalars())

    def close_dead_orders(self, sleeve: str, why: str, ts: datetime | None = None) -> list[str]:
        """F-REFRESH-RES (HoE, Drop A SHOULD): a paper Strategy's process is confirmed stopped, so the simulated venue
        its orders rested at went with it and none of them can fill. Each order the journal still shows open is
        closed as canceled with `why` (its reservation released as 'cancel' with it, as any cancel's is), then every
        reservation of the Strategy still active (one holding no order, or one an order left behind) is released as
        'cancel' too, in one transaction with one event. Call it only after the process has exited (the caller checks
        the process is gone). Refused for a live account: a venue order outlives the process. Returns the orders
        closed."""
        ts = ts or utcnow()
        r = gate_reservations_t
        held: list[str] = []  # a second leg's retry hold dies with its process too; put back if nothing commits
        try:
            with self.engine.begin() as c:
                account = c.execute(select(sleeve_accounts_t.c.account)
                                    .where(sleeve_accounts_t.c.sleeve == sleeve)).scalar() or "paper"
                if account != "paper":
                    raise ValueError(f"{sleeve} trades on a live account: its orders outlive its process, so none is "
                                     "closed")
                open_ids = [o for (o,) in c.execute(select(orders_t.c.order_id).where(
                    orders_t.c.sleeve == sleeve, orders_t.c.status.in_(OPEN_ORDER_STATUSES)).order_by(orders_t.c.id))]
                held = [o for o in open_ids if o in self._retry_held]
                self._retry_held.difference_update(held)  # so each cancel below releases its room
                for oid in open_ids:
                    self._update_order(c, oid, status="canceled", message=f"closed: {why}")
                sid = c.execute(select(sleeves_t.c.id).where(sleeves_t.c.name == sleeve)).scalar()
                left = c.execute(update(r).where(r.c.sleeve_id == sid, r.c.released_at.is_(None))
                                 .values(released_at=ts, release_reason="cancel")).rowcount
                if open_ids or left:
                    c.execute(insert(events_t).values(
                        sleeve=sleeve, ts=ts, level="info", kind="orders_closed_stopped",
                        message=f"{why}: its process has stopped, so {len(open_ids)} order(s) still shown open can't "
                                "fill and are closed, and the room they or the Strategy held in the book is released"))
        except Exception:
            self._retry_held.update(held)
            raise
        return open_ids

    def _settle_reservation(self, c, order_id: str, fill_qty, status: str | None) -> None:
        """An entry's fills and end settle the reservation it holds, in the order's own transaction (HoQA w2b, w2c):
        a fill reduces it (filled to nothing, it is released as 'fill'); a filled order releases what is left as
        'fill', a rejected or denied one as 'reject', a cancelled or expired one as 'cancel' (a part-filled one keeps
        what filled: its position is in the book). A failed write raises, so the fill or status isn't written either."""
        d, r = gate_decisions_t, gate_reservations_t  # the decision that reserved (a fill-stage check reserves nothing)
        rid = c.execute(select(d.c.id).join(r, r.c.decision_id == d.c.id).where(d.c.order_id == order_id)).scalar()
        if rid is None:
            return
        led = self._gate_ledger()
        with led.using(c):
            if fill_qty:
                led.reduce(rid, fill_qty)
            reason = {"filled": "fill", "rejected": "reject", "denied": "reject", "unfilled_unknown": "reject",
                      "canceled": "cancel", "expired": "cancel"}.get(status)
            if reason in ("reject", "cancel") and order_id in self._retry_held:
                reason = None  # a two-leg Strategy's second leg: what is left stays reserved for its retry (hand_on)
            if reason is not None:
                led.release(rid, reason)
                self._retry_held.discard(order_id)

    # --- a two-leg Strategy's second-leg retry (P2-8; Advisor 14 Oct 05:12 UK, MUST) ---------------------------------
    # The second leg's share, reserved with the first leg's by the pair's one check, stays reserved until it fills, the
    # pair unwinds or the retry deadline passes: a retry draws on what is left and is never checked alone (a spurious
    # refusal would force the unwind the reservation exists to avoid). The hold is the order's own writer's, in memory:
    # a restart cancels the order, which then releases it as any other's.

    @property
    def _retry_held(self) -> set:
        return self.__dict__.setdefault("_retry_held_orders", set())

    def hold_for_retry(self, order_id: str) -> None:
        """The order's reject, cancel or expiry keeps what is left of its reservation (its fills still reduce it)."""
        self._retry_held.add(order_id)

    def hand_on(self, order_id: str) -> tuple[int, Decimal] | None:
        """What is left of the reservation `order_id` held, now held by no order, for the retry to attach: (its
        reservation id, remaining quantity), or None when nothing is left (filled, or released)."""
        d, r = gate_decisions_t, gate_reservations_t
        with self.engine.begin() as c:
            row = c.execute(select(r.c.decision_id, r.c.remaining_qty).join(d, d.c.id == r.c.decision_id)
                            .where(d.c.order_id == order_id, r.c.released_at.is_(None)).with_for_update()).first()
            if row is None:
                self._retry_held.discard(order_id)
                return None
            c.execute(update(d).where(d.c.id == row.decision_id).values(order_id=None))
        self._retry_held.discard(order_id)
        return row.decision_id, to_decimal(row.remaining_qty)

    def release_reservation(self, reservation_id: int, reason: str = "cancel") -> None:
        """A handed-on reservation no retry took up: back to the book."""
        self._gate_ledger().release(reservation_id, reason)

    def release_held(self, order_id: str, reason: str = "cancel") -> None:
        """The pair unwound or its retry deadline passed: what `order_id` still holds goes back to the book."""
        self._retry_held.discard(order_id)
        d, r = gate_decisions_t, gate_reservations_t
        with self.engine.connect() as c:
            rid = c.execute(select(d.c.id).join(r, r.c.decision_id == d.c.id)
                            .where(d.c.order_id == order_id, r.c.released_at.is_(None))).scalar()
        if rid is not None:
            self.release_reservation(rid, reason)

    def _update_order(self, c, order_id: str, *, status: str | None = None, message: str | None = None,
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
        if row.intent in GATED_INTENTS and (fill_qty or values.get("status", row.status) != row.status):
            self._settle_reservation(c, order_id, fill_qty, values.get("status"))

    def rebook_liquidation(self, order_id: str, reason: str, signal: dict, ts: datetime | None = None) -> None:
        """GAP-LIQ (Advisor): a resting stop whose fill was at or past the liquidation price is re-booked as the
        liquidation it was; since F283-P1 any close filled there is (REBOOKABLE). The only intent change the journal
        allows (Data Architect): a close to liquidation, on an order that has filled; its fills stay as they are, and an order_rebooked event keeps the lineage."""
        with self.engine.begin() as c:
            row = c.execute(select(orders_t).where(orders_t.c.order_id == order_id)).first()
            _check_rebook(order_id, row and row._mapping)
            c.execute(update(orders_t).where(orders_t.c.order_id == order_id)
                      .values(intent="liquidation", reason=reason, signal=signal, updated_at=utcnow()))
        self.event(row.sleeve, "info", "order_rebooked", _rebook_words(order_id, row.intent, reason), ts=ts)
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

    def closing_orders(self, sleeve: str, since: datetime | None = None) -> list[dict]:
        """F-RX-1: a Strategy's closing orders (CLOSING_INTENTS) sent at or after since, oldest first, each with its
        status, the venue's or risk engine's words (message) and when it last moved (updated_at). Every one is
        listed, refused or not, so a reader counts refused closes in a row from the tail (F-RX-2)."""
        o = orders_t.c
        q = (select(o.order_id, o.intent, o.status, o.message, o.ts, o.updated_at, o.instrument)
             .where(o.sleeve == sleeve, o.intent.in_(CLOSING_INTENTS)))
        if since is not None:
            q = q.where(o.ts >= since)
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(o.ts, o.id)))

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
        """A move onto or off a test account its venue names moves the Strategy's equity out of or into the fund's
        book (F229-6), with its flow (HALT-REF-1).

        F-ASSIGN-STALE (Advisor ADV-ASSIGN, 14 Oct 17:41 UK): on a marked book only the supervisor's poll (land_move),
        after its own mark, moves capital, so P&L made since the last mark is judged at the book's scale first. Asked for
        anywhere else, a move that would flow waits (book_moves) and the Strategy stays on its old account and its old
        side of the book until the poll lands it (land_move); a later request replaces it. One whose create is still
        waiting takes the new account as where it joins. A move that flows nothing is made at once, as is any move on
        a book never marked."""
        ts = utcnow()
        joining = self.book_moves_pending().get(sleeve, {}).get("kind") == "create"
        moved, waits = [], []

        def amount() -> Decimal:
            if is_backtest(sleeve) or sleeve in self.withdrawn():  # out of the book already (F-C3B-A)
                return Decimal(0)
            s = self.sleeve(sleeve)
            if joining:
                return Decimal(1)  # its create waits: where it joins is this account, landed by the poll
            was = _outside_book(self, s)
            if was == _outside_book(self, s, account):
                return Decimal(0)
            from sleeve_fund.supervisor import book_part

            moved.append("into" if was else "out of")
            part = book_part(self, s)
            return part if was else -part

        def wait(c) -> None:
            self._wait_move(c, sleeve, account, "create" if joining else "move", ts)

        with self._book_move(amount, lambda: f"{sleeve} moved {moved[0]} the fund's book (account {account})",
                             defer=wait, waits_out=waits) as c:
            self._ensure_paper_account(c)
            if c.execute(select(accounts_t.c.name).where(accounts_t.c.name == account)).first() is None:
                raise ValueError(f"no account called {account}")
            if not waits[0]:
                c.execute(sleeve_accounts_t.delete().where(sleeve_accounts_t.c.sleeve == sleeve))
                c.execute(insert(sleeve_accounts_t).values(sleeve=sleeve, account=account, assigned_at=ts))
                # a move made at once (no flow: back to the side it is on) replaces any that waited (DA 0024 rule 2)
                self._end_waiting(c, sleeve, ts, f"a move to account {account} on the same side of the book was "
                                                 "made at once")

    # --- F-ASSIGN-STALE: moves waiting for the supervisor's poll -------------------------------------------------

    def _end_waiting(self, c, sleeve: str, ts: datetime, why: str | None = None) -> list:
        """Supersede `sleeve`'s waiting move, if any, journaled when `why` says what replaced it; the rows superseded.
        Each row ends once (DA 0024 rule 1): a row another writer ended first is left as it is."""
        rows = c.execute(select(book_moves_t).where(book_moves_t.c.sleeve == sleeve, book_moves_t.c.landed_at.is_(None),
                                                    book_moves_t.c.superseded_at.is_(None))).all()
        ended = [r for r in rows if _end_move(c, r, superseded_at=ts)]
        for r in ended if why else ():
            c.execute(insert(events_t).values(
                sleeve=sleeve, ts=ts, level="info", kind=BOOK_MOVE_SUPERSEDED,
                message=f"{sleeve}'s move to account {r.account}, waiting for the supervisor's next mark, is "
                        f"cancelled: {why}"))
        return ended

    def _wait_move(self, c, sleeve: str, account: str, kind: str, ts: datetime) -> None:
        """Record `sleeve`'s move to `account` as waiting for the poll (latest wins), journaled."""
        before = self._end_waiting(c, sleeve, ts, f"a later request moves it to account {account}")
        c.execute(insert(book_moves_t).values(sleeve=sleeve, account=account, kind=kind, requested_at=ts))
        what = (f"{sleeve} joins the fund's book on account {account}" if kind == "create"
                else f"{sleeve} moves to account {account}")
        was = f" (replacing its move to account {before[-1].account})" if before and before[-1].account != account else ""
        c.execute(insert(events_t).values(
            sleeve=sleeve, ts=ts, level="info", kind=BOOK_MOVE_WAITING,
            message=f"{what} at the supervisor's next mark{was}, so what the book made since its last mark counts "
                    "first; its entries are refused until then"))

    def book_moves_pending(self) -> dict[str, dict]:
        """Each Strategy's waiting move: {name: {"id", "account", "kind", "requested_at"}}, oldest first."""
        q = (select(book_moves_t).where(book_moves_t.c.landed_at.is_(None), book_moves_t.c.superseded_at.is_(None))
             .order_by(book_moves_t.c.id))
        with self.engine.connect() as c:
            return {r.sleeve: {"id": r.id, "account": r.account, "kind": r.kind, "requested_at": _aware(r.requested_at)}
                    for r in c.execute(q)}

    def joining(self, sleeve: str) -> bool:
        """Whether `sleeve` was created on a marked book and its joining deposit still waits for the poll."""
        with self.engine.connect() as c:
            return c.execute(select(book_moves_t.c.id).where(
                book_moves_t.c.sleeve == sleeve, book_moves_t.c.kind == "create", book_moves_t.c.landed_at.is_(None),
                book_moves_t.c.superseded_at.is_(None))).first() is not None

    def land_move(self, sleeve: str, ts: datetime | None = None) -> bool:
        """The supervisor's poll, after its mark: land `sleeve`'s waiting move with its flow at the fresh scale, in one
        transaction with the account row and the landing (Advisor ADV-ASSIGN; DA 0024 rule 3). A create joins the book
        with its equity; a move goes as assign_account would. A two-leg Strategy's create lands both legs together, on
        its first leg's call, with one flow of their sum; its second leg's own call leaves it to the first. False when
        nothing (of its own) waits."""
        from sleeve_fund.supervisor import book_part

        ts = ts or utcnow()
        got, moved, gone, on = [], [], [], {}

        def waiting(r, name: str):
            return r.execute(select(book_moves_t).where(
                book_moves_t.c.sleeve == name, book_moves_t.c.landed_at.is_(None),
                book_moves_t.c.superseded_at.is_(None))).first()

        def amount() -> Decimal:
            with self.engine.connect() as r:
                row = waiting(r, sleeve)
                first = self.sleeve(sleeve).parent if row is not None else None
                if row is not None and row.kind == "create" and first is not None and waiting(r, first) is not None:
                    return Decimal(0)  # the second leg of a two-leg create: its first leg lands both (got stays empty)
                rows = [row] if row is not None else []
                if row is not None and row.kind == "create":
                    rows += [w for leg in self.legs(sleeve) if leg != sleeve and (w := waiting(r, leg)) is not None
                             and w.kind == "create"]
            got.extend(rows)
            # Read here, before this move writes (F286-ENG: on Store.in_memory a read after the writes loses them).
            on.update({w.sleeve: self.account_of(w.sleeve) for w in rows})
            if not rows:
                return Decimal(0)
            if sleeve in self.withdrawn() or sleeve in self.archived():
                gone.append(True)  # put away while it waited (a put-away supersedes it, DA rule 4): it never lands
                return Decimal(0)
            total = Decimal(0)
            for w in rows:
                s = self.sleeve(w.sleeve)
                into = not _outside_book(self, s, w.account)
                was = True if w.kind == "create" else _outside_book(self, s)
                if was == (not into):
                    continue
                moved.append("joins" if w.kind == "create" else "moved into" if was else "moved out of")
                part = book_part(self, s)
                total += part if was else -part
            return total

        def said() -> str:
            names = " and ".join(w.sleeve for w in got)
            verb = "join" if len(moved) > 1 else moved[0]  # a two-leg create's legs join together
            return f"{names} {verb} the fund's book (account {got[0].account}) at the supervisor's mark"

        with self._book_move(amount, said, ts) as c:
            if not got:
                return False
            for row in got:
                if not _end_move(c, row, **({"superseded_at": ts} if gone else {"landed_at": ts})):
                    raise RuntimeError(f"{row.sleeve}'s move to account {row.account} had already ended, so it isn't "
                                       "landed twice")
                if gone:
                    c.execute(insert(events_t).values(
                        sleeve=row.sleeve, ts=ts, level="info", kind=BOOK_MOVE_SUPERSEDED,
                        message=f"{row.sleeve} was put away while its move to account {row.account} waited, so it "
                                "never lands"))
                    continue
                if not (row.kind == "create" and row.account == "paper" and on[row.sleeve] == "paper"):
                    c.execute(sleeve_accounts_t.delete().where(sleeve_accounts_t.c.sleeve == row.sleeve))
                    c.execute(insert(sleeve_accounts_t).values(sleeve=row.sleeve, account=row.account, assigned_at=ts))
                c.execute(insert(events_t).values(
                    sleeve=row.sleeve, ts=ts, level="info", kind=BOOK_MOVE_LANDED,
                    message=(f"{row.sleeve} joined the fund's book" if row.kind == "create"
                             else f"{row.sleeve} moved to account {row.account}") + " at the supervisor's mark"))
        return True

    def archived(self) -> dict[str, datetime]:
        with self.engine.connect() as c:
            return {r.sleeve: _aware(r.archived_at) for r in c.execute(select(sleeve_archive_t))}

    def withdrawn(self) -> dict[str, Decimal]:
        """Strategies put away whose equity left the fund's book with its flow (F-C3B-A): {name: the equity taken out}.
        fund_equity counts none of them."""
        with self.engine.connect() as c:
            return {r.sleeve: to_decimal(r.withdrawn) for r in c.execute(
                select(sleeve_archive_t).where(sleeve_archive_t.c.withdrawn.is_not(None)))}

    @contextmanager
    def _put_away(self, sleeve: str, ts: datetime, why: str, polled: bool = False):
        """F-C3B-A (Advisor 14 Oct 09:45 UK): a Strategy put away for good (archived, retired, or archived by a clean
        slate) takes its whole equity out of the fund's book, as a flow (_book_move), in the same transaction as its
        archive row, which this writes; the caller's own writes go on the connection this yields. One outside the book
        (a test account, a saved backtest) takes nothing out, as it brought nothing in.

        Not settled flat yet (withdrawal_blocker: a position, an open order, room an order holds), it takes nothing
        out now: it stays in the book in full until it is flat (F229-6 C3), journaled as WITHDRAWAL_DEFERRED, and the
        supervisor's poll withdraws it then, after that poll's mark (Advisor 14 Oct 10:45 UK). A settled equity of 0 or
        less is never withdrawn: it stays in the book, journaled as WITHDRAWAL_NONE. Each is decided under the gate's
        lock, so the poll and withdraw-archived racing take it out once.

        Only the supervisor's poll (polled), after its own mark, takes equity out of a marked book (F286-CR1, HoE on the
        Advisor's option (b)): put away anywhere else, flat or not, it is WITHDRAWAL_DEFERRED, so the P&L made since the
        last mark is judged at the book's scale before its equity leaves, never landing on the Strategies that remain.
        A book never marked has no scale to keep: it is taken out at once, with no flow.

        Once the fund holds the cash (Bridge Drop A) a Strategy has no capital of its own: settled flat, it leaves the
        book with withdrawn = 0 and an info line (PUT_AWAY_FUND_HELD), never WITHDRAWAL_NONE (F-WN-1, HoE + DA)."""
        if self.two_leg(sleeve):  # the backstop behind every put-away: never one leg alone (TWO_LEG_FENCE)
            raise ValueError(f"putting a Strategy away {TWO_LEG_FENCE}")
        seen: list[tuple[str, Decimal | str]] = []
        dust_qty: list[Decimal] = []

        def decide(marked: bool) -> tuple[str, Decimal | str]:
            if sleeve in self.withdrawn():
                return "out already", Decimal(0)
            if self._outside_or_backtest(sleeve):
                return "out", Decimal(0)  # it brought nothing into the book
            blocked = self.withdrawal_blocker(sleeve)
            if blocked is not None:
                return "deferred", blocked
            if _fund_held(self):  # F-WN-1 (DA): nothing of its own moves, so no book scale to wait for; out at 0 now
                equity = self.in_book_equity(sleeve)
                # Anything but exactly 0 here is a broken invariant (book_part is 0 once the fund holds the cash): it
                # stays in the book and warns, never written off as nothing moved.
                return ("held", Decimal(0)) if equity == 0 else ("none", equity)
            if marked and not polled:
                return "deferred", NEXT_MARK
            equity = self.in_book_equity(sleeve)
            return ("out", equity) if equity > 0 else ("none", equity)

        def amount() -> Decimal:
            seen.append(decide(True))
            # Read here, before this move writes: on a store sharing one connection (Store.in_memory), a read after the
            # writes ends the lock's transaction and loses them (F286-ENG).
            dust_qty.append(self.lifecycle_flat(sleeve)[0])
            kind, value = seen[0]
            return -value if kind == "out" else Decimal(0)

        def said() -> str:
            qty = dust_qty[0]
            dust = f", with dust of {qty_words(qty)} no order can close, which leaves with it" if qty else ""
            return f"{sleeve} put away ({why}): its equity {seen[0][1]:,.2f} leaves the fund's book{dust}"

        # A clean slate's put-away (previous_book) starts a fresh book, so its emptying carries nothing (MUST-F286-H4)
        with self._book_move(amount, said, ts, leaving=sleeve, carry=sleeve not in self.previous_book()) as c:
            yield c
            # A create or move still waiting never lands for a Strategy put away (DA 0024 rule 4, F-ASSIGN-STALE)
            self._end_waiting(c, sleeve, ts, f"it was put away ({why})")
            # Put away again (archived, then retired; a second archive): already out, so it keeps its row, its
            # amount and when it went, with no second flow (DA, 14 Oct).
            if c.execute(select(sleeve_archive_t.c.sleeve).where(sleeve_archive_t.c.sleeve == sleeve,
                                                                 sleeve_archive_t.c.withdrawn.is_not(None))).first():
                return
            kind, value = seen[0] if seen else decide(False)  # nothing marked yet: no flow, but decided all the same
            out = stored(value) if kind in ("out", "held") else None
            # An archive from before 0023 (withdraw-archived), or one waiting to go flat, keeps when it went:
            # previous_book reads archived_at against the clean slate's start (DA, 14 Oct). A fresh one is written now.
            if not c.execute(update(sleeve_archive_t).where(sleeve_archive_t.c.sleeve == sleeve)
                             .values(withdrawn=out)).rowcount:
                c.execute(insert(sleeve_archive_t).values(sleeve=sleeve, archived_at=ts, withdrawn=out))
            if kind == "deferred" and value == NEXT_MARK:
                c.execute(insert(events_t).values(
                    sleeve=sleeve, ts=ts, level="info", kind=WITHDRAWAL_DEFERRED,
                    message=f"{sleeve} put away ({why}): it stays in the fund's book until the supervisor's next mark, "
                            "and its equity leaves the book then, so what it made since the last mark counts first"))
            elif kind == "deferred":
                c.execute(insert(events_t).values(
                    sleeve=sleeve, ts=ts, level="info", kind=WITHDRAWAL_DEFERRED,
                    message=f"{sleeve} put away ({why}) while not yet flat ({value}): it stays in the fund's book in "
                            "full, exiting but never entering, and its settled equity leaves the book when it is flat"))
            elif kind == "held":
                c.execute(insert(events_t).values(
                    sleeve=sleeve, ts=ts, level="info", kind=PUT_AWAY_FUND_HELD,
                    message=f"{sleeve} put away ({why}), settled flat: the fund holds the cash, so nothing moves and "
                            "what it made stays in the fund's book"))
            elif kind == "none":
                c.execute(insert(events_t).values(
                    sleeve=sleeve, ts=ts, level="warning", kind=WITHDRAWAL_NONE,
                    message=f"{sleeve} put away ({why}) with a settled equity of {value:,.2f}: nothing is withdrawn, "
                            "as a withdrawal is never negative, and it stays in the fund's book at that"))

    def _outside_or_backtest(self, sleeve: str) -> bool:
        return is_backtest(sleeve) or _outside_book(self, self.sleeve(sleeve))

    def withdrawal_blocker(self, sleeve: str) -> str | None:
        """Why `sleeve` isn't settled flat for its withdrawal, or None (Advisor 14 Oct 10:45 UK): a position (dust no
        venue takes an order for counts as flat, as for a retire: lifecycle_flat), an order still open, or room an
        order still holds in the portfolio gate. Fills, fees and funding are booked with the fill, so a flat journal
        with no open order has them all."""
        qty, flat = self.lifecycle_flat(sleeve)
        if not flat:
            return f"it holds {float(qty):g}"
        if self.orders(sleeve, statuses=OPEN_ORDER_STATUSES, limit=1):
            return "an order of it is still open"
        if any(r.strategy == sleeve for r in self._gate_ledger().reservations()):
            return "an order of it still holds room in the book"
        return None

    def withdrawal_pending(self) -> list[str]:
        """Strategies archived while not flat whose equity waits to leave the book (WITHDRAWAL_DEFERRED, and no
        WITHDRAWAL_NONE since): the supervisor's poll withdraws each once it is flat. The dashboard shows them as
        archived, waiting to go flat. Every archive since F-C3B-A either withdraws (withdrawn) or journals one of the
        two, so the latest of them is the current archive's, whatever clocks wrote them."""
        archived, out = self.archived(), self.withdrawn()
        last: dict[str, tuple] = {}
        for e in self.events_of((WITHDRAWAL_DEFERRED, WITHDRAWAL_NONE), limit=10_000):  # newest first
            last.setdefault(e["sleeve"], (e["kind"], e["ts"]))
        return [n for n, (kind, _) in last.items() if kind == WITHDRAWAL_DEFERRED and n in archived and n not in out]

    def nothing_withdrawn(self) -> dict[str, Decimal]:
        """Strategies archived with a settled equity of 0 or less, so nothing left the book (WITHDRAWAL_NONE since
        their archive): {name: that equity, as the book still counts it}. For the dashboard's "archived, nothing to
        withdraw" (Advisor 14 Oct 11:10 UK)."""
        archived, out = self.archived(), self.withdrawn()
        last: dict[str, tuple] = {}
        for e in self.events_of((WITHDRAWAL_DEFERRED, WITHDRAWAL_NONE), limit=10_000):  # newest first
            last.setdefault(e["sleeve"], (e["kind"], e["ts"]))
        return {n: self.in_book_equity(n) for n, (kind, _) in last.items()
                if kind == WITHDRAWAL_NONE and n in archived and n not in out}

    def in_book_equity(self, sleeve: str) -> Decimal:
        """`sleeve`'s equity as the fund's book counts it (supervisor.strategy_equity), or nothing when it sits outside
        the book (a saved backtest, a test account, F229-6)."""
        s = self.sleeve(sleeve)
        if is_backtest(sleeve) or _outside_book(self, s):
            return Decimal(0)
        from sleeve_fund.supervisor import book_part

        return book_part(self, s)

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
        """Put a stopped, flat sleeve away, its equity out of the fund's book with its flow (F-C3B-A). Raises
        ValueError if it is running or still holds a position."""
        s = self.sleeve(sleeve)
        if s.desired_state != "stopped":
            raise ValueError("stop the strategy before archiving it")
        if self.two_leg(sleeve):
            raise ValueError(f"archiving {TWO_LEG_FENCE}")
        with self._put_away(sleeve, utcnow(), "archived"):
            pass

    def unarchive(self, sleeve: str) -> None:
        """Bring a put-away sleeve back: the equity it took out comes back into the book with its flow, at what it is
        worth now (F-C3B-A, the undo of archive)."""
        if self.sleeve(sleeve).lifecycle == lifecycle.RETIRED:
            raise ValueError("this strategy is retired, and a retired strategy can't be brought back")
        if self.two_leg(sleeve):  # TWO_LEG_FENCE
            raise ValueError(f"putting a Strategy away {TWO_LEG_FENCE}")
        back: list[Decimal] = []

        def amount() -> Decimal:
            if sleeve not in self.withdrawn():
                return Decimal(0)  # an archive from before F-C3B-A: it never left the book
            back.append(self.in_book_equity(sleeve))
            return back[0]

        with self._book_move(amount, lambda: f"{sleeve} brought back: its equity {back[0]:,.2f} comes into the fund's "
                                             "book") as c:
            c.execute(sleeve_archive_t.delete().where(sleeve_archive_t.c.sleeve == sleeve))

    # --- lifecycle (P2-6) ------------------------------------------------------------

    def held_qty(self, sleeve: str) -> Decimal:
        """The position the journal's fills add up to (the figure the flatten and archive checks read)."""
        return to_decimal(self.journal_book(sleeve, self.sleeve(sleeve).starting_balance)["qty"])

    def lifecycle_flat(self, sleeve: str) -> tuple[Decimal, bool]:
        """(the position the journal holds, whether the lifecycle counts it flat): none, or dust too small for any
        venue to take an order for (is_dust, as a reset judges it). The engine reads below its smallest order as flat,
        so a remainder no flatten can close must never keep a strategy from retiring (HoQA P26-1)."""
        book = self.journal_book(sleeve, self.sleeve(sleeve).starting_balance)
        return to_decimal(book["qty"]), abs(book["qty"]) <= 1e-12 or is_dust(book)

    def lifecycle_holds(self, sleeve: str) -> tuple[list[timedelta], list[timedelta], str]:
        """The holding times a wind-down's deadline is set from, as (backtest, paper, where from). Paper: the
        strategy's own closed round trips in the journal. Backtest: none yet, as no saved backtest is linked to the
        strategy it came from, so lifecycle.window falls back to the longest hold (fewer than 20 backtest trades)."""
        from sleeve_fund import markets
        from sleeve_fund.research.metrics import trades

        rows = list(reversed(self.fills(sleeve, limit=10_000_000)))
        paper = [t["closed"] - t["opened"] for t in trades(rows, shorts=markets.is_perp(self.sleeve(sleeve).params))
                 if t["opened"] and t["closed"]]
        return [], paper, f"no linked backtest; {len(paper)} closed paper trade{'' if len(paper) == 1 else 's'}"

    def set_lifecycle(self, sleeve: str, action: str, actor: str, reason: str, now: datetime | None = None) -> Sleeve:
        """The PM's lifecycle move, and the one writer of it (the dashboard's lifecycle route is its one caller).
        action: wind_down, retire or close_now; the rules are lifecycle.move and lifecycle.close_now, which refuse
        anyone but the PM. Writes the row, a decision and a lifecycle event in one transaction; a strategy retired
        here is also stopped and archived, so every reader of the archive sees it put away."""
        if action not in LIFECYCLE_ACTIONS:
            raise ValueError(f"bad lifecycle action {action!r}")
        if is_backtest(sleeve):
            raise ValueError("a saved backtest has no lifecycle")
        now = now or utcnow()
        s = self.sleeve(sleeve)
        two_leg = self.two_leg(sleeve)
        if two_leg and (action != "close_now" or s.parent is not None):
            raise ValueError(f"{action.replace('_', ' ')} on its second leg {TWO_LEG_FENCE}" if action == "close_now"
                             else f"{action.replace('_', ' ')} {TWO_LEG_FENCE}")
        multi = self.multi_asset(sleeve)
        if multi and s.parent is not None:
            raise ValueError(f"{sleeve} is an asset leg of {s.parent}: a lifecycle move is on the Strategy, every leg")
        names = self.legs(sleeve) if multi else [sleeve]
        held, flat = self.lifecycle_flat(sleeve)
        flat = all(self.lifecycle_flat(n)[1] for n in names) if multi else flat  # retired once every leg is flat
        started, deadline_at, basis = s.wind_down_started, s.wind_down_deadline, ""
        if action == "wind_down":
            to = lifecycle.move(s.lifecycle, lifecycle.WINDING_DOWN, actor, flat=flat)
            backtest, paper, basis = self.lifecycle_holds(sleeve)
            started, deadline_at = now, lifecycle.deadline(now, backtest, paper)
            basis = f" Window {lifecycle.span(lifecycle.window(backtest, paper))}, from the holds: {basis}."
        elif action == "retire":
            to = lifecycle.move(s.lifecycle, lifecycle.RETIRED, actor, flat=flat)
        else:
            m = lifecycle.close_now(s.lifecycle, actor, now, flat=flat)
            to = m.lifecycle
            if to == lifecycle.RETIRED and two_leg:
                # Close-now is never refused (Advisor): flat, a two-leg Strategy still only winds down, due now, so it
                # opens nothing; its retire waits for the put-away design (TWO_LEG_FENCE).
                to = lifecycle.WINDING_DOWN
            if to == lifecycle.WINDING_DOWN:
                started, deadline_at = started or now, now
        text = lifecycle.moved_words(s.lifecycle, to, action, deadline_at, held) + basis
        if multi:
            self._lifecycle_legs(names, s.lifecycle, to, started, deadline_at, now, action, reason, text)
            return self.sleeve(sleeve)
        # A retire puts it away for good: its equity leaves the book with the lifecycle row (F-C3B-A).
        with (self._put_away(sleeve, now, "retired") if to == lifecycle.RETIRED else self.engine.begin()) as c:
            values = {"lifecycle": to, "wind_down_started": started, "wind_down_deadline": deadline_at,
                      "updated_at": now}
            if to == lifecycle.RETIRED:
                values["desired_state"] = "stopped"
            # Only over the lifecycle this move was judged on: a retire the runtime completed meanwhile is never
            # written back to winding down (CR F26-A); the PM sees the refusal and the row as it now stands.
            done = c.execute(update(sleeves_t).where(sleeves_t.c.name == sleeve,
                                                     sleeves_t.c.lifecycle == s.lifecycle).values(**values)).rowcount
            if not done:
                raise lifecycle.LifecycleError("This strategy's lifecycle changed meanwhile; look again and retry.")
            c.execute(insert(decisions_t).values(ts=now, actor="PM", action=action, sleeve=sleeve,
                                                 reason=reason.strip()))
            c.execute(insert(events_t).values(sleeve=sleeve, ts=now, level="info", kind=LIFECYCLE_EVENT, message=text))
        return self.sleeve(sleeve)

    def _lifecycle_legs(self, names: list[str], was: str, to: str, started, deadline_at, now: datetime, action: str,
                        reason: str, text: str) -> None:
        """A multi-asset Strategy's lifecycle move, on every leg (names: the parent first). A move short of retired is one
        transaction over every leg. A retire puts each leg away (F-C3B-A: its own flow under the gate's lock, as any
        put-away), the asset legs first and the parent last, each with its row; one that fails leaves the rest winding
        down and flat, and complete_retire finishes them. Journaled once, on the parent."""
        values = {"lifecycle": to, "wind_down_started": started, "wind_down_deadline": deadline_at, "updated_at": now}
        parent = names[0]

        def journal(c) -> None:
            c.execute(insert(decisions_t).values(ts=now, actor="PM", action=action, sleeve=parent, reason=reason.strip()))
            c.execute(insert(events_t).values(sleeve=parent, ts=now, level="info", kind=LIFECYCLE_EVENT, message=text))

        if to != lifecycle.RETIRED:
            with self.engine.begin() as c:
                done = c.execute(update(sleeves_t).where(sleeves_t.c.name.in_(names), sleeves_t.c.lifecycle == was)
                                 .values(**values)).rowcount
                if done != len(names):
                    raise lifecycle.LifecycleError("This strategy's lifecycle changed meanwhile; look again and retry.")
                journal(c)
            return
        values["desired_state"] = "stopped"
        for name in names[1:] + names[:1]:
            with self._put_away(name, now, "retired") as c:
                c.execute(update(sleeves_t).where(sleeves_t.c.name == name).values(**values))
                if name == parent:
                    journal(c)

    def complete_retire(self, sleeve: str, now: datetime | None = None, how: str = "flat") -> bool:
        """Retire a winding-down strategy once the journal shows it flat (dust counts as flat: lifecycle_flat): the end of the PM's own wind-down, never a
        new move (lifecycle.due's RETIRE). False, and nothing written, unless it is winding down and flat. how: flat,
        deadline or close_now, for the journal's words. Its equity leaves the fund's book with the row (F-C3B-A)."""
        if self.two_leg(sleeve):
            return False  # it stays winding down, flat, until the put-away design lands (TWO_LEG_FENCE)
        if self.multi_asset(sleeve):
            return self._complete_retire_legs(self.legs(sleeve), now or utcnow(), how)
        try:
            return self._complete_retire(sleeve, now, how)
        except _NotRetired:
            return False

    def _complete_retire_legs(self, names: list[str], now: datetime, how: str) -> bool:
        """A multi-asset Strategy's wind-down ends once every leg is flat: each leg not yet retired is put away (the
        parent last), so one leg's process finishing first never retires the Strategy alone."""
        rows = {n: self.sleeve(n) for n in names}
        left = [n for n in names if rows[n].lifecycle != lifecycle.RETIRED]
        if not left or any(rows[n].lifecycle != lifecycle.WINDING_DOWN for n in left):
            return False
        if not all(self.lifecycle_flat(n)[1] for n in names):
            return False
        text = lifecycle.retired_words(how)
        for name in [n for n in left if n != names[0]] + [n for n in left if n == names[0]]:
            with self._put_away(name, now, "retired") as c:
                c.execute(update(sleeves_t).where(sleeves_t.c.name == name, sleeves_t.c.lifecycle ==
                                                  lifecycle.WINDING_DOWN)
                          .values(lifecycle=lifecycle.RETIRED, desired_state="stopped", updated_at=now))
                if name == names[0]:
                    c.execute(insert(decisions_t).values(ts=now, actor="PM", action="retire", sleeve=name,
                                                         reason="Triggered by the system once every leg was flat, "
                                                                f"ending the PM's wind-down. {text}"))
                    c.execute(insert(events_t).values(sleeve=name, ts=now, level="info", kind=LIFECYCLE_EVENT,
                                                      message=text))
        return True

    def _complete_retire(self, sleeve: str, now: datetime | None, how: str) -> bool:
        now = now or utcnow()
        s = self.sleeve(sleeve)
        if s.lifecycle != lifecycle.WINDING_DOWN or not self.lifecycle_flat(sleeve)[1]:
            return False
        text = lifecycle.retired_words(how)
        with self._put_away(sleeve, now, "retired") as c:  # its equity leaves the book with the row (F-C3B-A)
            done = c.execute(update(sleeves_t).where(sleeves_t.c.name == sleeve,
                                                     sleeves_t.c.lifecycle == lifecycle.WINDING_DOWN)
                             .values(lifecycle=lifecycle.RETIRED, desired_state="stopped", updated_at=now)).rowcount
            if not done:  # another process retired it first: nothing taken out (the raise rolls the flow back)
                raise _NotRetired
            # The PM's wind-down, ended by the system once flat: the row names both, so it never reads as the PM
            # pressing retire (Advisor 13 Oct 16:40)
            c.execute(insert(decisions_t).values(ts=now, actor="PM", action="retire", sleeve=sleeve,
                                                 reason="Triggered by the system once flat, ending the PM's "
                                                        f"wind-down. {text}"))
            c.execute(insert(events_t).values(sleeve=sleeve, ts=now, level="info", kind=LIFECYCLE_EVENT, message=text))
        return True

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

    def event(self, sleeve: str | None, level: str, kind: str, message: str, ts: datetime | None = None,
              command_id: int | None = None) -> None:
        """`command_id`: only on a liquidation_reset (or a pair's re-base with one, PAIR_REBASED), the command it carries
        out (unique: a second raises)."""
        if level not in LEVELS:
            raise ValueError(f"bad level {level!r}")
        if command_id is not None and kind not in (LIQUIDATION_RESET, PAIR_REBASED):
            raise ValueError("only a liquidation_reset names the command it carries out")
        with self.engine.begin() as c:
            c.execute(insert(events_t).values(sleeve=sleeve, ts=ts or utcnow(), level=level, kind=kind, message=message,
                                              command_id=command_id))

    # --- a two-leg Strategy's coordinator (P2-8, Advisor M3; DA 14 Oct 05:02 UK) ---------------------------------------

    def pair_state(self, parent: str) -> dict | None:
        with self.engine.connect() as c:
            rows = _rows(c.execute(select(pair_state_t).where(pair_state_t.c.parent == parent)))
        return rows[0] if rows else None

    def set_pair_state(self, parent: str, *, expected: str | None, state: str, trade_id: str | None = None,
                       deadline: datetime | None = None, ts: datetime | None = None,
                       expected_trade_id: str | None = None) -> bool:
        """Move the pair from `expected` (None: it has no row yet) and `expected_trade_id` (the trade it holds; None
        when flat) to `state`, with one pair_state event in the same transaction. A compare-and-set on both, so a
        stale writer from an earlier trade passing through the same state can't move a later one: False when the row
        wasn't there (another writer moved it first, or it already exists), and nothing is written; re-read and
        decide again. ValueError for a move a pair never makes (PAIR_MOVES), before anything is written."""
        if state not in PAIR_MOVES.get(expected, ()):
            raise ValueError(f"{expected or 'new'} -> {state} isn't a move a two-leg Strategy makes")
        if expected not in (None, "flat") and expected_trade_id is None:
            raise ValueError(f"a move from {expected} names the trade it moves (expected_trade_id)")
        ts = ts or utcnow()
        values = dict(state=state, trade_id=trade_id, deadline=deadline, updated_at=ts)
        try:
            with self.engine.begin() as c:
                row = c.execute(select(sleeves_t.c.parent).where(sleeves_t.c.name == parent)).first()
                if row is None or row.parent is not None:
                    raise ValueError(f"{parent!r} is not a two-leg Strategy's first leg")
                if expected is None:
                    c.execute(insert(pair_state_t).values(parent=parent, **values))
                else:
                    held = (pair_state_t.c.trade_id.is_(None) if expected_trade_id is None
                            else pair_state_t.c.trade_id == expected_trade_id)
                    done = c.execute(update(pair_state_t).where(pair_state_t.c.parent == parent,
                                                                pair_state_t.c.state == expected, held).values(**values))
                    if done.rowcount != 1:
                        return False
                c.execute(insert(events_t).values(
                    sleeve=parent, ts=ts, level="info", kind="pair_state",
                    message=f"{expected or 'new'} -> {state}" + (f" ({trade_id})" if trade_id else "")))
        except IntegrityError:
            if expected is None:
                return False  # it has a row already: another writer made it
            raise
        return True

    def mark_pair(self, parent: str, equity, day, ts: datetime | None = None) -> dict:
        """The summed guard's figures at the pair's first mark, all at once (reference, high-water mark and the day's
        start at `equity`); kept as they are once set. Returns the row."""
        ts, v = ts or utcnow(), stored(equity, "equity")
        with self.engine.begin() as c:
            c.execute(update(pair_state_t).where(pair_state_t.c.parent == parent, pair_state_t.c.marked_at.is_(None))
                      .values(reference_equity=v, hwm=v, day_start_equity=v, day_start=day, marked_at=ts, updated_at=ts))
        return self.pair_state(parent)

    def raise_pair_peak(self, parent: str, equity, ts: datetime | None = None) -> None:
        """A new summed high: the halt reference and the high-water mark rise to it, never fall (a stale writer's
        lower figure changes nothing)."""
        ts, v = ts or utcnow(), stored(equity, "equity")
        t, e = pair_state_t, literal(v, EXACT)
        with self.engine.begin() as c:
            c.execute(update(t).where(t.c.parent == parent, t.c.marked_at.is_not(None),
                                      or_(t.c.hwm < v, t.c.reference_equity < v))
                      .values(hwm=case((t.c.hwm < v, e), else_=t.c.hwm),
                              reference_equity=case((t.c.reference_equity < v, e), else_=t.c.reference_equity),
                              marked_at=ts, updated_at=ts))

    def set_pair_day(self, parent: str, day, equity, ts: datetime | None = None, *, rebase: bool = False) -> None:
        """The UTC day's start at `equity`: once a day (a day already started keeps its figure), or on a Resume
        (`rebase`), which re-bases it."""
        ts, v = ts or utcnow(), stored(equity, "equity")
        t = pair_state_t
        q = update(t).where(t.c.parent == parent, t.c.marked_at.is_not(None))
        if not rebase:
            q = q.where(t.c.day_start < day)
        with self.engine.begin() as c:
            c.execute(q.values(day_start=day, day_start_equity=v, marked_at=ts, updated_at=ts))

    def rebase_pair(self, parent: str, equity, ts: datetime | None = None, *, hwm: bool = False) -> None:
        """A Resume after a halt re-bases the halt reference at `equity`; a book reset (`hwm`) the high-water mark too."""
        ts, v = ts or utcnow(), stored(equity, "equity")
        values = dict(reference_equity=v, marked_at=ts, updated_at=ts, **({"hwm": v} if hwm else {}))
        with self.engine.begin() as c:
            c.execute(update(pair_state_t).where(pair_state_t.c.parent == parent,
                                                 pair_state_t.c.marked_at.is_not(None)).values(**values))

    def reset_for_command(self, command_id: int, kind: str = LIQUIDATION_RESET) -> dict | None:
        """The liquidation_reset journaled for this command (F217-3), else None. kind=PAIR_REBASED: the other leg's
        half of a two-leg Strategy's reset after liquidation."""
        q = select(events_t).where(events_t.c.command_id == command_id, events_t.c.kind == kind)
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

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

    def book_equity_at_or_before(self, ts: datetime) -> Decimal | None:
        """The fund's book at its last minute mark at or before `ts` (book_marks), or None before the first."""
        q = select(book_marks_t.c.equity).where(book_marks_t.c.ts <= ts).order_by(book_marks_t.c.ts.desc()).limit(1)
        with self.engine.connect() as c:
            return c.execute(q).scalar()

    def first_book_mark(self) -> dict | None:
        q = select(book_marks_t.c.ts, book_marks_t.c.equity).order_by(book_marks_t.c.ts).limit(1)
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def refused_ideas(self, sleeve: str | None, since: datetime | None = None, limit: int = 200) -> list[dict]:
        """The gate's refusals and trims of `sleeve`'s ideas (every Strategy's when None), newest first, one per idea
        (its submit check; the fill backstop only when it decided alone): sleeve, decided_at, underlying, intent,
        requested_qty, approved_qty, outcome, test_hit, limit_hit, compliance_version, price, book_equity,
        open_risk, would_be (the refused test's measure at the requested size, a share of the book) and side (+1/-1)."""
        d = gate_decisions_t
        q = (select(sleeves_t.c.name.label("sleeve"), d.c.decided_at, d.c.underlying, d.c.intent, d.c.requested_qty,
                    d.c.approved_qty, d.c.outcome, d.c.test_hit, d.c.limit_hit, d.c.compliance_version, d.c.price,
                    d.c.book_equity, d.c.open_risk, d.c.stage, d.c.would_be, d.c.side)
             .join(sleeves_t, sleeves_t.c.id == d.c.sleeve_id)
             .where(d.c.outcome.in_(("rejected", "trimmed")))
             .order_by(d.c.decided_at.desc(), d.c.seq.desc()).limit(limit))
        if sleeve is not None:
            q = q.where(sleeves_t.c.name == sleeve)
        if since is not None:
            q = q.where(d.c.decided_at >= since)
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def fills(self, sleeve: str | None = None, limit: int = 200) -> list[dict]:
        q = select(fills_t)
        q = q.where(fills_t.c.sleeve == sleeve) if sleeve else q.where(_not_backtest(fills_t.c.sleeve))
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(fills_t.c.ts.desc(), fills_t.c.id.desc()).limit(limit)))

    def journal_book(self, sleeve: str, starting_balance: Decimal | float) -> dict:
        """Cash, signed position and average entry implied by the journal: the paper book's source of
        truth (replay_book), with the perp's funding and any insurance-fund cover booked to cash."""
        # Every column the replay reads, not the table: the replay never needs the instrument (one per Strategy until
        # Drop B), and a store opened on a journal migrated no further than DA-9 still reads it (QA DA-9 c1)
        q = (select(*(c for c in fills_t.c if c.name != "instrument")).where(fills_t.c.sleeve == sleeve)
             .order_by(fills_t.c.ts, fills_t.c.id))
        with self.engine.connect() as c:
            fills = _rows(c.execute(q))
        return replay_book(fills, starting_balance, self.funding_total(sleeve), self.insurance_total(sleeve))

    def record_funding(self, sleeve: str, *, qty: float, price: float, rate: float, amount: float,
                       ts: datetime | None = None, kind: str = "settled", arrived_at: datetime | None = None) -> str:
        """Book a funding movement once (CASH-1): "new" when written; "same" when this settlement already holds it
        (a second process, a restart); "differs" when it holds another figure, or the other charge kind, which raises
        an error event and moves nothing. Only "new" moved cash. A true-up moves cash at `arrived_at`, by default
        when it is written; record_true_up works it out from the baseline (CASH-4)."""
        qty, price, amount = _exact(qty=qty, price=price, amount=amount)
        ts = ts or utcnow()
        if kind == "true_up":
            arrived_at = arrived_at or utcnow()
        elif arrived_at is not None:
            raise ValueError(f"only a true-up moves cash at an arrival time, not a {kind} row")
        key = (funding_t.c.sleeve == sleeve) & (funding_t.c.ts == ts)
        key &= funding_t.c.kind.in_(FUNDING_CHARGE_KINDS) if kind in FUNDING_CHARGE_KINDS else funding_t.c.kind == kind
        values = dict(sleeve=sleeve, ts=ts, qty=qty, price=price, rate=rate, amount=amount, kind=kind)
        if arrived_at is not None:  # named only when set, so a journal from before 0016 still takes every other row
            values["arrived_at"] = arrived_at
        booked = self._book_once(funding_t, key, values,
                                 lambda row: row.kind == kind and stored(row.amount) == stored(amount),
                                 cols=(funding_t.c.kind, funding_t.c.amount))
        if booked == "differs":
            self.event(sleeve, "error", "funding_conflict",
                       f"Funding for the {ts:%d %b %H:%M} UTC settlement ({kind}, {amount:,.2f}) arrived again with "
                       "different figures. The booked entry is kept and nothing moved; check the venue's record "
                       "before trusting this strategy's book", ts=ts)
        return booked

    def _book_once(self, table, key, values: dict, same, cols=None) -> str:
        """Insert `values` unless a row matching `key` exists: "new", "same" (same(row) holds) or "differs". The
        table's unique index makes a concurrent second writer fail; its retry reads the first writer's row. `cols`:
        the columns same() reads, when not every column (one a later migration adds may not be there yet)."""
        for _ in range(2):
            try:
                with self.engine.begin() as c:
                    row = c.execute((select(*cols) if cols else select(table)).where(key)).first()
                    if row is not None:
                        return "same" if same(row) else "differs"
                    c.execute(insert(table).values(**values))
                    return "new"
            except IntegrityError:
                continue
        raise RuntimeError(f"a {table.name} row of {values.get('sleeve')} could not be booked or read")

    def record_true_up(self, sleeve: str, ts: datetime, *, rate: float, amount: float | Decimal,
                       arrived_at: datetime) -> tuple[str, Decimal | None]:
        """CASH-4 (Advisor 8 Oct 04:10, ruling 2): the venue's rate for settlement `ts`, charged at the baseline,
        arrived at `arrived_at`; `amount` is what that rate settles to (signed as the cash moves). Books one true-up of
        amount - baseline at the settlement's ts, moving cash at its arrival, and never edits the baseline. Returns
        ("new", true-up) when written; ("same", true-up) when this settlement already holds that one (a second
        process, a restart); ("differs", None) when it holds another, which raises an error event and moves nothing;
        ("none", None) when there is nothing to true up: no baseline at ts (it was settled at the venue's rate) or
        the baseline was reversed (no settlement after all). Only "new" moved cash."""
        amount, = _exact(amount=amount)
        at = (funding_t.c.sleeve == sleeve) & (funding_t.c.ts == ts)
        for _ in range(2):
            try:
                with self.engine.begin() as c:
                    rows = {r.kind: r for r in c.execute(select(funding_t).where(at))}
                    base = rows.get("baseline")
                    if base is None or "reversal" in rows:
                        return "none", None
                    up = amount - to_decimal(base.amount)
                    if "true_up" in rows:
                        if stored(rows["true_up"].amount) == stored(up):
                            return "same", to_decimal(rows["true_up"].amount)
                        break
                    c.execute(insert(funding_t).values(sleeve=sleeve, ts=ts, qty=base.qty, price=base.price,
                                                       rate=rate, amount=up, kind="true_up", arrived_at=arrived_at))
                    return "new", up
            except IntegrityError:
                continue  # a second writer booked it first: read theirs
        else:
            raise RuntimeError(f"a funding true-up of {sleeve} could not be booked or read")
        self.event(sleeve, "error", "funding_conflict",
                   f"The true-up for the {ts:%d %b %H:%M} UTC settlement arrived again as {up:+,.2f}, not as booked. "
                   "The booked true-up is kept and nothing moved; check the venue's record before trusting this "
                   "strategy's book", ts=arrived_at)
        return "differs", None

    def open_baselines(self, sleeve: str) -> list[dict]:
        """Settlements charged the baseline and neither trued up nor reversed yet, oldest first: what a strategy
        with no process of its own is still owed a true-up on (Supervisor.true_up_stopped, CASH-4)."""
        other = funding_t.alias("other")
        done = (select(other.c.id).where(other.c.sleeve == funding_t.c.sleeve, other.c.ts == funding_t.c.ts,
                                         other.c.kind.in_(("true_up", "reversal"))).exists())
        q = (select(funding_t).where(funding_t.c.sleeve == sleeve, funding_t.c.kind == "baseline", ~done)
             .order_by(funding_t.c.ts))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    @staticmethod
    def _cash_moved_at(t):
        """When a row moved cash: a funding true-up at its arrival, everything else at its ts (CASH-4)."""
        return func.coalesce(t.c.arrived_at, t.c.ts) if t is funding_t else t.c.ts

    def funding(self, sleeve: str, limit: int | None = 1000) -> list[dict]:
        """Newest first: the last `limit` rows, or all of them with limit=None, as a check that must see every
        settlement reads them (CASH-1)."""
        q = select(funding_t).where(funding_t.c.sleeve == sleeve).order_by(funding_t.c.ts.desc()).limit(limit)
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def funding_total(self, sleeve: str, before: datetime | None = None) -> Decimal:
        """Funding booked to the strategy's cash, all of it or (before) only what moved it before then: a true-up
        from the arrival of the rate it corrects (CASH-4). Exact on either database, as strategy_cash_pnl: a true-up's
        difference must not come back as a float's."""
        w = funding_t.c.sleeve == sleeve
        if before is not None:
            w &= self._cash_moved_at(funding_t) < before
        with self.engine.connect() as c:
            if self.engine.dialect.name == "postgresql":
                return to_decimal(c.execute(select(func.sum(funding_t.c.amount)).where(w)).scalar() or 0)
            return sum((to_decimal(a) for a in c.execute(select(funding_t.c.amount).where(w)).scalars()), Decimal(0))

    def record_insurance(self, sleeve: str, *, price: float, amount: float, order_id: str | None = None,
                         trade_id: str | None = None, ts: datetime | None = None) -> str:
        """Book the insurance fund's credit for the fill (order_id, trade_id) that left the strategy flat below zero,
        once (CASH-1): "new", "same" or "differs", as record_funding. A credit that names no fill could not be
        refused the second time, so it is refused now (CR F226-2); only rows from before 0013 are unkeyed."""
        price, amount = _exact(price=price, amount=amount)
        if not order_id or not trade_id:
            raise ValueError(f"an insurance credit for {sleeve} must name the fill it covers (order_id and trade_id)")
        # Up to the cent, here in the one write path so no caller can skip it: it exists to bring equity back to zero
        # or above (Advisor 8 Oct 04:10, ruling 3)
        amount = amount.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        ts = ts or utcnow()
        key = ((insurance_t.c.sleeve == sleeve) & (insurance_t.c.order_id == order_id)
               & (insurance_t.c.trade_id == trade_id))
        booked = self._book_once(insurance_t, key, dict(sleeve=sleeve, ts=ts, price=price, amount=amount,
                                                        order_id=order_id, trade_id=trade_id),
                                 lambda row: stored(row.amount) == stored(amount))
        if booked == "differs":
            self.event(sleeve, "error", "insurance_conflict",
                       f"The insurance credit for order {order_id} (trade {trade_id}) arrived again as {amount:,.2f}. "
                       "The booked credit is kept and nothing moved", ts=ts)
        return booked

    def insurance(self, sleeve: str, limit: int | None = 1000) -> list[dict]:
        """The insurance fund's covers, newest first, as the trades are paired with them (the Trades tab, attribution):
        an old book close-out (CLOSE_OUT) is a transfer, not a trade's, so it isn't one (HoQA RESET-NEG cell 9)."""
        i = insurance_t.c
        q = (select(insurance_t).where(i.sleeve == sleeve, (i.order_id.is_(None)) | (i.order_id != CLOSE_OUT))
             .order_by(i.ts.desc()).limit(limit))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def insurance_total(self, sleeve: str, before: datetime | None = None) -> Decimal:
        """What the venue's insurance fund covered, all of it or (before) only what it covered before then."""
        q = select(func.sum(insurance_t.c.amount)).where(insurance_t.c.sleeve == sleeve)
        if before is not None:
            q = q.where(insurance_t.c.ts < before)
        with self.engine.connect() as c:
            return to_decimal(c.execute(q).scalar() or 0)

    def strategy_cash_pnl(self, sleeve: str, since: datetime | None = None, at: datetime | None = None) -> Decimal:
        """The one cash reader (CASH-2): what the journal moved the strategy's cash by over (since, at], every row
        counted. That is each fill's cash leg less its fee, plus funding and the insurance fund's cover. since None
        is from the first row and at None is to the last; a row at `at` counts and a row at `since` does not, so
        consecutive windows add up to the whole. A funding true-up counts from the arrival of the rate it corrects,
        not its settlement's ts (CASH-4). Exact: summed by Postgres' NUMERIC, or row by row on SQLite, whose own sum
        would be a float's."""
        def window(t):
            w, moved = t.c.sleeve == sleeve, self._cash_moved_at(t)
            if since is not None:
                w &= moved > since
            if at is not None:
                w &= moved <= at
            return w

        f = fills_t.c
        with self.engine.connect() as c:
            if self.engine.dialect.name == "postgresql":
                # Postgres' round(numeric, 8) is half away from zero: per_fill's ROUND_HALF_UP
                leg = (case((f.side == "BUY", -func.round(f.qty * f.price, 8)), else_=func.round(f.qty * f.price, 8))
                       - func.round(f.fee, 8))
                sums = [select(func.sum(leg)).where(window(fills_t)),
                        select(func.sum(funding_t.c.amount)).where(window(funding_t)),
                        select(func.sum(insurance_t.c.amount)).where(window(insurance_t))]
                return sum((to_decimal(c.execute(q).scalar() or 0) for q in sums), Decimal(0))
            total = sum((fill_cash(*r) for r in c.execute(select(f.side, f.qty, f.price, f.fee)
                                                          .where(window(fills_t)))), Decimal(0))
            for t in (funding_t, insurance_t):
                total += sum((to_decimal(a) for a in c.execute(select(t.c.amount).where(window(t))).scalars()),
                             Decimal(0))
            return total

    def close_out_cash(self, sleeve: str, since: datetime | None = None) -> Decimal:
        """What old book close-outs (CLOSE_OUT) moved the strategy's cash by after `since`, as strategy_cash_pnl counts
        them: a transfer, so what compares that cash with the trips (paired from insurance, without them) takes it
        back out (CR F-RN-5)."""
        i = insurance_t.c
        w = (i.sleeve == sleeve) & (i.order_id == CLOSE_OUT)
        if since is not None:
            w &= i.ts > since
        with self.engine.connect() as c:
            return sum((to_decimal(a) for a in c.execute(select(i.amount).where(w)).scalars()), Decimal(0))

    def fill_totals(self, sleeve: str) -> tuple[int, Decimal]:
        """Every fill the strategy has and the fees they paid, uncapped (CASH-2), exact as strategy_cash_pnl."""
        with self.engine.connect() as c:
            if self.engine.dialect.name == "postgresql":
                n, fees = c.execute(select(func.count(), func.sum(func.round(fills_t.c.fee, 8)))
                                    .where(fills_t.c.sleeve == sleeve)).one()
                return n, to_decimal(fees or 0)
            fees = list(c.execute(select(fills_t.c.fee).where(fills_t.c.sleeve == sleeve)).scalars())
            return len(fees), sum((per_fill(to_decimal(x)) for x in fees), Decimal(0))

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
        if s.parent is not None or self._leg_of(sleeve) is not None:  # CR F283-1: split_run makes one leg's run
            raise ValueError("a reset isn't built for a two-leg Strategy yet, so nothing was changed")
        if self._asset_legs(sleeve):  # DA (Drop B MUST): split_run would put away the parent's run alone
            raise ValueError("a reset isn't built for a multi-asset Strategy yet, so nothing was changed")
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
                    words = (flattened_words(asked[-1].reason) if asked[-1].command == "flatten"
                             else f"paused by PM: {asked[-1].reason}")
                    hold = {"status": "paused", "status_reason": words, "paused_until": None}
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
            _drop_carried(c)  # MUST-F286-H4 (HoQA): a reset while the book is empty is a fresh start
            c.execute(insert(decisions_t).values(
                ts=now, actor=actor, action=BOOK_RESET, sleeve=None,
                reason=f"Book reset: {reason.strip()} (cleared: {', '.join(s.name for s, *_ in cleared) or 'none'})"))
        return [s.name for s, *_ in cleared]

    def drop_carried(self) -> bool:
        """A clean slate while the book is empty (F286-4): the next capital starts afresh, not at the drawdown a
        withdrawal that emptied the book carried (MUST-F286-H4). True when there was one to drop."""
        with self.engine.begin() as c:
            return _drop_carried(c)

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

    def reset_from(self) -> dict[str, str]:
        """Each run a reset put away, and the Strategy it was reset from: {run name: Strategy name}."""
        with self.engine.connect() as c:
            return {r.run: r.sleeve for r in c.execute(
                select(resets_t.c.run, resets_t.c.sleeve).where(resets_t.c.done_at.is_not(None), resets_t.c.run != ""))}

    def close_out_run(self, run: str, into: str, amount: Decimal, key: str, ts: datetime, actor: str, action: str,
                      said: str) -> bool:
        """RESET-NEG (Advisor, 15 Oct): a reset run of the old book below zero closed out against the Strategy it was
        reset from, whose capital it ran on (split_run): `amount` (the run's loss, > 0) credited to the run, bringing it
        to exactly 0, and debited from `into`, with the decision `said`, in one transaction. A transfer inside the book,
        so its total doesn't move. The rows are cash rows both of the book's readers count (journal_book and
        strategy_cash_pnl), so its identity doesn't move either, keyed by `key` and the run: once only (insurance_once),
        a second call writes nothing and returns False."""
        if amount <= 0:
            raise ValueError(f"a close-out credits a run below zero, not {amount}")
        i = insurance_t.c
        with self.engine.begin() as c:
            if c.execute(select(i.id).where(i.sleeve == run, i.order_id == key, i.trade_id == run)).first():
                return False
            c.execute(insert(insurance_t), [
                {"sleeve": run, "ts": ts, "price": 0, "amount": amount, "order_id": key, "trade_id": run},
                {"sleeve": into, "ts": ts, "price": 0, "amount": -amount, "order_id": key, "trade_id": run}])
            c.execute(insert(decisions_t).values(ts=ts, actor=actor, action=action, sleeve=into, reason=said.strip()))
        return True

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
        with self.engine.begin() as c:
            # The reset row is locked before the hold is read, as Store.command locks it: a PM pause or flatten pressed
            # meanwhile is either in the hold read here or waits and stays a pending command of the fresh run.
            locked = c.execute(select(resets_t.c.restart).where(resets_t.c.id == request["id"]).with_for_update()).first()
            if locked is not None:  # a PM Stop since the request (Store.hold_on_reset) keeps the fresh run stopped
                request["restart"] = locked.restart
            row = dict(c.execute(select(sleeves_t).where(sleeves_t.c.name == name)).first()._mapping)
            row_id = row.pop("id")
            c.execute(insert(sleeves_t).values(**{**row, "name": run, "desired_state": "stopped", "status": "stopped",
                                                  "status_reason": f"run put away by a reset on {now:%d %b %Y %H:%M}",
                                                  "updated_at": now}))
            # The pin is copied before the orders and fills move: the database refuses a pin to a strategy that has
            # traded (F-WED1-3), and the put-away run hasn't yet.
            for t in RESET_COPIES:
                for r in c.execute(select(t).where(t.c.sleeve == name)).all():
                    c.execute(insert(t).values(**{**dict(r._mapping), "sleeve": run}))
            for t in RESET_MOVES:
                c.execute(update(t).where(t.c.sleeve == name).values(sleeve=run))
            # Feed state, and the old run's conditions on the Signals tab, until the fresh process writes its own
            # (m13-E2).
            for t in RESET_CLEARS:
                c.execute(t.delete().where(t.c.sleeve == name))
            # A reservation still active is room the old run held for an order that can no longer fill (the process
            # is stopped before the split): released, so no claim from before the reset counts after it (HoE, DA-6).
            r = gate_reservations_t
            c.execute(update(r).where(r.c.sleeve_id == row_id, r.c.released_at.is_(None))
                      .values(released_at=now, release_reason="cancel"))
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

    def write_incident_note(self, incident_id: int, *, author: str, why_stop_did_not_protect: str,
                            why_clamp_did_not_prevent: str = "") -> None:
        """The note on a liquidation's incident (Advisor 18:17): why the half-liquidation stop did not protect the
        position, and who wrote it. Both are required. A later note on the same incident replaces it. A two-leg
        Strategy's leg's note also says why closing the pair at half its distance to liquidation didn't prevent it
        (WHY_CLAMP_FIELD, Advisor F283-O3): its reset after liquidation is refused without it."""
        why, author = (why_stop_did_not_protect or "").strip(), (author or "").strip()
        clamp = (why_clamp_did_not_prevent or "").strip()
        if not why:
            raise ValueError(f"the note needs {WHY_STOP_FIELD}")
        if not author:
            raise ValueError("the note needs its author")
        ev = self.event_by_id(incident_id)
        if ev is None or ev["kind"] != "incident" or ev["sleeve"] is None:
            raise ValueError(f"#{incident_id} is not a strategy's incident")
        self.event(ev["sleeve"], "info", INCIDENT_NOTE,
                   f"Incident #{incident_id} note by {author}: {WHY_STOP_FIELD}: {why}"
                   + (f" | {WHY_CLAMP_FIELD}: {clamp}" if clamp else ""))

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
        clamp = note["message"].split(f" | {WHY_CLAMP_FIELD}: ", 1)
        note["clamp"] = clamp[1] if len(clamp) == 2 else None
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

    def portfolio_profile(self, version: int | None = None):
        """The PM's portfolio limits as a risk.PortfolioProfile: the latest version, or the one named. Raises
        LookupError when there is none, so a caller never falls back to limits nobody accepted."""
        from sleeve_fund.risk import PortfolioProfile
        t = portfolio_profile_t
        q = select(t).where(t.c.version == version) if version is not None else \
            select(t).order_by(t.c.version.desc()).limit(1)
        with self.engine.connect() as c:
            r = c.execute(q).first()
        if r is None:
            raise LookupError(f"no portfolio profile {version if version is not None else '(none recorded)'}")
        return PortfolioProfile(version=r.version, gross=float(str(r.gross_max)),
                                net_instrument=float(str(r.net_underlying_max)), margin=float(str(r.margin_max)),
                                open_risk=float(str(r.open_risk_max)), drawdown=float(str(r.drawdown_halt)),
                                daily_loss=float(str(r.daily_pause)))

    def add_portfolio_profile(self, limits, created_by: str, note: str | None = None) -> int:
        """Record new portfolio limits as the next version and return it. Append-only: there is no update, because
        every gate decision names the version it was taken under. limits.version is ignored. A limit finer than the
        column's 4 places is refused, never rounded (QA F211-1: 0.00125 would be kept as a looser 0.0013)."""
        t = portfolio_profile_t
        for name in ("gross", "net_instrument", "margin", "open_risk", "drawdown", "daily_loss"):
            d = to_decimal(getattr(limits, name), name)
            if d != d.quantize(LIMIT_PLACES):
                raise ValueError(f"a portfolio limit has at most 4 decimal places: {name} is {d}")
        with self.engine.begin() as c:
            version = (c.execute(select(func.max(t.c.version))).scalar() or 0) + 1
            c.execute(insert(t), [{
                "version": version, "created_at": utcnow(), "created_by": created_by, "note": note,
                "gross_max": to_decimal(limits.gross), "net_underlying_max": to_decimal(limits.net_instrument),
                "margin_max": to_decimal(limits.margin), "open_risk_max": to_decimal(limits.open_risk),
                "drawdown_halt": to_decimal(limits.drawdown), "daily_pause": to_decimal(limits.daily_loss)}])
        return version

    # --- the fund's capital (bridge A, DA 0025) ----------------------------------------------------------------

    def fund_flow(self, kind: str, amount, *, actor: str, reason: str, rebase: bool = False,
                  now: datetime | None = None) -> int:
        """Journal an external deposit or withdrawal of the fund's capital and return its id: the row and its decision
        in one transaction. It moves nothing yet: the poll ends it in id order (settle_flow or refuse_flow). The dry run
        is the command's (sleeve_fund.fund), which calls this only on --apply."""
        if kind not in FUND_FLOW_KINDS:
            raise ValueError(f"a fund flow is a {' or a '.join(FUND_FLOW_KINDS)}, not {kind!r}")
        amount = to_decimal(amount, "amount")
        if not amount > 0:
            raise ValueError(f"a fund flow's amount is above zero: {amount}")
        if rebase and kind != "deposit":
            raise ValueError("only a deposit can re-base the book (the clean slate's)")
        if not actor or not (reason or "").strip():
            raise ValueError("a fund flow names who asked for it and why")
        now = now or utcnow()
        with self.engine.begin() as c:
            fid = c.execute(insert(fund_flows_t).values(kind=kind, amount=amount, rebase=rebase, actor=actor,
                                                        reason=reason.strip(), requested_at=now)).inserted_primary_key[0]
            c.execute(insert(decisions_t).values(ts=now, actor=actor, action=f"fund_{kind}", sleeve=None,
                                                 reason=f"{kind} {amount} (flow {fid}{', re-base' if rebase else ''}): "
                                                        f"{reason.strip()}"))
        return fid

    def pending_flows(self) -> list[dict]:
        """The flows the poll hasn't ended, in the order it must take them (id)."""
        t = fund_flows_t
        with self.engine.connect() as c:
            return _rows(c.execute(select(t).where(t.c.applied_at.is_(None), t.c.refused_at.is_(None))
                                   .order_by(t.c.id)))

    def fund_flows(self) -> list[dict]:
        """Every flow, oldest first."""
        with self.engine.connect() as c:
            return _rows(c.execute(select(fund_flows_t).order_by(fund_flows_t.c.id)))

    def settle_flow(self, flow_id: int, at: datetime, conn=None) -> None:
        """End a pending flow as applied at `at`. conn: the poll's own transaction, so the flow lands with the book's
        figures; raises when the flow is not pending, or an earlier one is."""
        self._end_flow(flow_id, {"applied_at": at}, conn)

    def refuse_flow(self, flow_id: int, at: datetime, reason: str, conn=None) -> None:
        """End a pending flow as refused, with the reason the PM sees."""
        if not (reason or "").strip():
            raise ValueError("a refused flow says why")
        self._end_flow(flow_id, {"refused_at": at, "refuse_reason": reason.strip()}, conn)

    def _end_flow(self, flow_id: int, values: dict, conn) -> None:
        t = fund_flows_t
        pending = and_(t.c.applied_at.is_(None), t.c.refused_at.is_(None))

        def end(c):
            first = c.execute(select(func.min(t.c.id)).where(pending)).scalar()
            if first != flow_id:
                raise ValueError(f"flow {flow_id} is not the first pending one ({first}): flows end in order")
            c.execute(update(t).where(t.c.id == flow_id, pending).values(**values))

        if conn is not None:
            end(conn)
        else:
            with self.engine.begin() as c:
                end(c)

    def fund_capital(self, at: datetime | None = None) -> Decimal:
        """The fund's external capital: applied deposits less applied withdrawals, all of them or (at) those applied
        at or before then. Exact on either database."""
        t = fund_flows_t
        w = t.c.applied_at.is_not(None)
        if at is not None:
            w &= t.c.applied_at <= at
        with self.engine.connect() as c:
            rows = c.execute(select(t.c.kind, t.c.amount).where(w)).all()
        return sum((-to_decimal(a) if k == "withdraw" else to_decimal(a) for k, a in rows), Decimal(0))

    # --- instruments ------------------------------------------------------------------------------------------

    def instrument_id(self, venue: str, symbol: str, contract: str, underlying: str, conn=None) -> int:
        """The instrument's row, written if it is missing; conn: the caller's transaction."""
        if conn is not None:
            return _ensure_instrument(conn, venue, symbol, contract, underlying)
        with self.engine.begin() as c:
            return _ensure_instrument(c, venue, symbol, contract, underlying)

    def instruments(self) -> list[dict]:
        with self.engine.connect() as c:
            return _rows(c.execute(select(instruments_t).order_by(instruments_t.c.id)))

    # --- (Strategy, asset): multi-asset Strategies (Drop B, DA 0028; PE1 multi-asset-spec) ---------------------

    def strategy_assets(self, name: str) -> list[dict]:
        """Every asset of the Strategy `name` (its own row's name, or any of its legs'), one row each: the parent row
        first, then its legs in the order they were made (the definition's asset order). A one-asset Strategy is one
        row. Each row: sleeve (the row trading it), instrument, underlying, venue and whether it is a leg."""
        from sleeve_fund.portfolio.holding import underlying
        from sleeve_fund.venues import DEFAULT_VENUE

        t, v = sleeves_t, sleeve_venues_t
        with self.engine.connect() as c:
            row = c.execute(select(t.c.name, t.c.parent).where(t.c.name == name)).first()
            if row is None:
                raise KeyError(f"no strategy {name!r}")
            strategy = row.parent or row.name
            rows = c.execute(select(t.c.name, t.c.instrument, t.c.parent, v.c.venue)
                             .select_from(t.outerjoin(v, v.c.sleeve == t.c.name))
                             .where(or_(t.c.name == strategy, t.c.parent == strategy))
                             .order_by(t.c.parent.is_not(None), t.c.id)).all()
        return [{"strategy": strategy, "sleeve": r.name, "instrument": r.instrument,
                 "underlying": underlying(r.instrument), "venue": (r.venue or DEFAULT_VENUE).upper(),
                 "leg": r.parent is not None} for r in rows]

    def asset_book(self, strategy: str) -> dict:
        """The Strategy's book per asset and in total, rebuilt from the journal (fills are the truth): each asset's
        row is journal_book of the row trading it, and the total is their exact sum, so per-asset P&L, fees and
        funding add up to the Strategy's to the last digit (nightly rebuild, Drop B Done-when 6). Position and entry
        are per asset only: a total of BTC and ETH units means nothing."""
        assets = []
        for a in self.strategy_assets(strategy):
            book = self.journal_book(a["sleeve"], self.sleeve(a["sleeve"]).starting_balance)
            fees = self._fees(a["sleeve"])
            assets.append(a | {"starting_balance": to_decimal(self.sleeve(a["sleeve"]).starting_balance),
                               "cash": book["cash"], "qty": book["qty"], "entry_px": book["entry_px"],
                               "fills": book["fills"], "fees": fees, "funding": book["funding"],
                               "insurance": book["insurance"]})
        total = {k: sum((a[k] for a in assets), Decimal(0))
                 for k in ("starting_balance", "cash", "fees", "funding", "insurance")}
        total["fills"] = sum(a["fills"] for a in assets)
        return {"strategy": assets[0]["strategy"], "assets": assets, "total": total}

    def _fees(self, sleeve: str) -> Decimal:
        with self.engine.connect() as c:
            fees = c.execute(select(fills_t.c.fee).where(fills_t.c.sleeve == sleeve)).scalars().all()
        return sum((to_decimal(f, "fee") for f in fees), Decimal(0))

    def positions_by_asset(self) -> dict[tuple[str, str], Decimal]:
        """{(Strategy, asset): net quantity} as the orders record what filled, every journal row but saved backtests'
        (NIGHTLY-RECON's stored side, reconcile.stored_positions; reconcile.rebuild_positions is the fills' side)."""
        from sleeve_fund import reconcile

        return reconcile.stored_positions(self)

    def asset_mismatches(self, limit: int = 50) -> list[dict]:
        """Reconcile by (Strategy, asset): journal rows whose instrument is not their row's (an orders, fills, funding
        or insurance row written with another asset, or left '' because its Strategy row was gone). Empty when every
        row is keyed right; reconcile names each one (0026 left the check here, never a foreign key)."""
        out = []
        with self.engine.connect() as c:
            for t in (orders_t, fills_t, funding_t, insurance_t):
                q = (select(t.c.id, t.c.sleeve, t.c.instrument, sleeves_t.c.instrument.label("expected"))
                     .select_from(t.outerjoin(sleeves_t, sleeves_t.c.name == t.c.sleeve))
                     .where(or_(sleeves_t.c.instrument.is_(None), t.c.instrument != sleeves_t.c.instrument))
                     .order_by(t.c.id).limit(limit))
                out += [{"table": t.name, "id": r.id, "sleeve": r.sleeve, "instrument": r.instrument,
                         "expected": r.expected} for r in c.execute(q)]
        return out[:limit]

    def record_shadow_outcome(self, decision_id: int, *, kind: str, pnl, method: str,
                              computed_at: datetime | None = None) -> str:
        """A trimmed or refused idea's outcome (Advisor MA-SPEC): "new" when written, "same" when this decision holds
        the same figure already (a rerun), and an error when it holds another (write-once: never overwritten)."""
        pnl = to_decimal(pnl, "pnl")
        row = dict(decision_id=decision_id, kind=kind, pnl=pnl, hypothetical=kind == "refusal", method=method,
                   computed_at=computed_at or utcnow())
        t = shadow_outcomes_t
        with self.engine.begin() as c:
            have = c.execute(select(t).where(t.c.decision_id == decision_id)).first()
            if have is None:
                c.execute(insert(t).values(**row))
                return "new"
        if have.kind == kind and have.method == method and stored(have.pnl) == stored(pnl):
            return "same"
        raise ValueError(f"decision {decision_id} already has a {have.kind} outcome of {stored(have.pnl)}; "
                         "an outcome is never rewritten")

    def shadow_outcomes(self, strategy: str | None = None, limit: int = 200) -> list[dict]:
        """Trimmed and refused ideas with their outcomes, newest first: the decision (Strategy row, asset, bar, asked,
        approved, the test, planned stop and target, side) and its outcome once computed (None until then). Every
        Strategy by default; a Strategy's name takes in all its legs."""
        g, t, o = gate_decisions_t, sleeves_t, shadow_outcomes_t
        q = (select(t.c.name.label("sleeve"), func.coalesce(t.c.parent, t.c.name).label("strategy"), t.c.instrument,
                    g.c.id.label("decision_id"), g.c.bar_ts, g.c.stage, g.c.outcome, g.c.test_hit, g.c.compliance_version,
                    g.c.requested_qty, g.c.approved_qty, g.c.price, g.c.planned_stop, g.c.planned_target, g.c.side,
                    g.c.would_be, o.c.kind, o.c.pnl, o.c.hypothetical, o.c.method, o.c.computed_at)
             .select_from(g.join(t, t.c.id == g.c.sleeve_id).outerjoin(o, o.c.decision_id == g.c.id))
             .where(g.c.outcome.in_(("trimmed", "rejected")))
             .order_by(g.c.decided_at.desc(), g.c.id.desc()).limit(limit))
        if strategy is not None:
            q = q.where(func.coalesce(t.c.parent, t.c.name) == strategy)
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    # --- compliance tests (bridge A, DA 0027; Advisor BRIDGE-3) -----------------------------------------------

    def compliance(self, version: int | None = None, conn=None) -> Compliance:
        """The tests and the asset-group map of one version, the latest by default. The gate calls this inside its
        lock, at evaluation, with its conn, and records the version on the decision (BRIDGE-3-QA)."""
        def read(c) -> Compliance:
            v = version or c.execute(select(func.max(compliance_versions_t.c.version))).scalar()
            head = c.execute(select(compliance_versions_t).where(compliance_versions_t.c.version == v)).first()
            if head is None:
                raise ValueError(f"no compliance version {v}")
            t, m = compliance_tests_t, compliance_asset_types_t
            tests = [{"test": r.test, "label": r.label, "scope": r.scope, "measure": r.measure,
                      "limit": Decimal(r.limit_value).quantize(LIMIT_PLACES), "action": r.action,
                      "enabled": bool(r.enabled)}
                     for r in c.execute(select(t).where(t.c.version == v).order_by(t.c.test))]
            groups = {r.underlying: r.asset_type for r in c.execute(select(m).where(m.c.version == v))}
            return Compliance(head.version, _aware(head.created_at), head.created_by, head.note, tests, groups)

        if conn is not None:
            return read(conn)
        with self.engine.connect() as c:
            return read(c)

    def compliance_versions(self, limit: int | None = None) -> list[dict]:
        """Every version, newest first: version, created_at, created_by, note."""
        q = select(compliance_versions_t).order_by(compliance_versions_t.c.version.desc())
        with self.engine.connect() as c:
            return _rows(c.execute(q.limit(limit) if limit else q))

    def unmapped(self, version: int | None = None) -> list[str]:
        """Underlyings in instruments that the version's map leaves out: the gate refuses their entries while the
        asset-group test is on (Advisor BRIDGE-A-FE-2), and Portfolio Settings shows them in amber."""
        groups = self.compliance(version).asset_types
        return sorted({r["underlying"] for r in self.instruments()} - set(groups))

    def add_compliance_version(self, tests: list[dict], asset_types: dict[str, str], *, expected_version: int,
                               note: str, created_by: str = "pm", now: datetime | None = None) -> int:
        """The PM's edit: a whole new version (every test and the whole map), written only if the latest is still
        expected_version, so two open forms never overwrite each other. Refused with the reasons in the PM's words.
        Journaled with what changed, old and new. It applies from the gate's next evaluation."""
        from sleeve_fund import compliance_rules as rules

        if created_by != "pm" and not created_by.startswith("migration "):
            raise ValueError(f"compliance tests are changed by the PM only, not {created_by!r}")
        if not (note or "").strip():
            raise ValueError("say why the tests change")
        errors = rules.validate(tests)
        rows = []
        for t in tests:
            limit = to_decimal(t.get("limit", 0), "limit")
            if limit != limit.quantize(LIMIT_PLACES):
                errors.append(f"{t.get('label') or t.get('test')}: a limit has at most four decimal places")
            rows.append({"test": t["test"], "label": rules.LABELS.get(t["test"], t.get("label") or t["test"]),
                         "scope": t.get("scope"), "measure": t.get("measure"), "limit_value": limit,
                         "action": t.get("action"), "enabled": bool(t.get("enabled"))})
        groups = {}
        for u, g in (asset_types or {}).items():
            u, g = str(u).strip().upper(), str(g or "").strip()
            if not u or not g:
                errors.append(f"asset {u or '?'} needs a group name")
            groups[u] = g
        if errors:
            raise ValueError("; ".join(errors))
        now = now or utcnow()
        try:
            with self.engine.begin() as c:
                latest = c.execute(select(func.max(compliance_versions_t.c.version))).scalar() or 0
                if latest != expected_version:
                    raise ValueError("the compliance tests changed since this page was opened "
                                     f"(now version {latest}): reload and edit again")
                old = self.compliance(latest, conn=c) if latest else None
                v = latest + 1
                c.execute(insert(compliance_versions_t).values(version=v, created_at=now, created_by=created_by,
                                                               note=note.strip()))
                c.execute(insert(compliance_tests_t), [r | {"version": v} for r in rows])
                if groups:
                    c.execute(insert(compliance_asset_types_t),
                              [{"version": v, "underlying": u, "asset_type": g} for u, g in sorted(groups.items())])
                c.execute(insert(decisions_t).values(
                    ts=now, actor=created_by, action="compliance_version", sleeve=None,
                    reason=f"compliance tests v{v} ({note.strip()}): {_compliance_changes(old, rows, groups)}"))
        except IntegrityError as exc:  # another write took the same version number first
            raise ValueError("the compliance tests changed since this page was opened: reload and edit again") from exc
        return v

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
        row = self.sleeve(sleeve)  # raises if unknown
        two_leg = row.parent is not None or self._leg_of(sleeve) is not None
        if command == BOOK_RESET and two_leg:
            raise ValueError("a reset isn't built for a two-leg Strategy yet, so nothing was changed")  # P2-8
        pair = None  # a two-leg Strategy's reset after liquidation: the leg it resets and the leg re-based with it
        if command == RAL and two_leg:
            pair = self._pair_ral(row, incident, actor)
            sleeve = pair["leg"]  # journaled on the liquidated leg's row, so one reset per incident still holds
        here = None  # a reset after liquidation applied here, as no process will apply it
        if command == RAL:
            from sleeve_fund.paper.runtime import ral_refusal  # the engine's liquidation rule (liquidation_head)

            if (why := ral_refusal(self, sleeve, incident, actor)) is not None:
                raise ValueError(why)
            reason = f"{reason.strip()} (incident #{incident})"
            here = self._ral_here(sleeve, reason, incident, pair)
        elif incident is not None:
            raise ValueError("only a reset after liquidation names an incident")
        now = utcnow()
        try:
            with self.engine.begin() as c:
                cid = c.execute(insert(commands_t).values(sleeve=sleeve, command=command, reason=reason.strip(),
                                                          created_at=now, incident=incident)).inserted_primary_key[0]
                if holds_through_reset and command in ("pause", "flatten"):
                    self._hold_on_reset(c, sleeve, command, reason)
                # A two-leg Strategy's pause, resume and flatten reach both legs' runtimes, in the same transaction:
                # each leg keeps its own status, and a leg left halted or paused would refuse its half of the next
                # pair (P2-8, Advisor M2). A multi-asset Strategy's reach every asset leg's runtime the same way
                # (F-B5-FLATTEN, HoE): each asset runs on its own row, so a flatten told to the parent alone would
                # close its first asset only, and a resume after it would leave the others paused.
                # The leg's queue follows the parent's in order: only a repeat of the leg's own last waiting command
                # is not queued twice (CR F283-3: resume, pause, resume must leave both legs resumed).
                legs = [] if command not in ("pause", "resume", "flatten") else (
                    [leg] if (leg := self._leg_of(sleeve, c)) is not None else self._asset_legs(sleeve, c))
                for leg in legs:
                    if c.execute(select(commands_t.c.command).where(
                            commands_t.c.sleeve == leg, commands_t.c.applied_at.is_(None))
                            .order_by(commands_t.c.id.desc()).limit(1)).scalar() == command:
                        continue
                    c.execute(insert(commands_t).values(sleeve=leg, command=command, reason=reason.strip(),
                                                        created_at=now))
                    if holds_through_reset and command in ("pause", "flatten"):
                        self._hold_on_reset(c, leg, command, reason)
                if pair:
                    # The other leg's half (Advisor F283-O3 (b)): its process re-bases the pair's summed figures and
                    # its own with the liquidated leg's; it resets nothing and lifts no halt of its own.
                    pair["companion"] = c.execute(insert(commands_t).values(
                        sleeve=pair["other"], command=RAL, created_at=now,
                        reason=f"{PAIR_REBASE}: {sleeve}'s reset after liquidation (command #{cid})"
                    )).inserted_primary_key[0]
                if here:
                    self._apply_ral(c, sleeve, cid, here, now)
                    if pair:
                        self._apply_pair_rebase(c, pair, pair["companion"], now)
        except IntegrityError:
            if incident is None:
                raise
            raise ValueError(f"already reset for this liquidation: incident #{incident} has its reset after "
                             "liquidation") from None
        if here:
            self._lapse_resets_before(sleeve, now)
        self.decide(actor, command, reason, pair["parent"] if pair else sleeve)

    def _pair_ral(self, row: Sleeve, incident: int | None, actor: str) -> dict:
        """A two-leg Strategy's reset after liquidation (Advisor F283-O3, 14 Oct 13:15 UK), sent to the Strategy (its
        first leg's row) naming a liquidated leg's incident. Raises ValueError, in words naming what is in the way,
        unless: that leg takes it under the one-leg rule (ral_refusal); its incident's note also says why the pair's
        close at half its distance to liquidation didn't prevent it (WHY_CLAMP_FIELD); the pair is flat in its journal
        (pair_state); neither leg holds anything but dust; and neither has an order working. Returns the parent, the
        liquidated leg and the other. A leg liquidated too needs its own reset, naming its own incident: the pair stays
        blocked until each liquidated leg has one."""
        from sleeve_fund.paper.runtime import liquidation_head, ral_refusal

        if row.parent is not None:
            raise ValueError(f"this is the second leg of {row.parent}: send the reset after liquidation to {row.parent}, "
                             "naming the liquidated leg's incident")
        parent, legs = row.name, self.legs(row.name)
        ev = self.event_by_id(incident) if incident is not None else None
        # The leg the incident is on; else (none named, or not a leg's) the one liquidated leg, so the refusal is
        # its own (name the incident, or not this liquidation's), not "nothing to clear" from the leg that is fine.
        liquidated = [n for n in legs if liquidation_head(self, n) is not None]
        leg = (ev["sleeve"] if ev is not None and ev["sleeve"] in legs else
               liquidated[0] if len(liquidated) == 1 else parent)
        if (why := ral_refusal(self, leg, incident, actor)) is not None:
            raise ValueError(why)
        if (self.incident_note(incident) or {}).get("clamp") is None:
            raise ValueError(f"incident #{incident}'s note needs {WHY_CLAMP_FIELD} too, as {leg} is a leg of the "
                             f"two-leg Strategy {parent}: write it first")
        state = self.pair_state(parent)
        if state is not None and state["state"] != "flat":
            raise ValueError(f"the pair is still {state['state']}: a reset after liquidation waits until both legs "
                             "are closed")
        for name in legs:
            book = self.journal_book(name, self.sleeve(name).starting_balance)
            if abs(book["qty"]) > 1e-12 and not is_dust(book):
                raise ValueError(f"{name} still holds {book['qty']}: a reset after liquidation waits until both legs "
                                 "are closed")
            if self.orders(name, statuses=OPEN_ORDER_STATUSES, limit=1):
                raise ValueError(f"{name} has an order working: a reset after liquidation waits until it is done")
        return {"parent": parent, "leg": leg, "other": next(n for n in legs if n != leg)}

    def pair_rebase_words(self, parent: str, leg: str, old, equity) -> str:
        """What the other leg's half of a pair's reset after liquidation journals."""
        return (f"Pair re-base with {leg}'s reset after liquidation: {parent}'s high-water mark {to_decimal(old):,.2f} "
                f"becomes {to_decimal(equity):,.2f}, both legs' equity summed, which is also the day's loss baseline. "
                "Nothing of this leg is reset and no halt of its own is lifted.")

    def pair_figures(self, pair: dict, now: datetime) -> dict:
        """What a stopped pair's re-base writes, read before its transaction: each leg's cash (both flat), their sum,
        the pair's high-water mark before, and the other leg's last mark (its price and benchmark)."""
        cash = {n: to_decimal(self.journal_book(n, self.sleeve(n).starting_balance)["cash"], "cash")
                for n in (pair["leg"], pair["other"])}
        summed = sum(cash.values(), Decimal(0))
        state = self.pair_state(pair["parent"])
        return {"cash": cash, "summed": summed,
                "old": state["hwm"] if state is not None and state["hwm"] is not None else summed,
                "last": self.equity_at_or_before(pair["other"], now)}

    def _apply_pair_rebase(self, c, pair: dict, command_id: int, now: datetime) -> bool:
        """A stopped two-leg Strategy's reset after liquidation, the other leg's half, in the command's transaction (as
        _apply_ral is the liquidated leg's): the pair's summed high-water mark, halt reference and day's start re-based
        at both legs' equity now (both flat: their cash), a mark of this leg's at it, and the re-base journaled, so a
        Start restores the summed figures (_pair_first_mark). The book-level figures are not touched."""
        took = c.execute(update(commands_t).where(commands_t.c.id == command_id, commands_t.c.applied_at.is_(None))
                         .values(applied_at=now)).rowcount
        if not took:
            return False
        other, f = pair["other"], pair["figures"]
        c.execute(insert(events_t).values(sleeve=other, ts=now, level="info", kind=PAIR_REBASED,
                                          message=self.pair_rebase_words(pair["parent"], pair["leg"], f["old"],
                                                                         f["summed"]),
                                          command_id=command_id))
        if f["last"] is not None:
            v = float(f["cash"][other])
            c.execute(insert(equity_t).values(sleeve=other, ts=now, equity=v, cash=v, qty=0.0,
                                              price=f["last"]["price"], benchmark=f["last"]["benchmark"]))
        from sleeve_fund import risk

        v = stored(f["summed"], "equity")
        c.execute(update(pair_state_t).where(pair_state_t.c.parent == pair["parent"],
                                             pair_state_t.c.marked_at.is_not(None))
                  .values(reference_equity=v, hwm=v, day_start=risk.trading_day(now), day_start_equity=v,
                          marked_at=now, updated_at=now))
        return True

    def apply_waiting_ral(self, sleeve: str) -> bool:
        """A reset after liquidation still waiting when its strategy was stopped (QA RAL-F2: Stop after the RAL): no
        process will apply it once the strategy is stopped and flat, so it is applied here as Store.command applies
        one sent to a stopped strategy. True if one was applied; a process that got there first wins. A two-leg
        Strategy: each leg's waiting reset, with the other leg's half that came with it (PAIR_REBASE)."""
        legs = self.legs(sleeve)
        if len(legs) > 1:
            return any([self._apply_waiting_pair_ral(legs, leg) for leg in legs])
        waiting = [c for c in self.pending_commands(sleeve) if c["command"] == RAL]
        if not waiting or (here := self._ral_here(sleeve, waiting[-1]["reason"], waiting[-1]["incident"])) is None:
            return False
        now = utcnow()
        with self.engine.begin() as c:
            if not self._apply_ral(c, sleeve, waiting[-1]["id"], here, now):
                return False
        self._lapse_resets_before(sleeve, now)
        return True

    def _apply_waiting_pair_ral(self, legs: list[str], leg: str) -> bool:
        waiting = [c for c in self.pending_commands(leg) if c["command"] == RAL and c["incident"] is not None]
        if not waiting:
            return False
        cmd, other = waiting[-1], next(n for n in legs if n != leg)
        half = [c for c in self.pending_commands(other) if c["command"] == RAL and c["incident"] is None
                and c["reason"].endswith(f"(command #{cmd['id']})")]
        pair = {"parent": legs[0], "leg": leg, "other": other}
        if (here := self._ral_here(leg, cmd["reason"], cmd["incident"], pair)) is None:
            return False
        now = utcnow()
        with self.engine.begin() as c:
            if not self._apply_ral(c, leg, cmd["id"], here, now):
                return False
            if half:
                self._apply_pair_rebase(c, pair, half[-1]["id"], now)
        self._lapse_resets_before(leg, now)
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
                                          message=here["words"], command_id=command_id))
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

    def _ral_here(self, sleeve: str, reason: str, incident: int, pair: dict | None = None) -> dict | None:
        """A reset after liquidation for a strategy no process will apply it for (CR on #193): stopped and flat, so
        the supervisor runs nothing, and Start is refused while it is liquidated. Returns what Store.command journals
        for it in the command's own transaction (the engine's words, and a mark at the remaining equity, so a later
        Start restores the new high-water mark and day baseline from it), else None for the process to apply. A
        two-leg Strategy's leg (`pair`, from _pair_ral): the words carry the pair's summed figures, and the figures
        the other leg's half writes (pair_figures) are read into `pair` here, before the transaction."""
        from sleeve_fund.paper.runtime import SleeveRuntime, liquidation_event, ral_words

        s = self.sleeve(sleeve)
        book = self.journal_book(sleeve, s.starting_balance)
        liq = liquidation_event(self, sleeve)
        if s.desired_state == "running" or abs(book["qty"]) > 1e-12 or liq is None:
            return None  # liq None: nothing left to reset (a book reset ended it), so it isn't applied (QA D-1)
        equity = book["cash"]
        last = self.equity_at_or_before(sleeve, utcnow())
        cmd = {"incident": incident, "reason": reason}
        if pair is not None:
            pair["figures"] = self.pair_figures(pair, utcnow())
            words = ral_words(self, sleeve, cmd, liq, pair["figures"]["old"], pair["figures"]["summed"])
        else:
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
            words = flattened_words(reason.strip()) if command == "flatten" else f"paused by PM: {reason.strip()}"
            c.execute(reset_holds_t.delete().where(reset_holds_t.c.reset_id == req.id))
            c.execute(insert(reset_holds_t).values(reset_id=req.id, status="paused", status_reason=words,
                                                   paused_until=None))
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
        names = [sleeve]
        if self.multi_asset(sleeve):
            # One settings block for every leg (multi-asset spec §2); the asset list is the Strategy's identity (QT3 b3c).
            if s.parent is not None:
                raise ValueError(f"{sleeve} is an asset leg of {s.parent}: change the Strategy's settings")
            for key in LEG_IDENTITY:
                if params.get(key) != s.params.get(key):
                    raise ValueError(f"a multi-asset Strategy's {key} is fixed when it is made: a new list is a new "
                                     "Strategy")
            names = self.legs(sleeve)
        if trial is not None:
            check_trial(trial)
        with self.engine.begin() as c:
            c.execute(update(sleeves_t).where(sleeves_t.c.name.in_(names)).values(
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
        """Retire a strategy's waiting commands unapplied, each noted in the decision log: a two-leg Strategy's
        second leg's too, as its commands are the first leg's (P2-8, CR F283-3)."""
        # A reset after liquidation doesn't lapse: its incident can't be answered twice, so it is applied instead
        # (apply_waiting_ral, or the process still running for its exits) (QA RAL-F2).
        n = 0
        for name in self.legs(sleeve):
            pending = [c for c in self.pending_commands(name) if c["command"] != RAL]
            for cmd in pending:
                self.mark_applied(cmd["id"])
                if cmd["command"] != RELOAD:  # saved settings don't lapse: the next start trades under them
                    self.decide("system", f"drop {cmd['command']}", f"{why} ({cmd['reason']})", name)
            n += len(pending)
        return n

    def command_by_id(self, command_id: int) -> dict | None:
        with self.engine.connect() as c:
            rows = _rows(c.execute(select(commands_t).where(commands_t.c.id == command_id)))
        return rows[0] if rows else None

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
