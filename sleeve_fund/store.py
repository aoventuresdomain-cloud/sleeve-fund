"""The one database every part reads from: sleeves, equity marks, fills, events,
PM commands and the decision log.

Postgres in production (the same engine Supabase runs, so pointing DATABASE_URL at
a Supabase project is a config change). SQLite works for tests and quick local
runs. Tables are plain and typed so reports and BI tools can query them directly.
"""

from __future__ import annotations

from decimal import Decimal

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
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
    create_engine,
    delete,
    event,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.engine import Engine

DEFAULT_URL = "sqlite:///data/sleeve_fund.db"

COMMANDS = {"pause", "resume", "flatten"}
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
    Column("starting_balance", Float, nullable=False),
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
    Column("equity", Float, nullable=False),
    Column("cash", Float, nullable=False),
    Column("qty", Float, nullable=False),
    Column("price", Float, nullable=False),
    Column("benchmark", Float, nullable=False),
    Index("equity_sleeve_ts", "sleeve", "ts"),
)

fills_t = Table(
    "fills",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("ts", TS, nullable=False),
    Column("side", String(8), nullable=False),
    Column("qty", Float, nullable=False),
    Column("price", Float, nullable=False),
    Column("fee", Float, nullable=False),
    Column("order_id", String(64), nullable=False),
    Column("trade_id", String(64), nullable=False),
    Index("fills_sleeve_ts", "sleeve", "ts"),
)

# A perpetual's funding, exchanged at each funding time while a position is held (sleeve_fund.markets):
# amount is what the strategy received (negative: paid), in the quote currency, booked to cash.
funding_t = Table(
    "funding",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sleeve", String(64), ForeignKey("sleeves.name"), nullable=False),
    Column("ts", TS, nullable=False),
    Column("qty", Float, nullable=False),
    Column("price", Float, nullable=False),
    Column("rate", Float, nullable=False),
    Column("amount", Float, nullable=False),
    Index("funding_sleeve_ts", "sleeve", "ts"),
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
    Column("command", String(16), nullable=False),
    Column("reason", Text, nullable=False),
    Column("created_at", TS, nullable=False),
    Column("applied_at", TS),
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
    Column("qty", Float, nullable=False),
    Column("status", String(16), nullable=False),  # one of ORDER_STATUSES
    Column("filled_qty", Float, nullable=False, default=0.0),
    Column("avg_px", Float),
    Column("fee", Float, nullable=False, default=0.0),
    Column("intent", String(16), nullable=False),  # one of INTENTS: what the order was for
    Column("reason", Text, nullable=False),  # plain English, e.g. "10-bar average crossed above 30-bar"
    Column("signal", JSON, nullable=False, default=dict),  # indicator values and price at the decision
    Column("message", Text, nullable=False, default=""),  # venue or risk-engine text on reject/cancel
    Index("orders_sleeve_ts", "sleeve", "ts"),
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
    Column("risk_amount", Float),  # the trade's 1R from now on: the larger of the entry's and this plan's
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
    Column("venue", String(16), nullable=False, default="kraken"),
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
# Events that say the strategy's own code raised: a handler, or the risk check's tick (see
# LongFlatStrategy._report).
ERROR_KINDS = ("handler_failed", "tick_failed")
# Backtest names can't collide with a strategy's: those are lower-case letters, digits and dashes.
BACKTEST_PREFIX = "bt:"
ORDER_STATUSES = ("submitted", "accepted", "partially_filled", "filled", "canceled", "rejected", "denied", "expired")
OPEN_ORDER_STATUSES = ("submitted", "accepted", "partially_filled")
INTENTS = ("entry", "exit", "stop_loss", "take_profit", "risk_halt", "risk_pause", "pm_flatten", "rebalance")


DUST = Decimal("1e-10")  # a position closer to flat than this is flat: the smallest lot is 1e-8


def replay_book(fills, starting_balance: float, funding: float = 0.0) -> dict:
    """Cash, signed position and average entry from fills in time order, plus funding received.

    Spot-style cash: a buy costs qty * price + fee and a sell returns qty * price - fee, so a short holds
    the sale's proceeds as cash against a negative position and equity is cash + qty * price on either
    side. Fills that add to a position average its entry; fills that reduce it leave the entry alone; a
    fill that crosses through flat opens the remainder at its own price. The position is summed in
    Decimal from each fill as written: a float sum of many XRP-sized fills carries noise of a few 1e-12
    that could tip a one-lot difference over reconcile's tolerance. A long-only strategy's journal never
    goes negative; if it does, the negative stays visible so reconciliation catches it."""
    cash, qty, entry, n = float(starting_balance), Decimal(0), None, 0
    for f in fills:
        n += 1
        price = float(f["price"])
        q = Decimal(repr(float(f["qty"])))
        sign = 1 if f["side"] == "BUY" else -1
        cash -= sign * float(f["qty"]) * price + float(f["fee"])
        new = qty + sign * q
        if abs(new) < DUST:  # float residue from a fill's own arithmetic, far below any lot
            new = Decimal(0)
        if new == 0:
            entry = None
        elif qty == 0 or (qty > 0) == (sign > 0):  # opening or adding
            entry = ((entry or 0.0) * float(abs(qty)) + float(q) * price) / float(abs(new))
        elif (new > 0) != (qty > 0):  # through flat: what is left opened at this fill's price
            entry = price
        qty = new
    return {"cash": cash + float(funding), "qty": float(qty), "entry_px": entry, "fills": n,
            "funding": float(funding)}


def is_backtest(name: str | None) -> bool:
    return bool(name) and name.startswith(BACKTEST_PREFIX)


def _not_backtest(col):
    """Rows that belong to paper or live (or to no strategy), not to a saved backtest."""
    return or_(col.is_(None), ~col.like(BACKTEST_PREFIX + "%"))


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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

    @classmethod
    def from_row(cls, row) -> "Sleeve":
        d = dict(row._mapping)
        for k in ("paused_until", "heartbeat_at", "created_at", "updated_at"):
            d[k] = _aware(d[k])
        d["params"] = dict(d["params"] or {})
        return cls(**d)


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
    ) -> Sleeve:
        ts = utcnow()
        with self.engine.begin() as c:
            c.execute(insert(sleeves_t).values(
                name=name, strategy=strategy, instrument=instrument, bar_spec=bar_spec, params=params or {},
                starting_balance=starting_balance, risk_profile=risk_profile, warmup_bars=warmup_bars,
                desired_state=desired_state, status="starting" if desired_state == "running" else "stopped",
                status_reason="", created_at=ts, updated_at=ts,
            ))
        return self.sleeve(name)

    def sleeve(self, name: str) -> Sleeve:
        with self.engine.connect() as c:
            row = c.execute(select(sleeves_t).where(sleeves_t.c.name == name)).first()
        if row is None:
            raise KeyError(f"no strategy {name!r}")
        return Sleeve.from_row(row)

    def sleeves(self, include_backtests: bool = False) -> list[Sleeve]:
        """Paper and live strategies. Saved backtests are left out unless asked for: they never run,
        count towards the book, or raise alerts."""
        q = select(sleeves_t).order_by(sleeves_t.c.created_at, sleeves_t.c.id)
        if not include_backtests:
            q = q.where(_not_backtest(sleeves_t.c.name))
        with self.engine.connect() as c:
            rows = c.execute(q).all()
        return [Sleeve.from_row(r) for r in rows]

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

    def heartbeat(self, name: str) -> None:
        with self.engine.begin() as c:
            c.execute(update(sleeves_t).where(sleeves_t.c.name == name).values(heartbeat_at=utcnow()))

    # --- journal -----------------------------------------------------------------

    def record_equity(self, sleeve: str, *, equity: float, cash: float, qty: float, price: float,
                      benchmark: float, ts: datetime | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(insert(equity_t).values(sleeve=sleeve, ts=ts or utcnow(), equity=equity, cash=cash,
                                              qty=qty, price=price, benchmark=benchmark))

    def record_fill(self, sleeve: str, *, side: str, qty: float, price: float, fee: float,
                    order_id: str, trade_id: str, ts: datetime | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(insert(fills_t).values(sleeve=sleeve, ts=ts or utcnow(), side=side, qty=qty, price=price,
                                             fee=fee, order_id=order_id, trade_id=trade_id))

    def record_order(self, sleeve: str, *, order_id: str, side: str, qty: float, intent: str, reason: str,
                     signal: dict | None = None, order_type: str = "MARKET", ts: datetime | None = None) -> None:
        if intent not in INTENTS:
            raise ValueError(f"bad intent {intent!r}")
        now = ts or utcnow()
        with self.engine.begin() as c:
            c.execute(insert(orders_t).values(sleeve=sleeve, order_id=order_id, ts=now, updated_at=now, side=side,
                                              order_type=order_type, qty=qty, status="submitted", filled_qty=0.0,
                                              fee=0.0, intent=intent, reason=reason, signal=signal or {},
                                              message=""))

    def update_order(self, order_id: str, *, status: str | None = None, message: str | None = None,
                     fill_qty: float = 0.0, fill_px: float | None = None, fee: float = 0.0,
                     qty: float | None = None) -> None:
        """Move an order on (accepted, cancelled, rejected...) or add a fill to it. Unknown ids are ignored:
        orders sent before this journal existed have no row."""
        if status is not None and status not in ORDER_STATUSES:
            raise ValueError(f"bad order status {status!r}")
        with self.engine.begin() as c:
            row = c.execute(select(orders_t).where(orders_t.c.order_id == order_id)).first()
            if row is None:
                return
            values = {"updated_at": utcnow()}
            if qty is not None:  # resized at the venue (a backtest's resting stop growing with its entry)
                values["qty"] = qty
            if fill_qty:
                filled = row.filled_qty + fill_qty
                values["avg_px"] = ((row.avg_px or 0.0) * row.filled_qty + fill_qty * fill_px) / filled
                values["filled_qty"] = filled
                values["fee"] = row.fee + fee
                values["status"] = "filled" if filled >= row.qty - 1e-12 else "partially_filled"
            if status is not None and row.status not in ("filled", "canceled", "rejected", "denied", "expired"):
                values["status"] = status  # a late "accepted" never reopens a finished order
            if message:
                values["message"] = message
            c.execute(update(orders_t).where(orders_t.c.order_id == order_id).values(**values))

    def orders(self, sleeve: str | None = None, statuses: tuple[str, ...] | None = None, limit: int = 500) -> list[dict]:
        q = select(orders_t)
        q = q.where(orders_t.c.sleeve == sleeve) if sleeve else q.where(_not_backtest(orders_t.c.sleeve))
        if statuses:
            q = q.where(orders_t.c.status.in_(statuses))
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(orders_t.c.ts.desc(), orders_t.c.id.desc()).limit(limit)))

    def order_counts(self, sleeve: str | None = None) -> dict[str, int]:
        q = select(orders_t.c.status, func.count()).group_by(orders_t.c.status)
        q = q.where(orders_t.c.sleeve == sleeve) if sleeve else q.where(_not_backtest(orders_t.c.sleeve))
        with self.engine.connect() as c:
            return {s: n for s, n in c.execute(q)}

    # --- accounts ------------------------------------------------------------------

    def _ensure_paper_account(self, c) -> None:
        if c.execute(select(accounts_t.c.name).where(accounts_t.c.name == "paper")).first() is None:
            c.execute(insert(accounts_t).values(name="paper", kind="paper", venue="kraken",
                                                note="Simulated money at live Kraken prices and fees", created_at=utcnow()))

    def create_account(self, name: str, kind: str, note: str = "") -> None:
        from sleeve_fund.accounts import KINDS, NAME_RE

        if not NAME_RE.fullmatch(name):
            raise ValueError("account name: lower-case letters, digits and dashes, 2 to 41 characters")
        if kind not in KINDS:
            raise ValueError(f"account kind must be one of {KINDS}")
        with self.engine.begin() as c:
            self._ensure_paper_account(c)
            if c.execute(select(accounts_t.c.name).where(accounts_t.c.name == name)).first():
                raise ValueError(f"an account called {name} already exists")
            c.execute(insert(accounts_t).values(name=name, kind=kind, venue="kraken", note=note, created_at=utcnow()))

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
        earlier book's. Their history stays and their pages still open; the book's figures leave them out."""
        start = self.book_start()
        if start is None:
            return {}
        return {name: at for name, at in self.archived().items() if at <= start}

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

    def latest_spread(self, venue: str, instrument: str) -> dict | None:
        q = (select(spreads_t).where(spreads_t.c.venue == venue.upper(), spreads_t.c.instrument == instrument)
             .order_by(spreads_t.c.measured_at.desc(), spreads_t.c.id.desc()).limit(1))
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def latest_fees(self, venue: str, account: str | None = None) -> dict | None:
        """The most recent fetched schedule for a venue (or one account), or None."""
        q = select(fee_schedules_t).where(fee_schedules_t.c.venue == venue.upper())
        if account:
            q = q.where(fee_schedules_t.c.account == account)
        with self.engine.connect() as c:
            rows = _rows(c.execute(q.order_by(fee_schedules_t.c.fetched_at.desc(), fee_schedules_t.c.id.desc()).limit(1)))
        return rows[0] if rows else None

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
            return c.execute(q).scalar()

    def max_drawdown(self, sleeve: str) -> float:
        """The deepest fall from a running peak over every mark kept, however many: the screens read only
        the latest marks, and a paper strategy marks every few seconds."""
        peak = func.max(equity_t.c.equity).over(order_by=(equity_t.c.ts, equity_t.c.id),
                                                rows=(None, 0)).label("peak")
        marks = select(equity_t.c.equity, peak).where(equity_t.c.sleeve == sleeve).subquery()
        q = select(func.max(1 - marks.c.equity / marks.c.peak)).where(marks.c.peak > 0)
        with self.engine.connect() as c:
            return float(c.execute(q).scalar() or 0.0)

    def day_open_equity(self, sleeve: str, day_start: datetime) -> float | None:
        """The equity the day opened at: the last mark before `day_start` (00:00 UTC), else the day's
        first mark. A restart reads it so the daily-loss guard keeps the day's real baseline."""
        before = (select(equity_t.c.equity).where(equity_t.c.sleeve == sleeve, equity_t.c.ts < day_start)
                  .order_by(equity_t.c.ts.desc(), equity_t.c.id.desc()).limit(1))
        first = (select(equity_t.c.equity).where(equity_t.c.sleeve == sleeve, equity_t.c.ts >= day_start)
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

    def journal_book(self, sleeve: str, starting_balance: float) -> dict:
        """Cash, signed position and average entry implied by the journal: the paper book's source of
        truth (replay_book), with the perp's funding booked to cash."""
        q = select(fills_t).where(fills_t.c.sleeve == sleeve).order_by(fills_t.c.ts, fills_t.c.id)
        with self.engine.connect() as c:
            fills = _rows(c.execute(q))
        return replay_book(fills, starting_balance, self.funding_total(sleeve))

    def record_funding(self, sleeve: str, *, qty: float, price: float, rate: float, amount: float,
                       ts: datetime | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(funding_t.insert().values(sleeve=sleeve, ts=ts or utcnow(), qty=qty, price=price, rate=rate,
                                                amount=amount))

    def funding(self, sleeve: str, limit: int = 1000) -> list[dict]:
        q = select(funding_t).where(funding_t.c.sleeve == sleeve).order_by(funding_t.c.ts.desc()).limit(limit)
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def funding_total(self, sleeve: str) -> float:
        with self.engine.connect() as c:
            return float(c.execute(select(func.coalesce(func.sum(funding_t.c.amount), 0.0))
                                   .where(funding_t.c.sleeve == sleeve)).scalar() or 0.0)

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

    def sleeve_events_since(self, sleeve: str, kinds: tuple[str, ...], after_id: int = 0) -> list[dict]:
        """One sleeve's events of these kinds newer than an id, oldest first."""
        q = (select(events_t).where(events_t.c.sleeve == sleeve, events_t.c.kind.in_(kinds), events_t.c.id > after_id)
             .order_by(events_t.c.id))
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

    def last_event(self, sleeve: str, kinds: tuple[str, ...]) -> dict | None:
        q = (select(events_t).where(events_t.c.sleeve == sleeve, events_t.c.kind.in_(kinds))
             .order_by(events_t.c.id.desc()).limit(1))
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
                      bar_spec: str | None = None) -> str:
        """Copy a backtest's in-memory journal (sleeve_fund.paper.journal) into these tables under its
        own name, with the result the backtest page shows. Returns the backtest's strategy name.
        `bar_spec` is the interval the PM picked, where the run's own bars only match its length."""
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
            for t in (equity_t, fills_t, funding_t, orders_t, events_t, exit_plans_t):
                c.execute(delete(t).where(t.c.sleeve.in_(old)))
            c.execute(delete(backtests_t).where(backtests_t.c.sleeve.in_(old)))
            c.execute(delete(sleeves_t).where(sleeves_t.c.name.in_(old)))
        return len(old)

    # --- PM commands and decisions ----------------------------------------------

    def command(self, sleeve: str, command: str, reason: str, actor: str = "PM") -> None:
        if command not in COMMANDS:
            raise ValueError(f"bad command {command!r}")
        if not reason.strip():
            raise ValueError("every command needs a reason")
        if is_backtest(sleeve):
            raise ValueError("a saved backtest takes no commands")
        self.sleeve(sleeve)  # raises if unknown
        with self.engine.begin() as c:
            c.execute(insert(commands_t).values(sleeve=sleeve, command=command, reason=reason.strip(),
                                                created_at=utcnow()))
        self.decide(actor, command, reason, sleeve)

    def change_settings(self, sleeve: str, *, risk_profile: str, params: dict, warmup_bars: int) -> bool:
        """Save new risk settings. A running strategy is restarted by the supervisor to trade under
        them (True); a stopped one picks them up when it next starts (False)."""
        s = self.sleeve(sleeve)
        if is_backtest(sleeve):
            raise ValueError("a saved backtest's settings are what it tested; run a new backtest instead")
        self._update_sleeve(sleeve, risk_profile=risk_profile, params=params, warmup_bars=warmup_bars)
        if s.desired_state != "running":
            return False
        with self.engine.begin() as c:
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
        pending = self.pending_commands(sleeve)
        for cmd in pending:
            self.mark_applied(cmd["id"])
            if cmd["command"] != RELOAD:  # saved settings don't lapse: the next start trades under them
                self.decide("system", f"drop {cmd['command']}", f"{why} ({cmd['reason']})", sleeve)
        return len(pending)

    def mark_applied(self, command_id: int) -> None:
        with self.engine.begin() as c:
            c.execute(update(commands_t).where(commands_t.c.id == command_id).values(applied_at=utcnow()))

    def decide(self, actor: str, action: str, reason: str, sleeve: str | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(insert(decisions_t).values(ts=utcnow(), actor=actor, action=action, sleeve=sleeve,
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
