"""The one database every part reads from: sleeves, equity marks, fills, events,
PM commands and the decision log.

Postgres in production (the same engine Supabase runs, so pointing DATABASE_URL at
a Supabase project is a config change). SQLite works for tests and quick local
runs. Tables are plain and typed so reports and BI tools can query them directly.
"""

from __future__ import annotations

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
    event,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

DEFAULT_URL = "sqlite:///data/sleeve_fund.db"

COMMANDS = {"pause", "resume", "flatten"}
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
        metadata.create_all(self.engine)

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
                desired_state=desired_state, status="starting", status_reason="", created_at=ts, updated_at=ts,
            ))
        return self.sleeve(name)

    def sleeve(self, name: str) -> Sleeve:
        with self.engine.connect() as c:
            row = c.execute(select(sleeves_t).where(sleeves_t.c.name == name)).first()
        if row is None:
            raise KeyError(f"no sleeve {name!r}")
        return Sleeve.from_row(row)

    def sleeves(self) -> list[Sleeve]:
        with self.engine.connect() as c:
            rows = c.execute(select(sleeves_t).order_by(sleeves_t.c.created_at, sleeves_t.c.id)).all()
        return [Sleeve.from_row(r) for r in rows]

    def _update_sleeve(self, name: str, **values) -> None:
        with self.engine.begin() as c:
            c.execute(update(sleeves_t).where(sleeves_t.c.name == name).values(updated_at=utcnow(), **values))

    def set_desired_state(self, name: str, state: str) -> None:
        if state not in {"running", "stopped"}:
            raise ValueError(f"bad desired state {state!r}")
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

    def event(self, sleeve: str | None, level: str, kind: str, message: str) -> None:
        if level not in LEVELS:
            raise ValueError(f"bad level {level!r}")
        with self.engine.begin() as c:
            c.execute(insert(events_t).values(sleeve=sleeve, ts=utcnow(), level=level, kind=kind, message=message))

    def equity_series(self, sleeve: str, limit: int = 5000) -> list[dict]:
        q = select(equity_t).where(equity_t.c.sleeve == sleeve).order_by(equity_t.c.ts.desc(), equity_t.c.id.desc())
        with self.engine.connect() as c:
            rows = _rows(c.execute(q.limit(limit)))
        return rows[::-1]

    def last_equity(self, sleeve: str) -> dict | None:
        rows = self.equity_series(sleeve, limit=1)
        return rows[0] if rows else None

    def first_equity(self, sleeve: str) -> dict | None:
        q = select(equity_t).where(equity_t.c.sleeve == sleeve).order_by(equity_t.c.ts, equity_t.c.id).limit(1)
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def peak_equity(self, sleeve: str) -> float | None:
        with self.engine.connect() as c:
            return c.execute(select(func.max(equity_t.c.equity)).where(equity_t.c.sleeve == sleeve)).scalar()

    def equity_at_or_before(self, sleeve: str, ts: datetime) -> dict | None:
        q = (select(equity_t).where(equity_t.c.sleeve == sleeve, equity_t.c.ts <= ts)
             .order_by(equity_t.c.ts.desc(), equity_t.c.id.desc()).limit(1))
        with self.engine.connect() as c:
            rows = _rows(c.execute(q))
        return rows[0] if rows else None

    def fills(self, sleeve: str | None = None, limit: int = 200) -> list[dict]:
        q = select(fills_t)
        if sleeve:
            q = q.where(fills_t.c.sleeve == sleeve)
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(fills_t.c.ts.desc(), fills_t.c.id.desc()).limit(limit)))

    def journal_book(self, sleeve: str, starting_balance: float) -> dict:
        """Cash, coin and average entry implied by the journal: the paper book's source of truth.

        Fees are charged in the quote currency (as Kraken spot does), so a buy costs
        qty * price + fee and a sell returns qty * price - fee.
        """
        q = select(fills_t).where(fills_t.c.sleeve == sleeve).order_by(fills_t.c.ts, fills_t.c.id)
        with self.engine.connect() as c:
            fills = _rows(c.execute(q))
        cash, qty, entry = float(starting_balance), 0.0, None
        for f in fills:
            notional = f["qty"] * f["price"]
            if f["side"] == "BUY":
                entry = ((entry or 0.0) * qty + notional) / (qty + f["qty"])
                cash -= notional + f["fee"]
                qty += f["qty"]
            else:
                cash += notional - f["fee"]
                qty -= f["qty"]
                if abs(qty) <= 1e-12:
                    qty = 0.0
                if qty <= 0:  # a negative qty is left visible so reconciliation catches it
                    entry = None
        return {"cash": cash, "qty": qty, "entry_px": entry, "fills": len(fills)}

    def events(self, sleeve: str | None = None, limit: int = 100, min_level: str = "info") -> list[dict]:
        q = select(events_t).where(events_t.c.level.in_(LEVELS[LEVELS.index(min_level):]))
        if sleeve:
            q = q.where(events_t.c.sleeve == sleeve)
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(events_t.c.id.desc()).limit(limit)))

    # --- PM commands and decisions ----------------------------------------------

    def command(self, sleeve: str, command: str, reason: str, actor: str = "PM") -> None:
        if command not in COMMANDS:
            raise ValueError(f"bad command {command!r}")
        if not reason.strip():
            raise ValueError("every command needs a reason")
        self.sleeve(sleeve)  # raises if unknown
        with self.engine.begin() as c:
            c.execute(insert(commands_t).values(sleeve=sleeve, command=command, reason=reason.strip(),
                                                created_at=utcnow()))
        self.decide(actor, command, reason, sleeve)

    def pending_commands(self, sleeve: str) -> list[dict]:
        q = (select(commands_t).where(commands_t.c.sleeve == sleeve, commands_t.c.applied_at.is_(None))
             .order_by(commands_t.c.id))
        with self.engine.connect() as c:
            return _rows(c.execute(q))

    def mark_applied(self, command_id: int) -> None:
        with self.engine.begin() as c:
            c.execute(update(commands_t).where(commands_t.c.id == command_id).values(applied_at=utcnow()))

    def decide(self, actor: str, action: str, reason: str, sleeve: str | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(insert(decisions_t).values(ts=utcnow(), actor=actor, action=action, sleeve=sleeve,
                                                 reason=reason.strip()))

    def decisions(self, sleeve: str | None = None, limit: int = 200) -> list[dict]:
        q = select(decisions_t)
        if sleeve:
            q = q.where(decisions_t.c.sleeve == sleeve)
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(decisions_t.c.id.desc()).limit(limit)))
