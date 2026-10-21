"""B-24 leverage figures (DA, migration 0032; Advisor B24-LEVERAGE, QD). A gate decision keeps the four figures the
gate computed for it (margin leverage, share of the fund, gross after, net per underlying after) and what the Strategy
held before it (signed, so a fill's share of the fund can be worked from the decision alone), and each fill of it keeps
the four at the fill price in gate_fill_marks. The decision's five are all there or none (rows before 0032, error rows),
and once written never change; a fill mark is write-once. The database holds both, on SQLite and on
Postgres (TEST_DATABASE_URL). Expected figures are written out here, not computed by the code under test."""
from __future__ import annotations

import itertools
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, insert, select, text, update
from sqlalchemy.exc import DatabaseError

from sleeve_fund.store import Store, gate_decisions_t

PG = os.environ.get("TEST_DATABASE_URL")
FIGURES = ("margin_leverage", "share_of_fund", "gross_after", "net_underlying_after")
# A short: net per underlying is negative; the others are not. Exact on a float too, so SQLite keeps them as written.
LEV = {"margin_leverage": Decimal("2.25"), "share_of_fund": Decimal("0.0375"), "gross_after": Decimal("1.5125"),
       "net_underlying_after": Decimal("-0.4875")}
DEC = {**LEV, "held_before": Decimal("-0.25")}  # the gate writes all five: held before is signed too
DECISION = (*FIGURES, "held_before")
AT = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)
_SCHEMAS: list[str] = []
_ENGINES: list = []
_SEQ = itertools.count(1)  # gate_decisions.seq is unique: a refused row is refused for its figures alone


def _engine(schema: str | None = None):
    args = {"options": f"-csearch_path={schema}"} if schema else {}
    return create_engine(PG, connect_args=args)


@pytest.fixture(scope="module", autouse=True)
def _drop_schemas():
    yield
    for eng in _ENGINES:
        eng.dispose()
    if PG and _SCHEMAS:
        admin = _engine()
        for s in _SCHEMAS:
            with admin.begin() as c:
                c.execute(text(f"DROP SCHEMA IF EXISTS {s} CASCADE"))
        admin.dispose()


def _stores():
    out = [Store.in_memory()]
    if PG:
        name = f"da_{uuid.uuid4().hex[:12]}"
        with _engine().begin() as c:
            c.execute(text(f"CREATE SCHEMA {name}"))
        _SCHEMAS.append(name)
        eng = _engine(name)
        _ENGINES.append(eng)
        out.append(Store(engine=eng))
    return out


def _row(sid, **over):
    return {"seq": 1, "sleeve_id": sid, "bar_ts": AT, "intent_id": "i1", "intent": "open_short", "underlying": "BTC",
            "profile_version": 1, "outcome": "approved", "limit_hit": None, "requested_qty": Decimal("1"),
            "approved_qty": Decimal("1"), "price": Decimal("60000"), "stage": "submit", "decided_at": AT, **over}


def _decision(store, **over) -> int:
    sid = store.create_sleeve(name=f"s{uuid.uuid4().hex[:6]}", strategy="buy_and_hold", instrument="BTC/USD",
                              bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=1_000).id
    with store.engine.begin() as c:
        return c.execute(insert(gate_decisions_t).returning(gate_decisions_t.c.id),
                         [_row(sid, seq=next(_SEQ), **over)]).scalar_one()


def _figures(store, did):
    with store.engine.connect() as c:
        r = c.execute(select(*(gate_decisions_t.c[f] for f in DECISION)).where(gate_decisions_t.c.id == did)).one()
    return dict(zip(DECISION, r))


@pytest.mark.parametrize("store", _stores(), ids=lambda s: s.engine.dialect.name)
def test_a_decision_keeps_its_four_figures_exactly_and_older_rows_keep_none(store):
    did = _decision(store, **DEC)
    assert _figures(store, did) == DEC
    old = _decision(store)  # an error row, or a writer before 0032: all four NULL
    assert _figures(store, old) == dict.fromkeys(DECISION)


@pytest.mark.parametrize("store", _stores(), ids=lambda s: s.engine.dialect.name)
def test_the_four_are_all_there_or_none_and_only_net_may_be_below_zero(store):
    for f in DECISION:  # four of five is refused, whichever is missing
        with pytest.raises(DatabaseError):
            _decision(store, **{**DEC, f: None})
    for f in ("margin_leverage", "share_of_fund", "gross_after"):
        with pytest.raises(DatabaseError):
            _decision(store, **{**DEC, f: Decimal("-0.01")})
    assert _figures(store, _decision(store, **{**DEC, "margin_leverage": Decimal("0")}))["margin_leverage"] == 0
    assert _figures(store, _decision(store, **{**DEC, "held_before": Decimal("0.3")}))["held_before"] == Decimal("0.3")


@pytest.mark.parametrize("store", _stores(), ids=lambda s: s.engine.dialect.name)
def test_written_figures_never_change_but_the_rest_of_the_row_still_updates(store):
    did = _decision(store, **DEC)
    t = gate_decisions_t
    for f in DECISION:
        with pytest.raises(DatabaseError), store.engine.begin() as c:
            c.execute(update(t).where(t.c.id == did).values({f: Decimal("9")}))
    with pytest.raises(DatabaseError), store.engine.begin() as c:
        c.execute(update(t).where(t.c.id == did).values(dict.fromkeys(DECISION)))
    with store.engine.begin() as c:  # the rest of the row still updates, as before (the order id once sent)
        c.execute(update(t).where(t.c.id == did).values(limit_hit=None, regime_state="x"))
    assert _figures(store, did) == DEC
    old = _decision(store)  # a row without them may get them once, all four together
    with store.engine.begin() as c:
        c.execute(update(t).where(t.c.id == old).values(**DEC))
    assert _figures(store, old) == DEC


@pytest.mark.parametrize("store", _stores(), ids=lambda s: s.engine.dialect.name)
def test_a_fill_mark_is_written_once_per_fill_and_read_back_oldest_first(store):
    did = _decision(store, **DEC)
    second = {"margin_leverage": Decimal("2.5"), "share_of_fund": Decimal("0.04"), "gross_after": Decimal("1.6"),
              "net_underlying_after": Decimal("-0.5")}
    assert store.record_fill_mark(did, "t-1", Decimal("60010.5"), "fill", marked_at=AT, **LEV) == "new"
    assert store.record_fill_mark(did, "t-2", Decimal("60020"), "fill", marked_at=AT.replace(minute=5),
                                  **second) == "new"
    assert store.record_fill_mark(did, "t-1", Decimal("60010.5"), "fill", marked_at=AT, **LEV) == "same"
    with pytest.raises(ValueError, match="never rewritten"):
        store.record_fill_mark(did, "t-1", Decimal("60011"), "fill", marked_at=AT, **LEV)
    got = store.fill_marks(did)
    assert [(m["trade_id"], m["price"], m["price_source"]) for m in got] == [
        ("t-1", Decimal("60010.5"), "fill"), ("t-2", Decimal("60020"), "fill")]
    assert {f: got[0][f] for f in FIGURES} == LEV and {f: got[1][f] for f in FIGURES} == second
    assert store.fill_marks(_decision(store)) == []


@pytest.mark.parametrize("store", _stores(), ids=lambda s: s.engine.dialect.name)
def test_a_fill_mark_refuses_an_unknown_source_a_missing_figure_a_bad_price_or_decision(store):
    did = _decision(store, **DEC)
    for source in ("expected_fill", "mark"):  # the three the CHECK holds; the gate writes "fill"
        assert store.record_fill_mark(did, f"t-{source}", Decimal("60000"), source, **LEV) == "new"
    with pytest.raises(DatabaseError):
        store.record_fill_mark(did, "t-x", Decimal("60000"), "last", **LEV)
    with pytest.raises(DatabaseError):
        store.record_fill_mark(did, "t-y", Decimal("0"), "fill", **LEV)
    with pytest.raises(DatabaseError):
        store.record_fill_mark(did, "t-z", Decimal("60000"), "fill", **{**LEV, "gross_after": Decimal("-1")})
    with pytest.raises((TypeError, ValueError)):
        store.record_fill_mark(did, "t-n", Decimal("60000"), "fill", **{**LEV, "share_of_fund": None})
    with pytest.raises(LookupError):
        store.record_fill_mark(10**9, "t-1", Decimal("60000"), "fill", **LEV)


@pytest.mark.parametrize("store", _stores(), ids=lambda s: s.engine.dialect.name)
def test_a_fill_mark_rides_the_callers_transaction_and_the_database_keeps_it_write_once(store):
    did = _decision(store, **DEC)
    with pytest.raises(RuntimeError), store.engine.begin() as c:  # the fill path's own transaction (reduce)
        store.record_fill_mark(did, "t-1", Decimal("60000"), "fill", conn=c, **LEV)
        raise RuntimeError("the reservation's reduce failed")
    assert store.fill_marks(did) == []
    store.record_fill_mark(did, "t-1", Decimal("60000"), "fill", **LEV)
    for sql in ("UPDATE gate_fill_marks SET price = 1", "DELETE FROM gate_fill_marks"):
        with pytest.raises(DatabaseError), store.engine.begin() as c:
            c.execute(text(sql))
    assert len(store.fill_marks(did)) == 1


@pytest.fixture
def mig_engine(tmp_path):
    from sleeve_fund.store import make_engine, metadata

    eng = make_engine(PG or f"sqlite:///{tmp_path}/m.db")

    def clean():
        with eng.begin() as c:
            c.execute(text("DROP TABLE IF EXISTS alembic_version"))
        metadata.drop_all(eng)

    clean()
    yield eng
    clean()
    eng.dispose()


def test_0032_adds_the_figures_from_0031_with_the_migrations_own_rules_and_refuses_to_drop_kept_ones(mig_engine):
    from alembic import command

    from sleeve_fund import schema

    with mig_engine.begin() as c:
        command.upgrade(schema._config(c), "0031")
    store = Store(engine=mig_engine)
    old = _decision(store)  # a row from before 0032
    with mig_engine.begin() as c:  # to 0032 itself: a downgrade from head runs the later ones first (TEC, DA 0034)
        command.upgrade(schema._config(c), "0032")
    assert _figures(store, old) == dict.fromkeys(DECISION)
    for f in DECISION:  # the migration's own triggers, not create_all's: four of five is refused, whichever is missing
        with pytest.raises(DatabaseError):
            _decision(store, **{**DEC, f: None})
    did = _decision(store, **DEC)
    with pytest.raises(DatabaseError), mig_engine.begin() as c:
        c.execute(update(gate_decisions_t).where(gate_decisions_t.c.id == did).values(gross_after=Decimal("9")))
    with pytest.raises(RuntimeError, match="not dropped"), mig_engine.begin() as c:  # figures kept, no fill mark yet
        command.downgrade(schema._config(c), "0031")
    store.record_fill_mark(did, "t-1", Decimal("60000"), "fill", **LEV)
    with pytest.raises(DatabaseError), mig_engine.begin() as c:
        c.execute(text("DELETE FROM gate_fill_marks"))
    with pytest.raises(RuntimeError, match="not dropped"), mig_engine.begin() as c:
        command.downgrade(schema._config(c), "0031")
    assert store.fill_marks(did)[0]["trade_id"] == "t-1" and _figures(store, did) == DEC


def test_0032_downgrades_while_nothing_is_kept_and_upgrades_again(mig_engine):
    from alembic import command
    from sqlalchemy import inspect

    from sleeve_fund import schema

    schema.migrate(mig_engine, log=lambda _: None)
    _decision(Store(engine=mig_engine))  # a decision without figures doesn't hold the downgrade
    with mig_engine.begin() as c:
        command.downgrade(schema._config(c), "0031")
    assert "gate_fill_marks" not in inspect(mig_engine).get_table_names()
    assert not set(DECISION) & {c["name"] for c in inspect(mig_engine).get_columns("gate_decisions")}
    schema.migrate(mig_engine, log=lambda _: None)
    assert "gate_fill_marks" in inspect(mig_engine).get_table_names()
    assert set(DECISION) <= {c["name"] for c in inspect(mig_engine).get_columns("gate_decisions")}


def test_0032_writes_the_same_trigger_rules_as_the_store():
    """CR2: a database built by 0032 and one built by the store hold the same rules, written out twice (a migration is
    frozen); the SQL must match, or one of them drifts unseen."""
    import importlib

    from sleeve_fund import store

    m = importlib.import_module("sleeve_fund.migrations.versions.0032_gate_leverage_figures")
    assert m.BAD == store.GATE_LEVERAGE_BAD and m.MESSAGE == store.GATE_LEVERAGE_MESSAGE
    assert m.FROZEN_PG == store.GATE_LEVERAGE_FROZEN["postgresql"]
    assert m.FROZEN_SQLITE == store.GATE_LEVERAGE_FROZEN["sqlite"]
    assert m.PG == store.GATE_LEVERAGE_PG and m.SQLITE == store.GATE_LEVERAGE_SQLITE
    assert m.PG_MARKS == store.GATE_FILL_MARKS_PG and m.SQLITE_MARKS == store.GATE_FILL_MARKS_SQLITE
