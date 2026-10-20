"""PAGE-SIZE-T1 (DA, migration 0034; Advisor PAGE-SIZE, HoE option (a)): the columns each gate decision keeps for T1
(the loss at the stops per asset), which QD2's writer fills and the page reads: t1_used_before, t1_cap and risk_per_unit
(fund currency, as the check compared them) and size_share (approved x risk per unit over T1's room, a fraction; NULL
with no room). The database holds their shape: the first three all or none, size_share only with them, none below 0,
risk per unit above 0, and once written they never change. Older rows keep none. SQLite and Postgres
(TEST_DATABASE_URL). The writer's cells are QD2's."""
from __future__ import annotations

import importlib
import os
import uuid
from decimal import Decimal as D

import pytest
from sqlalchemy import create_engine, select, text, update
from sqlalchemy.exc import DatabaseError
from test_gate_leverage_figures import _decision, mig_engine  # noqa: F401

from sleeve_fund.store import Store, gate_decisions_t

COLUMNS = ("t1_used_before", "t1_cap", "risk_per_unit", "size_share")
# 1.57 approved at 300 a unit of a room of 800 - 15 = 785: size_share 471 / 785 = 0.6
GOOD = {"t1_used_before": D("15"), "t1_cap": D("800"), "risk_per_unit": D("300"), "size_share": D("0.6")}
NONE = dict.fromkeys(COLUMNS)


@pytest.fixture(params=["sqlite", "postgresql"])
def store(request):
    """Its own store, made in the cell: test_gate_leverage_figures drops its schemas when that module ends."""
    if request.param == "sqlite":
        yield Store.in_memory()
        return
    pg = os.environ.get("TEST_DATABASE_URL")
    if not pg:
        pytest.skip("Postgres only with TEST_DATABASE_URL")
    name = f"t1_{uuid.uuid4().hex[:12]}"
    admin = create_engine(pg)
    with admin.begin() as c:
        c.execute(text(f"CREATE SCHEMA {name}"))
    eng = create_engine(pg, connect_args={"options": f"-csearch_path={name}"})
    try:
        yield Store(engine=eng)
    finally:
        eng.dispose()
        with admin.begin() as c:
            c.execute(text(f"DROP SCHEMA IF EXISTS {name} CASCADE"))
        admin.dispose()


def _t1_of(store, did):
    with store.engine.connect() as c:
        r = c.execute(select(*(gate_decisions_t.c[k] for k in COLUMNS)).where(gate_decisions_t.c.id == did)).one()
    return dict(zip(COLUMNS, r))


def _shape(store):
    """The rules on the database `store` sits on."""
    for k in COLUMNS[:3]:  # two of the three is refused, whichever is missing, with or without size_share
        for over in ({k: None}, {k: None, "size_share": None}):
            with pytest.raises(DatabaseError):
                _decision(store, **{**GOOD, **over})
    with pytest.raises(DatabaseError):  # size_share alone
        _decision(store, size_share=D("0.5"))
    for k, bad in (("t1_used_before", D("-0.01")), ("t1_cap", D("-1")), ("risk_per_unit", D(0)),
                   ("size_share", D("-0.01"))):
        with pytest.raises(DatabaseError):
            _decision(store, **{**GOOD, k: bad})
    full = {**GOOD, "t1_used_before": D("800"), "size_share": None}  # no room left: the three, and no share
    assert _t1_of(store, _decision(store, **full)) == full
    did = _decision(store, **GOOD)
    for k, v in (("t1_cap", D(900)), ("size_share", D("0.5")), ("size_share", None)):
        with pytest.raises(DatabaseError), store.engine.begin() as c:
            c.execute(update(gate_decisions_t).where(gate_decisions_t.c.id == did).values({k: v}))
    assert _t1_of(store, did) == GOOD


def test_the_columns_are_all_or_none_never_below_zero_and_never_rewritten(store):
    _shape(store)
    assert _t1_of(store, _decision(store)) == NONE  # a row without them: T1 off, not reached, or an error


def test_0034_adds_them_from_0033_with_its_own_rules_and_refuses_to_drop_kept_ones(mig_engine):
    from alembic import command

    from sleeve_fund import schema

    with mig_engine.begin() as c:
        command.upgrade(schema._config(c), "0033")
    store = Store(engine=mig_engine)
    old = _decision(store)  # a row from before 0034
    schema.migrate(mig_engine, log=lambda _: None)
    assert _t1_of(store, old) == NONE
    _shape(store)  # the migration's own triggers, not create_all's
    with pytest.raises(RuntimeError, match="not dropped"), mig_engine.begin() as c:
        command.downgrade(schema._config(c), "0033")


def test_0034_downgrades_while_none_is_kept_and_upgrades_again(mig_engine):
    from alembic import command
    from sqlalchemy import inspect

    from sleeve_fund import schema

    schema.migrate(mig_engine, log=lambda _: None)
    _decision(Store(engine=mig_engine))  # a decision without them doesn't hold the downgrade
    with mig_engine.begin() as c:
        command.downgrade(schema._config(c), "0033")
    assert not set(COLUMNS) & {c["name"] for c in inspect(mig_engine).get_columns("gate_decisions")}
    with mig_engine.begin() as c:  # its triggers went with it (one left naming a dropped column would fail this)
        c.execute(text("UPDATE gate_decisions SET outcome = outcome"))
    schema.migrate(mig_engine, log=lambda _: None)
    assert set(COLUMNS) <= {c["name"] for c in inspect(mig_engine).get_columns("gate_decisions")}


def test_0034_writes_the_same_rules_and_columns_as_the_store():
    from sleeve_fund import store

    m = importlib.import_module("sleeve_fund.migrations.versions.0034_gate_t1_inputs")
    assert m.BAD == store.GATE_T1_BAD and m.MESSAGE == store.GATE_T1_MESSAGE
    assert m.FROZEN_PG == store.GATE_T1_FROZEN["postgresql"] and m.FROZEN_SQLITE == store.GATE_T1_FROZEN["sqlite"]
    assert m.PG == store.GATE_T1_PG and m.SQLITE == store.GATE_T1_SQLITE and m.COLUMNS == COLUMNS
    money = gate_decisions_t.c.book_equity.type
    for k in COLUMNS:  # the same Numeric(38, 18) as the other money columns, so the page multiplies like with like
        t = gate_decisions_t.c[k].type
        assert gate_decisions_t.c[k].nullable and (t.precision, t.scale) == (money.precision, money.scale) == (38, 18)
        assert (m.EXACT.precision, m.EXACT.scale) == (38, 18)
