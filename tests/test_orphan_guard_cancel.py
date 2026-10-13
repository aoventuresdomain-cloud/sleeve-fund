"""ORPHAN-GUARD cancel (migration 0015; Advisor 8 Oct 06:15 and 06:20 UK): the supervisor's cancel names one resting
order, is open at most once per order, is never applied by a Strategy's process or lapsed, and is applied only on the
venue's answer, with the reservation and the order in the same transaction; "not found" with a fill in the journal
releases nothing. CI also runs these against Postgres."""

import ast
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path

import pytest
from sqlalchemy import func, insert, inspect, select, text, update
from sqlalchemy.exc import IntegrityError

import sleeve_fund.store as store_mod
from sleeve_fund import schema
from sleeve_fund.store import (
    CANCEL,
    COMMAND_KINDS,
    Store,
    commands_t,
    gate_decisions_t,
    gate_reservations_t,
    make_engine,
    metadata,
    orders_t,
    portfolio_profile_t,
    sleeves_t,
)
from sleeve_fund.store import _aware

T = datetime(2026, 10, 13, 9, tzinfo=timezone.utc)
RAL_KIND = "reset_after_liquidation"
MIGRATION = Path(store_mod.__file__).parent / "migrations" / "versions" / "0015_orphan_guard_cancel.py"


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db")
    _clean(eng)
    yield eng
    _clean(eng)


@pytest.fixture
def store(engine):
    s = Store(engine=engine)
    s.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT-PERP", bar_spec="1-HOUR-LAST-INTERNAL",
                    starting_balance=1_000)
    s.create_sleeve(name="other", strategy="buy_and_hold", instrument="BTC/USDT-PERP",
                    bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=1_000)
    return s


def _resting(store, order_id="E-1", qty="0.01", sleeve="s"):
    """A resting limit entry with its approved gate decision and the reservation it holds."""
    store.record_order(sleeve, order_id=order_id, side="BUY", qty=float(qty), intent="entry", reason="dip entry",
                       order_type="LIMIT", ts=T)
    with store.engine.begin() as c:
        sid = c.execute(select(sleeves_t.c.id).where(sleeves_t.c.name == sleeve)).scalar()
        version = c.execute(select(func.max(portfolio_profile_t.c.version))).scalar()
        seq = (c.execute(select(func.max(gate_decisions_t.c.seq))).scalar() or 0) + 1
        did = c.execute(insert(gate_decisions_t).values(
            seq=seq, sleeve_id=sid, bar_ts=T, intent_id=order_id, intent="entry", underlying="BTC", order_id=order_id,
            profile_version=version, outcome="approved", requested_qty=D(qty), approved_qty=D(qty), price=D("60000"),
            stage="submit", decided_at=T)).inserted_primary_key[0]
        c.execute(insert(gate_reservations_t).values(
            decision_id=did, sleeve_id=sid, underlying="BTC", remaining_qty=D(qty), notional=D(qty) * 60000,
            margin=D(qty) * 60000, open_risk=D(qty) * 600, created_at=T, expires_at=T))
    return did


def _reservation(store, did):
    with store.engine.connect() as c:
        return c.execute(select(gate_reservations_t).where(gate_reservations_t.c.decision_id == did)).first()


def _order(store, order_id="E-1"):
    with store.engine.connect() as c:
        return c.execute(select(orders_t).where(orders_t.c.order_id == order_id)).first()


def _cancel_rows(store):
    with store.engine.connect() as c:
        return c.execute(select(commands_t).where(commands_t.c.command == CANCEL)).fetchall()


def test_the_kinds_the_code_writes_and_the_migrations_check_are_one_list():
    tree = ast.parse(MIGRATION.read_text())
    kinds = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", None) == "KINDS")
    assert kinds == COMMAND_KINDS
    assert {*store_mod.COMMANDS, store_mod.RELOAD, CANCEL} == set(COMMAND_KINDS)


def test_the_database_refuses_a_cancel_without_its_order_an_order_on_anything_else_and_an_unknown_kind(store):
    _resting(store)
    bad = [dict(command=CANCEL), dict(command="pause", order_id="E-1"), dict(command="sell_everything"),
           dict(command="pause", not_found_at=T)]
    for row in bad:
        with pytest.raises(IntegrityError), store.engine.begin() as c:
            c.execute(insert(commands_t).values(sleeve="s", reason="x", created_at=T, **row))
    assert _cancel_rows(store) == []


def test_one_open_cancel_per_order_and_asking_again_returns_the_first(store):
    _resting(store)
    first = store.request_cancel("s", "E-1", "owner process down")
    assert store.request_cancel("s", "E-1", "second sweep") == first
    with pytest.raises(IntegrityError), store.engine.begin() as c:  # the database itself refuses a second
        c.execute(insert(commands_t).values(sleeve="s", command=CANCEL, order_id="E-1", reason="x", created_at=T))
    assert [r.id for r in _cancel_rows(store)] == [first]
    assert [c["id"] for c in store.open_cancels("s")] == [first]
    # one decision-log line, written with the cancel itself (CR F241-2)
    assert [d["reason"] for d in store.decisions("s", action=CANCEL)] == ["owner process down (order E-1)"]


def test_a_cancel_whose_decision_line_fails_is_not_written_either(store, monkeypatch):
    """CR F241-2: the cancel and its decision-log line commit together, so a crash between them can't leave one."""
    import sleeve_fund.store as store_mod

    _resting(store)
    real = store_mod.insert

    def failing(table):
        if table is store_mod.decisions_t:
            raise RuntimeError("crash before the log line")
        return real(table)

    monkeypatch.setattr(store_mod, "insert", failing)
    with pytest.raises(RuntimeError, match="log line"):
        store.request_cancel("s", "E-1", "owner process down")
    monkeypatch.undo()
    assert _cancel_rows(store) == [] and store.decisions("s", action=CANCEL) == []


def test_a_cancel_for_another_strategys_order_an_unknown_one_or_a_finished_one_is_refused(store):
    _resting(store)
    _resting(store, "E-2")
    store.update_order("E-2", status="rejected")
    for sleeve, oid in (("other", "E-1"), ("s", "nope"), ("s", "E-2")):
        with pytest.raises(ValueError):
            store.request_cancel(sleeve, oid, "sweep")
    assert _cancel_rows(store) == []


def test_a_strategys_process_never_sees_a_cancel_and_nothing_lapses_or_marks_it_applied(store):
    _resting(store)
    cid = store.request_cancel("s", "E-1", "owner process down")
    store.command("s", "pause", "PM pause")
    assert [c["command"] for c in store.pending_commands("s")] == ["pause"]
    store.drop_pending("s", "lapsed: the strategy was stopped")
    store.mark_applied(cid)
    assert [c["id"] for c in store.open_cancels("s")] == [cid]


@pytest.mark.parametrize("not_found_first", [False, True], ids=["confirmed", "not-found-then-confirmed"])
def test_the_venues_answer_releases_the_reservation_and_ends_the_order_with_the_cancel_in_one_go(store, not_found_first):
    """Released only on the venue's confirmed end: a confirmed cancel, or after "not found" its own order query and fill
    history showing a terminal state with zero fills (Advisor 13 Oct 17:00 UK)."""
    did = _resting(store)
    cid = store.request_cancel("s", "E-1", "owner process down")
    if not_found_first:
        assert store.apply_cancel(cid, "not_found") == "unconfirmed"
    assert store.apply_cancel(cid, "cancelled") == "applied"
    r = _reservation(store, did)
    assert r.released_at is not None and r.release_reason == "cancel"
    assert _order(store).status == "canceled"
    assert store.open_cancels() == []
    assert store.apply_cancel(cid, "cancelled") == store.apply_cancel(cid, "not_found") == "already"
    # once applied, a later cancel for the same order may be opened (the index covers open ones only), but the order
    # is no longer resting, so none is
    with pytest.raises(ValueError, match="not resting"):
        store.request_cancel("s", "E-1", "again")


def test_not_found_with_a_fill_in_the_journal_releases_nothing_and_leaves_the_cancel_open(store):
    did = _resting(store)
    cid = store.request_cancel("s", "E-1", "owner process down")
    store.record_fill("s", side="BUY", qty=0.004, price=60_000.0, fee=0.1, order_id="E-1", trade_id="T-1", ts=T)
    before = (_reservation(store, did), _order(store))
    assert store.apply_cancel(cid, "not_found") == "filled"
    assert (_reservation(store, did), _order(store)) == before
    assert [c["id"] for c in store.open_cancels()] == [cid]


def test_not_found_with_no_fill_releases_nothing_however_long_the_journal_stays_silent(store):
    """Advisor 13 Oct 17:00 UK (HoQA og2c): silence in our own journal is no confirmation. The first answer's time is
    kept on the supervisor's clock (a second doesn't move it), and the cancel stays open, which is what blocks the
    Strategy's entries and what the bar close + 2 min alert reads, until the venue settles the order's end."""
    did = _resting(store)
    cid = store.request_cancel("s", "E-1", "owner process down")
    before = (_reservation(store, did), _order(store))
    assert store.apply_cancel(cid, "not_found", at=T) == "unconfirmed"
    assert store.apply_cancel(cid, "not_found", at=T + timedelta(hours=6)) == "unconfirmed"
    assert [(c["id"], _aware(c["not_found_at"])) for c in store.open_cancels("s")] == [(cid, T)]
    assert (_reservation(store, did), _order(store)) == before
    assert not hasattr(store, "confirm_cancels")  # no timer releases it


def test_a_fill_found_after_not_found_is_never_released_as_cancel_and_closes_as_filled(store):
    """og2c, fill second: cancel -> not found -> the venue's fill. Once the fills are booked (each reduces the
    reservation, W2) the caller answers "filled": applied, nothing released as 'cancel'."""
    did = _resting(store)
    cid = store.request_cancel("s", "E-1", "owner process down")
    with pytest.raises(ValueError, match="no fill"):
        store.apply_cancel(cid, "filled")
    assert store.apply_cancel(cid, "not_found") == "unconfirmed"
    store.record_fill("s", side="BUY", qty=0.01, price=60_000.0, fee=0.1, order_id="E-1", trade_id="T-1", ts=T)
    assert store.apply_cancel(cid, "not_found") == "filled"
    assert _reservation(store, did).release_reason != "cancel"
    assert store.apply_cancel(cid, "filled", at=T) == "applied"
    assert store.apply_cancel(cid, "filled") == "already" and store.open_cancels() == []
    assert _reservation(store, did).release_reason != "cancel"


def test_a_confirmed_cancel_of_a_partly_filled_order_releases_what_no_fill_used(store):
    did = _resting(store)
    cid = store.request_cancel("s", "E-1", "owner process down")
    store.record_fill("s", side="BUY", qty=0.004, price=60_000.0, fee=0.1, order_id="E-1", trade_id="T-1", ts=T)
    assert store.apply_cancel(cid, "cancelled") == "partly_filled"
    assert _reservation(store, did).release_reason == "cancel"
    assert _order(store).status == "canceled"


def test_a_failure_inside_the_answer_leaves_reservation_order_and_cancel_untouched(store, monkeypatch):
    did = _resting(store)
    cid = store.request_cancel("s", "E-1", "owner process down")
    before = (_reservation(store, did), _order(store))
    real = store.engine.begin

    class Boom(Exception):
        pass

    def begin():  # fail on the order update, after the command and the reservation were written in this transaction
        ctx = real()
        conn = ctx.__enter__()
        orig = conn.execute

        def execute(stmt, *a, **k):
            if getattr(stmt, "table", None) is orders_t and stmt.is_update:
                raise Boom
            return orig(stmt, *a, **k)

        conn.execute = execute

        class Ctx:
            def __enter__(self):
                return conn

            def __exit__(self, *exc):
                return ctx.__exit__(*exc)

        return Ctx()

    monkeypatch.setattr(store.engine, "begin", begin)
    with pytest.raises(Boom):
        store.apply_cancel(cid, "cancelled")
    monkeypatch.undo()
    assert (_reservation(store, did), _order(store)) == before
    assert [c["id"] for c in store.open_cancels()] == [cid]


def test_0015_stops_on_a_command_of_unknown_kind_and_changes_nothing(engine):
    from alembic import command

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0014")
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, starting_balance, created_at, "
                          "updated_at, desired_state, status, status_reason, risk_profile, params, warmup_bars) "
                          "SELECT 's', 'buy_and_hold', 'BTC/USDT-PERP', 'b', 1000, :t, :t, 'stopped', 'stopped', '', "
                          "'balanced', '{}', 0"), {"t": T})
        conn.execute(text("INSERT INTO commands (sleeve, command, reason, created_at) VALUES "
                          "('s', 'legacy_thing', 'x', :t)"), {"t": T})
    with pytest.raises(Exception, match="legacy_thing"):
        schema.migrate(engine, log=lambda _: None)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0014"
    assert "order_id" not in {c["name"] for c in inspect(engine).get_columns("commands")}
    with engine.begin() as conn:  # to head, so teardown's drop_all finds every constraint the code names
        conn.execute(text("DELETE FROM commands WHERE command = 'legacy_thing'"))
    schema.migrate(engine, log=lambda _: None)


def test_0015_migrated_commands_equal_create_alls(engine, tmp_path):
    def shape(eng):
        i = inspect(eng)
        return (sorted((c["name"], c["nullable"]) for c in i.get_columns("commands")),
                sorted(c["name"] for c in i.get_check_constraints("commands")),
                sorted((x["name"], tuple(x["column_names"]), bool(x["unique"])) for x in i.get_indexes("commands")),
                sorted((tuple(f["constrained_columns"]), f["referred_table"]) for f in i.get_foreign_keys("commands")))

    schema.migrate(engine, log=lambda _: None)
    migrated = shape(engine)
    assert ("gate_decisions_order_id", ("order_id",), False) in [
        (x["name"], tuple(x["column_names"]), bool(x["unique"])) for x in inspect(engine).get_indexes("gate_decisions")]
    fresh = make_engine(f"sqlite:///{tmp_path}/fresh.db") if engine.dialect.name == "sqlite" else engine
    if fresh is engine:
        _clean(engine)
    Store(engine=fresh)
    assert migrated == shape(fresh)
    assert {"commands_kind", "commands_cancel_order", "commands_not_found_cancel"} <= set(migrated[1])
    assert ("commands_cancel_open", ("order_id",), True) in migrated[2]
    assert "gate_decisions_order_id" in {i["name"] for i in inspect(engine).get_indexes("gate_decisions")}


def test_0015_keeps_a_reset_keyed_to_its_command_when_it_rebuilds_commands(engine):
    """0014's events.command_id references commands; 0015's batch rebuild of commands (SQLite) must keep that row
    and its key valid."""
    from alembic import command

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0014")
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, starting_balance, created_at, "
                          "updated_at, desired_state, status, status_reason, risk_profile, params, warmup_bars) "
                          "SELECT 's', 'buy_and_hold', 'BTC/USDT-PERP', 'b', 1000, :t, :t, 'stopped', 'stopped', '', "
                          "'balanced', '{}', 0"), {"t": T})
        cid = conn.execute(text("INSERT INTO commands (sleeve, command, reason, created_at, applied_at) VALUES "
                                "('s', 'reset_after_liquidation', 'x', :t, :t) RETURNING id"), {"t": T}).scalar()
        conn.execute(text("INSERT INTO events (sleeve, ts, level, kind, message, command_id) VALUES "
                          "('s', :t, 'info', 'liquidation_reset', 'reset', :c)"), {"t": T, "c": cid})
    schema.migrate(engine, log=lambda _: None)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT command_id FROM events WHERE kind = 'liquidation_reset'")).scalar() == cid
        assert conn.execute(text("SELECT command FROM commands WHERE id = :c"), {"c": cid}).scalar() == RAL_KIND
        if engine.dialect.name == "sqlite":
            assert conn.execute(text("PRAGMA foreign_key_check")).fetchall() == []


def test_a_reset_is_refused_while_a_cancel_is_open_and_runs_once_the_venue_answers(store):
    """F241-4 (HoE 13 Oct, after #241): split_run itself refuses a Strategy with an open cancel, in its own
    transaction, so a cancel requested after the supervisor's check (F241-3) can't move to the archived run and leave
    the fresh run trading with nothing blocking it. Refused before anything is written."""
    _resting(store)
    cid = store.request_cancel("s", "E-1", "owner process down")
    store.request_reset("s", "Settings changed")
    req = store.pending_resets()[0]
    with pytest.raises(ValueError, match="cancel the venue hasn't answered"):
        store.split_run(req)
    assert store.archived() == {} and store.pending_resets()[0]["id"] == req["id"]
    assert _order(store).sleeve == "s" and [c.sleeve for c in _cancel_rows(store)] == ["s"]
    assert [c["id"] for c in store.open_cancels("s")] == [cid]

    assert store.apply_cancel(cid, "cancelled") == "applied"
    run = store.split_run(store.pending_resets()[0])
    assert run in store.archived() and store.open_cancels() == []
    assert [c.sleeve for c in _cancel_rows(store)] == [run]  # the answered cancel goes with its run


def test_another_strategys_open_cancel_does_not_refuse_a_reset(store):
    _resting(store, sleeve="other")
    store.request_cancel("other", "E-1", "owner process down")
    store.request_reset("s", "Settings changed")
    assert store.split_run(store.pending_resets()[0]) in store.archived()
