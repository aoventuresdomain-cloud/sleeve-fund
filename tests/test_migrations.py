"""Schema migrations (sleeve_fund.schema.migrate, sleeve_fund/migrations): the newest migration builds exactly
store.py's schema, an existing database without history is stamped and never rebuilt, and a drifted one stops
the deploy with nothing changed. CI also runs these against Postgres via TEST_DATABASE_URL."""

import os

import pytest
from sqlalchemy import event, inspect, text

from sleeve_fund import schema
from sleeve_fund.store import Store, make_engine, metadata


def _clean(eng):
    with eng.begin() as conn:
        for t in ("alembic_version", "stray"):
            conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
    metadata.drop_all(eng)


@pytest.fixture
def engine(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    eng = make_engine(url or f"sqlite:///{tmp_path}/t.db")
    _clean(eng)
    yield eng
    _clean(eng)


def _statements(eng):
    seen = []
    event.listen(eng, "before_cursor_execute", lambda *a: seen.append(" ".join(a[2].split()[:3]).upper()))
    return seen


def _version(eng):
    with eng.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def test_an_empty_database_is_built_by_the_migrations_and_matches_the_code(engine):
    # This is the "every schema change has a migration" check: a table or column added to store.py
    # without one makes the migrated database differ from the code here.
    done = schema.migrate(engine, log=lambda _: None)
    assert done == f"at migration {schema.head()}" and _version(engine) == schema.head()
    _, diffs = schema.report(engine)
    assert diffs == []


def test_an_existing_database_without_history_is_stamped_not_rebuilt(engine):
    Store(engine=engine)  # what production is today: tables made by create_all, no migration history
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO accounts (name, kind, venue, note, created_at) "
                          "VALUES ('paper', 'paper', 'X', '', CURRENT_TIMESTAMP)"))
    seen = _statements(engine)
    done = schema.migrate(engine, log=lambda _: None)
    assert done.startswith("stamped existing schema")
    assert _version(engine) == schema.head()
    touched = [s for s in seen if s.split()[0] in ("CREATE", "ALTER", "DROP", "DELETE", "UPDATE", "TRUNCATE")]
    assert all("ALEMBIC_VERSION" in s for s in touched), touched  # only alembic's own bookkeeping table
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM accounts")).scalar() == 1  # the data is untouched


def test_a_drifted_database_without_history_stops_the_migration_and_nothing_changes(engine):
    Store(engine=engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE fills ADD COLUMN extra INTEGER"))
    with pytest.raises(schema.Drift, match="fills.extra"):
        schema.migrate(engine, log=lambda _: None)
    assert "alembic_version" not in inspect(engine).get_table_names()


def test_trials_made_before_0003_become_ok_rows_and_a_status_outside_the_two_is_refused(engine):
    # QA P1-T8 (Data Architect): status is NOT NULL with a server default, so every row before it reads "ok".
    from alembic import command
    from sqlalchemy.exc import IntegrityError

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0002")
        conn.execute(text("INSERT INTO trials (id, definition_hash, idea_hash, code_version, definition_name, family, "
                          "settings, dataset, stage, source, created_at) VALUES ('old', 'd', 'i', 'c', 'n', 'f', "
                          "'{}', 'ds', 'in_sample', 'study', CURRENT_TIMESTAMP)"))
    schema.migrate(engine, log=lambda _: None)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT status, error FROM trials WHERE id = 'old'")).one() == ("ok", None)
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text("UPDATE trials SET status = 'pending' WHERE id = 'old'"))


def test_migrating_again_changes_nothing(engine):
    schema.migrate(engine, log=lambda _: None)
    seen = _statements(engine)
    schema.migrate(engine, log=lambda _: None)
    assert not [s for s in seen if s.split()[0] in ("CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE")]


def test_cli_holds_migrations_on_first_stamp_drift_and_raises_an_alert(engine, monkeypatch, capfd):
    # Before the first stamp the app runs as it always has, so the deploy goes on; the drift is an error
    # event in the alerts inbox, and nothing is stamped or migrated until it is fixed.
    store = Store(engine=engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE stray (x INTEGER)"))
    monkeypatch.setattr(schema, "make_engine", lambda: engine)
    assert schema.main(["migrate"]) == 0
    assert "migrations held, nothing changed" in capfd.readouterr().err
    (ev,) = store.events_of(("schema_drift",))
    assert ev["level"] == "error" and "table stray" in ev["message"]
    assert "alembic_version" not in inspect(engine).get_table_names()


def test_cli_reports_drift_after_migrating_too(engine, monkeypatch, capfd):
    schema.migrate(engine, log=lambda _: None)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE stray (x INTEGER)"))  # a hand edit on the server
    monkeypatch.setattr(schema, "make_engine", lambda: engine)
    assert schema.main(["migrate"]) == 0
    assert "the migrated database differs" in capfd.readouterr().err
    (ev,) = Store(engine=engine).events_of(("schema_drift",))
    assert "table stray" in ev["message"]


def test_a_clean_migrate_raises_no_alert(engine, monkeypatch):
    monkeypatch.setattr(schema, "make_engine", lambda: engine)
    assert schema.main(["migrate"]) == 0
    assert Store(engine=engine).events_of(("schema_drift",)) == []


def test_cli_migrate_fails_the_deploy_when_a_migration_fails(engine, monkeypatch):
    schema.migrate(engine, log=lambda _: None)
    monkeypatch.setattr(schema.command, "upgrade", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(schema, "make_engine", lambda: engine)
    with pytest.raises(RuntimeError):
        schema.main(["migrate"])  # uncaught: the process exits non-zero and the services don't start


def test_a_liquidation_incident_is_answered_by_one_command_only(engine):
    """0007 (P1-RAL): commands.command holds 'reset_after_liquidation' (23 characters), and commands.incident, the
    liquidation event a reset answers, is unique where set: a retry or a double click can't reset twice."""
    from sqlalchemy import insert
    from sqlalchemy.exc import IntegrityError

    from sleeve_fund.store import commands_t, utcnow

    schema.migrate(engine)
    store = Store(engine=engine)
    store.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=1_000)
    store.event("s", "error", "liquidation", "liquidated")
    (inc,) = [e["id"] for e in store.events("s", limit=5) if e["kind"] == "liquidation"]
    row = dict(sleeve="s", command="reset_after_liquidation", reason="r", created_at=utcnow(), incident=inc)
    with engine.begin() as c:
        c.execute(insert(commands_t).values(**row))
    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(insert(commands_t).values(**row))
    with engine.begin() as c:  # commands without an incident are unaffected
        c.execute(insert(commands_t).values(**{**row, "command": "pause", "incident": None}))
        c.execute(insert(commands_t).values(**{**row, "command": "resume", "incident": None}))
    if engine.dialect.name == "postgresql":  # SQLite here doesn't enforce foreign keys
        with pytest.raises(IntegrityError), engine.begin() as c:  # the incident must be a real event
            c.execute(insert(commands_t).values(**{**row, "incident": 10**9}))


def _gate_row(sleeve_id, **over):
    from datetime import datetime, timezone
    from decimal import Decimal

    from sleeve_fund.store import utcnow

    return {"seq": 1, "sleeve_id": sleeve_id, "bar_ts": datetime(2026, 10, 7, tzinfo=timezone.utc), "intent_id": "i1",
            "intent": "open_long", "underlying": "BTC", "profile_version": 1, "outcome": "approved", "limit_hit": None,
            "requested_qty": Decimal("1"), "approved_qty": Decimal("1"), "price": Decimal("100"),
            "regime_weight": Decimal("0.5"), "stage": "submit", "decided_at": utcnow(), **over}


def test_0010_seeds_the_pms_accepted_limits_and_profiles_are_append_only(engine):
    """P2-2: the migration and create_all both hold version 1, the limits the PM accepted (risk.PORTFOLIO); a change
    is a new version and an old one is never rewritten (Store has no update for it)."""
    from sleeve_fund.risk import PORTFOLIO, PortfolioProfile

    schema.migrate(engine)
    store = Store(engine=engine)
    assert store.portfolio_profile() == PORTFOLIO == store.portfolio_profile(1)
    got = store.portfolio_profile()  # Numeric(10,4) back as the floats the core's ratios are written in
    assert (got.gross, got.net_instrument, got.margin, got.open_risk, got.drawdown, got.daily_loss) == \
        (1.5, 0.5, 0.5, 0.05, 0.15, 0.03)
    assert Store.in_memory().portfolio_profile() == PORTFOLIO
    v2 = store.add_portfolio_profile(PortfolioProfile(version=9, gross=1.2), created_by="pm", note="tighter gross")
    assert v2 == 2 and store.portfolio_profile() == PortfolioProfile(version=2, gross=1.2)
    assert store.portfolio_profile(1) == PORTFOLIO
    with pytest.raises(LookupError):
        store.portfolio_profile(3)
    assert [m for m in dir(Store) if "portfolio_profile" in m] == ["add_portfolio_profile", "portfolio_profile"]


def test_0010_refuses_names_and_limits_outside_the_spec(engine):
    """The CHECKs of v2/p2-2-tables.md: a gate decision's outcome, limit and stage, a reservation's release reason,
    the supervisor's single row and status, and a profile whose halt is not above its pause."""
    from datetime import date
    from decimal import Decimal

    from sqlalchemy import insert
    from sqlalchemy.exc import IntegrityError

    from sleeve_fund.store import (PORTFOLIO_PROFILE_V1, gate_decisions_t, gate_reservations_t, portfolio_profile_t,
                                   portfolio_state_t, utcnow)

    schema.migrate(engine)
    store = Store(engine=engine)
    sid = store.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USD",
                              bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=1_000).id

    def refused(table, row):
        with pytest.raises(IntegrityError), engine.begin() as c:
            c.execute(insert(table), [row])

    for bad in ({"outcome": "ok"}, {"limit_hit": "portfolio_halt"}, {"stage": "exit"}):
        refused(gate_decisions_t, _gate_row(sid, **bad))
    with engine.begin() as c:
        did = c.execute(insert(gate_decisions_t).returning(gate_decisions_t.c.id),
                        [_gate_row(sid, outcome="trimmed", limit_hit="below_min")]).scalar_one()
    refused(gate_decisions_t, _gate_row(sid, seq=2))  # one decision per strategy, bar, intent and stage
    now = utcnow()
    res = {"decision_id": did, "sleeve_id": sid, "underlying": "BTC", "remaining_qty": Decimal("1"),
           "notional": Decimal("100"), "margin": Decimal("50"), "open_risk": Decimal("2"), "created_at": now,
           "expires_at": now, "released_at": now}
    refused(gate_reservations_t, {**res, "release_reason": "expired"})
    with engine.begin() as c:
        c.execute(insert(gate_reservations_t), [{**res, "release_reason": "ttl"}])
    state = {"id": 1, "status": "ok", "reference_equity": Decimal("1000"), "hwm": Decimal("1000"),
             "day_start_equity": Decimal("1000"), "day_start": date(2026, 10, 7), "book_equity": Decimal("1000"),
             "marked_at": now, "profile_version": 1, "updated_at": now}
    refused(portfolio_state_t, {**state, "id": 2})
    refused(portfolio_state_t, {**state, "status": "stopped"})
    refused(portfolio_state_t, {**state, "status": "paused"})  # paused needs its paused_until
    refused(portfolio_state_t, {**state, "status": "halted"})  # halted needs its reason
    refused(portfolio_profile_t, {**PORTFOLIO_PROFILE_V1, "version": 2, "drawdown_halt": Decimal("0.03")})
    with engine.begin() as c:
        c.execute(insert(portfolio_state_t), [state])


def test_0010_refuses_impossible_quantities_and_half_released_reservations(engine):
    """DA's review: approved is 0..requested and 0 when rejected; a reservation's release time and reason come
    together; remaining_qty is never below 0."""
    from decimal import Decimal

    from sqlalchemy import insert
    from sqlalchemy.exc import IntegrityError

    from sleeve_fund.store import gate_decisions_t, gate_reservations_t, utcnow

    schema.migrate(engine)
    store = Store(engine=engine)
    sid = store.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USD",
                              bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=1_000).id

    def refused(table, row):
        with pytest.raises(IntegrityError), engine.begin() as c:
            c.execute(insert(table), [row])

    refused(gate_decisions_t, _gate_row(sid, approved_qty=Decimal("2")))  # more than asked
    refused(gate_decisions_t, _gate_row(sid, approved_qty=Decimal("-1")))
    refused(gate_decisions_t, _gate_row(sid, outcome="rejected", limit_hit="gross"))  # rejected with a quantity
    with engine.begin() as c:
        did = c.execute(insert(gate_decisions_t).returning(gate_decisions_t.c.id),
                        [_gate_row(sid, outcome="trimmed", approved_qty=Decimal("0.5"), limit_hit="gross")]
                        ).scalar_one()
    now = utcnow()
    res = {"decision_id": did, "sleeve_id": sid, "underlying": "BTC", "remaining_qty": Decimal("0.5"),
           "notional": Decimal("50"), "margin": Decimal("25"), "open_risk": Decimal("1"), "created_at": now,
           "expires_at": now}
    refused(gate_reservations_t, {**res, "released_at": now})  # released with no reason
    refused(gate_reservations_t, {**res, "release_reason": "fill"})  # a reason but still active
    refused(gate_reservations_t, {**res, "remaining_qty": Decimal("-0.1")})
    with engine.begin() as c:
        c.execute(insert(gate_reservations_t), [res])


def test_0010_seed_equals_the_stores_copy_field_by_field(engine):
    """The migration's frozen seed and store.PORTFOLIO_PROFILE_V1 (create_all's) are deliberate copies that must
    never drift (DA)."""
    from datetime import datetime, timezone

    from sqlalchemy import select

    from sleeve_fund.store import PORTFOLIO_PROFILE_V1, portfolio_profile_t

    schema.migrate(engine)
    with engine.connect() as c:
        row = dict(c.execute(select(portfolio_profile_t).where(portfolio_profile_t.c.version == 1)).one()._mapping)
    at = row["created_at"]
    row["created_at"] = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    assert set(row) == set(PORTFOLIO_PROFILE_V1)
    for k, want in PORTFOLIO_PROFILE_V1.items():
        assert row[k] == want, (k, row[k], want)
    assert isinstance(row["created_at"], datetime)


def test_0010_on_a_database_a_newer_store_already_opened_keeps_its_tables_and_one_seed(engine):
    """A Store opened on a database at 0009 builds the new tables itself (create_all, seeding v1); 0010 then leaves
    them as they are, adds no second seed, and the result matches the code (DA-9's c1 cell opens one that way)."""
    from alembic import command
    from sqlalchemy import func, select

    from sleeve_fund.store import portfolio_profile_t

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0009")
    Store(engine=engine)
    assert schema.migrate(engine, log=lambda _: None) == f"at migration {schema.head()}"
    with engine.connect() as c:
        assert c.execute(select(func.count()).select_from(portfolio_profile_t)).scalar() == 1
    assert schema.report(engine)[1] == []


def test_qa_f211_1_a_limit_finer_than_four_places_is_refused_never_rounded(engine):
    """Before: open_risk 0.00125 was stored as 0.0013, a looser limit than asked. Now it is refused, nothing written."""
    from sleeve_fund.risk import PortfolioProfile

    schema.migrate(engine)
    store = Store(engine=engine)
    with pytest.raises(ValueError, match="at most 4 decimal places: open_risk is 0.00125"):
        store.add_portfolio_profile(PortfolioProfile(open_risk=0.00125), created_by="pm")
    assert store.portfolio_profile().version == 1
    assert store.portfolio_profile(store.add_portfolio_profile(PortfolioProfile(open_risk=0.0125),
                                                               created_by="pm")).open_risk == 0.0125


def _gate_shape(eng):
    i = inspect(eng)
    out = {}
    for t in ("portfolio_profile", "portfolio_state", "book_marks", "gate_decisions", "gate_reservations"):
        out[t] = (sorted((c["name"], c["nullable"]) for c in i.get_columns(t)),
                  sorted(c["name"] for c in i.get_check_constraints(t)),
                  sorted((x["name"], tuple(x["column_names"]), bool(x["unique"])) for x in i.get_indexes(t)),
                  sorted(u["name"] for u in i.get_unique_constraints(t)),
                  sorted((tuple(f["constrained_columns"]), f["referred_table"]) for f in i.get_foreign_keys(t)))
    return out


def test_0011_migrated_gate_tables_equal_create_alls_check_for_check(engine, tmp_path):
    """The migrations (0010 then 0011's batch rebuilds) give the gate's tables exactly what store.py's create_all
    does: columns and nullability, every named CHECK, the partial unique index in place of the old UNIQUE, and FKs."""
    schema.migrate(engine)
    migrated = _gate_shape(engine)
    fresh = make_engine(f"sqlite:///{tmp_path}/fresh.db") if engine.dialect.name == "sqlite" else engine
    if fresh is engine:
        _clean(engine)
    Store(engine=fresh)
    built = _gate_shape(fresh)
    assert migrated == built
    assert "portfolio_state_pause_reason" in built["portfolio_state"][1]
    assert ("gate_decisions_sleeve_bar_intent_stage", ("sleeve_id", "bar_ts", "intent_id", "stage"), True) in \
        built["gate_decisions"][2] and "gate_decisions_sleeve_bar_intent_stage" not in built["gate_decisions"][3]


def test_0011_an_error_row_never_blocks_the_retry_but_a_second_real_decision_is_refused(engine):
    from datetime import datetime, timezone

    from sqlalchemy import insert
    from sqlalchemy.exc import IntegrityError

    from sleeve_fund.store import gate_decisions_t

    schema.migrate(engine)
    store = Store(engine=engine)
    sid = store.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USD",
                              bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=1_000).id
    bar = datetime(2026, 10, 7, tzinfo=timezone.utc)
    with engine.begin() as c:  # two transient failures on the bar, then the retry decides
        c.execute(insert(gate_decisions_t), [_gate_row(sid, seq=1, bar_ts=bar, outcome="error", limit_hit="lock_error",
                                                       approved_qty=0),
                                             _gate_row(sid, seq=2, bar_ts=bar, outcome="error", approved_qty=0),
                                             _gate_row(sid, seq=3, bar_ts=bar)])
    with pytest.raises(IntegrityError), engine.begin() as c:  # a replay of the real one fails closed
        c.execute(insert(gate_decisions_t), [_gate_row(sid, seq=4, bar_ts=bar, outcome="rejected", approved_qty=0)])


def test_0012_trials_before_it_are_single_and_a_kind_outside_the_three_is_refused(engine):
    # DA 8 Oct (QD M4): kind records what was evaluated; every row before it was one strategy's variant.
    from alembic import command
    from sqlalchemy.exc import IntegrityError

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0011")
        conn.execute(text("INSERT INTO trials (id, definition_hash, idea_hash, code_version, definition_name, family, "
                          "settings, dataset, stage, source, created_at) VALUES ('old', 'd', 'i', 'c', 'n', 'f', "
                          "'{}', 'ds', 'in_sample', 'study', CURRENT_TIMESTAMP)"))
    schema.migrate(engine, log=lambda _: None)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT kind FROM trials WHERE id = 'old'")).scalar() == "single"
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text("UPDATE trials SET kind = 'basket' WHERE id = 'old'"))


def test_0012_on_a_database_a_newer_store_already_opened_keeps_its_column(engine):
    from alembic import command

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0011")
    with engine.begin() as conn:  # a Store on this code adds the column before the migration runs
        conn.execute(text("ALTER TABLE trials ADD COLUMN kind VARCHAR(16) DEFAULT 'single' NOT NULL"))
    schema.migrate(engine, log=lambda _: None)
    assert "kind" in {c["name"] for c in inspect(engine).get_columns("trials")}


def test_0012_a_halt_before_the_first_mark_is_kept_and_a_pause_still_needs_a_mark(engine):
    # CR F219-3: 0011's CHECK let an unmarked book be 'ok' only, so a halt before the first mark was refused (fail open).
    from alembic import command
    from sqlalchemy.exc import IntegrityError

    halt = ("INSERT INTO portfolio_state (id, status, halt_reason, profile_version, updated_at) "
            "VALUES (1, 'halted', 'drawdown', 1, CURRENT_TIMESTAMP)")
    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0011")
    with pytest.raises(IntegrityError), engine.begin() as conn:  # what 0012 changes
        conn.execute(text(halt))
    schema.migrate(engine, log=lambda _: None)
    with engine.begin() as conn:
        conn.execute(text(halt))
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text("UPDATE portfolio_state SET status = 'paused', halt_reason = NULL, "
                          "paused_until = CURRENT_TIMESTAMP, pause_reason = 'daily loss' WHERE id = 1"))
    sql = {c["name"]: c["sqltext"] for c in inspect(engine).get_check_constraints("portfolio_state")}
    assert "'paused'" in sql["portfolio_state_marked"]
