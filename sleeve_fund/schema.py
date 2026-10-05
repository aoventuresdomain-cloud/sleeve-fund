"""Does the database's schema match the code's? A read-only check of a live database against store.py.

    python -m sleeve_fund.schema check     # exit 0: matches; exit 1: drift, each difference listed
    python -m sleeve_fund.schema migrate   # bring the database to the newest migration (run by the deploy)
    python -m sleeve_fund.schema new "add x to y"   # draft a migration from store.py (developers, on SQLite)

The app creates missing tables when it starts but never alters one that exists, so a column changed in
store.py can pass every test and never reach the server. This check says so out loud. It only reads: on
Postgres it runs inside a READ ONLY transaction, so even a bug here cannot write.
"""

from __future__ import annotations

import sys

from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Float, insert, inspect, text
from sqlalchemy.engine import Connection, Engine

from sleeve_fund.store import events_t, make_engine, metadata, utcnow

# alembic's own bookkeeping table, once migrations are in (DA-1): not part of the app's schema.
IGNORED_TABLES = {"alembic_version"}


def _same_type(context, inspected_column, metadata_column, inspected_type, metadata_type):
    # Float reflects back from Postgres as DOUBLE PRECISION: the same column, not drift.
    if isinstance(metadata_type, Float) and isinstance(inspected_type, Float):
        return False
    return None  # alembic's default comparison


def _include(obj, name, type_, reflected, compare_to):
    return not (type_ == "table" and name in IGNORED_TABLES)


def differences(conn: Connection) -> list[str]:
    """Each way the connected database differs from store.py's tables, as one line, empty when they match."""
    ctx = MigrationContext.configure(conn, opts={"compare_type": _same_type, "include_object": _include})
    return [_describe(d) for d in compare_metadata(ctx, metadata)]


def _describe(diff) -> str:
    if isinstance(diff, list):  # column modifications come as a list of (kind, schema, table, column, ...)
        return "; ".join(_describe(d) for d in diff)
    kind = diff[0]
    if kind in ("add_table", "remove_table"):
        where = "missing from the database" if kind == "add_table" else "in the database but not in the code"
        return f"table {diff[1].name}: {where}"
    if kind in ("add_column", "remove_column"):
        where = "missing from the database" if kind == "add_column" else "in the database but not in the code"
        return f"column {diff[2]}.{diff[3].name}: {where}"
    if kind in ("add_index", "remove_index", "add_constraint", "remove_constraint"):
        obj = diff[1]
        where = "missing from the database" if kind.startswith("add") else "in the database but not in the code"
        return f"{kind.split('_')[1]} {obj.name} on {obj.table.name}: {where}"
    if kind.startswith("modify_"):
        _, _, table, column, _, db_value, code_value = diff
        return f"column {table}.{column}: {kind[len('modify_'):]} is {db_value} in the database, {code_value} in the code"
    return repr(diff)


def report(engine: Engine) -> tuple[list[str], list[str]]:
    """(facts about the database, differences from the code)."""
    with engine.connect() as conn:
        if conn.dialect.name == "postgresql":
            conn.execute(text("SET TRANSACTION READ ONLY"))
        tables = set(inspect(conn).get_table_names())
        version = None
        if "alembic_version" in tables:
            version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        diffs = differences(conn)
        conn.rollback()
    facts = [f"engine: {engine.dialect.name}",
             f"tables: {len(tables - IGNORED_TABLES)} in the database, {len(metadata.tables)} in the code",
             f"migration version: {version or 'none (no migrations applied yet)'}"]
    return facts, diffs


def _config(conn: Connection) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    cfg.attributes["connection"] = conn
    return cfg


def head() -> str:
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(_config(None)).get_current_head()


class Drift(RuntimeError):
    """An existing database without migration history that does not match the code: nothing was changed."""


def migrate(engine: Engine, log=print) -> str:
    """Bring the database to the newest migration and return what was done.

    - Migration history present: apply the newer migrations, in one transaction where the engine allows.
    - Empty database: build it by running every migration.
    - Tables but no history (production before migrations existed): it is never rebuilt. If its schema
      matches the code exactly it is STAMPED at the newest migration (history recorded, no table touched);
      if it differs, Drift is raised listing each difference, and nothing is changed."""
    with engine.connect() as conn:
        tables = set(inspect(conn).get_table_names())
        app_tables = tables & set(metadata.tables)
        diffs = differences(conn) if "alembic_version" not in tables and app_tables else []
        conn.rollback()  # end the reads' transaction; the writes below each run in their own
        if "alembic_version" not in tables and app_tables:
            if diffs:
                raise Drift("the database has tables but no migration history, and differs from the code:\n"
                            + "\n".join(f"  - {d}" for d in diffs))
            with conn.begin():
                command.stamp(_config(conn), "head")
            done = f"stamped existing schema at {head()} (no table changed)"
        else:
            with conn.begin():
                command.upgrade(_config(conn), "head")
            done = f"at migration {head()}"
    log(f"schema: {done}")
    return done


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["migrate"]:
        engine = make_engine()
        try:
            migrate(engine)
        except Drift as exc:
            # Only before the first stamp: the database runs as it did before migrations existed, so the app
            # may start as it always has. Nothing is migrated until the drift is fixed, and the alerts inbox
            # (and the status workflow's events) say so on every deploy. A failing migration still exits 1.
            msg = f"migrations held, nothing changed: {exc}"
        else:
            # Every deploy re-checks the migrated schema against the code: a hand edit on the server or a
            # migration that doesn't match store.py shows up here, without stopping the deploy.
            _, diffs = report(engine)
            msg = ("the migrated database differs from the code:\n" + "\n".join(f"  - {d}" for d in diffs)
                   if diffs else None)
        if msg:
            print(f"SCHEMA DRIFT: {msg}", file=sys.stderr)
            with engine.begin() as conn:
                conn.execute(insert(events_t).values(sleeve=None, ts=utcnow(), level="error", kind="schema_drift",
                                                     message=msg))
        return 0
    if len(argv) == 2 and argv[0] == "new":
        with make_engine().connect() as conn, conn.begin():
            command.revision(_config(conn), message=argv[1], autogenerate=True)
        return 0
    if argv != ["check"]:
        print("usage: python -m sleeve_fund.schema check | migrate | new MESSAGE", file=sys.stderr)
        return 2
    facts, diffs = report(make_engine())
    for line in facts:
        print(line)
    if not diffs:
        print("schema: matches the code")
        return 0
    print(f"SCHEMA DRIFT: {len(diffs)} difference(s) between the database and the code")
    for line in diffs:
        print(f"  - {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
