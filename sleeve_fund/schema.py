"""Does the database's schema match the code's? A read-only check of a live database against store.py.

    python -m sleeve_fund.schema check   # exit 0: matches; exit 1: drift, each difference listed

The app creates missing tables when it starts but never alters one that exists, so a column changed in
store.py can pass every test and never reach the server. This check says so out loud. It only reads: on
Postgres it runs inside a READ ONLY transaction, so even a bug here cannot write.
"""

from __future__ import annotations

import sys

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Float, inspect, text
from sqlalchemy.engine import Connection, Engine

from sleeve_fund.store import make_engine, metadata

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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["check"]:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
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
