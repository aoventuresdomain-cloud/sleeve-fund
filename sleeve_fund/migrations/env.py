"""alembic's entry point. The connection comes from sleeve_fund.schema (config.attributes["connection"])."""

from alembic import context

from sleeve_fund.store import metadata

conn = context.config.attributes["connection"]
context.configure(connection=conn, target_metadata=metadata, compare_type=True,
                  render_as_batch=conn.dialect.name == "sqlite")  # SQLite can only alter a table by copying it
with context.begin_transaction():
    context.run_migrations()
