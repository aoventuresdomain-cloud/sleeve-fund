"""Bridge A (DA, 14 Oct): orders, fills, funding and insurance say which instrument they were in.

A Strategy on several assets (Drop B) needs its fills, funding and insurance read per asset; today every row's
instrument is its Strategy's, so the backfill copies sleeves.instrument through the row's sleeve. A writer that names
no instrument gets its Strategy's at the database (Postgres: a BEFORE INSERT trigger; SQLite: '' by default, then an
AFTER INSERT trigger fills that row), so no writer changes now. No foreign key to instruments: a fill is never refused
for a missing reference row; reconcile checks it instead. A row whose Strategy row is gone keeps '' (reconcile names
it).

Additive only. Each column is skipped when a Store on the same code built it first.

Revision ID: 0026
Revises: 0025
Create Date: 2026-10-14 22:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

TABLES = ('orders', 'fills', 'funding', 'insurance')
# Written out, not imported: a migration is frozen at what it was when it ran
PG_FUNCTION = (
    "CREATE OR REPLACE FUNCTION fill_instrument() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "IF NEW.instrument IS NULL OR NEW.instrument = '' THEN "
    "NEW.instrument := COALESCE((SELECT instrument FROM sleeves WHERE name = NEW.sleeve), ''); END IF; "
    "RETURN NEW; END $$"
)


def _pg_trigger(table):
    return (f"CREATE TRIGGER {table}_fill_instrument BEFORE INSERT ON {table} FOR EACH ROW "
            "EXECUTE FUNCTION fill_instrument()")


def _sqlite_trigger(table):
    return (f"CREATE TRIGGER {table}_fill_instrument AFTER INSERT ON {table} WHEN NEW.instrument = '' BEGIN "
            f"UPDATE {table} SET instrument = COALESCE((SELECT instrument FROM sleeves WHERE name = NEW.sleeve), '') "
            f"WHERE id = NEW.id; END")


revision = '0026'
down_revision = '0025'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    pg = bind.dialect.name == 'postgresql'
    insp = sa.inspect(bind)
    if pg:
        op.execute(PG_FUNCTION)
    for table in TABLES:
        if 'instrument' in {c['name'] for c in insp.get_columns(table)}:
            continue
        op.add_column(table, sa.Column('instrument', sa.String(32), nullable=False, server_default=''))
        op.execute(f"UPDATE {table} SET instrument = COALESCE((SELECT s.instrument FROM sleeves s "
                   f"WHERE s.name = {table}.sleeve), '')")
        op.execute(_pg_trigger(table) if pg else _sqlite_trigger(table))
    if 'fills_sleeve_instrument_ts' not in {i['name'] for i in insp.get_indexes('fills')}:
        op.create_index('fills_sleeve_instrument_ts', 'fills', ['sleeve', 'instrument', 'ts'])


def downgrade() -> None:
    bind = op.get_bind()
    pg = bind.dialect.name == 'postgresql'
    op.drop_index('fills_sleeve_instrument_ts', table_name='fills')
    for table in TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS {table}_fill_instrument" + (f" ON {table}" if pg else ""))
        # A plain DROP COLUMN (SQLite 3.35+), not batch mode: batch rebuilds the table, and renaming it back trips
        # 0020's sleeve_venues triggers, which name orders and fills
        op.drop_column(table, 'instrument')
    if pg:
        op.execute('DROP FUNCTION IF EXISTS fill_instrument()')
