"""CASH-1: each cash movement is booked once, refused by the database the second time.

funding: unique (sleeve, ts, kind) (a settlement reversed or trued up at most once) and a partial unique (sleeve, ts)
WHERE kind IN ('settled', 'baseline') (one charge per settlement, never both kinds). Until now only the engine's
memory kept a settlement from being charged twice.

insurance: order_id and trade_id, the fill that left the strategy flat below zero, with a partial unique index on
(sleeve, order_id, trade_id) WHERE order_id IS NOT NULL: one credit per closing fill. Rows before this stay unkeyed.

Additive only: no row is changed or removed. If the journal already holds a key twice, the migration stops and lists
them (never deletes: PM rule), as 0008 does; the HoE decides what to do with them before it is run again. Each step is
skipped when a Store on the same code built that shape first (create_all).

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-08 04:30:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0013'
down_revision = '0012'
branch_labels = None
depends_on = None


CHARGES = "kind IN ('settled', 'baseline')"
DUPES = {
    "funding (sleeve, ts, kind)": "SELECT sleeve, ts, kind AS what, COUNT(*) AS n FROM funding "
                                  "GROUP BY sleeve, ts, kind HAVING COUNT(*) > 1",
    "funding charges (sleeve, ts)": f"SELECT sleeve, ts, 'charge' AS what, COUNT(*) AS n FROM funding WHERE {CHARGES} "
                                    "GROUP BY sleeve, ts HAVING COUNT(*) > 1",
}


def upgrade() -> None:
    bind = op.get_bind()
    found = []
    for name, sql in DUPES.items():
        rows = bind.execute(sa.text(sql + " ORDER BY sleeve, ts LIMIT 50")).fetchall()
        found += [f"{name}: {r.sleeve} {r.ts} {r.what} x{r.n}" for r in rows]
    if found:
        raise RuntimeError(f"cash booked more than once ({len(found)} keys shown, none changed): {'; '.join(found)}. "
                           "Ask the Data Architect and the HoE before running this migration again")
    have = sa.inspect(bind)
    if 'funding_once' not in {i['name'] for i in have.get_indexes('funding')}:
        with op.batch_alter_table('funding', schema=None) as batch_op:
            batch_op.create_index('funding_once', ['sleeve', 'ts', 'kind'], unique=True)
            batch_op.create_index('funding_charged_once', ['sleeve', 'ts'], unique=True,
                                  sqlite_where=sa.text(CHARGES), postgresql_where=sa.text(CHARGES))
    if 'order_id' not in {c['name'] for c in have.get_columns('insurance')}:
        with op.batch_alter_table('insurance', schema=None) as batch_op:
            batch_op.add_column(sa.Column('order_id', sa.String(length=64), nullable=True))
            batch_op.add_column(sa.Column('trade_id', sa.String(length=64), nullable=True))
    if 'insurance_once' not in {i['name'] for i in sa.inspect(bind).get_indexes('insurance')}:
        with op.batch_alter_table('insurance', schema=None) as batch_op:
            batch_op.create_index('insurance_once', ['sleeve', 'order_id', 'trade_id'], unique=True,
                                  sqlite_where=sa.text('order_id IS NOT NULL'),
                                  postgresql_where=sa.text('order_id IS NOT NULL'))


def downgrade() -> None:
    raise NotImplementedError('the once-only keys keep each cash movement booked once: never dropped')
