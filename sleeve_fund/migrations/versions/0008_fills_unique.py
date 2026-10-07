"""DA-2: a fill is booked once. A unique index on fills (sleeve, order_id, trade_id). Keyed with the order because the
paper venue's trade ids are deterministic per process (Nautilus sandbox) and can repeat after a restart on a new
order; the client order id is unique (orders.order_id). Additive only: no row is changed or removed.

If the journal already holds the same key twice, the migration stops and lists them (never deletes: PM rule); the
HoE decides what to do with them before it is run again.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-07 21:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    dupes = op.get_bind().execute(sa.text(
        "SELECT sleeve, order_id, trade_id, COUNT(*) AS n FROM fills GROUP BY sleeve, order_id, trade_id "
        "HAVING COUNT(*) > 1 ORDER BY sleeve, order_id, trade_id LIMIT 50")).fetchall()
    if dupes:
        listed = "; ".join(f"{r.sleeve} {r.order_id}/{r.trade_id} x{r.n}" for r in dupes)
        raise RuntimeError(f"fills booked more than once ({len(dupes)} keys shown, none changed): {listed}. "
                           "Ask the Data Architect and the HoE before running this migration again")
    with op.batch_alter_table('fills', schema=None) as batch_op:
        batch_op.create_index('fills_sleeve_order_trade', ['sleeve', 'order_id', 'trade_id'], unique=True)


def downgrade() -> None:
    raise NotImplementedError('fills_sleeve_order_trade keeps each fill booked once: never dropped')
