"""orders.venue (P2-4 router v1): the venue each order was sent to, the strategy's one venue. Stamped by
store.record_order from router.venue_for, never by the caller; upper case, held by CHECK orders_venue_upper.

Backfill: every order already journaled gets its strategy's venue exactly as paper/config resolved it when the order
was sent: its sleeve_venues row, else the default venue of the day. The default is spelled here as a literal (DA), so
this migration's result never moves if the code's default does. Then the column is made NOT NULL.

Each step is skipped when a Store on the same code built that shape first (create_all), as 0010 allows. SQLite adds the
column, its CHECK and NOT NULL through a batch rebuild. Downgrade is refused: an order's venue is never dropped.

Revision ID: 0020
Revises: 0014
Create Date: 2026-10-13 19:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0020'
down_revision = '0014'
branch_labels = None
depends_on = None

# The default venue when this migration was written (sleeve_fund.venues.DEFAULT_VENUE on 13 Oct 2026): the venue of
# every strategy saved before strategies carried their own.
DEFAULT_VENUE_THEN = 'KRAKEN'
UPPER = 'venue = upper(venue)'


def upgrade() -> None:
    bind = op.get_bind()
    have = sa.inspect(bind)
    if 'venue' not in {c['name'] for c in have.get_columns('orders')}:
        with op.batch_alter_table('orders', schema=None) as batch_op:
            batch_op.add_column(sa.Column('venue', sa.String(length=16), nullable=True))
    orders = sa.table('orders', sa.column('sleeve', sa.String), sa.column('venue', sa.String))
    pins = sa.table('sleeve_venues', sa.column('sleeve', sa.String), sa.column('venue', sa.String))
    pinned = (sa.select(sa.func.upper(pins.c.venue)).where(pins.c.sleeve == orders.c.sleeve).scalar_subquery())
    bind.execute(sa.update(orders).where(orders.c.venue.is_(None))
                 .values(venue=sa.func.coalesce(pinned, DEFAULT_VENUE_THEN)))
    have = sa.inspect(bind)
    nullable = next(c['nullable'] for c in have.get_columns('orders') if c['name'] == 'venue')
    checked = 'orders_venue_upper' in {c['name'] for c in have.get_check_constraints('orders')}
    if nullable or not checked:
        with op.batch_alter_table('orders', schema=None) as batch_op:
            if nullable:
                batch_op.alter_column('venue', existing_type=sa.String(length=16), nullable=False)
            if not checked:
                batch_op.create_check_constraint('orders_venue_upper', UPPER)


def downgrade() -> None:
    raise NotImplementedError('orders.venue records where each order was sent: never dropped')
