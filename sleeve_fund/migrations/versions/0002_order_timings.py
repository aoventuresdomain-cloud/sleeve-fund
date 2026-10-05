"""Add order_timings: each paper or live order's bar close, arrival, decision, send, acceptance and fills, to
the microsecond (v2 P1-2). A new table, so nothing existing changes.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-05
"""

from alembic import op
import sqlalchemy as sa


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('order_timings',
    sa.Column('order_id', sa.String(length=64), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('bar_close', sa.DateTime(timezone=True), nullable=True),
    sa.Column('bar_recv', sa.DateTime(timezone=True), nullable=True),
    sa.Column('decided', sa.DateTime(timezone=True), nullable=True),
    sa.Column('sent', sa.DateTime(timezone=True), nullable=True),
    sa.Column('accepted', sa.DateTime(timezone=True), nullable=True),
    sa.Column('first_fill', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_fill', sa.DateTime(timezone=True), nullable=True),
    sa.Column('venue_ts', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['order_id'], ['orders.order_id'], ),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('order_id')
    )
    op.create_index('order_timings_sleeve_decided', 'order_timings', ['sleeve', 'decided'], unique=False)


def downgrade() -> None:
    op.drop_index('order_timings_sleeve_decided', table_name='order_timings')
    op.drop_table('order_timings')
