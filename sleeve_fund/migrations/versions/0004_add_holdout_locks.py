"""Add holdout_locks: each idea's held-back period, opened once per underlying (v2 P1-7, C4). Additive only.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-05 19:10:35.782256
"""

from alembic import op
import sqlalchemy as sa


revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('holdout_locks',
    sa.Column('id', sa.String(length=16), nullable=False),
    sa.Column('idea_hash', sa.String(length=64), nullable=False),
    sa.Column('underlying', sa.String(length=16), nullable=False),
    sa.Column('period_start', sa.DateTime(timezone=True), nullable=True),
    sa.Column('period_end', sa.DateTime(timezone=True), nullable=True),
    sa.Column('opened_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('trial_id', sa.String(length=16), nullable=True),
    sa.Column('source', sa.String(length=16), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('idea_hash', 'underlying', name='holdout_locks_idea_underlying')
    )


def downgrade() -> None:
    raise NotImplementedError('holdout_locks is an append-only record of every holdout opened: never dropped')
