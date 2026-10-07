"""Add funding.kind: whether a funding row is the venue's settled rate, the baseline charged for a missing one, or
a true-up once the missing rate arrives (QA P1-O17, Advisor 6 Oct 2026). Additive only: existing rows were all
charged at a rate the venue settled or the fixed fallback, and are marked "settled".

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-06 20:30:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('funding', schema=None) as batch_op:
        batch_op.add_column(sa.Column('kind', sa.String(length=16), server_default='settled', nullable=False))
        batch_op.create_check_constraint('funding_kind', "kind IN ('settled', 'baseline', 'true_up', 'reversal')")


def downgrade() -> None:
    with op.batch_alter_table('funding', schema=None) as batch_op:
        batch_op.drop_constraint('funding_kind', type_='check')
        batch_op.drop_column('kind')
