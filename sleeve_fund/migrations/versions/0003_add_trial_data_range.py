"""Add trials.data_start and data_end: the bars each evaluation read, for the holdout overlap check (v2 P1-7,
C4). Additive: two nullable columns; NULL is unknown and counts as overlapping.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-05 19:10:34.756510
"""

from alembic import op
import sqlalchemy as sa


revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('trials', schema=None) as batch_op:
        batch_op.add_column(sa.Column('data_start', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('data_end', sa.DateTime(timezone=True), nullable=True))



def downgrade() -> None:
    raise NotImplementedError('the trial dates guard the holdout locks: never dropped')
