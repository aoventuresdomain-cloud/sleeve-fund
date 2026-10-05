"""Add trials: the register of every research variant run (v2 P1-6), counted by definition hash for the
deflated Sharpe. Additive only: a new table and its two indexes.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-05 18:50:40.494073
"""

from alembic import op
import sqlalchemy as sa


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('trials',
    sa.Column('id', sa.String(length=16), nullable=False),
    sa.Column('definition_hash', sa.String(length=64), nullable=False),
    sa.Column('idea_hash', sa.String(length=64), nullable=False),
    sa.Column('code_version', sa.String(length=40), nullable=False),
    sa.Column('definition_name', sa.Text(), nullable=False),
    sa.Column('family', sa.String(length=32), nullable=False),
    sa.Column('settings', sa.Text(), nullable=False),
    sa.Column('dataset', sa.String(length=128), nullable=False),
    sa.Column('stage', sa.String(length=16), nullable=False),
    sa.Column('source', sa.String(length=16), nullable=False),
    sa.Column('sharpe', sa.Float(), nullable=True),
    sa.Column('trades', sa.Integer(), nullable=True),
    sa.Column('oos_trades', sa.Integer(), nullable=True),
    sa.Column('backtest_id', sa.String(length=16), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('trials', schema=None) as batch_op:
        batch_op.create_index('trials_definition_dataset', ['definition_hash', 'dataset'], unique=False)
        batch_op.create_index('trials_idea_hash', ['idea_hash'], unique=False)



def downgrade() -> None:
    with op.batch_alter_table('trials', schema=None) as batch_op:
        batch_op.drop_index('trials_idea_hash')
        batch_op.drop_index('trials_definition_dataset')

    op.drop_table('trials')
