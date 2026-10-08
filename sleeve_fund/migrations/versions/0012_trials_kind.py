"""Add trials.kind, and let portfolio_state be halted before its first mark.

trials.kind: what a row evaluated, one strategy's variant ('single'), an ablation of one, or a multi-strategy
portfolio run (DA 8 Oct 2026, for QD's M4; the ablation kind is PE1's 0015 plan). source keeps meaning where the row
came from, so a portfolio run is source 'backtest', kind 'portfolio_run'. Additive only: every existing row was one
strategy's variant and is 'single'. Whether ablations and portfolio runs count in N is the Advisor's call.

portfolio_state_marked becomes (marked_at IS NOT NULL OR status <> 'paused') (CR F219-3): a halt, the PM's included, is
never refused for coming before the supervisor's first mark, since refusing it would fail open; a daily pause always
follows a mark, so it still needs one. The per-figure CHECKs stay: the figures are null until that mark.

Each step is skipped when a Store on the same code built that shape first (create_all), as 0010 allows.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-08 02:40:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0012'
down_revision = '0011'
branch_labels = None
depends_on = None


MARKED = "marked_at IS NOT NULL OR status <> 'paused'"


def upgrade() -> None:
    have = sa.inspect(op.get_bind())
    if 'kind' not in {c['name'] for c in have.get_columns('trials')}:
        with op.batch_alter_table('trials', schema=None) as batch_op:
            batch_op.add_column(sa.Column('kind', sa.String(length=16), server_default='single', nullable=False))
            batch_op.create_check_constraint('trials_kind', "kind IN ('single', 'ablation', 'portfolio_run')")
    marked = {c['name']: c['sqltext'] for c in have.get_check_constraints('portfolio_state')}.get('portfolio_state_marked')
    if marked is not None and "'paused'" not in marked:  # still 0011's "OR status = 'ok'"
        with op.batch_alter_table('portfolio_state', schema=None) as batch_op:
            batch_op.drop_constraint('portfolio_state_marked', type_='check')
            batch_op.create_check_constraint('portfolio_state_marked', MARKED)


def downgrade() -> None:
    raise NotImplementedError('the trials register is append-only: never dropped')
