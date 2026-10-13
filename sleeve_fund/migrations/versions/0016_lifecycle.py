"""P2-6: a strategy's lifecycle, Active -> Winding down -> Retired, set only by the PM (sleeve_fund.lifecycle).

sleeves.lifecycle: active, winding_down or retired, kept apart from running/stopped/paused (DA 6 Oct). Every existing
strategy is active. wind_down_started and wind_down_deadline: when a wind-down began and when the strategy must be
flat; both set while winding down (CHECK), and kept on a retired row as its record. The row holds the current state
only: each move is journaled in events and decisions (Store.set_lifecycle, the one writer).

Additive only. Each step is skipped when a Store on the same code built that shape first (create_all), as 0012 does.
The chain is renumbered onto the register's head when the reserved 0014 and 0015 merge (DA register rule).

Revision ID: 0016
Revises: 0013
Create Date: 2026-10-13 16:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0016'
down_revision = '0013'
branch_labels = None
depends_on = None


STATES = "lifecycle IN ('active', 'winding_down', 'retired')"
WINDING = ("lifecycle <> 'winding_down' OR (wind_down_started IS NOT NULL AND wind_down_deadline IS NOT NULL "
           "AND wind_down_deadline >= wind_down_started)")


def upgrade() -> None:
    have = sa.inspect(op.get_bind())
    if 'lifecycle' not in {c['name'] for c in have.get_columns('sleeves')}:
        with op.batch_alter_table('sleeves', schema=None) as batch_op:
            batch_op.add_column(sa.Column('lifecycle', sa.String(length=16), server_default='active', nullable=False))
            batch_op.add_column(sa.Column('wind_down_started', sa.DateTime(timezone=True), nullable=True))
            batch_op.add_column(sa.Column('wind_down_deadline', sa.DateTime(timezone=True), nullable=True))
            batch_op.create_check_constraint('sleeves_lifecycle', STATES)
            batch_op.create_check_constraint('sleeves_wind_down', WINDING)


def downgrade() -> None:
    raise NotImplementedError('a strategy\'s lifecycle is its record: never dropped')
