"""Commands for a reset after liquidation (P1-RAL): command widened to 32 characters ("reset_after_liquidation" is
23), and commands.incident, the liquidation event (events.id) the reset answers, unique where set so one incident is
answered once. Additive only: existing commands keep incident NULL.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-07 00:45:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('commands', schema=None) as batch_op:
        batch_op.alter_column('command', existing_type=sa.String(length=16), type_=sa.String(length=32),
                              existing_nullable=False)
        batch_op.add_column(sa.Column('incident', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('commands_incident_fkey', 'events', ['incident'], ['id'], ondelete='RESTRICT')
        batch_op.create_index('commands_incident', ['incident'], unique=True,
                              postgresql_where=sa.text('incident IS NOT NULL'),
                              sqlite_where=sa.text('incident IS NOT NULL'))


def downgrade() -> None:
    raise NotImplementedError('commands.incident records which liquidation each reset answered: never dropped')
