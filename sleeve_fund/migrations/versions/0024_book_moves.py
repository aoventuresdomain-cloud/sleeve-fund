"""F-ASSIGN-STALE: a create or an account move asked for on a marked book, waiting for the supervisor's poll.

book_moves (DA, 14 Oct 17:45 UK; Advisor ADV-ASSIGN option 1): one row per request; at most one waiting per Strategy
(ux_book_moves_waiting). Only the poll, after its own mark, lands it (landed_at), with its flow and the Strategy's
sleeve_accounts row in the same transaction; a later request or a put-away supersedes it (superseded_at). Rows end
once and are never deleted.

Additive only; nothing existing is edited. Skipped when a Store on the same code built the table first (create_all).

Revision ID: 0024
Revises: 0022
Create Date: 2026-10-14 22:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

TS = sa.DateTime(timezone=True)
WAITING = 'landed_at IS NULL AND superseded_at IS NULL'

revision = '0024'
down_revision = '0022'
branch_labels = None
depends_on = None


def upgrade() -> None:
    if 'book_moves' in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        'book_moves',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sleeve', sa.String(length=64), nullable=False),
        sa.Column('account', sa.String(length=41), nullable=False),
        sa.Column('kind', sa.String(length=8), nullable=False),
        sa.Column('requested_at', TS, nullable=False),
        sa.Column('landed_at', TS, nullable=True),
        sa.Column('superseded_at', TS, nullable=True),
        sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name']),
        sa.ForeignKeyConstraint(['account'], ['accounts.name']),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("kind IN ('create', 'move')", name='book_moves_kind'),
        sa.CheckConstraint('landed_at IS NULL OR superseded_at IS NULL', name='book_moves_one_end'),
        sa.CheckConstraint('(landed_at IS NULL OR landed_at >= requested_at) '
                           'AND (superseded_at IS NULL OR superseded_at >= requested_at)', name='book_moves_order'),
    )
    op.create_index('ux_book_moves_waiting', 'book_moves', ['sleeve'], unique=True,
                    postgresql_where=sa.text(WAITING), sqlite_where=sa.text(WAITING))


def downgrade() -> None:
    raise NotImplementedError("dropping it would lose the creates and moves waiting for the poll: never")
