"""P2-2 DB ledger: portfolio_state holds what the gate keeps in it. pause_reason (the pause's why, beside halt_reason),
stale_told_at (a stale book is alerted once per spell, across processes) and a row before the first mark: the book's
figures are null until then, all set or all null (portfolio_state_marked). Nothing has written the table yet (the
ledger lands with this), so it is rebuilt; a row in it stops the migration rather than be lost.

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-08 00:45:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0011'
down_revision = '0010'
branch_labels = None
depends_on = None

EXACT = sa.Numeric(38, 18)
TS = sa.DateTime(timezone=True)
FIGURES = ('reference_equity', 'hwm', 'day_start_equity', 'day_start', 'book_equity', 'marked_at')
MARKED = ("(" + " AND ".join(f"{c} IS NULL" for c in FIGURES) + " AND status = 'ok') OR ("
          + " AND ".join(f"{c} IS NOT NULL" for c in FIGURES) + ")")


def upgrade() -> None:
    bind = op.get_bind()
    have = sa.inspect(bind)
    if 'stale_told_at' in {c['name'] for c in have.get_columns('portfolio_state')}:
        return  # built by a newer Store's create_all first (as 0010 allows): already this shape
    if bind.execute(sa.text('SELECT count(*) FROM portfolio_state')).scalar():
        raise RuntimeError('portfolio_state has a row: 0011 rebuilds it, so it stops here rather than lose it')
    op.drop_table('portfolio_state')
    op.create_table(
        'portfolio_state',
        sa.Column('id', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('paused_until', TS, nullable=True),
        sa.Column('halt_reason', sa.Text(), nullable=True),
        sa.Column('pause_reason', sa.Text(), nullable=True),
        sa.Column('reference_equity', EXACT, nullable=True),
        sa.Column('hwm', EXACT, nullable=True),
        sa.Column('day_start_equity', EXACT, nullable=True),
        sa.Column('day_start', sa.Date(), nullable=True),
        sa.Column('book_equity', EXACT, nullable=True),
        sa.Column('marked_at', TS, nullable=True),
        sa.Column('stale_told_at', TS, nullable=True),
        sa.Column('profile_version', sa.Integer(), nullable=False),
        sa.Column('updated_at', TS, nullable=False),
        sa.CheckConstraint('id = 1', name='portfolio_state_one_row'),
        sa.CheckConstraint("status IN ('ok', 'paused', 'halted')", name='portfolio_state_status'),
        sa.CheckConstraint("status <> 'paused' OR paused_until IS NOT NULL", name='portfolio_state_paused_until'),
        sa.CheckConstraint("status <> 'halted' OR halt_reason IS NOT NULL", name='portfolio_state_halt_reason'),
        sa.CheckConstraint(MARKED, name='portfolio_state_marked'),
        sa.ForeignKeyConstraint(['profile_version'], ['portfolio_profile.version']),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    raise NotImplementedError('the portfolio state is kept: never dropped')
