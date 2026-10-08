"""P2-2 DB ledger (the DA's 0011): portfolio_state holds what the gate keeps in it, and an 'error' row never blocks a
retry.

- portfolio_state: pause_reason (with paused, as halt_reason with halted), stale_told_at (a stale book is alerted once
  per spell, across processes) and a row before the first mark: the book's figures are null until then, one CHECK
  per figure tying it to marked_at, and an unmarked book is 'ok' (the gate still treats it as stale).
- gate_decisions: the UNIQUE (sleeve_id, bar_ts, intent_id, stage) becomes a partial unique index WHERE outcome <>
  'error', same name: a transient lock or read failure is kept as audit and the retry on the same bar can decide.

Each step is skipped when a newer Store's create_all built that shape first (as 0010 allows). SQLite rebuilds the
tables (batch); nothing has written portfolio_state yet.

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
FIGURES = ('reference_equity', 'hwm', 'day_start_equity', 'day_start', 'book_equity')
UNIQUE = 'gate_decisions_sleeve_bar_intent_stage'
NOT_ERROR = "outcome <> 'error'"


def upgrade() -> None:
    bind = op.get_bind()
    have = sa.inspect(bind)
    cols = {c['name'] for c in have.get_columns('portfolio_state')}
    if 'stale_told_at' not in cols:
        with op.batch_alter_table('portfolio_state') as t:
            if 'pause_reason' not in cols:
                t.add_column(sa.Column('pause_reason', sa.Text(), nullable=True))
            t.add_column(sa.Column('stale_told_at', TS, nullable=True))
            for c in (*FIGURES, 'marked_at'):
                t.alter_column(c, existing_type=sa.Date() if c == 'day_start' else TS if c == 'marked_at' else EXACT,
                               nullable=True)
            t.create_check_constraint('portfolio_state_pause_reason', "status <> 'paused' OR pause_reason IS NOT NULL")
            t.create_check_constraint('portfolio_state_marked', "marked_at IS NOT NULL OR status = 'ok'")
            for c in FIGURES:
                t.create_check_constraint(f'portfolio_state_{c}_marked', f'({c} IS NULL) = (marked_at IS NULL)')
    if UNIQUE in {u['name'] for u in have.get_unique_constraints('gate_decisions')}:  # still the old UNIQUE
        with op.batch_alter_table('gate_decisions') as t:
            t.drop_constraint(UNIQUE, type_='unique')
        op.create_index(UNIQUE, 'gate_decisions', ['sleeve_id', 'bar_ts', 'intent_id', 'stage'], unique=True,
                        sqlite_where=sa.text(NOT_ERROR), postgresql_where=sa.text(NOT_ERROR))


def downgrade() -> None:
    raise NotImplementedError('the portfolio state and the gate journal are kept: never dropped')
