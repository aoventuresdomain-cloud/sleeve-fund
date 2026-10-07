"""P2-2: the portfolio limits gate's tables (Data Architect's shapes, v2/p2-2-tables.md). portfolio_profile (the PM's
limits, append-only, seeded with version 1 as accepted on 6 Oct), portfolio_state (the supervisor's one row),
book_marks (the book's history, a cache rebuilt from fills), gate_decisions (one row per check of an order that raises
a position, with its first-come seq) and gate_reservations (headroom held by resting entries, with remaining_qty for
partial fills and cancel_sent_at for the sweep's cancel). New tables only: no existing row is changed.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-07 23:30:00.000000
"""

from datetime import datetime, timezone
from decimal import Decimal

from alembic import op
import sqlalchemy as sa


revision = '0010'
down_revision = '0009'
branch_labels = None
depends_on = None

EXACT = sa.Numeric(38, 18)
LIMIT = sa.Numeric(10, 4)
TS = sa.DateTime(timezone=True)
BIG_ID = sa.BigInteger().with_variant(sa.Integer(), 'sqlite')
OUTCOMES = ('approved', 'trimmed', 'rejected', 'late', 'portfolio_state_stale', 'error')
LIMITS = ('gross', 'net_instrument', 'margin', 'open_risk', 'halt', 'pause', 'portfolio_state_stale', 'lock_error',
          'below_min')


def _in(column, values):
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def upgrade() -> None:
    if op.get_bind().dialect.name == 'postgresql':
        op.execute(sa.schema.CreateSequence(sa.Sequence('gate_decision_seq')))
    profile = op.create_table(
        'portfolio_profile',
        sa.Column('version', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('created_at', TS, nullable=False),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.Column('gross_max', LIMIT, nullable=False),
        sa.Column('net_underlying_max', LIMIT, nullable=False),
        sa.Column('margin_max', LIMIT, nullable=False),
        sa.Column('open_risk_max', LIMIT, nullable=False),
        sa.Column('drawdown_halt', LIMIT, nullable=False),
        sa.Column('daily_pause', LIMIT, nullable=False),
        sa.Column('note', sa.Text(), nullable=True),
        sa.CheckConstraint('gross_max > 0 AND net_underlying_max > 0 AND margin_max > 0 AND open_risk_max > 0 '
                           'AND daily_pause > 0 AND drawdown_halt > daily_pause', name='portfolio_profile_limits'),
        sa.PrimaryKeyConstraint('version'),
    )
    op.create_table(
        'portfolio_state',
        sa.Column('id', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('paused_until', TS, nullable=True),
        sa.Column('halt_reason', sa.Text(), nullable=True),
        sa.Column('reference_equity', EXACT, nullable=False),
        sa.Column('hwm', EXACT, nullable=False),
        sa.Column('day_start_equity', EXACT, nullable=False),
        sa.Column('day_start', sa.Date(), nullable=False),
        sa.Column('book_equity', EXACT, nullable=False),
        sa.Column('marked_at', TS, nullable=False),
        sa.Column('profile_version', sa.Integer(), nullable=False),
        sa.Column('updated_at', TS, nullable=False),
        sa.CheckConstraint('id = 1', name='portfolio_state_one_row'),
        sa.CheckConstraint("status IN ('ok', 'paused', 'halted')", name='portfolio_state_status'),
        sa.ForeignKeyConstraint(['profile_version'], ['portfolio_profile.version']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'book_marks',
        sa.Column('ts', TS, nullable=False),
        sa.Column('equity', EXACT, nullable=False),
        sa.Column('gross', EXACT, nullable=False),
        sa.Column('net_max', EXACT, nullable=False),
        sa.Column('net_underlying', sa.String(length=16), nullable=True),
        sa.Column('margin_used', EXACT, nullable=False),
        sa.Column('open_risk', EXACT, nullable=False),
        sa.Column('hwm', EXACT, nullable=False),
        sa.Column('reference_equity', EXACT, nullable=False),
        sa.Column('day_start_equity', EXACT, nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('profile_version', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['profile_version'], ['portfolio_profile.version']),
        sa.PrimaryKeyConstraint('ts'),
    )
    op.create_table(
        'gate_decisions',
        sa.Column('id', BIG_ID, nullable=False),
        sa.Column('seq', sa.BigInteger(), nullable=False),
        sa.Column('sleeve_id', sa.Integer(), nullable=False),
        sa.Column('bar_ts', TS, nullable=False),
        sa.Column('intent_id', sa.Text(), nullable=False),
        sa.Column('intent', sa.String(length=16), nullable=False),
        sa.Column('underlying', sa.String(length=16), nullable=False),
        sa.Column('order_id', sa.String(length=64), nullable=True),
        sa.Column('profile_version', sa.Integer(), nullable=False),
        sa.Column('outcome', sa.String(length=24), nullable=False),
        sa.Column('limit_hit', sa.String(length=24), nullable=True),
        sa.Column('requested_qty', EXACT, nullable=False),
        sa.Column('approved_qty', EXACT, nullable=False),
        sa.Column('price', EXACT, nullable=False),
        sa.Column('book_equity', EXACT, nullable=True),
        sa.Column('gross', EXACT, nullable=True),
        sa.Column('net_underlying', EXACT, nullable=True),
        sa.Column('margin_used', EXACT, nullable=True),
        sa.Column('open_risk', EXACT, nullable=True),
        sa.Column('regime_weight', sa.Numeric(10, 6), nullable=True),
        sa.Column('regime_state', sa.Text(), nullable=True),
        sa.Column('stage', sa.String(length=8), nullable=False),
        sa.Column('decided_at', TS, nullable=False),
        sa.CheckConstraint(_in('outcome', OUTCOMES), name='gate_decisions_outcome'),
        sa.CheckConstraint(_in('limit_hit', LIMITS), name='gate_decisions_limit_hit'),
        sa.CheckConstraint(_in('stage', ('submit', 'fill')), name='gate_decisions_stage'),
        sa.ForeignKeyConstraint(['order_id'], ['orders.order_id']),
        sa.ForeignKeyConstraint(['profile_version'], ['portfolio_profile.version']),
        sa.ForeignKeyConstraint(['sleeve_id'], ['sleeves.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('seq'),
        sa.UniqueConstraint('sleeve_id', 'bar_ts', 'intent_id', 'stage', name='gate_decisions_sleeve_bar_intent_stage'),
    )
    op.create_index('gate_decisions_sleeve_decided_at', 'gate_decisions', ['sleeve_id', 'decided_at'], unique=False)
    op.create_table(
        'gate_reservations',
        sa.Column('decision_id', BIG_ID, autoincrement=False, nullable=False),
        sa.Column('sleeve_id', sa.Integer(), nullable=False),
        sa.Column('underlying', sa.String(length=16), nullable=False),
        sa.Column('remaining_qty', EXACT, nullable=False),
        sa.Column('notional', EXACT, nullable=False),
        sa.Column('margin', EXACT, nullable=False),
        sa.Column('open_risk', EXACT, nullable=False),
        sa.Column('created_at', TS, nullable=False),
        sa.Column('cancel_sent_at', TS, nullable=True),
        sa.Column('expires_at', TS, nullable=False),
        sa.Column('released_at', TS, nullable=True),
        sa.Column('release_reason', sa.String(length=8), nullable=True),
        sa.CheckConstraint(_in('release_reason', ('fill', 'reject', 'cancel', 'ttl')),
                           name='gate_reservations_release_reason'),
        sa.ForeignKeyConstraint(['decision_id'], ['gate_decisions.id']),
        sa.ForeignKeyConstraint(['sleeve_id'], ['sleeves.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('decision_id'),
    )
    op.create_index('gate_reservations_active', 'gate_reservations', ['underlying'], unique=False,
                    sqlite_where=sa.text('released_at IS NULL'), postgresql_where=sa.text('released_at IS NULL'))
    op.create_index('gate_reservations_active_sleeve', 'gate_reservations', ['sleeve_id'], unique=False,
                    sqlite_where=sa.text('released_at IS NULL'), postgresql_where=sa.text('released_at IS NULL'))
    # The PM's accepted limits (6 Oct 14:17), as sleeve_fund.risk.PortfolioProfile v1 holds them.
    op.bulk_insert(profile, [{
        'version': 1, 'created_at': datetime(2026, 10, 6, 14, 17, tzinfo=timezone.utc), 'created_by': 'migration 0010',
        'gross_max': Decimal('1.5'), 'net_underlying_max': Decimal('0.5'), 'margin_max': Decimal('0.5'),
        'open_risk_max': Decimal('0.05'), 'drawdown_halt': Decimal('0.15'), 'daily_pause': Decimal('0.03'),
        'note': "the PM's accepted portfolio limits (6 Oct 2026)",
    }])


def downgrade() -> None:
    raise NotImplementedError('the gate journal and the PM limits history are kept: never dropped')
