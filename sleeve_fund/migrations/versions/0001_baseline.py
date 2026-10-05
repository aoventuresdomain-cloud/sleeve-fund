"""Baseline: the 27 tables as store.py created them on 5 Oct 2026. An existing production database is
STAMPED at this revision (sleeve_fund.schema.migrate), never built from it; an empty one is built from it.

Revision ID: 0001
Revises: 
Create Date: 2026-10-05
"""

from alembic import op
import sqlalchemy as sa


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('accounts',
    sa.Column('name', sa.String(length=41), nullable=False),
    sa.Column('kind', sa.String(length=8), nullable=False),
    sa.Column('venue', sa.String(length=16), nullable=False),
    sa.Column('note', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('name')
    )
    op.create_table('backtests',
    sa.Column('id', sa.String(length=16), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('key', sa.String(length=64), nullable=False),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('query', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('result', sa.Text(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('backtests', schema=None) as batch_op:
        batch_op.create_index('backtests_key', ['key', 'created_at'], unique=False)

    op.create_table('decisions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('actor', sa.String(length=32), nullable=False),
    sa.Column('action', sa.String(length=32), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=True),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('events',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=True),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('level', sa.String(length=8), nullable=False),
    sa.Column('kind', sa.String(length=32), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('events', schema=None) as batch_op:
        batch_op.create_index('events_ts', ['ts'], unique=False)

    op.create_table('history_requests',
    sa.Column('venue', sa.String(length=16), nullable=False),
    sa.Column('instrument', sa.String(length=32), nullable=False),
    sa.Column('since', sa.DateTime(timezone=True), nullable=False),
    sa.Column('requested_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('venue', 'instrument')
    )
    op.create_table('sleeves',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('strategy', sa.String(length=64), nullable=False),
    sa.Column('instrument', sa.String(length=32), nullable=False),
    sa.Column('bar_spec', sa.String(length=64), nullable=False),
    sa.Column('params', sa.JSON(), nullable=False),
    sa.Column('starting_balance', sa.Float(), nullable=False),
    sa.Column('risk_profile', sa.String(length=32), nullable=False),
    sa.Column('warmup_bars', sa.Integer(), nullable=False),
    sa.Column('desired_state', sa.String(length=16), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('status_reason', sa.Text(), nullable=False),
    sa.Column('paused_until', sa.DateTime(timezone=True), nullable=True),
    sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('spreads',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('venue', sa.String(length=16), nullable=False),
    sa.Column('instrument', sa.String(length=32), nullable=False),
    sa.Column('half_spread', sa.Float(), nullable=False),
    sa.Column('samples', sa.Integer(), nullable=False),
    sa.Column('measured_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('strategy_reset_holds',
    sa.Column('reset_id', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('status_reason', sa.Text(), nullable=False),
    sa.Column('paused_until', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('reset_id')
    )
    op.create_table('strategy_resets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('actor', sa.String(length=64), nullable=False),
    sa.Column('restart', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('done_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('run', sa.String(length=64), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('account_keys',
    sa.Column('account', sa.String(length=41), nullable=False),
    sa.Column('present', sa.Integer(), nullable=False),
    sa.Column('checked_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['account'], ['accounts.name'], ),
    sa.PrimaryKeyConstraint('account')
    )
    op.create_table('account_retired',
    sa.Column('account', sa.String(length=41), nullable=False),
    sa.Column('retired_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['account'], ['accounts.name'], ),
    sa.PrimaryKeyConstraint('account')
    )
    op.create_table('alert_acks',
    sa.Column('event_id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('actor', sa.String(length=32), nullable=False),
    sa.Column('note', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['event_id'], ['events.id'], ),
    sa.PrimaryKeyConstraint('event_id')
    )
    op.create_table('commands',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('command', sa.String(length=16), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('applied_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('demo_mirror',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('fill_id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('instrument', sa.String(length=32), server_default='', nullable=False),
    sa.Column('amount', sa.Float(), server_default='0', nullable=False),
    sa.Column('price', sa.Float(), nullable=True),
    sa.Column('order_id', sa.String(length=64), server_default='', nullable=False),
    sa.Column('message', sa.Text(), server_default='', nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('demo_mirror', schema=None) as batch_op:
        batch_op.create_index('demo_mirror_sleeve_fill', ['sleeve', 'fill_id'], unique=False)

    op.create_table('equity',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('equity', sa.Float(), nullable=False),
    sa.Column('cash', sa.Float(), nullable=False),
    sa.Column('qty', sa.Float(), nullable=False),
    sa.Column('price', sa.Float(), nullable=False),
    sa.Column('benchmark', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('equity', schema=None) as batch_op:
        batch_op.create_index('equity_sleeve_ts', ['sleeve', 'ts'], unique=False)

    op.create_table('exit_plans',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('entry_order', sa.String(length=64), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('stop_frac', sa.Float(), nullable=True),
    sa.Column('tp_frac', sa.Float(), nullable=True),
    sa.Column('basis', sa.Text(), nullable=False),
    sa.Column('stop_cfg', sa.JSON(), nullable=True),
    sa.Column('risk_amount', sa.Float(), nullable=True),
    sa.Column('planned_r', sa.Float(), nullable=True),
    sa.Column('event_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('exit_plans', schema=None) as batch_op:
        batch_op.create_index('exit_plans_entry', ['sleeve', 'entry_order', 'id'], unique=False)

    op.create_table('fee_schedules',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('venue', sa.String(length=16), nullable=False),
    sa.Column('account', sa.String(length=41), nullable=False),
    sa.Column('maker', sa.Float(), nullable=False),
    sa.Column('taker', sa.Float(), nullable=False),
    sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['account'], ['accounts.name'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('feed_seen',
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('seen_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('sleeve')
    )
    op.create_table('fills',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('side', sa.String(length=8), nullable=False),
    sa.Column('qty', sa.Float(), nullable=False),
    sa.Column('price', sa.Float(), nullable=False),
    sa.Column('fee', sa.Float(), nullable=False),
    sa.Column('order_id', sa.String(length=64), nullable=False),
    sa.Column('trade_id', sa.String(length=64), nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('fills', schema=None) as batch_op:
        batch_op.create_index('fills_sleeve_ts', ['sleeve', 'ts'], unique=False)

    op.create_table('funding',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('qty', sa.Float(), nullable=False),
    sa.Column('price', sa.Float(), nullable=False),
    sa.Column('rate', sa.Float(), nullable=False),
    sa.Column('amount', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('funding', schema=None) as batch_op:
        batch_op.create_index('funding_sleeve_ts', ['sleeve', 'ts'], unique=False)

    op.create_table('insurance',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('price', sa.Float(), nullable=False),
    sa.Column('amount', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('insurance', schema=None) as batch_op:
        batch_op.create_index('insurance_sleeve_ts', ['sleeve', 'ts'], unique=False)

    op.create_table('mirror_requests',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=True),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('actor', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('done_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('result', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('orders',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('order_id', sa.String(length=64), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('side', sa.String(length=8), nullable=False),
    sa.Column('order_type', sa.String(length=16), nullable=False),
    sa.Column('qty', sa.Float(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('filled_qty', sa.Float(), nullable=False),
    sa.Column('avg_px', sa.Float(), nullable=True),
    sa.Column('fee', sa.Float(), nullable=False),
    sa.Column('intent', sa.String(length=16), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('signal', sa.JSON(), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('order_id')
    )
    with op.batch_alter_table('orders', schema=None) as batch_op:
        batch_op.create_index('orders_sleeve_ts', ['sleeve', 'ts'], unique=False)

    op.create_table('signal_state',
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('payload', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('sleeve')
    )
    op.create_table('sleeve_accounts',
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('account', sa.String(length=41), nullable=False),
    sa.Column('assigned_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['account'], ['accounts.name'], ),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('sleeve')
    )
    op.create_table('sleeve_archive',
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('archived_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('sleeve')
    )
    op.create_table('sleeve_venues',
    sa.Column('sleeve', sa.String(length=64), nullable=False),
    sa.Column('venue', sa.String(length=16), nullable=False),
    sa.ForeignKeyConstraint(['sleeve'], ['sleeves.name'], ),
    sa.PrimaryKeyConstraint('sleeve')
    )


def downgrade() -> None:
    raise NotImplementedError("the baseline is never downgraded: it would drop every table")
