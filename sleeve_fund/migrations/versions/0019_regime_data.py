"""P2-5 regime, the data side (DA design data-architect/p2-5-data-design.md): instruments, universe_snapshots and
regime_series, with daily_bars beside them, each write-once in the database itself.

- instruments: every contract a venue lists or listed, delisted included, unique (venue, symbol). Only delisted_at
  (set once, from NULL) and seen_at may change; no row is deleted.
- daily_bars: each closed day's close and quote notional per contract (api or archive), unique (venue, symbol, day),
  FK to instruments: the only input the universe and the ER read.
- universe_snapshots: the members on each day under one config, unique per (venue, date, config) by symbol and by
  underlying (one contract per underlying), FK (venue, symbol) to instruments, data_through before date.
- regime_series: one row per (date, reference_venue, lookback, config): the median ER, its state and the cuts used;
  data_through before date; a known state carries its ER and cuts.
universe_snapshots and regime_series rows are never edited or deleted.

Additive only: four new tables, no row read or changed. Each table and its triggers are skipped when a Store on the
same code built them first (create_all).

Revision ID: 0019
Revises: 0014 (0015 ORPHAN-GUARD, 0016 CASH-4, 0017 P2-6 and 0018 TRIALS-WORM are reserved and not merged: the PR
that merges later re-chains, per the register)
Create Date: 2026-10-13 18:40:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0019'
down_revision = '0014'
branch_labels = None
depends_on = None


EXACT = sa.Numeric(38, 18)
TS = sa.DateTime(timezone=True)
# Written out, not imported: a migration is frozen at what it was when it ran
TRIGGERS = {
    'instruments': {
        'postgresql': (
            "CREATE OR REPLACE FUNCTION instruments_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF TG_OP = 'DELETE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'instruments are write-once: row ' || OLD.id || ' cannot be deleted'; END IF; IF (NEW.id, NEW.venue, NEW.symbol, NEW.underlying, NEW.quote, NEW.contract_type, NEW.listed_at, NEW.source) IS DISTINCT FROM (OLD.id, OLD.venue, OLD.symbol, OLD.underlying, OLD.quote, OLD.contract_type, OLD.listed_at, OLD.source) THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'instruments are write-once: row ' || OLD.id || ' cannot be edited'; END IF; IF OLD.delisted_at IS NOT NULL AND NEW.delisted_at IS DISTINCT FROM OLD.delisted_at THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'instruments are write-once: delisted_at of row ' || OLD.id || ' is set once'; END IF; RETURN NEW; END $$",
            'CREATE TRIGGER instruments_write_once BEFORE UPDATE OR DELETE ON instruments FOR EACH ROW EXECUTE FUNCTION instruments_write_once()',
        ),
        'sqlite': (
            "CREATE TRIGGER instruments_no_delete BEFORE DELETE ON instruments BEGIN SELECT RAISE(ABORT, 'instruments are write-once: a row cannot be deleted'); END",
            "CREATE TRIGGER instruments_no_edit BEFORE UPDATE ON instruments WHEN NEW.id IS NOT OLD.id OR NEW.venue IS NOT OLD.venue OR NEW.symbol IS NOT OLD.symbol OR NEW.underlying IS NOT OLD.underlying OR NEW.quote IS NOT OLD.quote OR NEW.contract_type IS NOT OLD.contract_type OR NEW.listed_at IS NOT OLD.listed_at OR NEW.source IS NOT OLD.source OR (OLD.delisted_at IS NOT NULL AND NEW.delisted_at IS NOT OLD.delisted_at) BEGIN SELECT RAISE(ABORT, 'instruments are write-once: a row cannot be edited'); END",
        ),
    },
    'daily_bars': {
        'postgresql': (
            "CREATE OR REPLACE FUNCTION daily_bars_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF TG_OP = 'DELETE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'daily bars are write-once: row ' || OLD.id || ' cannot be deleted'; END IF; IF (NEW.id, NEW.venue, NEW.symbol, NEW.day, NEW.close, NEW.notional, NEW.source, NEW.fetched_at) IS DISTINCT FROM (OLD.id, OLD.venue, OLD.symbol, OLD.day, OLD.close, OLD.notional, OLD.source, OLD.fetched_at) THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'daily bars are write-once: row ' || OLD.id || ' cannot be edited'; END IF; RETURN NEW; END $$",
            'CREATE TRIGGER daily_bars_write_once BEFORE UPDATE OR DELETE ON daily_bars FOR EACH ROW EXECUTE FUNCTION daily_bars_write_once()',
        ),
        'sqlite': (
            "CREATE TRIGGER daily_bars_no_delete BEFORE DELETE ON daily_bars BEGIN SELECT RAISE(ABORT, 'daily bars are write-once: a row cannot be deleted'); END",
            "CREATE TRIGGER daily_bars_no_edit BEFORE UPDATE ON daily_bars WHEN NEW.id IS NOT OLD.id OR NEW.venue IS NOT OLD.venue OR NEW.symbol IS NOT OLD.symbol OR NEW.day IS NOT OLD.day OR NEW.close IS NOT OLD.close OR NEW.notional IS NOT OLD.notional OR NEW.source IS NOT OLD.source OR NEW.fetched_at IS NOT OLD.fetched_at BEGIN SELECT RAISE(ABORT, 'daily bars are write-once: a row cannot be edited'); END",
        ),
    },
    'universe_snapshots': {
        'postgresql': (
            "CREATE OR REPLACE FUNCTION universe_snapshots_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF TG_OP = 'DELETE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'universe snapshots are write-once: row ' || OLD.id || ' cannot be deleted'; END IF; IF (NEW.id, NEW.date, NEW.venue, NEW.config, NEW.symbol, NEW.underlying, NEW.rank, NEW.notional, NEW.data_through, NEW.computed_at) IS DISTINCT FROM (OLD.id, OLD.date, OLD.venue, OLD.config, OLD.symbol, OLD.underlying, OLD.rank, OLD.notional, OLD.data_through, OLD.computed_at) THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'universe snapshots are write-once: row ' || OLD.id || ' cannot be edited'; END IF; RETURN NEW; END $$",
            'CREATE TRIGGER universe_snapshots_write_once BEFORE UPDATE OR DELETE ON universe_snapshots FOR EACH ROW EXECUTE FUNCTION universe_snapshots_write_once()',
        ),
        'sqlite': (
            "CREATE TRIGGER universe_snapshots_no_delete BEFORE DELETE ON universe_snapshots BEGIN SELECT RAISE(ABORT, 'universe snapshots are write-once: a row cannot be deleted'); END",
            "CREATE TRIGGER universe_snapshots_no_edit BEFORE UPDATE ON universe_snapshots WHEN NEW.id IS NOT OLD.id OR NEW.date IS NOT OLD.date OR NEW.venue IS NOT OLD.venue OR NEW.config IS NOT OLD.config OR NEW.symbol IS NOT OLD.symbol OR NEW.underlying IS NOT OLD.underlying OR NEW.rank IS NOT OLD.rank OR NEW.notional IS NOT OLD.notional OR NEW.data_through IS NOT OLD.data_through OR NEW.computed_at IS NOT OLD.computed_at BEGIN SELECT RAISE(ABORT, 'universe snapshots are write-once: a row cannot be edited'); END",
        ),
    },
    'regime_series': {
        'postgresql': (
            "CREATE OR REPLACE FUNCTION regime_series_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF TG_OP = 'DELETE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'the regime series is write-once: row ' || OLD.id || ' cannot be deleted'; END IF; IF (NEW.id, NEW.date, NEW.reference_venue, NEW.lookback, NEW.config, NEW.universe_date, NEW.data_through, NEW.er, NEW.state, NEW.lo, NEW.hi, NEW.members, NEW.computed_at) IS DISTINCT FROM (OLD.id, OLD.date, OLD.reference_venue, OLD.lookback, OLD.config, OLD.universe_date, OLD.data_through, OLD.er, OLD.state, OLD.lo, OLD.hi, OLD.members, OLD.computed_at) THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'the regime series is write-once: row ' || OLD.id || ' cannot be edited'; END IF; RETURN NEW; END $$",
            'CREATE TRIGGER regime_series_write_once BEFORE UPDATE OR DELETE ON regime_series FOR EACH ROW EXECUTE FUNCTION regime_series_write_once()',
        ),
        'sqlite': (
            "CREATE TRIGGER regime_series_no_delete BEFORE DELETE ON regime_series BEGIN SELECT RAISE(ABORT, 'the regime series is write-once: a row cannot be deleted'); END",
            "CREATE TRIGGER regime_series_no_edit BEFORE UPDATE ON regime_series WHEN NEW.id IS NOT OLD.id OR NEW.date IS NOT OLD.date OR NEW.reference_venue IS NOT OLD.reference_venue OR NEW.lookback IS NOT OLD.lookback OR NEW.config IS NOT OLD.config OR NEW.universe_date IS NOT OLD.universe_date OR NEW.data_through IS NOT OLD.data_through OR NEW.er IS NOT OLD.er OR NEW.state IS NOT OLD.state OR NEW.lo IS NOT OLD.lo OR NEW.hi IS NOT OLD.hi OR NEW.members IS NOT OLD.members OR NEW.computed_at IS NOT OLD.computed_at BEGIN SELECT RAISE(ABORT, 'the regime series is write-once: a row cannot be edited'); END",
        ),
    },
}


def _tables():
    return {
        'instruments': [
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('venue', sa.String(length=32), nullable=False),
            sa.Column('symbol', sa.String(length=32), nullable=False),
            sa.Column('underlying', sa.String(length=16), nullable=False),
            sa.Column('quote', sa.String(length=16), nullable=False),
            sa.Column('contract_type', sa.String(length=8), nullable=False),
            sa.Column('listed_at', sa.Date(), nullable=False),
            sa.Column('delisted_at', sa.Date(), nullable=True),
            sa.Column('source', sa.String(length=8), nullable=False),
            sa.Column('seen_at', TS, nullable=False),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('venue', 'symbol', name='instruments_once'),
            sa.CheckConstraint("contract_type IN ('perp', 'spot')", name='instruments_contract_type'),
            sa.CheckConstraint("source IN ('api', 'archive')", name='instruments_source'),
            sa.CheckConstraint('delisted_at IS NULL OR delisted_at >= listed_at',
                               name='instruments_delisted_after_listed'),
        ],
        'daily_bars': [
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('venue', sa.String(length=32), nullable=False),
            sa.Column('symbol', sa.String(length=32), nullable=False),
            sa.Column('day', sa.Date(), nullable=False),
            sa.Column('close', EXACT, nullable=False),
            sa.Column('notional', EXACT, nullable=False),
            sa.Column('source', sa.String(length=8), nullable=False),
            sa.Column('fetched_at', TS, nullable=False),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('venue', 'symbol', 'day', name='daily_bars_once'),
            sa.ForeignKeyConstraint(['venue', 'symbol'], ['instruments.venue', 'instruments.symbol'],
                                    name='daily_bars_instrument'),
            sa.CheckConstraint('close > 0', name='daily_bars_close'),
            sa.CheckConstraint('notional >= 0', name='daily_bars_notional'),
            sa.CheckConstraint("source IN ('api', 'archive')", name='daily_bars_source'),
        ],
        'universe_snapshots': [
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('date', sa.Date(), nullable=False),
            sa.Column('venue', sa.String(length=32), nullable=False),
            sa.Column('config', sa.String(length=64), nullable=False),
            sa.Column('symbol', sa.String(length=32), nullable=False),
            sa.Column('underlying', sa.String(length=16), nullable=False),
            sa.Column('rank', sa.Integer(), nullable=False),
            sa.Column('notional', EXACT, nullable=False),
            sa.Column('data_through', sa.Date(), nullable=False),
            sa.Column('computed_at', TS, nullable=False),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('venue', 'date', 'config', 'symbol', name='universe_snapshots_once'),
            sa.UniqueConstraint('venue', 'date', 'config', 'underlying', name='universe_snapshots_underlying_once'),
            sa.ForeignKeyConstraint(['venue', 'symbol'], ['instruments.venue', 'instruments.symbol'],
                                    name='universe_snapshots_instrument'),
            sa.CheckConstraint('data_through < date', name='universe_snapshots_point_in_time'),
            sa.CheckConstraint('rank >= 1', name='universe_snapshots_rank'),
        ],
        'regime_series': [
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('date', sa.Date(), nullable=False),
            sa.Column('reference_venue', sa.String(length=32), nullable=False),
            sa.Column('lookback', sa.Integer(), nullable=False),
            sa.Column('config', sa.String(length=64), nullable=False),
            sa.Column('universe_date', sa.Date(), nullable=False),
            sa.Column('data_through', sa.Date(), nullable=False),
            sa.Column('er', EXACT, nullable=True),
            sa.Column('state', sa.String(length=8), nullable=False),
            sa.Column('lo', EXACT, nullable=True),
            sa.Column('hi', EXACT, nullable=True),
            sa.Column('members', sa.Integer(), nullable=False),
            sa.Column('computed_at', TS, nullable=False),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('date', 'reference_venue', 'lookback', 'config', name='regime_series_once'),
            sa.CheckConstraint('lookback IN (20, 30)', name='regime_series_lookback'),
            sa.CheckConstraint("state IN ('low', 'mid', 'high', 'unknown')", name='regime_series_state'),
            sa.CheckConstraint('data_through < date', name='regime_series_point_in_time'),
            sa.CheckConstraint("state = 'unknown' OR (er IS NOT NULL AND lo IS NOT NULL AND hi IS NOT NULL "
                               "AND lo <= hi)", name='regime_series_known_has_cuts'),
            sa.CheckConstraint('members >= 0', name='regime_series_members'),
        ],
    }


def upgrade() -> None:
    bind = op.get_bind()
    have = set(sa.inspect(bind).get_table_names())
    for name, columns in _tables().items():  # in order: universe_snapshots refers to instruments
        if name in have:
            continue  # a Store on this code built it, triggers included (create_all)
        op.create_table(name, *columns)
        for sql in TRIGGERS[name]['postgresql' if bind.dialect.name == 'postgresql' else 'sqlite']:
            op.execute(sql)


def downgrade() -> None:
    raise NotImplementedError('the regime data is write-once: never dropped')
