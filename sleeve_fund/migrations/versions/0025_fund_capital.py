"""Bridge A (DA, 14 Oct; PM go 21:38 UK): the fund holds the capital. A Strategy has none of its own.

fund_flows: the fund's only external capital, append-only. A deposit or withdrawal is journaled with its decision and
ended once at the poll, applied or refused with the reason, in id order (an inflow never overtakes a pending outflow,
Advisor CAP5-RESET). rebase marks the clean slate's deposit only (carry=False, onto a book of exactly 0). Write-once at
the database: Postgres, one plpgsql function and a BEFORE UPDATE OR DELETE trigger; SQLite, two BEFORE triggers.

instruments: what can be traded, one row per venue, symbol and contract, with the underlying the gate nets by. Seeded
from the Strategies already there (backtests skipped): venue from sleeve_venues or the default, contract from
params.market. The asset type is not here: it is versioned with the compliance tests (0026).

Additive only: nothing existing is edited. Each part is skipped when a Store on the same code built it first.

Revision ID: 0025
Revises: 0024 (PE2's book_moves, on main's 0022)
Create Date: 2026-10-14 22:00:00.000000
"""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from alembic import op
import sqlalchemy as sa

EXACT = sa.Numeric(38, 18)  # as sleeve_fund.exact, frozen here
BIG_ID = sa.BigInteger().with_variant(sa.Integer(), 'sqlite')
TS = sa.DateTime(timezone=True)


def _default_venue_then() -> str:
    """The default venue as 0020 froze it (a Strategy with no pin trades there): read from that migration, so the seed
    agrees with the orders backfill and no module outside the venue ones names a venue."""
    spec = importlib.util.spec_from_file_location('m0020', Path(__file__).with_name('0020_orders_venue.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.DEFAULT_VENUE_THEN


ALIASES = {'XBT': 'BTC', 'XDG': 'DOGE'}  # sleeve_fund.portfolio.holding.ALIASES, frozen
BACKTEST_PREFIX = 'bt:'

# Written out, not imported: a migration is frozen at what it was when it ran
FROZEN = ('id', 'kind', 'amount', 'rebase', 'actor', 'reason', 'requested_at')
PG = (
    "CREATE OR REPLACE FUNCTION fund_flows_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "IF TG_OP = 'DELETE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    "MESSAGE = 'fund_flows is append-only: ' || OLD.id || ' cannot be deleted'; END IF; "
    f"IF ({', '.join('NEW.' + c for c in FROZEN)}) IS DISTINCT FROM ({', '.join('OLD.' + c for c in FROZEN)}) "
    "OR OLD.applied_at IS NOT NULL OR OLD.refused_at IS NOT NULL THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', "
    "MESSAGE = 'fund_flows is append-only: ' || OLD.id || ' may only be ended once'; END IF; "
    "RETURN NEW; END $$",
    "CREATE TRIGGER fund_flows_write_once BEFORE UPDATE OR DELETE ON fund_flows FOR EACH ROW "
    "EXECUTE FUNCTION fund_flows_write_once()",
)
SQLITE = (
    "CREATE TRIGGER fund_flows_no_delete BEFORE DELETE ON fund_flows BEGIN "
    "SELECT RAISE(ABORT, 'fund_flows is append-only: a row cannot be deleted'); END",
    "CREATE TRIGGER fund_flows_no_edit BEFORE UPDATE ON fund_flows WHEN "
    + " OR ".join(f"NEW.{c} IS NOT OLD.{c}" for c in FROZEN)
    + " OR OLD.applied_at IS NOT NULL OR OLD.refused_at IS NOT NULL"
    + " BEGIN SELECT RAISE(ABORT, 'fund_flows is append-only: a row may only be ended once'); END",
)


revision = '0025'
down_revision = '0024'
branch_labels = None
depends_on = None


def _underlying(symbol: str) -> str:
    base = symbol.split('/')[0].strip().upper()
    return ALIASES.get(base, base)


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if 'fund_flows' not in tables:
        op.create_table(
            'fund_flows',
            sa.Column('id', BIG_ID, primary_key=True),
            sa.Column('kind', sa.String(8), nullable=False),
            sa.Column('amount', EXACT, nullable=False),
            sa.Column('rebase', sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column('actor', sa.String(32), nullable=False),
            sa.Column('reason', sa.Text(), nullable=False),
            sa.Column('requested_at', TS, nullable=False),
            sa.Column('applied_at', TS),
            sa.Column('refused_at', TS),
            sa.Column('refuse_reason', sa.Text()),
            sa.CheckConstraint("kind IN ('deposit', 'withdraw')", name='fund_flows_kind'),
            sa.CheckConstraint('amount > 0', name='fund_flows_amount'),
            sa.CheckConstraint("NOT rebase OR kind = 'deposit'", name='fund_flows_rebase'),
            sa.CheckConstraint("actor <> ''", name='fund_flows_actor'),
            sa.CheckConstraint('applied_at IS NULL OR refused_at IS NULL', name='fund_flows_one_end'),
            sa.CheckConstraint('(refused_at IS NULL) = (refuse_reason IS NULL)', name='fund_flows_refuse_reason'),
        )
        op.create_index('fund_flows_pending', 'fund_flows', ['id'],
                        sqlite_where=sa.text('applied_at IS NULL AND refused_at IS NULL'),
                        postgresql_where=sa.text('applied_at IS NULL AND refused_at IS NULL'))
        for sql in (PG if bind.dialect.name == 'postgresql' else SQLITE):
            op.execute(sql)
    if 'instruments' not in tables:
        op.create_table(
            'instruments',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('venue', sa.String(16), nullable=False),
            sa.Column('symbol', sa.String(32), nullable=False),
            sa.Column('contract', sa.String(8), nullable=False),
            sa.Column('underlying', sa.String(16), nullable=False),
            sa.Column('created_at', TS, nullable=False),
            sa.CheckConstraint('venue = upper(venue)', name='instruments_venue_upper'),
            sa.CheckConstraint("contract IN ('spot', 'perp')", name='instruments_contract'),
            sa.UniqueConstraint('venue', 'symbol', 'contract', name='instruments_venue_symbol_contract'),
        )
    # Seed from the Strategies there (idempotent: a row already there is kept)
    have = {(r.venue, r.symbol, r.contract) for r in bind.execute(sa.text(
        'SELECT venue, symbol, contract FROM instruments'))}
    pins = dict(bind.execute(sa.text('SELECT sleeve, venue FROM sleeve_venues')).all())
    default = _default_venue_then()
    now, rows = datetime.now(timezone.utc), []
    for name, symbol, params in bind.execute(sa.text('SELECT name, instrument, params FROM sleeves')).all():
        if name.startswith(BACKTEST_PREFIX):
            continue
        params = json.loads(params) if isinstance(params, str) else (params or {})
        contract = 'spot' if (params.get('market') or 'spot') == 'spot' else 'perp'
        key = ((pins.get(name) or default).upper(), symbol, contract)
        if key not in have:
            have.add(key)
            rows.append({'venue': key[0], 'symbol': symbol, 'contract': contract, 'underlying': _underlying(symbol),
                         'created_at': now})
    if rows:
        bind.execute(sa.text('INSERT INTO instruments (venue, symbol, contract, underlying, created_at) '
                             'VALUES (:venue, :symbol, :contract, :underlying, :created_at)'), rows)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.execute(sa.text('SELECT count(*) FROM fund_flows')).scalar():
        raise NotImplementedError("the fund's capital ledger has rows: it is never dropped with money in it")
    op.drop_table('instruments')
    if bind.dialect.name == 'postgresql':
        op.execute('DROP TRIGGER IF EXISTS fund_flows_write_once ON fund_flows')
        op.execute('DROP FUNCTION IF EXISTS fund_flows_write_once()')
    op.drop_index('fund_flows_pending', table_name='fund_flows')
    op.drop_table('fund_flows')
