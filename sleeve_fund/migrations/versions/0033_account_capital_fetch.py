"""B-23 (PE5, 20 Oct; Advisor B23-ACCOUNT-CAPITAL): account_capital_fetch, one row per read of a venue account's
marked equity (wallet plus unrealised, in USDT), good or failed, append-only, shaped like venue_terms_fetch. anchor = 1
marks the paper reading a gap is measured from.

Additive only: one new table, its CHECKs, its index and the triggers that refuse an edit or a delete. Skipped when a
Store on the same code built it first.

Revision ID: 0033 (HoE 20 Oct: DA's B-24 takes 0031 and 0032 in D2)
Revises: 0032 (re-chained at the D3 fold, HoE 20 Oct)
Create Date: 2026-10-20 20:15:00.000000
"""

from alembic import op
import sqlalchemy as sa

TS = sa.DateTime(timezone=True)
# Written out, not imported: a migration is frozen at what it was when it ran
WORM = "account_capital_fetch is append-only: a read is never edited or deleted, a new read is a new row"
PG = (
    "CREATE OR REPLACE FUNCTION account_capital_fetch_append_only() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    f"RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = '{WORM}'; END $$",
    "CREATE TRIGGER account_capital_fetch_append_only BEFORE UPDATE OR DELETE ON account_capital_fetch FOR EACH ROW "
    "EXECUTE FUNCTION account_capital_fetch_append_only()",
)
SQLITE = tuple(f"CREATE TRIGGER account_capital_fetch_no_{o.lower()} BEFORE {o} ON account_capital_fetch "
               f"BEGIN SELECT RAISE(ABORT, '{WORM}'); END" for o in ('UPDATE', 'DELETE'))

revision = '0033'
down_revision = '0032'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if 'account_capital_fetch' in sa.inspect(bind).get_table_names():
        return
    op.create_table(
        'account_capital_fetch',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('account', sa.String(length=41), nullable=False),
        sa.Column('fetched_at', TS, nullable=False),
        sa.Column('ok', sa.Integer(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.Column('anchor', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint('ok IN (0, 1)', name='account_capital_fetch_ok'),
        sa.CheckConstraint('anchor IN (0, 1)', name='account_capital_fetch_anchor'),
        sa.CheckConstraint("(ok = 1 AND reason = '') OR (ok = 0 AND reason <> '')", name='account_capital_fetch_reason'),
        sa.CheckConstraint('ok = 0 OR payload IS NOT NULL', name='account_capital_fetch_payload'),
        sa.CheckConstraint('anchor = 0 OR ok = 1', name='account_capital_fetch_anchor_good'),
    )
    op.create_index('ix_account_capital_latest', 'account_capital_fetch', ['account', 'ok', 'fetched_at', 'id'])
    for sql in PG if bind.dialect.name == 'postgresql' else SQLITE:
        op.execute(sql)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.execute(sa.text('SELECT count(*) FROM account_capital_fetch')).scalar():
        raise NotImplementedError('account_capital_fetch holds reads: they are never dropped')
    op.drop_table('account_capital_fetch')  # its triggers go with it
    if bind.dialect.name == 'postgresql':
        op.execute('DROP FUNCTION IF EXISTS account_capital_fetch_append_only()')
