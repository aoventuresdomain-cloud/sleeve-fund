"""FEE-LIVE (DA, head 3, 15 Oct; data-architect/fee-live-storage.md): venue_terms_fetch, one row per fetch of a
venue's fee rates (from the account) or margin tiers (public), good or failed, append-only. The engine reads the
newest good row per (venue, kind, instrument) as last-known-good.

Additive only: one new table, its CHECKs, its index and the triggers that refuse an edit or a delete. Skipped when a
Store on the same code built it first.

Revision ID: 0029
Revises: 0028 (re-chained onto 0015 when A2 folds into head 3: data-architect/migration-register-head2.md)
Create Date: 2026-10-15 12:30:00.000000
"""

from alembic import op
import sqlalchemy as sa

TS = sa.DateTime(timezone=True)
# Written out, not imported: a migration is frozen at what it was when it ran
WORM = "venue_terms_fetch is append-only: a fetch is never edited or deleted, a new fetch is a new row"
PG = (
    "CREATE OR REPLACE FUNCTION venue_terms_fetch_append_only() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    f"RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = '{WORM}'; END $$",
    "CREATE TRIGGER venue_terms_fetch_append_only BEFORE UPDATE OR DELETE ON venue_terms_fetch FOR EACH ROW "
    "EXECUTE FUNCTION venue_terms_fetch_append_only()",
)
SQLITE = tuple(f"CREATE TRIGGER venue_terms_fetch_no_{o.lower()} BEFORE {o} ON venue_terms_fetch BEGIN "
               f"SELECT RAISE(ABORT, '{WORM}'); END" for o in ('UPDATE', 'DELETE'))

revision = '0029'
down_revision = '0028'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if 'venue_terms_fetch' in sa.inspect(bind).get_table_names():
        return
    op.create_table(
        'venue_terms_fetch',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('venue', sa.String(length=16), nullable=False),
        sa.Column('kind', sa.String(length=8), nullable=False),
        sa.Column('instrument', sa.String(length=32), nullable=False),
        sa.Column('account', sa.String(length=41), nullable=True),
        sa.Column('source', sa.String(length=64), nullable=False),
        sa.Column('fetched_at', TS, nullable=False),
        sa.Column('ok', sa.Integer(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['account'], ['accounts.name']),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint('venue = upper(venue)', name='venue_terms_fetch_venue_upper'),
        sa.CheckConstraint("kind IN ('fee', 'tiers')", name='venue_terms_fetch_kind'),
        sa.CheckConstraint('ok IN (0, 1)', name='venue_terms_fetch_ok'),
        sa.CheckConstraint("(ok = 1 AND reason = '') OR (ok = 0 AND reason <> '')", name='venue_terms_fetch_reason'),
        sa.CheckConstraint('ok = 0 OR payload IS NOT NULL', name='venue_terms_fetch_payload'),
        sa.CheckConstraint("kind <> 'tiers' OR account IS NULL", name='venue_terms_fetch_public_tiers'),
    )
    op.create_index('ix_venue_terms_latest', 'venue_terms_fetch',
                    ['venue', 'kind', 'instrument', 'ok', 'fetched_at', 'id'])
    for sql in PG if bind.dialect.name == 'postgresql' else SQLITE:
        op.execute(sql)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.execute(sa.text('SELECT count(*) FROM venue_terms_fetch')).scalar():
        raise NotImplementedError('venue_terms_fetch holds fetches: they are never dropped')
    op.drop_table('venue_terms_fetch')  # its triggers go with it
    if bind.dialect.name == 'postgresql':
        op.execute('DROP FUNCTION IF EXISTS venue_terms_fetch_append_only()')
