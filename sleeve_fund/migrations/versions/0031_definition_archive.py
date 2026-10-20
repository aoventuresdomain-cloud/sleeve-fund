"""B-16: definition_archive, the research Strategies put away from the Strategies list (PM 15 Oct 17:22 UK; mock-up v5
approved 17:39 UK; shape agreed by DA and FE2, 20 Oct).

One row per archive, keyed by the rule file's id; a restore sets restored_at, restored_by and restore_reason once, and
archiving again is a new row. A partial unique index allows one open row per file. Write-once is held by the database:
no delete, and an update may only set the restore, once, all three together (Postgres: one plpgsql function and a
BEFORE UPDATE OR DELETE trigger; SQLite: two BEFORE triggers), as trials (0018). No money columns, no Strategy column.

Additive only: a new table. Skipped when a Store on the same code built it first (create_all). Downgrade drops it only
while it is empty, and refuses otherwise: the archive is the PM's record (never destroy data without a DA review).

Revision ID: 0031
Revises: 0030
Create Date: 2026-10-20 16:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0031'
down_revision = '0030'
branch_labels = None
depends_on = None


# Written out, not imported: a migration is frozen at what it was when it ran
REASONS = ('no_edge', 'superseded', 'duplicate', 'not_wanted')
OPEN = 'restored_at IS NULL'
PG = ("CREATE OR REPLACE FUNCTION definition_archive_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF TG_OP = 'DELETE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'definition_archive is write-once: a row cannot be deleted'; END IF; IF (NEW.id, NEW.file_id, NEW.definition_hash, NEW.status_at_archive, NEW.reason, NEW.note, NEW.actor, NEW.archived_at) IS DISTINCT FROM (OLD.id, OLD.file_id, OLD.definition_hash, OLD.status_at_archive, OLD.reason, OLD.note, OLD.actor, OLD.archived_at) OR OLD.restored_at IS NOT NULL OR NEW.restored_at IS NULL OR NEW.restored_by IS NULL OR NEW.restore_reason IS NULL THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'definition_archive is write-once: only a restore, once, with its time, actor and reason'; END IF; RETURN NEW; END $$", 'CREATE TRIGGER definition_archive_write_once BEFORE UPDATE OR DELETE ON definition_archive FOR EACH ROW EXECUTE FUNCTION definition_archive_write_once()')
SQLITE = ("CREATE TRIGGER definition_archive_no_delete BEFORE DELETE ON definition_archive BEGIN SELECT RAISE(ABORT, 'definition_archive is write-once: a row cannot be deleted'); END", "CREATE TRIGGER definition_archive_no_edit BEFORE UPDATE ON definition_archive WHEN NEW.id IS NOT OLD.id OR NEW.file_id IS NOT OLD.file_id OR NEW.definition_hash IS NOT OLD.definition_hash OR NEW.status_at_archive IS NOT OLD.status_at_archive OR NEW.reason IS NOT OLD.reason OR NEW.note IS NOT OLD.note OR NEW.actor IS NOT OLD.actor OR NEW.archived_at IS NOT OLD.archived_at OR OLD.restored_at IS NOT NULL OR NEW.restored_at IS NULL OR NEW.restored_by IS NULL OR NEW.restore_reason IS NULL BEGIN SELECT RAISE(ABORT, 'definition_archive is write-once: only a restore, once, with its time, actor and reason'); END")


def upgrade() -> None:
    bind = op.get_bind()
    if 'definition_archive' in sa.inspect(bind).get_table_names():
        return  # a Store on this code built the whole shape, triggers included (create_all)
    big_id = sa.BigInteger().with_variant(sa.Integer(), 'sqlite')
    op.create_table(
        'definition_archive',
        sa.Column('id', big_id, primary_key=True, autoincrement=True),
        sa.Column('file_id', sa.String(length=64), nullable=False),
        sa.Column('definition_hash', sa.String(length=64), nullable=True),
        sa.Column('status_at_archive', sa.String(length=32), nullable=False),
        sa.Column('reason', sa.String(length=32), nullable=False),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('actor', sa.String(length=32), nullable=False),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('restored_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('restored_by', sa.String(length=32), nullable=True),
        sa.Column('restore_reason', sa.Text(), nullable=True),
        sa.CheckConstraint(f"reason IN ({', '.join(repr(r) for r in REASONS)})", name='definition_archive_reason'),
    )
    op.create_index('definition_archive_open', 'definition_archive', ['file_id'], unique=True,
                    postgresql_where=sa.text(OPEN), sqlite_where=sa.text(OPEN))
    for sql in (PG if bind.dialect.name == 'postgresql' else SQLITE):
        op.execute(sql)


def downgrade() -> None:
    bind = op.get_bind()
    n = bind.execute(sa.text('SELECT COUNT(*) FROM definition_archive')).scalar_one()
    if n:
        raise RuntimeError(f'definition_archive holds {n} rows, the PM\'s archive record: not dropped. Ask the Data '
                           'Architect and the HoE')
    if bind.dialect.name == 'postgresql':
        op.execute('DROP TRIGGER IF EXISTS definition_archive_write_once ON definition_archive')
        op.execute('DROP FUNCTION IF EXISTS definition_archive_write_once()')
    op.drop_index('definition_archive_open', table_name='definition_archive')
    op.drop_table('definition_archive')
