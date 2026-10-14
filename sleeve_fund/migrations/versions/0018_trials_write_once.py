"""TRIALS-WORM (QD #238, HoE 13 Oct): the trials register is write-once, held by the database itself.

A row is never deleted, and an UPDATE may change only status and error (a failed count re-counted). Anything else,
created_at, definition_hash, settings and the data range above all, is refused with a check violation, so N, the
single-setting exemption's earliest-row rule and the holdout check can't be edited after the fact. Postgres: one
plpgsql function and a BEFORE UPDATE OR DELETE trigger; SQLite: two BEFORE triggers.

Additive only: nothing is changed or removed; no row is read. Skipped when a Store on the same code built the triggers
first (create_all).

Revision ID: 0018
Revises: 0016 (CASH-4, merged; 0015 ORPHAN-GUARD is reserved and not merged: the PR that merges later re-chains,
per the register)
Create Date: 2026-10-13 17:40:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0018'
down_revision = '0016'
branch_labels = None
depends_on = None


# Written out, not imported: a migration is frozen at what it was when it ran
PG = ("CREATE OR REPLACE FUNCTION trials_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF TG_OP = 'DELETE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'trials are write-once: ' || OLD.id || ' cannot be deleted'; END IF; IF (NEW.id, NEW.definition_hash, NEW.idea_hash, NEW.code_version, NEW.definition_name, NEW.family, NEW.settings, NEW.dataset, NEW.stage, NEW.source, NEW.sharpe, NEW.trades, NEW.oos_trades, NEW.backtest_id, NEW.created_at, NEW.data_start, NEW.data_end, NEW.kind) IS DISTINCT FROM (OLD.id, OLD.definition_hash, OLD.idea_hash, OLD.code_version, OLD.definition_name, OLD.family, OLD.settings, OLD.dataset, OLD.stage, OLD.source, OLD.sharpe, OLD.trades, OLD.oos_trades, OLD.backtest_id, OLD.created_at, OLD.data_start, OLD.data_end, OLD.kind) THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'trials are write-once: only status and error of ' || OLD.id || ' may change'; END IF; RETURN NEW; END $$", 'CREATE TRIGGER trials_write_once BEFORE UPDATE OR DELETE ON trials FOR EACH ROW EXECUTE FUNCTION trials_write_once()')
SQLITE = ("CREATE TRIGGER trials_no_delete BEFORE DELETE ON trials BEGIN SELECT RAISE(ABORT, 'trials are write-once: a row cannot be deleted'); END", "CREATE TRIGGER trials_no_edit BEFORE UPDATE ON trials WHEN NEW.id IS NOT OLD.id OR NEW.definition_hash IS NOT OLD.definition_hash OR NEW.idea_hash IS NOT OLD.idea_hash OR NEW.code_version IS NOT OLD.code_version OR NEW.definition_name IS NOT OLD.definition_name OR NEW.family IS NOT OLD.family OR NEW.settings IS NOT OLD.settings OR NEW.dataset IS NOT OLD.dataset OR NEW.stage IS NOT OLD.stage OR NEW.source IS NOT OLD.source OR NEW.sharpe IS NOT OLD.sharpe OR NEW.trades IS NOT OLD.trades OR NEW.oos_trades IS NOT OLD.oos_trades OR NEW.backtest_id IS NOT OLD.backtest_id OR NEW.created_at IS NOT OLD.created_at OR NEW.data_start IS NOT OLD.data_start OR NEW.data_end IS NOT OLD.data_end OR NEW.kind IS NOT OLD.kind BEGIN SELECT RAISE(ABORT, 'trials are write-once: only status and error may change'); END")


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        have = bind.execute(sa.text("SELECT 1 FROM pg_trigger WHERE tgname = 'trials_write_once' "
                                    "AND tgrelid = 'trials'::regclass")).first()
        statements = PG
    else:
        have = bind.execute(sa.text("SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
                                    "AND name = 'trials_no_edit'")).first()
        statements = SQLITE
    if have:
        return  # a Store on this code built them (create_all)
    for sql in statements:
        op.execute(sql)


def downgrade() -> None:
    raise NotImplementedError('the trials register stays write-once: never dropped')
