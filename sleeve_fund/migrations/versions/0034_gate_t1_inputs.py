"""PAGE-SIZE-T1 (DA, 20 Oct; Advisor PAGE-SIZE, HoE option (a)): the inputs T1 judged each gate decision on.

gate_decisions: t1_used_before (the loss at the stops already in the order's underlying, reservations in), t1_cap and
risk_per_unit (the order's, expected fill to stop with the fee), in fund currency like book_equity, and size_share
(approved_qty x risk_per_unit over the room, cap less used; a fraction, NULL with no room). Exact, nullable; QD2's
writer fills them, the page's "sized at x% of the room T1 allows" and "risk if the stop is hit" read them. A trigger
holds their shape, so gate_decisions is not rebuilt on SQLite: the first three all or none (T1 off, or not reached),
size_share exactly when they leave room, none below 0, risk per unit above 0. None of the four ever changes, a NULL
included (no backfill onto an old row), nor book_equity once set: the page's "% of the fund" divides by it.

Additive only, skipped when a Store on the same code built it first (create_all). Downgrade drops them only while no
row holds them, and refuses otherwise (never destroy data without a DA review).

Revision ID: 0034
Revises: 0033
Create Date: 2026-10-20 21:30:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0034'
down_revision = '0033'
branch_labels = None
depends_on = None


# Written out, not imported: a migration is frozen at what it was when it ran
EXACT = sa.Numeric(38, 18)
COLUMNS = ('t1_used_before', 't1_cap', 'risk_per_unit', 'size_share')
BAD = ("NOT ((NEW.t1_used_before IS NULL AND NEW.t1_cap IS NULL AND NEW.risk_per_unit IS NULL "
       "AND NEW.size_share IS NULL) OR (NEW.t1_used_before IS NOT NULL AND NEW.t1_cap IS NOT NULL "
       "AND NEW.risk_per_unit IS NOT NULL)) OR NEW.t1_used_before < 0 OR NEW.t1_cap < 0 "
       "OR NEW.risk_per_unit <= 0 OR NEW.size_share < 0 "
       "OR (NEW.t1_cap <= NEW.t1_used_before AND NEW.size_share IS NOT NULL) "
       "OR (NEW.t1_cap > NEW.t1_used_before AND NEW.size_share IS NULL)")
MESSAGE = ("gate_decisions: the T1 used, cap and risk per unit are all set or none, size share exactly when "
           "they leave room, none below 0, risk per unit above 0; they never change, nor book_equity once set")
FROZEN_PG = ("(NEW.t1_used_before, NEW.t1_cap, NEW.risk_per_unit, NEW.size_share) IS DISTINCT FROM (OLD.t1_used_before, "
             "OLD.t1_cap, OLD.risk_per_unit, OLD.size_share) OR (OLD.book_equity IS NOT NULL AND NEW.book_equity IS DISTINCT "
             "FROM OLD.book_equity)")
FROZEN_SQLITE = ("NEW.t1_used_before IS NOT OLD.t1_used_before OR NEW.t1_cap IS NOT OLD.t1_cap OR NEW.risk_per_unit IS NOT "
                 "OLD.risk_per_unit OR NEW.size_share IS NOT OLD.size_share OR (OLD.book_equity IS NOT NULL AND NEW.book_equity "
                 "IS NOT OLD.book_equity)")
PG = (
    "CREATE OR REPLACE FUNCTION gate_t1_shape() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    f"IF {BAD} OR (TG_OP = 'UPDATE' AND {FROZEN_PG}) THEN RAISE EXCEPTION "
    f"USING ERRCODE = 'check_violation', MESSAGE = '{MESSAGE}'; END IF; RETURN NEW; END $$",
    "CREATE TRIGGER gate_decisions_t1 BEFORE INSERT OR UPDATE ON gate_decisions FOR EACH ROW "
    "EXECUTE FUNCTION gate_t1_shape()",
)
SQLITE = (
    f"CREATE TRIGGER gate_decisions_t1_insert BEFORE INSERT ON gate_decisions WHEN {BAD} "
    f"BEGIN SELECT RAISE(ABORT, '{MESSAGE}'); END",
    f"CREATE TRIGGER gate_decisions_t1_update BEFORE UPDATE ON gate_decisions WHEN {BAD} "
    f"OR ({FROZEN_SQLITE}) BEGIN SELECT RAISE(ABORT, '{MESSAGE}'); END",
)


def upgrade() -> None:
    bind = op.get_bind()
    if 't1_cap' in {c['name'] for c in sa.inspect(bind).get_columns('gate_decisions')}:
        return
    for name in COLUMNS:
        op.add_column('gate_decisions', sa.Column(name, EXACT, nullable=True))
    for sql in PG if bind.dialect.name == 'postgresql' else SQLITE:
        op.execute(sql)


def downgrade() -> None:
    bind = op.get_bind()
    kept = bind.execute(sa.text('SELECT COUNT(*) FROM gate_decisions WHERE '
                                + ' OR '.join(f'{c} IS NOT NULL' for c in COLUMNS))).scalar_one()
    if kept:
        raise RuntimeError(f'{kept} gate decisions hold the T1 inputs they were judged on: not dropped. Ask the Data '
                           'Architect and the HoE')
    if bind.dialect.name == 'postgresql':
        op.execute('DROP TRIGGER IF EXISTS gate_decisions_t1 ON gate_decisions')
        op.execute('DROP FUNCTION IF EXISTS gate_t1_shape()')
    else:
        for o in ('insert', 'update'):
            op.execute(f'DROP TRIGGER IF EXISTS gate_decisions_t1_{o}')
    for name in reversed(COLUMNS):
        op.drop_column('gate_decisions', name)  # plain DROP COLUMN on SQLite (3.35+): no foreign key on these
