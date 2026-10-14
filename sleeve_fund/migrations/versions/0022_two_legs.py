"""P2-8 two legs: a two-leg Strategy's second leg is a sleeves row of its own, and both legs' orders share a trade id.

sleeves.parent: the first leg's name on the second leg's row "<name>~2", NULL on every other row; FK to sleeves.name,
ON DELETE RESTRICT. CHECK sleeves_leg_name holds the shape on both engines without a trigger (DA, 14 Oct): a "~" only as
the "~2" suffix, never self, one level (a parent holds no "~"), and at most one leg (its name is fixed and unique).
orders.parent_trade_id: one id for every order of a two-leg trade across both rows, indexed alone, since the restart
check joins the two legs across two sleeve names.
pair_state: a two-leg Strategy's coordinator, one row per parent (FK sleeves.name, ON DELETE RESTRICT): its state
(a hint after a restart; the fills by parent_trade_id are the truth), the open trade's id and the legged state's
deadline, and the summed guard's figures, split as portfolio_state's (Advisor M3; DA 14 Oct 05:02 UK).

Additive only. Each step is skipped when a Store on the same code built that shape first (create_all), as 0017 does.
A sleeve name holding "~" already would fail the CHECK: the migration stops and lists them (never edits or renames: PM
rule), for the Data Architect and the HoE. On SQLite the CHECK rebuilds sleeves in batch mode; its FK and CHECK survive
(pinned in test_two_legs_schema). 0022 in the DA's register (14 Oct); 0021 is reserved for netting, and this chains
onto whatever is main's head when its PR opens.

Revision ID: 0022
Revises: 0020 (main's head after #259: TRIALS-WORM 0018 and orders.venue 0020)
Create Date: 2026-10-14 00:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0022'
down_revision = '0020'
branch_labels = None
depends_on = None


EXACT = sa.Numeric(38, 18)
TS = sa.DateTime(timezone=True)
FIGURES = ('reference_equity', 'hwm', 'day_start_equity', 'day_start')
SHAPE = ("(state = 'flat' AND trade_id IS NULL AND deadline IS NULL) "
         "OR (state = 'hedged' AND trade_id IS NOT NULL AND deadline IS NULL) "
         "OR (state IN ('legging', 'closing') AND trade_id IS NOT NULL AND deadline IS NOT NULL)")

# "parent IS NOT NULL" is needed: with a NULL parent the second term is NULL, and a CHECK passes on NULL.
LEG_NAME = ("(parent IS NULL AND name NOT LIKE '%~%') "
            "OR (parent IS NOT NULL AND name = parent || '~2' AND parent NOT LIKE '%~%')")


def upgrade() -> None:
    bind = op.get_bind()
    have = sa.inspect(bind)
    if 'parent' not in {c['name'] for c in have.get_columns('sleeves')}:
        rows = bind.execute(sa.text("SELECT name FROM sleeves WHERE name LIKE '%~%' ORDER BY name LIMIT 50"))
        found = [r.name for r in rows]
        if found:
            raise RuntimeError(f"sleeve names holding '~' ({len(found)} shown, none changed): {'; '.join(found)}. "
                               "Ask the Data Architect and the HoE before running this migration again")
        with op.batch_alter_table('sleeves', schema=None) as batch_op:
            batch_op.add_column(sa.Column('parent', sa.String(length=64), nullable=True))
            batch_op.create_foreign_key('sleeves_parent_fk', 'sleeves', ['parent'], ['name'], ondelete='RESTRICT')
            batch_op.create_check_constraint('sleeves_leg_name', sa.text(LEG_NAME))
    if 'parent_trade_id' not in {c['name'] for c in have.get_columns('orders')}:
        with op.batch_alter_table('orders', schema=None) as batch_op:
            batch_op.add_column(sa.Column('parent_trade_id', sa.String(length=64), nullable=True))
            batch_op.create_index('orders_parent_trade_id', ['parent_trade_id'], unique=False)
    if 'pair_state' not in have.get_table_names():
        op.create_table(
            'pair_state',
            sa.Column('parent', sa.String(length=64), nullable=False),
            sa.Column('state', sa.String(length=8), nullable=False),
            sa.Column('trade_id', sa.String(length=64), nullable=True),
            sa.Column('deadline', TS, nullable=True),
            sa.Column('reference_equity', EXACT, nullable=True),
            sa.Column('hwm', EXACT, nullable=True),
            sa.Column('day_start_equity', EXACT, nullable=True),
            sa.Column('day_start', sa.Date(), nullable=True),
            sa.Column('marked_at', TS, nullable=True),
            sa.Column('updated_at', TS, nullable=False),
            sa.ForeignKeyConstraint(['parent'], ['sleeves.name'], name='pair_state_parent_fk', ondelete='RESTRICT'),
            sa.PrimaryKeyConstraint('parent'),
            sa.CheckConstraint("state IN ('flat', 'legging', 'hedged', 'closing')", name='pair_state_values'),
            sa.CheckConstraint(SHAPE, name='pair_state_shape'),
            *(sa.CheckConstraint(f'({c} IS NULL) = (marked_at IS NULL)', name=f'pair_state_{c}_marked')
              for c in FIGURES),
        )


def downgrade() -> None:
    raise NotImplementedError('a two-leg Strategy\'s legs and their trade ids are its record: never dropped')
