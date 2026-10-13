"""P2-8 two legs: a two-leg Strategy's second leg is a sleeves row of its own, and both legs' orders share a trade id.

sleeves.parent: the first leg's name on the second leg's row "<name>~2", NULL on every other row; FK to sleeves.name,
ON DELETE RESTRICT. CHECK sleeves_leg_name holds the shape on both engines without a trigger (DA, 14 Oct): a "~" only as
the "~2" suffix, never self, one level (a parent holds no "~"), and at most one leg (its name is fixed and unique).
orders.parent_trade_id: one id for every order of a two-leg trade across both rows, indexed alone, since the restart
check joins the two legs across two sleeve names.

Additive only. Each step is skipped when a Store on the same code built that shape first (create_all), as 0017 does.
A sleeve name holding "~" already would fail the CHECK: the migration stops and lists them (never edits or renames: PM
rule), for the Data Architect and the HoE. On SQLite the CHECK rebuilds sleeves in batch mode; its FK and CHECK survive
(pinned in test_two_legs_schema). 0022 in the DA's register (14 Oct); 0021 is reserved for netting, and this chains
onto whatever is main's head when its PR opens.

Revision ID: 0022
Revises: 0014 (re-chained at PR time per the register)
Create Date: 2026-10-14 00:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0022'
down_revision = '0014'
branch_labels = None
depends_on = None


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


def downgrade() -> None:
    raise NotImplementedError('a two-leg Strategy\'s legs and their trade ids are its record: never dropped')
