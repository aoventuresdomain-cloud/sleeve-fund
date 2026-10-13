"""CASH-4: a funding true-up moves cash when the venue's real rate arrives, not at its settlement's ts.

funding.arrived_at (nullable timestamp), set on a true-up only, with CHECK funding_true_up_arrived:
(kind = 'true_up') = (arrived_at IS NOT NULL). The as-of readers count a true-up from arrived_at, so a replay before
the rate arrived shows the baseline alone (Advisor 8 Oct 04:10, ruling 2).

Additive only. Nothing has written a true-up before this (the kind has been allowed since 0006): if one is there, it
has no arrival to give it, so the migration stops and lists it (never edits or deletes: PM rule) for the Data
Architect and the HoE. Skipped when a Store on the same code built the column first (create_all).

Revision ID: 0016
Revises: 0014
Create Date: 2026-10-13 16:40:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0016'
down_revision = '0014'
branch_labels = None
depends_on = None


ARRIVED = "(kind = 'true_up') = (arrived_at IS NOT NULL)"


def upgrade() -> None:
    bind = op.get_bind()
    if 'arrived_at' in {c['name'] for c in sa.inspect(bind).get_columns('funding')}:
        return
    rows = bind.execute(sa.text("SELECT sleeve, ts FROM funding WHERE kind = 'true_up' ORDER BY sleeve, ts LIMIT 50"))
    found = [f"{r.sleeve} {r.ts}" for r in rows]
    if found:
        raise RuntimeError(f"funding true-ups with no arrival time ({len(found)} shown, none changed): "
                           f"{'; '.join(found)}. Ask the Data Architect and the HoE before running this migration again")
    with op.batch_alter_table('funding', schema=None) as batch_op:
        batch_op.add_column(sa.Column('arrived_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.create_check_constraint('funding_true_up_arrived', sa.text(ARRIVED))


def downgrade() -> None:
    raise NotImplementedError('a true-up without its arrival would move cash at the wrong time: never dropped')
