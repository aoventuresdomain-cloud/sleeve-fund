"""ORPHAN-GUARD: the supervisor's cancel of one resting order (Advisor 8 Oct 06:15 and 06:20 UK).

commands: order_id, the resting order a cancel names (FK orders.order_id); a CHECK that a cancel always names one and
nothing else does (NULL never means "every order"); a partial unique index, one open cancel per order, so a retry, a
second sweep or a restart never sends two; not_found_at, when the venue first answered a cancel "not found" with no
fill (no proof of a cancel: its reservation waits for CANCEL_CONFIRM, PE2 13 Oct), with a CHECK that only a cancel has
one; and a CHECK on the kinds a row may hold (store.COMMAND_KINDS), which until
now only Python checked. gate_decisions: an index on order_id, which every fill and status change of a gated order and
every cancel reads (PE2 and QA Tester 2, 13 Oct).

Additive only: no row is changed or removed. If a row already holds a kind outside the list, the migration stops and
lists them (never deletes: PM rule), as 0008 and 0013 do. Each step is skipped when a Store on the same code built
that shape first (create_all).

Revision ID: 0015
Revises: 0013 (0014, FE's F217-3, is reserved and not merged: the PR that merges second re-chains, per the register)
Create Date: 2026-10-13 15:30:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0015'
down_revision = '0013'
branch_labels = None
depends_on = None


# Written out, not imported: a migration is frozen at what it was when it ran
KINDS = ('book_reset', 'flatten', 'pause', 'reset_after_liquidation', 'resume', 'reload', 'cancel')
KIND_IN = f"command IN ({', '.join(repr(k) for k in KINDS)})"
OPEN = "command = 'cancel' AND applied_at IS NULL"


def upgrade() -> None:
    bind = op.get_bind()
    odd = bind.execute(sa.text(f"SELECT command, COUNT(*) AS n FROM commands WHERE NOT ({KIND_IN}) "
                               "GROUP BY command ORDER BY command LIMIT 50")).fetchall()
    if odd:
        raise RuntimeError(f"commands of a kind no code writes ({len(odd)} kinds shown, none changed): "
                           f"{'; '.join(f'{r.command!r} x{r.n}' for r in odd)}. Ask the Data Architect and the HoE "
                           "before running this migration again")
    have = sa.inspect(bind)
    if 'gate_decisions_order_id' not in {i['name'] for i in have.get_indexes('gate_decisions')}:
        op.create_index('gate_decisions_order_id', 'gate_decisions', ['order_id'])
    if 'order_id' in {c['name'] for c in have.get_columns('commands')}:
        return  # a Store on this code built the whole shape (create_all)
    with op.batch_alter_table('commands', schema=None) as batch_op:
        batch_op.add_column(sa.Column('order_id', sa.String(length=64), nullable=True))
        batch_op.create_foreign_key('commands_order_id_fkey', 'orders', ['order_id'], ['order_id'],
                                    ondelete='RESTRICT')
        batch_op.create_check_constraint('commands_kind', KIND_IN)
        batch_op.create_check_constraint('commands_cancel_order', "(command = 'cancel') = (order_id IS NOT NULL)")
        batch_op.add_column(sa.Column('not_found_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.create_check_constraint('commands_not_found_cancel', "not_found_at IS NULL OR command = 'cancel'")
        batch_op.create_index('commands_cancel_open', ['order_id'], unique=True,
                              postgresql_where=sa.text(OPEN), sqlite_where=sa.text(OPEN))


def downgrade() -> None:
    raise NotImplementedError('a cancel names its order and is sent once: never dropped')
