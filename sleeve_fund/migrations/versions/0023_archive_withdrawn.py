"""F-C3B-A: the equity a Strategy took out of the fund's book when it was put away.

sleeve_archive.withdrawn (nullable exact number). Set, with its portfolio flow, when a Strategy is archived, retired
or archived by a clean slate from now on; fund_equity skips those. NULL on every row already there: those archives
still count in the book, as they did, until the one-off `python -m sleeve_fund.supervisor withdraw-archived` takes
each out with its own flow (Advisor 14 Oct 09:45 UK). A reset's run is archived with NULL too: its cash stays.
A Strategy archived while not flat is NULL as well until it is flat: it stays in the book in full meanwhile, and
the supervisor's withdrawal fills it in (Advisor 14 Oct 10:45 UK, option b). NULL always means it still counts.

portfolio_state.carried_* (MUST-F286-H4, Advisor 14 Oct 13:35 UK): what a withdrawal that empties the book carries
to the next capital's first mark, which restores the drawdown and the day's change from them and clears them.

Additive only; nothing existing is edited. Skipped when a Store on the same code built the column first (create_all).

Revision ID: 0023
Revises: 0020
Create Date: 2026-10-14 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

EXACT = sa.Numeric(38, 18)  # as sleeve_fund.exact, frozen here
# What a withdrawal that empties the book carries to the next capital's first mark, from the book before it:
CARRIED = (
    ('carried_hwm_drawdown', EXACT),  # 1 - book / hwm
    ('carried_reference_drawdown', EXACT),  # 1 - book / reference_equity; NULL when the reference was the HWM
    ('carried_day_return', EXACT),  # book / day_start_equity - 1, for carried_day
    ('carried_day', sa.Date()),  # the UTC day carried_day_return belongs to
)
# As sleeve_fund.store.PORTFOLIO_STATE_CARRIED_CHECKS, frozen here
CARRIED_CHECKS = (
    ('unmarked', 'marked_at IS NULL OR (carried_hwm_drawdown IS NULL AND carried_reference_drawdown IS NULL '
                 'AND carried_day_return IS NULL AND carried_day IS NULL)'),
    ('anchor', 'carried_hwm_drawdown IS NOT NULL OR (carried_reference_drawdown IS NULL AND carried_day_return IS NULL)'),
    ('day', '(carried_day_return IS NULL) = (carried_day IS NULL)'),
    ('hwm_range', 'carried_hwm_drawdown IS NULL OR (carried_hwm_drawdown >= 0 AND carried_hwm_drawdown < 1)'),
    ('reference_range', 'carried_reference_drawdown IS NULL OR '
                        '(carried_reference_drawdown >= 0 AND carried_reference_drawdown < 1)'),
    ('day_range', 'carried_day_return IS NULL OR carried_day_return > -1'),
)


revision = '0023'
down_revision = '0020'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if 'withdrawn' not in {c['name'] for c in sa.inspect(bind).get_columns('sleeve_archive')}:
        with op.batch_alter_table('sleeve_archive', schema=None) as batch_op:
            batch_op.add_column(sa.Column('withdrawn', EXACT, nullable=True))
    # MUST-F286-H4 (Advisor 14 Oct 13:35 UK; DA): each guarded, as above
    have = {c['name'] for c in sa.inspect(bind).get_columns('portfolio_state')}
    checks = {c['name'] for c in sa.inspect(bind).get_check_constraints('portfolio_state')}
    with op.batch_alter_table('portfolio_state', schema=None) as batch_op:
        for name, kind in CARRIED:
            if name not in have:
                batch_op.add_column(sa.Column(name, kind, nullable=True))
        for name, sql in CARRIED_CHECKS:
            if f'portfolio_state_carried_{name}' not in checks:
                batch_op.create_check_constraint(f'portfolio_state_carried_{name}', sql)


def downgrade() -> None:
    raise NotImplementedError('dropping it would put every withdrawn Strategy back in the book with no flow, and an '
                              "emptied book's drawdown would be lost: never")
