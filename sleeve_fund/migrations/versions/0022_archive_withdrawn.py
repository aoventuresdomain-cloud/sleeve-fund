"""F-C3B-A: the equity a Strategy took out of the fund's book when it was put away.

sleeve_archive.withdrawn (nullable exact number). Set, with its portfolio flow, when a Strategy is archived, retired
or archived by a clean slate from now on; fund_equity skips those. NULL on every row already there: those archives
still count in the book, as they did, until the one-off `python -m sleeve_fund.supervisor withdraw-archived` takes
each out with its own flow (Advisor 14 Oct 09:45 UK). A reset's run is archived with NULL too: its cash stays.

Additive only; nothing existing is edited. Skipped when a Store on the same code built the column first (create_all).

Revision ID: 0022
Revises: 0020
Create Date: 2026-10-14 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

EXACT = sa.Numeric(38, 18)  # as sleeve_fund.exact, frozen here


revision = '0022'
down_revision = '0020'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if 'withdrawn' in {c['name'] for c in sa.inspect(bind).get_columns('sleeve_archive')}:
        return
    with op.batch_alter_table('sleeve_archive', schema=None) as batch_op:
        batch_op.add_column(sa.Column('withdrawn', EXACT, nullable=True))


def downgrade() -> None:
    raise NotImplementedError('dropping it would put every withdrawn Strategy back in the book with no flow: never')
