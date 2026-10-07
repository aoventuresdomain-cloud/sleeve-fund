"""DA-9: money and quantity columns are exact. Each moves from a binary float to NUMERIC(38, 18): 20 places before
the point and 18 after. Pure ratios (rates, fractions, fee and spread shares, Sharpe) stay float.

Each value is converted as the decimal it prints as (Postgres: float -> text -> numeric, the shortest text that reads
back as the same float, which is what Python's repr gives), so 0.1 becomes 0.1 and history is not re-rounded beyond
the 18 places the column keeps. No row is added or removed.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-07 21:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0009'
down_revision = '0008'
branch_labels = None
depends_on = None

# table: (column, nullable)
COLUMNS = {
    'sleeves': [('starting_balance', False)],
    'equity': [('equity', False), ('cash', False), ('qty', False), ('price', False), ('benchmark', False)],
    'fills': [('qty', False), ('price', False), ('fee', False)],
    'funding': [('qty', False), ('price', False), ('amount', False)],
    'insurance': [('price', False), ('amount', False)],
    'demo_mirror': [('amount', False), ('price', True)],
    'orders': [('qty', False), ('filled_qty', False), ('avg_px', True), ('fee', False)],
    'exit_plans': [('risk_amount', True)],
}
EXACT = sa.Numeric(38, 18)


def upgrade() -> None:
    pg = op.get_bind().dialect.name == 'postgresql'
    for table, columns in COLUMNS.items():
        with op.batch_alter_table(table, schema=None) as batch_op:
            for column, nullable in columns:
                batch_op.alter_column(column, existing_type=sa.Float(), type_=EXACT, existing_nullable=nullable,
                                      postgresql_using=f'{column}::text::numeric(38, 18)' if pg else None)


def downgrade() -> None:
    raise NotImplementedError('money columns stay exact: a float would re-introduce binary rounding')
