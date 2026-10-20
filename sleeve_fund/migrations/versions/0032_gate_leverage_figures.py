"""B-24 (DA, 20 Oct; Advisor B24-LEVERAGE, QD): the gate's leverage figures, on each decision and at each of its fills.

gate_decisions: margin_leverage, share_of_fund, gross_after, net_underlying_after (signed) and held_before (the
Strategy's own position before the order, signed; QD), nullable. A trigger holds their shape, so gate_decisions is not
rebuilt on SQLite: all five or none (rows before it, error rows), only net and held_before below 0, and once set they never
change (a row without them may get them once).

gate_fill_marks: the same four at a fill's price, one per (decision, trade id), write-once. Keyed by the trade id, not
fills.id: a research run's gate fills have no fills row. price_source is expected_fill, fill or mark.

Additive only. Each part is skipped when a Store on the same code built it first (create_all). Downgrade drops them only
while no figure is kept, and refuses otherwise (never destroy data without a DA review).

Revision ID: 0032
Revises: 0031
Create Date: 2026-10-20 18:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0032'
down_revision = '0031'
branch_labels = None
depends_on = None


# Written out, not imported: a migration is frozen at what it was when it ran
EXACT = sa.Numeric(38, 18)
TS = sa.DateTime(timezone=True)
BIG_ID = sa.BigInteger().with_variant(sa.Integer(), 'sqlite')
FIGURES = ('margin_leverage', 'share_of_fund', 'gross_after', 'net_underlying_after')
DECISION = (*FIGURES, 'held_before')
SOURCES = ('expected_fill', 'fill', 'mark')
BAD = ("NOT ((NEW.margin_leverage IS NULL AND NEW.share_of_fund IS NULL AND NEW.gross_after IS NULL "
       "AND NEW.net_underlying_after IS NULL AND NEW.held_before IS NULL) OR (NEW.margin_leverage IS NOT NULL "
       "AND NEW.share_of_fund IS NOT NULL AND NEW.held_before IS NOT NULL "
       "AND NEW.gross_after IS NOT NULL AND NEW.net_underlying_after IS NOT NULL)) "
       "OR NEW.margin_leverage < 0 OR NEW.share_of_fund < 0 OR NEW.gross_after < 0")
MESSAGE = ('gate_decisions: the leverage figures and held_before are all set or none, only net and held_before '
           'may be below 0, and once set they never change')
FROZEN_PG = ("OLD.margin_leverage IS NOT NULL AND (NEW.margin_leverage, NEW.share_of_fund, NEW.gross_after, "
             "NEW.net_underlying_after, NEW.held_before) IS DISTINCT FROM (OLD.margin_leverage, OLD.share_of_fund, "
             "OLD.gross_after, OLD.net_underlying_after, OLD.held_before)")
FROZEN_SQLITE = ("OLD.margin_leverage IS NOT NULL AND (NEW.margin_leverage IS NOT OLD.margin_leverage OR NEW.share_of_fund "
                 "IS NOT OLD.share_of_fund OR NEW.gross_after IS NOT OLD.gross_after OR NEW.net_underlying_after IS NOT "
                 "OLD.net_underlying_after OR NEW.held_before IS NOT OLD.held_before)")
PG = (
    "CREATE OR REPLACE FUNCTION gate_leverage_shape() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    f"IF {BAD} OR (TG_OP = 'UPDATE' AND {FROZEN_PG}) THEN RAISE EXCEPTION "
    f"USING ERRCODE = 'check_violation', MESSAGE = '{MESSAGE}'; END IF; RETURN NEW; END $$",
    "CREATE TRIGGER gate_decisions_leverage BEFORE INSERT OR UPDATE ON gate_decisions FOR EACH ROW "
    "EXECUTE FUNCTION gate_leverage_shape()",
)
SQLITE = (
    f"CREATE TRIGGER gate_decisions_leverage_insert BEFORE INSERT ON gate_decisions WHEN {BAD} "
    f"BEGIN SELECT RAISE(ABORT, '{MESSAGE}'); END",
    f"CREATE TRIGGER gate_decisions_leverage_update BEFORE UPDATE ON gate_decisions WHEN {BAD} "
    f"OR ({FROZEN_SQLITE}) BEGIN SELECT RAISE(ABORT, '{MESSAGE}'); END",
)
PG_MARKS = (
    "CREATE OR REPLACE FUNCTION gate_fill_mark_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'gate_fill_marks is write-once'; END $$",
    "CREATE TRIGGER gate_fill_marks_write_once BEFORE UPDATE OR DELETE ON gate_fill_marks FOR EACH ROW "
    "EXECUTE FUNCTION gate_fill_mark_write_once()",
)
SQLITE_MARKS = tuple(f"CREATE TRIGGER gate_fill_marks_no_{o.lower()} BEFORE {o} ON gate_fill_marks BEGIN "
                     "SELECT RAISE(ABORT, 'gate_fill_marks is write-once'); END" for o in ('UPDATE', 'DELETE'))


def upgrade() -> None:
    bind = op.get_bind()
    pg = bind.dialect.name == 'postgresql'
    have = sa.inspect(bind)
    if 'margin_leverage' not in {c['name'] for c in have.get_columns('gate_decisions')}:
        for name in DECISION:
            op.add_column('gate_decisions', sa.Column(name, EXACT, nullable=True))
        for sql in PG if pg else SQLITE:
            op.execute(sql)
    if 'gate_fill_marks' not in have.get_table_names():
        op.create_table(
            'gate_fill_marks',
            sa.Column('id', BIG_ID, primary_key=True, autoincrement=True),
            sa.Column('decision_id', BIG_ID, nullable=False),
            sa.Column('trade_id', sa.String(length=64), nullable=False),
            sa.Column('price', EXACT, nullable=False),
            sa.Column('price_source', sa.String(length=16), nullable=False),
            *(sa.Column(name, EXACT, nullable=False) for name in FIGURES),
            sa.Column('marked_at', TS, nullable=False),
            sa.ForeignKeyConstraint(['decision_id'], ['gate_decisions.id'], ondelete='RESTRICT'),
            sa.UniqueConstraint('decision_id', 'trade_id', name='gate_fill_marks_decision_trade'),
            sa.CheckConstraint(f"price_source IN ({', '.join(repr(s) for s in SOURCES)})",
                               name='gate_fill_marks_price_source'),
            sa.CheckConstraint('price > 0', name='gate_fill_marks_price'),
            sa.CheckConstraint('margin_leverage >= 0 AND share_of_fund >= 0 AND gross_after >= 0',
                               name='gate_fill_marks_figures'),
        )
        for sql in PG_MARKS if pg else SQLITE_MARKS:
            op.execute(sql)


def downgrade() -> None:
    bind = op.get_bind()
    pg = bind.dialect.name == 'postgresql'
    marks = bind.execute(sa.text('SELECT COUNT(*) FROM gate_fill_marks')).scalar_one()
    kept = bind.execute(sa.text('SELECT COUNT(*) FROM gate_decisions WHERE '
                                + ' OR '.join(f'{c} IS NOT NULL' for c in DECISION))).scalar_one()
    if marks or kept:
        raise RuntimeError(f'{kept} gate decisions and {marks} fill marks hold leverage figures: not dropped. Ask the '
                           'Data Architect and the HoE')
    op.drop_table('gate_fill_marks')
    if pg:
        op.execute('DROP FUNCTION IF EXISTS gate_fill_mark_write_once()')
        op.execute('DROP TRIGGER IF EXISTS gate_decisions_leverage ON gate_decisions')
        op.execute('DROP FUNCTION IF EXISTS gate_leverage_shape()')
    else:
        for o in ('insert', 'update'):
            op.execute(f'DROP TRIGGER IF EXISTS gate_decisions_leverage_{o}')
    for name in reversed(DECISION):
        op.drop_column('gate_decisions', name)  # plain DROP COLUMN on SQLite (3.35+): no foreign key on these
