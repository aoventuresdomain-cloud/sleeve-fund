"""Drop B (DA, 15 Oct; PE1 multi-asset-spec, Advisor MA-SPEC): multi-asset Strategies keyed by (Strategy, asset).

sleeves: a multi-asset Strategy's legs are rows of their own, "<name>~<ASSET>" (e.g. trend-x~ETH), many under one
parent; the two-leg "<name>~2" still fits. CHECK sleeves_leg_name is widened from "~2" to a suffix of 1-16 upper-case
letters or digits, one level (a parent holds no "~"). A unique index on (COALESCE(parent, name), instrument) keeps one
row per (Strategy, asset): no multi-asset leg trades its parent's instrument or another leg's. Two-leg "~2" rows are
left out of it: a carry trades one symbol on spot and perp. Positions, P&L and reconcile read
that key from the rows (Store.strategy_assets, asset_book, asset_mismatches); no positions table.

gate_decisions: planned_stop, planned_target (the idea's prices, for a refusal's replay), would_be (the bound test's
measure at the asked size, a share of book_equity) and side (+1 or -1), nullable; a trigger holds their shape, so
gate_decisions is not rebuilt on SQLite.

shadow_outcomes: what a trimmed or refused idea would have made, one per decision, write-once; a trigger checks its
kind matches the decision's outcome.

Additive only. Each part is skipped when a Store on the same code built it first. Before it no row can break the new
index: every leg is a "~2" row, which the index leaves out, and names are unique.

Revision ID: 0028
Revises: 0027
Create Date: 2026-10-15 00:30:00.000000
"""

from alembic import op
import sqlalchemy as sa

EXACT = sa.Numeric(38, 18)
TS = sa.DateTime(timezone=True)
BIG_ID = sa.BigInteger().with_variant(sa.Integer(), 'sqlite')
# Written out, not imported: a migration is frozen at what it was when it ran
OLD_LEG_NAME = ("(parent IS NULL AND name NOT LIKE '%~%') "
                "OR (parent IS NOT NULL AND name = parent || '~2' AND parent NOT LIKE '%~%')")
_HEAD = ("(parent IS NULL AND name NOT LIKE '%~%') "
         "OR (parent IS NOT NULL AND parent NOT LIKE '%~%' AND substr(name, 1, length(parent) + 1) = parent || '~' "
         "AND ")
_SUFFIX = "substr(name, length(parent) + 2)"
LEG_NAME = {
    'postgresql': _HEAD + f"{_SUFFIX} ~ '^[A-Z0-9]{{1,16}}$')",
    'sqlite': _HEAD + f"length({_SUFFIX}) BETWEEN 1 AND 16 AND {_SUFFIX} NOT GLOB '*[^A-Z0-9]*')",
}
ASSET_ROWS = "parent IS NULL OR name <> parent || '~2'"
IDEA = ('planned_stop', 'planned_target', 'would_be', 'side')
IDEA_BAD = ("(NEW.side IS NOT NULL AND NEW.side NOT IN (-1, 1)) "
            "OR (NEW.would_be IS NOT NULL AND NEW.test_hit IS NULL) "
            "OR NEW.planned_stop <= 0 OR NEW.planned_target <= 0")
IDEA_MESSAGE = 'gate_decisions: side is +1 or -1, would_be needs test_hit, a planned stop or target is above 0'
WRONG_KIND = ("NOT EXISTS (SELECT 1 FROM gate_decisions WHERE id = NEW.decision_id AND outcome = "
              "CASE NEW.kind WHEN 'trim' THEN 'trimmed' WHEN 'refusal' THEN 'rejected' END)")
KIND_MESSAGE = 'a shadow outcome is a trim of a trimmed decision or a refusal of a rejected one'
PG = (
    "CREATE OR REPLACE FUNCTION gate_idea_shape() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    f"IF {IDEA_BAD} THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = '{IDEA_MESSAGE}'; "
    "END IF; RETURN NEW; END $$",
    "CREATE TRIGGER gate_decisions_idea BEFORE INSERT OR UPDATE ON gate_decisions FOR EACH ROW "
    "EXECUTE FUNCTION gate_idea_shape()",
)
PG_SHADOW = (
    "CREATE OR REPLACE FUNCTION shadow_outcome_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'shadow_outcomes is write-once'; END $$",
    "CREATE TRIGGER shadow_outcomes_write_once BEFORE UPDATE OR DELETE ON shadow_outcomes FOR EACH ROW "
    "EXECUTE FUNCTION shadow_outcome_write_once()",
    "CREATE OR REPLACE FUNCTION shadow_outcome_kind() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    f"IF {WRONG_KIND} THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = '{KIND_MESSAGE}'; "
    "END IF; RETURN NEW; END $$",
    "CREATE TRIGGER shadow_outcomes_kind BEFORE INSERT ON shadow_outcomes FOR EACH ROW "
    "EXECUTE FUNCTION shadow_outcome_kind()",
)
SQLITE = tuple(f"CREATE TRIGGER gate_decisions_idea_{o.lower()} BEFORE {o} ON gate_decisions WHEN {IDEA_BAD} "
               f"BEGIN SELECT RAISE(ABORT, '{IDEA_MESSAGE}'); END" for o in ('INSERT', 'UPDATE'))
SQLITE_SHADOW = (
    *(f"CREATE TRIGGER shadow_outcomes_no_{o.lower()} BEFORE {o} ON shadow_outcomes BEGIN "
      "SELECT RAISE(ABORT, 'shadow_outcomes is write-once'); END" for o in ('UPDATE', 'DELETE')),
    f"CREATE TRIGGER shadow_outcomes_kind BEFORE INSERT ON shadow_outcomes WHEN {WRONG_KIND} BEGIN "
    f"SELECT RAISE(ABORT, '{KIND_MESSAGE}'); END",
)

revision = '0028'
down_revision = '0027'
branch_labels = None
depends_on = None


def _set_leg_name(bind, sql: str) -> None:
    if bind.dialect.name == 'postgresql':
        op.drop_constraint('sleeves_leg_name', 'sleeves', type_='check')
        op.create_check_constraint('sleeves_leg_name', 'sleeves', sa.text(sql))
    else:  # SQLite changes a CHECK only by rebuilding the table; its FK and other CHECKs survive (as 0022)
        # The rebuild drops sleeves and renames the copy to it. 0026's fill_instrument triggers name sleeves, and a
        # modern rename re-checks every trigger while sleeves is gone; the legacy rename leaves them alone, and they
        # name the table by the name it gets back. sleeves itself has no trigger to lose.
        op.execute('PRAGMA legacy_alter_table = ON')
        try:
            with op.batch_alter_table('sleeves', recreate='always') as b:
                b.drop_constraint('sleeves_leg_name', type_='check')
                b.create_check_constraint('sleeves_leg_name', sa.text(sql))
        finally:
            op.execute('PRAGMA legacy_alter_table = OFF')


def upgrade() -> None:
    bind = op.get_bind()
    pg = bind.dialect.name == 'postgresql'
    have = sa.inspect(bind)
    # The old CHECK names the fixed "~2"; the widened one doesn't. (SQLite can't reflect the expression index, so the
    # index is made IF NOT EXISTS rather than looked up.)
    if any(c['name'] == 'sleeves_leg_name' and "'~2'" in c['sqltext'] for c in have.get_check_constraints('sleeves')):
        _set_leg_name(bind, LEG_NAME[bind.dialect.name])
    op.execute('CREATE UNIQUE INDEX IF NOT EXISTS sleeves_strategy_asset ON sleeves (coalesce(parent, name), '
               f'instrument) WHERE {ASSET_ROWS}')
    cols = {c['name'] for c in have.get_columns('gate_decisions')}
    if 'side' not in cols:
        for name in IDEA:
            op.add_column('gate_decisions', sa.Column(name, sa.SmallInteger() if name == 'side' else EXACT,
                                                      nullable=True))
        for sql in PG if pg else SQLITE:
            op.execute(sql)
    if 'shadow_outcomes' not in have.get_table_names():
        op.create_table(
            'shadow_outcomes',
            sa.Column('decision_id', BIG_ID, nullable=False, autoincrement=False),
            sa.Column('kind', sa.String(length=8), nullable=False),
            sa.Column('pnl', EXACT, nullable=False),
            sa.Column('hypothetical', sa.Boolean(), nullable=False),
            sa.Column('method', sa.String(length=16), nullable=False),
            sa.Column('computed_at', TS, nullable=False),
            sa.ForeignKeyConstraint(['decision_id'], ['gate_decisions.id'], ondelete='RESTRICT'),
            sa.PrimaryKeyConstraint('decision_id'),
            sa.CheckConstraint("kind IN ('trim', 'refusal')", name='shadow_outcomes_kind'),
            sa.CheckConstraint("method IN ('trade_scaled', 'replay')", name='shadow_outcomes_method'),
            sa.CheckConstraint("(kind = 'refusal' AND hypothetical AND method = 'replay') "
                               "OR (kind = 'trim' AND NOT hypothetical AND method = 'trade_scaled')",
                               name='shadow_outcomes_shape'),
        )
        for sql in PG_SHADOW if pg else SQLITE_SHADOW:
            op.execute(sql)


def downgrade() -> None:
    bind = op.get_bind()
    pg = bind.dialect.name == 'postgresql'
    shadows = bind.execute(sa.text('SELECT count(*) FROM shadow_outcomes')).scalar()
    ideas = bind.execute(sa.text('SELECT count(*) FROM gate_decisions WHERE '
                                 + ' OR '.join(f'{c} IS NOT NULL' for c in IDEA))).scalar()
    legs = bind.execute(sa.text("SELECT count(*) FROM sleeves WHERE parent IS NOT NULL "
                                "AND name <> parent || '~2'")).scalar()
    if shadows or ideas or legs:
        raise NotImplementedError('multi-asset legs, shadow outcomes or idea columns hold rows: they are never dropped')
    op.drop_table('shadow_outcomes')
    if pg:
        op.execute('DROP FUNCTION IF EXISTS shadow_outcome_write_once()')
        op.execute('DROP FUNCTION IF EXISTS shadow_outcome_kind()')
        op.execute('DROP TRIGGER IF EXISTS gate_decisions_idea ON gate_decisions')
        op.execute('DROP FUNCTION IF EXISTS gate_idea_shape()')
    else:
        for o in ('insert', 'update'):
            op.execute(f'DROP TRIGGER IF EXISTS gate_decisions_idea_{o}')
    for name in reversed(IDEA):
        op.drop_column('gate_decisions', name)  # plain DROP COLUMN on SQLite (3.35+): no foreign key on these
    op.execute('DROP INDEX IF EXISTS sleeves_strategy_asset')
    _set_leg_name(bind, OLD_LEG_NAME)
