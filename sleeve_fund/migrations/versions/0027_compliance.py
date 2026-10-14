"""Bridge A (DA, 14 Oct; Advisor BRIDGE-3, PM DEFAULTS-2): the PM's compliance tests, versioned whole.

compliance_versions, compliance_tests and compliance_asset_types: one version is every test and the asset-group map
together, written once (Store.add_compliance_version) and never edited or deleted (write-once triggers). Version 1 is
the PM's defaults, every test on: per asset 4%, per asset group 6%, whole fund 10% (loss if every stop is hit, trim);
gross 1.5x, net per asset 0.5x, margin 50% (trim); daily loss 3% (pause new entries to 00:00 UTC); drawdown 15%
(flatten and halt). Map: BTC, ETH large; SOL, XRP mid; every other underlying in instruments small.

gate_decisions, book_marks and portfolio_state get compliance_version (nullable: rows before it have the profile only;
portfolio_state's row is set to 1), and gate_decisions gets test_hit, which a trigger checks names a test of the
recorded version. profile_version stays as it is until the gate stops writing it.

Additive only. Each part is skipped when a Store on the same code built it first.

Revision ID: 0027
Revises: 0026
Create Date: 2026-10-14 22:30:00.000000
"""

from datetime import datetime, timezone
from decimal import Decimal

from alembic import op
import sqlalchemy as sa

TS = sa.DateTime(timezone=True)
LIMIT = sa.Numeric(10, 4)
# Written out, not imported: a migration is frozen at what it was when it ran
V1_AT = datetime(2026, 10, 14, 21, 0, tzinfo=timezone.utc)
V1_NOTE = "the PM's defaults (DEFAULTS-2, 14 Oct 2026); Advisor BRIDGE-3"
V1 = (  # test, label, scope, measure, limit, action
    ('asset_stop_loss', 'Per asset: loss if every stop is hit', 'asset', 'open_risk', '0.04', 'trim'),
    ('type_stop_loss', 'Per asset group: loss if every stop is hit', 'asset_type', 'open_risk', '0.06', 'trim'),
    ('fund_stop_loss', 'Whole fund: loss if every stop is hit', 'fund', 'open_risk', '0.10', 'trim'),
    ('fund_gross', "Gross: every position's size added up", 'fund', 'gross', '1.5', 'trim'),
    ('asset_net', 'Net per asset: long less short on one asset', 'asset', 'net', '0.5', 'trim'),
    ('fund_margin', 'Margin posted', 'fund', 'margin', '0.5', 'trim'),
    ('fund_day_loss', "Today's loss pauses new trades until the next day begins", 'fund', 'day_loss', '0.03', 'pause_day'),
    ('fund_drawdown', 'Drawdown from the peak closes everything and halts', 'fund', 'drawdown', '0.15', 'halt'),
)
GROUPS = {'BTC': 'large', 'ETH': 'large', 'SOL': 'mid', 'XRP': 'mid'}
OTHER = 'small'
SCOPES = "scope IN ('fund', 'asset', 'asset_type', 'strategy')"
MEASURES = ("measure IN ('open_risk', 'gross', 'net', 'margin', 'day_loss', 'drawdown', 'strategy_risk', "
            "'contrib_dd')")
ACTIONS = "action IN ('refuse', 'trim', 'pause_day', 'pause_pm', 'halt')"
TABLES = ('compliance_versions', 'compliance_tests', 'compliance_asset_types')
WORM_PG = ("CREATE OR REPLACE FUNCTION compliance_write_once() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
           "RAISE EXCEPTION USING ERRCODE = 'check_violation', "
           "MESSAGE = TG_TABLE_NAME || ' is write-once: a version is never edited or deleted, a change is a new "
           "version'; END $$")
HIT_PG = (
    "CREATE OR REPLACE FUNCTION gate_test_hit_known() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    "IF NEW.test_hit IS NOT NULL AND NOT EXISTS (SELECT 1 FROM compliance_tests WHERE version = NEW.compliance_version "
    "AND test = NEW.test_hit) THEN RAISE EXCEPTION USING ERRCODE = 'foreign_key_violation', "
    "MESSAGE = 'gate_decisions.test_hit ' || NEW.test_hit || ' is not a test of version ' "
    "|| COALESCE(NEW.compliance_version::text, 'none'); END IF; RETURN NEW; END $$",
    "CREATE TRIGGER gate_decisions_test_hit BEFORE INSERT OR UPDATE ON gate_decisions FOR EACH ROW "
    "EXECUTE FUNCTION gate_test_hit_known()",
)
HIT_SQLITE = tuple(
    f"CREATE TRIGGER gate_decisions_test_hit_{o.lower()} BEFORE {o} ON gate_decisions WHEN NEW.test_hit IS NOT NULL "
    "AND NOT EXISTS (SELECT 1 FROM compliance_tests WHERE version = NEW.compliance_version AND test = NEW.test_hit) "
    "BEGIN SELECT RAISE(ABORT, 'gate_decisions.test_hit is not a test of the version it records'); END"
    for o in ('INSERT', 'UPDATE'))


def _worm(table, pg):
    if pg:
        return (f"CREATE TRIGGER {table}_write_once BEFORE UPDATE OR DELETE ON {table} FOR EACH ROW "
                "EXECUTE FUNCTION compliance_write_once()",)
    return tuple(f"CREATE TRIGGER {table}_no_{o.lower()} BEFORE {o} ON {table} BEGIN SELECT RAISE(ABORT, "
                 f"'{table} is write-once: a version is never edited or deleted, a change is a new version'); END"
                 for o in ('UPDATE', 'DELETE'))


revision = '0027'
down_revision = '0026'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    pg = bind.dialect.name == 'postgresql'
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())
    if pg:
        op.execute(WORM_PG)
    if 'compliance_versions' not in tables:
        op.create_table(
            'compliance_versions',
            sa.Column('version', sa.Integer(), primary_key=True, autoincrement=False),
            sa.Column('created_at', TS, nullable=False),
            sa.Column('created_by', sa.Text(), nullable=False),
            sa.Column('note', sa.Text()),
            sa.CheckConstraint("created_by <> 'pm' OR (note IS NOT NULL AND note <> '')",
                               name='compliance_versions_pm_note'),
        )
        for sql in _worm('compliance_versions', pg):
            op.execute(sql)
    if 'compliance_tests' not in tables:
        op.create_table(
            'compliance_tests',
            sa.Column('version', sa.Integer(), sa.ForeignKey('compliance_versions.version'), primary_key=True),
            sa.Column('test', sa.String(32), primary_key=True),
            sa.Column('label', sa.Text(), nullable=False),
            sa.Column('scope', sa.String(12), nullable=False),
            sa.Column('measure', sa.String(16), nullable=False),
            sa.Column('limit_value', LIMIT, nullable=False),
            sa.Column('action', sa.String(12), nullable=False),
            sa.Column('enabled', sa.Boolean(), nullable=False),
            sa.CheckConstraint(SCOPES, name='compliance_tests_scope'),
            sa.CheckConstraint(MEASURES, name='compliance_tests_measure'),
            sa.CheckConstraint(ACTIONS, name='compliance_tests_action'),
            sa.CheckConstraint('limit_value > 0', name='compliance_tests_limit'),
        )
        for sql in _worm('compliance_tests', pg):
            op.execute(sql)
    if 'compliance_asset_types' not in tables:
        op.create_table(
            'compliance_asset_types',
            sa.Column('version', sa.Integer(), sa.ForeignKey('compliance_versions.version'), primary_key=True),
            sa.Column('underlying', sa.String(16), primary_key=True),
            sa.Column('asset_type', sa.String(32), nullable=False),
            sa.CheckConstraint('underlying = upper(underlying)', name='compliance_asset_types_upper'),
            sa.CheckConstraint("asset_type <> ''", name='compliance_asset_types_named'),
        )
        for sql in _worm('compliance_asset_types', pg):
            op.execute(sql)
    if not bind.execute(sa.text('SELECT count(*) FROM compliance_versions')).scalar():
        bind.execute(sa.text("INSERT INTO compliance_versions (version, created_at, created_by, note) "
                             "VALUES (1, :at, 'migration 0027', :note)"), {'at': V1_AT, 'note': V1_NOTE})
        bind.execute(sa.text("INSERT INTO compliance_tests (version, test, label, scope, measure, limit_value, action, "
                             "enabled) VALUES (1, :t, :l, :s, :m, :v, :a, :e)")
                     .bindparams(sa.bindparam('v', type_=LIMIT), sa.bindparam('e', type_=sa.Boolean())),
                     [{'t': t, 'l': l, 's': s, 'm': m, 'v': Decimal(v), 'a': a, 'e': True} for t, l, s, m, v, a in V1])
        held = {u for (u,) in bind.execute(sa.text('SELECT DISTINCT underlying FROM instruments'))}
        groups = GROUPS | {u: OTHER for u in held if u not in GROUPS}
        bind.execute(sa.text("INSERT INTO compliance_asset_types (version, underlying, asset_type) VALUES (1, :u, :g)"),
                     [{'u': u, 'g': g} for u, g in sorted(groups.items())])
    for table in ('portfolio_state', 'book_marks', 'gate_decisions'):
        if 'compliance_version' not in {c['name'] for c in insp.get_columns(table)}:
            if pg:
                op.add_column(table, sa.Column('compliance_version', sa.Integer(),
                                               sa.ForeignKey('compliance_versions.version')))
            else:  # SQLite adds a column's foreign key only written into the column (alembic's add_column won't)
                op.execute(f'ALTER TABLE {table} ADD COLUMN compliance_version INTEGER '
                           'REFERENCES compliance_versions (version)')
    if 'test_hit' not in {c['name'] for c in insp.get_columns('gate_decisions')}:
        op.add_column('gate_decisions', sa.Column('test_hit', sa.String(32)))
        for sql in (HIT_PG if pg else HIT_SQLITE):
            op.execute(sql)
    op.execute('UPDATE portfolio_state SET compliance_version = 1 WHERE compliance_version IS NULL')


def downgrade() -> None:
    bind = op.get_bind()
    pg = bind.dialect.name == 'postgresql'
    used = bind.execute(sa.text('SELECT count(*) FROM gate_decisions WHERE compliance_version IS NOT NULL')).scalar()
    edited = bind.execute(sa.text('SELECT count(*) FROM compliance_versions WHERE version > 1')).scalar()
    if used or edited:
        raise NotImplementedError('the compliance tests have been edited or decided under: they are never dropped')
    if pg:
        op.execute('DROP TRIGGER IF EXISTS gate_decisions_test_hit ON gate_decisions')
        op.execute('DROP FUNCTION IF EXISTS gate_test_hit_known()')
    else:
        for o in ('insert', 'update'):
            op.execute(f'DROP TRIGGER IF EXISTS gate_decisions_test_hit_{o}')
    op.drop_column('gate_decisions', 'test_hit')
    for table in ('gate_decisions', 'book_marks', 'portfolio_state'):
        if pg:
            for fk in sa.inspect(bind).get_foreign_keys(table):
                if fk['constrained_columns'] == ['compliance_version']:
                    op.drop_constraint(fk['name'], table, type_='foreignkey')
            op.drop_column(table, 'compliance_version')
        else:  # SQLite drops a column with a foreign key only by rebuilding the table
            with op.batch_alter_table(table) as b:
                b.drop_column('compliance_version')
    for table in reversed(TABLES):
        op.drop_table(table)
    if pg:
        op.execute('DROP FUNCTION IF EXISTS compliance_write_once()')
