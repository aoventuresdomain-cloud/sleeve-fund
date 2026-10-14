"""orders.venue (P2-4 router v1): the venue each order was sent to, the strategy's one venue. Stamped by
store.record_order from router.venue_for, never by the caller; upper case, held by CHECK orders_venue_upper.

Backfill: every order already journaled gets its strategy's venue exactly as paper/config resolved it when the order
was sent: its sleeve_venues row; else, for a saved backtest's strategy (save_backtest wrote no pin), the venue its
backtests row was run on, read from the run's query as the dashboard read it (F-WED1-1); else the default venue of the
day. The default is spelled here as a literal (DA), so
this migration's result never moves if the code's default does. Then the column is made NOT NULL.

Each step is skipped when a Store on the same code built that shape first (create_all), as 0010 allows. SQLite adds the
column, its CHECK and NOT NULL through a batch rebuild. Last, the pin is made insert-only (F-WED1-3): a venue is never
changed, a pin is deleted only with its saved backtest, and a strategy that has traded can't be given one. Downgrade is
refused: an order's venue is never dropped.

Revision ID: 0020
Revises: 0018
Create Date: 2026-10-13 19:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0020'
down_revision = '0018'
branch_labels = None
depends_on = None

# The default venue when this migration was written (sleeve_fund.venues.DEFAULT_VENUE on 13 Oct 2026): the venue of
# every strategy saved before strategies carried their own.
DEFAULT_VENUE_THEN = 'KRAKEN'
# Every venue with a profile on 13 Oct 2026: the dashboard ran a backtest on the one its query named (any case), and on
# the default when the name was blank or had no profile (dashboard _venue_name).
VENUES_THEN = ('KRAKEN', 'BINANCE')
UPPER = 'venue = upper(venue)'

# F-WED1-3: written out, not imported, as 0018's are: a migration is frozen at what it was when it ran
PIN_PG = ("CREATE OR REPLACE FUNCTION sleeve_venue_pinned() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF TG_OP = 'UPDATE' THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'a strategy''s venue is pinned: ' || OLD.sleeve || ' cannot change venue'; END IF; IF TG_OP = 'DELETE' THEN IF substr(OLD.sleeve, 1, 3) <> 'bt:' OR NOT EXISTS (SELECT 1 FROM backtests WHERE sleeve = OLD.sleeve) THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'a strategy''s venue is pinned: ' || OLD.sleeve || ' is not a saved backtest, so its pin stays'; END IF; RETURN OLD; END IF; IF EXISTS (SELECT 1 FROM orders WHERE sleeve = NEW.sleeve) OR EXISTS (SELECT 1 FROM fills WHERE sleeve = NEW.sleeve) THEN RAISE EXCEPTION USING ERRCODE = 'check_violation', MESSAGE = 'a strategy''s venue is pinned: ' || NEW.sleeve || ' has traded, so it keeps its venue'; END IF; RETURN NEW; END $$", 'CREATE TRIGGER sleeve_venue_pinned BEFORE INSERT OR UPDATE OR DELETE ON sleeve_venues FOR EACH ROW EXECUTE FUNCTION sleeve_venue_pinned()')
PIN_SQLITE = ("CREATE TRIGGER sleeve_venues_no_edit BEFORE UPDATE ON sleeve_venues BEGIN SELECT RAISE(ABORT, 'a strategy''s venue is pinned: a venue cannot change'); END", "CREATE TRIGGER sleeve_venues_no_delete BEFORE DELETE ON sleeve_venues WHEN substr(OLD.sleeve, 1, 3) <> 'bt:' OR NOT EXISTS (SELECT 1 FROM backtests WHERE sleeve = OLD.sleeve) BEGIN SELECT RAISE(ABORT, 'a strategy''s venue is pinned: only a saved backtest''s pin is deleted'); END", "CREATE TRIGGER sleeve_venues_no_late_pin BEFORE INSERT ON sleeve_venues WHEN EXISTS (SELECT 1 FROM orders WHERE sleeve = NEW.sleeve) OR EXISTS (SELECT 1 FROM fills WHERE sleeve = NEW.sleeve) BEGIN SELECT RAISE(ABORT, 'a strategy''s venue is pinned: a strategy that has traded keeps its venue'); END")


def upgrade() -> None:
    bind = op.get_bind()
    have = sa.inspect(bind)
    if 'venue' not in {c['name'] for c in have.get_columns('orders')}:
        with op.batch_alter_table('orders', schema=None) as batch_op:
            batch_op.add_column(sa.Column('venue', sa.String(length=16), nullable=True))
    orders = sa.table('orders', sa.column('sleeve', sa.String), sa.column('venue', sa.String))
    pins = sa.table('sleeve_venues', sa.column('sleeve', sa.String), sa.column('venue', sa.String))
    _backfill_backtests(bind, orders, pins)
    pinned = (sa.select(sa.func.upper(pins.c.venue)).where(pins.c.sleeve == orders.c.sleeve).scalar_subquery())
    bind.execute(sa.update(orders).where(orders.c.venue.is_(None))
                 .values(venue=sa.func.coalesce(pinned, DEFAULT_VENUE_THEN)))
    have = sa.inspect(bind)
    nullable = next(c['nullable'] for c in have.get_columns('orders') if c['name'] == 'venue')
    checked = 'orders_venue_upper' in {c['name'] for c in have.get_check_constraints('orders')}
    if nullable or not checked:
        with op.batch_alter_table('orders', schema=None) as batch_op:
            if nullable:
                batch_op.alter_column('venue', existing_type=sa.String(length=16), nullable=False)
            if not checked:
                batch_op.create_check_constraint('orders_venue_upper', UPPER)
    _pin_triggers(bind)


def _backtest_venue(query: str | None) -> str:
    """The venue a saved backtest ran on, from its query's venue param, as the dashboard resolved it then."""
    from urllib.parse import parse_qs

    name = (parse_qs(query or '').get('venue') or [''])[0].strip().upper()
    return name if name in VENUES_THEN else DEFAULT_VENUE_THEN


def _backfill_backtests(bind, orders, pins) -> None:
    """F-WED1-1: a strategy with no pin row whose backtests rows all ran on one venue gets that venue; one whose rows
    disagree is left to the default below (no single venue to name)."""
    backtests = sa.table('backtests', sa.column('sleeve', sa.String), sa.column('query', sa.Text))
    unpinned = bind.execute(sa.select(orders.c.sleeve).distinct().where(
        orders.c.venue.is_(None), orders.c.sleeve.not_in(sa.select(pins.c.sleeve)))).scalars().all()
    for sleeve in unpinned:
        ran_on = {_backtest_venue(q) for q in bind.execute(
            sa.select(backtests.c.query).where(backtests.c.sleeve == sleeve)).scalars()}
        if len(ran_on) == 1:
            bind.execute(sa.update(orders).where(orders.c.sleeve == sleeve, orders.c.venue.is_(None))
                         .values(venue=ran_on.pop()))


def _pin_triggers(bind) -> None:
    """F-WED1-3 (Advisor, HoE 14 Oct): the pin is insert-only. Skipped when a Store on the same code built them."""
    if bind.dialect.name == 'postgresql':
        have = bind.execute(sa.text("SELECT 1 FROM pg_trigger WHERE tgname = 'sleeve_venue_pinned' "
                                    "AND tgrelid = 'sleeve_venues'::regclass")).first()
        statements = PIN_PG
    else:
        have = bind.execute(sa.text("SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
                                    "AND name = 'sleeve_venues_no_edit'")).first()
        statements = PIN_SQLITE
    if not have:
        for sql in statements:
            op.execute(sql)


def downgrade() -> None:
    raise NotImplementedError('orders.venue records where each order was sent: never dropped')
