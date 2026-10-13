"""events.command_id (F217-3): the command a liquidation_reset carries out, unique where set, so a reset after
liquidation is journaled once per command and the process's retry after a crash reads whether its own reset was
journaled from this key instead of the message's wording.

One-off backfill (DA): a reset after liquidation already journaled before this migration, under the old wording, gets
the command it answered, matched as the process used to match it ("answering incident #N (" against
commands.incident, the newest such reset per command). So a command still pending across the deploy finishes without
a second reset; the wording key lives here and nowhere else. Every other row keeps NULL.

Each step is skipped when a Store on the same code built that shape first (create_all), as 0010 allows. SQLite adds the
column and its foreign key through a batch rebuild.

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-08 04:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0014'
down_revision = '0013'
branch_labels = None
depends_on = None

SET = 'command_id IS NOT NULL'


def upgrade() -> None:
    have = sa.inspect(op.get_bind())
    if 'command_id' not in {c['name'] for c in have.get_columns('events')}:
        with op.batch_alter_table('events', schema=None) as batch_op:
            batch_op.add_column(sa.Column('command_id', sa.Integer(), nullable=True))
            batch_op.create_foreign_key('events_command_id_fkey', 'commands', ['command_id'], ['id'],
                                        ondelete='RESTRICT')
    if 'events_command_once' not in {i['name'] for i in sa.inspect(op.get_bind()).get_indexes('events')}:
        op.create_index('events_command_once', 'events', ['command_id'], unique=True,
                        sqlite_where=sa.text(SET), postgresql_where=sa.text(SET))
    _backfill(op.get_bind())


def _backfill(bind) -> None:
    events = sa.table('events', sa.column('id', sa.Integer), sa.column('sleeve', sa.String), sa.column('kind', sa.String),
                      sa.column('message', sa.Text), sa.column('command_id', sa.Integer))
    commands = sa.table('commands', sa.column('id', sa.Integer), sa.column('sleeve', sa.String),
                        sa.column('incident', sa.Integer))
    taken = {r[0] for r in bind.execute(sa.select(events.c.command_id).where(events.c.command_id.is_not(None)))}
    for cid, sleeve, incident in bind.execute(sa.select(commands.c.id, commands.c.sleeve, commands.c.incident)
                                              .where(commands.c.incident.is_not(None))).all():
        if cid in taken:
            continue
        newest = bind.execute(sa.select(events.c.id).where(
            events.c.sleeve == sleeve, events.c.kind == 'liquidation_reset', events.c.command_id.is_(None),
            events.c.message.contains(f'answering incident #{incident} (')).order_by(events.c.id.desc())
            .limit(1)).scalar()
        if newest is not None:
            bind.execute(sa.update(events).where(events.c.id == newest).values(command_id=cid))


def downgrade() -> None:
    raise NotImplementedError('events.command_id records which command each reset carried out: never dropped')
