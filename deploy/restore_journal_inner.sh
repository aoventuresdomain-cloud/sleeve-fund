#!/usr/bin/env bash
# Runs INSIDE the backup container, piped in by deploy/restore_journal.sh (restore-journal.yml, by hand, only on the
# PM's go): one proved dump beside the backups put back as the journal. The dump is restored into a stage database and
# counted first; only then is the journal renamed aside, kept whole as <journal>_before_restore_<UTC stamp> and never
# dropped here, and the stage renamed into its place. Anything wrong before that leaves the journal exactly as it was.
# The log is public: the step, the migration and row counts only; the tools' own words go to restore-last.err beside
# the dumps, readable by its owner only. Exit 0 when the journal is the dump's, else 1.
set -euo pipefail
umask 077
DIR=${BACKUP_DIR:-/backups}
DB=${BACKUP_DB:-sleeve_fund}
STAGE=${RESTORE_STAGE_DB:-restore_stage}
DUMP=${DUMP:-}
export PGHOST=${PGHOST:-db} PGUSER=${PGUSER:-sleeve}  # as backup.sh: the container holds only the password
fail() { echo "restore journal: FAIL, $1"; exit 1; }
# The stage is dropped on every exit before the swap, so its name can only be its own.
[[ "$STAGE" =~ ^restore_[a-z0-9_]+$ ]] && [ "$STAGE" != "$DB" ] || fail "not a stage database name"
# Only a dump this server made, by its file name beside the backups: no path, nothing else.
[[ "$DUMP" =~ ^(predeploy-[0-9]{8}-[0-9]{6}\.pgdump|sleeve_fund-[0-9A-Za-z_-]+\.dump)$ ]] || fail "not a dump's name"
[ -f "$DIR/$DUMP" ] || fail "no such dump beside the backups"
ERR="$DIR/restore-last.err"
: > "$ERR"
swapped=0
trap '[ "$swapped" = 1 ] || dropdb --if-exists "$STAGE" >/dev/null 2>&1 || true' EXIT
COUNT="select (select version_num from alembic_version) || ' ' ||
  (select count(*) from orders where sleeve not like 'bt:%') || ' ' ||
  (select count(*) from fills where sleeve not like 'bt:%') || ' ' ||
  (select count(*) from events where sleeve is null or sleeve not like 'bt:%') || ' ' ||
  (select count(*) from fund_flows) || ' ' || (select count(*) from book_moves) || ' ' || (select count(*) from decisions)"
NAMES="migration orders fills events fund_flows book_moves decisions"
dropdb --if-exists "$STAGE" >/dev/null 2>&1 || true
createdb "$STAGE" 2>>"$ERR" || { rc=$?; fail "couldn't make the stage database (exit $rc), details beside the dumps"; }
pg_restore --no-owner --no-privileges --exit-on-error -d "$STAGE" "$DIR/$DUMP" 2>>"$ERR" \
  || { rc=$?; fail "it didn't restore (exit $rc), details beside the dumps"; }
copy=$(psql -d "$STAGE" -At -c "$COUNT" 2>>"$ERR") || { rc=$?; fail "couldn't read the restored copy (exit $rc), details beside the dumps"; }
read -r -a c <<< "$copy"; read -r -a n <<< "$NAMES"
[ "${#c[@]}" = 7 ] || fail "couldn't read the restored copy"
words="$DUMP restored at migration ${c[0]}"
for i in 1 2 3 4 5 6; do words="$words, ${n[$i]} ${c[$i]}"; done
# The app is stopped by the host side first; a session still on the journal means something wasn't, so no swap.
busy=$(psql -d postgres -At -c "select count(*) from pg_stat_activity where datname = '$DB' and pid <> pg_backend_pid()" \
  2>>"$ERR") || { rc=$?; fail "couldn't check the journal is free (exit $rc), details beside the dumps"; }
[ "$busy" = 0 ] || fail "the journal is still in use ($busy sessions)"
KEEP="${DB}_before_restore_$(date -u +%Y%m%d%H%M%S)"
psql -d postgres -v ON_ERROR_STOP=1 -c "ALTER DATABASE \"$DB\" RENAME TO \"$KEEP\"" >/dev/null 2>>"$ERR" \
  || { rc=$?; fail "couldn't set the journal aside (exit $rc), it is as it was"; }
if ! psql -d postgres -v ON_ERROR_STOP=1 -c "ALTER DATABASE \"$STAGE\" RENAME TO \"$DB\"" >/dev/null 2>>"$ERR"; then
  rc=$?
  psql -d postgres -v ON_ERROR_STOP=1 -c "ALTER DATABASE \"$KEEP\" RENAME TO \"$DB\"" >/dev/null 2>>"$ERR" \
    || fail "the restored copy couldn't take the journal's place (exit $rc) and the journal is still named $KEEP"
  fail "the restored copy couldn't take the journal's place (exit $rc), the journal is back as it was"
fi
swapped=1
echo "restore journal: PASS, $words; the journal it replaced is kept as $KEEP"
