#!/usr/bin/env bash
# Runs INSIDE the backup container, piped in by remote_up.sh before a deploy starts the new code (Advisor's pre-merge
# condition, 15 Oct): a dump of the journal as the running code left it, proved by restoring it into a scratch database
# on the same server (as backup.sh's nightly check does), so a deploy that goes wrong has a known-good copy from minutes
# before it. Kept as predeploy-<UTC stamp>.pgdump, the newest 3: not *.dump, so backup.sh's 14-dump rotation and the
# restore drill never count or remove one. Any failure exits non-zero, and remote_up.sh (set -e) stops the deploy
# before the new code starts. The log is the deploy's (public): the migration and row counts only.
set -euo pipefail
umask 077  # docker exec runs with umask 0000: the copy is readable by its owner only (QT3 N1)
DIR=${BACKUP_DIR:-/backups}
DB=${BACKUP_DB:-sleeve_fund}
CHECK=${PREDEPLOY_CHECK_DB:-predeploy_check}
export PGHOST=${PGHOST:-db} PGUSER=${PGUSER:-sleeve}  # as backup.sh: the container holds only the password
# The scratch database is dropped on every exit, so its name can only be its own: never the journal or another database.
[[ "$CHECK" =~ ^predeploy_[a-z0-9_]+$ ]] && [ "$CHECK" != "$DB" ] \
  || { echo "pre-deploy dump: FAIL, not a scratch database name"; exit 1; }
f="$DIR/predeploy-$(date -u +%Y%m%d-%H%M%S).pgdump"
# Until the copy is proved it keeps its .part name, removed on any exit: an unproved copy never stays under a good one.
trap 'rm -f "$f.part"; dropdb --if-exists "$CHECK" >/dev/null 2>&1 || true' EXIT
# Twice the newest dump (nightly or pre-deploy) free beside them, or nothing is dumped (HoE).
last=$(ls -1t "$DIR"/*.dump "$DIR"/predeploy-*.pgdump 2>/dev/null | head -n 1 || true)
size=$(if [ -n "$last" ]; then stat -c %s "$last"; else echo 0; fi)
free=$(df -PB1 "$DIR" | awk 'NR == 2 { print $4 }')
[[ "$free" =~ ^[0-9]+$ ]] && [ "$free" -ge $((2 * size)) ] || { echo "pre-deploy dump: FAIL, not enough disk"; exit 1; }
# The journal's own rows (backtests excluded: the dashboard may prune those meanwhile), counted before and after the
# dump: the running code only adds them, so the copy must hold at least the first count and at most the second.
COUNT="select (select version_num from alembic_version) || ' ' ||
  (select count(*) from orders where sleeve not like 'bt:%') || ' ' ||
  (select count(*) from fills where sleeve not like 'bt:%') || ' ' ||
  (select count(*) from events where sleeve is null or sleeve not like 'bt:%') || ' ' ||
  (select count(*) from fund_flows) || ' ' || (select count(*) from book_moves) || ' ' || (select count(*) from decisions)"
NAMES="migration orders fills events fund_flows book_moves decisions"
# Each tool's own words can name a host or a row: they go nowhere, and a failure says which step (QT3 N2).
before=$(psql -d "$DB" -At -c "$COUNT" 2>/dev/null) || { echo "pre-deploy dump: FAIL, couldn't read the journal"; exit 1; }
pg_dump -d "$DB" -Fc -f "$f.part" 2>/dev/null || { echo "pre-deploy dump: FAIL, the dump didn't complete"; exit 1; }
after=$(psql -d "$DB" -At -c "$COUNT" 2>/dev/null) || { echo "pre-deploy dump: FAIL, couldn't read the journal"; exit 1; }
dropdb --if-exists "$CHECK" >/dev/null 2>&1 || true
createdb "$CHECK" 2>/dev/null || { echo "pre-deploy dump: FAIL, couldn't make the scratch database"; exit 1; }
pg_restore --no-owner --no-privileges --exit-on-error -d "$CHECK" "$f.part" 2>/dev/null \
  || { echo "pre-deploy dump: FAIL, it didn't restore"; exit 1; }
copy=$(psql -d "$CHECK" -At -c "$COUNT" 2>/dev/null) || { echo "pre-deploy dump: FAIL, couldn't read the copy"; exit 1; }
dropdb "$CHECK"
read -r -a b <<< "$before"; read -r -a a <<< "$after"; read -r -a c <<< "$copy"; read -r -a n <<< "$NAMES"
[ "${c[0]}" = "${b[0]}" ] && [ "${c[0]}" = "${a[0]}" ] \
  || { echo "pre-deploy dump: FAIL, the copy is at another migration"; exit 1; }
words="restored at migration ${c[0]}"
for i in 1 2 3 4 5 6; do
  { [ "${c[$i]}" -ge "${b[$i]}" ] && [ "${c[$i]}" -le "${a[$i]}" ]; } \
    || { echo "pre-deploy dump: FAIL, ${n[$i]} doesn't match the journal"; exit 1; }
  words="$words, ${n[$i]} ${c[$i]}"
done
mv "$f.part" "$f"
ls -1t "$DIR"/predeploy-*.pgdump | tail -n +4 | xargs -r rm -f
echo "pre-deploy dump: PASS, $words"
