#!/usr/bin/env bash
# DA-7, the restore drill (restore-drill.yml runs it on the server, from /opt/sleeve-fund, as root): the newest nightly
# dump (deploy/backup.sh) restored into a throwaway Postgres on a network and volume of its own (internal: nothing
# published on the host, the journal's database never reached), then checked by python -m sleeve_fund.restore_drill
# in the supervisor's image, with the backups mounted read only. Everything made here is removed at the end, on
# failure too. The log is public: only step names, counts, timings and PASS or FAIL are printed; the tools' own output
# (which can carry names, paths or rows) goes nowhere. Exit 0 when the drill passes, else 1.
set -uo pipefail
DRILL=${DRILL_ID:-restore-drill-$(date -u +%Y%m%d%H%M%S)}
NET=$DRILL VOL=$DRILL PG=$DRILL-db
cleanup() {
  docker rm -f "$PG" >/dev/null 2>&1
  docker network rm "$NET" >/dev/null 2>&1
  docker volume rm "$VOL" >/dev/null 2>&1
  echo "cleanup: done"
}
trap cleanup EXIT
fail() { echo "$1: FAIL"; echo "restore drill: FAIL"; exit 1; }
now() { date +%s.%N; }
since() { awk -v a="$1" -v b="$(now)" 'BEGIN { printf "%.1f", b - a }'; }

t=$(now)
mount_of() {  # the named volume a running service mounts at $2
  local id; id=$(docker compose ps -q "$1" 2>/dev/null) && [ -n "$id" ] || return 1
  docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$2\"}}{{.Name}}{{end}}{{end}}" "$id" 2>/dev/null
}
image_of() {
  local id; id=$(docker compose ps -q "$1" 2>/dev/null) && [ -n "$id" ] || return 1
  docker inspect -f '{{.Config.Image}}' "$id" 2>/dev/null
}
BACKUPS=$(mount_of backup /backups) && [ -n "$BACKUPS" ] || fail "find the backups"
PG_IMAGE=$(image_of db) && [ -n "$PG_IMAGE" ] || fail "find the database image"
APP_IMAGE=$(image_of supervisor) && [ -n "$APP_IMAGE" ] || fail "find the app image"
docker network create --internal "$NET" >/dev/null 2>&1 || fail "make the drill's network"
docker volume create "$VOL" >/dev/null 2>&1 || fail "make the drill's volume"
docker run -d --name "$PG" --network "$NET" -v "$VOL:/var/lib/postgresql/data" -v "$BACKUPS:/backups:ro" \
  -e POSTGRES_HOST_AUTH_METHOD=trust -e POSTGRES_DB=drill "$PG_IMAGE" >/dev/null 2>&1 || fail "start the drill's database"
for _ in $(seq 1 60); do
  docker exec "$PG" pg_isready -U postgres -d drill >/dev/null 2>&1 && break
  sleep 1
done
docker exec "$PG" pg_isready -U postgres -d drill >/dev/null 2>&1 || fail "start the drill's database"
echo "setup: PASS ($(since "$t")s), own network and volume, nothing published"

t=$(now)
DUMP=$(docker exec "$PG" sh -c 'ls -1 /backups/sleeve_fund-*.dump 2>/dev/null | sort | tail -n 1')
[ -n "$DUMP" ] || fail "restore"
# The owner and grants are the journal's role, which the drill's database hasn't: the rows are what is proved.
docker exec "$PG" pg_restore -U postgres -d drill --no-owner --no-privileges --exit-on-error "$DUMP" >/dev/null 2>&1 \
  || fail "restore"
RESTORE=$(since "$t")

docker run --rm --network "$NET" -v "$BACKUPS:/backups:ro" \
  -e DATABASE_URL="postgresql+psycopg://postgres@$PG:5432/drill" "$APP_IMAGE" \
  python -m sleeve_fund.restore_drill --backups /backups --restore-seconds "$RESTORE" 2>/dev/null \
  || { echo "checks: exit $?"; exit 1; }
