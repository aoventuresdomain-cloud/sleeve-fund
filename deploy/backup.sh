#!/usr/bin/env bash
# The backup service's loop (docker-compose.yml): a compressed dump of the journal every night,
# written whole or not at all, then proved by restoring it into a scratch database and counting
# its rows. The newest 14 dumps are kept. A failed night is tried again an hour later, not a day.
# The result goes to status.json beside the dumps, which the Operations page and the supervisor's
# alerts read. These dumps sit on this machine's disk: a copy elsewhere covers losing the machine.
set -uo pipefail
DIR=${BACKUP_DIR:-/backups}
DB=${BACKUP_DB:-sleeve_fund}
export PGHOST=${PGHOST:-db} PGUSER=${PGUSER:-sleeve}
status() {  # ok, file, message, restored counts (JSON object)
  printf '{"ts": "%s", "ok": %s, "file": "%s", "message": "%s", "restored": %s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" "${4:-null}" > "$DIR/status.json.part"
  mv "$DIR/status.json.part" "$DIR/status.json"
}
while true; do
  f="$DIR/$DB-$(date -u +%Y%m%d-%H%M).dump"
  wait=3600
  if ! pg_dump -d "$DB" -Fc -f "$f.part" 2>"$DIR/last_error.txt"; then
    rm -f "$f.part"
    status false "" "pg_dump failed: $(head -c 200 "$DIR/last_error.txt" | tr -d '"\n')"
  else
    mv "$f.part" "$f"
    dropdb --if-exists restore_check 2>/dev/null
    if createdb restore_check 2>"$DIR/last_error.txt" \
       && pg_restore --no-owner -d restore_check "$f" 2>"$DIR/last_error.txt"; then
      counts=$(psql -d restore_check -At -c "select json_build_object('sleeves', (select count(*) from sleeves), \
        'orders', (select count(*) from orders), 'fills', (select count(*) from fills), \
        'events', (select count(*) from events))" 2>"$DIR/last_error.txt")
      if [ -n "$counts" ]; then
        status true "$(basename "$f")" "dumped and restored into a scratch database" "$counts"
        wait=86400
      else
        status false "$(basename "$f")" "restored but couldn't read it back: $(head -c 200 "$DIR/last_error.txt" | tr -d '"\n')"
      fi
    else
      status false "$(basename "$f")" "dump didn't restore: $(head -c 200 "$DIR/last_error.txt" | tr -d '"\n')"
    fi
    dropdb --if-exists restore_check 2>/dev/null
  fi
  ls -1t "$DIR"/*.dump 2>/dev/null | tail -n +15 | xargs -r rm -f
  [ -n "${BACKUP_ONCE:-}" ] && break  # one round, for testing
  sleep "$wait"
done
