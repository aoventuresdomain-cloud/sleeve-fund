#!/usr/bin/env bash
# The backup service's loop (docker-compose.yml): a compressed dump of the journal every night,
# written whole or not at all, then proved by restoring it into a scratch database and counting
# its rows. The newest 14 dumps that restored are kept; a dump that didn't restore is renamed
# .failed and only the newest of those is kept, so a run of failures can never push out a good copy.
# A failed night is tried again an hour later, not a day. The result goes to status.json beside the
# dumps, which the Operations page and the supervisor's alerts read. These dumps sit on this
# machine's disk; the server's automatic Lightsail snapshots are the copy that survives losing it.
set -uo pipefail
DIR=${BACKUP_DIR:-/backups}
DB=${BACKUP_DB:-sleeve_fund}
export PGHOST=${PGHOST:-db} PGUSER=${PGUSER:-sleeve}
json_text() {  # a JSON string's inside: control characters to spaces, backslashes and quotes escaped
  printf '%s' "$1" | tr '\000-\037' ' ' | sed 's/\\/\\\\/g; s/"/\\"/g'
}
why() {  # the first 200 characters of the last error, as one line
  head -c 200 "$DIR/last_error.txt" | iconv -c -f utf-8 -t utf-8 2>/dev/null || head -c 200 "$DIR/last_error.txt"
}
status() {  # ok, file, message, restored counts (JSON object)
  printf '{"ts": "%s", "ok": %s, "file": "%s", "message": "%s", "restored": %s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$(json_text "$2")" "$(json_text "$3")" "${4:-null}" > "$DIR/status.json.part"
  mv "$DIR/status.json.part" "$DIR/status.json"
}
while true; do
  f="$DIR/$DB-$(date -u +%Y%m%d-%H%M).dump"
  wait=3600
  if ! pg_dump -d "$DB" -Fc -f "$f.part" 2>"$DIR/last_error.txt"; then
    rm -f "$f.part"
    status false "" "pg_dump failed: $(why)"
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
        mv "$f" "$f.failed"
        status false "$(basename "$f").failed" "restored but couldn't read it back: $(why)"
      fi
    else
      mv "$f" "$f.failed"
      status false "$(basename "$f").failed" "dump didn't restore: $(why)"
    fi
    dropdb --if-exists restore_check 2>/dev/null
  fi
  ls -1t "$DIR"/*.dump 2>/dev/null | tail -n +15 | xargs -r rm -f
  ls -1t "$DIR"/*.dump.failed 2>/dev/null | tail -n +2 | xargs -r rm -f
  [ -n "${BACKUP_ONCE:-}" ] && break  # one round, for testing
  sleep "$wait"
done
