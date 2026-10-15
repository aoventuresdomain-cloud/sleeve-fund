#!/usr/bin/env bash
# The server side of restore-journal.yml (by hand, from main, only on the PM's go; run as root in /opt/sleeve-fund):
# stop the app, put the chosen dump back as the journal (restore_journal_inner.sh, in the backup container), bring it to
# the code's migration and start the app again. The database and backup containers keep running throughout. A failure
# before the swap starts the app again on the untouched journal; a failure after it keeps the app stopped, with the
# replaced journal kept beside the restored one, for the operator, as does a journal left out of its place (exit 2).
# Exit 0 when the app is back on the restored journal.
set -uo pipefail
fail() { echo "restore journal: FAIL, $1"; exit 1; }
# The PM's go, and the dump typed twice: the workflow passes both from its inputs, through the environment only.
[ "${RESTORE_JOURNAL_GO:-}" = "pm-go" ] || fail "no go from the PM"
DUMP=${DUMP:-}
[[ "$DUMP" =~ ^(predeploy-[0-9]{8}-[0-9]{6}\.pgdump|sleeve_fund-[0-9A-Za-z_-]+\.dump)$ ]] || fail "not a dump's name"
[ "${CONFIRM:-}" = "restore $DUMP" ] || fail "the confirmation doesn't name the dump"
running=$(docker compose ps --services --status running 2>/dev/null) || fail "docker didn't answer"
app=$(grep -vxE 'db|backup|caddy' <<< "$running" | tr '\n' ' ' | sed 's/ *$//')
if [ -n "$app" ]; then
  # shellcheck disable=SC2086 # one word per service
  docker compose stop $app >/dev/null 2>&1 || fail "couldn't stop the app"
fi
echo "app: stopped (${app:-nothing running})"
start_app() {
  # shellcheck disable=SC2086
  [ -z "$app" ] || docker compose start $app >/dev/null 2>&1
}
out=$(docker compose exec -T -e DUMP="$DUMP" backup bash -s < deploy/restore_journal_inner.sh)
rc=$?
printf '%s\n' "$out"
if [ "$rc" != 0 ]; then
  # Exit 2 (or its words, from an older inner script): no database carries the journal's name. Never start the app.
  if [ "$rc" = 2 ] || grep -q "journal is still named" <<< "$out"; then
    echo "restore journal: FAIL, the journal is out of its place; the app stays stopped for the operator"
    exit 2
  fi
  start_app || fail "the journal is as it was, but the app didn't start again"
  echo "app: started again on the journal as it was"
  fail "the journal is as it was"
fi
if ! docker compose run --rm --no-deps migrate >/dev/null 2>&1; then
  fail "the restored journal didn't migrate to the code's schema; the app stays stopped"
fi
echo "migrate: PASS"
start_app || fail "the restored journal is in place but the app didn't start"
echo "app: started on the restored journal. Next: the start check and the books check (nightly-recon.yml)"
