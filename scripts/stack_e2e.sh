#!/usr/bin/env bash
# End-to-end check of the full stack on live Kraken data (paper only, no credentials):
# compose up, wait, then assert the sleeves are heartbeating and marking equity, and
# the dashboard serves the portfolio page behind its password.
set -euo pipefail
WAIT="${1:-150}"
cat > .env <<ENV
POSTGRES_PASSWORD=e2e-db-pw
DASHBOARD_PASSWORD=e2e-dash-pw
SITE_ADDRESS=localhost
ENV
trap 'docker compose logs --no-color --tail=80 supervisor dashboard; docker compose down -v' EXIT
docker compose up -d --build
echo "waiting ${WAIT}s for sleeves to connect and mark..."
sleep "$WAIT"
q() { docker compose exec -T db psql -U sleeve -d sleeve_fund -tAc "$1"; }
echo "sleeves:"; q "select name, status, heartbeat_at from sleeves order by id"
HEART=$(q "select count(*) from sleeves where heartbeat_at > now() - interval '2 minutes'")
MARKS=$(q "select count(*) from equity")
ERRS=$(q "select count(*) from events where level = 'error'")
echo "heartbeating sleeves: $HEART, equity marks: $MARKS, error events: $ERRS"
q "select ts, sleeve, level, kind, message from events order by id" | tail -20
CODE=$(docker compose exec -T dashboard python -c "
import base64, urllib.request
req = urllib.request.Request('http://localhost:8000/', headers={'Authorization': 'Basic ' + base64.b64encode(b'pm:e2e-dash-pw').decode()})
print(urllib.request.urlopen(req).status)")
NOAUTH=$(docker compose exec -T dashboard python -c "
import urllib.request, urllib.error
try: urllib.request.urlopen('http://localhost:8000/'); print(200)
except urllib.error.HTTPError as e: print(e.code)")
echo "dashboard: with password $CODE, without $NOAUTH"
[ "$HEART" -ge 2 ] || { echo "FAIL: expected 2 heartbeating sleeves"; exit 1; }
[ "$MARKS" -ge 2 ] || { echo "FAIL: no equity marks"; exit 1; }
[ "$ERRS" -eq 0 ] || { echo "FAIL: error events recorded"; exit 1; }
[ "$CODE" = "200" ] && [ "$NOAUTH" = "401" ] || { echo "FAIL: dashboard auth"; exit 1; }
echo PASS
