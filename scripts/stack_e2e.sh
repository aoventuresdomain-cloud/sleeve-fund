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
# Prove any asset works: add a non-BTC sleeve through the dashboard form, as the PM would.
for i in $(seq 1 30); do
  docker compose exec -T dashboard python -c "
import base64, urllib.parse, urllib.request
form = urllib.parse.urlencode({'name': 'sui-e2e', 'strategy': 'trend_filter', 'instrument': 'SUI/USD',
    'bar_spec': '1-MINUTE-LAST-INTERNAL', 'starting_balance': '5000', 'risk_profile': 'balanced',
    'warmup_bars': '0', 'p_trend_filter__fast': '5', 'p_trend_filter__slow': '20', 'reason': 'e2e any-asset check'}).encode()
req = urllib.request.Request('http://localhost:8000/sleeves/new', data=form, headers={
    'Authorization': 'Basic ' + base64.b64encode(b'pm:e2e-dash-pw').decode(),
    'Origin': 'http://localhost:8000', 'Host': 'localhost:8000'})
urllib.request.urlopen(req)" && break
  sleep 2
done
echo "waiting ${WAIT}s for sleeves to connect and mark..."
sleep "$WAIT"
q() { docker compose exec -T db psql -U sleeve -d sleeve_fund -tAc "$1"; }
echo "sleeves:"; q "select name, instrument, status, heartbeat_at from sleeves order by id"
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
[ "$HEART" -ge 3 ] || { echo "FAIL: expected 3 heartbeating sleeves (BTC smoke, BTC daily, SUI)"; exit 1; }
[ "$MARKS" -ge 3 ] || { echo "FAIL: no equity marks"; exit 1; }
[ "$ERRS" -eq 0 ] || { echo "FAIL: error events recorded"; exit 1; }
[ "$CODE" = "200" ] && [ "$NOAUTH" = "401" ] || { echo "FAIL: dashboard auth"; exit 1; }

# Restart check: a position in the journal must survive a restart and reconcile with the
# rebuilt paper engine. Journal a 10 SUI buy, restart the sleeves, and expect them to carry it.
q "insert into fills (sleeve, ts, side, qty, price, fee, order_id, trade_id)
   values ('sui-e2e', now(), 'BUY', 10, 1.0, 0.01, 'e2e-carry', 'e2e-carry')"
docker compose restart supervisor
echo "waiting ${WAIT}s after restart..."
sleep "$WAIT"
q "select ts, sleeve, level, kind, message from events where kind in ('restore', 'reconcile', 'reconcile_mismatch') order by id"
RESTORED=$(q "select count(*) from events where sleeve = 'sui-e2e' and kind = 'restore' and message like '%coin 10%'")
RECONCILED=$(q "select count(*) from events where sleeve = 'sui-e2e' and kind = 'reconcile'")
ERRS=$(q "select count(*) from events where level = 'error'")
HEART=$(q "select count(*) from sleeves where heartbeat_at > now() - interval '2 minutes'")
echo "restored: $RESTORED, reconciled: $RECONCILED, error events: $ERRS, heartbeating: $HEART"
[ "$RESTORED" -ge 1 ] || { echo "FAIL: SUI position not restored from the journal"; exit 1; }
[ "$RECONCILED" -ge 2 ] || { echo "FAIL: expected a reconciliation before and after the restart"; exit 1; }
[ "$ERRS" -eq 0 ] || { echo "FAIL: error events after restart (a reconcile mismatch?)"; exit 1; }
[ "$HEART" -ge 3 ] || { echo "FAIL: sleeves not heartbeating after restart"; exit 1; }
echo PASS
