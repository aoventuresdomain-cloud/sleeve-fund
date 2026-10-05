#!/usr/bin/env bash
# Runs ON the server after each deploy. Idempotent: installs Docker once, creates
# .env once (random database password), then (re)builds and starts the stack.
set -euo pipefail
cd /opt/sleeve-fund
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
if [ ! -f .env ]; then
  umask 077
  cat > .env <<ENV
POSTGRES_PASSWORD=$(openssl rand -hex 24)
DASHBOARD_PASSWORD=${DASHBOARD_PASSWORD:?DASHBOARD_PASSWORD missing}
SITE_ADDRESS=${SITE_ADDRESS:?SITE_ADDRESS missing}
ENV
fi
# Kraken keys for live accounts go here by hand, on the server only (see the dashboard's Accounts
# page). Created empty and private so the supervisor can always load it; deploys never touch it.
if [ ! -f kraken.env ]; then
  umask 077
  : > kraken.env
fi
# Keep the dashboard password in step with the GitHub secret (no sed: passwords may contain /).
umask 077
grep -v '^DASHBOARD_PASSWORD=' .env > .env.tmp
printf 'DASHBOARD_PASSWORD=%s\n' "$DASHBOARD_PASSWORD" >> .env.tmp
mv .env.tmp .env
# Optional alert settings and the demo mirror's demo keys follow their GitHub secrets too; an unset
# secret removes the line. Only the mirror container is given the demo keys (docker-compose.yml).
for var in ALERT_WEBHOOK_URL HEALTHCHECK_PING_URL DEMO_MIRROR DERIBIT_TESTNET_API_KEY DERIBIT_TESTNET_API_SECRET \
           BYBIT_DEMO_API_KEY BYBIT_DEMO_API_SECRET; do
  grep -v "^$var=" .env > .env.tmp || true
  if [ -n "${!var:-}" ]; then printf '%s=%s\n' "$var" "${!var}" >> .env.tmp; fi
  mv .env.tmp .env
done
docker compose up -d --build --remove-orphans
docker image prune -f >/dev/null
docker compose ps
# The demo mirror's first lines say whether each demo key signed in (sleeve_fund/mirror.py); never a key.
sleep 15
docker compose logs --no-log-prefix --tail 10 mirror || true
