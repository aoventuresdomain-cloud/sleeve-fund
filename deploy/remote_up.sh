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
# Keep the dashboard password in step with the GitHub secret (no sed: passwords may contain /).
umask 077
grep -v '^DASHBOARD_PASSWORD=' .env > .env.tmp
printf 'DASHBOARD_PASSWORD=%s\n' "$DASHBOARD_PASSWORD" >> .env.tmp
mv .env.tmp .env
docker compose up -d --build --remove-orphans
docker image prune -f >/dev/null
docker compose ps
