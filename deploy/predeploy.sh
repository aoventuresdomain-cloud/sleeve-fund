#!/usr/bin/env bash
# Runs from remote_up.sh, in /opt/sleeve-fund, before compose up starts the new code: the pre-deploy dump
# (predeploy_dump.sh, inside the backup container). Skipped only on a first deploy, when the journal's volume doesn't
# exist yet; never because the database happens to be down (HoQA F-PDD-1). A stopped database and backup are started
# as they are (`compose start` never recreates, so the new config can't start before the dump), then dumped. Any
# failure exits non-zero and remote_up.sh (set -e) stops the deploy with the old containers as they were.
set -euo pipefail
vols=$(docker volume ls -q)  # docker not answering fails here (set -e): never read as "no journal"
if ! grep -qE '(^|_)pgdata$' <<< "$vols"; then
  echo "pre-deploy dump: SKIP, no journal yet (first deploy)"
  exit 0
fi
docker compose start db backup >/dev/null
for _ in $(seq 1 60); do
  docker compose exec -T backup pg_isready -q -h db >/dev/null 2>&1 && break
  sleep 1
done
docker compose exec -T backup pg_isready -q -h db >/dev/null 2>&1 \
  || { echo "pre-deploy dump: FAIL, the journal's database didn't answer"; exit 1; }
docker compose exec -T backup bash -s < deploy/predeploy_dump.sh
