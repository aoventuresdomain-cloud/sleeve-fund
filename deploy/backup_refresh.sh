#!/usr/bin/env bash
# Runs from remote_up.sh, in /opt/sleeve-fund, after compose up. backup.sh is bind-mounted into the backup container,
# so compose up leaves it running the old script; it is recreated when the script changed since the last recreate
# (.backup_sha; the first deploy too). Never on an unchanged script: each recreate dumps at once, and backup.sh keeps
# 14 dumps by count, so a recreate per deploy would push nights of history out. The stamp is written only after the
# recreate succeeded; a failed one fails the deploy and the next deploy tries again.
set -euo pipefail
STAMP=.backup_sha
new=$(sha256sum deploy/backup.sh | cut -d ' ' -f 1)
if [ "$new" = "$(cat "$STAMP" 2>/dev/null || true)" ]; then
  echo "backup: script unchanged, left running"
  exit 0
fi
docker compose up -d --force-recreate --no-deps backup
printf '%s\n' "$new" > "$STAMP.tmp"
mv "$STAMP.tmp" "$STAMP"
echo "backup: recreated to run the new script"
