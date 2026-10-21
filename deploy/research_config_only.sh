#!/usr/bin/env bash
# R-29: research_config_only.sh <sha the server runs> <sha picked to ship>. Exits 0 when every change between the two
# is a research config the study job reads from main (study.yml mounts it), so the deploy can skip the restart and keep
# .deployed_sha; exits 1 when anything else changed, and the deploy ships as before.
# A change is research config only when it adds or modifies a regular file (mode 100644) at exactly
#   configs/definitions/<a-z0-9_>.toml, configs/definitions/variants/<a-z0-9_>.toml (R-06/R-07)
#   or configs/studies/<a-z0-9_->.md
# A deletion, a rename (seen as a deletion and an addition: renames are not detected), a mode or type change, a
# symlink, a nested path, an upper-case name, a quoted (unusual) path, or no change at all: not config only.
set -euo pipefail
from=${1:-}
to=${2:-}
sha_re='^[0-9a-f]{40}$'
[[ $from =~ $sha_re && $to =~ $sha_re ]] || exit 1
[ "$from" != "$to" ] || exit 1
raw=$(git diff --raw --no-renames --no-abbrev "$from" "$to") || exit 1
[ -n "$raw" ] || exit 1
path_re='^configs/(definitions/(variants/)?[a-z0-9_]+\.toml|studies/[a-z0-9_-]+\.md)$'
while IFS=$'\t' read -r meta path; do
  # meta: ":<old mode> <new mode> <old blob> <new blob> <status>"
  read -r _ new_mode _ _ status <<<"$meta"
  [[ $status == A || $status == M ]] || exit 1
  [ "$new_mode" = 100644 ] || exit 1
  [[ $path =~ $path_re ]] || exit 1
done <<<"$raw"
exit 0
