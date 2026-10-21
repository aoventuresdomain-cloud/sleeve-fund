#!/usr/bin/env bash
# R-29: research_config_only.sh <sha the server runs> <sha picked to ship>. Exits 0 when every change between the two
# is an ADDITIVE research config the study job reads from main (study.yml mounts it), so the deploy can skip the restart
# and keep .deployed_sha; exits 1 when anything else changed, and the deploy ships as before.
# A change is research config only when it is a regular file (mode 100644) and one of (HoE R-29 v3, CR M1):
#   a NEW definition, configs/definitions/<a-z0-9_>.toml or configs/definitions/variants/<a-z0-9_>.toml (status A);
#   a study list, configs/studies/<a-z0-9_->.md (added or modified);
#   configs/definition_pins.toml with lines added only (new pins): none removed or changed.
# An edited definition is a new trials key, and a changed pin hides one: a code change, a deploy and a TEC. A
# deletion, a rename (seen as a deletion and an addition: renames are not detected), a mode or type change, a
# symlink, a nested path, an upper-case name, a quoted (unusual) path, or no change at all: not config only.
set -euo pipefail
from=${1:-}
to=${2:-}
sha_re='^[0-9a-f]{40}$'
[[ $from =~ $sha_re && $to =~ $sha_re ]] || exit 1
[ "$from" != "$to" ] || exit 1
raw=$(git diff --raw --no-renames --no-abbrev "$from" "$to") || exit 1
[ -n "$raw" ] || exit 1
pins=configs/definition_pins.toml
while IFS=$'\t' read -r meta path; do
  # meta: ":<old mode> <new mode> <old blob> <new blob> <status>"
  read -r _ new_mode _ _ status <<<"$meta"
  [ "$new_mode" = 100644 ] || exit 1
  if [[ $path =~ ^configs/definitions/(variants/)?[a-z0-9_]+\.toml$ ]]; then
    [ "$status" = A ] || exit 1
  elif [[ $path =~ ^configs/studies/[a-z0-9_-]+\.md$ ]]; then
    [[ $status == A || $status == M ]] || exit 1
  elif [ "$path" = "$pins" ]; then
    [[ $status == A || $status == M ]] || exit 1
    removed=$(git diff --unified=0 --no-color "$from" "$to" -- "$pins" | grep -v '^--- ' | grep -c '^-' || true)
    [ "$removed" = 0 ] || exit 1
  else
    exit 1
  fi
done <<<"$raw"
exit 0
