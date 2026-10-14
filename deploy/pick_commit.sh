#!/usr/bin/env bash
# Picks the commit deploy.yml ships: pick_commit.sh <tested sha> [<sha the server runs>].
# Prints the sha to ship, or nothing when this run ships nothing; its reasons go to stderr as run annotations.
# Run from a full clone of the repository (origin = GitHub) with gh signed in.
# - main still at the tested commit: that commit.
# - main moved past it: the newest commit on main whose tests passed on a push to main (gh run list), never one
#   whose run is in flight or failed; the commit its own run would ship, so a run that waited behind another
#   deploy ships what is green now instead of nothing.
# - never the commit the server already runs, nor one that doesn't descend from it (no rollback): a run that
#   lost the race to a newer deploy stops.
set -euo pipefail
ship=$1
deployed=${2:-}
sha_re='^[0-9a-f]{40}$'
[[ $ship =~ $sha_re ]] || { echo "::error::not a commit sha: '$ship'" >&2; exit 1; }
if [ -n "$deployed" ] && ! [[ $deployed =~ $sha_re ]]; then
  echo "::error::the server's deployed-commit record is not a commit sha: '$deployed'" >&2; exit 1
fi

git fetch -q origin main
tip=$(git rev-parse FETCH_HEAD)
pick=$ship
if [ "$tip" != "$ship" ]; then
  green=$(gh run list --workflow tests.yml --branch main --event push --status success --limit 100 \
            --json headSha --jq '.[].headSha')
  pick=
  for c in $(git rev-list --first-parent --max-count 100 "$tip"); do
    if grep -qxF "$c" <<<"$green"; then pick=$c; break; fi
  done
  if [ -z "$pick" ]; then
    echo "::notice::main is at $tip and none of its last 100 commits has green tests on main. Nothing shipped." >&2
    exit 0
  fi
  echo "::notice::main is at $tip, past $ship; shipping $pick, main's newest commit with green tests." >&2
fi

if [ -z "$deployed" ]; then
  echo "::notice::the server has no record of the commit it runs (first deploy with the record)." >&2
elif [ "$deployed" = "$pick" ]; then
  echo "::notice::the server already runs $pick. Nothing shipped." >&2
  exit 0
elif ! git merge-base --is-ancestor "$deployed" "$pick" 2>/dev/null; then
  echo "::warning::$pick does not descend from $deployed, which the server runs; no rollback. Nothing shipped." >&2
  exit 0
fi
echo "$pick"
