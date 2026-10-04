#!/usr/bin/env bash
# Live smoke check: run the smoke sleeve against Kraken's public feed for a few
# minutes and fail unless data arrived. Paper only; uses no credentials.
set -euo pipefail
MINUTES="${1:-4}"
LOG="$(mktemp)"
python -m sleeve_fund.paper configs/examples/btc_trend_smoke.toml --minutes "$MINUTES" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee "$LOG"
if grep -q "Failed to connect" "$LOG"; then echo "FAIL: data client did not connect"; exit 1; fi
BARS=$(grep -c "\] .*TrendFilter.*: bar " "$LOG" || true)
echo "live bars received: $BARS"
[ "$BARS" -ge 1 ] || { echo "FAIL: no live bars from Kraken"; exit 1; }
echo "PASS"
