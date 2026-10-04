#!/usr/bin/env bash
# Record live paper sessions on Kraken's public feed, with the orders each sleeve sent, so
# tests/test_replay.py can replay them through the backtest engine. Paper only; no credentials.
#   scripts/record_session.sh [minutes] [out dir]
set -euo pipefail
MINUTES="${1:-45}"
OUT="${2:-recordings}"
mkdir -p "$OUT"
export DATABASE_URL="sqlite:///$OUT/record.db"
STAMP="$(date -u +%Y%m%d-%H%M)"
# Two sleeves that trade often enough to be worth replaying: market orders on 1-minute bars, and
# maker-first orders on 5-minute bars. Settings chosen for activity, not returns.
python - <<'PY'
from sleeve_fund.store import Store
s = Store()
s.create_sleeve(name="rec-1m", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                starting_balance=10_000, risk_profile="aggressive", params={"fast": 2, "slow": 4, "max_notional": 1000})
s.create_sleeve(name="rec-5m-maker", strategy="trend_filter", instrument="BTC/USD", bar_spec="5-MINUTE-LAST-INTERNAL",
                starting_balance=10_000, risk_profile="aggressive",
                params={"fast": 2, "slow": 3, "max_notional": 1000, "maker_wait_minutes": 2})
PY
for name in ${RECORD_ONLY:-rec-1m rec-5m-maker}; do
  python -m sleeve_fund.paper --db-sleeve "$name" --minutes "$MINUTES" \
    --record "$OUT/kraken-btcusd-$name-$STAMP.jsonl.gz" > "$OUT/$name.log" 2>&1 &
done
wait
python - "$OUT" "$STAMP" <<'PY'
import json, sys
from pathlib import Path
from sleeve_fund.store import Store
out, stamp = Path(sys.argv[1]), sys.argv[2]
s = Store()
for name in ("rec-1m", "rec-5m-maker"):
    orders = list(reversed(s.orders(name, limit=100_000)))
    rows = [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in o.items()} for o in orders]
    (out / f"kraken-btcusd-{name}-{stamp}.orders.json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"{name}: {len(rows)} orders")
PY
ls -la "$OUT"
