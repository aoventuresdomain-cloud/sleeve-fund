# Multi-day batch M1 and M3: the 2024 signal count and gross edge, not a try (18 Nov 2026, QD)

Purpose: the Advisor's pre-checks before the Researcher's multi-day batch (research/quant-researcher/next-round/
idea-batch-18nov-b.md) is listed for tries: each definition's entries per asset over the 2024 train window only,
expected test trades pro-rated (at least 60), no asset over half the signals, the first entries' times, each asset's
median size and share capped under the volatility target, and the gross edge of 2024's closed trades before fees and
funding (at least 15 bps a trade pooled, positive on at least 2 of 3 assets; verdicts only). M2 joins once its daily
ATR(50) settles inside the train year: a Wilder ATR(50) needs 1,000 daily candles.

**Count only.** Dispatch with `dry_run` and `count_signals` on, `--pair BTC/USDT,ETH/USDT,SOL/USDT`. The count reads no
bar on or after 2025-01-01, runs each asset alone at no fee and no spread, writes no sheet and records no try. A size
scale is set so that a run dispatched without the dry run by mistake still records no try.

## Common settings

- **Instrument:** BTC, ETH and SOL perpetuals (`market = "perp"`, long and short), on the stored history. No other asset.
- **Fees per side:** none in the count (0%, no spread): the gross edge is before fees.
- **Sizing:** `vol_target_size = { annual = 0.40, lookback_days = 30 }` on every row (variants.py), lag-1 daily
  volatility, capped at 1x.
- **Holdout:** sealed, 2026-05-01 to 2026-10-30. `--use-holdout` is never passed.
- **Walk-forward:** train 366 days (2024: the only days the count reads), test 485 days (2025-01-01 to 2026-04-30).
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded).
- **Fund guards:** off
- **Size scale:** 1
- **Count:** not a try: a count before the batch's list.

## Pinned dates

Every row reads these days only, whatever day it runs (B-37). UTC calendar days; the decision candles close on them.

- **Data start:** 2024-01-01
- **Research end:** 2026-04-30
- **Holdout:** 2026-05-01 to 2026-10-30

| Fold | Train start | Test start | Test end |
| --- | --- | --- | --- |
| 1 | 2024-01-01 | 2025-01-01 | 2026-04-30 |

## Studies

| # | Definition | Idea | Candle |
| --- | --- | --- | --- |
| 1 | qrm1_tsmom_20d | M1, time-series momentum over 20 days | 4h |
| 2 | qrm3_crowd_squeeze_breakout | M3, crowded-side squeeze (funding block: studies only) | 4h |
