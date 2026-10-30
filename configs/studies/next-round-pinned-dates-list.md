# Next round: families A and B, round 0 (30 Oct 2026, Quant Researcher; PM approved the plan 30 Oct 14:57 UTC; Advisor to clear)

Draft for the Quant Developer to commit under configs/studies/ (the Researcher opens no PRs). Nothing is dispatched until
M1's builds are live and the Advisor has cleared both pre-registrations
(research/quant-researcher/next-round/prereg-family-a-funding-fade-30oct.md and
prereg-family-b-range-reversion-30oct.md). One study at a time, dry run first.
Backtest studies only: no fund, Strategy, demo or paper change. Dispatched with `--pair BTC/USDT,ETH/USDT,SOL/USDT`.

## Common settings

- **Instrument:** BTC, ETH and SOL perpetuals as one basket (`market = "perp"`, long and short), on the stored history.
  No other asset (PM 30 Oct 14:46 UTC).
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker), ladder to 2x (A) and 3x (B). Market orders only. Half the
  measured spread on top, as studies charge it.
- **Holdout:** sealed, 2026-05-01 to 2026-10-30. `--use-holdout` is never passed in round 0 or any improve round; it is
  passed once, for a frozen version only, on its own list.
- **Walk-forward:** train 366 days (2024, fits nothing: settings are fixed), test 485 days (2025-01-01 to 2026-04-30).
- **Warm-up:** inside the train year; no bar before the data start is read (Advisor 30 Oct). A: 90 days (funding
  window). B: 141 days (R-11, 10 x daily ADX(14), checked with main's `continuous_warmup`).
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded).
- **Fund guards:** off
- **Count:** A0 is family A try 1; B0 is family B try 14 (13 prior reversal studies).

## Pinned dates

Every row reads these days only, whatever day it runs (B-37). UTC calendar days; the decision candles close on them.

- **Data start:** 2024-01-01
- **Research end:** 2026-04-30
- **Holdout:** 2026-05-01 to 2026-10-30

| Fold | Train start | Test start | Test end |
| --- | --- | --- | --- |
| 1 | 2024-01-01 | 2025-01-01 | 2026-04-30 |

## Studies (definition files to be added under configs/definitions/variants/ by the Quant Developer)

| # | Definition | Idea family | Decision candle |
|---|---|---|---|
| 1 | qra0_funding_fade | A, funding-rate fade (needs build 1) | 4h |
| 2 | qrb0_range_reversion_adx | B, range mean-reversion | 1h |

Row 1 waits for build 1 (funding rate and percentile-rank block) and the funding counts in its pre-registration.
Row 2 needs no build. No PASS or "edge found" wording.

## Checks for the Quant Developer before the first dry run

1. The history must cover 2024-01-01 to 2026-10-30 or `pinned.cut` refuses the run. The holdout's last day ends at
   2026-10-31 00:00 UTC, so no run can start before then.
2. The study's holdout length setting is 183 days, to match the pinned holdout (`study.py` refuses a mismatch).
3. The test fold's indicators carry the train year's history and are not restarted at 2025-01-01.
