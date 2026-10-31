# H0 check 6a: V5 daily series on the round's pinned dates, unguarded, not a try (31 Oct 2026, Quant Researcher; Advisor 31 Oct 05:57 UTC)

Purpose: the V5 side of H0's pre-registered correlation check (section 6a of
research/quant-researcher/next-round/prereg-family-h-trend-pullback-31oct.md). Frozen V5 (hash 5e2fd5a2055facf4) as one
unguarded BTC/ETH/SOL basket on the round's pinned dates, so that its pooled daily returns cover H0's test window
(2025-01-01 to 2026-04-30). **Diagnostic only, never a try.** Its figures are used for the correlation and nothing
else. They are never quoted as V5 performance; the paper test is V5's evidence (Advisor 31 Oct). Backtest studies only:
no fund, Strategy, demo or paper change. Dispatched with `--pair BTC/USDT,ETH/USDT,SOL/USDT`.

Why "Size scale: 1": a size scale is how a list marks a confirming run that records no try (basket_run.py:156). At
scale 1 every leg's account holds its full even third with no idle cash, and with fund guards off the book is the
same as the unguarded verdict run's (QD, 31 Oct). So the sheet's "every leg at 1 of its size" line means "full size,
not counted".

## Common settings

- **Instrument:** BTC, ETH and SOL perpetuals as one basket (`market = "perp"`, long only), on the stored history.
  No other asset (PM 30 Oct 14:46 UTC).
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker). Market orders only. Half the measured spread on top,
  as studies charge it.
- **Holdout:** sealed, 2026-05-01 to 2026-10-30. `--use-holdout` is never passed, and the runner refuses it for a
  confirming run (run.py:133).
- **Walk-forward:** train 366 days (2024, fits nothing: settings are frozen), test 485 days (2025-01-01 to 2026-04-30).
- **Warm-up:** V5's longest lookback is 1,280 4h candles (about 213 days, ends about 1 Aug 2024) and 140 daily
  candles, all inside the train year (check_definition on main, 31 Oct). No bar before 2024-01-01 is read.
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded).
- **Fund guards:** off
- **Size scale:** 1
- **Count:** not a new try: a confirming run of V5's frozen settings, already counted (N = 42), for H0's check 6a
  only (Advisor 31 Oct).

## Pinned dates

Every row reads these days only, whatever day it runs (B-37). UTC calendar days; the decision candles close on them.

- **Data start:** 2024-01-01
- **Research end:** 2026-04-30
- **Holdout:** 2026-05-01 to 2026-10-30

| Fold | Train start | Test start | Test end |
| --- | --- | --- | --- |
| 1 | 2024-01-01 | 2025-01-01 | 2026-04-30 |

## Studies (definition file in configs/definitions/variants/ on main, by its pre-registered name)

| # | Definition | Idea family | Decision candle |
|---|---|---|---|
| 1 | v5_lo_adx | QR-B2 multi-speed trend | 4h |

Row 1 is configs/definitions/variants/qr01b_trend_multispeed_lo_v5_adx.toml (sleeve_fund/research/variants.py,
V5_LO_ADX). No PASS or "edge found" wording.

## Checks before the first dry run

1. The dry run's Tries line must read "not a new try"; if it shows a count, nothing runs.
2. The holdout length setting is 183 days, matching the pinned holdout.
3. Run only after H0's sheet is exported, one study at a time.
