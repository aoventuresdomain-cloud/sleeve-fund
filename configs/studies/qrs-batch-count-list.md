# Idea batch S1-S4: the 2024 signal count, not a try (18 Nov 2026, QD; Advisor cleared the batch for the count)

Purpose: the Advisor's pre-checks before S1-S4 are listed as family B tries 27-30
(research/quant-researcher/next-round/idea-batch-18nov.md): each definition's entries per asset over the 2024 train
window only, expected test trades pro-rated (at least 60), no asset over half the signals, the first entries' times,
the exit-rule exits that land on a later UTC day than their entry (S4: its VWAP exits after a midnight reset), and each
asset's median size and share capped at 1x under the volatility target. Rows 5 and 6 are S3's and S4's pre-registered
fallbacks, read only if the main version's count is under 60.

**Count only.** Dispatch with `dry_run` and `count_signals` on, `--pair BTC/USDT,ETH/USDT,SOL/USDT`. The count reads no
bar on or after 2025-01-01, runs each asset alone, writes no sheet and records no try. The rows' studies proper go on
the round's own list once the count clears them. A size scale is set so that a run dispatched without the dry run by
mistake still records no try.

## Common settings

- **Instrument:** BTC, ETH and SOL perpetuals (`market = "perp"`, long and short), on the stored history. No other asset.
- **Fees per side:** 0.055% (VIP 0 perpetual taker), market orders only.
- **Sizing:** `vol_target_size = { annual = 0.40, lookback_days = 30 }` on every row (variants.py), lag-1 daily
  volatility, capped at 1x.
- **Holdout:** sealed, 2026-05-01 to 2026-10-30. `--use-holdout` is never passed.
- **Walk-forward:** train 366 days (2024: the only days the count reads), test 485 days (2025-01-01 to 2026-04-30).
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded).
- **Fund guards:** off
- **Size scale:** 1
- **Count:** not a try: a count before the round's list.

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
| 1 | qrs1_short_pullback_downtrend | S1, short pullback in a downtrend | 4h |
| 2 | qrs2_london_asia_range_fade | S2, London fade of the Asia range | 4h |
| 3 | qrs3_shock_fade_4h | S3, shock fade at 2 ATR | 4h |
| 4 | qrs4_vwap_range_reversion_4h | S4, day-VWAP reversion in quiet markets at 1.5 ATR | 4h |
| 5 | qrs3_shock_fade_4h_fb175 | S3's fallback, 1.75 ATR | 4h |
| 6 | qrs4_vwap_range_reversion_4h_fb125 | S4's fallback, 1.25 ATR | 4h |
