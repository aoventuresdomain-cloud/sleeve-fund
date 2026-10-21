# QR trend exit-set variants study list (21 Oct 2026, Quant Researcher)

Purpose: the trend exit set (3 ATR stop + 3 ATR ratchet trail, re-armed on a fresh signal) on the three leading trend
Strategies, pre-registered after the PM's 21 Oct 14:42-14:44 UK rule that every strategy carries stop-loss and
take-profit logic (the fixed-TP versions wait for TP-EXIT). Pre-registration: research/quant-researcher/
trend-variants-preregistration.md. Backtest studies only: no fund, Strategy, demo or paper change.

## Common settings

- **Instrument:** each pool asset in turn (BTC, ETH, SOL, XRP perpetuals, `market = "perp"`), on the stored history.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker). Market orders only. Half the measured spread on top,
  as studies charge it.
- **Holdout:** locked. Every study's data ends 2025-10-19, so the last 365 days are never read, and `--use-holdout`
  is never passed.
- **Walk-forward:** train 730 days, test 365 days. This fits the history before 2025-10-19.
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded), so signals are judged at fixed margin 1x.
- **Count:** each row is one pre-registered setting, so one try per idea in the trials register. Ablation variants
  are not counted (Advisor rule).
- **Scored on the Advisor's Edge found definition**, in order: net return at or above the 95th percentile of
  random entries (weak if in the market over 60%); long-only rows must also beat buy-and-hold; still profitable at
  2x fees (0.11%); break-even fee shown. The holdout and paper checks come later.

## Pinned dates

Every row reads these days only, whatever day it runs (B-37; Researcher 20 Oct, Advisor-checked). UTC calendar days;
the decision candles close on them. Minute data for the 15m and 1h rows and the first_touch loader stays inside them.

- **Data start:** 2020-10-21
- **Research end:** 2025-10-19
- **Holdout:** 2025-10-20 to 2026-10-19

| Fold | Train start | Test start | Test end |
| --- | --- | --- | --- |
| 1 | 2020-10-21 | 2022-10-21 | 2023-10-20 |
| 2 | 2021-10-21 | 2023-10-21 | 2024-10-19 |
| 3 | 2022-10-21 | 2024-10-20 | 2025-10-19 |

## Studies (definition file in configs/definitions/ on main)

| # | Definition | Idea family | Decision candle |
|---|---|---|---|
| 1 | qr23b_vma_band_sma50_trail_long | QR-B12 price vs average | 1d |
| 2 | qr21b_trend_sma200_trail_long | QR-B12 price vs average | 1d |
| 3 | qr01e_trend_multispeed_trail_long | QR-B2 multi-speed trend | 4h |

Each row is judged against its parent (qr23, qr21, qr01b long): G1 on its own; OOS max drawdown at least 15 pp
shallower; OOS Sharpe at least 0.9 x the parent's (Advisor, 21 Oct). Exits are counted by type.
