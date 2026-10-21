# QR stop-loss and take-profit exit-set study list (21 Oct 2026, Quant Researcher)

Purpose: every long Strategy re-run as its parent PLUS an explicit stop-loss and take-profit, fixed per type before
any run (PM 21 Oct 14:42-14:44 UK; Advisor 21 Oct, "add, don't replace"): trend adds a 3 ATR stop, 3R target and 3 ATR
trail; breakout adds a 2R target (from the parent's own stop) and a 10-candle time stop where the parent has none;
mean reversion adds a target resting at its 20-candle average. Parents' own stops are kept. Parents' own exits and time stops stay; only the
re-arm changes. Needs TP-EXIT live (Deploy 2). Pre-registration: research/quant-researcher/
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
| 1 | qr23_vma_band_sma50_sltp_long | QR-B12 price vs average | 1d |
| 2 | qr21_trend_sma200_sltp_long | QR-B12 price vs average | 1d |
| 3 | qr22_price_above_sma20_sltp_long | QR-B12 price vs average | 1d |
| 4 | qr01c_trend_donchian_sltp_long | QR-B3 Donchian trend | 1d |
| 5 | qr01d_trend_donchian20_sltp_long | QR-B3 Donchian trend | 1d |
| 6 | qr01b_trend_multispeed_sltp_long | QR-B2 multi-speed trend | 4h |
| 7 | qr01_trend_ema_sltp_long | QR-B1 trend EMA 21/55 | 4h |
| 8 | qr03_squeeze_sltp_long | QR-B5 squeeze breakout | 1h |
| 9 | qr12_overnight_breakout_sltp_long | QR-B8 overnight breakout | 1h |
| 10 | qr20_volume_surge_breakout_sltp_long | QR-B11 volume surge breakout | 4h |
| 11 | qr02_dip_sltp_long | QR-B4 trend filter + dip | 4h |
| 12 | qr10_capitulation_sltp_long | QR-B6 capitulation rebound | 1d |
| 13 | qr11_exhaustion_sltp_long | QR-B7 exhaustion reversal | 15m |

Each row is judged against its parent (the same name without `_sltp`): G1 on its own; OOS max drawdown at least 15 pp
shallower; OOS Sharpe at least 0.9 x the parent's (Advisor, 21 Oct). Exits are counted by type and the share of profit
from take-profit exits is reported. Rows 1, 2 and 6 also compare with their trail-only versions (qr23b, qr21b, qr01e).
