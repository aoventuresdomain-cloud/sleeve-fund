# QR round 1 study list (21 Oct 2026, Quant Researcher; PM 17:56 UK "yes to all")

Purpose: the PM's RSI 80/20 idea and the three leaders, each leader's SL/TP version beside its parent, run as
BTC/ETH/SOL/XRP baskets (1d rows), and as four single-asset studies reported together as ONE fixed asset set for the
15m and 4h rows until intraday baskets land (Advisor 18:30 UK). Spot rows: rows 3, 5 and 7 again under the research
spot market setting (Deploy 3). 10 tries in all (Advisor 18:01 UK). Pre-registration: research/quant-researcher/
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
| 1 | qr30_rsi_8020_reversal | QR-B14 RSI 80/20 reversal | 15m |
| 2 | qr23_vma_band_sma50_sltp_long | QR-B12 price vs average | 1d |
| 3 | qr23_vma_band_sma50_long | QR-B12 price vs average | 1d |
| 4 | qr01c_trend_donchian_sltp_long | QR-B3 Donchian trend | 1d |
| 5 | qr01c_trend_donchian_long | QR-B3 Donchian trend | 1d |
| 6 | qr01b_trend_multispeed_sltp_long | QR-B2 multi-speed trend | 4h |
| 7 | qr01b_trend_multispeed_long | QR-B2 multi-speed trend | 4h |

Rows 2, 4 and 6 are judged against rows 3, 5 and 7 (G1 on its own; OOS max drawdown at least 15 pp shallower; OOS
Sharpe at least 0.9 x the parent's). Row 1 has a short side, so no hold bar. Exits are counted by kind. No PASS or
"edge found" wording until PR E and BH-102 are live.
