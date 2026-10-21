# QR re-score study list (20 Oct 2026, Quant Researcher)

Purpose: re-score the 16 research definitions at the new paper-venue fees. BTC first; other instruments only after
the BTC table is read. These are backtest studies only: no fund, Strategy, demo or paper change.

## Common settings

- **Instrument:** BTC/USDT perpetual (`market = "perp"`; `allow_short` only on the short-leg definitions), on the paper
  venue's stored history.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker). Market orders only. Half the measured spread on top,
  as studies charge it.
- **Holdout:** locked. Every study's data ends 2025-10-19, so the last 365 days are never read, and `--use-holdout`
  is never passed.
- **Walk-forward:** train 730 days, test 365 days. This fits the history before 2025-10-19.
- **Scoring window:** continuous: a warm-up as each indicator needs to settle (exact rolling windows such as SMA,
  Donchian and relative volume: their period; recursive EMA and Wilder blocks: 10x their period; the longest applies),
  then scored to 2025-10-19 (the day before the holdout). QR-B1, QR-B2 and QR-B3 rows also show the old three-year
  walk-forward verdict and pass only if both pass (SCORE-WINDOW addendum).
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
| 1 | qr01_trend_ema_long | QR-B1 trend EMA 21/55 | 4h |
| 2 | qr01_trend_ema_short | QR-B1 trend EMA 21/55 | 4h |
| 3 | qr01b_trend_multispeed_long | QR-B2 multi-speed trend | 4h |
| 4 | qr01b_trend_multispeed_short | QR-B2 multi-speed trend | 4h |
| 5 | qr01c_trend_donchian_long | QR-B3 Donchian trend | 1d |
| 6 | qr01c_trend_donchian_short | QR-B3 Donchian trend | 1d |
| 7 | qr02_dip_long | QR-B4 trend filter + dip | 4h |
| 8 | qr02_dip_short | QR-B4 trend filter + dip | 4h |
| 9 | qr03_squeeze_long | QR-B5 squeeze breakout | 1h |
| 10 | qr03_squeeze_short | QR-B5 squeeze breakout | 1h |
| 11 | qr10_capitulation_long | QR-B6 capitulation rebound | 1d |
| 12 | qr11_exhaustion_long | QR-B7 exhaustion reversal | 15m |
| 13 | qr11_exhaustion_short | QR-B7 exhaustion reversal | 15m |
| 14 | qr12_overnight_breakout_long | QR-B8 overnight breakout | 1h |
| 15 | qr13_level_continuation_long | QR-B9 level continuation | 1h |
| 16 | qr13_level_continuation_short | QR-B9 level continuation | 1h |

Run order: 1 to 8 first, because they make the first ranked table the PM wants for tomorrow morning; then 9 to 16.

## Notes for whoever runs them

- The CLI's `study` takes a model name and has no definition argument. The run needs the `rules` model with
  `params = to_params(definition)` plus `market = "perp"`, as the Research page does for a rule-builder Strategy.
- Rows 15 and 16 use first_touch, so they need the 1-minute loader (`minute_loader`).
- My cancelled GitHub lab (research/qr-lab/ on my branch, manual-only, never to be dispatched) holds the scoring
  I'd apply to the results: the fee ladder, break-even and random entry. It may help as a reference.
