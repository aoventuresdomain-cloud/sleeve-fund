# QR01b round 2 study list (24 Oct 2026, Quant Researcher; Advisor cleared; improve loop pre-approved by the PM 24 Oct)

Purpose: round 2 of qr01b's improve-and-retest loop. V4, "shorts only in a down market": an exact copy of V3
(qr01b-v3-adx-list.md) with one change, no new short unless the last completed daily close is below its 200-day
average (no short while the average has no value). The V3 short leg made money in 2022, below the long average, and
lost in the 2023 to 2025 uptrend. BTC only on this list. One try. Backtest studies only: no fund, Strategy, demo or
paper change.

## Common settings

- **Instrument:** BTC perpetual (`market = "perp"`, shorts allowed), on the stored history.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker). Market orders only. Half the measured spread on top,
  as studies charge it.
- **Holdout:** locked. Every study's data ends 2025-10-19, so the last 365 days are never read, and `--use-holdout`
  is never passed.
- **Walk-forward:** train 730 days, test 365 days. This fits the history before 2025-10-19.
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded), so signals are judged at fixed margin 1x.
- **Count:** the row is one pre-registered setting, so one try for the qr01b family in the trials register.
- **Judged against V3** (qr01b_trend_multispeed_ls_v3_adx) out of sample, on the same dates.

## Pinned dates

Every row reads these days only, whatever day it runs (B-37; Researcher 20 Oct, Advisor-checked). UTC calendar days;
the decision candles close on them.

- **Data start:** 2020-10-21
- **Research end:** 2025-10-19
- **Holdout:** 2025-10-20 to 2026-10-19

| Fold | Train start | Test start | Test end |
| --- | --- | --- | --- |
| 1 | 2020-10-21 | 2022-10-21 | 2023-10-20 |
| 2 | 2021-10-21 | 2023-10-21 | 2024-10-19 |
| 3 | 2022-10-21 | 2024-10-20 | 2025-10-19 |

## Studies (definition file in configs/definitions/variants/ on main, by its pre-registered name)

| # | Definition | Idea family | Decision candle |
|---|---|---|---|
| 1 | v4_short_sma200 | QR-B2 multi-speed trend | 4h |

Row 1 is configs/definitions/variants/qr01b_trend_multispeed_ls_v4_short_sma200.toml
(sleeve_fund/research/variants.py, V4_SHORT_SMA200). No PASS or "edge found" wording until PR E and BH-102 are live.
