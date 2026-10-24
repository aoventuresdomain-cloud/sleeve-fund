# Improve-loop round 1 study list (24 Oct 2026, Quant Researcher; Advisor cleared; improve loop pre-approved by the PM)

Purpose: round 1 of the improve-and-retest loop for qr23 and qr21, one change each to round 0's long and short
(qr-improve-r0-list.md). qr23: shorts only while the last daily close is below its 200-day average, qr01b V4's rule.
qr21: each entry also needs the 200-day average moving its way (above, or below, where it stood 20 daily candles
ago). qr01b's round 2 is qr01b-r2-list.md. BTC only. Backtest studies only: no fund, Strategy, demo or paper change.

## Common settings

- **Instrument:** BTC perpetual (`market = "perp"`, shorts allowed), on the stored history.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker). Market orders only. Half the measured spread on top,
  as studies charge it.
- **Holdout:** locked. Every study's data ends 2025-10-19, so the last 365 days are never read, and `--use-holdout`
  is never passed.
- **Walk-forward:** train 730 days, test 365 days. This fits the history before 2025-10-19.
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded), so signals are judged at fixed margin 1x.
- **Count:** each row is one pre-registered setting, so one try each; qr23, qr21 and qr01b are one family for the
  deflated Sharpe (Advisor 24 Oct).
- **Labelled "development":** BTC round results are not out of sample (Advisor 24 Oct). Each row is judged against its
  round 0 sheet on the same dates.

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
| 1 | qr23_r1 | QR-B12 price-vs-average | 1d |
| 2 | qr21_r1 | QR-B12 200-day trend | 1d |

Row 1 is configs/definitions/variants/qr23_vma_band_sma50_ls_r1_short_sma200.toml and row 2
qr21_trend_sma200_ls_r1_slope.toml (sleeve_fund/research/variants.py, QR23_R1 and QR21_R1). No PASS or "edge found" wording until the Advisor clears it.
