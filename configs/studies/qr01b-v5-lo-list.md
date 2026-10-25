# QR01b V5 long-only study list (25 Oct 2026, Quant Researcher; Advisor cleared as try 28; the PM chose one long-only test 25 Oct)

Purpose: one long-only test of qr01b. V5, "long-only with the ADX gate": an exact copy of V4 (qr01b-r2-list.md) with
the short leg deleted, so the long-only parent plus the daily ADX(14) <= 25 entry gate. V4's short leg lost on BTC, ETH
and SOL in the test years while the long leg held; the idea comes from these results. One row, as V4's list. One try.
Backtest studies only: no fund, Strategy, demo or paper change.

## Common settings

- **Instrument:** BTC perpetual (`market = "perp"`, long only: the definition has no short side), on the stored history.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker). Market orders only. Half the measured spread on top,
  as studies charge it.
- **Holdout:** locked. Every study's data ends 2025-10-19, so the last 365 days are never read, and `--use-holdout`
  is never passed.
- **Walk-forward:** train 730 days, test 365 days. This fits the history before 2025-10-19.
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded), so signals are judged at fixed margin 1x.
- **Count:** the row is one pre-registered setting, so one try for the qr01b family in the trials register.
- **Judged against V4** (qr01b_trend_multispeed_ls_v4_short_sma200) and the PM's bar, out of sample, on the same dates.

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
| 1 | v5_lo_adx | QR-B2 multi-speed trend | 4h |

Row 1 is configs/definitions/variants/qr01b_trend_multispeed_lo_v5_adx.toml
(sleeve_fund/research/variants.py, V5_LO_ADX). No PASS or "edge found" wording until PR E and BH-102 are live.
