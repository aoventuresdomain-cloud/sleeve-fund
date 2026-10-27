# QR01b V5 long-only basket, fund guards off (27 Oct 2026, Quant Researcher; PM yes 27 Oct 18:52; Advisor 27 Oct)

Purpose: the edge verdict and joint random draws for V5 long-only on BTC, ETH and SOL as one basket, on 4-hour
decisions filled on 1-minute bars, with every fund limit switched off (Advisor 27 Oct 19:13). The guarded run at a
fixed size scale k (10% / this run's pooled worst drawdown) follows in its own list, fitness only.
Backtest studies only: no fund, Strategy, demo or paper change. Dispatched with `--pair BTC/USDT,ETH/USDT,SOL/USDT`.

## Common settings

- **Instrument:** BTC, ETH and SOL perpetuals as one basket (`market = "perp"`, long only), on the stored history.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker). Market orders only. Half the measured spread on top,
  as studies charge it.
- **Holdout:** locked. Every study's data ends 2025-10-19, so the last 365 days are never read, and `--use-holdout`
  is never passed.
- **Walk-forward:** train 730 days, test 365 days. This fits the history before 2025-10-19.
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded), so signals are judged at fixed margin 1x.
- **Fund guards:** off
- **Count:** the basket's one trials row, standing for the V5 pooled try already in the hand count (Advisor 27 Oct:
  it replaces that entry, N stays 28).

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
