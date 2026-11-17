# QR01b V5 long-only basket, BTC and ETH only, fund guards on at size scale 0.346, under the aggressive profile at a 2x cap (16 Nov 2026, HoE; PM 16 Nov: V5 paper at 1x exposure on 2x isolated margin; Advisor 16 Nov)

Purpose: the 30 Oct reference backtest (qr01b-v5-basket-guarded-btc-eth-list.md, unchanged) re-run as paper will run
V5: every leg under the aggressive profile's caps with a 2x leverage cap as the Strategy's own. Same definition, dates,
fees, guards and size scale; only the risk lines differ (pre-registration rev 2,
research/quant-researcher/v5-paper-forward-test-preregistration-30oct.md). BTC and ETH as one basket, on 4-hour
decisions filled on 1-minute bars, under the fund's guards, each leg at 0.519 of an even one-third share with SOL's
third idle: on two assets that is size scale 0.346 (0.519 x 2/3) of each leg's half share, so each leg's account holds
0.173 of the fund and the fund keeps 0.654 as cash no Strategy holds. Fitness only, not a new try. Backtest studies
only: no fund, Strategy, demo or paper change. Dispatched with `--pair BTC/USDT,ETH/USDT`.

## Common settings

- **Instrument:** BTC and ETH perpetuals as one basket (`market = "perp"`, long only), on the stored history.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker). Market orders only. Half the measured spread on top,
  as studies charge it.
- **Holdout:** locked. Every study's data ends 2025-10-19, so the last 365 days are never read, and `--use-holdout`
  is never passed.
- **Walk-forward:** train 730 days, test 365 days. This fits the history before 2025-10-19.
- **Risk profile:** aggressive
- **Leverage cap:** 2x
- **Fund guards:** on
- **Size scale:** 0.346
- **Count:** not a new try: a benchmark under paper's risk caps, never counted (Advisor 16 Nov).

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
