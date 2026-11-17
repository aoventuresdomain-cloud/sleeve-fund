# Idea batch N1 to N4: four new ideas on daily signals, 4h candles (17 Nov 2026, HoE; Advisor cleared 17 Nov ~08:15 UK)

Four new ideas from the Researcher's batch (research/quant-researcher/next-round/idea-batch-17nov.md), cleared by the
Advisor with pre-run implementation fixes, no try spent (the hash record is in that file). Each decides at the 4h candle
closing 00:00 UTC on daily blocks (N3: Monday 00:00 only) and fills on 1-minute bars. The rows run back to back in one
dispatch, dry run first. Backtest studies only: no fund, Strategy, demo or paper change. Dispatched with
`--pair BTC/USDT,ETH/USDT,SOL/USDT`.

## Common settings

- **Instrument:** BTC, ETH and SOL perpetuals as one basket (`market = "perp"`), on the stored history. No other asset.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker), ladder to 2x. Market orders only. Half the measured spread
  on top, as studies charge it.
- **Holdout:** sealed, 2026-05-01 to 2026-10-30. `--use-holdout` is never passed on this list.
- **Walk-forward:** train 366 days (2024, fits nothing: settings are fixed), test 485 days (2025-01-01 to 2026-04-30).
- **Warm-up:** inside the train year. Every daily ATR is ATR(14), which settles in 280 daily candles (281 with the
  lag-1 trigger ATR); the longest simple window is the daily SMA(200).
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded).
- **Fund guards:** off
- **Count:** row 1 is family B N 25; rows 2 and 4 are trend N 44 and 45; row 3 is family E N 2 (Advisor 17 Nov). The
  stated priors below give those numbers when the register holds none of these ideas; the deflated Sharpe uses the
  larger of stated and register.
- **Round:** batch-17nov
- **Budget per family:** 6 tries
- **Family B:** rows 1
- **Family trend:** rows 2, 4
- **Family E:** rows 3
- **Prior tries, family B:** 24
- **Prior tries, family trend:** 43
- **Prior tries, family E:** 1

## Pinned dates

Every row reads these days only, whatever day it runs (B-37). UTC calendar days; the decision candles close on them.

- **Data start:** 2024-01-01
- **Research end:** 2026-04-30
- **Holdout:** 2026-05-01 to 2026-10-30

| Fold | Train start | Test start | Test end |
| --- | --- | --- | --- |
| 1 | 2024-01-01 | 2025-01-01 | 2026-04-30 |

## Studies (definitions in configs/definitions/variants/)

| # | Definition | Idea family | Decision candle |
|---|---|---|---|
| 1 | qrn1_capitulation_reversal | B, N1: overlaps qr10 (BTC long-only capitulation, 2020-23 already seen); range not body, SMA200 regime, both sides | 4h |
| 2 | qrn2_short_trend_sleeve | trend, N2: proposed after a falling test window | 4h |
| 3 | qrn3_weekend_fade | E, N3: weekend fade | 4h |
| 4 | qrn4_wide_range_continuation | trend, N4: filtered Donchian breakout | 4h |

The sheets print each row's definition hash. The report on each row adds by hand (no code prints these): "pre-run
fixes 17 Nov, no try spent"; for rows 1 and 4, "pre-run parameter change: ATR(20) to ATR(14) for warm-up; trigger ATR
lag 1"; for row 4, its daily-return correlation to V5, computed by the Researcher from the two sheets' daily ledgers
(above 0.5 it doesn't count toward the four). No PASS or "edge found" wording.
