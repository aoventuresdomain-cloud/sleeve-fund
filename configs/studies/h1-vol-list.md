# H improve row: H1 + vol_target 0.40/30d, family B try 30 (18 Nov 2026, HoE; Advisor ruling 16:41 UK)

One H improve row: H1 (qrh1_trend_pullback_ts6, time stop 10 to 6 candles), its definition unchanged, sized to 40% a
year of 30-day volatility on the study list. It is an H-line row, not a new family B design. Backtest study only: no
fund, Strategy, demo or paper change. Dispatched with `--pair BTC/USDT,ETH/USDT,SOL/USDT`, dry run first.

Its basket key carries the params, so the sizing makes it a new variant of H1's idea, never a re-run of H1's
registered try. The sheet is read with H's keep rule against H1: return a year and return over worst drop both rise,
and random entry doesn't worsen; the best asset left out and the run without SOL are read before any seal decision.
The independence rule of the H improve list holds: the test window can only kill an H variant, never validate one.

## Common settings

- **Instrument:** BTC, ETH and SOL perpetuals as one basket (`market = "perp"`, long and short), on the stored history.
  No other asset.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker), ladder to 2x. Market orders only. Half the measured spread
  on top, as studies charge it.
- **Sizing:** `vol_target_size = { annual = 0.40, lookback_days = 30 }` (variants.py, QRH1_TREND_PULLBACK_TS6_VOL):
  lag-1 daily volatility, capped at 1x, set at entry. The sheet prints the target and each asset's median size and
  the share of its entries the target didn't cut.
- **Holdout:** sealed, 2026-05-01 to 2026-10-30. `--use-holdout` is never passed on this list.
- **Walk-forward:** train 366 days (2024, fits nothing: settings are fixed), test 485 days (2025-01-01 to 2026-04-30).
- **Warm-up:** inside the train year, as H0 (201 days for the daily SMA(200) gate).
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded).
- **Fund guards:** off
- **Count:** family B try 30 by the Advisor's slot count. Prior 24 = slot 30 - 1 - 5 confirmed H0-idea rows (H0-H4);
  H5/H8 unconfirmed. The register adds every row of H0's idea that has run (H0 to H5 and H8 share it), so if H5 or H8
  has run the sheet prints 31 or 32: accepted, never adjusted (Advisor). The sheet shows the stated and register
  counts. The deflated Sharpe uses the larger of stated and register.
- **Round:** h1-vol
- **Budget per family:** 6 tries
- **Family B:** rows 1
- **Prior tries, family B:** 24

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
| --- | --- | --- | --- |
| 1 | qrh1_trend_pullback_ts6_vol | B, H improve row: H1 + vol_target 0.40/30d | 4h |
