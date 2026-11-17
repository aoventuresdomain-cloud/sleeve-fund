# Family H improve items: trend pullback, each change alone against H0 (17 Nov 2026, HoE; Advisor ruling 16 Nov ~22:10 UK)

The PM reopened family H after its round-0 kill (16 Nov 21:06 UTC) and asked for each idea to be tried one by one (16 Nov
21:13 UTC). The Advisor cleared items 1 to 5, 7, 8 and P1, each one change from H0 (hash 2161c99b4cf29d08), each a
counted family B try (research/quant-researcher/next-round/h1-improve-round-time-stop-16nov.md). Item 6 (8-hour candles)
is dropped (studies refuse 8-hour bars; Advisor 17 Nov); item 9 (the combination) is built after these by the fixed rule. The
rows run back to back in one dispatch, dry run first. Backtest studies only: no fund, Strategy, demo or paper change.
Dispatched with `--pair BTC/USDT,ETH/USDT,SOL/USDT`.

Independence rule (Advisor 16 Nov): the changes were chosen after seeing the test months, so the test window can only
kill an H variant, never validate one. A variant that clears the test-window bar is "candidate (test window not
independent)". Validation is on the sealed months, opened once by one frozen H version under the seal rule.

## Common settings

- **Instrument:** BTC, ETH and SOL perpetuals as one basket (`market = "perp"`, long and short), on the stored history.
  No other asset.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker), ladder to 2x. Market orders only. Half the measured spread
  on top, as studies charge it.
- **Holdout:** sealed, 2026-05-01 to 2026-10-30. `--use-holdout` is never passed on this list.
- **Walk-forward:** train 366 days (2024, fits nothing: settings are fixed), test 485 days (2025-01-01 to 2026-04-30).
- **Warm-up:** inside the train year, as H0 (201 days for the daily SMA(200) gate; row 4 needs less).
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded).
- **Fund guards:** off
- **Count:** each row is a counted family B try. The stated prior is family B's 13 earlier reversal studies plus B0;
  the register adds H0 and every H row as it runs, so rows 1 to 8 print as family B tries 16 to 23 (item 6 is
  dropped; the combination is try 24). The deflated Sharpe uses the larger N (Advisor: max of stated and
  register).
- **Round:** h-improve
- **Budget per family:** 12 tries
- **Family B:** rows 1-8
- **Prior tries, family B:** 14

## Pinned dates

Every row reads these days only, whatever day it runs (B-37). UTC calendar days; the decision candles close on them.

- **Data start:** 2024-01-01
- **Research end:** 2026-04-30
- **Holdout:** 2026-05-01 to 2026-10-30

| Fold | Train start | Test start | Test end |
| --- | --- | --- | --- |
| 1 | 2024-01-01 | 2025-01-01 | 2026-04-30 |

## Studies (definitions in configs/definitions/variants/, each one change from qrh0_trend_pullback)

| # | Definition | Idea family | Decision candle |
|---|---|---|---|
| 1 | qrh1_trend_pullback_ts6 | H item 1, time stop 10 to 6 candles | 4h |
| 2 | qrh2_trend_pullback_runners | H item 2, exit RSI(2) 90/10 or SMA10 (added by the PM after seeing H0's trades) | 4h |
| 3 | qrh3_trend_pullback_stop15 | H item 3, 1.5 ATR stop (added by the PM after seeing H0's trades) | 4h |
| 4 | qrh4_trend_pullback_sma100 | H item 4, daily SMA100 gate | 4h |
| 5 | qrh5_trend_pullback_stop35 | H item 5, 3.5 ATR stop | 4h |
| 6 | qrh7_trend_pullback_short_only | H item 7, short only (the winning side picked after the fact) | 4h |
| 7 | qrh8_trend_pullback_rsi595 | H item 8, RSI(2) 5/95 entries | 4h |
| 8 | qrhp1_trend_pullback_break_filter | H item P1, break filter RSI(2) 2 to 98 | 4h |

Each sheet carries the PM's risk/reward lines, gross and net: win rate, average win, average loss, break-even win rate,
and exits by reason. Keep rule, against H0: return a year and return over worst drop both rise, and random entry doesn't
worsen. No PASS or "edge found" wording.
