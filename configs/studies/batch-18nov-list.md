# Idea batch S2 and S3: family B tries 28 and 29 (18 Nov 2026, HoE; Advisor ruled after the 2024 signal count)

The two rows the Advisor cleared from the Researcher's batch (research/quant-researcher/next-round/idea-batch-18nov.md)
after the train-window count (qrs-batch-count-list.md, no try): S2 and S3 at its main 2 ATR threshold. S1 and S4 were
dropped with no try, and S3's fallback is not run. Each definition is the Researcher's frozen file (raw sha256 pinned in
tests/test_qrs_batch_count.py), sized by the study list only. The rows run back to back in one dispatch, dry run
first. Backtest studies only: no fund, Strategy, demo or paper change. Dispatched with
`--pair BTC/USDT,ETH/USDT,SOL/USDT`.

## Common settings

- **Instrument:** BTC, ETH and SOL perpetuals as one basket (`market = "perp"`, long and short), on the stored history.
  No other asset.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker), ladder to 2x. Market orders only. Half the measured spread
  on top, as studies charge it.
- **Sizing:** `vol_target_size = { annual = 0.40, lookback_days = 30 }` on both rows (variants.py): lag-1 daily
  volatility, capped at 1x, set at entry. Each sheet prints the target and each asset's median size and the share of
  its entries the target didn't cut.
- **Holdout:** sealed, 2026-05-01 to 2026-10-30. `--use-holdout` is never passed on this list.
- **Walk-forward:** train 366 days (2024, fits nothing: settings are fixed), test 485 days (2025-01-01 to 2026-04-30).
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded).
- **Fund guards:** off
- **Count:** row 1 is family B try 28 and row 2 is try 29 (Advisor 18 Nov). The stated prior is 27: family B's 26
  earlier tries and slot 27, which S1 held when the batch was cleared as tries 27 to 30 and which was dropped with no
  run. The deflated Sharpe uses the larger of stated and register.
- **Round:** batch-18nov
- **Budget per family:** 6 tries
- **Family B:** rows 1-2
- **Prior tries, family B:** 27

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
| 1 | qrs2_london_asia_range_fade | B, S2: London fade of the Asia range | 4h |
| 2 | qrs3_shock_fade_4h | B, S3: shock fade at 2 ATR | 4h |

The sheets print each row's definition hash and its volatility target line. No PASS or "edge found" wording in the
report beyond the sheet's own verdict.
