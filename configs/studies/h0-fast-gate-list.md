# H0 fast-path gate: H0's base re-run on the 4-rung study, unguarded, not a try (18 Nov 2026, QD; Advisor ruling 17 Nov)

Purpose: the Advisor's gate before the study-speed change (#365) counts. H0 (configs/definitions/variants/
qrh0_trend_pullback.toml, hash 2161c99b4cf29d08) re-run once on main after #365, so its base run's trades can be
compared byte for byte with the 17 Nov run (20261117-165205), and its base net and its 0% and 0.02% rungs to the cent.
**Diagnostic only, never a try.** Backtest studies only: no fund, Strategy, demo or paper change. Dispatched with
`--pair BTC/USDT,ETH/USDT,SOL/USDT`.

Why "Size scale: 1": a size scale is how a list marks a confirming run that records no try (basket_run.py). With fund
guards off and a scale of 1 every leg's account holds its full even third and the fund keeps no idle cash, so the run's
legs, sizes and book are those of H0's unguarded verdict run (basket._run: share 1, nothing unallocated, each leg sized
off its own account either way). A run that records no try is never screened (#365), so it runs whole: base, the 2x
fees run, the best asset left out, the draws, and the four rungs.

## Common settings

- **Instrument:** BTC, ETH and SOL perpetuals as one basket (`market = "perp"`, long and short), on the stored history.
  No other asset.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker), ladder to 2x. Market orders only. Half the measured spread
  on top, as studies charge it.
- **Holdout:** sealed, 2026-05-01 to 2026-10-30. `--use-holdout` is never passed, and the runner refuses it for a
  confirming run.
- **Walk-forward:** train 366 days (2024, fits nothing: settings are fixed), test 485 days (2025-01-01 to 2026-04-30).
- **Warm-up:** inside the train year, as H0.
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded).
- **Fund guards:** off
- **Size scale:** 1
- **Count:** not a new try: a confirming run of H0's settings, already counted, for the study-speed gate only.

## Pinned dates

Every row reads these days only, whatever day it runs (B-37). UTC calendar days; the decision candles close on them.

- **Data start:** 2024-01-01
- **Research end:** 2026-04-30
- **Holdout:** 2026-05-01 to 2026-10-30

| Fold | Train start | Test start | Test end |
| --- | --- | --- | --- |
| 1 | 2024-01-01 | 2025-01-01 | 2026-04-30 |

## Studies (definition file in configs/definitions/variants/ on main, by its pre-registered name)

| # | Definition | Idea family | Decision candle |
|---|---|---|---|
| 1 | qrh0_trend_pullback | H, pullback inside a trend (H0, the gate re-run) | 4h |

## Checks

1. The dry run's Tries line must read "not new tries"; if it shows a count, nothing runs.
2. Gate (Advisor 17 Nov): the base run's trade file byte-identical to 20261117-165205's; the base net, and the 0% and
   0.02% rungs, equal to the cent. The sheet's pooled break-even line names the real-fee rung ("base reused").
