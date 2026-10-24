# Improve-loop round 0 study list (24 Oct 2026, Quant Researcher; PM confirmed the plan and said "execute it", 24 Oct)

Purpose: round 0 of the improve-and-retest loop (research/quant-researcher/improve-loop-plan-24oct.md, Advisor cleared
24 Oct): qr23 and qr21 as long and short in one Strategy each, every long leg its parent's exactly and the short leg
its mirror. qr01b's round is qr01b-v3-adx-list.md. BTC only. Backtest studies only: no fund, Strategy, demo or paper
change.

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
- **Labelled "development":** BTC round results are not out of sample (Advisor 24 Oct). qr23's long and short is
  nearly always in the market, so its funding and flip costs are read every round.

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
| 1 | qr23_ls | QR-B12 price-vs-average | 1d |
| 2 | qr21_ls | QR-B12 200-day trend | 1d |

Row 1 is configs/definitions/variants/qr23_vma_band_sma50_ls.toml and row 2 qr21_trend_sma200_ls.toml
(sleeve_fund/research/variants.py, QR23_LS and QR21_LS). No PASS or "edge found" wording until the Advisor clears it.
