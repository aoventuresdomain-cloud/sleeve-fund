# QR01b V3 ADX study list (24 Oct 2026, Quant Researcher; Advisor cleared, PM chose "Test it" 18:03 UK)

Purpose: qr01b's V3, "skip strong-trend entries": the multi-speed trend's long and short legs in one Strategy, each
with one change, no new entry while Wilder's ADX(14) on daily candles is above 25 (read from the last completed daily
close; no filter while it has no value). Long and short in one definition by the PM's change of 24 Oct. BTC only on
this list: BTC is reported but does not decide, because the idea came from BTC's own trades. One try in all.
Pre-registration: research/quant-researcher/qr01b-v3-adx-preregistration.md. Backtest studies only: no fund,
Strategy, demo or paper change.

## Common settings

- **Instrument:** BTC perpetual (`market = "perp"`, shorts allowed), on the stored history.
- **Fees per side:** 0% and 0.055% (VIP 0 perpetual taker). Market orders only. Half the measured spread on top,
  as studies charge it.
- **Holdout:** locked. Every study's data ends 2025-10-19, so the last 365 days are never read, and `--use-holdout`
  is never passed.
- **Walk-forward:** train 730 days, test 365 days. This fits the history before 2025-10-19.
- **Risk:** risk profile "none" (research: 1x, uncapped and unguarded), so signals are judged at fixed margin 1x.
- **Count:** the row is one pre-registered setting, so one try for the qr01b family in the trials register.
- **Judged against the parents** (qr01b_trend_multispeed_long and qr01b_trend_multispeed_short) out of sample, as the
  pre-registration says.

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
| 1 | v3_adx | QR-B2 multi-speed trend | 4h |

Row 1 is configs/definitions/variants/qr01b_trend_multispeed_ls_v3_adx.toml (sleeve_fund/research/variants.py,
V3_ADX). No PASS or "edge found" wording until PR E and BH-102 are live.
