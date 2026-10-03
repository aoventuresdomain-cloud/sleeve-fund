# Strategy 3: staged screen (15-minute bars, core levels)

Run 2026-10-03 19:01 UTC. Development history only: the most recent 365 days are held back untouched. Short windows are smoke tests, not verdicts. Expectancy is the average net return per trade in basis points of equity. Sharpe uses daily returns of an equal-weight basket (per instrument in the right-hand columns). Costs are per side. Levels: prior session high and low, Asia session (00:00 to 08:00 UTC) high and low. Risk 0.5% per trade, leverage up to 4x (1x long-only).

## History

| Instrument | From | To (holdout starts) | Years |
|---|---|---|---|
| BTC/USD | 2017-08-17 | 2025-10-01 | 8.12 |
| ETH/USD | 2017-08-17 | 2025-10-01 | 8.12 |
| SOL/USD | 2020-08-11 | 2025-10-01 | 5.14 |
| SUI/USD | 2023-05-03 | 2025-10-01 | 2.41 |
| XRP/USD | 2018-05-04 | 2025-10-01 | 7.41 |

## long and short, kraken pro taker costs (40 bp fee + 2 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | BTC Sharpe | ETH Sharpe | SOL Sharpe | SUI Sharpe | XRP Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 sweep reversal | 1 week | 11 | -41 | 27% | -21.51 | -0.9% | -0.6% | -9.95 | -11.05 | -5.28 | -6.05 | -7.36 | no |
| 3 sweep reversal | 1 month | 58 | -36 | 17% | -20.29 | -4.1% | -4.1% | -12.51 | -10.15 | -7.97 | -5.78 | -8.75 | no |
| 3 sweep reversal | 3 months | 205 | -31 | 25% | -17.50 | -12.0% | -12.0% | -11.54 | -10.89 | -7.93 | -5.25 | -5.36 | no |
| 3 sweep reversal | 6 months | 411 | -29 | 26% | -17.30 | -21.3% | -21.0% | -11.40 | -9.17 | -6.30 | -5.02 | -6.93 | no |
| 3 sweep reversal | 1 year | 801 | -27 | 26% | -16.16 | -35.6% | -35.5% | -10.56 | -8.43 | -6.96 | -5.30 | -6.38 | no |
| 3 sweep reversal | 2 years | 1682 | -27 | 26% | -15.66 | -59.5% | -59.4% | -10.35 | -8.83 | -6.84 | -4.06 | -7.45 | no |
| 3 sweep reversal | all development history | 4951 | -25 | 25% | -12.41 | -92.0% | -92.0% | -9.39 | -7.30 | -5.63 | -4.39 | -6.65 | no |
| 3 sweep reversal, no trend filter | 1 week | 11 | -41 | 27% | -21.51 | -0.9% | -0.6% | -9.95 | -11.05 | -5.28 | -6.05 | -7.36 | no |
| 3 sweep reversal, no trend filter | 1 month | 59 | -36 | 17% | -20.43 | -4.1% | -4.2% | -12.51 | -10.15 | -8.21 | -5.78 | -8.75 | no |
| 3 sweep reversal, no trend filter | 3 months | 206 | -31 | 25% | -17.54 | -12.0% | -12.0% | -11.54 | -10.89 | -8.01 | -5.25 | -5.36 | no |
| 3 sweep reversal, no trend filter | 6 months | 412 | -29 | 26% | -17.32 | -21.3% | -21.1% | -11.40 | -9.17 | -6.34 | -5.02 | -6.93 | no |
| 3 sweep reversal, no trend filter | 1 year | 805 | -27 | 26% | -16.16 | -35.6% | -35.5% | -10.56 | -8.44 | -6.95 | -5.30 | -6.38 | no |
| 3 sweep reversal, no trend filter | 2 years | 1711 | -27 | 27% | -15.65 | -59.7% | -59.6% | -10.31 | -8.83 | -6.84 | -4.10 | -7.47 | no |
| 3 sweep reversal, no trend filter | all development history | 5028 | -25 | 25% | -12.45 | -92.2% | -92.2% | -9.42 | -7.33 | -5.63 | -4.44 | -6.71 | no |
| bench: breakout of the same levels | 1 week | 10 | -29 | 30% | -15.34 | -0.6% | -0.6% | -2.40 | 11.14 | -11.17 | -8.59 | -10.92 | no |
| bench: breakout of the same levels | 1 month | 56 | -47 | 23% | -16.37 | -5.1% | -4.8% | -11.52 | -4.96 | -8.95 | -6.16 | -4.40 | no |
| bench: breakout of the same levels | 3 months | 197 | -33 | 29% | -11.48 | -12.3% | -12.1% | -9.55 | -2.55 | -4.48 | -5.17 | -5.99 | no |
| bench: breakout of the same levels | 6 months | 388 | -29 | 30% | -9.20 | -20.3% | -20.3% | -8.12 | -2.83 | -5.64 | -3.97 | -3.77 | no |
| bench: breakout of the same levels | 1 year | 758 | -26 | 30% | -8.10 | -32.7% | -32.6% | -7.58 | -2.86 | -4.89 | -3.30 | -2.66 | no |
| bench: breakout of the same levels | 2 years | 1607 | -29 | 27% | -9.52 | -60.9% | -60.8% | -7.49 | -3.61 | -4.82 | -4.48 | -4.83 | no |
| bench: breakout of the same levels | all development history | 4749 | -26 | 27% | -7.60 | -91.7% | -91.7% | -5.96 | -3.83 | -4.34 | -4.33 | -3.74 | no |

## long and short, high volume taker costs (10 bp fee + 2 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | BTC Sharpe | ETH Sharpe | SOL Sharpe | SUI Sharpe | XRP Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 sweep reversal | 1 week | 11 | -18 | 36% | -10.30 | -0.4% | -0.2% | -7.26 | -11.17 | -2.27 | 7.15 | -2.21 | no |
| 3 sweep reversal | 1 month | 58 | -9 | 45% | -6.69 | -1.1% | -1.3% | -1.81 | -6.71 | -4.82 | -1.39 | -1.67 | no |
| 3 sweep reversal | 3 months | 205 | -8 | 47% | -5.40 | -3.1% | -3.3% | -3.16 | -5.44 | -3.04 | -1.23 | 0.59 | no |
| 3 sweep reversal | 6 months | 411 | -8 | 46% | -5.50 | -6.0% | -6.0% | -4.86 | -4.32 | -1.22 | -1.23 | -1.38 | no |
| 3 sweep reversal | 1 year | 801 | -8 | 45% | -6.22 | -12.6% | -12.9% | -4.32 | -3.94 | -2.65 | -1.88 | -1.62 | no |
| 3 sweep reversal | 2 years | 1682 | -8 | 46% | -5.48 | -22.4% | -22.4% | -4.44 | -3.92 | -2.70 | -0.25 | -2.01 | no |
| 3 sweep reversal | all development history | 4951 | -7 | 46% | -4.63 | -51.5% | -51.6% | -4.01 | -2.79 | -1.78 | -0.66 | -2.01 | no |
| 3 sweep reversal, no trend filter | 1 week | 11 | -18 | 36% | -10.30 | -0.4% | -0.2% | -7.26 | -11.17 | -2.27 | 7.15 | -2.21 | no |
| 3 sweep reversal, no trend filter | 1 month | 59 | -9 | 46% | -6.66 | -1.1% | -1.3% | -1.81 | -6.71 | -4.77 | -1.39 | -1.67 | no |
| 3 sweep reversal, no trend filter | 3 months | 206 | -8 | 47% | -5.39 | -3.1% | -3.3% | -3.16 | -5.44 | -3.02 | -1.23 | 0.59 | no |
| 3 sweep reversal, no trend filter | 6 months | 412 | -8 | 46% | -5.50 | -6.0% | -6.0% | -4.86 | -4.32 | -1.21 | -1.23 | -1.38 | no |
| 3 sweep reversal, no trend filter | 1 year | 805 | -8 | 45% | -6.16 | -12.5% | -12.8% | -4.32 | -3.93 | -2.54 | -1.88 | -1.62 | no |
| 3 sweep reversal, no trend filter | 2 years | 1711 | -7 | 47% | -5.37 | -22.0% | -22.0% | -4.39 | -3.82 | -2.58 | -0.23 | -1.98 | no |
| 3 sweep reversal, no trend filter | all development history | 5028 | -7 | 46% | -4.61 | -51.6% | -51.8% | -4.00 | -2.78 | -1.76 | -0.65 | -2.03 | no |
| bench: breakout of the same levels | 1 week | 10 | -6 | 40% | -7.22 | -0.1% | -0.1% | 4.16 | 10.99 | -11.18 | -6.20 | -10.61 | no |
| bench: breakout of the same levels | 1 month | 56 | -20 | 32% | -9.34 | -2.2% | -2.0% | -7.97 | -1.16 | -6.50 | -4.27 | -0.81 | no |
| bench: breakout of the same levels | 3 months | 197 | -10 | 36% | -3.82 | -3.8% | -4.4% | -4.99 | 1.49 | -1.21 | -2.92 | -2.90 | no |
| bench: breakout of the same levels | 6 months | 388 | -7 | 36% | -2.63 | -5.7% | -6.2% | -3.36 | 0.51 | -2.40 | -1.65 | -0.68 | no |
| bench: breakout of the same levels | 1 year | 758 | -7 | 36% | -2.35 | -10.0% | -9.9% | -3.26 | 0.15 | -1.92 | -1.27 | -0.15 | no |
| bench: breakout of the same levels | 2 years | 1607 | -10 | 34% | -3.51 | -27.2% | -27.0% | -3.00 | -0.52 | -1.88 | -2.45 | -1.74 | no |
| bench: breakout of the same levels | all development history | 4749 | -8 | 35% | -2.55 | -53.7% | -53.8% | -1.97 | -0.94 | -1.98 | -2.25 | -1.07 | no |

## long and short, institutional costs (2 bp fee + 1 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | BTC Sharpe | ETH Sharpe | SOL Sharpe | SUI Sharpe | XRP Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 sweep reversal | 1 week | 11 | -11 | 55% | -6.19 | -0.2% | -0.2% | -5.33 | -11.19 | -1.14 | 9.06 | 1.12 | no |
| 3 sweep reversal | 1 month | 58 | -1 | 59% | -0.97 | -0.2% | -0.6% | 3.51 | -4.52 | -3.54 | 0.38 | 1.68 | no |
| 3 sweep reversal | 3 months | 205 | -1 | 56% | -0.39 | -0.2% | -0.9% | 1.41 | -3.05 | -1.26 | 0.19 | 2.56 | no |
| 3 sweep reversal | 6 months | 411 | -1 | 53% | -0.78 | -0.9% | -2.1% | -0.93 | -2.23 | 0.47 | 0.09 | 0.57 | no |
| 3 sweep reversal | 1 year | 801 | -3 | 52% | -2.01 | -4.3% | -5.4% | -0.99 | -2.04 | -1.06 | -0.68 | 0.10 | no |
| 3 sweep reversal | 2 years | 1682 | -2 | 53% | -1.26 | -5.7% | -7.0% | -1.25 | -1.69 | -1.18 | 0.98 | 0.09 | no |
| 3 sweep reversal | all development history | 4951 | -2 | 53% | -1.20 | -16.9% | -18.0% | -1.33 | -0.90 | -0.43 | 0.58 | -0.27 | no |
| 3 sweep reversal, no trend filter | 1 week | 11 | -11 | 55% | -6.19 | -0.2% | -0.2% | -5.33 | -11.19 | -1.14 | 9.06 | 1.12 | no |
| 3 sweep reversal, no trend filter | 1 month | 59 | -1 | 59% | -0.90 | -0.1% | -0.6% | 3.51 | -4.52 | -3.38 | 0.38 | 1.68 | no |
| 3 sweep reversal, no trend filter | 3 months | 206 | -1 | 56% | -0.37 | -0.2% | -0.9% | 1.41 | -3.05 | -1.22 | 0.19 | 2.56 | no |
| 3 sweep reversal, no trend filter | 6 months | 412 | -1 | 53% | -0.77 | -0.9% | -2.1% | -0.93 | -2.23 | 0.49 | 0.09 | 0.57 | no |
| 3 sweep reversal, no trend filter | 1 year | 805 | -3 | 52% | -1.95 | -4.1% | -5.3% | -0.99 | -2.03 | -0.93 | -0.68 | 0.10 | no |
| 3 sweep reversal, no trend filter | 2 years | 1711 | -1 | 53% | -1.10 | -5.0% | -6.5% | -1.19 | -1.56 | -1.04 | 1.01 | 0.13 | no |
| 3 sweep reversal, no trend filter | all development history | 5028 | -2 | 53% | -1.17 | -16.5% | -17.7% | -1.30 | -0.87 | -0.40 | 0.61 | -0.28 | no |
| bench: breakout of the same levels | 1 week | 10 | +1 | 40% | 1.81 | +0.0% | -0.1% | 5.71 | 10.94 | -11.19 | -5.13 | -10.41 | no |
| bench: breakout of the same levels | 1 month | 56 | -12 | 34% | -5.87 | -1.3% | -1.2% | -5.62 | 0.25 | -5.36 | -3.43 | 0.32 | no |
| bench: breakout of the same levels | 3 months | 197 | -3 | 37% | -1.04 | -1.0% | -2.7% | -2.79 | 2.66 | -0.10 | -2.10 | -1.84 | no |
| bench: breakout of the same levels | 6 months | 388 | -1 | 39% | -0.34 | -0.8% | -2.8% | -1.23 | 1.51 | -1.30 | -0.87 | 0.28 | no |
| bench: breakout of the same levels | 1 year | 758 | -1 | 38% | -0.40 | -1.8% | -3.5% | -1.54 | 1.07 | -0.94 | -0.62 | 0.63 | no |
| bench: breakout of the same levels | 2 years | 1607 | -4 | 36% | -1.44 | -12.3% | -13.7% | -1.25 | 0.45 | -0.91 | -1.79 | -0.70 | no |
| bench: breakout of the same levels | all development history | 4749 | -3 | 38% | -0.84 | -22.6% | -24.4% | -0.56 | -0.00 | -1.21 | -1.57 | -0.22 | no |

## long only (spot), kraken pro taker costs (40 bp fee + 2 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | BTC Sharpe | ETH Sharpe | SOL Sharpe | SUI Sharpe | XRP Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 sweep reversal | 1 week | 7 | -35 | 43% | -10.65 | -0.5% | -0.1% | -7.22 | -7.22 | 7.22 | -6.05 | -7.22 | no |
| 3 sweep reversal | 1 month | 34 | -38 | 18% | -13.79 | -2.6% | -2.6% | -8.79 | -8.12 | -5.54 | -4.36 | -7.69 | no |
| 3 sweep reversal | 3 months | 115 | -32 | 25% | -12.11 | -7.1% | -7.1% | -9.31 | -7.27 | -5.96 | -4.55 | -4.37 | no |
| 3 sweep reversal | 6 months | 225 | -29 | 24% | -11.28 | -12.3% | -12.1% | -8.21 | -6.81 | -4.48 | -4.30 | -5.04 | no |
| 3 sweep reversal | 1 year | 429 | -28 | 25% | -10.78 | -21.6% | -21.5% | -7.77 | -6.07 | -4.93 | -4.47 | -4.84 | no |
| 3 sweep reversal | 2 years | 889 | -27 | 27% | -10.06 | -38.2% | -38.0% | -7.09 | -6.42 | -4.77 | -3.01 | -4.95 | no |
| 3 sweep reversal | all development history | 2716 | -25 | 26% | -8.38 | -73.9% | -73.9% | -6.81 | -5.18 | -3.81 | -3.30 | -4.58 | no |
| 3 sweep reversal, no trend filter | 1 week | 7 | -35 | 43% | -10.65 | -0.5% | -0.1% | -7.22 | -7.22 | 7.22 | -6.05 | -7.22 | no |
| 3 sweep reversal, no trend filter | 1 month | 35 | -37 | 17% | -13.89 | -2.6% | -2.6% | -8.79 | -8.12 | -5.81 | -4.36 | -7.69 | no |
| 3 sweep reversal, no trend filter | 3 months | 116 | -32 | 25% | -12.14 | -7.1% | -7.1% | -9.31 | -7.27 | -6.07 | -4.55 | -4.37 | no |
| 3 sweep reversal, no trend filter | 6 months | 226 | -29 | 24% | -11.30 | -12.3% | -12.1% | -8.21 | -6.81 | -4.53 | -4.30 | -5.04 | no |
| 3 sweep reversal, no trend filter | 1 year | 431 | -28 | 25% | -10.79 | -21.6% | -21.5% | -7.77 | -6.08 | -4.96 | -4.47 | -4.84 | no |
| 3 sweep reversal, no trend filter | 2 years | 906 | -27 | 27% | -10.18 | -38.5% | -38.4% | -7.11 | -6.52 | -4.85 | -3.06 | -4.96 | no |
| 3 sweep reversal, no trend filter | all development history | 2756 | -25 | 26% | -8.44 | -74.3% | -74.3% | -6.85 | -5.22 | -3.88 | -3.34 | -4.62 | no |
| bench: breakout of the same levels | 1 week | 4 | -36 | 25% | -13.65 | -0.3% | -0.3% | -7.22 | 7.22 | -7.22 | 0.00 | -7.22 | no |
| bench: breakout of the same levels | 1 month | 24 | -48 | 21% | -11.10 | -2.3% | -2.1% | -6.76 | -2.44 | -5.82 | -5.16 | -7.59 | no |
| bench: breakout of the same levels | 3 months | 88 | -27 | 31% | -6.01 | -4.7% | -5.2% | -5.07 | -0.62 | -2.50 | -4.91 | -3.95 | no |
| bench: breakout of the same levels | 6 months | 179 | -24 | 31% | -5.22 | -8.3% | -9.1% | -4.34 | -1.75 | -3.28 | -3.18 | -1.32 | no |
| bench: breakout of the same levels | 1 year | 366 | -26 | 29% | -5.62 | -17.2% | -17.1% | -4.73 | -1.96 | -2.62 | -4.78 | -1.16 | no |
| bench: breakout of the same levels | 2 years | 780 | -27 | 28% | -6.00 | -34.3% | -34.2% | -4.95 | -1.95 | -2.36 | -3.19 | -2.69 | no |
| bench: breakout of the same levels | all development history | 2219 | -25 | 28% | -4.96 | -66.7% | -66.7% | -4.05 | -2.38 | -2.55 | -3.05 | -1.84 | no |

## long only (spot), high volume taker costs (10 bp fee + 2 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | BTC Sharpe | ETH Sharpe | SOL Sharpe | SUI Sharpe | XRP Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 sweep reversal | 1 week | 7 | -14 | 43% | -5.82 | -0.2% | -0.0% | -7.22 | -7.22 | 7.22 | 7.15 | -7.22 | no |
| 3 sweep reversal | 1 month | 34 | -12 | 50% | -5.51 | -0.8% | -0.9% | -2.11 | -5.79 | -3.47 | -0.09 | -4.59 | no |
| 3 sweep reversal | 3 months | 115 | -9 | 47% | -4.20 | -2.1% | -2.1% | -3.51 | -3.98 | -3.21 | -1.30 | 1.02 | no |
| 3 sweep reversal | 6 months | 225 | -8 | 46% | -3.64 | -3.5% | -3.6% | -3.49 | -3.72 | -1.33 | -1.26 | 0.03 | no |
| 3 sweep reversal | 1 year | 429 | -10 | 42% | -4.57 | -8.0% | -8.0% | -3.53 | -3.34 | -2.02 | -2.01 | -1.19 | no |
| 3 sweep reversal | 2 years | 889 | -8 | 45% | -3.80 | -13.5% | -13.6% | -3.26 | -3.30 | -2.21 | -0.27 | -1.03 | no |
| 3 sweep reversal | all development history | 2716 | -7 | 47% | -3.00 | -31.6% | -31.8% | -2.86 | -2.04 | -1.28 | -0.66 | -1.02 | no |
| 3 sweep reversal, no trend filter | 1 week | 7 | -14 | 43% | -5.82 | -0.2% | -0.0% | -7.22 | -7.22 | 7.22 | 7.15 | -7.22 | no |
| 3 sweep reversal, no trend filter | 1 month | 35 | -11 | 51% | -5.47 | -0.8% | -0.9% | -2.11 | -5.79 | -3.40 | -0.09 | -4.59 | no |
| 3 sweep reversal, no trend filter | 3 months | 116 | -9 | 47% | -4.19 | -2.1% | -2.1% | -3.51 | -3.98 | -3.18 | -1.30 | 1.02 | no |
| 3 sweep reversal, no trend filter | 6 months | 226 | -8 | 46% | -3.64 | -3.5% | -3.6% | -3.49 | -3.72 | -1.32 | -1.26 | 0.03 | no |
| 3 sweep reversal, no trend filter | 1 year | 431 | -10 | 42% | -4.56 | -8.0% | -8.0% | -3.53 | -3.34 | -2.02 | -2.01 | -1.19 | no |
| 3 sweep reversal, no trend filter | 2 years | 906 | -8 | 46% | -3.84 | -13.6% | -13.7% | -3.24 | -3.42 | -2.23 | -0.25 | -1.00 | no |
| 3 sweep reversal, no trend filter | all development history | 2756 | -7 | 47% | -3.04 | -32.2% | -32.4% | -2.86 | -2.09 | -1.35 | -0.64 | -1.05 | no |
| bench: breakout of the same levels | 1 week | 4 | -8 | 25% | -4.26 | -0.1% | -0.1% | -7.22 | 7.22 | -7.22 | 0.00 | -7.22 | no |
| bench: breakout of the same levels | 1 month | 24 | -21 | 29% | -6.75 | -1.0% | -0.9% | -2.73 | 0.27 | -4.67 | -4.25 | -6.52 | no |
| bench: breakout of the same levels | 3 months | 88 | -3 | 38% | -0.70 | -0.5% | -2.2% | -1.50 | 1.91 | -0.01 | -4.07 | -1.71 | no |
| bench: breakout of the same levels | 6 months | 179 | -2 | 37% | -0.53 | -0.8% | -2.2% | -0.51 | 0.55 | -0.96 | -2.06 | 0.57 | no |
| bench: breakout of the same levels | 1 year | 366 | -6 | 36% | -1.49 | -4.5% | -4.5% | -1.45 | 0.14 | -0.53 | -3.74 | 0.34 | no |
| bench: breakout of the same levels | 2 years | 780 | -7 | 35% | -1.65 | -10.4% | -10.8% | -1.56 | 0.03 | -0.29 | -2.06 | -0.62 | no |
| bench: breakout of the same levels | all development history | 2219 | -6 | 36% | -1.29 | -23.8% | -24.2% | -1.20 | -0.38 | -0.84 | -1.86 | -0.20 | no |

## long only (spot), institutional costs (2 bp fee + 1 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | BTC Sharpe | ETH Sharpe | SOL Sharpe | SUI Sharpe | XRP Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 sweep reversal | 1 week | 7 | -8 | 57% | -3.42 | -0.1% | -0.0% | -7.22 | -7.22 | 7.22 | 9.06 | -7.22 | no |
| 3 sweep reversal | 1 month | 34 | -4 | 62% | -1.76 | -0.2% | -0.5% | 1.75 | -4.56 | -2.63 | 1.61 | -1.01 | no |
| 3 sweep reversal | 3 months | 115 | -2 | 56% | -1.04 | -0.5% | -1.2% | -0.14 | -2.49 | -2.12 | -0.10 | 2.95 | no |
| 3 sweep reversal | 6 months | 225 | -1 | 53% | -0.66 | -0.6% | -1.6% | -0.76 | -2.31 | -0.26 | -0.16 | 1.85 | no |
| 3 sweep reversal | 1 year | 429 | -4 | 50% | -1.99 | -3.5% | -3.8% | -1.21 | -2.13 | -0.97 | -1.12 | 0.22 | no |
| 3 sweep reversal | 2 years | 889 | -3 | 51% | -1.19 | -4.4% | -5.2% | -1.17 | -1.84 | -1.24 | 0.62 | 0.44 | no |
| 3 sweep reversal | all development history | 2716 | -2 | 53% | -0.74 | -8.9% | -9.6% | -0.98 | -0.75 | -0.38 | 0.23 | 0.27 | no |
| 3 sweep reversal, no trend filter | 1 week | 7 | -8 | 57% | -3.42 | -0.1% | -0.0% | -7.22 | -7.22 | 7.22 | 9.06 | -7.22 | no |
| 3 sweep reversal, no trend filter | 1 month | 35 | -3 | 63% | -1.67 | -0.2% | -0.5% | 1.75 | -4.56 | -2.44 | 1.61 | -1.01 | no |
| 3 sweep reversal, no trend filter | 3 months | 116 | -2 | 56% | -1.02 | -0.5% | -1.2% | -0.14 | -2.49 | -2.05 | -0.10 | 2.95 | no |
| 3 sweep reversal, no trend filter | 6 months | 226 | -1 | 54% | -0.65 | -0.6% | -1.6% | -0.76 | -2.31 | -0.23 | -0.16 | 1.85 | no |
| 3 sweep reversal, no trend filter | 1 year | 431 | -4 | 50% | -1.98 | -3.5% | -3.8% | -1.21 | -2.12 | -0.95 | -1.12 | 0.22 | no |
| 3 sweep reversal, no trend filter | 2 years | 906 | -2 | 52% | -1.19 | -4.4% | -5.2% | -1.12 | -1.95 | -1.24 | 0.66 | 0.47 | no |
| 3 sweep reversal, no trend filter | all development history | 2756 | -2 | 53% | -0.78 | -9.3% | -10.1% | -0.97 | -0.79 | -0.45 | 0.27 | 0.24 | no |
| bench: breakout of the same levels | 1 week | 4 | +1 | 25% | 0.31 | +0.0% | -0.1% | -7.22 | 7.22 | -7.22 | 0.00 | -7.22 | no |
| bench: breakout of the same levels | 1 month | 24 | -13 | 33% | -4.39 | -0.6% | -0.6% | -0.75 | 1.24 | -4.12 | -3.78 | -5.71 | no |
| bench: breakout of the same levels | 3 months | 88 | +4 | 39% | 1.00 | +0.8% | -1.3% | -0.07 | 2.60 | 0.78 | -3.70 | -0.93 | no |
| bench: breakout of the same levels | 6 months | 179 | +4 | 40% | 1.03 | +1.5% | -1.3% | 0.84 | 1.21 | -0.19 | -1.67 | 1.13 | no |
| bench: breakout of the same levels | 1 year | 366 | -0 | 37% | -0.10 | -0.4% | -2.6% | -0.26 | 0.77 | 0.13 | -3.35 | 0.79 | no |
| bench: breakout of the same levels | 2 years | 780 | -1 | 37% | -0.24 | -1.7% | -4.8% | -0.33 | 0.63 | 0.36 | -1.69 | 0.04 | no |
| bench: breakout of the same levels | all development history | 2219 | -0 | 39% | -0.10 | -2.4% | -5.3% | -0.22 | 0.25 | -0.30 | -1.47 | 0.30 | no |
