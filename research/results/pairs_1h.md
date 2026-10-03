# Strategy 5: staged screen (hourly bars, every pair in the basket)

Run 2026-10-03 19:01 UTC. Development history only: the most recent 365 days are held back untouched. Short windows are smoke tests, not verdicts. Expectancy is the average net return per trade in basis points of equity. Sharpe uses daily returns of an equal-weight basket (per instrument in the right-hand columns). Costs are per side. Each trade risks 0.75% of equity to the z = 3.5 stop; gross leverage up to 3x; the short leg pays 5% a year. Exits act on hourly closes. Trend-regime entries are the EMA crossover itself, not yet the pullback to the spread's anchored VWAP.

## History

| Instrument | From | To (holdout starts) | Years |
|---|---|---|---|
| ETH/BTC | 2017-08-17 | 2025-10-01 | 8.12 |
| SOL/BTC | 2020-08-11 | 2025-10-01 | 5.14 |
| XRP/BTC | 2018-05-04 | 2025-10-01 | 7.41 |
| SUI/BTC | 2023-05-03 | 2025-10-01 | 2.41 |
| SOL/ETH | 2020-08-11 | 2025-10-01 | 5.14 |
| XRP/ETH | 2018-05-04 | 2025-10-01 | 7.41 |
| SUI/ETH | 2023-05-03 | 2025-10-01 | 2.41 |
| XRP/SOL | 2020-08-11 | 2025-10-01 | 5.14 |
| SUI/SOL | 2023-05-03 | 2025-10-01 | 2.41 |
| SUI/XRP | 2023-05-03 | 2025-10-01 | 2.41 |

## long and short, kraken pro taker costs (40 bp fee + 2 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | ETH Sharpe | SOL Sharpe | XRP Sharpe | SUI Sharpe | SOL Sharpe | XRP Sharpe | SUI Sharpe | XRP Sharpe | SUI Sharpe | SUI Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 regime-switching pairs | 1 week | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 1 month | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 3 months | 10 | -240 | 10% | -4.52 | -2.4% | -2.4% | 0.00 | -2.00 | 0.00 | 0.00 | 0.00 | -2.39 | 0.00 | -3.75 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 6 months | 16 | -163 | 19% | -3.36 | -2.6% | -2.6% | 0.00 | -1.42 | 0.00 | 0.00 | 0.00 | -1.50 | 0.00 | -3.09 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 1 year | 20 | -149 | 20% | -2.45 | -2.9% | -2.9% | 0.00 | 0.12 | 0.00 | 0.00 | -1.00 | -1.06 | 0.00 | -2.17 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 2 years | 22 | -137 | 23% | -1.75 | -3.0% | -3.0% | 0.00 | -0.05 | 0.00 | 0.00 | -0.71 | -0.75 | 0.00 | -1.53 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | all development history | 54 | -148 | 19% | -1.46 | -7.7% | -7.7% | -1.05 | -0.84 | 0.04 | 0.00 | -0.77 | -0.77 | 0.00 | -1.01 | 0.00 | 0.00 | no |
| 5 revert regime only | 1 week | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 revert regime only | 1 month | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 revert regime only | 3 months | 10 | -240 | 10% | -4.52 | -2.4% | -2.4% | 0.00 | -2.00 | 0.00 | 0.00 | 0.00 | -2.39 | 0.00 | -3.75 | 0.00 | 0.00 | no |
| 5 revert regime only | 6 months | 16 | -163 | 19% | -3.36 | -2.6% | -2.6% | 0.00 | -1.42 | 0.00 | 0.00 | 0.00 | -1.50 | 0.00 | -3.09 | 0.00 | 0.00 | no |
| 5 revert regime only | 1 year | 20 | -149 | 20% | -2.45 | -2.9% | -2.9% | 0.00 | 0.12 | 0.00 | 0.00 | -1.00 | -1.06 | 0.00 | -2.17 | 0.00 | 0.00 | no |
| 5 revert regime only | 2 years | 22 | -137 | 23% | -1.75 | -3.0% | -3.0% | 0.00 | -0.05 | 0.00 | 0.00 | -0.71 | -0.75 | 0.00 | -1.53 | 0.00 | 0.00 | no |
| 5 revert regime only | all development history | 54 | -148 | 19% | -1.46 | -7.7% | -7.7% | -1.05 | -0.84 | 0.04 | 0.00 | -0.77 | -0.77 | 0.00 | -1.01 | 0.00 | 0.00 | no |
| bench: static hedge, no regime switch | 1 week | 4 | -48 | 0% | -9.74 | -0.2% | -0.2% | 0.00 | -7.22 | -7.22 | -8.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| bench: static hedge, no regime switch | 1 month | 9 | -13 | 22% | -1.61 | -0.1% | -0.2% | 0.00 | -3.49 | -3.49 | 0.79 | 0.00 | 0.00 | 3.49 | 0.00 | 0.00 | -3.49 | no |
| bench: static hedge, no regime switch | 3 months | 55 | -97 | 16% | -4.75 | -5.2% | -5.2% | -1.92 | -2.15 | -2.15 | 0.67 | -2.82 | -1.59 | -2.70 | -3.37 | -3.24 | -1.38 | no |
| bench: static hedge, no regime switch | 6 months | 106 | -75 | 18% | -4.59 | -7.6% | -7.6% | -2.87 | -2.61 | -2.56 | 0.75 | -1.49 | -1.03 | -2.14 | -2.29 | -2.40 | -2.54 | no |
| bench: static hedge, no regime switch | 1 year | 261 | -66 | 22% | -4.04 | -15.7% | -15.7% | -2.65 | -0.81 | -1.70 | -0.63 | -2.16 | -2.58 | -1.01 | -3.10 | -1.82 | 0.22 | no |
| bench: static hedge, no regime switch | 2 years | 468 | -60 | 29% | -2.58 | -24.5% | -24.5% | -1.62 | -1.22 | -0.87 | 0.04 | -1.88 | -1.63 | -0.81 | -2.25 | -0.14 | -0.71 | no |
| bench: static hedge, no regime switch | all development history | 1082 | -66 | 28% | -2.37 | -51.2% | -51.2% | -1.33 | -1.30 | -1.36 | 0.04 | -1.48 | -1.56 | -0.81 | -1.76 | -0.14 | -0.71 | no |

## long and short, high volume taker costs (10 bp fee + 2 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | ETH Sharpe | SOL Sharpe | XRP Sharpe | SUI Sharpe | SOL Sharpe | XRP Sharpe | SUI Sharpe | XRP Sharpe | SUI Sharpe | SUI Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 regime-switching pairs | 1 week | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 1 month | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 3 months | 10 | -143 | 10% | -3.81 | -1.4% | -1.5% | 0.00 | -2.00 | 0.00 | 0.00 | 0.00 | -2.22 | 0.00 | -3.20 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 6 months | 16 | -85 | 25% | -2.44 | -1.3% | -1.5% | 0.00 | -1.42 | 0.00 | 0.00 | 0.00 | -1.22 | 0.00 | -2.30 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 1 year | 20 | -66 | 30% | -1.48 | -1.3% | -1.5% | 0.00 | 0.83 | 0.00 | 0.00 | -1.00 | -0.86 | 0.00 | -1.62 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 2 years | 22 | -58 | 32% | -1.01 | -1.3% | -1.5% | 0.00 | 0.58 | 0.00 | 0.00 | -0.60 | -0.61 | 0.00 | -1.14 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | all development history | 54 | -61 | 31% | -0.81 | -3.2% | -3.2% | -0.24 | -0.40 | 0.23 | 0.00 | -0.57 | -0.66 | 0.00 | -0.75 | 0.00 | 0.00 | no |
| 5 revert regime only | 1 week | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 revert regime only | 1 month | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 revert regime only | 3 months | 10 | -143 | 10% | -3.81 | -1.4% | -1.5% | 0.00 | -2.00 | 0.00 | 0.00 | 0.00 | -2.22 | 0.00 | -3.20 | 0.00 | 0.00 | no |
| 5 revert regime only | 6 months | 16 | -85 | 25% | -2.44 | -1.3% | -1.5% | 0.00 | -1.42 | 0.00 | 0.00 | 0.00 | -1.22 | 0.00 | -2.30 | 0.00 | 0.00 | no |
| 5 revert regime only | 1 year | 20 | -66 | 30% | -1.48 | -1.3% | -1.5% | 0.00 | 0.83 | 0.00 | 0.00 | -1.00 | -0.86 | 0.00 | -1.62 | 0.00 | 0.00 | no |
| 5 revert regime only | 2 years | 22 | -58 | 32% | -1.01 | -1.3% | -1.5% | 0.00 | 0.58 | 0.00 | 0.00 | -0.60 | -0.61 | 0.00 | -1.14 | 0.00 | 0.00 | no |
| 5 revert regime only | all development history | 54 | -61 | 31% | -0.81 | -3.2% | -3.2% | -0.24 | -0.40 | 0.23 | 0.00 | -0.57 | -0.66 | 0.00 | -0.75 | 0.00 | 0.00 | no |
| bench: static hedge, no regime switch | 1 week | 4 | -16 | 50% | -5.75 | -0.1% | -0.1% | 0.00 | 7.22 | -7.22 | 1.37 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| bench: static hedge, no regime switch | 1 month | 9 | +26 | 56% | 3.04 | +0.2% | -0.1% | 0.00 | 3.49 | -3.49 | 4.08 | 0.00 | 0.00 | 3.49 | 0.00 | 0.00 | -3.49 | no |
| bench: static hedge, no regime switch | 3 months | 55 | -57 | 29% | -3.51 | -3.1% | -3.3% | -1.89 | -1.94 | -1.48 | 2.57 | -2.42 | -0.82 | -1.99 | -2.78 | -3.55 | -0.37 | no |
| bench: static hedge, no regime switch | 6 months | 106 | -36 | 33% | -2.78 | -3.7% | -4.0% | -2.41 | -0.92 | -1.71 | 1.91 | -1.00 | -0.36 | -1.83 | -1.81 | -2.66 | -1.94 | no |
| bench: static hedge, no regime switch | 1 year | 261 | -32 | 32% | -2.27 | -8.0% | -8.6% | -2.08 | 0.18 | -0.65 | 0.72 | -1.83 | -2.02 | -0.16 | -2.67 | -1.46 | 0.55 | no |
| bench: static hedge, no regime switch | 2 years | 468 | -29 | 38% | -1.38 | -13.0% | -13.3% | -1.16 | -0.52 | 0.00 | 0.62 | -1.50 | -0.99 | -0.09 | -1.87 | 0.19 | -0.53 | no |
| bench: static hedge, no regime switch | all development history | 1082 | -32 | 36% | -1.27 | -29.1% | -29.3% | -0.91 | -0.51 | -0.56 | 0.62 | -0.83 | -0.83 | -0.09 | -1.15 | 0.19 | -0.53 | no |

## long and short, institutional costs (2 bp fee + 1 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | ETH Sharpe | SOL Sharpe | XRP Sharpe | SUI Sharpe | SOL Sharpe | XRP Sharpe | SUI Sharpe | XRP Sharpe | SUI Sharpe | SUI Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 regime-switching pairs | 1 week | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 1 month | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 3 months | 10 | -114 | 30% | -3.41 | -1.1% | -1.3% | 0.00 | 2.00 | 0.00 | 0.00 | 0.00 | -2.15 | 0.00 | -2.78 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 6 months | 16 | -61 | 44% | -1.96 | -1.0% | -1.3% | 0.00 | 1.42 | 0.00 | 0.00 | 0.00 | -1.09 | 0.00 | -1.79 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 1 year | 20 | -41 | 45% | -1.00 | -0.8% | -1.3% | 0.00 | 0.93 | 0.00 | 0.00 | -1.00 | -0.77 | 0.00 | -1.26 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | 2 years | 22 | -35 | 50% | -0.65 | -0.8% | -1.3% | 0.00 | 0.67 | 0.00 | 0.00 | -0.51 | -0.55 | 0.00 | -0.89 | 0.00 | 0.00 | no |
| 5 regime-switching pairs | all development history | 54 | -34 | 43% | -0.49 | -1.9% | -1.9% | 0.25 | -0.22 | 0.27 | 0.00 | -0.40 | -0.61 | 0.00 | -0.59 | 0.00 | 0.00 | no |
| 5 revert regime only | 1 week | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 revert regime only | 1 month | 0 | n/a | n/a | 0.00 | +0.0% | -0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 5 revert regime only | 3 months | 10 | -114 | 30% | -3.41 | -1.1% | -1.3% | 0.00 | 2.00 | 0.00 | 0.00 | 0.00 | -2.15 | 0.00 | -2.78 | 0.00 | 0.00 | no |
| 5 revert regime only | 6 months | 16 | -61 | 44% | -1.96 | -1.0% | -1.3% | 0.00 | 1.42 | 0.00 | 0.00 | 0.00 | -1.09 | 0.00 | -1.79 | 0.00 | 0.00 | no |
| 5 revert regime only | 1 year | 20 | -41 | 45% | -1.00 | -0.8% | -1.3% | 0.00 | 0.93 | 0.00 | 0.00 | -1.00 | -0.77 | 0.00 | -1.26 | 0.00 | 0.00 | no |
| 5 revert regime only | 2 years | 22 | -35 | 50% | -0.65 | -0.8% | -1.3% | 0.00 | 0.67 | 0.00 | 0.00 | -0.51 | -0.55 | 0.00 | -0.89 | 0.00 | 0.00 | no |
| 5 revert regime only | all development history | 54 | -34 | 43% | -0.49 | -1.9% | -1.9% | 0.25 | -0.22 | 0.27 | 0.00 | -0.40 | -0.61 | 0.00 | -0.59 | 0.00 | 0.00 | no |
| bench: static hedge, no regime switch | 1 week | 4 | -7 | 50% | -2.86 | -0.0% | -0.1% | 0.00 | 7.22 | -7.22 | 4.13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| bench: static hedge, no regime switch | 1 month | 9 | +38 | 67% | 4.20 | +0.3% | -0.1% | 0.00 | 3.49 | -3.49 | 4.73 | 0.00 | 0.00 | 3.49 | 0.00 | 0.00 | 3.49 | no |
| bench: static hedge, no regime switch | 3 months | 55 | -44 | 35% | -2.93 | -2.4% | -2.9% | -1.88 | -1.86 | -1.24 | 2.92 | -2.26 | -0.54 | -0.98 | -2.53 | -3.72 | 0.13 | no |
| bench: static hedge, no regime switch | 6 months | 106 | -24 | 40% | -1.99 | -2.6% | -3.1% | -2.18 | -0.17 | -1.38 | 2.18 | -0.84 | -0.12 | -1.32 | -1.61 | -2.81 | -1.61 | no |
| bench: static hedge, no regime switch | 1 year | 261 | -22 | 36% | -1.60 | -5.6% | -6.5% | -1.81 | 0.44 | -0.30 | 1.14 | -1.70 | -1.78 | 0.13 | -2.49 | -1.29 | 0.65 | no |
| bench: static hedge, no regime switch | 2 years | 468 | -20 | 42% | -0.97 | -9.2% | -9.7% | -0.99 | -0.31 | 0.29 | 0.79 | -1.37 | -0.76 | 0.15 | -1.72 | 0.29 | -0.47 | no |
| bench: static hedge, no regime switch | all development history | 1082 | -21 | 41% | -0.87 | -20.6% | -21.0% | -0.77 | -0.21 | -0.27 | 0.79 | -0.59 | -0.56 | 0.15 | -0.91 | 0.29 | -0.47 | no |
