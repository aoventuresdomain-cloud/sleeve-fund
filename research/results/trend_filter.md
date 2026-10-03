# Long-only trend filter: staged screen

Run 2026-10-03 19:29 UTC. Development history only: the most recent 365 days are held back untouched. Short windows are smoke tests, not verdicts. Expectancy is the average net return per trade in basis points of equity. Sharpe uses daily returns of an equal-weight basket (per instrument in the right-hand columns). Costs are per side. Long-only, spot, no leverage. Equal-weight basket of the instruments with history at each date (missing instruments count as flat). Vol target sizes each position to 40% annualised volatility (30-day realised), capped at 100%, re-set only on a 25% drift.

## History

| Instrument | From | To (holdout starts) | Years |
|---|---|---|---|
| BTC/USD | 2017-08-17 | 2025-10-01 | 8.12 |
| ETH/USD | 2017-08-17 | 2025-10-01 | 8.12 |
| SOL/USD | 2020-08-11 | 2025-10-01 | 5.14 |
| SUI/USD | 2023-05-03 | 2025-10-01 | 2.41 |
| XRP/USD | 2018-05-04 | 2025-10-01 | 7.41 |

## long only (spot), kraken pro taker costs (40 bp fee + 2 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | BTC Sharpe | ETH Sharpe | SOL Sharpe | SUI Sharpe | XRP Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4h 21/55, full | 1 week | always in | n/a | n/a | 5.37 | +0.5% | -0.1% | 5.72 | -7.22 | 0.00 | 0.00 | -7.22 | n/a |
| 4h 21/55, full | 1 month | always in | n/a | n/a | 1.18 | +2.4% | -7.4% | 0.77 | 0.40 | 2.80 | -0.18 | 0.00 | n/a |
| 4h 21/55, full | 3 months | always in | n/a | n/a | 1.86 | +17.6% | -17.1% | 0.83 | 2.12 | 2.06 | 0.30 | 1.86 | n/a |
| 4h 21/55, full | 6 months | always in | n/a | n/a | 1.41 | +27.8% | -23.7% | 1.77 | 1.99 | 1.57 | 0.50 | 0.10 | n/a |
| 4h 21/55, full | 1 year | always in | n/a | n/a | 1.27 | +55.1% | -37.1% | 1.17 | 0.37 | 1.28 | 0.18 | 1.61 | n/a |
| 4h 21/55, full | 2 years | always in | n/a | n/a | 1.34 | +144.8% | -37.1% | 1.10 | 0.49 | 1.71 | 0.54 | 0.69 | n/a |
| 4h 21/55, full | all development history | always in | n/a | n/a | 1.11 | +1078.7% | -48.3% | 0.70 | 0.78 | 1.34 | 0.54 | 0.67 | n/a |
| 4h 21/55, vol target 40% | 1 week | always in | n/a | n/a | 5.42 | +0.5% | -0.1% | 5.72 | -7.22 | 0.00 | 0.00 | -7.22 | n/a |
| 4h 21/55, vol target 40% | 1 month | always in | n/a | n/a | 0.79 | +1.0% | -4.8% | 0.77 | 0.36 | 2.06 | -0.50 | -0.11 | n/a |
| 4h 21/55, vol target 40% | 3 months | always in | n/a | n/a | 2.08 | +12.3% | -9.7% | 0.83 | 2.21 | 1.87 | 0.25 | 2.60 | n/a |
| 4h 21/55, vol target 40% | 6 months | always in | n/a | n/a | 1.36 | +15.1% | -14.3% | 1.43 | 1.97 | 1.25 | 0.21 | 0.47 | n/a |
| 4h 21/55, vol target 40% | 1 year | always in | n/a | n/a | 1.54 | +36.4% | -17.4% | 1.34 | 0.50 | 1.27 | 0.24 | 1.92 | n/a |
| 4h 21/55, vol target 40% | 2 years | always in | n/a | n/a | 1.50 | +82.9% | -19.3% | 1.31 | 0.77 | 1.79 | 0.40 | 0.85 | n/a |
| 4h 21/55, vol target 40% | all development history | always in | n/a | n/a | 1.20 | +369.4% | -27.8% | 0.91 | 0.84 | 1.38 | 0.40 | 0.88 | n/a |
| daily 20/50, full | 1 week | always in | n/a | n/a | -0.10 | -0.3% | -0.7% | -8.12 | 3.46 | 2.89 | -7.22 | -7.78 | n/a |
| daily 20/50, full | 1 month | always in | n/a | n/a | -0.85 | -3.1% | -13.9% | -4.59 | 0.43 | 2.38 | -3.41 | -3.19 | n/a |
| daily 20/50, full | 3 months | always in | n/a | n/a | 1.28 | +14.4% | -18.4% | -0.96 | 3.44 | 2.04 | -0.76 | 0.52 | n/a |
| daily 20/50, full | 6 months | always in | n/a | n/a | 0.68 | +10.8% | -18.4% | 0.63 | 2.06 | 1.21 | -0.79 | -0.11 | n/a |
| daily 20/50, full | 1 year | always in | n/a | n/a | 1.62 | +87.3% | -23.6% | 1.68 | 1.09 | 0.63 | 0.95 | 1.69 | n/a |
| daily 20/50, full | 2 years | always in | n/a | n/a | 1.72 | +273.2% | -40.2% | 1.39 | 1.12 | 1.44 | 1.34 | 0.86 | n/a |
| daily 20/50, full | all development history | always in | n/a | n/a | 1.09 | +1209.5% | -53.7% | 1.01 | 0.91 | 1.06 | 1.34 | 0.35 | n/a |
| daily 20/50, vol target 40% | 1 week | always in | n/a | n/a | -0.79 | -0.7% | -0.5% | -8.12 | 3.46 | 2.89 | -7.22 | -7.78 | n/a |
| daily 20/50, vol target 40% | 1 month | always in | n/a | n/a | -1.59 | -3.5% | -9.9% | -4.59 | 0.16 | 1.57 | -3.41 | -3.48 | n/a |
| daily 20/50, vol target 40% | 3 months | always in | n/a | n/a | 1.34 | +9.9% | -13.3% | -1.02 | 3.20 | 1.83 | -1.00 | 1.07 | n/a |
| daily 20/50, vol target 40% | 6 months | always in | n/a | n/a | 0.73 | +8.1% | -13.3% | 0.45 | 2.06 | 0.96 | -1.15 | 0.21 | n/a |
| daily 20/50, vol target 40% | 1 year | always in | n/a | n/a | 1.53 | +41.4% | -13.3% | 1.58 | 0.96 | 0.85 | 0.71 | 1.48 | n/a |
| daily 20/50, vol target 40% | 2 years | always in | n/a | n/a | 1.71 | +117.2% | -23.6% | 1.51 | 1.16 | 1.66 | 1.07 | 0.76 | n/a |
| daily 20/50, vol target 40% | all development history | always in | n/a | n/a | 1.10 | +342.2% | -36.5% | 1.01 | 1.02 | 1.15 | 1.07 | 0.35 | n/a |
| daily 50/200, full | 1 week | always in | n/a | n/a | 3.08 | +4.0% | -1.1% | 5.26 | 3.46 | 2.89 | 3.00 | 0.93 | n/a |
| daily 50/200, full | 1 month | always in | n/a | n/a | 2.13 | +7.9% | -15.2% | 3.75 | 0.43 | 2.38 | 2.00 | 2.05 | n/a |
| daily 50/200, full | 3 months | always in | n/a | n/a | 1.85 | +24.4% | -15.6% | 1.30 | 2.67 | 0.76 | 1.34 | 1.94 | n/a |
| daily 50/200, full | 6 months | always in | n/a | n/a | 1.46 | +32.4% | -15.6% | 2.20 | 1.89 | 0.54 | 0.51 | 1.45 | n/a |
| daily 50/200, full | 1 year | always in | n/a | n/a | 1.40 | +85.6% | -37.3% | 1.75 | 0.55 | 0.44 | 0.83 | 1.76 | n/a |
| daily 50/200, full | 2 years | always in | n/a | n/a | 1.46 | +246.8% | -37.3% | 1.76 | 0.74 | 1.41 | 0.80 | 0.91 | n/a |
| daily 50/200, full | all development history | always in | n/a | n/a | 0.99 | +1091.2% | -42.8% | 0.83 | 0.61 | 1.33 | 0.80 | 0.55 | n/a |
| daily 50/200, vol target 40% | 1 week | always in | n/a | n/a | 3.05 | +2.9% | -0.8% | 5.26 | 3.46 | 2.89 | 3.00 | 0.93 | n/a |
| daily 50/200, vol target 40% | 1 month | always in | n/a | n/a | 1.60 | +3.9% | -10.4% | 3.75 | 0.06 | 1.67 | 1.34 | 1.51 | n/a |
| daily 50/200, vol target 40% | 3 months | always in | n/a | n/a | 1.87 | +15.2% | -10.4% | 1.30 | 2.73 | 0.38 | 1.18 | 2.32 | n/a |
| daily 50/200, vol target 40% | 6 months | always in | n/a | n/a | 1.51 | +20.1% | -10.4% | 2.15 | 1.93 | 0.27 | 0.40 | 1.52 | n/a |
| daily 50/200, vol target 40% | 1 year | always in | n/a | n/a | 1.42 | +45.1% | -19.0% | 1.73 | 0.50 | 0.56 | 0.70 | 1.74 | n/a |
| daily 50/200, vol target 40% | 2 years | always in | n/a | n/a | 1.52 | +122.9% | -19.0% | 1.78 | 0.77 | 1.57 | 0.66 | 0.84 | n/a |
| daily 50/200, vol target 40% | all development history | always in | n/a | n/a | 1.07 | +377.1% | -27.4% | 0.93 | 0.68 | 1.37 | 0.66 | 0.67 | n/a |
| bench: buy and hold | 1 week | always in | n/a | n/a | 3.08 | +4.0% | -1.1% | 5.26 | 3.46 | 2.89 | 3.00 | 0.93 | n/a |
| bench: buy and hold | 1 month | always in | n/a | n/a | 2.13 | +7.9% | -15.2% | 3.75 | 0.43 | 2.38 | 2.00 | 2.05 | n/a |
| bench: buy and hold | 3 months | always in | n/a | n/a | 2.42 | +36.0% | -15.5% | 1.30 | 3.44 | 2.41 | 1.34 | 1.94 | n/a |
| bench: buy and hold | 6 months | always in | n/a | n/a | 2.10 | +76.4% | -22.6% | 2.20 | 2.69 | 2.07 | 1.34 | 1.45 | n/a |
| bench: buy and hold | 1 year | always in | n/a | n/a | 1.65 | +147.4% | -50.1% | 1.75 | 1.14 | 0.92 | 1.17 | 2.10 | n/a |
| bench: buy and hold | 2 years | always in | n/a | n/a | 1.76 | +536.7% | -50.1% | 1.76 | 1.04 | 1.70 | 1.33 | 1.46 | n/a |
| bench: buy and hold | all development history | always in | n/a | n/a | 1.06 | +2743.5% | -73.7% | 0.87 | 0.76 | 1.10 | 1.33 | 0.81 | n/a |

## long only (spot), high volume taker costs (10 bp fee + 2 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | BTC Sharpe | ETH Sharpe | SOL Sharpe | SUI Sharpe | XRP Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4h 21/55, full | 1 week | always in | n/a | n/a | 6.34 | +0.7% | -0.1% | 6.38 | 7.22 | 0.00 | 0.00 | -7.22 | n/a |
| 4h 21/55, full | 1 month | always in | n/a | n/a | 1.56 | +3.2% | -7.0% | 1.67 | 0.83 | 2.89 | -0.05 | 0.54 | n/a |
| 4h 21/55, full | 3 months | always in | n/a | n/a | 2.11 | +20.5% | -15.8% | 1.39 | 2.31 | 2.17 | 0.50 | 2.01 | n/a |
| 4h 21/55, full | 6 months | always in | n/a | n/a | 1.67 | +34.8% | -22.1% | 2.18 | 2.15 | 1.73 | 0.68 | 0.38 | n/a |
| 4h 21/55, full | 1 year | always in | n/a | n/a | 1.57 | +74.8% | -33.9% | 1.53 | 0.62 | 1.45 | 0.37 | 1.76 | n/a |
| 4h 21/55, full | 2 years | always in | n/a | n/a | 1.64 | +209.2% | -33.9% | 1.42 | 0.73 | 1.87 | 0.71 | 0.89 | n/a |
| 4h 21/55, full | all development history | always in | n/a | n/a | 1.35 | +2123.3% | -42.7% | 0.97 | 0.96 | 1.48 | 0.71 | 0.82 | n/a |
| 4h 21/55, vol target 40% | 1 week | always in | n/a | n/a | 6.35 | +0.7% | -0.1% | 6.38 | 7.22 | 0.00 | 0.00 | -7.22 | n/a |
| 4h 21/55, vol target 40% | 1 month | always in | n/a | n/a | 1.30 | +1.7% | -4.5% | 1.67 | 0.86 | 2.19 | -0.34 | 0.48 | n/a |
| 4h 21/55, vol target 40% | 3 months | always in | n/a | n/a | 2.37 | +14.1% | -8.8% | 1.39 | 2.40 | 1.99 | 0.45 | 2.75 | n/a |
| 4h 21/55, vol target 40% | 6 months | always in | n/a | n/a | 1.67 | +19.1% | -13.2% | 1.89 | 2.14 | 1.42 | 0.41 | 0.74 | n/a |
| 4h 21/55, vol target 40% | 1 year | always in | n/a | n/a | 1.86 | +46.1% | -15.5% | 1.71 | 0.75 | 1.44 | 0.43 | 2.10 | n/a |
| 4h 21/55, vol target 40% | 2 years | always in | n/a | n/a | 1.82 | +110.4% | -16.9% | 1.62 | 1.02 | 1.95 | 0.58 | 1.09 | n/a |
| 4h 21/55, vol target 40% | all development history | always in | n/a | n/a | 1.47 | +581.0% | -23.9% | 1.18 | 1.04 | 1.54 | 0.58 | 1.07 | n/a |
| daily 20/50, full | 1 week | always in | n/a | n/a | 0.09 | -0.1% | -0.7% | -7.48 | 3.46 | 2.89 | -7.22 | -7.38 | n/a |
| daily 20/50, full | 1 month | always in | n/a | n/a | -0.71 | -2.7% | -13.9% | -4.10 | 0.43 | 2.38 | -3.27 | -2.88 | n/a |
| daily 20/50, full | 3 months | always in | n/a | n/a | 1.34 | +15.2% | -18.1% | -0.83 | 3.44 | 2.06 | -0.69 | 0.59 | n/a |
| daily 20/50, full | 6 months | always in | n/a | n/a | 0.74 | +12.1% | -18.1% | 0.72 | 2.07 | 1.24 | -0.74 | -0.04 | n/a |
| daily 20/50, full | 1 year | always in | n/a | n/a | 1.66 | +91.4% | -22.9% | 1.72 | 1.13 | 0.67 | 0.98 | 1.72 | n/a |
| daily 20/50, full | 2 years | always in | n/a | n/a | 1.77 | +288.0% | -39.7% | 1.43 | 1.15 | 1.47 | 1.36 | 0.89 | n/a |
| daily 20/50, full | all development history | always in | n/a | n/a | 1.12 | +1360.5% | -52.1% | 1.05 | 0.94 | 1.08 | 1.36 | 0.38 | n/a |
| daily 20/50, vol target 40% | 1 week | always in | n/a | n/a | -0.58 | -0.5% | -0.5% | -7.48 | 3.46 | 2.89 | -7.22 | -7.38 | n/a |
| daily 20/50, vol target 40% | 1 month | always in | n/a | n/a | -1.42 | -3.1% | -9.8% | -4.10 | 0.19 | 1.59 | -3.27 | -3.19 | n/a |
| daily 20/50, vol target 40% | 3 months | always in | n/a | n/a | 1.42 | +10.5% | -13.0% | -0.88 | 3.22 | 1.86 | -0.92 | 1.16 | n/a |
| daily 20/50, vol target 40% | 6 months | always in | n/a | n/a | 0.80 | +9.1% | -13.0% | 0.54 | 2.08 | 1.01 | -1.09 | 0.29 | n/a |
| daily 20/50, vol target 40% | 1 year | always in | n/a | n/a | 1.60 | +43.8% | -13.0% | 1.63 | 1.02 | 0.89 | 0.74 | 1.54 | n/a |
| daily 20/50, vol target 40% | 2 years | always in | n/a | n/a | 1.77 | +124.2% | -23.2% | 1.56 | 1.20 | 1.70 | 1.10 | 0.83 | n/a |
| daily 20/50, vol target 40% | all development history | always in | n/a | n/a | 1.16 | +380.7% | -34.9% | 1.06 | 1.06 | 1.18 | 1.10 | 0.41 | n/a |
| daily 50/200, full | 1 week | always in | n/a | n/a | 3.08 | +4.0% | -1.1% | 5.26 | 3.46 | 2.89 | 3.00 | 0.93 | n/a |
| daily 50/200, full | 1 month | always in | n/a | n/a | 2.13 | +7.9% | -15.2% | 3.75 | 0.43 | 2.38 | 2.00 | 2.05 | n/a |
| daily 50/200, full | 3 months | always in | n/a | n/a | 1.86 | +24.5% | -15.6% | 1.30 | 2.69 | 0.78 | 1.34 | 1.94 | n/a |
| daily 50/200, full | 6 months | always in | n/a | n/a | 1.47 | +32.6% | -15.6% | 2.20 | 1.90 | 0.55 | 0.52 | 1.45 | n/a |
| daily 50/200, full | 1 year | always in | n/a | n/a | 1.41 | +86.6% | -37.1% | 1.75 | 0.57 | 0.45 | 0.83 | 1.76 | n/a |
| daily 50/200, full | 2 years | always in | n/a | n/a | 1.47 | +251.4% | -37.1% | 1.76 | 0.75 | 1.42 | 0.81 | 0.92 | n/a |
| daily 50/200, full | all development history | always in | n/a | n/a | 1.00 | +1130.4% | -42.5% | 0.84 | 0.62 | 1.33 | 0.81 | 0.55 | n/a |
| daily 50/200, vol target 40% | 1 week | always in | n/a | n/a | 3.05 | +2.9% | -0.8% | 5.26 | 3.46 | 2.89 | 3.00 | 0.93 | n/a |
| daily 50/200, vol target 40% | 1 month | always in | n/a | n/a | 1.63 | +4.0% | -10.3% | 3.75 | 0.09 | 1.69 | 1.35 | 1.56 | n/a |
| daily 50/200, vol target 40% | 3 months | always in | n/a | n/a | 1.89 | +15.4% | -10.3% | 1.30 | 2.77 | 0.40 | 1.19 | 2.33 | n/a |
| daily 50/200, vol target 40% | 6 months | always in | n/a | n/a | 1.53 | +20.3% | -10.3% | 2.15 | 1.96 | 0.29 | 0.41 | 1.53 | n/a |
| daily 50/200, vol target 40% | 1 year | always in | n/a | n/a | 1.44 | +46.0% | -18.9% | 1.74 | 0.53 | 0.57 | 0.71 | 1.76 | n/a |
| daily 50/200, vol target 40% | 2 years | always in | n/a | n/a | 1.55 | +126.5% | -18.9% | 1.80 | 0.80 | 1.58 | 0.67 | 0.88 | n/a |
| daily 50/200, vol target 40% | all development history | always in | n/a | n/a | 1.09 | +394.3% | -27.2% | 0.94 | 0.70 | 1.39 | 0.67 | 0.68 | n/a |
| bench: buy and hold | 1 week | always in | n/a | n/a | 3.08 | +4.0% | -1.1% | 5.26 | 3.46 | 2.89 | 3.00 | 0.93 | n/a |
| bench: buy and hold | 1 month | always in | n/a | n/a | 2.13 | +7.9% | -15.2% | 3.75 | 0.43 | 2.38 | 2.00 | 2.05 | n/a |
| bench: buy and hold | 3 months | always in | n/a | n/a | 2.42 | +36.0% | -15.5% | 1.30 | 3.44 | 2.41 | 1.34 | 1.94 | n/a |
| bench: buy and hold | 6 months | always in | n/a | n/a | 2.10 | +76.4% | -22.6% | 2.20 | 2.69 | 2.07 | 1.34 | 1.45 | n/a |
| bench: buy and hold | 1 year | always in | n/a | n/a | 1.65 | +147.4% | -50.1% | 1.75 | 1.14 | 0.92 | 1.17 | 2.10 | n/a |
| bench: buy and hold | 2 years | always in | n/a | n/a | 1.76 | +536.7% | -50.1% | 1.76 | 1.04 | 1.70 | 1.33 | 1.46 | n/a |
| bench: buy and hold | all development history | always in | n/a | n/a | 1.06 | +2743.5% | -73.7% | 0.87 | 0.76 | 1.10 | 1.33 | 0.81 | n/a |

## long only (spot), institutional costs (2 bp fee + 1 bp slippage per side)

| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | BTC Sharpe | ETH Sharpe | SOL Sharpe | SUI Sharpe | XRP Sharpe | Cross-asset pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4h 21/55, full | 1 week | always in | n/a | n/a | 6.58 | +0.7% | -0.1% | 6.57 | 7.22 | 0.00 | 0.00 | -7.22 | n/a |
| 4h 21/55, full | 1 month | always in | n/a | n/a | 1.67 | +3.5% | -6.9% | 1.96 | 0.96 | 2.92 | -0.01 | 0.70 | n/a |
| 4h 21/55, full | 3 months | always in | n/a | n/a | 2.19 | +21.4% | -15.4% | 1.56 | 2.36 | 2.20 | 0.56 | 2.06 | n/a |
| 4h 21/55, full | 6 months | always in | n/a | n/a | 1.75 | +37.0% | -21.6% | 2.31 | 2.20 | 1.78 | 0.73 | 0.46 | n/a |
| 4h 21/55, full | 1 year | always in | n/a | n/a | 1.65 | +81.1% | -32.9% | 1.64 | 0.69 | 1.50 | 0.42 | 1.81 | n/a |
| 4h 21/55, full | 2 years | always in | n/a | n/a | 1.73 | +231.7% | -32.9% | 1.52 | 0.81 | 1.92 | 0.76 | 0.95 | n/a |
| 4h 21/55, full | all development history | always in | n/a | n/a | 1.42 | +2589.4% | -40.9% | 1.05 | 1.02 | 1.52 | 0.76 | 0.87 | n/a |
| 4h 21/55, vol target 40% | 1 week | always in | n/a | n/a | 6.58 | +0.7% | -0.1% | 6.57 | 7.22 | 0.00 | 0.00 | -7.22 | n/a |
| 4h 21/55, vol target 40% | 1 month | always in | n/a | n/a | 1.45 | +1.9% | -4.4% | 1.96 | 1.01 | 2.23 | -0.29 | 0.66 | n/a |
| 4h 21/55, vol target 40% | 3 months | always in | n/a | n/a | 2.46 | +14.7% | -8.6% | 1.56 | 2.46 | 2.03 | 0.51 | 2.79 | n/a |
| 4h 21/55, vol target 40% | 6 months | always in | n/a | n/a | 1.76 | +20.3% | -12.8% | 2.03 | 2.19 | 1.47 | 0.47 | 0.83 | n/a |
| 4h 21/55, vol target 40% | 1 year | always in | n/a | n/a | 1.95 | +49.2% | -14.9% | 1.82 | 0.83 | 1.49 | 0.49 | 2.15 | n/a |
| 4h 21/55, vol target 40% | 2 years | always in | n/a | n/a | 1.92 | +119.5% | -16.2% | 1.72 | 1.09 | 2.00 | 0.63 | 1.16 | n/a |
| 4h 21/55, vol target 40% | all development history | always in | n/a | n/a | 1.55 | +661.4% | -22.7% | 1.27 | 1.10 | 1.58 | 0.63 | 1.13 | n/a |
| daily 20/50, full | 1 week | always in | n/a | n/a | 0.15 | -0.1% | -0.7% | -7.29 | 3.46 | 2.89 | -7.22 | -7.26 | n/a |
| daily 20/50, full | 1 month | always in | n/a | n/a | -0.67 | -2.6% | -13.8% | -3.95 | 0.43 | 2.38 | -3.22 | -2.78 | n/a |
| daily 20/50, full | 3 months | always in | n/a | n/a | 1.36 | +15.4% | -18.0% | -0.79 | 3.44 | 2.06 | -0.67 | 0.61 | n/a |
| daily 20/50, full | 6 months | always in | n/a | n/a | 0.75 | +12.5% | -18.0% | 0.75 | 2.07 | 1.25 | -0.72 | -0.02 | n/a |
| daily 20/50, full | 1 year | always in | n/a | n/a | 1.68 | +92.6% | -22.7% | 1.73 | 1.14 | 0.68 | 0.98 | 1.73 | n/a |
| daily 20/50, full | 2 years | always in | n/a | n/a | 1.78 | +292.6% | -39.5% | 1.45 | 1.16 | 1.48 | 1.36 | 0.91 | n/a |
| daily 20/50, full | all development history | always in | n/a | n/a | 1.14 | +1409.1% | -51.6% | 1.06 | 0.94 | 1.09 | 1.36 | 0.39 | n/a |
| daily 20/50, vol target 40% | 1 week | always in | n/a | n/a | -0.52 | -0.5% | -0.5% | -7.29 | 3.46 | 2.89 | -7.22 | -7.26 | n/a |
| daily 20/50, vol target 40% | 1 month | always in | n/a | n/a | -1.37 | -3.0% | -9.8% | -3.95 | 0.19 | 1.59 | -3.22 | -3.10 | n/a |
| daily 20/50, vol target 40% | 3 months | always in | n/a | n/a | 1.44 | +10.7% | -12.9% | -0.84 | 3.22 | 1.87 | -0.90 | 1.18 | n/a |
| daily 20/50, vol target 40% | 6 months | always in | n/a | n/a | 0.82 | +9.4% | -12.9% | 0.57 | 2.08 | 1.02 | -1.07 | 0.32 | n/a |
| daily 20/50, vol target 40% | 1 year | always in | n/a | n/a | 1.62 | +44.5% | -12.9% | 1.65 | 1.03 | 0.91 | 0.75 | 1.55 | n/a |
| daily 20/50, vol target 40% | 2 years | always in | n/a | n/a | 1.79 | +126.3% | -23.0% | 1.58 | 1.22 | 1.71 | 1.11 | 0.84 | n/a |
| daily 20/50, vol target 40% | all development history | always in | n/a | n/a | 1.18 | +392.9% | -34.4% | 1.07 | 1.07 | 1.19 | 1.11 | 0.43 | n/a |
| daily 50/200, full | 1 week | always in | n/a | n/a | 3.08 | +4.0% | -1.1% | 5.26 | 3.46 | 2.89 | 3.00 | 0.93 | n/a |
| daily 50/200, full | 1 month | always in | n/a | n/a | 2.13 | +7.9% | -15.2% | 3.75 | 0.43 | 2.38 | 2.00 | 2.05 | n/a |
| daily 50/200, full | 3 months | always in | n/a | n/a | 1.86 | +24.6% | -15.5% | 1.30 | 2.70 | 0.78 | 1.34 | 1.94 | n/a |
| daily 50/200, full | 6 months | always in | n/a | n/a | 1.48 | +32.7% | -15.5% | 2.20 | 1.90 | 0.55 | 0.52 | 1.45 | n/a |
| daily 50/200, full | 1 year | always in | n/a | n/a | 1.41 | +86.9% | -37.1% | 1.75 | 0.58 | 0.46 | 0.83 | 1.76 | n/a |
| daily 50/200, full | 2 years | always in | n/a | n/a | 1.48 | +252.8% | -37.1% | 1.76 | 0.76 | 1.42 | 0.81 | 0.93 | n/a |
| daily 50/200, full | all development history | always in | n/a | n/a | 1.00 | +1142.4% | -42.5% | 0.84 | 0.62 | 1.33 | 0.81 | 0.55 | n/a |
| daily 50/200, vol target 40% | 1 week | always in | n/a | n/a | 3.05 | +2.9% | -0.8% | 5.26 | 3.46 | 2.89 | 3.00 | 0.93 | n/a |
| daily 50/200, vol target 40% | 1 month | always in | n/a | n/a | 1.64 | +4.0% | -10.3% | 3.75 | 0.10 | 1.69 | 1.35 | 1.57 | n/a |
| daily 50/200, vol target 40% | 3 months | always in | n/a | n/a | 1.90 | +15.4% | -10.3% | 1.30 | 2.78 | 0.41 | 1.19 | 2.34 | n/a |
| daily 50/200, vol target 40% | 6 months | always in | n/a | n/a | 1.54 | +20.4% | -10.3% | 2.15 | 1.96 | 0.29 | 0.42 | 1.53 | n/a |
| daily 50/200, vol target 40% | 1 year | always in | n/a | n/a | 1.45 | +46.3% | -18.8% | 1.74 | 0.53 | 0.58 | 0.71 | 1.77 | n/a |
| daily 50/200, vol target 40% | 2 years | always in | n/a | n/a | 1.56 | +127.6% | -18.8% | 1.80 | 0.80 | 1.59 | 0.68 | 0.89 | n/a |
| daily 50/200, vol target 40% | all development history | always in | n/a | n/a | 1.10 | +399.6% | -27.1% | 0.95 | 0.70 | 1.39 | 0.68 | 0.69 | n/a |
| bench: buy and hold | 1 week | always in | n/a | n/a | 3.08 | +4.0% | -1.1% | 5.26 | 3.46 | 2.89 | 3.00 | 0.93 | n/a |
| bench: buy and hold | 1 month | always in | n/a | n/a | 2.13 | +7.9% | -15.2% | 3.75 | 0.43 | 2.38 | 2.00 | 2.05 | n/a |
| bench: buy and hold | 3 months | always in | n/a | n/a | 2.42 | +36.0% | -15.5% | 1.30 | 3.44 | 2.41 | 1.34 | 1.94 | n/a |
| bench: buy and hold | 6 months | always in | n/a | n/a | 2.10 | +76.4% | -22.6% | 2.20 | 2.69 | 2.07 | 1.34 | 1.45 | n/a |
| bench: buy and hold | 1 year | always in | n/a | n/a | 1.65 | +147.4% | -50.1% | 1.75 | 1.14 | 0.92 | 1.17 | 2.10 | n/a |
| bench: buy and hold | 2 years | always in | n/a | n/a | 1.76 | +536.7% | -50.1% | 1.76 | 1.04 | 1.70 | 1.33 | 1.46 | n/a |
| bench: buy and hold | all development history | always in | n/a | n/a | 1.06 | +2743.5% | -73.7% | 0.87 | 0.76 | 1.10 | 1.33 | 0.81 | n/a |

## By calendar year (long-only, Kraken retail costs, equal-weight basket)

| Variant | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Sharpe | Max DD | Avg exposure | Deflated Sharpe prob. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| daily 20/50, full | -14.7% | +15.0% | +79.1% | +242.2% | -32.0% | +41.7% | +112.7% | +6.3% | 1.09 | -53.7% | 53% | 0.00 |
| daily 50/200, full | -14.5% | -1.5% | +36.4% | +258.5% | -17.6% | +51.8% | +114.2% | +7.9% | 0.99 | -42.8% | 59% | 0.00 |
| 4h 21/55, full | -8.2% | +13.9% | +91.3% | +186.2% | -33.8% | +78.7% | +81.7% | -4.2% | 1.11 | -48.3% | 50% | 0.00 |
| daily 20/50, vol target 40% | -8.1% | +18.7% | +58.3% | +63.2% | -22.1% | +24.9% | +52.2% | +5.9% | 1.10 | -36.5% | 29% | 0.00 |
| daily 50/200, vol target 40% | -6.1% | +2.5% | +40.4% | +65.0% | -9.3% | +34.9% | +58.3% | +10.4% | 1.07 | -27.4% | 32% | 0.00 |
| 4h 21/55, vol target 40% | -8.3% | +16.6% | +70.9% | +50.2% | -18.4% | +42.9% | +43.9% | +1.8% | 1.20 | -27.8% | 27% | 0.00 |
| bench: buy and hold | -28.9% | +9.7% | +127.3% | +339.7% | -65.1% | +179.8% | +196.5% | +26.1% | 1.06 | -73.7% | 100% | 0.00 |

Deflated Sharpe probability: the chance the true Sharpe beats the best you would expect by luck from 22 recorded attempts (above 0.95 is the plan's bar).
