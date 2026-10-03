# Classifier test (2026-10-03 19:07 UTC)

Does the day-type classifier tell trending sessions from balanced ones, using only bars closed by the decision time? Continuation is the move after the decision in the trend's direction, in daily ATR; the 95% range is a bootstrap. Move is the absolute move. t compares trend and balance moves. Development history only (the most recent year is held back). Variants: doc as the doc; relaxed {'side_share': 0.7, 'trend_slope': 0.1, 'relvol_min': 1.0}.

## History used

| Instrument | From | To | Years | Missing minutes |
|---|---|---|---|---|
| BTC/USD | 2017-08-17 | 2025-10-01 | 8.12 | 0.2% |
| ETH/USD | 2017-08-17 | 2025-10-01 | 8.12 | 0.2% |
| SOL/USD | 2020-08-11 | 2025-10-01 | 5.14 | 0.053% |
| SUI/USD | 2023-05-03 | 2025-10-01 | 2.41 | 0.0% |
| XRP/USD | 2018-05-04 | 2025-10-01 | 7.41 | 0.151% |

## Results

| Instrument | Variant | Bar | Session | Decide | Horizon | Sessions | Trend share | Balance share | Continuation (95% range) | Hit | Move trend | Move balance | t | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTC/USD | doc | 5m | utc | 60m | 4h | 2927 | 3% | 3% | 0.01 (-0.07 to 0.10) | 44% | 0.26 | 0.13 | 3.2 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | doc | 5m | utc | 60m | session | 2927 | 3% | 3% | -0.02 (-0.19 to 0.15) | 45% | 0.61 | 0.54 | 0.8 | fails: trend days don't behave differently enough |
| BTC/USD | relaxed | 5m | utc | 60m | 4h | 2927 | 6% | 3% | 0.02 (-0.03 to 0.08) | 46% | 0.24 | 0.13 | 3.4 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | relaxed | 5m | utc | 60m | session | 2927 | 6% | 3% | 0.01 (-0.12 to 0.13) | 43% | 0.60 | 0.54 | 0.7 | fails: trend days don't behave differently enough |
| BTC/USD | doc | 5m | utc | 90m | 4h | 2927 | 3% | 12% | 0.09 (0.02 to 0.17) | 55% | 0.25 | 0.15 | 3.4 | works: trend days keep going and move more than balance days |
| BTC/USD | doc | 5m | utc | 90m | session | 2927 | 3% | 12% | 0.08 (-0.11 to 0.29) | 43% | 0.66 | 0.49 | 2.2 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | relaxed | 5m | utc | 90m | 4h | 2927 | 7% | 12% | 0.04 (-0.01 to 0.09) | 48% | 0.24 | 0.15 | 4.4 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | relaxed | 5m | utc | 90m | session | 2927 | 7% | 12% | 0.01 (-0.10 to 0.14) | 42% | 0.62 | 0.49 | 2.3 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | doc | 5m | us_open | 60m | 4h | 2927 | 4% | 3% | 0.12 (0.05 to 0.20) | 57% | 0.31 | 0.14 | 4.9 | works: trend days keep going and move more than balance days |
| BTC/USD | doc | 5m | us_open | 60m | session | 2927 | 4% | 3% | 0.14 (0.00 to 0.30) | 53% | 0.54 | 0.38 | 2.2 | works: trend days keep going and move more than balance days |
| BTC/USD | relaxed | 5m | us_open | 60m | 4h | 2927 | 7% | 3% | 0.04 (-0.01 to 0.10) | 54% | 0.29 | 0.14 | 5.4 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | relaxed | 5m | us_open | 60m | session | 2927 | 7% | 3% | 0.13 (0.02 to 0.24) | 53% | 0.54 | 0.38 | 2.6 | works: trend days keep going and move more than balance days |
| BTC/USD | doc | 5m | us_open | 90m | 4h | 2927 | 4% | 12% | 0.11 (0.04 to 0.19) | 57% | 0.30 | 0.15 | 5.1 | works: trend days keep going and move more than balance days |
| BTC/USD | doc | 5m | us_open | 90m | session | 2927 | 4% | 12% | 0.16 (0.02 to 0.31) | 58% | 0.54 | 0.40 | 2.2 | works: trend days keep going and move more than balance days |
| BTC/USD | relaxed | 5m | us_open | 90m | 4h | 2927 | 8% | 12% | 0.06 (0.01 to 0.11) | 53% | 0.27 | 0.15 | 5.3 | works: trend days keep going and move more than balance days |
| BTC/USD | relaxed | 5m | us_open | 90m | session | 2927 | 8% | 12% | 0.13 (0.03 to 0.24) | 59% | 0.54 | 0.40 | 2.9 | works: trend days keep going and move more than balance days |
| BTC/USD | doc | 15m | utc | 60m | 4h | 2928 | 3% | 3% | 0.05 (-0.02 to 0.13) | 48% | 0.25 | 0.14 | 3.0 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | doc | 15m | utc | 60m | session | 2928 | 3% | 3% | 0.05 (-0.14 to 0.24) | 47% | 0.63 | 0.58 | 0.5 | fails: trend days don't behave differently enough |
| BTC/USD | relaxed | 15m | utc | 60m | 4h | 2928 | 8% | 3% | 0.01 (-0.03 to 0.06) | 43% | 0.24 | 0.14 | 3.7 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | relaxed | 15m | utc | 60m | session | 2928 | 8% | 3% | 0.02 (-0.09 to 0.12) | 44% | 0.57 | 0.58 | -0.1 | fails: trend days don't behave differently enough |
| BTC/USD | doc | 15m | utc | 90m | 4h | 2928 | 4% | 13% | 0.04 (-0.03 to 0.12) | 50% | 0.27 | 0.15 | 4.3 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | doc | 15m | utc | 90m | session | 2928 | 4% | 13% | 0.01 (-0.14 to 0.18) | 38% | 0.68 | 0.49 | 3.1 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | relaxed | 15m | utc | 90m | 4h | 2928 | 7% | 13% | 0.02 (-0.02 to 0.08) | 47% | 0.25 | 0.15 | 4.7 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | relaxed | 15m | utc | 90m | session | 2928 | 7% | 13% | -0.02 (-0.14 to 0.10) | 41% | 0.62 | 0.49 | 2.6 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | doc | 15m | us_open | 60m | 4h | 2928 | 4% | 4% | 0.15 (0.06 to 0.25) | 60% | 0.33 | 0.13 | 4.6 | works: trend days keep going and move more than balance days |
| BTC/USD | doc | 15m | us_open | 60m | session | 2928 | 4% | 4% | 0.20 (0.04 to 0.37) | 54% | 0.57 | 0.41 | 2.0 | works: trend days keep going and move more than balance days |
| BTC/USD | relaxed | 15m | us_open | 60m | 4h | 2928 | 10% | 4% | 0.05 (0.00 to 0.10) | 53% | 0.28 | 0.13 | 6.3 | works: trend days keep going and move more than balance days |
| BTC/USD | relaxed | 15m | us_open | 60m | session | 2928 | 10% | 4% | 0.11 (0.02 to 0.20) | 50% | 0.52 | 0.41 | 2.0 | works: trend days keep going and move more than balance days |
| BTC/USD | doc | 15m | us_open | 90m | 4h | 2928 | 6% | 13% | 0.08 (0.02 to 0.14) | 55% | 0.28 | 0.16 | 5.5 | works: trend days keep going and move more than balance days |
| BTC/USD | doc | 15m | us_open | 90m | session | 2928 | 6% | 13% | 0.12 (0.02 to 0.24) | 56% | 0.49 | 0.40 | 2.0 | partial: trend days keep going, but don't move clearly more than balance days |
| BTC/USD | relaxed | 15m | us_open | 90m | 4h | 2928 | 10% | 13% | 0.06 (0.01 to 0.10) | 52% | 0.27 | 0.16 | 5.7 | works: trend days keep going and move more than balance days |
| BTC/USD | relaxed | 15m | us_open | 90m | session | 2928 | 10% | 13% | 0.14 (0.05 to 0.23) | 57% | 0.54 | 0.40 | 3.3 | works: trend days keep going and move more than balance days |
| ETH/USD | doc | 5m | utc | 60m | 4h | 2927 | 4% | 2% | -0.01 (-0.09 to 0.06) | 48% | 0.26 | 0.14 | 3.5 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | doc | 5m | utc | 60m | session | 2927 | 4% | 2% | -0.04 (-0.19 to 0.12) | 41% | 0.67 | 0.40 | 3.7 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | relaxed | 5m | utc | 60m | 4h | 2927 | 7% | 2% | -0.01 (-0.05 to 0.04) | 49% | 0.23 | 0.14 | 3.7 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | relaxed | 5m | utc | 60m | session | 2927 | 7% | 2% | -0.02 (-0.13 to 0.09) | 44% | 0.61 | 0.40 | 3.3 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | doc | 5m | utc | 90m | 4h | 2927 | 4% | 12% | -0.00 (-0.05 to 0.05) | 43% | 0.19 | 0.15 | 2.3 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | doc | 5m | utc | 90m | session | 2927 | 4% | 12% | 0.01 (-0.12 to 0.15) | 39% | 0.58 | 0.42 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | relaxed | 5m | utc | 90m | 4h | 2927 | 8% | 12% | 0.02 (-0.02 to 0.05) | 47% | 0.20 | 0.15 | 3.6 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | relaxed | 5m | utc | 90m | session | 2927 | 8% | 12% | 0.03 (-0.08 to 0.14) | 42% | 0.58 | 0.42 | 3.5 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | doc | 5m | us_open | 60m | 4h | 2927 | 4% | 3% | 0.09 (0.03 to 0.15) | 57% | 0.27 | 0.19 | 2.8 | works: trend days keep going and move more than balance days |
| ETH/USD | doc | 5m | us_open | 60m | session | 2927 | 4% | 3% | 0.20 (0.09 to 0.33) | 56% | 0.51 | 0.42 | 1.3 | partial: trend days keep going, but don't move clearly more than balance days |
| ETH/USD | relaxed | 5m | us_open | 60m | 4h | 2927 | 8% | 3% | 0.04 (-0.00 to 0.09) | 52% | 0.26 | 0.19 | 2.6 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | relaxed | 5m | us_open | 60m | session | 2927 | 8% | 3% | 0.12 (0.03 to 0.21) | 53% | 0.51 | 0.42 | 1.5 | partial: trend days keep going, but don't move clearly more than balance days |
| ETH/USD | doc | 5m | us_open | 90m | 4h | 2927 | 4% | 13% | 0.06 (-0.01 to 0.14) | 54% | 0.28 | 0.18 | 3.3 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | doc | 5m | us_open | 90m | session | 2927 | 4% | 13% | 0.19 (0.06 to 0.34) | 55% | 0.51 | 0.42 | 1.5 | partial: trend days keep going, but don't move clearly more than balance days |
| ETH/USD | relaxed | 5m | us_open | 90m | 4h | 2927 | 9% | 13% | 0.05 (0.00 to 0.09) | 52% | 0.26 | 0.18 | 3.8 | works: trend days keep going and move more than balance days |
| ETH/USD | relaxed | 5m | us_open | 90m | session | 2927 | 9% | 13% | 0.09 (0.01 to 0.19) | 53% | 0.50 | 0.42 | 2.0 | partial: trend days keep going, but don't move clearly more than balance days |
| ETH/USD | doc | 15m | utc | 60m | 4h | 2928 | 4% | 3% | 0.00 (-0.07 to 0.07) | 53% | 0.24 | 0.17 | 2.5 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | doc | 15m | utc | 60m | session | 2928 | 4% | 3% | 0.00 (-0.16 to 0.17) | 40% | 0.67 | 0.36 | 5.0 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | relaxed | 15m | utc | 60m | 4h | 2928 | 9% | 3% | -0.02 (-0.06 to 0.02) | 46% | 0.22 | 0.17 | 2.5 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | relaxed | 15m | utc | 60m | session | 2928 | 9% | 3% | -0.07 (-0.16 to 0.03) | 38% | 0.61 | 0.36 | 5.2 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | doc | 15m | utc | 90m | 4h | 2928 | 5% | 13% | 0.00 (-0.04 to 0.05) | 43% | 0.21 | 0.15 | 3.3 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | doc | 15m | utc | 90m | session | 2928 | 5% | 13% | -0.01 (-0.14 to 0.12) | 41% | 0.61 | 0.43 | 3.8 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | relaxed | 15m | utc | 90m | 4h | 2928 | 8% | 13% | 0.01 (-0.03 to 0.05) | 45% | 0.20 | 0.15 | 3.3 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | relaxed | 15m | utc | 90m | session | 2928 | 8% | 13% | -0.02 (-0.13 to 0.10) | 40% | 0.62 | 0.43 | 4.4 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | doc | 15m | us_open | 60m | 4h | 2928 | 4% | 4% | 0.10 (0.03 to 0.18) | 56% | 0.30 | 0.18 | 3.7 | works: trend days keep going and move more than balance days |
| ETH/USD | doc | 15m | us_open | 60m | session | 2928 | 4% | 4% | 0.18 (0.05 to 0.31) | 55% | 0.51 | 0.44 | 0.8 | partial: trend days keep going, but don't move clearly more than balance days |
| ETH/USD | relaxed | 15m | us_open | 60m | 4h | 2928 | 10% | 4% | 0.04 (-0.00 to 0.08) | 50% | 0.26 | 0.18 | 4.0 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | relaxed | 15m | us_open | 60m | session | 2928 | 10% | 4% | 0.09 (0.01 to 0.16) | 50% | 0.49 | 0.44 | 0.7 | partial: trend days keep going, but don't move clearly more than balance days |
| ETH/USD | doc | 15m | us_open | 90m | 4h | 2928 | 6% | 14% | 0.08 (0.03 to 0.15) | 56% | 0.28 | 0.18 | 3.8 | works: trend days keep going and move more than balance days |
| ETH/USD | doc | 15m | us_open | 90m | session | 2928 | 6% | 14% | 0.14 (0.03 to 0.26) | 53% | 0.50 | 0.42 | 1.6 | partial: trend days keep going, but don't move clearly more than balance days |
| ETH/USD | relaxed | 15m | us_open | 90m | 4h | 2928 | 10% | 14% | 0.05 (0.01 to 0.09) | 54% | 0.26 | 0.18 | 3.9 | works: trend days keep going and move more than balance days |
| ETH/USD | relaxed | 15m | us_open | 90m | session | 2928 | 10% | 14% | 0.11 (0.03 to 0.19) | 52% | 0.50 | 0.42 | 1.8 | partial: trend days keep going, but don't move clearly more than balance days |
| SOL/USD | doc | 5m | utc | 60m | 4h | 1856 | 3% | 2% | 0.01 (-0.10 to 0.14) | 40% | 0.27 | 0.14 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | doc | 5m | utc | 60m | session | 1856 | 3% | 2% | -0.07 (-0.29 to 0.17) | 42% | 0.62 | 0.39 | 2.4 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | relaxed | 5m | utc | 60m | 4h | 1856 | 6% | 2% | -0.01 (-0.08 to 0.06) | 44% | 0.26 | 0.14 | 3.6 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | relaxed | 5m | utc | 60m | session | 1856 | 6% | 2% | -0.01 (-0.15 to 0.13) | 44% | 0.57 | 0.39 | 2.4 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | doc | 5m | utc | 90m | 4h | 1856 | 2% | 12% | 0.15 (0.01 to 0.29) | 56% | 0.36 | 0.15 | 3.5 | works: trend days keep going and move more than balance days |
| SOL/USD | doc | 5m | utc | 90m | session | 1856 | 2% | 12% | -0.00 (-0.31 to 0.29) | 49% | 0.68 | 0.39 | 2.6 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | relaxed | 5m | utc | 90m | 4h | 1856 | 6% | 12% | 0.07 (0.00 to 0.14) | 52% | 0.27 | 0.15 | 4.3 | works: trend days keep going and move more than balance days |
| SOL/USD | relaxed | 5m | utc | 90m | session | 1856 | 6% | 12% | -0.04 (-0.22 to 0.14) | 47% | 0.69 | 0.39 | 4.2 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | doc | 5m | us_open | 60m | 4h | 1857 | 4% | 2% | 0.03 (-0.05 to 0.11) | 48% | 0.27 | 0.24 | 0.6 | fails: trend days don't behave differently enough |
| SOL/USD | doc | 5m | us_open | 60m | session | 1857 | 4% | 2% | 0.06 (-0.11 to 0.22) | 53% | 0.57 | 0.49 | 0.9 | fails: trend days don't behave differently enough |
| SOL/USD | relaxed | 5m | us_open | 60m | 4h | 1857 | 8% | 2% | 0.03 (-0.02 to 0.09) | 51% | 0.26 | 0.24 | 0.3 | fails: trend days don't behave differently enough |
| SOL/USD | relaxed | 5m | us_open | 60m | session | 1857 | 8% | 2% | 0.03 (-0.07 to 0.15) | 53% | 0.53 | 0.49 | 0.5 | fails: trend days don't behave differently enough |
| SOL/USD | doc | 5m | us_open | 90m | 4h | 1857 | 3% | 10% | 0.03 (-0.08 to 0.15) | 48% | 0.36 | 0.21 | 3.3 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | doc | 5m | us_open | 90m | session | 1857 | 3% | 10% | 0.22 (0.02 to 0.43) | 63% | 0.65 | 0.41 | 3.3 | works: trend days keep going and move more than balance days |
| SOL/USD | relaxed | 5m | us_open | 90m | 4h | 1857 | 9% | 10% | 0.03 (-0.03 to 0.09) | 53% | 0.28 | 0.21 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | relaxed | 5m | us_open | 90m | session | 1857 | 9% | 10% | 0.14 (0.03 to 0.24) | 60% | 0.53 | 0.41 | 2.5 | works: trend days keep going and move more than balance days |
| SOL/USD | doc | 15m | utc | 60m | 4h | 1856 | 3% | 3% | 0.05 (-0.09 to 0.20) | 50% | 0.35 | 0.18 | 2.7 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | doc | 15m | utc | 60m | session | 1856 | 3% | 3% | 0.05 (-0.17 to 0.27) | 52% | 0.57 | 0.51 | 0.6 | fails: trend days don't behave differently enough |
| SOL/USD | relaxed | 15m | utc | 60m | 4h | 1856 | 8% | 3% | 0.03 (-0.04 to 0.10) | 47% | 0.28 | 0.18 | 3.0 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | relaxed | 15m | utc | 60m | session | 1856 | 8% | 3% | 0.06 (-0.07 to 0.18) | 50% | 0.61 | 0.51 | 1.3 | fails: trend days don't behave differently enough |
| SOL/USD | doc | 15m | utc | 90m | 4h | 1856 | 4% | 12% | 0.08 (-0.02 to 0.19) | 52% | 0.34 | 0.17 | 4.3 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | doc | 15m | utc | 90m | session | 1856 | 4% | 12% | -0.03 (-0.26 to 0.19) | 51% | 0.72 | 0.42 | 3.5 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | relaxed | 15m | utc | 90m | 4h | 1856 | 7% | 12% | 0.06 (-0.01 to 0.13) | 54% | 0.29 | 0.17 | 4.2 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | relaxed | 15m | utc | 90m | session | 1856 | 7% | 12% | -0.03 (-0.20 to 0.13) | 49% | 0.69 | 0.42 | 4.0 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | doc | 15m | us_open | 60m | 4h | 1857 | 4% | 2% | 0.04 (-0.06 to 0.14) | 53% | 0.29 | 0.26 | 0.5 | fails: trend days don't behave differently enough |
| SOL/USD | doc | 15m | us_open | 60m | session | 1857 | 4% | 2% | 0.00 (-0.19 to 0.19) | 53% | 0.61 | 0.43 | 2.1 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | relaxed | 15m | us_open | 60m | 4h | 1857 | 11% | 2% | 0.05 (-0.00 to 0.10) | 51% | 0.27 | 0.26 | 0.2 | fails: trend days don't behave differently enough |
| SOL/USD | relaxed | 15m | us_open | 60m | session | 1857 | 11% | 2% | 0.02 (-0.08 to 0.12) | 53% | 0.55 | 0.43 | 1.6 | fails: trend days don't behave differently enough |
| SOL/USD | doc | 15m | us_open | 90m | 4h | 1857 | 5% | 10% | 0.05 (-0.04 to 0.13) | 51% | 0.34 | 0.20 | 3.8 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | doc | 15m | us_open | 90m | session | 1857 | 5% | 10% | 0.10 (-0.06 to 0.26) | 57% | 0.62 | 0.40 | 3.5 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | relaxed | 15m | us_open | 90m | 4h | 1857 | 10% | 10% | 0.03 (-0.03 to 0.08) | 51% | 0.28 | 0.20 | 2.9 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | relaxed | 15m | us_open | 90m | session | 1857 | 10% | 10% | 0.07 (-0.04 to 0.18) | 54% | 0.54 | 0.40 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | doc | 5m | utc | 60m | 4h | 869 | 5% | 3% | 0.02 (-0.08 to 0.13) | 41% | 0.23 | 0.15 | 1.7 | too few sessions to judge |
| SUI/USD | doc | 5m | utc | 60m | session | 869 | 5% | 3% | -0.01 (-0.20 to 0.20) | 46% | 0.56 | 0.51 | 0.4 | too few sessions to judge |
| SUI/USD | relaxed | 5m | utc | 60m | 4h | 869 | 8% | 3% | -0.01 (-0.08 to 0.07) | 40% | 0.25 | 0.15 | 2.6 | too few sessions to judge |
| SUI/USD | relaxed | 5m | utc | 60m | session | 869 | 8% | 3% | -0.05 (-0.20 to 0.10) | 45% | 0.51 | 0.51 | 0.0 | too few sessions to judge |
| SUI/USD | doc | 5m | utc | 90m | 4h | 869 | 4% | 12% | 0.04 (-0.08 to 0.19) | 48% | 0.24 | 0.14 | 1.8 | fails: trend days don't behave differently enough |
| SUI/USD | doc | 5m | utc | 90m | session | 869 | 4% | 12% | 0.01 (-0.26 to 0.29) | 39% | 0.62 | 0.37 | 2.6 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | relaxed | 5m | utc | 90m | 4h | 869 | 7% | 12% | -0.02 (-0.11 to 0.07) | 37% | 0.26 | 0.14 | 3.4 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | relaxed | 5m | utc | 90m | session | 869 | 7% | 12% | -0.00 (-0.20 to 0.20) | 41% | 0.60 | 0.37 | 3.0 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | doc | 5m | us_open | 60m | 4h | 869 | 4% | 2% | -0.04 (-0.14 to 0.05) | 47% | 0.23 | 0.16 | 1.3 | too few sessions to judge |
| SUI/USD | doc | 5m | us_open | 60m | session | 869 | 4% | 2% | 0.01 (-0.20 to 0.26) | 44% | 0.49 | 0.53 | -0.3 | too few sessions to judge |
| SUI/USD | relaxed | 5m | us_open | 60m | 4h | 869 | 8% | 2% | 0.02 (-0.06 to 0.09) | 54% | 0.24 | 0.16 | 1.6 | too few sessions to judge |
| SUI/USD | relaxed | 5m | us_open | 60m | session | 869 | 8% | 2% | 0.07 (-0.09 to 0.24) | 48% | 0.49 | 0.53 | -0.3 | too few sessions to judge |
| SUI/USD | doc | 5m | us_open | 90m | 4h | 869 | 3% | 12% | 0.08 (-0.06 to 0.21) | 59% | 0.26 | 0.18 | 1.6 | too few sessions to judge |
| SUI/USD | doc | 5m | us_open | 90m | session | 869 | 3% | 12% | 0.26 (-0.11 to 0.65) | 45% | 0.70 | 0.37 | 2.3 | too few sessions to judge |
| SUI/USD | relaxed | 5m | us_open | 90m | 4h | 869 | 9% | 12% | 0.08 (0.01 to 0.15) | 54% | 0.24 | 0.18 | 1.8 | partial: trend days keep going, but don't move clearly more than balance days |
| SUI/USD | relaxed | 5m | us_open | 90m | session | 869 | 9% | 12% | 0.17 (0.00 to 0.36) | 56% | 0.59 | 0.37 | 3.0 | works: trend days keep going and move more than balance days |
| SUI/USD | doc | 15m | utc | 60m | 4h | 869 | 4% | 4% | 0.08 (-0.03 to 0.20) | 53% | 0.26 | 0.16 | 2.0 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | doc | 15m | utc | 60m | session | 869 | 4% | 4% | 0.06 (-0.17 to 0.31) | 47% | 0.60 | 0.40 | 1.7 | fails: trend days don't behave differently enough |
| SUI/USD | relaxed | 15m | utc | 60m | 4h | 869 | 10% | 4% | 0.02 (-0.04 to 0.09) | 46% | 0.24 | 0.16 | 2.6 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | relaxed | 15m | utc | 60m | session | 869 | 10% | 4% | 0.01 (-0.11 to 0.14) | 49% | 0.49 | 0.40 | 0.9 | fails: trend days don't behave differently enough |
| SUI/USD | doc | 15m | utc | 90m | 4h | 869 | 5% | 12% | -0.01 (-0.11 to 0.11) | 46% | 0.25 | 0.15 | 2.3 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | doc | 15m | utc | 90m | session | 869 | 5% | 12% | -0.03 (-0.25 to 0.19) | 39% | 0.57 | 0.37 | 2.5 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | relaxed | 15m | utc | 90m | 4h | 869 | 7% | 12% | -0.01 (-0.09 to 0.08) | 43% | 0.24 | 0.15 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | relaxed | 15m | utc | 90m | session | 869 | 7% | 12% | -0.00 (-0.19 to 0.20) | 44% | 0.57 | 0.37 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | doc | 15m | us_open | 60m | 4h | 869 | 3% | 2% | -0.03 (-0.13 to 0.08) | 44% | 0.23 | 0.15 | 2.0 | too few sessions to judge |
| SUI/USD | doc | 15m | us_open | 60m | session | 869 | 3% | 2% | -0.02 (-0.26 to 0.26) | 37% | 0.49 | 0.43 | 0.5 | too few sessions to judge |
| SUI/USD | relaxed | 15m | us_open | 60m | 4h | 869 | 10% | 2% | 0.01 (-0.06 to 0.08) | 51% | 0.26 | 0.15 | 3.4 | too few sessions to judge |
| SUI/USD | relaxed | 15m | us_open | 60m | session | 869 | 10% | 2% | 0.10 (-0.04 to 0.26) | 49% | 0.53 | 0.43 | 0.9 | too few sessions to judge |
| SUI/USD | doc | 15m | us_open | 90m | 4h | 869 | 4% | 12% | 0.08 (-0.01 to 0.17) | 59% | 0.23 | 0.20 | 0.9 | fails: trend days don't behave differently enough |
| SUI/USD | doc | 15m | us_open | 90m | session | 869 | 4% | 12% | 0.32 (0.04 to 0.62) | 51% | 0.70 | 0.42 | 2.3 | works: trend days keep going and move more than balance days |
| SUI/USD | relaxed | 15m | us_open | 90m | 4h | 869 | 9% | 12% | 0.06 (0.00 to 0.13) | 54% | 0.22 | 0.20 | 0.8 | partial: trend days keep going, but don't move clearly more than balance days |
| SUI/USD | relaxed | 15m | us_open | 90m | session | 869 | 9% | 12% | 0.19 (0.04 to 0.37) | 55% | 0.54 | 0.42 | 1.7 | partial: trend days keep going, but don't move clearly more than balance days |
| XRP/USD | doc | 5m | utc | 60m | 4h | 2672 | 3% | 2% | 0.03 (-0.05 to 0.10) | 59% | 0.23 | 0.13 | 2.7 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 5m | utc | 60m | session | 2672 | 3% | 2% | 0.14 (-0.13 to 0.50) | 58% | 0.79 | 0.39 | 2.9 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 5m | utc | 60m | 4h | 2672 | 5% | 2% | 0.01 (-0.04 to 0.06) | 53% | 0.21 | 0.13 | 3.1 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 5m | utc | 60m | session | 2672 | 5% | 2% | 0.09 (-0.09 to 0.28) | 53% | 0.67 | 0.39 | 3.1 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 5m | utc | 90m | 4h | 2672 | 2% | 11% | 0.02 (-0.06 to 0.10) | 45% | 0.25 | 0.15 | 3.0 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 5m | utc | 90m | session | 2672 | 2% | 11% | 0.14 (-0.13 to 0.47) | 50% | 0.77 | 0.40 | 2.7 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 5m | utc | 90m | 4h | 2672 | 6% | 11% | 0.02 (-0.05 to 0.11) | 43% | 0.27 | 0.15 | 3.1 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 5m | utc | 90m | session | 2672 | 6% | 11% | -0.03 (-0.19 to 0.16) | 44% | 0.65 | 0.40 | 3.4 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 5m | us_open | 60m | 4h | 2672 | 3% | 3% | 0.03 (-0.04 to 0.10) | 47% | 0.24 | 0.15 | 3.7 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 5m | us_open | 60m | session | 2672 | 3% | 3% | 0.04 (-0.11 to 0.20) | 51% | 0.50 | 0.37 | 1.9 | fails: trend days don't behave differently enough |
| XRP/USD | relaxed | 5m | us_open | 60m | 4h | 2672 | 6% | 3% | 0.04 (-0.01 to 0.10) | 48% | 0.25 | 0.15 | 4.3 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 5m | us_open | 60m | session | 2672 | 6% | 3% | 0.03 (-0.09 to 0.15) | 52% | 0.54 | 0.37 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 5m | us_open | 90m | 4h | 2672 | 3% | 14% | 0.05 (-0.05 to 0.15) | 54% | 0.31 | 0.17 | 3.1 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 5m | us_open | 90m | session | 2672 | 3% | 14% | 0.08 (-0.14 to 0.30) | 55% | 0.65 | 0.42 | 2.5 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 5m | us_open | 90m | 4h | 2672 | 7% | 14% | 0.01 (-0.05 to 0.07) | 48% | 0.27 | 0.17 | 3.7 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 5m | us_open | 90m | session | 2672 | 7% | 14% | 0.02 (-0.11 to 0.16) | 51% | 0.60 | 0.42 | 3.2 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 15m | utc | 60m | 4h | 2672 | 3% | 3% | 0.02 (-0.07 to 0.10) | 61% | 0.24 | 0.15 | 2.5 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 15m | utc | 60m | session | 2672 | 3% | 3% | 0.07 (-0.22 to 0.43) | 48% | 0.79 | 0.38 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 15m | utc | 60m | 4h | 2672 | 8% | 3% | -0.02 (-0.07 to 0.02) | 46% | 0.22 | 0.15 | 2.7 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 15m | utc | 60m | session | 2672 | 8% | 3% | 0.01 (-0.13 to 0.16) | 51% | 0.63 | 0.38 | 3.2 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 15m | utc | 90m | 4h | 2672 | 4% | 12% | 0.04 (-0.04 to 0.15) | 46% | 0.28 | 0.15 | 3.4 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 15m | utc | 90m | session | 2672 | 4% | 12% | 0.01 (-0.22 to 0.26) | 47% | 0.75 | 0.41 | 3.4 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 15m | utc | 90m | 4h | 2672 | 7% | 12% | 0.05 (-0.02 to 0.13) | 49% | 0.28 | 0.15 | 4.0 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 15m | utc | 90m | session | 2672 | 7% | 12% | -0.02 (-0.17 to 0.14) | 46% | 0.66 | 0.41 | 3.7 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 15m | us_open | 60m | 4h | 2672 | 3% | 4% | 0.02 (-0.05 to 0.10) | 46% | 0.24 | 0.14 | 3.5 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 15m | us_open | 60m | session | 2672 | 3% | 4% | 0.08 (-0.07 to 0.22) | 51% | 0.49 | 0.38 | 1.6 | fails: trend days don't behave differently enough |
| XRP/USD | relaxed | 15m | us_open | 60m | 4h | 2672 | 8% | 4% | 0.03 (-0.02 to 0.07) | 46% | 0.24 | 0.14 | 5.0 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 15m | us_open | 60m | session | 2672 | 8% | 4% | 0.04 (-0.06 to 0.14) | 52% | 0.54 | 0.38 | 2.7 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 15m | us_open | 90m | 4h | 2672 | 4% | 14% | 0.04 (-0.04 to 0.12) | 53% | 0.27 | 0.17 | 3.1 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | doc | 15m | us_open | 90m | session | 2672 | 4% | 14% | 0.05 (-0.12 to 0.22) | 52% | 0.61 | 0.44 | 2.5 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 15m | us_open | 90m | 4h | 2672 | 8% | 14% | 0.03 (-0.02 to 0.08) | 52% | 0.24 | 0.17 | 3.0 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | relaxed | 15m | us_open | 90m | session | 2672 | 8% | 14% | 0.06 (-0.05 to 0.16) | 51% | 0.53 | 0.44 | 1.9 | fails: trend days don't behave differently enough |
