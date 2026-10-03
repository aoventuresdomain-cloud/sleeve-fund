# Classifier test (2026-10-03 18:53 UTC)

Does the day-type classifier tell trending sessions from balanced ones, using only bars closed by the decision time? Continuation is the move after the decision in the trend's direction, in daily ATR; the 95% range is a bootstrap. Move is the absolute move. t compares trend and balance moves.

## History used

| Instrument | From | To | Years | Missing minutes |
|---|---|---|---|---|
| BTC/USD | 2017-08-17 | 2026-10-02 | 9.13 | 1.078% |
| ETH/USD | 2017-08-17 | 2026-10-02 | 9.13 | 1.078% |
| SOL/USD | 2020-08-11 | 2026-10-02 | 6.14 | 1.381% |
| SUI/USD | 2023-05-03 | 2026-10-02 | 3.42 | 2.403% |
| XRP/USD | 2018-05-04 | 2026-10-02 | 8.41 | 1.109% |

## Results

| Instrument | Bar | Session | Decide | Horizon | Sessions | Trend share | Balance share | Continuation (95% range) | Hit | Move trend | Move balance | t | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTC/USD | 1m | utc | 30m | 4h | 3262 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 1m | utc | 30m | session | 3262 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 1m | utc | 60m | 4h | 3262 | 2% | 46% | 0.05 (-0.04 to 0.14) | 49% | 0.25 | 0.14 | 3.0 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | 1m | utc | 60m | session | 3262 | 2% | 46% | -0.01 (-0.22 to 0.22) | 44% | 0.66 | 0.44 | 2.6 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | 1m | utc | 90m | 4h | 3262 | 2% | 56% | 0.10 (0.02 to 0.18) | 61% | 0.22 | 0.14 | 2.3 | works: trend days keep going and move more than balance days |
| BTC/USD | 1m | utc | 90m | session | 3262 | 2% | 56% | 0.11 (-0.11 to 0.38) | 51% | 0.65 | 0.43 | 2.6 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | 1m | us_open | 30m | 4h | 3260 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 1m | us_open | 30m | session | 3260 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 1m | us_open | 60m | 4h | 3261 | 2% | 40% | 0.13 (0.04 to 0.22) | 59% | 0.30 | 0.17 | 3.9 | works: trend days keep going and move more than balance days |
| BTC/USD | 1m | us_open | 60m | session | 3261 | 2% | 40% | 0.12 (-0.03 to 0.28) | 49% | 0.48 | 0.42 | 1.0 | fails: trend days don't behave differently enough |
| BTC/USD | 1m | us_open | 90m | 4h | 3261 | 3% | 46% | 0.11 (0.05 to 0.18) | 62% | 0.27 | 0.17 | 3.8 | works: trend days keep going and move more than balance days |
| BTC/USD | 1m | us_open | 90m | session | 3261 | 3% | 46% | 0.13 (-0.02 to 0.30) | 57% | 0.54 | 0.41 | 2.0 | fails: trend days don't behave differently enough |
| BTC/USD | 5m | utc | 30m | 4h | 3263 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 5m | utc | 30m | session | 3263 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 5m | utc | 60m | 4h | 3263 | 3% | 28% | 0.01 (-0.06 to 0.10) | 45% | 0.25 | 0.15 | 3.4 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | 5m | utc | 60m | session | 3263 | 3% | 28% | -0.02 (-0.18 to 0.14) | 46% | 0.60 | 0.47 | 2.3 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | 5m | utc | 90m | 4h | 3263 | 3% | 42% | 0.08 (0.00 to 0.15) | 56% | 0.25 | 0.14 | 3.7 | works: trend days keep going and move more than balance days |
| BTC/USD | 5m | utc | 90m | session | 3263 | 3% | 42% | 0.10 (-0.08 to 0.30) | 45% | 0.66 | 0.44 | 3.0 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | 5m | us_open | 30m | 4h | 3261 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 5m | us_open | 30m | session | 3261 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 5m | us_open | 60m | 4h | 3262 | 4% | 28% | 0.12 (0.05 to 0.20) | 58% | 0.31 | 0.17 | 5.1 | works: trend days keep going and move more than balance days |
| BTC/USD | 5m | us_open | 60m | session | 3262 | 4% | 28% | 0.13 (0.00 to 0.28) | 52% | 0.52 | 0.41 | 2.0 | works: trend days keep going and move more than balance days |
| BTC/USD | 5m | us_open | 90m | 4h | 3262 | 4% | 38% | 0.12 (0.05 to 0.18) | 58% | 0.30 | 0.17 | 5.1 | works: trend days keep going and move more than balance days |
| BTC/USD | 5m | us_open | 90m | session | 3262 | 4% | 38% | 0.15 (0.02 to 0.30) | 56% | 0.52 | 0.40 | 2.0 | works: trend days keep going and move more than balance days |
| BTC/USD | 15m | utc | 30m | 4h | 3264 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 15m | utc | 30m | session | 3264 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 15m | utc | 60m | 4h | 3264 | 2% | 3% | 0.05 (-0.02 to 0.14) | 49% | 0.25 | 0.13 | 3.2 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | 15m | utc | 60m | session | 3264 | 2% | 3% | 0.06 (-0.11 to 0.25) | 48% | 0.61 | 0.57 | 0.4 | fails: trend days don't behave differently enough |
| BTC/USD | 15m | utc | 90m | 4h | 3264 | 4% | 13% | 0.03 (-0.03 to 0.11) | 50% | 0.27 | 0.15 | 4.5 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | 15m | utc | 90m | session | 3264 | 4% | 13% | 0.02 (-0.12 to 0.19) | 40% | 0.69 | 0.49 | 3.2 | partial: trend days move more, but not reliably in the trend's direction |
| BTC/USD | 15m | us_open | 30m | 4h | 3262 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 15m | us_open | 30m | session | 3262 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| BTC/USD | 15m | us_open | 60m | 4h | 3263 | 4% | 4% | 0.14 (0.06 to 0.23) | 61% | 0.32 | 0.14 | 4.6 | works: trend days keep going and move more than balance days |
| BTC/USD | 15m | us_open | 60m | session | 3263 | 4% | 4% | 0.17 (0.03 to 0.33) | 52% | 0.54 | 0.40 | 1.9 | partial: trend days keep going, but don't move clearly more than balance days |
| BTC/USD | 15m | us_open | 90m | 4h | 3263 | 6% | 13% | 0.10 (0.05 to 0.15) | 56% | 0.29 | 0.16 | 5.6 | works: trend days keep going and move more than balance days |
| BTC/USD | 15m | us_open | 90m | session | 3263 | 6% | 13% | 0.14 (0.03 to 0.26) | 54% | 0.51 | 0.40 | 2.3 | works: trend days keep going and move more than balance days |
| ETH/USD | 1m | utc | 30m | 4h | 3262 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 1m | utc | 30m | session | 3262 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 1m | utc | 60m | 4h | 3262 | 2% | 44% | 0.06 (-0.01 to 0.15) | 54% | 0.24 | 0.15 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | 1m | utc | 60m | session | 3262 | 2% | 44% | 0.08 (-0.10 to 0.28) | 48% | 0.64 | 0.43 | 3.4 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | 1m | utc | 90m | 4h | 3262 | 2% | 53% | 0.04 (-0.01 to 0.10) | 54% | 0.18 | 0.14 | 1.8 | fails: trend days don't behave differently enough |
| ETH/USD | 1m | utc | 90m | session | 3262 | 2% | 53% | 0.09 (-0.05 to 0.25) | 50% | 0.53 | 0.43 | 1.9 | fails: trend days don't behave differently enough |
| ETH/USD | 1m | us_open | 30m | 4h | 3260 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 1m | us_open | 30m | session | 3260 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 1m | us_open | 60m | 4h | 3261 | 3% | 39% | 0.13 (0.06 to 0.21) | 60% | 0.28 | 0.18 | 3.2 | works: trend days keep going and move more than balance days |
| ETH/USD | 1m | us_open | 60m | session | 3261 | 3% | 39% | 0.25 (0.10 to 0.39) | 62% | 0.55 | 0.41 | 2.4 | works: trend days keep going and move more than balance days |
| ETH/USD | 1m | us_open | 90m | 4h | 3261 | 3% | 48% | 0.19 (0.10 to 0.28) | 66% | 0.32 | 0.18 | 3.8 | works: trend days keep going and move more than balance days |
| ETH/USD | 1m | us_open | 90m | session | 3261 | 3% | 48% | 0.23 (0.09 to 0.40) | 58% | 0.54 | 0.42 | 1.9 | partial: trend days keep going, but don't move clearly more than balance days |
| ETH/USD | 5m | utc | 30m | 4h | 3263 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 5m | utc | 30m | session | 3263 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 5m | utc | 60m | 4h | 3263 | 4% | 27% | -0.01 (-0.08 to 0.06) | 49% | 0.25 | 0.15 | 3.6 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | 5m | utc | 60m | session | 3263 | 4% | 27% | -0.03 (-0.19 to 0.13) | 43% | 0.65 | 0.44 | 4.0 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | 5m | utc | 90m | 4h | 3263 | 4% | 40% | -0.01 (-0.05 to 0.04) | 44% | 0.19 | 0.15 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | 5m | utc | 90m | session | 3263 | 4% | 40% | 0.01 (-0.13 to 0.15) | 41% | 0.57 | 0.44 | 2.7 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | 5m | us_open | 30m | 4h | 3261 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 5m | us_open | 30m | session | 3261 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 5m | us_open | 60m | 4h | 3262 | 4% | 28% | 0.09 (0.04 to 0.15) | 58% | 0.27 | 0.19 | 3.6 | works: trend days keep going and move more than balance days |
| ETH/USD | 5m | us_open | 60m | session | 3262 | 4% | 28% | 0.22 (0.12 to 0.33) | 57% | 0.51 | 0.41 | 2.3 | works: trend days keep going and move more than balance days |
| ETH/USD | 5m | us_open | 90m | 4h | 3262 | 4% | 39% | 0.08 (0.01 to 0.15) | 55% | 0.28 | 0.18 | 3.6 | works: trend days keep going and move more than balance days |
| ETH/USD | 5m | us_open | 90m | session | 3262 | 4% | 39% | 0.21 (0.08 to 0.35) | 57% | 0.52 | 0.41 | 2.0 | partial: trend days keep going, but don't move clearly more than balance days |
| ETH/USD | 15m | utc | 30m | 4h | 3264 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 15m | utc | 30m | session | 3264 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 15m | utc | 60m | 4h | 3264 | 4% | 3% | 0.00 (-0.06 to 0.07) | 54% | 0.23 | 0.17 | 2.4 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | 15m | utc | 60m | session | 3264 | 4% | 3% | 0.01 (-0.14 to 0.17) | 43% | 0.64 | 0.36 | 4.9 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | 15m | utc | 90m | 4h | 3264 | 4% | 13% | 0.00 (-0.04 to 0.05) | 46% | 0.21 | 0.15 | 3.5 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | 15m | utc | 90m | session | 3264 | 4% | 13% | 0.01 (-0.12 to 0.13) | 44% | 0.61 | 0.44 | 3.5 | partial: trend days move more, but not reliably in the trend's direction |
| ETH/USD | 15m | us_open | 30m | 4h | 3262 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 15m | us_open | 30m | session | 3262 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| ETH/USD | 15m | us_open | 60m | 4h | 3263 | 4% | 4% | 0.10 (0.03 to 0.17) | 57% | 0.30 | 0.19 | 3.6 | works: trend days keep going and move more than balance days |
| ETH/USD | 15m | us_open | 60m | session | 3263 | 4% | 4% | 0.18 (0.07 to 0.30) | 56% | 0.51 | 0.43 | 1.0 | partial: trend days keep going, but don't move clearly more than balance days |
| ETH/USD | 15m | us_open | 90m | 4h | 3263 | 6% | 14% | 0.09 (0.04 to 0.15) | 57% | 0.28 | 0.18 | 4.1 | works: trend days keep going and move more than balance days |
| ETH/USD | 15m | us_open | 90m | session | 3263 | 6% | 14% | 0.16 (0.06 to 0.27) | 55% | 0.51 | 0.42 | 1.8 | partial: trend days keep going, but don't move clearly more than balance days |
| SOL/USD | 1m | utc | 30m | 4h | 2192 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 1m | utc | 30m | session | 2192 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 1m | utc | 60m | 4h | 2192 | 2% | 43% | -0.01 (-0.14 to 0.13) | 47% | 0.27 | 0.16 | 2.1 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 1m | utc | 60m | session | 2192 | 2% | 43% | -0.15 (-0.38 to 0.09) | 41% | 0.53 | 0.46 | 0.9 | fails: trend days don't behave differently enough |
| SOL/USD | 1m | utc | 90m | 4h | 2192 | 2% | 50% | 0.03 (-0.11 to 0.20) | 45% | 0.35 | 0.15 | 3.4 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 1m | utc | 90m | session | 2192 | 2% | 50% | 0.03 (-0.23 to 0.33) | 47% | 0.64 | 0.43 | 2.1 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 1m | us_open | 30m | 4h | 2191 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 1m | us_open | 30m | session | 2191 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 1m | us_open | 60m | 4h | 2192 | 3% | 35% | 0.01 (-0.08 to 0.10) | 52% | 0.27 | 0.19 | 2.4 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 1m | us_open | 60m | session | 2192 | 3% | 35% | 0.03 (-0.14 to 0.22) | 50% | 0.57 | 0.41 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 1m | us_open | 90m | 4h | 2192 | 2% | 43% | -0.02 (-0.13 to 0.11) | 43% | 0.31 | 0.19 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 1m | us_open | 90m | session | 2192 | 2% | 43% | 0.16 (-0.03 to 0.36) | 63% | 0.53 | 0.42 | 1.5 | fails: trend days don't behave differently enough |
| SOL/USD | 5m | utc | 30m | 4h | 2192 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 5m | utc | 30m | session | 2192 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 5m | utc | 60m | 4h | 2192 | 3% | 27% | 0.00 (-0.09 to 0.12) | 45% | 0.27 | 0.15 | 2.9 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 5m | utc | 60m | session | 2192 | 3% | 27% | -0.08 (-0.27 to 0.12) | 41% | 0.57 | 0.44 | 1.8 | fails: trend days don't behave differently enough |
| SOL/USD | 5m | utc | 90m | 4h | 2192 | 2% | 38% | 0.10 (-0.03 to 0.24) | 51% | 0.34 | 0.15 | 3.8 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 5m | utc | 90m | session | 2192 | 2% | 38% | -0.03 (-0.29 to 0.22) | 47% | 0.64 | 0.44 | 2.1 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 5m | us_open | 30m | 4h | 2191 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 5m | us_open | 30m | session | 2191 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 5m | us_open | 60m | 4h | 2192 | 4% | 26% | 0.05 (-0.02 to 0.13) | 52% | 0.27 | 0.20 | 2.3 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 5m | us_open | 60m | session | 2192 | 4% | 26% | 0.09 (-0.05 to 0.22) | 56% | 0.55 | 0.43 | 2.4 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 5m | us_open | 90m | 4h | 2192 | 4% | 34% | 0.06 (-0.04 to 0.16) | 53% | 0.33 | 0.20 | 3.5 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 5m | us_open | 90m | session | 2192 | 4% | 34% | 0.20 (0.03 to 0.38) | 66% | 0.58 | 0.41 | 2.8 | works: trend days keep going and move more than balance days |
| SOL/USD | 15m | utc | 30m | 4h | 2192 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 15m | utc | 30m | session | 2192 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 15m | utc | 60m | 4h | 2192 | 3% | 3% | 0.05 (-0.07 to 0.18) | 54% | 0.33 | 0.16 | 3.1 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 15m | utc | 60m | session | 2192 | 3% | 3% | 0.03 (-0.15 to 0.24) | 52% | 0.53 | 0.47 | 0.6 | fails: trend days don't behave differently enough |
| SOL/USD | 15m | utc | 90m | 4h | 2192 | 4% | 13% | 0.06 (-0.04 to 0.16) | 51% | 0.33 | 0.16 | 4.7 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 15m | utc | 90m | session | 2192 | 4% | 13% | -0.06 (-0.28 to 0.14) | 48% | 0.70 | 0.43 | 3.4 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 15m | us_open | 30m | 4h | 2191 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 15m | us_open | 30m | session | 2191 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SOL/USD | 15m | us_open | 60m | 4h | 2192 | 4% | 3% | 0.07 (-0.01 to 0.15) | 57% | 0.28 | 0.25 | 0.5 | fails: trend days don't behave differently enough |
| SOL/USD | 15m | us_open | 60m | session | 2192 | 4% | 3% | 0.03 (-0.12 to 0.18) | 56% | 0.57 | 0.46 | 1.2 | fails: trend days don't behave differently enough |
| SOL/USD | 15m | us_open | 90m | 4h | 2192 | 6% | 10% | 0.06 (-0.01 to 0.14) | 54% | 0.31 | 0.20 | 3.7 | partial: trend days move more, but not reliably in the trend's direction |
| SOL/USD | 15m | us_open | 90m | session | 2192 | 6% | 10% | 0.14 (0.01 to 0.29) | 61% | 0.59 | 0.40 | 3.2 | works: trend days keep going and move more than balance days |
| SUI/USD | 1m | utc | 30m | 4h | 1205 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 1m | utc | 30m | session | 1205 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 1m | utc | 60m | 4h | 1205 | 3% | 44% | 0.05 (-0.06 to 0.17) | 49% | 0.26 | 0.15 | 2.5 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | 1m | utc | 60m | session | 1205 | 3% | 44% | 0.02 (-0.19 to 0.22) | 49% | 0.53 | 0.40 | 2.2 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | 1m | utc | 90m | 4h | 1205 | 2% | 51% | 0.06 (-0.07 to 0.20) | 52% | 0.23 | 0.15 | 1.4 | too few sessions to judge |
| SUI/USD | 1m | utc | 90m | session | 1205 | 2% | 51% | 0.06 (-0.16 to 0.29) | 41% | 0.49 | 0.41 | 1.0 | too few sessions to judge |
| SUI/USD | 1m | us_open | 30m | 4h | 1204 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 1m | us_open | 30m | session | 1204 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 1m | us_open | 60m | 4h | 1204 | 2% | 36% | -0.05 (-0.18 to 0.09) | 42% | 0.28 | 0.17 | 2.6 | too few sessions to judge |
| SUI/USD | 1m | us_open | 60m | session | 1204 | 2% | 36% | 0.14 (-0.15 to 0.46) | 54% | 0.56 | 0.40 | 1.5 | too few sessions to judge |
| SUI/USD | 1m | us_open | 90m | 4h | 1204 | 2% | 41% | 0.07 (-0.02 to 0.17) | 57% | 0.18 | 0.17 | 0.3 | too few sessions to judge |
| SUI/USD | 1m | us_open | 90m | session | 1204 | 2% | 41% | 0.17 (-0.14 to 0.55) | 52% | 0.60 | 0.38 | 1.7 | too few sessions to judge |
| SUI/USD | 5m | utc | 30m | 4h | 1205 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 5m | utc | 30m | session | 1205 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 5m | utc | 60m | 4h | 1205 | 4% | 32% | -0.02 (-0.10 to 0.08) | 42% | 0.24 | 0.15 | 2.4 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | 5m | utc | 60m | session | 1205 | 4% | 32% | 0.01 (-0.17 to 0.18) | 48% | 0.51 | 0.42 | 1.5 | fails: trend days don't behave differently enough |
| SUI/USD | 5m | utc | 90m | 4h | 1205 | 3% | 41% | 0.01 (-0.09 to 0.14) | 45% | 0.23 | 0.15 | 1.8 | fails: trend days don't behave differently enough |
| SUI/USD | 5m | utc | 90m | session | 1205 | 3% | 41% | -0.01 (-0.23 to 0.21) | 40% | 0.56 | 0.42 | 1.9 | fails: trend days don't behave differently enough |
| SUI/USD | 5m | us_open | 30m | 4h | 1204 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 5m | us_open | 30m | session | 1204 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 5m | us_open | 60m | 4h | 1204 | 4% | 25% | -0.00 (-0.09 to 0.09) | 52% | 0.23 | 0.18 | 1.8 | fails: trend days don't behave differently enough |
| SUI/USD | 5m | us_open | 60m | session | 1204 | 4% | 25% | 0.04 (-0.14 to 0.24) | 48% | 0.46 | 0.40 | 0.9 | fails: trend days don't behave differently enough |
| SUI/USD | 5m | us_open | 90m | 4h | 1204 | 3% | 33% | 0.07 (-0.03 to 0.16) | 57% | 0.24 | 0.17 | 1.9 | fails: trend days don't behave differently enough |
| SUI/USD | 5m | us_open | 90m | session | 1204 | 3% | 33% | 0.15 (-0.09 to 0.42) | 51% | 0.57 | 0.38 | 1.9 | fails: trend days don't behave differently enough |
| SUI/USD | 15m | utc | 30m | 4h | 1205 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 15m | utc | 30m | session | 1205 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 15m | utc | 60m | 4h | 1205 | 4% | 4% | 0.05 (-0.04 to 0.15) | 53% | 0.24 | 0.17 | 1.9 | fails: trend days don't behave differently enough |
| SUI/USD | 15m | utc | 60m | session | 1205 | 4% | 4% | 0.05 (-0.15 to 0.25) | 49% | 0.54 | 0.38 | 1.7 | fails: trend days don't behave differently enough |
| SUI/USD | 15m | utc | 90m | 4h | 1205 | 4% | 13% | -0.02 (-0.10 to 0.09) | 45% | 0.24 | 0.16 | 2.4 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | 15m | utc | 90m | session | 1205 | 4% | 13% | -0.06 (-0.24 to 0.11) | 40% | 0.53 | 0.39 | 2.1 | partial: trend days move more, but not reliably in the trend's direction |
| SUI/USD | 15m | us_open | 30m | 4h | 1204 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 15m | us_open | 30m | session | 1204 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| SUI/USD | 15m | us_open | 60m | 4h | 1204 | 3% | 3% | 0.00 (-0.09 to 0.10) | 50% | 0.22 | 0.18 | 0.9 | fails: trend days don't behave differently enough |
| SUI/USD | 15m | us_open | 60m | session | 1204 | 3% | 3% | 0.01 (-0.20 to 0.24) | 44% | 0.47 | 0.36 | 1.1 | fails: trend days don't behave differently enough |
| SUI/USD | 15m | us_open | 90m | 4h | 1204 | 5% | 12% | 0.07 (-0.00 to 0.14) | 54% | 0.21 | 0.19 | 0.9 | fails: trend days don't behave differently enough |
| SUI/USD | 15m | us_open | 90m | session | 1204 | 5% | 12% | 0.20 (0.00 to 0.43) | 54% | 0.59 | 0.40 | 2.2 | works: trend days keep going and move more than balance days |
| XRP/USD | 1m | utc | 30m | 4h | 3007 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 1m | utc | 30m | session | 3007 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 1m | utc | 60m | 4h | 3007 | 2% | 45% | -0.02 (-0.11 to 0.08) | 52% | 0.23 | 0.14 | 2.6 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 1m | utc | 60m | session | 3007 | 2% | 45% | -0.08 (-0.41 to 0.27) | 54% | 0.74 | 0.41 | 2.6 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 1m | utc | 90m | 4h | 3007 | 2% | 55% | 0.01 (-0.09 to 0.11) | 46% | 0.27 | 0.13 | 3.9 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 1m | utc | 90m | session | 3007 | 2% | 55% | -0.08 (-0.35 to 0.21) | 44% | 0.74 | 0.41 | 3.3 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 1m | us_open | 30m | 4h | 3005 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 1m | us_open | 30m | session | 3005 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 1m | us_open | 60m | 4h | 3006 | 2% | 45% | 0.03 (-0.05 to 0.11) | 52% | 0.22 | 0.18 | 1.5 | fails: trend days don't behave differently enough |
| XRP/USD | 1m | us_open | 60m | session | 3006 | 2% | 45% | -0.03 (-0.21 to 0.13) | 48% | 0.44 | 0.38 | 1.1 | fails: trend days don't behave differently enough |
| XRP/USD | 1m | us_open | 90m | 4h | 3006 | 2% | 53% | 0.03 (-0.06 to 0.13) | 57% | 0.23 | 0.17 | 1.6 | fails: trend days don't behave differently enough |
| XRP/USD | 1m | us_open | 90m | session | 3006 | 2% | 53% | -0.03 (-0.25 to 0.16) | 55% | 0.50 | 0.39 | 1.4 | fails: trend days don't behave differently enough |
| XRP/USD | 5m | utc | 30m | 4h | 3008 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 5m | utc | 30m | session | 3008 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 5m | utc | 60m | 4h | 3008 | 3% | 28% | 0.02 (-0.05 to 0.10) | 58% | 0.23 | 0.14 | 3.4 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 5m | utc | 60m | session | 3008 | 3% | 28% | 0.15 (-0.12 to 0.46) | 59% | 0.75 | 0.41 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 5m | utc | 90m | 4h | 3008 | 2% | 42% | 0.01 (-0.07 to 0.09) | 47% | 0.25 | 0.14 | 3.8 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 5m | utc | 90m | session | 3008 | 2% | 42% | 0.14 (-0.14 to 0.47) | 49% | 0.78 | 0.40 | 3.0 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 5m | us_open | 30m | 4h | 3006 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 5m | us_open | 30m | session | 3006 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 5m | us_open | 60m | 4h | 3007 | 3% | 32% | 0.05 (-0.01 to 0.11) | 49% | 0.24 | 0.18 | 1.9 | fails: trend days don't behave differently enough |
| XRP/USD | 5m | us_open | 60m | session | 3007 | 3% | 32% | 0.09 (-0.06 to 0.24) | 52% | 0.52 | 0.38 | 2.1 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 5m | us_open | 90m | 4h | 3007 | 3% | 43% | 0.06 (-0.03 to 0.15) | 56% | 0.29 | 0.17 | 3.0 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 5m | us_open | 90m | session | 3007 | 3% | 43% | 0.11 (-0.10 to 0.34) | 56% | 0.66 | 0.39 | 3.1 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 15m | utc | 30m | 4h | 3008 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 15m | utc | 30m | session | 3008 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 15m | utc | 60m | 4h | 3008 | 2% | 3% | 0.02 (-0.06 to 0.10) | 61% | 0.24 | 0.15 | 2.6 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 15m | utc | 60m | session | 3008 | 2% | 3% | 0.08 (-0.19 to 0.41) | 49% | 0.77 | 0.39 | 2.8 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 15m | utc | 90m | 4h | 3008 | 4% | 13% | 0.04 (-0.04 to 0.14) | 47% | 0.29 | 0.15 | 3.7 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 15m | utc | 90m | session | 3008 | 4% | 13% | 0.03 (-0.19 to 0.27) | 47% | 0.78 | 0.42 | 3.8 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 15m | us_open | 30m | 4h | 3006 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 15m | us_open | 30m | session | 3006 | 0% | 0% | n/a (n/a to n/a) | n/a | n/a | n/a | n/a | too few sessions to judge |
| XRP/USD | 15m | us_open | 60m | 4h | 3007 | 3% | 4% | 0.02 (-0.04 to 0.09) | 45% | 0.25 | 0.15 | 3.5 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 15m | us_open | 60m | session | 3007 | 3% | 4% | 0.11 (-0.04 to 0.27) | 52% | 0.51 | 0.37 | 1.8 | fails: trend days don't behave differently enough |
| XRP/USD | 15m | us_open | 90m | 4h | 3007 | 4% | 14% | 0.04 (-0.03 to 0.11) | 54% | 0.26 | 0.17 | 3.1 | partial: trend days move more, but not reliably in the trend's direction |
| XRP/USD | 15m | us_open | 90m | session | 3007 | 4% | 14% | 0.07 (-0.09 to 0.24) | 52% | 0.62 | 0.42 | 2.9 | partial: trend days move more, but not reliably in the trend's direction |
