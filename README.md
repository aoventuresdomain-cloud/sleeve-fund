# Sleeve Fund

A personal multi-strategy trading platform run like a small fund. The plan, gates and
milestones live in the project's `sleeve-fund-project-plan.md`; this repo is the code.

**Paper only.** Nothing here can place a real order. Paper trading uses Kraken's public
prices with fills simulated locally, needs no API keys, and refuses to start if any venue
credentials are in the environment. Live trading waits for gate G2.

## Set up (VS Code)

Needs Python 3.12 or 3.13 and Git.

```bash
git clone https://github.com/aoventuresdomain-cloud/sleeve-fund
cd sleeve-fund
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest -q
```

Open the folder in VS Code, accept the recommended Python extension, and pick `.venv` as the
interpreter. The Run and Debug panel has ready-made launch entries for everything below.

## Paper trading

```bash
# Pipeline smoke test: 1-minute bars from live Kraken trades, trades within ~30 minutes
python -m sleeve_fund.paper configs/examples/btc_trend_smoke.toml --minutes 30

# Idea #1 as it would really run: 50/200-day trend filter, warms up on Kraken history
python -m sleeve_fund.paper configs/examples/btc_trend_daily.toml
```

What happens: Kraken public WebSocket/REST feed into NautilusTrader, bars go to the strategy,
orders go to the Nautilus sandbox matching engine (fed by the same live prices), and every fill
pays Kraken's UK entry fee (0.40% maker, 0.80% taker). A sleeve is a TOML file in
`configs/sleeves/` (seeded into the database on start; `configs/examples/` holds ones that are not): strategy, instrument, bar size, paper balance, per-order cap.

## Research loop

```bash
python -m sleeve_fund study trend_filter --synthetic                     # pipeline check
python -m sleeve_fund study trend_filter --data data/XBTUSD_1440.csv     # Kraken daily CSV
python -m sleeve_fund counter                                            # idea counter
```

A study holds back the last 365 days, runs the parameter grid (sensitivity), walk-forwards
3-year train / 1-year test, compares with buy-and-hold paying the same fee, logs every variant
to the append-only idea counter, and writes a tear sheet to `research/tearsheets/`.
G1's "beats buy-and-hold" means a higher out-of-sample Sharpe after fees.

## Layout

| Path | What |
| --- | --- |
| `sleeve_fund/strategies/` | Strategies. Same class runs in backtest and paper. Each has an `IdeaSpec`. |
| `sleeve_fund/paper/` | Paper node (Kraken data + sandbox fills), sleeve config, safety guard |
| `sleeve_fund/research/` | Backtest runner, walk-forward study, metrics, idea ledger, tear sheet |
| `sleeve_fund/instruments.py` | Instruments and the fee model used everywhere |
| `configs/sleeves/` | One TOML per strategy seeded on start; `configs/clear.toml` puts the book away once per entry |

## Known limits

- NautilusTrader 2.0 is a release candidate (2.0.0rc5); the Kraken adapter only exists in 2.x.
- Backtest fills happen at the signal bar's close; spread and slippage aren't modelled yet.
- Kraken's daily bars arrive after the first trade of the next day, not exactly at midnight UTC.
- Returns are in USD, not GBP; no UK CGT.
- Not yet built (plan M2): risk guard, reconciliation, heartbeat, dead man's switch, Telegram alerts, database.
