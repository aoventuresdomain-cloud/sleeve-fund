# Sleeve Fund

A personal multi-strategy trading platform run like a small fund. The plan, gates and
milestones live in the project's `sleeve-fund-project-plan.md`; this repo is the code.

**Paper only.** The order client ships OFF and guarded, and the paper process refuses to start if any venue
credentials are in its environment (`sleeve_fund/paper/safety.py`). Research reads Binance perpetual bars and
funding (`RESEARCH_VENUE` in `sleeve_fund/venues.py`); the paper venue's instrument terms and risk-limit tiers
come from Bybit. Live trading waits for gate G2.

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

Paper runs on the deployed server (see `docker-compose.yml` and `.github/workflows/deploy.yml`). A Strategy is
created and started from the dashboard; its settings are a rules definition (`configs/definitions/*.toml`) or one of
the older hand-coded models. Fills are simulated by the NautilusTrader sandbox matching engine, fed by the market
data hub, and every fill pays the venue profile's fees. Strategies are never started from this repo by hand.

The two sample files under `configs/examples/` and the files in `configs/sleeves/` are legacy test fixtures:
`docker-compose.yml` seeds `configs/sleeves/*.toml` on every start and many tests load them, so they stay until the
tests that use them are moved.

Nightly recon (NIGHTLY-RECON): once a night after 01:00 UTC the supervisor compares every Strategy's
dashboard figures with its journal, to under a cent, and writes `recon-YYYY-MM-DD.txt` (first line PASS or
FAIL) to `RECON_DIR`; a FAIL is an alert. Deploy note: compose mounts the `recon` volume at `/data/recon`
(created on the first `docker compose up`, made writable for the app's user by `volume-init`); read a night's
record with `docker compose exec supervisor cat /data/recon/recon-YYYY-MM-DD.txt`. Unset `RECON_DIR`, it
doesn't run.

## Research loop

Studies run on the paper server's stored history, from the `study` workflow (`.github/workflows/study.yml`), which
reads `configs/definitions/*.toml` and `configs/studies/*.md` from main without a deploy. The merge rule for a new
definition is in `docs/research-definitions.md`. A study holds back the last 365 days, runs the walk-forward
(730 days training, 365 testing), charges fees, spread, slippage and funding, compares with the perpetual held at 1x,
runs the random-entry and double-fee checks, logs every variant to the trials register, and writes a tear sheet.

## Layout

| Path | What |
| --- | --- |
| `sleeve_fund/strategies/` | Strategy models. The same class runs in backtest and paper. `rules.py` runs a definition from `configs/definitions/`. |
| `sleeve_fund/paper/` | Paper node, runtime, journal, order client (OFF), safety guard |
| `sleeve_fund/research/` | Backtest runner, walk-forward study, random-entry test, metrics, trials register, tear sheet |
| `sleeve_fund/portfolio/` | Fund limits, sizing and shadow netting across Strategies |
| `sleeve_fund/hub/` | Market data hub: one per venue, writes the history store |
| `sleeve_fund/dashboard/` | The PM dashboard |
| `sleeve_fund/instruments.py`, `markets.py`, `venues.py` | Instruments, perpetual terms, venue profiles and fees |
| `configs/definitions/`, `configs/studies/` | Research definitions and study lists (merge rule: `docs/research-definitions.md`) |
| `configs/sleeves/` | Legacy test Strategies seeded on start; `configs/clear.toml` puts the book away once per entry |

## Known limits

- Paper fills are simulated; nothing has yet sent a real order. The order client, venue-resting stops and the
  drills that prove them are not done.
- Margin tiers are kept for BTCUSDT only; paper refuses other perpetuals. Studies on ETH, SOL and XRP reuse BTC's
  tiers and say so on the sheet.
- Returns are in USD, not GBP; no UK CGT.
