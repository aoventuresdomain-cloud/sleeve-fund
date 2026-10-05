# Data-model standard

The Head of Engineering checks every PR that touches `sleeve_fund/store.py`, the history store or a new
data file against this page. Owner: Data Architect.

## Schema changes
- **Only through a migration.** A change to a table in `store.py` ships with a migration in
  `sleeve_fund/migrations/versions/` in the same PR. Draft one with
  `DATABASE_URL=sqlite:///scratch.db python -m sleeve_fund.schema new "add x to y"` against a database at
  the previous head, then read it and edit it by hand. `tests/test_migrations.py` fails if the migrated
  schema differs from `store.py`.
- **The deploy applies migrations** through the `migrate` service, before anything else starts. Nobody
  edits the server's database by hand. `python -m sleeve_fund.schema check` reports drift.
- **Additive first.** Add a column as nullable or with a server default, backfill it, then tighten it in a
  later migration. Never rename or drop a column in the same PR that stops using it.
- **Never destroy data in a migration** without a Data Architect review and a backup taken that day.
  `downgrade()` may raise rather than drop.

## Keys and links
- Every table has a primary key. A record that must not repeat has a unique constraint on its natural
  identity (for a fill: strategy and venue trade id), so the database refuses the second copy.
- Existing tables link to a strategy by its name (`sleeve`). **New v2 tables link by an immutable id**,
  not a name that a reset or rename can rewrite.
- Every table with a strategy column is either on the reset list or on its explicit exclude list (DA-6 test).
- Links are declared as foreign keys wherever both ends are tables.

## Types
- **Money and quantities in new columns are `Numeric`**, never `Float`. The existing float columns move
  in one migration before G2 (DA-9).
- Times are `DateTime(timezone=True)` in UTC. History bars are stamped at their open in storage and at
  their close when read.
- Enumerations are short strings validated in code. JSON columns hold display or audit payloads, never
  values another query filters on.

## Naming
- Tables are plural nouns and columns are `snake_case`. Indexes are named `<table>_<columns>`.
- Names and comments use institutional terms ("Strategy", "Model", "instrument"). Engine code contains
  no venue names.

## Lineage
- Every order records why it was sent: the intent, the reason in plain words, and the signal values.
- **Rule-builder (v2) decisions also record** the rule and settings hash, the indicator values, the data
  source and the range of bars read, and whether any bar in that range was refilled (DA-4).
- The market data hub writes history only through `HistoryStore.append_bars`. A bar's identity is
  (venue, instrument, open minute), writes are idempotent, and refills and conflicts are recorded in
  `provenance.jsonl`.
- Backtests and research never write into the paper journal, apart from a saved backtest under the
  `bt:` prefix. Parameter sweeps keep their trials elsewhere.
