"""START CHECKLIST (2), HoE 13 Oct: the server's read-only check before Strategy testing starts, and for HoQA's final
round. One page, pass or fail:

1. Keys. No live venue credential is set in this environment, and the supervisor reports no live key for any account.
   The demo mirror's demo and testnet keys are allowed (sleeve_fund.mirror, demo hosts only).
2. Venues. Only the paper account exists and every Strategy runs on it. A live account set up, or a Strategy assigned
   to one, fails.
3. Books. For every Strategy the dashboard shows, its open position and cash (the latest equity mark, which the
   dashboard reads) match the journal (fills, the source of truth) within the reconcile's tolerances.

Strictly read-only. It opens the database read-only (SQLite mode=ro; Postgres default_transaction_read_only), never
the Store's own engine, and imports nothing that places an order or opens a network connection.

    python -m sleeve_fund.start_check [--db URL]   # exit 0 pass, 1 fail; the page goes to stdout
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy import create_engine, make_url, select
from sqlalchemy.engine import Engine

PASS, FAIL, NOTE = "PASS", "FAIL", "NOTE"


@dataclass
class Section:
    title: str
    lines: list[tuple[str, str]] = field(default_factory=list)  # (PASS | FAIL | NOTE, what)

    def add(self, status: str, what: str) -> None:
        self.lines.append((status, what))

    @property
    def failed(self) -> bool:
        return any(s == FAIL for s, _ in self.lines)


def read_only_engine(url: str) -> Engine:
    """The journal opened so that nothing can be written: any INSERT, UPDATE, DELETE or DDL raises."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
        if path in ("", ":memory:") or not os.path.exists(path):
            raise FileNotFoundError(f"no journal at {path or url}")
        return create_engine(f"sqlite:///file:{os.path.abspath(path)}?mode=ro&uri=true")
    if url.startswith("postgresql+psycopg://"):
        return create_engine(url, connect_args={"options": "-c default_transaction_read_only=on"})
    raise ValueError("the journal URL is neither SQLite nor Postgres")


def read_only_store(url: str):
    """The Store's readers on the read-only engine; its constructor would create tables. The caller disposes of
    store.engine."""
    from sleeve_fund.store import Store

    store = Store.__new__(Store)
    store.engine, store.portfolio_gate, store.url = read_only_engine(url), False, None
    return store


def check_keys(environ: dict[str, str], engine: Engine) -> Section:
    from sleeve_fund.mirror import ACCOUNTS as DEMO_ACCOUNTS
    from sleeve_fund.paper.safety import credential_var
    from sleeve_fund.store import account_keys_t

    sec = Section("Live API keys")
    demo = {v for pair in DEMO_ACCOUNTS.values() for v in pair}
    found = sorted(k for k, v in environ.items() if v and v.strip() and credential_var(k) and k not in demo)
    for k in found:  # the name only, never the value
        sec.add(FAIL, f"a venue credential is set in the environment: {k}")
    with engine.connect() as c:
        held = [r.account for r in c.execute(select(account_keys_t).where(account_keys_t.c.present == 1))]
    for a in sorted(held):
        sec.add(FAIL, f"the supervisor reports a live key for account {a}")
    used = sorted(k for k in demo if environ.get(k, "").strip())
    if used:
        sec.add(NOTE, f"demo mirror keys set ({', '.join(used)}): demo and testnet hosts only")
    if not sec.failed:
        sec.add(PASS, "no live key in the environment or reported for any account")
    return sec


def check_venues(engine: Engine) -> Section:
    from sleeve_fund.accounts import PAPER
    from sleeve_fund.store import accounts_t, sleeve_accounts_t

    sec = Section("Venues and accounts")
    with engine.connect() as c:
        accounts = {r.name: r for r in c.execute(select(accounts_t))}
        links = {r.sleeve: r.account for r in c.execute(select(sleeve_accounts_t))}
    for name, r in sorted(accounts.items()):
        if r.kind != PAPER:
            sec.add(FAIL, f"account {name} is {r.kind}{f' on {r.venue}' if r.venue else ''}: only paper may exist")
    for sleeve, account in sorted(links.items()):
        if account != PAPER and (account not in accounts or accounts[account].kind != PAPER):
            sec.add(FAIL, f"Strategy {sleeve} is assigned to account {account}, not paper")
    if not sec.failed:
        sec.add(PASS, "only the paper account; every Strategy trades on it")
    return sec


def check_books(store) -> Section:
    """The dashboard's position and cash (book.sleeve_extras reads the latest equity mark) against the journal."""
    sec = Section("Dashboard against the journal")
    archived, runs = store.archived(), store.reset_runs()
    checked = 0
    for s in store.sleeves():
        if s.name in archived or s.name in runs:
            continue
        checked += 1
        book = store.journal_book(s.name, s.starting_balance)
        last = store.last_equity(s.name)
        if last is None:
            if abs(book["qty"]) > 1e-12 or book["fills"]:
                sec.add(FAIL, f"{s.name}: the journal has {book['fills']} fills and holds {book['qty']:.12g}, and "
                              "the dashboard has no mark to show it")
            else:
                sec.add(NOTE, f"{s.name}: not marked yet; the journal is flat with no fills")
            continue
        fills = store.fills(s.name, limit=1)
        if fills and fills[0]["ts"] > last["ts"]:
            sec.add(NOTE, f"{s.name}: a fill landed after its latest mark ({last['ts']:%H:%M:%S} UTC); compare "
                          "again after the next mark")
            continue
        # The reconcile's tolerances (SleeveRuntime.reconcile): a cent per fill for cash, float steps for the qty.
        cash_tol = 0.01 * (1 + 2 * book["fills"])
        scale = max(abs(float(last["qty"])), abs(float(book["qty"])))
        qty_tol = max(1e-8, (2 + book["fills"]) * math.ulp(scale))
        d_qty = float(Decimal(repr(float(last["qty"]))) - Decimal(repr(float(book["qty"]))))
        d_cash = float(last["cash"]) - float(book["cash"])
        what = (f"{s.name}: position {last['qty']:.12g} vs journal {book['qty']:.12g}; cash {last['cash']:,.2f} vs "
                f"journal {book['cash']:,.2f} ({book['fills']} fills)")
        sec.add(PASS if abs(d_qty) <= qty_tol and abs(d_cash) <= cash_tol else FAIL, what)
    if not checked:
        sec.add(NOTE, "no Strategy to compare")
    return sec


def run(url: str, environ: dict[str, str] | None = None) -> tuple[bool, str]:
    """The checks against the journal at url, and the one-page report."""
    from datetime import UTC, datetime

    env = dict(os.environ if environ is None else environ)
    store = read_only_store(url)
    engine = store.engine
    sections = [check_keys(env, engine), check_venues(engine), check_books(store)]
    ok = not any(s.failed for s in sections)
    where = make_url(url).render_as_string(hide_password=True)
    out = [f"Start check: {'PASS' if ok else 'FAIL'}", f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC, journal {where}, read-only",
           ""]
    for s in sections:
        out.append(f"{s.title}: {'FAIL' if s.failed else 'PASS'}")
        out += [f"  {st:<4}  {what}" for st, what in s.lines]
        out.append("")
    engine.dispose()
    return ok, "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    from sleeve_fund.store import DEFAULT_URL

    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.start_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=os.environ.get("DATABASE_URL", DEFAULT_URL), help="journal URL (DATABASE_URL)")
    args = ap.parse_args(argv)
    try:
        ok, page = run(args.db)
    except Exception as exc:  # noqa: BLE001 - a check that can't run is a fail, never a pass
        print(f"Start check: FAIL\n\nThe check couldn't run: {type(exc).__name__}: {exc}")
        return 1
    print(page, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
