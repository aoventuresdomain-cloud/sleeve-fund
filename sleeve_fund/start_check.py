"""START CHECKLIST (2), HoE 13 Oct: the server's read-only check before Strategy testing starts, and for HoQA's final
round. One page, pass or fail:

1. Keys. No live venue credential is set in this environment, and the supervisor reports no live key for any account.
   The demo mirror's demo and testnet keys are allowed (sleeve_fund.mirror, demo hosts only).
2. Venues. Only the paper account exists and every Strategy runs on it. A live account set up, or a Strategy assigned
   to one, fails.
3. Books. For every Strategy the dashboard's book counts, its open position and cash (the latest equity mark, which
   the dashboard reads) match the journal (fills, the source of truth) within the reconcile's tolerances, and so does
   the mark's equity, which Book, P&L and Return show. If every Strategy with fills had a fill after its latest mark,
   nothing was compared and the check fails.

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


def journal_kind(url: str) -> str:
    """Which journal was read, without its address: the page goes to a public workflow log, so no host, user or
    database name."""
    return "Postgres" if make_url(url).get_backend_name() in ("postgres", "postgresql") else "SQLite"


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


def _late_cash(store, name: str, since) -> str | None:
    """A funding settlement or true-up, or an insurance-fund cover, that moved the journal's cash after the latest
    mark (CR F276-1): the mark's cash is behind it until the next mark, as with a late fill."""
    from sleeve_fund.store import funding_t, insurance_t

    with store.engine.connect() as c:
        for what, t in (("a funding payment", funding_t), ("an insurance-fund cover", insurance_t)):
            moved = store._cash_moved_at(t)  # a true-up moves cash at its arrival (CASH-4)
            if c.execute(select(t.c.id).where(t.c.sleeve == name, moved > since).limit(1)).first() is not None:
                return what
    return None


def _qty(x):
    """A position for the page (F276-4): Postgres NUMERIC hands it back as a Decimal at its full scale, so a flat one
    would read 0e-18 and 0.1 would read 0.100000000000; a float's -0 reads 0."""
    return 0 if x == 0 else float(x)


def check_books(store) -> Section:
    """The dashboard's figures against the journal, for exactly the Strategies the dashboard's book counts
    (app.current_sleeves: every one but an earlier book's, store.previous_book, which holds the reset runs; an archived
    Strategy still holding stays in the book and is checked). Position and cash come from the latest equity mark
    (book.sleeve_extras); Book, P&L and Return from that mark's own equity column (metrics.sleeve_summary), which must
    be the journal's cash plus its position at the mark's price (the identity Strategy._mark writes)."""
    from sleeve_fund import markets

    sec = Section("Dashboard against the journal")
    earlier = store.previous_book()
    checked = compared = 0
    late_with_fills = []
    for s in store.sleeves():
        if s.name in earlier:
            continue
        checked += 1
        book = store.journal_book(s.name, s.starting_balance)
        last = store.last_equity(s.name)
        if last is None:
            if abs(book["qty"]) > 1e-12 or book["fills"]:
                sec.add(FAIL, f"{s.name}: the journal has {book['fills']} fills and holds "
                              f"{_qty(book['qty']):.12g}, and the dashboard has no mark to show it")
            else:
                sec.add(NOTE, f"{s.name}: not marked yet; the journal is flat with no fills")
            continue
        fills = store.fills(s.name, limit=1)
        # A fill in the mark's own second is late too: the runtime stamps both to the second, and the tick that sends
        # a flatten marks before its fill is booked (MARK-ON-STOP), so a tie can't show the mark saw the fill.
        late = "a fill" if fills and fills[0]["ts"] >= last["ts"] else _late_cash(store, s.name, last["ts"])
        if late:
            late_with_fills.append(f"{s.name} had {late}")
            sec.add(NOTE, f"{s.name}: {late} landed after its latest mark ({last['ts']:%H:%M:%S} UTC); compare "
                          "again after the next mark")
            continue
        compared += 1
        # The reconcile's tolerances (SleeveRuntime.reconcile): a cent per fill for cash, float steps for the qty.
        cash_tol = 0.01 * (1 + 2 * book["fills"])
        scale = max(abs(float(last["qty"])), abs(float(book["qty"])))
        qty_tol = max(1e-8, (2 + book["fills"]) * math.ulp(scale))
        d_qty = float(Decimal(repr(float(last["qty"]))) - Decimal(repr(float(book["qty"]))))
        d_cash = float(last["cash"]) - float(book["cash"])
        price, equity = float(last["price"]), float(last["equity"])
        want = float(book["cash"]) + float(book["qty"]) * price
        equity_tol = cash_tol + qty_tol * abs(price) + 4 * math.ulp(max(abs(want), abs(equity)))
        what = (f"{s.name}: position {_qty(last['qty']):.12g} vs journal {_qty(book['qty']):.12g}; "
                f"cash {last['cash']:,.2f} vs journal {book['cash']:,.2f} ({book['fills']} fills)")
        if abs(d_qty) > qty_tol or abs(d_cash) > cash_tol:
            sec.add(FAIL, what)
        elif equity == 0.0 and want < 0 and markets.is_perp(s.params):
            # Isolated margin: a Strategy wiped out can't lose more than it had, so its mark is floored at zero. A
            # spot mark is never floored, so there a zero against a negative journal figure fails below.
            sec.add(NOTE, f"{what}; equity 0.00, floored from {want:,.2f}: its isolated margin is wiped out")
        elif abs(equity - want) > equity_tol:
            sec.add(FAIL, f"{what}; but equity {equity:,.2f} vs journal {want:,.2f} (its cash plus its position at "
                          f"{price:,.2f}): the Book and P&L shown are wrong")
        else:
            sec.add(PASS, f"{what}; equity {equity:,.2f}")
    if not checked:
        sec.add(NOTE, "no Strategy to compare")
    elif not compared and late_with_fills:
        # Every Strategy with a book was skipped: nothing was actually checked, which is not a pass.
        sec.add(FAIL, f"nothing was compared: {'; '.join(late_with_fills)} after the latest mark; run again after "
                      "the next mark")
    return sec


def run(url: str, environ: dict[str, str] | None = None) -> tuple[bool, str]:
    """The checks against the journal at url, and the one-page report."""
    from datetime import UTC, datetime

    env = dict(os.environ if environ is None else environ)
    store = read_only_store(url)
    engine = store.engine
    sections = [check_keys(env, engine), check_venues(engine), check_books(store)]
    ok = not any(s.failed for s in sections)
    out = [f"Start check: {'PASS' if ok else 'FAIL'}",
           f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC, {journal_kind(url)} journal, read-only",
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
        # The error's type only: its text can carry the database's host or a path, and the page is public.
        print(f"Start check: FAIL\n\nThe check couldn't run: {type(exc).__name__}. Run it on the server for the "
              "detail.")
        return 1
    print(page, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
