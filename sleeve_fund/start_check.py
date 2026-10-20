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
4. Assets. A multi-asset Strategy has a row for every listed asset, and every journal row is keyed to its own row's
   asset. One failing asset fails, named.
5. Alerts (A15). Alerts have somewhere to go (ALERT_WEBHOOK_URL) and an outside monitor is pinged (HEALTHCHECK_PING_URL),
   named by host only.
6. Order client (head 3). Off is a note (the simulated paper client). On, the key type (demo or live) and the host from
   sleeve_fund.exec_target, the reader the client's guard shares; a production host in demo mode, a demo host in live
   mode, an unknown host or a missing key fails, named. Key values are never shown.

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


def check_store(engine: Engine) -> Section:
    """Paper journals to Postgres only (the Independent Quant Advisor's condition on #286): SQLite has none of the row
    locks the portfolio gate's reservations and the put-away flows rely on, so paper refuses to start on it."""
    sec = Section("Journal store")
    if engine.dialect.name == "postgresql":
        sec.add(PASS, "Postgres")
    else:
        sec.add(FAIL, f"{engine.dialect.name}: paper must journal to Postgres")
    return sec


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


def check_exec_target(environ: dict[str, str], extra=()) -> Section:
    """Head 3: where the exchange order client would send orders (sleeve_fund.exec_target, the reader its guard shares):
    off is a note; on, the key type and the host, each mismatch failing, named. Key values are never shown. `extra`:
    the client's own guard lines, as (status, text), added after these (PE3's EXEC-CLIENT hook)."""
    from sleeve_fund import exec_target

    sec = Section("Order client")
    t = exec_target.read(environ)
    if t is None:
        sec.add(NOTE, exec_target.OFF)
    else:
        found = exec_target.problems(t)
        for p in found:
            sec.add(FAIL, p)
        if not found:
            sec.add(PASS, exec_target.describe(t))
            sec.add(NOTE, exec_target.SCOPE_UNCHECKED)
    for status, text in extra:
        sec.add(status, text)
    return sec


def exec_gates(environ: dict[str, str], engine: Engine) -> list[tuple[str, str]]:
    """EXEC-CLIENT's own gates (paper.exec_client.gates: the PM's drill-go record, the G2 lock), for the "Order client"
    section's extra lines. The key's permissions are read from the venue only once the client goes on (drill prep)."""
    from sleeve_fund.paper import exec_client

    s = exec_client.exec_settings(environ)
    return exec_client.gates(s, drill_go=s.on and exec_client.recorded(engine, exec_client.DRILL_GO),
                             key_permissions=None)


def check_exec(environ: dict[str, str], engine: Engine) -> Section:
    """EXEC-CLIENT (Advisor HEAD3 (1)): every line of the client's guard on its own (exec_target's host and key lines,
    then the gates), for QA and the client. The start check itself shows these once, in check_exec_target's section."""
    from sleeve_fund.paper import exec_client

    sec = Section("Exchange order client")
    s = exec_client.exec_settings(environ)
    for status, what in exec_client.guard(s, drill_go=s.on and exec_client.recorded(engine, exec_client.DRILL_GO),
                                          key_permissions=None):
        sec.add(status, what)
    return sec


def check_alerts(environ: dict[str, str]) -> Section:
    """A15 (G2): where alerts go and the uptime ping an outside monitor watches, so a dead server says so. Hosts only:
    a webhook's path or a ping's id is its token, and the page is public."""
    from sleeve_fund.alerts import where

    sec = Section("Alerts")
    for var, what in (("ALERT_WEBHOOK_URL", "alerts go to"), ("HEALTHCHECK_PING_URL", "uptime pings go to")):
        url = (environ.get(var) or "").strip()
        sec.add(PASS, f"{what} {where(url)}") if url else sec.add(FAIL, f"not set: {var}")
    return sec


def check_account_sharing(environ: dict[str, str], engine: Engine) -> Section:
    """A7: with one account per Strategy on (ACCOUNT_PER_STRATEGY=on), every live or test-network account is one
    Strategy's; each shared one FAILs, named with its Strategies. Off, it says so and passes."""
    from sleeve_fund.accounts import per_strategy
    from sleeve_fund.store import (
        OPEN_ORDER_STATUSES,
        accounts_t,
        is_backtest,
        orders_t,
        sleeve_accounts_t,
        sleeve_archive_t,
        sleeves_t,
    )

    sec = Section("One account per Strategy")
    if not per_strategy(environ):
        sec.add(NOTE, "off (ACCOUNT_PER_STRATEGY): Strategies may share an account")
        return sec
    s, a, link = sleeves_t.c, accounts_t.c, sleeve_accounts_t.c
    with engine.connect() as c:
        rows = c.execute(select(link.account, link.sleeve, s.parent).join(sleeves_t, s.name == link.sleeve)
                         .join(accounts_t, a.name == link.account).where(a.kind == "live")).all()
        # put away, out of the book and no order open (Advisor (b)); a withdrawal is only ever made flat (F-C3B-A)
        released = set(c.execute(select(sleeve_archive_t.c.sleeve)
                                 .where(sleeve_archive_t.c.withdrawn.is_not(None))).scalars())
        released -= set(c.execute(select(orders_t.c.sleeve).where(orders_t.c.status.in_(OPEN_ORDER_STATUSES)))
                        .scalars())
    by: dict[str, set[str]] = {}
    for acc, row, parent in rows:
        if not is_backtest(row) and row not in released:  # put away and out of the book: released (A7 (b))
            by.setdefault(acc, set()).add(parent or row)
    for acc, who in sorted(by.items()):
        if len(who) > 1:
            sec.add(FAIL, f"account {acc} is shared by {', '.join(sorted(who))}: their new entries are refused "
                          "(exits and stops run) until each has its own")
    if not sec.failed:
        sec.add(PASS, "every live or test-network account trades for one Strategy")
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


def build_model(strategy: str, venue: str, instrument: str, params: dict, bar_spec: str):
    """A Strategy's model built as the engine builds it, from its model settings alone (store.model_params), and the
    units the warm-up seed check seeds: its decision candles' and each slower timeframe's (A10)."""
    from sleeve_fund.dashboard.charts import spec_minutes
    from sleeve_fund.instruments import spot_pair
    from sleeve_fund.store import model_params
    from sleeve_fund.strategies.series import build
    from sleeve_fund.venues import venue as venue_profile

    profile = venue_profile(venue)
    base, quote = instrument.split("/")
    inst = spot_pair(base, quote, fees=profile.fees, venue=profile.venue, price_precision=8)
    model = build(strategy, inst, model_params(params), spec_minutes(bar_spec))
    return model, model.seeding_units() + [u for us in model.seeding_timeframes().values() for u in us]


def check_warmups(store) -> Section:
    """A10 (Independent Quant Advisor, 15 Oct 07:02 UK): every Strategy in the book is built as the engine builds it and
    each indicator it decides on is classified for the warm-up's seed-twice check (strategies.indicators.seeding). One
    the check can't judge, a recursive average it doesn't know or one too long to settle within it, is a FAIL: the
    Strategy doesn't go to paper until it is covered."""
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.strategies.indicators import seeding

    sec = Section("Warm-up seed check")
    for s in store.sleeves():
        if s.lifecycle == "retired" or s.strategy not in REGISTRY:
            continue
        try:
            _, units = build_model(s.strategy, s.venue, s.instrument, s.params, s.bar_spec)
        except Exception as exc:  # noqa: BLE001 - said, as a failure: an unbuilt Strategy isn't checked
            sec.add(FAIL, f"{s.name}: couldn't be built to check its indicators ({exc})")
            continue
        for label, unit in units:
            for kind in seeding.uncovered(unit):
                sec.add(FAIL, f"{s.name}: {label} ({kind}) is a recursive average the warm-up seed check doesn't cover "
                              "(A10); it doesn't go to paper until it is")
            for why in seeding.too_long(unit):
                sec.add(FAIL, f"{s.name}: {label}: {why}")
    if not sec.failed:
        sec.add(PASS, "every indicator a Strategy decides on is covered by the warm-up seed check")
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


CENT = 0.01  # NIGHTLY-RECON (Advisor): a gap of a cent or more between the dashboard and the journal alerts


def check_books(store, cent: bool = False) -> Section:
    """The dashboard's figures against the journal, for exactly the Strategies the dashboard's book counts
    (app.current_sleeves: every one but an earlier book's, store.previous_book, which holds the reset runs; an archived
    Strategy still holding stays in the book and is checked). Position and cash come from the latest equity mark
    (book.sleeve_extras); Book, P&L and Return from that mark's own equity column (metrics.sleeve_summary), which must
    be the journal's cash plus its position at the mark's price (the identity Strategy._mark writes).

    cent (NIGHTLY-RECON): cash and equity must match the journal to under a CENT, not to the reconcile's cent per
    fill. A Strategy withdrawn from the book with dust no order can close is listed as known (Advisor, FC3BA-28B):
    the dust left with it, so it is no gap."""
    from sleeve_fund import markets

    sec = Section("Dashboard against the journal")
    earlier = store.previous_book()
    withdrawn = store.withdrawn()
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
        late = "a fill" if fills and fills[0]["ts"] > last["ts"] else _late_cash(store, s.name, last["ts"])
        if late:
            late_with_fills.append(f"{s.name} had {late}")
            sec.add(NOTE, f"{s.name}: {late} landed after its latest mark ({last['ts']:%H:%M:%S} UTC); compare "
                          "again after the next mark")
            continue
        # The reconcile's tolerances (SleeveRuntime.reconcile): a cent per fill for cash, float steps for the qty.
        cash_tol = CENT * (1 - 1e-6) if cent else 0.01 * (1 + 2 * book["fills"])
        scale = max(abs(float(last["qty"])), abs(float(book["qty"])))
        qty_tol = max(1e-8, (2 + book["fills"]) * math.ulp(scale))
        d_qty = float(Decimal(repr(float(last["qty"]))) - Decimal(repr(float(book["qty"]))))
        d_cash = float(last["cash"]) - float(book["cash"])
        if fills and fills[0]["ts"] == last["ts"] and (abs(d_qty) > qty_tol or abs(d_cash) > cash_tol):
            # A fill in the mark's own second (the runtime stamps both to the second) that the mark doesn't hold: the
            # tick that sends a flatten marks before its fill is booked (MARK-ON-STOP). Not a wrong book, and not
            # a skip that makes "nothing compared": a tie the mark does hold is compared below, as a tick that books
            # a fill and then marks in one second is the usual case (HALT-REF-1's re-base reads it). A FAIL, never a
            # NOTE (F289-2, HoE 18:48 UK 14 Oct): nothing is signed or re-based off a book the dashboard can't show yet.
            sec.add(FAIL, f"{s.name}: its latest mark predates a fill in the same second ({last['ts']:%H:%M:%S} UTC): "
                          "compare again after the next mark")
            continue
        compared += 1
        price, equity = float(last["price"]), float(last["equity"])
        want = float(book["cash"]) + float(book["qty"]) * price
        equity_tol = cash_tol + qty_tol * abs(price) + 4 * math.ulp(max(abs(want), abs(equity)))
        equity_tol = min(equity_tol, CENT * (1 - 1e-6)) if cent else equity_tol
        what = (f"{s.name}: position {_qty(last['qty']):.12g} vs journal {_qty(book['qty']):.12g}; "
                f"cash {last['cash']:,.2f} vs journal {book['cash']:,.2f} ({book['fills']} fills)")
        if abs(d_qty) > qty_tol:
            sec.add(FAIL, f"{what}: the mark doesn't hold the journal's position: it predates or misses a fill")
        elif abs(d_cash) > cash_tol:
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
        if s.name in withdrawn and abs(book["qty"]) > 1e-12:
            sec.add(NOTE, f"{s.name}: known, it left the fund's book holding dust of {_qty(book['qty']):.12g} no "
                          "order can close, which left with it")
    if not checked:
        sec.add(NOTE, "no Strategy to compare")
    elif not compared and late_with_fills:
        # Every Strategy with a book was skipped: nothing was actually checked, which is not a pass.
        sec.add(FAIL, f"nothing was compared: {'; '.join(late_with_fills)} after the latest mark; run again after "
                      "the next mark")
    return sec


def check_maintenance(store, now=None) -> Section:
    """VS-1 (Advisor, 15 Oct): a perp Strategy's liquidation is priced at its venue's published maintenance tiers for
    the instrument. One on a pair whose table isn't kept is a FAIL, named, never run at an assumed rate. A table read
    more than 30 days ago is noted, once per pair (VS-1-TABLE; the supervisor's alerts say it too)."""
    from datetime import UTC, datetime

    from sleeve_fund import markets

    sec, live = Section("Maintenance tables"), []
    for s in store.sleeves():
        if s.lifecycle == "retired" or (s.params or {}).get("market") != "perp":
            continue
        try:
            markets.terms(s.params, s.venue, s.instrument)
        except markets.NoMaintenanceTable as exc:
            sec.add(FAIL, f"{s.name}: {exc}")
            continue
        live.append(s)
    for line in markets.stale_maintenance(live, (now or datetime.now(UTC)).date()):
        sec.add(NOTE, line)
    if not sec.failed:
        sec.add(PASS, "every perp Strategy's instrument has its venue's maintenance tiers")
    return sec


def _live_tick(venue: str, pair: str):
    """The venue's live contract tick for `pair`, from its public listing (no key)."""
    from sleeve_fund.venues import venue as venue_profile

    profile = venue_profile(venue)
    if profile.live_tick is None:
        raise ValueError("this venue has no way to read its contract tick")
    return profile.live_tick(pair)


def check_price_ticks(store, live_tick=None) -> Section:
    """B-36b (HoE, 20 Oct): a perp Strategy's prices round to the contract tick kept beside its tiers
    (markets.price_tick). One whose venue's live tick differs from the kept one is a FAIL, named, with both ticks, as is
    one whose tick isn't kept or can't be read: never run on an assumed tick (the VS-1 pattern). One line per pair."""
    from decimal import Decimal

    from sleeve_fund import markets
    from sleeve_fund.venues import DEFAULT_VENUE

    read = live_tick or _live_tick
    sec, seen = Section("Price ticks"), {}
    for s in store.sleeves():
        if s.lifecycle == "retired" or (s.params or {}).get("market") != "perp":
            continue
        venue = getattr(s, "venue", None) or DEFAULT_VENUE
        kept = markets.price_tick(venue, s.instrument)
        if kept is None:
            sec.add(FAIL, f"{s.name}: {s.instrument} has no price tick kept, so its prices can't be rounded as the venue "
                          "does: refused until the venue's tick is checked in with its tiers")
            continue
        key = (venue, s.instrument)
        if key not in seen:
            try:
                seen[key] = Decimal(str(read(venue, s.instrument))).normalize()
            except Exception as exc:  # noqa: BLE001 - an unreachable venue is a FAIL, never an assumed tick
                seen[key] = exc
        live = seen[key]
        if isinstance(live, Exception):
            sec.add(FAIL, f"{s.name}: the venue's price tick for {s.instrument} couldn't be read ({live}), so the kept "
                          f"tick {kept} is unconfirmed")
        elif live != kept:
            sec.add(FAIL, f"{s.name}: the venue's price tick for {s.instrument} is now {live}, not the kept {kept}: "
                          "refused until the kept tick is checked in again")
    if not sec.failed:
        sec.add(PASS, "every perp Strategy's instrument trades at the price tick kept for it")
    return sec


def _hub_lot(venue: str, pair: str):
    """The (step, minimum) paper's instrument carries: the data venue's contract, as the hub relays its definition."""
    from sleeve_fund.venues import venue as venue_profile

    c = venue_profile(venue).contract(pair)
    return Decimal(10) ** -int(c["size_precision"]), Decimal(str(c["min_quantity"])).normalize()


def _paper_lot(venue: str, pair: str):
    """The paper venue's listed (step, minimum) for `pair` (B-44), from its public listing (no key)."""
    from sleeve_fund.venues import venue as venue_profile

    profile = venue_profile(venue)
    if profile.live_lot is None:
        raise ValueError("this venue has no way to read its lot size")
    return profile.live_lot(pair)


def check_lot_sizes(store, hub_lot=None, paper_lot=None) -> Section:
    """B-44 (HoE 23:19 UK, option ii; Advisor): paper sizes on the hub's instrument (the data venue's definition), so a
    perp Strategy whose hub step or minimum differs from the paper venue's listed lot is a FAIL, named, with both lots,
    as is one whose listed step the engine can't round to (not a power of ten) or whose lots can't be read: fail-closed,
    as the price tick. One line per pair. shortcut: paper doesn't read the paper venue's lot itself; a MUST before G2."""
    from sleeve_fund.venues import DEFAULT_VENUE, LotStepUnusable

    hub_read, paper_read = hub_lot or _hub_lot, paper_lot or _paper_lot
    sec, seen = Section("Lot sizes"), {}
    for s in store.sleeves():
        if s.lifecycle == "retired" or (s.params or {}).get("market") != "perp":
            continue
        venue = getattr(s, "venue", None) or DEFAULT_VENUE
        key = (venue, s.instrument)
        if key not in seen:
            try:
                hub = tuple(Decimal(str(x)).normalize() for x in hub_read(venue, s.instrument))
                listed = tuple(Decimal(str(x)).normalize() for x in paper_read(venue, s.instrument))
                seen[key] = (hub, listed)
            except LotStepUnusable as exc:
                seen[key] = f"the paper venue lists a quantity step of {exc.step}, not a power of ten, which the engine " \
                            "can't round to"
            except Exception as exc:  # noqa: BLE001 - an unreachable venue is a FAIL, never an assumed lot
                seen[key] = f"the lot size couldn't be read ({exc})"
        got = seen[key]
        if isinstance(got, str):
            sec.add(FAIL, f"{s.name}: for {s.instrument}, {got}: refused until it can be")
        elif got[0] != got[1]:
            (hs, hm), (ls, lm) = got
            sec.add(FAIL, f"{s.name}: {s.instrument} sizes on step {hs}, minimum {hm}, but the paper venue lists step "
                          f"{ls}, minimum {lm}: refused until they match")
    if not sec.failed:
        sec.add(PASS, "every perp Strategy's instrument sizes on the paper venue's listed lot")
    return sec


def check_venue_terms(store, now=None) -> Section:
    """FEE-LIVE (Advisor FEE-LIVE-DW): each perp Strategy's instrument's fee rates and margin tiers, as fetched from the
    venue. Over 72 h old, or never fetched, is a FAIL (its entries are refused; exits and stops run); over 24 h, or the
    published rate standing in for the account's, is a NOTE."""
    from datetime import UTC, datetime

    from sleeve_fund import venue_terms

    sec = Section("Venue fees and tiers")
    for t in venue_terms.live_terms(store, now or datetime.now(UTC)):
        if t.state == venue_terms.REFUSED:
            sec.add(FAIL, t.words)
        elif t.state == venue_terms.STALE or t.source == venue_terms.PUBLISHED_FEES:
            sec.add(NOTE, t.words)
    if not sec.failed:
        sec.add(PASS, "every followed Strategy's instrument has its fee rates (and on a perp its tiers) within 72 hours")
    return sec


def check_account_capital(store, environ=None, now=None) -> Section:
    """B-23 (Advisor B23-ACCOUNT-CAPITAL): the paper account against the book, as the dashboard shows it, and "Size from
    the account figure", which ships OFF. On paper both are NOTEs: sizing is from the book and a failed or stale read
    only warns. ON with a live account holding a Strategy is a FAIL: sizing from the account isn't wired into the
    entry check yet (after G2), so it must not look switched on."""
    from datetime import UTC, datetime

    from sleeve_fund import account_capital

    sec = Section("Account capital")
    now = now or datetime.now(UTC)
    sec.add(NOTE, account_capital.view(store, "paper", now).words)
    live = sorted({a for a in (store.account_of(s.name) for s in store.sleeves()) if store.account_kind(a) == "live"})
    takes_effect, words = account_capital.setting_words(environ, bool(live))
    if takes_effect:
        sec.add(FAIL, f"{words}, but it isn't wired into the entry check yet (after G2), so it is refused: switch "
                      f"{account_capital.SETTING_ENV} off ({', '.join(live)})")
    else:
        sec.add(NOTE, words)
    return sec


def check_assets(store) -> Section:
    """A multi-asset Strategy asset by asset (multi-asset spec §7): every listed asset has its row, and every journal
    row is keyed to its own row's asset (DA 0028's reconcile, store.asset_mismatches). One failing asset fails, named."""
    sec = Section("Multi-asset Strategies")
    for s in store.sleeves(include_legs=False):
        want = list((s.params or {}).get("assets") or [])
        if not want:
            continue
        have = [a["instrument"] for a in store.strategy_assets(s.name)]
        missing, extra = [a for a in want if a not in have], [a for a in have if a not in want]
        for a in missing:
            sec.add(FAIL, f"{s.name}: {a} has no row, so that asset can't trade or be put away")
        for a in extra:
            sec.add(FAIL, f"{s.name}: a row trades {a}, which is not on its list")
        if not missing and not extra:
            sec.add(PASS, f"{s.name}: {', '.join(want)}, each with its row")
    for m in store.asset_mismatches():
        sec.add(FAIL, f"{m['table']} row {m['id']}: {m['sleeve']} is keyed to {m['instrument'] or 'nothing'}, not its "
                      f"{m['expected'] or 'row (gone)'}")
    if not sec.lines:
        sec.add(NOTE, "no multi-asset Strategy; every journal row is keyed to its own asset")
    return sec


def _build_config(sleeve) -> None:
    """Build the Strategy's config from its row as the dashboard's create check does: bad settings raise here."""
    from nautilus_trader.model import BarType, InstrumentId

    from sleeve_fund.paper.config import from_store
    from sleeve_fund.strategies import REGISTRY

    cfg = from_store(sleeve)
    _, config_cls = REGISTRY[cfg.strategy]
    config_cls(instrument_id=InstrumentId.from_str(cfg.instrument_id), bar_type=BarType.from_str(cfg.bar_type),
               assumed_taker_fee=float(cfg.fees.taker), **cfg.params)


def check_strength(store) -> Section:
    """G5: every Strategy set to size by signal strength builds the config its node will: a valid setting, not
    rebalancing to a target weight (CR F-G5-1; a band the Model injects counts, QD F-G5-3) and not a pair's leg (QD
    F-G5-4). Any of them fails, named."""
    sec = Section("Sizing by signal strength")
    for s in store.sleeves(include_legs=False):
        params = s.params or {}
        if params.get("signal_strength") is None:
            continue
        try:  # the config the node builds (QD F-G5-3): a Model may inject a rebalance band of its own
            _build_config(s)
        except Exception as exc:  # noqa: BLE001 - whatever stops it building stops the Strategy at start: named
            sec.add(FAIL, f"{s.name}: {exc}")
            continue
        sec.add(PASS, f"{s.name}: sizes by signal strength ({params['signal_strength']['mapping']})")
    if not sec.lines:
        sec.add(NOTE, "no Strategy sizes by signal strength (it is off for every one)")
    return sec


def run(url: str, environ: dict[str, str] | None = None) -> tuple[bool, str]:
    """The checks against the journal at url, and the one-page report."""
    from datetime import UTC, datetime

    env = dict(os.environ if environ is None else environ)
    store = read_only_store(url)
    engine = store.engine
    sections = [check_store(engine), check_keys(env, engine), check_exec_target(env, exec_gates(env, engine)),
                check_alerts(env), check_venues(engine), check_maintenance(store), check_price_ticks(store), check_lot_sizes(store),
                check_venue_terms(store),
                check_books(store), check_assets(store), check_warmups(store), check_account_sharing(env, engine),
                check_strength(store), check_account_capital(store, env)]
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
