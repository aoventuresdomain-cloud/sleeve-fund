"""Final-head smoke (venue-swap plan §B; pe1-plans/b-smoke-runbook.md): read-only checks on the paper server after
the final head is deployed, before testing starts. Each subcommand prints its figures and one line
"<CHECK>: PASS" or "<CHECK>: FAIL (why)", and exits 1 on any FAIL. Nothing is written: selects only, a hub
subscription, and one dry check_order on an in-memory copy of the ledger (never reserved, never sent).

Runs over stdin inside a container on the server (.github/workflows/final-smoke.yml), from /app:
    python -u - safety [--mirror]                        # B6: in each container the workflow names
    python -u - hub [--wait 60]                          # B2: hub-fed trades and quotes, and the history store
    python -u - journal --strategy NAME --since ISO      # B4: B3's orders, fills, funding and events
    python -u - gate                                     # B5: the gate's book against the dashboard's tile
The round trip that B4 reads is scripts/round_trip.py, which is not read-only and is kept apart."""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time
import tomllib
from datetime import UTC, datetime, timedelta
from decimal import Decimal

MAX_AGE = 60  # seconds: a trade or quote older than this, or a hub silent this long, is stale
HUB_VENUE = "BINANCE"


def verdict(check: str, fails: list[str]) -> int:
    print(f"{check}: PASS" if not fails else f"{check}: FAIL ({'; '.join(fails)})")
    return 1 if fails else 0


def configured(pattern: str = "configs/sleeves/*.toml") -> list[tuple[str, str]]:
    """(VENUE, BASE/QUOTE) of every configured perpetual Strategy, each once (as scripts/b8_instrument_terms.py)."""
    from sleeve_fund import markets
    from sleeve_fund.venues import DEFAULT_VENUE

    out: list[tuple[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
        s, p = cfg.get("sleeve", {}), cfg.get("params", {})
        key = (str(s.get("venue") or DEFAULT_VENUE).upper(), str(s["instrument"]).upper())
        if markets.market_of(p) == markets.PERP and key not in out:
            out.append(key)
    return out


# --- B6 safety -------------------------------------------------------------------------------------------------------

def safety(environ: dict[str, str] | None = None, mirror_container: bool = False) -> int:
    """No venue credential is set in this container's environment (names only are printed, never values), and the
    demo mirror's hosts are demo hosts. In the mirror's container (mirror_container) only its demo and testnet
    accounts' names may be set (mirror.ACCOUNTS); any other credential name fails: never a live key."""
    from sleeve_fund import mirror
    from sleeve_fund.paper.safety import credential_var

    env = os.environ if environ is None else environ
    demo = {name for pair in mirror.ACCOUNTS.values() for name in pair} if mirror_container else set()
    found = sorted(k for k, v in env.items() if v and (credential_var(k) or k in demo))
    print(f"credential variables set: {found or 'none'}" + (f" (demo accounts allowed: {sorted(demo)})" if demo else ""))
    hosts = (mirror.TESTNET_HOST, mirror.BYBIT_DEMO_HOST)
    print(f"mirror hosts: {hosts}")
    live = [k for k in found if k not in demo]
    fails = [f"credential set: {', '.join(live)}"] if live else []
    fails += [f"mirror host {h} is not a demo host" for h in hosts if not h.startswith(("test.", "api-demo."))]
    return verdict("B6 safety", fails)


# --- B2 hub fresh ----------------------------------------------------------------------------------------------------

def _hub_ids(pairs: list[tuple[str, str]]) -> dict[str, str]:
    from sleeve_fund.venues import venue

    profile = venue(HUB_VENUE.lower())
    return {f"{profile.symbol_of(pair)}.{HUB_VENUE}": pair for v, pair in pairs if v == HUB_VENUE}


def hub_seen(address: str, ids: list[str], wait: float, connect=socket.create_connection, clock=time.time) -> dict:
    """Subscribe to the hub and read until a trade and a quote have come for every id, or `wait` seconds pass.
    Returns {id: {"trade": ts_ns, "quote": ts_ns}} with what came, and "hb" for the last heartbeat."""
    from sleeve_fund.hub import protocol

    host, _, port = address.rpartition(":")
    seen: dict[str, dict] = {i: {} for i in ids}
    end = clock() + wait
    with connect((host, int(port)), timeout=wait) as s:
        s.sendall(protocol.encode(protocol.subscription(ids)))
        for line in s.makefile("rb"):
            msg = protocol.decode(line)
            if msg["t"] in ("trade", "quote") and msg.get("id") in seen:
                seen[msg["id"]][msg["t"]] = int(msg["ts"])
            elif msg["t"] == "hb":
                seen.setdefault("hb", {})["ts"] = int(msg["ts"])
            elif msg["t"] == "error":
                raise RuntimeError(f"hub refused the subscription: {msg}")
            if all("trade" in seen[i] and "quote" in seen[i] for i in ids) or clock() > end:
                break
    return seen


def hub(wait: float = 60.0, address: str | None = None, history=None, now=None) -> int:
    from sleeve_fund.history import HistoryStore

    now = now or datetime.now(UTC)
    ids = _hub_ids(configured())
    fails = [] if ids else ["no configured Strategy is hub-fed"]
    address = address or os.environ.get(f"HUB_{HUB_VENUE}", "")
    if ids and not address:
        fails.append(f"HUB_{HUB_VENUE} is not set in this container")
    if ids and address:
        seen = hub_seen(address, list(ids), wait)
        for iid, pair in ids.items():
            for kind in ("trade", "quote"):
                ts = seen[iid].get(kind)
                age = None if ts is None else now.timestamp() - ts / 1e9
                print(f"{pair:12} last {kind:5} {'none' if age is None else f'{age:.1f} s ago'}")
                if age is None or age > MAX_AGE:
                    fails.append(f"{pair} {kind} {'not seen' if age is None else f'{age:.0f} s old'}")
    store = history or HistoryStore(os.environ.get("HISTORY_DIR"))
    last_closed = (now - timedelta(minutes=1)).replace(second=0, microsecond=0)
    for pair in ids.values():
        cov = store.coverage(HUB_VENUE, pair)
        closed = None if cov is None else cov.closed
        print(f"{pair:12} history newest closed minute {closed} (last closed now {last_closed:%Y-%m-%d %H:%M})")
        # The newest closed minute may still be landing: one minute behind the clock is in time, two is not.
        if closed is None or closed.to_pydatetime() < last_closed - timedelta(minutes=1):
            fails.append(f"{pair} history behind ({closed})")
    return verdict("B2 hub fresh", fails)


# --- B4 journal ------------------------------------------------------------------------------------------------------

def journal(strategy: str, since: datetime, store=None) -> int:
    """B3's orders, fills, funding rows and events since `since`, for HoQA's hand calculation. Fails on an open
    order (orphan or resting), a funding_conflict event, a true-up row stamped before its arrival, no entry fill,
    no exit fill, or a position still held."""
    from sleeve_fund import markets
    from sleeve_fund.store import OPEN_ORDER_STATUSES, Store
    from sleeve_fund.venues import venue

    store = store or Store()
    s = store.sleeve(strategy)
    taker = markets.fees_for(s.params, venue((s.venue or "").lower() or "kraken").fees, s.venue).taker
    orders = [o for o in store.orders(strategy, limit=500) if o["ts"] >= since]
    fills = sorted((f for f in store.fills(strategy, limit=500) if f["ts"] >= since), key=lambda f: f["ts"])
    print(f"strategy {strategy}, since {since:%Y-%m-%d %H:%M:%S} UTC, taker {taker}")
    for o in sorted(orders, key=lambda o: o["ts"]):
        print(f"order {o['order_id']} {o['ts']:%H:%M:%S} {o['intent']:12} {o['order_type']:16} {o['side']:4} "
              f"qty {o['qty']} status {o['status']} fee {o['fee']}")
    for f in fills:
        hand = Decimal(str(f["qty"])) * Decimal(str(f["price"])) * Decimal(str(taker))
        print(f"fill {f['trade_id']} {f['ts']:%H:%M:%S} {f['side']:4} qty {f['qty']} @ {f['price']} fee {f['fee']} "
              f"(qty x price x taker = {hand:.8f})")
    funding = [r for r in store.funding(strategy, limit=None) if r["ts"] >= since]
    for r in funding:
        print(f"funding {r['ts']:%H:%M} {r['kind']:8} rate {r['rate']} amount {r['amount']} "
              f"arrived {r.get('arrived_at')}")
    conflicts = [e for e in store.events(strategy, limit=1000, min_level="info")
                 if e["kind"] == "funding_conflict" and e["ts"] >= since]
    book = store.journal_book(strategy, s.starting_balance)
    print(f"book now: qty {book['qty']}, cash {book['cash']}")
    fails = [f"order {o['order_id']} still {o['status']}" for o in orders if o["status"] in OPEN_ORDER_STATUSES]
    fails += [f"funding_conflict at {e['ts']:%H:%M}" for e in conflicts]
    fails += [f"true-up at {r['ts']:%H:%M} stamped before it arrived" for r in funding
              if r["kind"] == "true_up" and (r.get("arrived_at") is None or r["arrived_at"] < r["ts"])]
    sides = {f["side"] for f in fills}
    if len(sides) < 2:
        fails.append(f"no round trip: fills on {sorted(sides) or 'no side'}")
    if book["qty"]:
        fails.append(f"still holding {book['qty']}")
    return verdict("B4 journal", fails)


# --- B5 gate ---------------------------------------------------------------------------------------------------------

def gate(store=None, now=None, atr_pct=None) -> int:
    """The gate's book (PaperGate: the supervisor's last mark and every Strategy's position) against the dashboard's
    open-risk tile (open_risk.book_open_risk, the tile's one formula) at the displayed precision; test-currency
    Strategies out of the book's totals; and one dry over-limit entry on an in-memory copy, refused or trimmed."""
    from sleeve_fund import markets
    from sleeve_fund.open_risk import book_open_risk, history_atr_pct
    from sleeve_fund.paper.portfolio_gate import PaperGate, fund_holdings
    from sleeve_fund.portfolio.gate import MemoryLedger, check_order, mark_book
    from sleeve_fund.portfolio.limits import Intent
    from sleeve_fund.risk import PORTFOLIO
    from sleeve_fund.store import Store, utcnow
    from sleeve_fund.supervisor import is_test_currency_account

    store = store or Store(portfolio_gate=True)
    now = now or utcnow()
    if atr_pct is None:
        def atr_pct(s):
            return history_atr_pct(s.venue or "", s.instrument, now)
    fails: list[str] = []
    try:
        equity = PaperGate(store).ledger.state().equity
    except Exception as e:  # noqa: BLE001 - reported, not raised: the check fails
        return verdict("B5 gate", [f"the gate's state can't be read ({type(e).__name__}: {e})"])
    if equity is None:
        return verdict("B5 gate", ["the fund's book has never been marked"])
    archived = store.archived()
    holders = [s for s in store.sleeves() if s.name not in archived
               and Decimal(str(store.journal_book(s.name, s.starting_balance)["qty"])) != 0]
    held = fund_holdings(store, now, atr_pct)  # the gate's own call, in store.sleeves() order
    by_name = dict(zip([s.name for s in holders], held, strict=True))
    perps = {n: h for n, h in by_name.items() if markets.is_perp(next(s for s in holders if s.name == n).params)}
    gate_risk = sum((h.risk for h in perps.values()), Decimal(0))
    tile = book_open_risk(store, atr_pct)
    tile_risk = sum((Decimal(str(r["risk"])) for r in tile if r["risk"] is not None), Decimal(0))
    gross = sum((abs(h.notional) for h in held), Decimal(0))
    print(f"fund equity (last mark) {equity}; holdings {len(held)}; gross {gross:.2f}; "
          f"margin {sum((h.margin for h in held), Decimal(0)):.2f}")
    print(f"open risk: gate {gate_risk:.2f}, tile {tile_risk:.2f} ({len(tile)} row(s))")
    if round(gate_risk, 2) != round(tile_risk, 2):
        fails.append(f"open risk differs: gate {gate_risk:.2f}, tile {tile_risk:.2f}")
    outside = [n for n in by_name if is_test_currency_account(store.account_of(n), store.sleeve(n).venue)]
    if outside:
        fails.append(f"test-currency Strategies counted in the book: {', '.join(outside)} (CCY-8DP-UI)")
    # A dry entry worth twice the open-risk room, on an in-memory copy: the limit must bite. Nothing is reserved.
    ledger = MemoryLedger(equity=equity, held=list(held))
    mark_book(ledger, equity, now, PORTFOLIO)
    price = Decimal(60000)
    risk_per_unit = price * Decimal("0.01")
    room = max(equity * Decimal(str(PORTFOLIO.open_risk)) - gate_risk, Decimal(0))
    qty = ((2 * room + risk_per_unit) / risk_per_unit).quantize(Decimal("0.001"))
    asked = Intent("BTC", 1, qty, price, price / 2, risk_per_unit, Decimal("0.001"), Decimal("0.001"))
    got = check_order(ledger, "final-smoke-dry", asked, PORTFOLIO, now).decision
    print(f"dry entry {qty} BTC at {price} with a 1% stop: {got.outcome} ({got.limit_hit}), approved {got.approved_qty}")
    if got.outcome == "approved" and got.approved_qty >= qty:
        fails.append("a dry entry at twice the open-risk room was approved in full")
    return verdict("B5 gate", fails)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="final_smoke")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sf = sub.add_parser("safety")
    sf.add_argument("--mirror", action="store_true", help="the mirror's container: its demo accounts' names allowed")
    h = sub.add_parser("hub")
    h.add_argument("--wait", type=float, default=60.0)
    j = sub.add_parser("journal")
    j.add_argument("--strategy", required=True)
    j.add_argument("--since", required=True, type=lambda t: datetime.fromisoformat(t).astimezone(UTC))
    sub.add_parser("gate")
    a = ap.parse_args(argv)
    print(json.dumps({"check": a.cmd, "at": datetime.now(UTC).isoformat(timespec="seconds")}))
    if a.cmd == "safety":
        return safety(mirror_container=a.mirror)
    if a.cmd == "hub":
        return hub(a.wait)
    if a.cmd == "journal":
        return journal(a.strategy, a.since)
    return gate()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
