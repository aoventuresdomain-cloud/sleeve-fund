"""B3 paper round trip (pe1-plans/b-smoke-runbook.md): start one allowlisted smoke Strategy on paper, let it enter,
flatten it, and stop it again, so the final-head smoke has one entry and one exit to check (final_smoke journal).
NOT read-only, so it is kept out of final_smoke.py. It acts only through the journal calls the dashboard's Start,
Flatten and Stop use, as actor "final-smoke round trip"; the supervisor does the rest as for the PM's buttons.

Runs over stdin in the supervisor container (.github/workflows/final-smoke.yml, its round-trip input), from /app:
    python -u - --strategy ping-pong-ls-binance [--timeout-min 30]

Guards: the Strategy must be on the allowlist, on the paper account, a perpetual on the hub-fed venue, stopped and
flat. It is stopped again whatever happens (an error, a timeout, an interrupt). Exit 0 only for one entry fill,
one exit fill and a flat book; the fills are printed for final_smoke journal and HoQA."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal

ALLOWED = frozenset({"ping-pong-ls-binance"})
ACTOR = "final-smoke round trip"
POLL = 10.0  # seconds


def guard(store, name: str) -> str | None:
    """Why the round trip refuses to start `name`; None when it may."""
    from sleeve_fund import markets

    if name not in ALLOWED:
        return f"{name} is not on the round-trip allowlist ({', '.join(sorted(ALLOWED))})"
    try:
        s = store.sleeve(name)
    except Exception as e:  # noqa: BLE001 - an unknown name refuses
        return f"{name} isn't in the journal ({type(e).__name__})"
    if store.account_of(name) != "paper":
        return f"{name} is on account {store.account_of(name)}, not paper"
    if (s.venue or "").upper() != "BINANCE" or not markets.is_perp(s.params):
        return f"{name} is not a perpetual on the hub-fed paper venue"
    if s.desired_state != "stopped":
        return f"{name} is {s.desired_state}: the round trip starts only a stopped Strategy"
    if Decimal(str(store.journal_book(name, s.starting_balance)["qty"])) != 0:
        return f"{name} holds a position: the round trip starts only from flat"
    return None


def stop(store, name: str, reason: str) -> None:
    """The dashboard's Stop (dashboard.app command handler), in its order."""
    accepted = datetime.now(UTC)
    store.set_desired_state(name, "stopped")
    store.drop_pending(name, "lapsed: the strategy was stopped before it acted")
    store.hold_on_reset(name, "stop", reason)
    store.decide(ACTOR, "stop", reason, name, ts=accepted)
    store.apply_waiting_ral(name)


def run(store, name: str, timeout: float, sleep=time.sleep, clock=time.monotonic) -> int:
    from sleeve_fund.paper.runtime import entry_blocked
    from sleeve_fund.store import utcnow

    if why := guard(store, name):
        print(f"B3 round trip: FAIL (refused: {why})")
        return 1
    blocked, why = entry_blocked(store, name, utcnow(), starting=True)  # the dashboard's start check (CHOKE)
    if blocked:
        print(f"B3 round trip: FAIL (not started: {why})")
        return 1
    s = store.sleeve(name)
    since = utcnow()
    fills: list[dict] = []
    flattened = False
    try:
        store.set_desired_state(name, "running")
        store.decide(ACTOR, "start", "final-head smoke: one paper round trip", name)
        print(f"started {name} at {since:%H:%M:%S} UTC")
        end = clock() + timeout
        while clock() < end:
            fills = sorted((f for f in store.fills(name, limit=50) if f["ts"] >= since), key=lambda f: f["ts"])
            qty = Decimal(str(store.journal_book(name, s.starting_balance)["qty"]))
            if fills and not flattened and qty != 0:
                store.command(name, "flatten", "final-head smoke: close the round trip", actor=ACTOR)
                flattened = True
                print(f"entry filled; flatten sent at {utcnow():%H:%M:%S} UTC")
            if flattened and qty == 0:
                break
            sleep(POLL)
    finally:
        stop(store, name, "final-head smoke: round trip done")
        print(f"stopped {name} at {utcnow():%H:%M:%S} UTC")
    fills = sorted((f for f in store.fills(name, limit=50) if f["ts"] >= since), key=lambda f: f["ts"])
    for f in fills:
        print(f"fill {f['trade_id']} {f['ts']:%H:%M:%S} {f['side']:4} qty {f['qty']} @ {f['price']} fee {f['fee']}")
    qty = Decimal(str(store.journal_book(name, s.starting_balance)["qty"]))
    sides = {f["side"] for f in fills}
    fails = [] if len(sides) == 2 else [f"fills on {sorted(sides) or 'no side'}"]
    fails += [f"still holding {qty}: it rests on its stop, exits only"] if qty else []
    print(f"since {since.isoformat()}")
    print("B3 round trip: PASS" if not fails else f"B3 round trip: FAIL ({'; '.join(fails)})")
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    from sleeve_fund.store import Store

    ap = argparse.ArgumentParser(prog="round_trip")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--timeout-min", type=float, default=30.0)
    a = ap.parse_args(argv)
    return run(Store(portfolio_gate=True), a.strategy, a.timeout_min * 60)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
