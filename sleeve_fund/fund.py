"""The fund's external capital (bridge A, DA spec 1): journal a deposit or a withdrawal for the poll to apply.

    python -m sleeve_fund.fund deposit 25000 --actor pm --reason "top up"            (dry run: writes nothing)
    python -m sleeve_fund.fund withdraw 5000 --actor pm --reason "take out" --apply

The default is a dry run: it checks the request as the real run would and prints the row it would write. --apply
writes the fund_flows row and its decision in one transaction (Store.fund_flow) and prints the row's id. Nothing moves
here: the poll ends each flow in id order, applied, or refused with the reason (e.g. a withdrawal above free cash).
The clean slate's re-base deposit is not made here: only the reset sets it (DA spec 7, PE1).

fund-flow.yml runs this on the server. Its log is public, so the output is only the kind, the amount, the row id and
the outcome: no balance, reason, host or address."""

from __future__ import annotations

import argparse
import re
import sys

from sleeve_fund.money import to_decimal
from sleeve_fund.store import FUND_FLOW_KINDS

# A cent at most: the PM moves money in currency units, and a finer amount is a typo, refused rather than rounded.
AMOUNT = re.compile(r"^[0-9]{1,12}(\.[0-9]{1,2})?$")
PLAIN = re.compile(r"^[A-Za-z0-9 ._,:()-]{1,200}$")


def check(kind: str, amount: str, actor: str, reason: str):
    """The request as the real run would take it, or a ValueError saying what is wrong."""
    if kind not in FUND_FLOW_KINDS:
        raise ValueError(f"kind: {' or '.join(FUND_FLOW_KINDS)}")
    if not AMOUNT.match(amount) or not to_decimal(amount) > 0:
        raise ValueError("amount: above zero, in units with at most two decimals, e.g. 25000 or 1250.50")
    if not re.match(r"^[A-Za-z0-9._-]{1,32}$", actor or ""):
        raise ValueError("actor: letters, digits, . _ - only, up to 32")
    if not PLAIN.match((reason or "").strip()):
        raise ValueError("reason: letters, digits, spaces and . , : ( ) _ - only, up to 200")
    return to_decimal(amount)


def run(argv: list[str] | None = None, store=None, out=print) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.fund")
    ap.add_argument("kind", choices=FUND_FLOW_KINDS)
    ap.add_argument("amount")
    ap.add_argument("--actor", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--apply", action="store_true", help="write it (default: a dry run that writes nothing)")
    args = ap.parse_args(argv)
    try:
        amount = check(args.kind, args.amount, args.actor, args.reason)
    except ValueError as exc:
        out(f"refused: {exc}")
        return 1
    if store is None:
        from sleeve_fund.store import Store

        store = Store()
    ahead = len(store.pending_flows())
    if not args.apply:
        out(f"dry run: would journal {args.kind} {amount:.2f}; {ahead} flow(s) ahead of it; nothing written")
        return 0
    fid = store.fund_flow(args.kind, amount, actor=args.actor, reason=args.reason.strip())
    out(f"journaled: {args.kind} {amount:.2f} as flow {fid}; {ahead} flow(s) ahead of it; the next poll ends it")
    return 0


if __name__ == "__main__":
    sys.exit(run())
