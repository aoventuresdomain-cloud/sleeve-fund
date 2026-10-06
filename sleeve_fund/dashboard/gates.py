"""Path to live: what a sleeve still needs before the PM can consider G2 for it.

The checklist only reports evidence. It never approves anything: G2 is the PM's decision, and live
trading stays locked in the shell until it is made.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sleeve_fund.dashboard.pipeline import studied_as
from sleeve_fund.store import Store
from sleeve_fund.venues import venue as venue_profile

PAPER_DAYS = 42  # G2: at least six weeks of paper trading
MIN_TRADES = 10  # fewer closed trades than this says nothing about the strategy


def path_to_live(store: Store, x: dict, g1: str | None, accounts: list[dict], now: datetime) -> list[dict]:
    """One row per G2 condition: label, ok (True, False, or None when only the PM can judge) and detail."""
    s = x["sleeve"]
    days = max(0, (now - s.created_at).days)
    trades = x["trades"]["trades"]
    since = now - timedelta(days=PAPER_DAYS)
    mismatch = store.last_event(s.name, ("reconcile_mismatch",))
    halt = store.last_event(s.name, ("risk_halt",))
    errors = [e for e in store.events(s.name, limit=200, min_level="error")
              if e["ts"] >= since and e["kind"] not in ("risk_halt", "reconcile_mismatch")]
    # Only a key for the strategy's own venue counts: a key elsewhere can't trade it.
    profile = venue_profile(s.venue)
    keyed = [a["name"] for a in accounts if a["kind"] == "live" and a["key_present"]
             and (a.get("venue") or "").upper() == profile.name]
    return [
        {"label": "Strategy passed G1", "ok": g1 == "PASS",
         "detail": "on real data, out of sample, for this instrument and interval" if g1 == "PASS"
         else "not possible yet for a perpetual or long/short strategy: studies run spot, long only"
         if not studied_as(s.params)
         else "not for this instrument and interval yet; paper results alone are not evidence"},
        {"label": "Six weeks of paper trading", "ok": days >= PAPER_DAYS,
         "detail": f"{days} of {PAPER_DAYS} days" if days < PAPER_DAYS else f"{days} days"},
        {"label": f"At least {MIN_TRADES} closed trades", "ok": trades >= MIN_TRADES,
         "detail": f"{trades} so far" if trades < MIN_TRADES else f"{trades} trades"},
        {"label": "Journal and engine always agreed", "ok": mismatch is None, "bad": mismatch is not None,
         "detail": "no reconcile mismatch" if mismatch is None else f"mismatch on {mismatch['ts']:%d %b %Y}"},
        {"label": "No risk halt or error in six weeks",
         "ok": not errors and not (halt and halt["ts"] >= since), "bad": bool(errors or (halt and halt["ts"] >= since)),
         "detail": (f"halted on {halt['ts']:%d %b %Y}" if halt and halt["ts"] >= since
                    else f"{len(errors)} error{'s' if len(errors) != 1 else ''}" if errors else "clean")},
        {"label": "Results inside the backtest's range", "ok": None,
         "detail": "you judge: compare its paper return and drawdown with the backtest"},
        {"label": f"Live {'perpetual' if profile.perpetual else 'spot'} account with its key installed",
         "ok": bool(keyed),
         "detail": ", ".join(keyed) if keyed else "add one on the Accounts page"},
        {"label": "Your G2 approval", "ok": False,
         "detail": "only you can approve G2; live mode stays locked until then"},
    ]
