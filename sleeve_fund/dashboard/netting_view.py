"""B-13: the netting shadow page (PM 13 Oct "shadow first"; Advisor specs-p2-3 §2). What one account per venue would
have saved had each decision bar's intents in an instrument gone out as one net order, from the paper journal as it
stands (portfolio.netting_paper). Shadow only: never Strategy evidence, and nothing here changes trading.

Display only: venue names stay off the page (an account is named by its Strategies)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sleeve_fund.portfolio import netting_paper


def _taker(store, now: datetime):
    """taker(strategy): the rate its paper fills are charged, worked out as the paper node does (node.py: FEE-LIVE's
    rate for its account, then its market's schedule)."""
    from sleeve_fund.fees import resolve
    from sleeve_fund.paper.config import from_store

    def taker(name: str) -> Decimal:
        row = store.sleeve(name)
        acct = store.account_of(name)
        quote = resolve(getattr(row, "venue", None), store, row.instrument, now,
                        live=store.account_kind(acct) == "live", account=acct)
        return from_store(row, fee_schedule=quote.fees).fees.taker

    return taker


def view(store, now: datetime) -> dict:
    """{"accounts": [...], "error": words or None}. Each account: its Strategies and the shadow's figures as floats
    for the page, with the residual's check and opposed exposure per Strategy pair (largest first)."""
    from sleeve_fund.supervisor import _in_book, fund_equity

    sleeves = {s.name: s for s in store.sleeves()}
    try:
        equity = Decimal(repr(float(fund_equity(store))))
        fills, funding = netting_paper.journal_rows(
            store, lambda name: name in sleeves and _in_book(store, sleeves[name]))
        shadow = netting_paper.shadow(fills, funding, _taker(store, now), equity)
    except (ValueError, LookupError) as e:  # an unmarked position, a fee that can't be read: said, never guessed
        return {"accounts": [], "error": str(e)}
    accounts = []
    for n, (_, s) in enumerate(sorted(shadow.items()), start=1):
        saving = {k: float(v) for k, v in s["saving"].items()}
        accounts.append({
            "n": n,
            "strategies": sorted(s["sub_ledgers"]),
            "fee": saving.get("fee", 0.0), "crossed": saving.get("crossed", 0.0), "total": float(s["saving_total"]),
            "orders_sent": s["orders_sent"], "orders_alone": s["orders_alone"], "bars_shared": s["bars_shared"],
            "crossed_notional": float(s["crossed_notional"]),
            "baseline": float(s["conservative_funding_baseline"]), "corrections": s["corrections"],
            "residual": float(s["residual"]["amount"]), "residual_bp": float(s["residual"]["bp"]),
            "residual_over": bool(s["residual"]["block"]),
            "opposed": sorted(({"pair": f"{a} / {b}", "share": v} for (a, b), v in s["opposed_exposure"].items()),
                              key=lambda r: -r["share"]),
        })
    return {"accounts": accounts, "error": None, "limit_bp": float(netting_paper.nt.RESIDUAL_BP)}
