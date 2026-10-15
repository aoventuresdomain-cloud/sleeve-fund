"""A multi-asset Strategy's page covers every asset, not only its first (F-B5-PAGE, HoQA, Drop B). Display only: each
asset's figures are the journal's, read from Store.asset_book (per asset and in total, to the last digit), never worked
out here."""

from __future__ import annotations

from decimal import Decimal

from sleeve_fund.dashboard import book as bookm
from sleeve_fund.dashboard.metrics import sleeve_summary


def _d(v) -> Decimal:
    """Exact: a Decimal as it is, a float through its shortest repr (the journal's floats are its stored digits)."""
    return v if isinstance(v, Decimal) else Decimal(str(v))


def view(store, s, x: dict) -> dict | None:
    """The Strategy `s`'s assets, one row each in its asset order, and its total; None for a one-asset or a two-leg
    Strategy, and on an asset leg's own row (the Strategy's page is its parent's). `x` is the parent row's summary.

    An asset's P&L is its cash less its starting balance once it is flat, which is exact; while it holds a position
    it is that row's own P&L since start, marked, as its leg's figures read it. `kpi` is the header's figures over
    every asset."""
    if s.parent or store._leg_of(s.name) is not None:
        return None
    book = store.asset_book(s.name)
    if len(book["assets"]) < 2:
        return None
    rows, unrealised, day, fills = [], 0.0, 0.0, 0
    for a in book["assets"]:
        lx = x if a["sleeve"] == s.name else bookm.sleeve_extras(
            store, sleeve_summary(store, store.sleeve(a["sleeve"])), bookm.daily(store, a["sleeve"]))
        held = a["qty"] != 0
        pnl = _d(lx["pnl"]) if held else _d(a["cash"]) - _d(a["starting_balance"])
        rows.append({"asset": a["underlying"], "instrument": a["instrument"], "sleeve": a["sleeve"], "qty": a["qty"],
                     "held": held, "fees": _d(a["fees"]), "funding": _d(a["funding"]), "pnl": pnl,
                     "fills": a["fills"]})
        unrealised += float(lx.get("unrealised", 0.0)) if held else 0.0
        day += float(lx.get("day_pnl", 0.0))
        fills += a["fills"]
    t = book["total"]
    total = {"pnl": sum((r["pnl"] for r in rows), Decimal(0)), "fees": _d(t["fees"]), "funding": _d(t["funding"]),
             "fills": t["fills"]}
    pnl = float(total["pnl"])
    kpi = {"pnl": pnl, "unrealised": unrealised, "realised": pnl - unrealised, "day_pnl": day,
           "fees": float(total["fees"]), "funding": float(total["funding"]),
           "fills": fills, "fees_funding": float(total["fees"] - total["funding"])}  # funding is + received
    names = [r["asset"] for r in rows]
    words = f"{', '.join(names[:-1])} and {names[-1]}" if len(names) <= 4 else f"{len(names)} assets"
    return {"rows": rows, "total": total, "kpi": kpi, "words": words, "legs": [r["sleeve"] for r in rows],
            "open": [r for r in rows if r["held"]]}  # what a flatten closes (F-B5-FLATTEN): every leg holding
