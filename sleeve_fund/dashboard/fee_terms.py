"""FEE-LIVE display (head 3; Independent Quant Advisor's ruling FEE-UI, 15 Oct 2026). Per instrument in use, the fee
and the margin tiers the engine uses, as two rows, each with its own source and fetch time: a stale tier table must
not hide behind a fresh fee.

Display only, and the engine's own reading: every row is a venue_terms.live_terms Term (the reader the start check, the
alerts and the entry refusal use), so the page can never say fresh where the engine refuses entries (a tier table that
changed, a live account on the published rate) or refused where it trades on (a spot schedule not recorded yet). Its
state, source words and age are the Term's; the rates are its values as fetched. The one line of the page's own is the
out-of-range note: the latest fetch was refused for its bounds after the last good one (DA's latest_venue_fetch)."""

from __future__ import annotations

from datetime import datetime

from sleeve_fund import venue_terms

OUT_OF_RANGE = "out of bounds"  # DA: the reason a refused fetch outside [lowest, VIP0] carries


def _venue_of(store, instrument: str) -> str | None:
    from sleeve_fund.venues import venue as venue_profile

    for s in store.sleeves():
        if s.instrument == instrument:
            return venue_profile(s.venue).name
    return None


def _line(store, t, venue: str | None) -> dict:
    out = {"kind": t.kind, "state": t.state, "words": t.words, "never": t.fetched_at is None, "fetched_at": t.fetched_at,
           "maker": None, "taker": None, "tiers": None, "out_of_range": False}
    if t.value is not None:
        if t.kind == "fee":
            out["maker"], out["taker"] = (str(v) for v in t.value)
        else:
            out["tiers"] = len(t.value)
    if venue and t.fetched_at is not None:
        last = store.latest_venue_fetch(venue, t.kind, t.instrument)
        # The newest fetch failing means it came after the last good one, which the engine still uses.
        out["out_of_range"] = bool(last and not last["ok"] and str(last["reason"]).startswith(OUT_OF_RANGE))
    return out


def view(store, now: datetime) -> dict:
    """{"instruments": [{instrument, market, fee: line, tiers: line or None on spot}]}, one per instrument and account
    the engine follows, in live_terms' order (fee rates, then a perp's tiers)."""
    rows, by_instrument = [], {}
    for t in venue_terms.live_terms(store, now):
        venue = _venue_of(store, t.instrument)
        if t.kind == "fee":
            row = {"instrument": t.instrument, "market": "Spot", "fee": _line(store, t, venue), "tiers": None}
            rows.append(row)
            by_instrument.setdefault(t.instrument, []).append(row)
        else:
            for row in by_instrument.get(t.instrument, []):
                row["market"], row["tiers"] = "Perpetual", _line(store, t, venue)
    return {"instruments": rows}
