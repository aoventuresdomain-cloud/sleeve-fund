"""FEE-LIVE display (head 3; Independent Quant Advisor's ruling FEE-UI, 15 Oct 2026). Per instrument in use, the fee
and the margin tiers the engine uses, as two rows, each with its own source and fetch time: a stale tier table must
not hide behind a fresh fee. Display only: the figures are the reader's decimal strings (DA's venue_terms_fetch,
Store.latest_good_venue_fetch and Store.latest_venue_fetch), shown as stored, never worked out here.

The age rules are FEE-LIVE's (hoe/plan-tonight-2130.md): past 24 hours the last good value is still used and alerted;
past 72 hours, or never fetched, new entries are refused."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sleeve_fund.store import PUBLISHED_FEES, is_backtest
from sleeve_fund.venues import venue as venue_profile

STALE_AFTER = timedelta(hours=24)
REFUSE_AFTER = timedelta(hours=72)
KINDS = ("fee", "tiers")
OUT_OF_RANGE = "out of bounds"  # DA: the reason a refused fetch outside [lowest, VIP0] carries


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _line(store, venue: str, kind: str, instrument: str, now: datetime) -> dict:
    good = store.latest_good_venue_fetch(venue, kind, instrument)
    last = store.latest_venue_fetch(venue, kind, instrument)
    out = {"kind": kind, "state": "refused", "never": good is None, "out_of_range": False, "fetched_at": None,
           "age_h": None, "source": None, "vip0": False, "maker": None, "taker": None, "tiers": None}
    # DA: the latest fetch was not used when it is not the last good one; out of range when refused for its bounds.
    if last is not None and (good is None or last["id"] != good["id"]) and not last["ok"] \
            and str(last["reason"]).startswith(OUT_OF_RANGE):
        out["out_of_range"] = True
    if good is None:
        return out
    at = _aware(good["fetched_at"])
    age = now - at
    payload = good["payload"] or {}
    out.update(fetched_at=at, age_h=int(age.total_seconds() // 3600), source=good["source"],
               vip0=good["source"] == PUBLISHED_FEES,
               state="refused" if age > REFUSE_AFTER else "stale" if age > STALE_AFTER else "fresh")
    if kind == "fee":
        out.update(maker=payload.get("maker"), taker=payload.get("taker"))
    else:
        out["tiers"] = len(payload.get("tiers") or [])
    return out


def view(store, now: datetime) -> dict:
    """{"instruments": [{venue, instrument, market, fee: line, tiers: line or None on spot}]} for every instrument a
    Strategy on the book trades."""
    now = _aware(now)
    gone = set(store.archived())
    seen, rows = set(), []
    for s in store.sleeves():
        if s.name in gone or is_backtest(s.name):
            continue
        profile = venue_profile(s.venue)
        key = (profile.name.upper(), s.instrument)
        if key in seen:
            continue
        seen.add(key)
        # Margin tiers apply to a perpetual only; a spot instrument has none to fetch.
        rows.append({"venue": key[0], "instrument": s.instrument,
                     "market": "Perpetual" if profile.perpetual else "Spot",
                     **{k: _line(store, key[0], k, s.instrument, now) if k == "fee" or profile.perpetual else None
                        for k in KINDS}})
    return {"instruments": sorted(rows, key=lambda r: (r["market"], r["instrument"]))}
