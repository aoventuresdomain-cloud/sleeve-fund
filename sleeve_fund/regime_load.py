"""P2-5 regime: loading what the venue lists into `instruments` (DA design data-architect/p2-5-data-design.md).
Network calls stay in sleeve_fund.venues; this only books their rows, once, and says what it refused."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta


def sync_instruments(store, venue: str, listings: list[dict], now: datetime | None = None) -> dict:
    """Book each listing (the venue loader's listing rows) into `instruments`: counts of "new", "same" and "delisted",
    and "refused": the listings that differ from the contract held (a reused symbol, a listing day or delisting
    that moved), each with the store's reason. A refusal changes nothing and stops nothing else: the caller raises
    it as an alert."""
    counts: Counter = Counter()
    refused = []
    for row in listings:
        try:
            counts[store.record_instrument(venue, row["symbol"], underlying=row["underlying"], quote=row["quote"],
                                           contract_type=row["contract_type"], listed_at=row["listed_at"],
                                           source=row["source"], delisted_at=row.get("delisted_at"),
                                           seen_at=now)] += 1
        except ValueError as exc:
            refused.append((row["symbol"], str(exc)))
    return {"new": counts["new"], "same": counts["same"], "delisted": counts["delisted"], "refused": refused}


def config_name(settings: dict | None = None) -> str:
    """The `config` the rows are written under: every pre-registered setting, so another setting writes new rows
    beside these instead of editing them (a changed setting is a new counted mapping, Advisor 13 Oct)."""
    from sleeve_fund.regime import SETTINGS

    s = settings or SETTINGS
    return (f"n{s['n']}-k{s['keep_within']}-age{s['min_age_days']}-lag{s['lag_days']}"
            f"-min{s['min_members']}-h{s['min_history']}")


def load_bars(store, venue: str, source: dict, now: datetime | None = None, log=print) -> dict:
    """Bring `instruments` and `daily_bars` up to date from the venue's loaders (sleeve_fund.venues.regime_source):
    the listing first, then each listed contract's closed days after the newest held, then every contract only the
    archive holds, once (its bars give its listed and delisted days). Returns the counts and every refusal, which
    the caller raises as an alert; a refusal stops only its own contract."""
    report = {"listed": sync_instruments(store, venue, source["listings"](), now), "bars_new": 0,
              "archive_only": 0, "refused": []}
    held = {r["symbol"]: r for r in store.instruments(venue)}
    for sym, row in sorted(held.items()):
        if row["source"] != "api":
            continue
        last = store.last_daily_bar(venue, sym)
        if last is not None and row["delisted_at"] is not None and last >= row["delisted_at"]:
            continue  # delisted and complete: nothing more will come
        try:
            bars = source["api_bars"](sym, start=None if last is None else last + timedelta(days=1))
            report["bars_new"] += store.record_daily_bars(venue, sym, bars, source="api", fetched_at=now)["new"]
        except (ValueError, OSError) as exc:
            report["refused"].append((sym, str(exc)))
    for sym in source["archive_symbols"]():
        if sym in held:
            continue
        try:
            bars = source["archive_bars"](sym)
            listing = source["archive_listing"](sym, bars)
            if listing is None:
                continue
            got = sync_instruments(store, venue, [listing], now)
            if got["refused"]:
                report["refused"] += got["refused"]
                continue
            store.record_daily_bars(venue, sym, bars, source="archive", fetched_at=now)
            report["archive_only"] += 1
        except (ValueError, OSError) as exc:
            report["refused"].append((sym, str(exc)))
    log(f"{venue}: {report['bars_new']} new daily bars, {report['archive_only']} archive-only contracts, "
        f"{len(report['refused'])} refused")
    return report


def build(store, venue: str, end: date, *, settings: dict | None = None, lookbacks=None,
          now: datetime | None = None, log=print) -> dict:
    """Compute the universe and the regime series from the stored daily bars for every day from the first bar to
    `end`, and write each day once under config_name(settings). Rows already written must come out the same (a
    differing one raises ValueError: the build is not reproducible, which is a bug to fix, never a row to edit).
    Recomputed from the first day every time, so a nightly run and a backfill write the same rows."""
    from sleeve_fund.regime import LOOKBACKS, SETTINGS, Listing, series

    s = settings or SETTINGS
    config = config_name(s)
    listings = [Listing(r["symbol"], r["underlying"], r["listed_at"], r["delisted_at"])
                for r in store.instruments(venue) if r["contract_type"] == "perp"]
    closes, notionals = store.daily_bars(venue)
    first = min((min(d) for d in closes.values() if d), default=None)
    out = {"config": config, "universe": Counter(), "regime": Counter()}
    if first is None or end <= first:
        return out
    days = [first + timedelta(days=i) for i in range((end - first).days + 1)]
    written_universe = False
    for lookback in lookbacks or LOOKBACKS:
        rows, universes = series(days, listings, notionals, closes, lookback=lookback, **s)
        if not written_universe:  # the universe doesn't depend on the lookback
            for day, members in universes.items():
                out["universe"][store.record_universe(venue, day, config, members,
                                                      data_through=day - timedelta(days=s["lag_days"]),
                                                      computed_at=now)] += 1
            written_universe = True
        for row in rows:
            out["regime"][store.record_regime(row, reference_venue=venue, config=config, computed_at=now)] += 1
    log(f"{venue} {config}: universe {dict(out['universe'])}, regime {dict(out['regime'])} through {end}")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    from datetime import timezone

    from sleeve_fund.store import Store
    from sleeve_fund.venues import regime_source

    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.regime_load",
                                 description="P2-5 regime data: load daily bars, then write the universe and series")
    ap.add_argument("cmd", choices=("backfill",), help="load what is new, then build every day through --end")
    ap.add_argument("--venue", required=True, help="the reference venue (lower-case)")
    ap.add_argument("--end", default=None, help="last day to write (YYYY-MM-DD, UTC); default today")
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    end = date.fromisoformat(args.end) if args.end else now.date()
    store = Store()
    report = load_bars(store, args.venue, regime_source(args.venue), now)
    for sym, why in report["refused"]:
        print(f"refused {sym}: {why}")
    build(store, args.venue, end, now=now)
    return 1 if report["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
