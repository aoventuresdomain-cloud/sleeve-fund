"""B8 instrument terms (venue-swap plan §B, Advisor 14 Oct 2026 03:05 UK): for every instrument a configured
Strategy trades on a perpetual venue with published terms, the engine's terms beside the venue's, with a verdict
per row, on two engine sides:
- "engine": the instrument backtest and research build (VenueProfile.instrument), its fallback included when the
  venue's contract can't be read.
- "(paper)": the definition the venue's hub serves in its hello, which is what every hub-fed paper process trades
  with (paper/hub_client.py). The hub reads it once, when it starts relaying the instrument, and never again, so a
  run of this check during testing also shows a venue term change paper has not seen (F267-2). A paper process
  connected before a hub restart keeps the copy it was given; the rows show the hub's. Only configured Strategies'
  instruments get these rows, and a hub that can't be reached or doesn't serve one is a MUST, never a pass.
With no instrument compared at all the check fails (B8: NOTHING CHECKED). Read-only: public endpoints and a hub
subscription, no key, writes nothing.

Runs where the venue's public API is reachable: on the paper server, inside the venue's hub container
(.github/workflows/instrument-terms.yml). The instruments come from configs/sleeves/*.toml at run time, so a later
Strategy is covered without a code change; extra BASE/QUOTE pairs on the command line are added to them.

    python -u - < scripts/b8_instrument_terms.py            # in the hub container, from /opt/sleeve-fund
    python scripts/b8_instrument_terms.py [ETH/USDT ...]

Verdicts (Advisor 03:05 UK):
- MUST: paper would accept an order the venue refuses (a finer tick or lot, or a lower minimum quantity or
  notional), a tick that isn't a power of ten (that instrument is not traded in testing), or a funding interval
  that differs (paper would charge funding at the wrong times). Exit status 1.
- FOLLOW-UP: a difference that can't make paper accept what the venue refuses: a coarser lot or a higher minimum,
  the funding cap, and the maintenance margin and half spread, which are one constant each and checked by hand
  (VS-1 is due before G2).
- PASS: equal.

With the journal (DATABASE_URL, which the hub container has), each traded pair also gets the venue data FEE-LIVE
fetched (venue_terms_fetch, its last-known-good row) beside the engine's: "maint. tiers (fetched)" against the kept
maintenance table, and "taker fee (fetched)" / "maker fee (fetched)" against the fees the engine charges
(fees.resolve); an account's fetched rate is never printed, the log being public, only whether the engine's is
equal, higher or lower (the published stand-in's rates print). A tier that differs, or an engine fee below the fetched one (paper cheaper than the venue), is a MUST;
an engine fee above it is a FOLLOW-UP. Age as FEE-LIVE judges it: over 24 h FOLLOW-UP, over 72 h or never fetched
MUST. The published stand-in, and a newer fetch that was refused, are FOLLOW-UPs that say so. It only reads them."""

from __future__ import annotations

import glob
import json
import os
import socket
import sys
import tomllib
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sleeve_fund import markets
from sleeve_fund.venues import (
    BINANCE_FAPI,
    DEFAULT_VENUE,
    binance_symbol,
    venue,
)

TIMEOUT = 30
STALE, REFUSED = timedelta(hours=24), timedelta(hours=72)  # FEE-LIVE: alert after a day, no new entries after three
WORST = ("PASS", "FOLLOW-UP", "MUST")
# The perpetual venues whose published terms this reads: venue name -> (exchangeInfo URL, fundingInfo URL).
READERS = {"BINANCE": (f"{BINANCE_FAPI}/exchangeInfo", f"{BINANCE_FAPI}/fundingInfo")}


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return json.load(r)


def configured(pattern: str = "configs/sleeves/*.toml") -> list[tuple[str, str]]:
    """(VENUE, BASE/QUOTE) for every configured perpetual Strategy, each pair once, in file order."""
    out: list[tuple[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
        s, p = cfg.get("sleeve", {}), cfg.get("params", {})
        key = (str(s.get("venue") or DEFAULT_VENUE).upper(), str(s["instrument"]).upper())
        if markets.market_of(p) == markets.PERP and key not in out:
            out.append(key)
    return out


def power_of_ten(x: Decimal) -> bool:
    d = x.normalize()
    return d == Decimal(1).scaleb(d.as_tuple().exponent)


def _floor_row(name: str, v: Decimal, e: Decimal | None) -> tuple[str, object, object, str]:
    """A minimum (step, quantity or notional): an engine figure below the venue's lets paper fill what it refuses."""
    if e is None or e < v:
        return name, v, e, "MUST"
    return name, v, e, "PASS" if e == v else "FOLLOW-UP"


def _dec(x) -> Decimal | None:
    """An instrument figure (Price, Quantity, Money or None) as a Decimal, None when the instrument has none."""
    if x is None:
        return None
    return x.as_decimal() if hasattr(x, "as_decimal") else Decimal(str(x))


def engine_instrument(pair: str):
    """The instrument backtest and research build for the pair (VenueProfile.instrument): the venue's contract limits
    when it answers, otherwise the cautious fallback (size precision capped at 3, no minimum quantity, minimum
    notional 5). Comparing this, not the contract parse alone, covers the fallback and perpetual()."""
    base, quote = pair.split("/")
    return venue("binance").instrument(base, quote)


def hub_instruments(name: str, pairs: list[str], connect=socket.create_connection) -> dict[str, object]:
    """{pair: instrument} as the venue's hub serves them in its hello: what a hub-fed paper process is given. Inside
    the hub's own container by default; HUB_<VENUE> (host:port) points elsewhere. A pair the hub doesn't serve is
    left out; a hub that can't be reached raises."""
    from sleeve_fund.hub import protocol
    from sleeve_fund.paper.hub_client import instrument_from

    profile = venue(name)
    ids = {f"{profile.symbol_of(p)}.{profile.name}": p for p in pairs}
    host, _, port = (os.environ.get(f"HUB_{profile.name}") or "localhost:7700").rpartition(":")
    with connect((host, int(port)), timeout=TIMEOUT) as s:
        s.sendall(protocol.encode(protocol.subscription(sorted(ids))))
        hello = protocol.decode(s.makefile("rb").readline())
    if hello.get("t") != "hello":
        raise RuntimeError(f"the hub answered {hello}")
    return {ids[d["id"]]: instrument_from(d) for d in hello.get("instruments", []) if d.get("id") in ids}


def paper_rows(pair: str, info: dict, inst) -> list[tuple[str, object, object, str]]:
    """The hub's definition against the venue: the four terms that decide what paper accepts, by the same rules."""
    return [(f"{term} (paper)", v, e, verdict) for term, v, e, verdict in rows(pair, info, {}, inst)[:4]]


def rows(pair: str, info: dict, funding: dict, inst) -> list[tuple[str, object, object, str]]:
    sym = binance_symbol(pair)
    listing = next(s for s in info["symbols"] if s["symbol"] == sym and s["contractType"] == "PERPETUAL")
    f = {x["filterType"]: x for x in listing["filters"]}
    tick, step = Decimal(f["PRICE_FILTER"]["tickSize"]), Decimal(f["LOT_SIZE"]["stepSize"])
    eng_tick = _dec(inst.price_increment)
    profile = venue("binance")
    fi = funding.get(sym, {})
    out = [
        ("tick", tick, eng_tick, "MUST" if not power_of_ten(tick) else _floor_row("", tick, eng_tick)[3]),
        _floor_row("lot step", step, _dec(inst.size_increment)),
        _floor_row("min quantity", Decimal(f["LOT_SIZE"]["minQty"]), _dec(inst.min_quantity)),
        _floor_row("min notional", Decimal(f["MIN_NOTIONAL"]["notional"]), _dec(inst.min_notional)),
    ]
    hours = int(fi.get("fundingIntervalHours", 8))  # a symbol missing from fundingInfo settles on the 8h default
    eng_hours = int(markets.funding_interval(profile.funding_hours).total_seconds() // 3600)
    out.append(("funding interval (h)", hours, eng_hours, "PASS" if hours == eng_hours else "MUST"))
    cap = Decimal(fi["adjustedFundingRateCap"]) if "adjustedFundingRateCap" in fi else None
    eng_cap = profile.published_funding_caps.get(pair)
    eng_cap = None if eng_cap is None else Decimal(str(eng_cap))
    out.append(("funding cap", cap, eng_cap, "PASS" if cap is not None and cap == eng_cap else "FOLLOW-UP"))
    out.append(("maint. margin, tier 1", "hand check", Decimal(str(markets.LOW_FEE_PERP.maintenance_margin)),
                "FOLLOW-UP"))
    out.append(("half spread", "hub quotes", Decimal(str(profile.assumed_half_spread)), "FOLLOW-UP"))
    return out


def _worst(*verdicts: str) -> str:
    return max(verdicts, key=WORST.index)


def _age(row: dict, now: datetime) -> tuple[str, str]:
    """(verdict, words) for a good fetch's age, as FEE-LIVE judges it."""
    at = row["fetched_at"]
    at = at if at.tzinfo else at.replace(tzinfo=UTC)
    age = now - at
    words = f"fetched {at:%d %b %H:%M} UTC, {age.total_seconds() / 3600:.0f} h ago"
    return ("MUST" if age > REFUSED else "FOLLOW-UP" if age > STALE else "PASS"), words


def _tier_gap(fetched: tuple, kept: tuple) -> str:
    """"" when the tables are the same, else the first tier that differs, in words."""
    import math

    for n, (a, b) in enumerate(zip(fetched, kept), 1):
        if not all(math.isclose(x, y, rel_tol=1e-12) for x, y in zip(a, b)):
            return (f"tier {n} differs: venue up to {a.ceiling:,.0f} at {a.rate:g} less {a.amount:,.0f}, "
                    f"engine up to {b.ceiling:,.0f} at {b.rate:g} less {b.amount:,.0f}")
    return "" if len(fetched) == len(kept) else f"the venue has {len(fetched)} tiers, the engine {len(kept)}"


def fetched_rows(store, name: str, pair: str, now: datetime) -> tuple[list[tuple[str, object, object, str]], list[str]]:
    """The fetched venue data (FEE-LIVE's last-known-good rows) beside the engine's maintenance tiers and fees, and a
    note for each row that needs words. Reads the journal only."""
    from sleeve_fund import fees
    from sleeve_fund.store import PUBLISHED_FEES, margin_tiers_of

    out: list[tuple[str, object, object, str]] = []
    notes: list[str] = []

    def said(term: str, words: str) -> None:
        notes.append(f"  {pair} {term}: {words}")

    def refused_since(kind: str, good: dict | None) -> str:
        last = store.latest_venue_fetch(name, kind, pair)
        if last and not last["ok"] and (good is None or last["id"] != good["id"]):
            return f"latest fetch not used: {last['reason']}"
        return ""

    term = "maint. tiers (fetched)"
    good = store.latest_good_venue_fetch(name, "tiers", pair)
    try:
        kept = markets.maintenance_for(venue(name), pair)
    except markets.NoMaintenanceTable:
        kept = None
    if good is None:
        out.append((term, "none", len(kept) if kept else "none", "MUST"))
        said(term, "never fetched")
    else:
        table = margin_tiers_of(good)
        verdict, words = _age(good, now)
        gap = "the engine keeps no table" if kept is None else _tier_gap(table, kept)
        if gap:
            verdict, words = "MUST", f"{gap}; {words}"
        out.append((term, len(table), len(kept) if kept else "none", verdict))
        said(term, words)
    if why := refused_since("tiers", good):
        out[-1] = (*out[-1][:3], _worst(out[-1][3], "FOLLOW-UP"))
        said(term, why)

    charged = fees.resolve(name.lower(), store).fees
    good = store.latest_good_venue_fetch(name, "fee", pair)
    why = refused_since("fee", good)
    for side in ("taker", "maker"):
        term, engine = f"{side} fee (fetched)", Decimal(str(getattr(charged, side)))
        if good is None:
            out.append((term, "none", engine, "MUST"))
            said(term, "never fetched")
            continue
        got = Decimal(good["payload"][side])
        verdict, words = _age(good, now)
        if engine < got:  # paper charges less than the venue: its results would look better than they are
            verdict, words = "MUST", f"the engine charges less than the venue; {words}"
        elif engine > got:
            verdict, words = _worst(verdict, "FOLLOW-UP"), f"the engine charges more than the venue; {words}"
        if good["source"] == PUBLISHED_FEES:
            verdict, words = _worst(verdict, "FOLLOW-UP"), f"published, not account: {words}"
        if why:
            verdict, words = _worst(verdict, "FOLLOW-UP"), f"{words}; {why}"
        if good["source"] == PUBLISHED_FEES:
            out.append((term, got, engine, verdict))
        else:  # an account's rate never reaches the public job log (HoE): only how the engine's compares with it
            out.append((term, "account", "equal" if engine == got else "higher" if engine > got else "lower", verdict))
        said(term, words)
    return out, notes


def main(argv: list[str], get: Callable[[str], object] = get_json, pattern: str = "configs/sleeves/*.toml",
         build: Callable[[str], object] = engine_instrument, hub: Callable[..., dict] | None = None, store=None,
         now: datetime | None = None) -> int:
    traded = configured(pattern)
    pairs = traded + [("BINANCE", a.upper()) for a in argv]
    fetched: dict[str, tuple[dict, dict]] = {}
    served: dict[str, dict | str] = {}  # venue -> {pair: hub definition}, or why the hub couldn't be read
    must = compared = 0
    now = now or datetime.now(UTC)
    print(f"{'instrument':14} {'term':22} {'venue':>14} {'engine':>14}  verdict")
    for name, pair in dict.fromkeys(pairs):
        if name not in READERS:
            # A perpetual on a venue without one (the simulated perp): no published terms to compare (FE M2).
            print(f"{pair:14} {'(simulated perp)':22} {'none':>14} {name.lower():>14}  FOLLOW-UP")
            continue
        if name not in fetched:
            info_url, funding_url = READERS[name]
            fetched[name] = (get(info_url), {x["symbol"]: x for x in get(funding_url)})
        out = rows(pair, *fetched[name], build(pair))
        if (name, pair) in traded:  # an extra pair from the command line isn't traded, so the hub isn't asked for it
            if name not in served:
                try:
                    served[name] = (hub or hub_instruments)(name, [p for v, p in traded if v == name])
                except Exception as exc:  # noqa: BLE001 - reported on each row it leaves unchecked
                    served[name] = f"hub unreachable: {exc!r}"
            got = served[name]
            if isinstance(got, str) or pair not in got:
                why = got if isinstance(got, str) else "the hub doesn't serve it"
                out.append(("hub definition (paper)", "-", "none", "MUST"))
                print(f"{pair}: no paper terms to compare ({why})")
            else:
                out += paper_rows(pair, fetched[name][0], got[pair])
        notes: list[str] = []
        if store is not None and (name, pair) in traded:
            more, notes = fetched_rows(store, name, pair, now)
            out += more
        for term, v, e, verdict in out:
            must += verdict == "MUST"
            compared += 1
            print(f"{pair:14} {term:22} {v!s:>14} {e!s:>14}  {verdict}")
        for line in notes:
            print(line)
    if not compared:  # F267-1: no instrument on a venue with published terms; a PASS here would mean nothing
        print("B8: NOTHING CHECKED")
        return 1
    print(f"B8: {'MUST' if must else 'PASS'} ({must} MUST row(s), {compared} row(s) compared)")
    return 1 if must else 0


if __name__ == "__main__":
    journal = None
    if os.environ.get("DATABASE_URL"):  # the hub container's journal: the fetched rows are read from it, never written
        from sleeve_fund.store import Store

        journal = Store()
    sys.exit(main(sys.argv[1:], store=journal))
