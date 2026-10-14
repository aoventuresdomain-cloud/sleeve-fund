"""B8 instrument terms (venue-swap plan §B, Advisor 14 Oct 2026 03:05 UK): for every instrument a configured
Strategy trades on a perpetual venue with published terms, the engine's terms beside the venue's, with a verdict
per row. The engine side is the instrument paper builds (VenueProfile.instrument), its fallback included when the
venue's contract can't be read; with no instrument compared at all the check fails (B8: NOTHING CHECKED). Read-only: public endpoints, no key, writes nothing.

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
- PASS: equal."""

from __future__ import annotations

import glob
import json
import sys
import tomllib
import urllib.request
from collections.abc import Callable
from decimal import Decimal

from sleeve_fund import markets
from sleeve_fund.venues import (
    BINANCE_FAPI,
    DEFAULT_VENUE,
    binance_symbol,
    venue,
)

TIMEOUT = 30
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
    """The instrument paper builds for the pair (VenueProfile.instrument, as paper's node does): the venue's contract
    limits when it answers, otherwise the cautious fallback (size precision capped at 3, no minimum quantity,
    minimum notional 5). Comparing this, not the contract parse alone, covers the fallback and perpetual()."""
    base, quote = pair.split("/")
    return venue("binance").instrument(base, quote)


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


def main(argv: list[str], get: Callable[[str], object] = get_json, pattern: str = "configs/sleeves/*.toml",
         build: Callable[[str], object] = engine_instrument) -> int:
    pairs = configured(pattern) + [("BINANCE", a.upper()) for a in argv]
    fetched: dict[str, tuple[dict, dict]] = {}
    must = compared = 0
    print(f"{'instrument':14} {'term':22} {'venue':>14} {'engine':>14}  verdict")
    for name, pair in dict.fromkeys(pairs):
        if name not in READERS:
            # A perpetual on a venue without one (the simulated perp): no published terms to compare (FE M2).
            print(f"{pair:14} {'(simulated perp)':22} {'none':>14} {name.lower():>14}  FOLLOW-UP")
            continue
        if name not in fetched:
            info_url, funding_url = READERS[name]
            fetched[name] = (get(info_url), {x["symbol"]: x for x in get(funding_url)})
        for term, v, e, verdict in rows(pair, *fetched[name], build(pair)):
            must += verdict == "MUST"
            compared += 1
            print(f"{pair:14} {term:22} {v!s:>14} {e!s:>14}  {verdict}")
    if not compared:  # F267-1: no instrument on a venue with published terms; a PASS here would mean nothing
        print("B8: NOTHING CHECKED")
        return 1
    print(f"B8: {'MUST' if must else 'PASS'} ({must} MUST row(s), {compared} row(s) compared)")
    return 1 if must else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
