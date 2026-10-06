"""Wording the PM reads, wherever it is shown: the dashboard and the outside alerts alike."""

from __future__ import annotations

import re

_DEMO, _PERP, _SPOT = "the demo account", "the perpetual venue", "the spot venue"
_WHAT = {"bybit": _DEMO, "deribit": _DEMO, "binance": _PERP, "kraken": _SPOT}
_NAMES = "bybit|binance|kraken|deribit"
_I = re.IGNORECASE


def _as(words: str | None = None):
    """Replace a venue name with what it is, keeping a possessive: "Bybit's API" reads "the demo account's API".
    With no words given, the venue named in the match decides."""
    def sub(m):
        found = re.search(_NAMES, m.group(0), _I)
        what = words or _WHAT[found.group(0).lower()]
        return what + ("'s" if m.group(0).lower().endswith("'s") else "")
    return sub


# Venue names in text the PM reads (mirror reasons, alerts, the decision log, anything a venue sends back).
# Any case: venues name themselves as they like ("binance", "ByBit Demo"). Left alone: the PM's own labels and
# keys, where the name runs on with "-" or "_" ("kraken-live", BYBIT_DEMO_API_KEY). Only Setup, Accounts
# names venues (QA U8, P1-U23). The first few rephrase whole sentences stored before the wording changed, so
# they read naturally (P1-U21).
_VENUE_WORDS = [
    # A URL or host on a venue's domain is the venue: "https://api.Binance.com/x" reads "the perpetual venue".
    # The host may run the name on ("testnet.binancefuture.com"), carry a port, and end a URL in "<…>".
    (re.compile(rf"(?:https?://)?[\w.-]*\b(?:{_NAMES})[a-z]*\.(?:com|io|exchange)\b(?::\d+)?"
                rf"(?:/[^\s,;)\]<>'\"]*[^\s,;)\]<>'\".:])?", _I),
     _as()),
    # An instrument id's venue suffix goes, the id stays: "BTCUSDT-PERP.BINANCE" reads "BTCUSDT-PERP".
    # Inside a file name the venue becomes "venue", so the name stays a name: "keys.binance.json" reads
    # "keys.venue.json", not "keys.json".
    (re.compile(rf"(?<=\w)\.(?:{_NAMES})(?=\.\w)", _I), ".venue"),
    (re.compile(rf"(?<=\S)\.(?:{_NAMES})\b", _I), ""),
    (re.compile(rf"(?<!\S)\.(?:{_NAMES})\b", _I), _as()),
    # An alert's tag names the venue and the pair ("[BINANCE BTC/USDT] funding stale"): the pair is what it is
    # about, so the tag reads "[BTC/USDT] funding stale".
    (re.compile(rf"\b(?:{_NAMES})\s+(?=(?-i:[A-Z0-9]{{2,}}/[A-Z0-9]{{2,}})\b)", _I), ""),
    # A venue's name at the start of a file name: "binance.py:42" reads "venue.py:42", "kraken.env" "venue.env".
    (re.compile(rf"\b(?:{_NAMES})(?=\.\w)", _I), "venue"),
    (re.compile(r"\bno (?:Bybit|Deribit) demo account set up\b", _I), "the demo account isn't set up"),
    (re.compile(r"\bno (?:Bybit|Deribit) demo perpetual set up for\b", _I), "the demo account has no perpetual set up for"),
    # "…to the demo account: Bybit Demo Trading /v5/order/create: …" said the account twice.
    (re.compile(r"\b(?:Bybit Demo Trading|Deribit testnet) (\S+): ", _I), r"refused (\1): "),
    (re.compile(r"\b(?:bybit|deribit)(?:[ ]?demo(?: trading| account)?| testnet)?(?:'s)?(?![\w-])", _I), _as(_DEMO)),
    # A venue's name run on into a word or an identifier keeps it readable: "BinanceClientError" reads
    # "VenueClientError", "KrakenAPIError" "VenueAPIError", "binancefutures" "venuefutures" (P1-U30).
    # A name that goes on with "-" or "_" is the PM's own label ("binancebot--2026…"), so it stays whole.
    (re.compile(rf"(?:{_NAMES})(?=[a-z]+(?![\w-]))", _I), lambda m: "Venue" if m.group(0)[0].isupper() else "venue"),
    (re.compile(r"\bbinance(?: USD-M perpetuals)?(?:'s)?(?![\w-])", _I), _as(_PERP)),
    (re.compile(r"\bkraken(?: spot)?(?:'s)?(?![\w-])", _I), _as(_SPOT)),
    (re.compile(r"\b(?:[Tt]he|[Aa]n?) the\b"), lambda m: "The" if m.group(0)[0].isupper() else "the"),
]


def no_venues(text):
    """Text the PM reads with any venue name swapped for what it is ("the demo account", "the perpetual venue",
    "the spot venue"). For messages stored before the wording changed, and anything a venue sends back."""
    if not text:
        return text
    out = str(text)
    for pattern, words in _VENUE_WORDS:
        out = pattern.sub(words, out)
    return out
