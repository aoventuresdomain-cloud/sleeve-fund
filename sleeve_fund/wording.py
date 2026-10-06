"""Wording the PM reads, wherever it is shown: the dashboard and the outside alerts alike."""

from __future__ import annotations

import re

_DEMO, _PERP, _SPOT = "the demo account", "the perpetual venue", "the spot venue"


def _as(words: str):
    """Replace a venue name with words, keeping a possessive: "Bybit's API" reads "the demo account's API"."""
    return lambda m: words + ("'s" if m.group(0).endswith("'s") else "")


# Venue names in text the PM reads (mirror reasons, alerts, the decision log), longest first so "Bybit Demo
# Trading" goes whole. Case-sensitive on purpose: account names such as "kraken-live" and env names such as
# BYBIT_DEMO_API_KEY are the PM's own labels and stay as they are. Only Setup, Accounts names venues (QA U8).
# The first few rephrase whole sentences stored before the wording changed, so they read naturally (P1-U21).
_VENUE_WORDS = [
    # An instrument id's venue suffix goes, the id stays: "BTCUSDT-PERP.BINANCE" reads "BTCUSDT-PERP".
    (re.compile(r"(?<=\S)\.(?:BINANCE|KRAKEN|BYBIT|DERIBIT)\b"), ""),
    (re.compile(r"\bno (?:Bybit|Deribit) demo account set up\b"), "the demo account isn't set up"),
    (re.compile(r"\bno (?:Bybit|Deribit) demo perpetual set up for\b"), "the demo account has no perpetual set up for"),
    # "…to the demo account: Bybit Demo Trading /v5/order/create: …" said the account twice.
    (re.compile(r"\b(?:Bybit Demo Trading|Deribit testnet) (\S+): "), r"refused (\1): "),
    (re.compile(r"\b(?:Bybit|Deribit)(?: [Dd]emo(?: Trading| account)?| [Tt]estnet)?(?:'s)?(?!\w)"), _as(_DEMO)),
    (re.compile(r"\bBinance(?: USD-M perpetuals)?(?:'s)?(?!\w)"), _as(_PERP)),
    (re.compile(r"\bKraken(?: spot)?(?:'s)?(?!\w)"), _as(_SPOT)),
    (re.compile(r"\b(?:BYBIT|DERIBIT)\b(?!_)"), _DEMO),
    (re.compile(r"\bBINANCE\b(?!_)"), _PERP),
    (re.compile(r"\bKRAKEN\b(?!_)"), _SPOT),
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
