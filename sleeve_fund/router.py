"""Execution router v1 (P2-4): one fixed venue per strategy. The pure core; paper's node builder and the strategy's
order path call it and keep no rule of their own. Nothing here reads the database, the clock or the environment.

- route() resolves the strategy's pinned venue through the registered venue profiles, so adding a profile needs no
  change here. It routes paper accounts only: paper trading only until G2.
- check() refuses any order whose instrument isn't on the strategy's pinned venue. The caller logs the refusal as an
  error event and sends nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

PAPER = "paper"


class RouteError(ValueError):
    """An order or a route the router refuses. `reason` is the sentence for the journal and the strategy page."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class VenueMismatch(RouteError):
    """An order for an instrument on a venue other than the strategy's pinned one."""


@dataclass(frozen=True)
class Route:
    strategy: str
    venue: str  # the venue profile's name, upper case, as instrument ids carry it
    account_id: str  # the paper execution account, "<VENUE>-PAPER-<tag>"


def route(strategy: str, venue: str, profiles: Mapping[str, object], *, tag: str, account_kind: str = PAPER) -> Route:
    """The strategy's route. venue: its pinned venue (the default venue already applied); profiles: the registered
    venue profiles by name; tag: the strategy's node tag, of which the account id keeps the first 20 characters."""
    if account_kind != PAPER:
        raise RouteError(f"{strategy} is on a {account_kind} account; orders are routed to paper accounts only.")
    if not venue:
        raise RouteError(f"{strategy} has no venue to route to.")
    name = venue.upper()
    if name not in profiles:
        raise RouteError(f"{strategy} is pinned to a venue with no registered profile.")
    if not tag:
        raise ValueError("a route needs the strategy's node tag")
    return Route(strategy=strategy, venue=name, account_id=f"{name}-PAPER-{tag[:20]}")


def venue_of(instrument_id: str) -> str:
    """The venue an instrument id names: BTC/USD.KRAKEN -> KRAKEN."""
    symbol, dot, venue = str(instrument_id).rpartition(".")
    if not dot or not symbol or not venue:
        raise ValueError(f"{instrument_id!r} is not an instrument id with a venue")
    return venue.upper()


def check(strategy: str, instrument_id: str, pinned_venue: str) -> None:
    """Refuse (VenueMismatch) an order whose instrument isn't on the strategy's pinned venue."""
    got, want = venue_of(instrument_id), pinned_venue.upper()
    if got != want:
        raise VenueMismatch(f"Order refused: {strategy} trades on its own venue only, and this order was for "
                            f"another venue's instrument.")
