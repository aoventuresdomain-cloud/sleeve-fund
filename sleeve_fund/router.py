"""Execution router v1 (P2-4): one fixed venue per strategy. The pure core; paper's node builder and the strategy's
order path call it and keep no rule of their own. Nothing here reads the database, the clock or the environment.

- route() resolves the strategy's pinned venue through the registered venue profiles, so adding a profile needs no
  change here. It routes paper accounts only: paper trading only until G2.
- check() refuses any order whose instrument isn't on the strategy's pinned venue. The caller logs the refusal as an
  error event and sends nothing.
- start_mode() decides how a strategy whose journal last traded on another venue than its pin may start (F-WED1-3).
- venue_for() is the one place a strategy's stored pin becomes its venue: the node's config, the runtime and every
  orders row (orders.venue) read it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sleeve_fund.venues import DEFAULT_VENUE, base_venue

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
    return Route(strategy=strategy, venue=name, account_id=f"{name}-PAPER-{cut_tag(tag)}")


def cut_tag(tag: str) -> str:
    """A node tag cut to the 20 characters ids keep, with no "-" left at the end (B-29): the engine refuses an id
    whose last part is empty, so "PAPER-CANDLE-STEP-BTCUSDT-" crashed the node of the form's own suggested name. A
    cut that didn't end in "-" is unchanged, so every name that started before keeps its ids."""
    cut = tag[:20].rstrip("-")
    if not cut:  # a name with no letter or digit: it would build "PAPER-", which the engine refuses
        raise ValueError(f"a Strategy's name needs a letter or digit to make its node ids: {tag!r}")
    return cut


def venue_for(pinned: str | None) -> str:
    """A strategy's venue from its stored pin (sleeve_venues): the pin, or the default venue for a strategy saved
    before strategies carried one; upper case, a portfolio run's clone (VENUE_P1) as its base venue."""
    return base_venue(pinned or DEFAULT_VENUE).upper()


def venue_of(instrument_id: str) -> str:
    """The venue an instrument id names: the part after its last dot, upper case."""
    symbol, dot, venue = str(instrument_id).rpartition(".")
    if not dot or not symbol or not venue:
        raise ValueError(f"{instrument_id!r} is not an instrument id with a venue")
    return venue.upper()


def check(strategy: str, instrument_id: str, pinned_venue: str) -> None:
    """Refuse (VenueMismatch) an order whose instrument isn't on the strategy's pinned venue."""
    got, want = base_venue(venue_of(instrument_id)), venue_for(pinned_venue)  # a run's clone is its base venue
    if got != want:
        raise VenueMismatch(f"Order refused: {strategy} trades on its own venue only, and this order was for "
                            f"another venue's instrument.")


NORMAL, EXITS_ONLY, REFUSE = "normal", "exits_only", "refuse"
# Why a strategy whose journal last traded on another venue than its pin doesn't start to trade, by start_mode. For
# the PM, so no venue names; the caller's log line names them.
VENUE_MOVED = {EXITS_ONLY: "its open position is on another venue than the one it is set to trade on, so it only "
                           "closes that position there",
               REFUSE: "it last traded on another venue than the one it is set to trade on; a reset starts a new run "
                       "on its own venue"}


def start_mode(pin_venue: str, traded_venue: str | None, holding: bool) -> str:
    """How a strategy starts when its journal last traded on another venue than its pin (F-WED1-3, Advisor 14 Oct
    03:35 UK). pin_venue: venue_for(its pin); traded_venue: the venue of the order behind its latest fill (None with
    no fills). The same venue, or no fills: NORMAL, on the pin. Another venue while holding: EXITS_ONLY, on the
    traded venue, so the open position's stop and target close it where it is and nothing opens. Another venue while
    flat: REFUSE; a reset starts a new run on the pin."""
    if traded_venue is None or base_venue(traded_venue).upper() == venue_for(pin_venue):
        return NORMAL
    return EXITS_ONLY if holding else REFUSE
