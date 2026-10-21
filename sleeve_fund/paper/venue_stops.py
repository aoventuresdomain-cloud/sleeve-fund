"""Own signed call for protective stops (Advisor OWN-STOP-SPEC and STOP-CLAIM, 15 Oct 2026; spec
/mnt/project-files/sleeve-fund/qd/own-stop-path-spec.md). Drill prep, built OFF: nothing calls it until the exchange
client is switched on, and resting stops stay refused ON until the docs check (OWN-STOP-SPEC MUST 4) is done.

The framework's exchange client can't send the last-price trigger and drops a trigger-price amend, so a resting stop
is placed, cancelled and read back here instead. It SENDS only: what to send is decided by paper.stop_mover. Only
reduce-only conditional market stops for the Strategy's own quantity (never a whole-position stop: the account is
shared), only with ids in the fund's namespace, only from the owner process, through the shared limiter.

Ownership of an order the framework client hands back is decided by the fund's journals (classify): this Strategy's
is booked, a sibling Strategy's is ignored silently, one of ours that no journal owns is an orphan (cancelled on a
restart), and one outside the fund's namespace is foreign (ForeignWatch: CRITICAL once fund-wide, entries on its
instrument held fund-wide until it is gone and the account reconciles; never cancelled, never booked)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from urllib.parse import urlencode

from sleeve_fund.paper.exec_client import LINK_ID, Limiter
from sleeve_fund.paper.stop_mover import ours

RECV_WINDOW = "5000"
DIRECTION = {"SELL": 2, "BUY": 1}  # a long's sell stop triggers on a fall (2), a short's buy stop on a rise (1)


class VenueStops:
    def __init__(self, base_url: str, *, symbol: str, api_key: str, api_secret: str, limiter: Limiter,
                 now_ms=None, timeout_s: float = 5.0, read_only: bool = False) -> None:
        """read_only: the connect check's reads (venue_reads); every write raises before anything is sent."""
        self.base_url, self.symbol, self.limiter, self.timeout_s = base_url.rstrip("/"), symbol, limiter, timeout_s
        self.read_only = read_only
        self._key, self._secret = api_key, api_secret
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    def __repr__(self) -> str:  # never the key or the secret
        return f"VenueStops({self.base_url}, {self.symbol})"

    # the three calls ------------------------------------------------------------------------------------------------
    def place(self, order_id: str, *, side: str, qty, level) -> tuple:
        """A reduce-only conditional market stop for `qty` at `level`, triggered on the last price."""
        self._check_id(order_id)
        return self._post("order/create", {
            "category": "linear", "symbol": self.symbol, "side": side.title(), "orderType": "Market",
            "qty": str(qty), "triggerPrice": str(level), "triggerBy": "LastPrice",
            "triggerDirection": DIRECTION[side.upper()], "reduceOnly": True, "positionIdx": 0,
            "orderLinkId": order_id})

    def cancel(self, order_id: str) -> tuple:
        self._check_id(order_id)
        return self._post("order/cancel", {"category": "linear", "symbol": self.symbol, "orderLinkId": order_id})

    def lookup(self, order_id: str) -> dict | None:
        """The venue's open-orders row for the order (for exec_client.stop_drift), or None when it reports none. Raises
        ConnectionError when there is no usable answer (none, or a refused query): a failed lookup is not evidence of
        anything (OWN-STOP-SPEC Q2)."""
        return self._find("order/realtime", order_id)

    def history(self, order_id: str) -> dict | None:
        """The venue's order-history row for the order (a filled, cancelled or refused order may show only here), or
        None when it reports none; ConnectionError as lookup (exec_client.resend_check reads both)."""
        return self._find("order/history", order_id)

    def _find(self, endpoint: str, order_id: str) -> dict | None:
        self.limiter.take(endpoint, opening=False)
        query = urlencode({"category": "linear", "symbol": self.symbol, "orderLinkId": order_id})
        answer = self._send("GET", f"/v5/{endpoint}?{query}", query, None)
        if answer is None or answer.get("retCode") != 0:
            raise ConnectionError(f"no answer looking up {order_id} ({endpoint})")
        rows = (answer.get("result") or {}).get("list") or []
        return next((r for r in rows if r.get("orderLinkId") == order_id), None)

    # plumbing --------------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_id(order_id: str) -> None:
        if not ours(order_id) or not LINK_ID.fullmatch(order_id):
            raise ValueError(f"{order_id!r} is not a fund order id of at most 36 characters: never sent")

    def _post(self, endpoint: str, body: dict) -> tuple:
        if self.read_only:  # place and cancel both come here: a read-only client never writes (TEC-B05-KEYS)
            raise PermissionError(f"a read-only venue client never writes ({endpoint})")
        self.limiter.take(endpoint, opening=False)  # protective: never held back
        raw = json.dumps(body, separators=(",", ":"))
        answer = self._send("POST", f"/v5/{endpoint}", raw, raw.encode())
        if answer is None:
            return ("unknown", "no answer")
        if answer.get("retCode") != 0:
            return ("rejected", answer.get("retCode"), answer.get("retMsg", ""))
        return ("accepted", (answer.get("result") or {}).get("orderId"))

    def _send(self, method: str, path: str, signed: str, data: bytes | None) -> dict | None:
        """The venue's signing scheme: HMAC-SHA256 of timestamp + key + recv window + (body or query string)."""
        if self.read_only and method != "GET":  # the one place a request goes out: no write gets round it
            raise PermissionError(f"a read-only venue client never writes ({method} {path})")
        ts = str(self._now_ms())
        sign = hmac.new(self._secret.encode(), f"{ts}{self._key}{RECV_WINDOW}{signed}".encode(),
                        hashlib.sha256).hexdigest()
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers={
            "X-BAPI-API-KEY": self._key, "X-BAPI-TIMESTAMP": ts, "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "X-BAPI-SIGN": sign, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return None


def classify(order_id: str, *, own: set, fund: set) -> str:
    """"book" (this Strategy's journal), "sibling" (another Strategy's in the fund), "orphan" (ours, owned by no
    journal) or "foreign" (not the fund's at all)."""
    if order_id in own:
        return "book"
    if order_id in fund:
        return "sibling"
    return "orphan" if ours(order_id) else "foreign"


class OwnershipGate:
    """One Strategy's view of the orders the framework client hands it (STOP-CLAIM 3): one of ours not yet in any
    journal is held for `grace` and re-checked before it is called an orphan, since its row may not be visible yet."""

    def __init__(self, own: set, grace: timedelta = timedelta(seconds=2)) -> None:
        self.own, self.grace = own, grace
        self.held: dict[str, datetime] = {}

    def on_claimed(self, order_id: str, fund: set, now: datetime) -> str:
        """"book", "ignore" (a sibling's), "hold" (ours, row not visible yet) or "foreign"."""
        kind = classify(order_id, own=self.own, fund=fund)
        if kind == "orphan":
            self.held.setdefault(order_id, now)
            return "hold"
        return {"book": "book", "sibling": "ignore", "foreign": "foreign"}[kind]

    def recheck(self, fund: set, now: datetime) -> list[tuple[str, str]]:
        out = []
        for oid, since in list(self.held.items()):
            kind = classify(oid, own=self.own, fund=fund)
            if kind in ("book", "sibling"):
                out.append((oid, "book" if kind == "book" else "ignore"))
                del self.held[oid]
            elif now - since > self.grace:
                out.append((oid, "orphan"))
                del self.held[oid]
        return out


class ForeignWatch:
    """Orders on the account that are not the fund's (OWN-STOP-SPEC Q1): a CRITICAL alert once fund-wide (the
    journal's record of it is shared through already_alerted / mark_alerted), new entries on its instrument held
    fund-wide until it is gone AND the account reconciles; never cancelled, never booked. Exits and stops run."""

    def __init__(self, *, already_alerted, mark_alerted) -> None:
        self._already, self._mark = already_alerted, mark_alerted
        self.open: dict[str, str] = {}  # foreign order id -> instrument

    def on_foreign(self, order_id: str, instrument: str, *, status: str) -> list[tuple]:
        out = []
        if instrument not in self.held():
            out.append(("hold", instrument))
        self.open[order_id] = instrument
        if not self._already(order_id):
            self._mark(order_id)
            out.append(("alert", "critical", (f"foreign order on the account: {order_id} on {instrument} ({status}) "
                                              "is in no journal of the fund. New entries on it are held fund-wide; "
                                              "it is never cancelled or booked by the fund")))
        return out

    def on_gone(self, order_id: str, *, reconcile_ties: bool) -> list[tuple]:
        if not reconcile_ties:
            return []
        instrument = self.open.pop(order_id, None)
        if instrument is None or instrument in self.held():
            return []
        return [("release", instrument)]

    def held(self) -> set[str]:
        return set(self.open.values())
