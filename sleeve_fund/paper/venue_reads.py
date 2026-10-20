"""The order client's connect check reads (B-05 M4; drill-prep MUST 3, 9 clock, 10, 12): the key's permissions, the
account's margin mode, each instrument's position settings and the venue's clock, by signed GETs through the shared
limiter (never held back) with venue_stops' signing. Reads only: nothing here writes to the account. An unanswered
read is None, which the check fails on (exec_client.connect_check), never a pass."""

from __future__ import annotations

from urllib.parse import urlencode

from sleeve_fund.paper.exec_client import ExecSettings, Limiter, connect_check
from sleeve_fund.paper.venue_stops import VenueStops


class VenueReads:
    def __init__(self, base_url: str, *, api_key: str, api_secret: str, limiter: Limiter, now_ms=None,
                 timeout_s: float = 5.0) -> None:
        self._client = VenueStops(base_url, symbol="", api_key=api_key, api_secret=api_secret, limiter=limiter,
                                  now_ms=now_ms, timeout_s=timeout_s)
        self.limiter = limiter

    def __repr__(self) -> str:  # never the key or the secret
        return f"VenueReads({self._client.base_url})"

    def _get(self, endpoint: str, query: dict | None = None) -> dict | None:
        if endpoint in self.limiter.limits:
            self.limiter.take(endpoint, opening=False)
        q = urlencode(query or {})
        return self._client._send("GET", f"/v5/{endpoint}" + (f"?{q}" if q else ""), q, None)

    def key(self) -> dict | None:
        return self._get("user/query-api")

    def account(self) -> dict | None:
        answer = self._get("account/info")
        return answer if answer and answer.get("retCode") == 0 else None

    def position(self, symbol: str) -> dict | None:
        answer = self._get("position/list", {"category": "linear", "symbol": symbol})
        if not answer or answer.get("retCode") != 0:
            return None
        rows = (answer.get("result") or {}).get("list") or []
        return next((r for r in rows if r.get("symbol") == symbol), None)

    def server_ms(self) -> int | None:
        answer = self._get("market/time")
        try:
            return int(int((answer.get("result") or {})["timeNano"]) // 1_000_000)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def check(self, s: ExecSettings, *, drill_go: bool, caps: dict[str, float], local_ms) -> list[tuple[str, str]]:
        """connect_check on what the venue answers now. local_ms: this machine's clock, read after the venue's."""
        key, account = self.key(), self.account()
        positions = {sym: self.position(sym) for sym in caps}
        server = self.server_ms()
        return connect_check(s, drill_go=drill_go, key=key, account=account, positions=positions, caps=caps,
                             server_ms=server, local_ms=local_ms())
