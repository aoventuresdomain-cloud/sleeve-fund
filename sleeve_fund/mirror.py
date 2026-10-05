"""Demo mirror: copies the fills of chosen paper strategies to a venue's demo account, so the PM can see the
same trades on a real venue's own screens. A strategy on Binance's USDT perpetuals is copied to Bybit Demo
Trading (the same linear perpetual; Binance's own demo is closed to the PM's account); any other is copied to the
Deribit testnet, the fallback. Demo money is not real and the paper journal
stays the record of truth: the mirror never feeds anything back into a strategy, and a mirror order that fails
is noted, not retried, so it can never trade twice.

Fail-closed by construction:
- it only ever talks to the demo hosts, api-demo.bybit.com and test.deribit.com; any other environment or
  host is refused. A live key is useless there: the demo systems don't know it;
- it runs only when DEMO_MIRROR is on and a demo account's key and secret are both set (BYBIT_DEMO_API_KEY
  and BYBIT_DEMO_API_SECRET, DERIBIT_TESTNET_API_KEY and DERIBIT_TESTNET_API_SECRET); otherwise it stays off
  quietly, and a strategy whose demo account isn't set up has its fills noted as skipped;
- it refuses to start if any other venue credential is in its environment, so a live key can't be picked up
  by mistake;
- it runs in its own container: the paper processes never see these variables (and refuse to start if
  they do, sleeve_fund.paper.safety).

A strategy is mirrored when its params carry demo_mirror = true. Each of its fills after the mirror first
sees it is sent as a market order: on Bybit the same quantity of the same perpetual, on Deribit the same
notional rounded to the testnet contract (inverse perpetuals sized in USD); labelled with the strategy and fill.

    python -m sleeve_fund.mirror run
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

FLAG_ENV = "DEMO_MIRROR"
KEY_ENV = "DERIBIT_TESTNET_API_KEY"
SECRET_ENV = "DERIBIT_TESTNET_API_SECRET"
BYBIT_KEY_ENV = "BYBIT_DEMO_API_KEY"
BYBIT_SECRET_ENV = "BYBIT_DEMO_API_SECRET"
# Each demo account: (key variable, secret variable). A strategy's venue picks its account (target_for).
ACCOUNTS = {"DERIBIT": (KEY_ENV, SECRET_ENV), "BYBIT": (BYBIT_KEY_ENV, BYBIT_SECRET_ENV)}
TESTNET_HOST = "test.deribit.com"
BYBIT_DEMO_HOST = "api-demo.bybit.com"
BYBIT_DEMO_URL = f"https://{BYBIT_DEMO_HOST}"
# Bybit linear perpetuals the mirror copies to: symbol -> (quantity step, which is also the smallest order, and the
# smallest notional in USDT), from Bybit's instrument list. A fill on anything else is noted as skipped.
BYBIT_CONTRACTS = {"BTCUSDT": (0.001, 5.0), "ETHUSDT": (0.01, 5.0)}
# Our instrument -> the testnet perpetual and its contract size in USD.
CONTRACTS = {"BTC/USD": ("BTC-PERPETUAL", 10.0), "ETH/USD": ("ETH-PERPETUAL", 1.0)}
POLL_SECONDS = 10
DRIFT_SECONDS = 300


class MirrorRefused(RuntimeError):
    pass


def testnet_url(environment=None) -> str:
    """Deribit's testnet API root. Anything but the testnet environment, or a URL that isn't the testnet
    host, is refused."""
    from nautilus_trader.adapters.deribit import DeribitEnvironment, get_deribit_http_base_url

    env = DeribitEnvironment.TESTNET if environment is None else environment
    if env != DeribitEnvironment.TESTNET:
        raise MirrorRefused(f"the demo mirror trades on Deribit's testnet only, not {env}")
    url = get_deribit_http_base_url(DeribitEnvironment.TESTNET)
    if urllib.parse.urlparse(url).hostname != TESTNET_HOST:
        raise MirrorRefused(f"the testnet URL {url} is not {TESTNET_HOST}")
    return url


@dataclass(frozen=True)
class Settings:
    key: str
    secret: str

    def __repr__(self) -> str:  # never print the key
        return "Settings(key=***, secret=***)"


def settings(environ: dict[str, str] | None = None) -> tuple[dict[str, Settings], str]:
    """The demo accounts the mirror should copy to, {"BYBIT" | "DERIBIT": credentials}: each one whose key
    and secret are both set, once DEMO_MIRROR is on. Empty, with why, when the mirror is off."""
    from sleeve_fund.paper.safety import credential_var

    env = os.environ if environ is None else environ
    demo_vars = {v for pair in ACCOUNTS.values() for v in pair}
    others = sorted(k for k, v in env.items() if v and credential_var(k) and k not in demo_vars)
    if others:
        raise MirrorRefused("the demo mirror refuses to start with other venue credentials in its environment: "
                            + ", ".join(others))
    if env.get(FLAG_ENV, "").strip().lower() not in ("1", "true", "on", "yes"):
        return {}, f"{FLAG_ENV} is off"
    out, why = {}, []
    for name, (key, secret) in ACCOUNTS.items():
        missing = [k for k in (key, secret) if not env.get(k)]
        if missing:
            why.append(f"{' and '.join(missing)} not set")
        else:
            out[name] = Settings(env[key], env[secret])
    return out, "" if out else "; ".join(why)


def _http_get(url: str, headers: dict[str, str]) -> dict:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # Deribit answers errors as JSON with a 400
        try:
            return json.load(e)
        except ValueError:
            raise e from None


class Testnet:
    """Deribit's JSON-RPC over HTTPS, on the testnet only. Mirrors strategies trading on simulated perpetuals:
    inverse perpetuals, sized in USD."""

    label, host, unit = "the Deribit testnet", TESTNET_HOST, "USD"

    def __init__(self, creds: Settings, get=_http_get, environment=None) -> None:
        self.url = testnet_url(environment)
        self._creds, self._get = creds, get
        self._token, self._expires = "", 0.0

    def call(self, method: str, params: dict, private: bool = False) -> dict:
        headers = {"Authorization": f"Bearer {self._auth()}"} if private else {}
        out = self._get(f"{self.url}/api/v2/{method}?{urllib.parse.urlencode(params)}", headers)
        if "error" in out:
            err = out["error"]
            raise RuntimeError(f"Deribit testnet {method}: {err.get('message', err)} ({err.get('data', '')})")
        return out["result"]

    def _auth(self) -> str:
        if not self._token or time.time() > self._expires - 60:
            r = self.call("public/auth", {"grant_type": "client_credentials", "client_id": self._creds.key,
                                          "client_secret": self._creds.secret})
            self._token, self._expires = r["access_token"], time.time() + float(r.get("expires_in", 0))
        return self._token

    @staticmethod
    def owns(instrument: str) -> bool:
        return instrument in {name for name, _ in CONTRACTS.values()}

    @staticmethod
    def size(sleeve, fill: dict) -> tuple[str | None, float, str]:
        """(testnet instrument, amount in USD, why it is skipped or "") for copying this fill."""
        contract = CONTRACTS.get(sleeve.instrument)
        if contract is None:
            return None, 0.0, f"no testnet perpetual for {sleeve.instrument}"
        instrument, size = contract
        amount = round(fill["qty"] * fill["price"] / size) * size
        if amount <= 0:
            return instrument, 0.0, f"{fill['qty'] * fill['price']:,.2f} USD is under one {size:g} USD contract"
        return instrument, amount, ""

    def market(self, side: str, instrument: str, amount: float, label: str) -> tuple[str, float | None]:
        """Send a market order; returns (order id, average price)."""
        method = "private/buy" if side == "BUY" else "private/sell"
        order = self.call(method, {"instrument_name": instrument, "amount": amount, "type": "market",
                                   "label": label[:64]}, private=True).get("order", {})
        return str(order.get("order_id", "")), order.get("average_price")

    def position(self, instrument: str) -> float:
        return float(self.call("private/get_position", {"instrument_name": instrument}, private=True)["size"])


def bybit_demo_url(url: str = BYBIT_DEMO_URL) -> str:
    """Bybit Demo Trading's V5 API root. Any other host, the live one and the testnet included, is refused."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != BYBIT_DEMO_HOST:
        raise MirrorRefused(f"the demo mirror trades on Bybit Demo Trading ({BYBIT_DEMO_HOST}) only, not {url}")
    return url.rstrip("/")


def _http(method: str, url: str, headers: dict[str, str], body: bytes | None = None):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # Bybit answers most errors as JSON: {"retCode": 10001, "retMsg": "..."}
        try:
            return json.load(e)
        except ValueError:
            raise e from None


class BybitDemo:
    """Bybit Demo Trading's V5 REST API, signed as Bybit signs it (HMAC-SHA256 of timestamp, key, receive window
    and the query or body), on the demo host only. Mirrors strategies on Binance's USD-M perpetuals to the same
    linear USDT perpetual on Bybit, the same quantity. Bybit's demo allows 5 requests a second; the mirror sends
    one or two per fill."""

    label, host, unit = "Bybit Demo Trading", BYBIT_DEMO_HOST, "contracts"
    RECV_WINDOW = "10000"

    def __init__(self, creds: Settings, http=_http, url: str = BYBIT_DEMO_URL, clock=time.time) -> None:
        self.url = bybit_demo_url(url)
        self._creds, self._http, self._clock = creds, http, clock

    def call(self, method: str, path: str, params: dict | None = None):
        """A private V5 call; returns its result. GET sends params as the query, POST as a JSON body."""
        params = dict(params or {})
        ts = str(int(self._clock() * 1000))
        if method == "GET":
            payload, body, url = urllib.parse.urlencode(params), None, f"{self.url}{path}"
            if params:
                url += f"?{payload}"
        else:
            payload = json.dumps(params, separators=(",", ":"))
            body, url = payload.encode(), f"{self.url}{path}"
        sig = hmac.new(self._creds.secret.encode(), (ts + self._creds.key + self.RECV_WINDOW + payload).encode(),
                       hashlib.sha256).hexdigest()
        headers = {"X-BAPI-API-KEY": self._creds.key, "X-BAPI-TIMESTAMP": ts, "X-BAPI-RECV-WINDOW": self.RECV_WINDOW,
                   "X-BAPI-SIGN": sig, "Content-Type": "application/json"}
        out = self._http(method, url, headers, body)
        if not isinstance(out, dict) or out.get("retCode") != 0:
            msg = out.get("retMsg", out) if isinstance(out, dict) else out
            code = out.get("retCode", "") if isinstance(out, dict) else ""
            raise RuntimeError(f"Bybit Demo Trading {path}: {msg} ({code})")
        return out.get("result") or {}

    @staticmethod
    def owns(instrument: str) -> bool:
        return instrument in BYBIT_CONTRACTS

    @staticmethod
    def size(sleeve, fill: dict) -> tuple[str | None, float, str]:
        """(symbol, quantity, why it is skipped or ""): the fill's own quantity, at the contract's step, if it
        clears the smallest order and notional Bybit takes."""
        symbol = sleeve.instrument.replace("/", "").upper()
        contract = BYBIT_CONTRACTS.get(symbol)
        if contract is None:
            return None, 0.0, f"no Bybit demo perpetual set up for {sleeve.instrument}"
        step, min_notional = contract
        qty = round(round(float(fill["qty"]) / step) * step, 8)
        if qty < step or qty * fill["price"] < min_notional:
            return symbol, 0.0, (f"{fill['qty']:.8g} ({fill['qty'] * fill['price']:,.2f} USDT) is under the smallest "
                                 f"order Bybit takes ({step:g}, {min_notional:g} USDT)")
        return symbol, qty, ""

    def market(self, side: str, instrument: str, amount: float, label: str) -> tuple[str, float | None]:
        # Bybit takes order link ids of up to 36 characters: letters, digits, - and _.
        link = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label)[:36]
        r = self.call("POST", "/v5/order/create", {
            "category": "linear", "symbol": instrument, "side": "Buy" if side == "BUY" else "Sell",
            "orderType": "Market", "qty": f"{amount:.8f}".rstrip("0").rstrip("."), "positionIdx": 0,
            "orderLinkId": link})
        order_id = str(r.get("orderId", ""))
        try:  # the create answer has no price; one look-up for it, and a miss costs only the price
            rows = self.call("GET", "/v5/order/realtime", {"category": "linear", "orderId": order_id}).get("list") or []
            avg = (float(rows[0].get("avgPrice") or 0) or None) if rows else None
        except Exception:  # noqa: BLE001 - the order went in; its price is a nicety
            avg = None
        return order_id, avg

    def position(self, instrument: str) -> float:
        rows = self.call("GET", "/v5/position/list", {"category": "linear", "symbol": instrument}).get("list") or []
        sides = {"Buy": 1.0, "Sell": -1.0}
        return sum(sides.get(r.get("side"), 0.0) * float(r.get("size") or 0) for r in rows)


def target_for(sleeve) -> str:
    """The demo account a strategy's fills go to: Bybit Demo Trading for a strategy on Binance's USDT
    perpetuals (Binance's own demo is closed to the PM's account), else the Deribit testnet."""
    return "BYBIT" if (sleeve.venue or "").upper() == "BINANCE" else "DERIBIT"


def mirrored(store) -> list:
    """The strategies whose params ask for the demo mirror."""
    return [s for s in store.sleeves() if s.params.get("demo_mirror")]


def mirror_once(store, targets: dict) -> int:
    """Copy every mirrored strategy's new fills to its demo account; returns how many orders were sent.
    targets: {"BYBIT" | "DERIBIT": the demo account's client}, those set up."""
    sent = 0
    for s in mirrored(store):
        name = target_for(s)
        venue = targets.get(name)
        mark = store.mirror_watermark(s.name)
        if mark is None:  # first sight: mirror from now on, not the strategy's history
            last = store.last_fill_id(s.name)
            store.record_mirror(s.name, fill_id=last, status="start", message="mirror started")
            where = (f"{venue.label} ({venue.host}); demo money is not real" if venue is not None else
                     f"its demo account, once that is set up ({' and '.join(ACCOUNTS[name])})")
            store.event(s.name, "info", "mirror_start",
                        f"Demo mirror on: each new fill is copied to {where}, and the paper book stays the record")
            continue
        for f in store.fills_after(s.name, mark):
            if venue is None:
                store.record_mirror(s.name, fill_id=f["id"], status="skipped",
                                    message=f"no {name.title()} demo account set up")
                continue
            try:
                instrument, amount, why = venue.size(s, f)
            except Exception as e:  # noqa: BLE001 - the demo venue's contract list unreachable: noted, not retried
                instrument, amount, why = None, 0.0, f"couldn't size it: {str(e)[:200]}"
            sign = 1 if f["side"] == "BUY" else -1
            if why:
                store.record_mirror(s.name, fill_id=f["id"], status="skipped", instrument=instrument or "", message=why)
                continue
            try:
                order_id, price = venue.market(f["side"], instrument, amount, f"{s.name}:{f['id']}")
            except Exception as e:  # noqa: BLE001 - noted, never retried, so it can't trade twice
                store.record_mirror(s.name, fill_id=f["id"], status="error", instrument=instrument,
                                    amount=sign * amount, message=str(e)[:500])
                store.event(s.name, "warning", "mirror_failed",
                            f"Demo mirror couldn't copy the {f['side'].lower()} of {f['qty']:.8g} at {f['price']:,.2f} "
                            f"to {venue.label}: {str(e)[:200]}. The paper book is unaffected.")
                continue
            store.record_mirror(s.name, fill_id=f["id"], status="filled", instrument=instrument, amount=sign * amount,
                                price=price, order_id=order_id)
            sent += 1
    return sent


def check_drift(store, targets: dict, last: dict[str, float]) -> dict[str, float]:
    """Warn once per change when a demo account's position differs from what the mirror put on (an order that
    failed, or something traded by hand on the demo account)."""
    out = dict(last)
    for instrument, expected in store.mirror_positions().items():
        venue = next((t for t in targets.values() if t.owns(instrument)), None)
        if venue is None:
            continue
        held = venue.position(instrument)
        gap = held - expected
        if abs(gap) >= 1e-9 and last.get(instrument) != gap:
            store.event(None, "warning", "mirror_drift",
                        f"Demo mirror: {venue.label} holds {held:+,.6g} {venue.unit} of {instrument}, the mirrored "
                        f"strategies {expected:+,.6g}. The paper book is unaffected.")
        out[instrument] = gap
    return out


def run(store, targets: dict, poll: float = POLL_SECONDS, stop=None) -> None:
    drift, checked = {}, 0.0
    while stop is None or not stop():
        try:
            mirror_once(store, targets)
            if time.monotonic() - checked > DRIFT_SECONDS:
                drift, checked = check_drift(store, targets, drift), time.monotonic()
        except Exception as e:  # noqa: BLE001 - a database or network blip: try again next poll
            print(f"demo mirror: {e}", file=sys.stderr)
        time.sleep(poll)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.mirror", description="Demo-account mirror of paper fills")
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args(argv)
    creds, why = settings()
    if not creds:
        # Off quietly: no events, no restarts; the container just idles.
        print(f"demo mirror off: {why}", flush=True)
        while True:
            time.sleep(3600)
    from sleeve_fund.store import Store

    clients = {"BYBIT": BybitDemo, "DERIBIT": Testnet}
    targets = {name: clients[name](c) for name, c in creds.items()}
    print("demo mirror on: " + ", ".join(f"{t.label} ({t.url})" for t in targets.values()), flush=True)
    run(Store(), targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
