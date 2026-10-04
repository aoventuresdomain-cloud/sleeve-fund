"""Demo mirror: copies the fills of chosen paper strategies to a Deribit TESTNET account, so the PM can see
the same trades on a real venue's demo screens. Testnet money is not real and the paper journal stays the
record of truth: the mirror never feeds anything back into a strategy, and a mirror order that fails is
noted, not retried, so it can never trade twice.

Fail-closed by construction:
- it only ever talks to Deribit's testnet: the URL comes from DeribitEnvironment.TESTNET and any other
  environment or host is refused;
- it runs only when DEMO_MIRROR is on and both DERIBIT_TESTNET_API_KEY and DERIBIT_TESTNET_API_SECRET are
  set; otherwise it stays off quietly;
- it refuses to start if any other venue credential is in its environment, so a live key can't be
  picked up by mistake;
- it runs in its own container: the paper processes never see these variables (and refuse to start if
  they do, sleeve_fund.paper.safety).

A strategy is mirrored when its params carry demo_mirror = true. Each of its fills after the mirror first
sees it is sent as a market order of the same notional, rounded to the testnet contract (inverse
perpetuals sized in USD), labelled with the strategy and fill id.

    python -m sleeve_fund.mirror run
"""

from __future__ import annotations

import argparse
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
TESTNET_HOST = "test.deribit.com"
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


def settings(environ: dict[str, str] | None = None) -> tuple[Settings | None, str]:
    """The testnet credentials when the mirror should run, else None and why it is off."""
    from sleeve_fund.paper.safety import credential_var

    env = os.environ if environ is None else environ
    others = sorted(k for k, v in env.items() if v and credential_var(k) and k not in (KEY_ENV, SECRET_ENV))
    if others:
        raise MirrorRefused("the demo mirror refuses to start with other venue credentials in its environment: "
                            + ", ".join(others))
    if env.get(FLAG_ENV, "").strip().lower() not in ("1", "true", "on", "yes"):
        return None, f"{FLAG_ENV} is off"
    missing = [k for k in (KEY_ENV, SECRET_ENV) if not env.get(k)]
    if missing:
        return None, f"{' and '.join(missing)} not set"
    return Settings(env[KEY_ENV], env[SECRET_ENV]), ""


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
    """Deribit's JSON-RPC over HTTPS, on the testnet only."""

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

    def market(self, side: str, instrument: str, amount: float, label: str) -> dict:
        method = "private/buy" if side == "BUY" else "private/sell"
        return self.call(method, {"instrument_name": instrument, "amount": amount, "type": "market",
                                  "label": label[:64]}, private=True)

    def position(self, instrument: str) -> float:
        return float(self.call("private/get_position", {"instrument_name": instrument}, private=True)["size"])


def mirrored(store) -> list:
    """The strategies whose params ask for the demo mirror."""
    return [s for s in store.sleeves() if s.params.get("demo_mirror")]


def mirror_once(store, venue: Testnet) -> int:
    """Copy every mirrored strategy's new fills to the testnet; returns how many orders were sent."""
    sent = 0
    for s in mirrored(store):
        mark = store.mirror_watermark(s.name)
        if mark is None:  # first sight: mirror from now on, not the strategy's history
            last = store.last_fill_id(s.name)
            store.record_mirror(s.name, fill_id=last, status="start", message="mirror started")
            store.event(s.name, "info", "mirror_start",
                        f"Demo mirror on: each new fill is copied to the Deribit testnet ({TESTNET_HOST}); "
                        "testnet money is not real and the paper book stays the record")
            continue
        contract = CONTRACTS.get(s.instrument)
        for f in store.fills_after(s.name, mark):
            if contract is None:
                store.record_mirror(s.name, fill_id=f["id"], status="skipped",
                                    message=f"no testnet perpetual for {s.instrument}")
                continue
            instrument, size = contract
            amount = round(f["qty"] * f["price"] / size) * size
            sign = 1 if f["side"] == "BUY" else -1
            if amount <= 0:
                store.record_mirror(s.name, fill_id=f["id"], status="skipped", instrument=instrument,
                                    message=f"{f['qty'] * f['price']:,.2f} USD is under one {size:g} USD contract")
                continue
            try:
                r = venue.market(f["side"], instrument, amount, f"{s.name}:{f['id']}")
            except Exception as e:  # noqa: BLE001 - noted, never retried, so it can't trade twice
                store.record_mirror(s.name, fill_id=f["id"], status="error", instrument=instrument,
                                    amount=sign * amount, message=str(e)[:500])
                store.event(s.name, "warning", "mirror_failed",
                            f"Demo mirror couldn't copy the {f['side'].lower()} of {f['qty']:.8g} at {f['price']:,.2f} "
                            f"to the testnet: {str(e)[:200]}. The paper book is unaffected.")
                continue
            order = r.get("order", {})
            store.record_mirror(s.name, fill_id=f["id"], status="filled", instrument=instrument, amount=sign * amount,
                                price=order.get("average_price"), order_id=str(order.get("order_id", "")))
            sent += 1
    return sent


def check_drift(store, venue: Testnet, last: dict[str, float]) -> dict[str, float]:
    """Warn once per change when the testnet position differs from what the mirror put on (an order that
    failed, or something traded by hand on the testnet account)."""
    out = dict(last)
    for instrument, expected in store.mirror_positions().items():
        held = venue.position(instrument)
        gap = held - expected
        if abs(gap) >= 1e-9 and last.get(instrument) != gap:
            store.event(None, "warning", "mirror_drift",
                        f"Demo mirror: the testnet holds {held:+,.0f} USD of {instrument}, the mirrored strategies "
                        f"{expected:+,.0f}. The paper book is unaffected.")
        out[instrument] = gap
    return out


def run(store, venue: Testnet, poll: float = POLL_SECONDS, stop=None) -> None:
    drift, checked = {}, 0.0
    while stop is None or not stop():
        try:
            mirror_once(store, venue)
            if time.monotonic() - checked > DRIFT_SECONDS:
                drift, checked = check_drift(store, venue, drift), time.monotonic()
        except Exception as e:  # noqa: BLE001 - a database or network blip: try again next poll
            print(f"demo mirror: {e}", file=sys.stderr)
        time.sleep(poll)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.mirror", description="Deribit testnet demo mirror")
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args(argv)
    creds, why = settings()
    if creds is None:
        # Off quietly: no events, no restarts; the container just idles.
        print(f"demo mirror off: {why}", flush=True)
        while True:
            time.sleep(3600)
    from sleeve_fund.store import Store

    venue = Testnet(creds)
    print(f"demo mirror on: {venue.url}", flush=True)
    run(Store(), venue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
