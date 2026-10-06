"""Demo mirror: copies the fills of chosen paper strategies to a venue's demo account, so the PM can see the
same trades on a real venue's own screens. A strategy on Binance's USDT perpetuals is copied to Bybit Demo
Trading (the same linear perpetual; Binance's own demo is closed to the PM's account); any other is copied to the
Deribit testnet, the fallback. Demo money is not real and the paper journal
stays the record of truth: the mirror never feeds anything back into a strategy, and a mirror order that fails
is noted, not retried, so it can never trade twice. Before copying to Bybit Demo it sets isolated margin at the
paper leverage (prepare_margin). Paper's isolated margin is the notional over that leverage (U13-2), so for one
strategy on a symbol the demo position's margin and liquidation price match the paper book. Strategies that share
a Bybit symbol share one demo position (one-way mode): it holds their net quantity at the lowest of their
leverages, so its quantity and P&L in total match the paper books but its margin and liquidation price match
neither; compare those per strategy on the paper book (m13-E8). On Bybit Demo, where
the quantity is the paper quantity, a catch-up then brings the account back to the paper position (catch_up),
only for a gap seen twice in a row and only as far as the account itself is short of it.

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
import itertools
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from sleeve_fund import markets, risk

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
CATCH_UP_SECONDS = 60  # a gap must be seen on two checks this far apart before the catch-up trades


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
            raise RuntimeError(f"the demo account {method}: {err.get('message', err)} ({err.get('data', '')})")
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

    def check(self) -> str:
        """One signed read at start, so a bad key shows in the logs at once, not on the first fill."""
        return f"key accepted, BTC-PERPETUAL position {self.position('BTC-PERPETUAL'):+g} USD"


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
            raise RuntimeError(f"the demo account {path}: {msg} ({code})")
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
            return None, 0.0, f"the demo account has no perpetual set up for {sleeve.instrument}"
        step, min_notional = contract
        qty = round(round(float(fill["qty"]) / step) * step, 8)
        if qty < step or qty * fill["price"] < min_notional:
            return symbol, 0.0, (f"{fill['qty']:.8g} ({fill['qty'] * fill['price']:,.2f} USDT) is under the smallest "
                                 f"order the demo account takes ({step:g}, {min_notional:g} USDT)")
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

    def margin_setup(self, symbol: str, leverage: float) -> str:
        """Put the account on isolated margin and the symbol at this leverage, as the paper book models a
        perpetual (isolated, the risk profile's leverage cap), so margin and liquidation price match it too.
        A unified account sets isolated margin account-wide; a classic one per symbol. Raises if Bybit refuses;
        returns what the account now says."""
        lev = f"{leverage:g}"
        try:
            self.call("POST", "/v5/account/set-margin-mode", {"setMarginMode": "ISOLATED_MARGIN"})
        except RuntimeError as unified:
            try:
                self.call("POST", "/v5/position/switch-isolated", {"category": "linear", "symbol": symbol,
                                                                    "tradeMode": 1, "buyLeverage": lev,
                                                                    "sellLeverage": lev})
            except RuntimeError as classic:
                if "(110026)" not in str(classic):  # 110026: already isolated
                    raise RuntimeError(f"isolated margin refused: {unified}; {classic}") from None
        try:
            self.call("POST", "/v5/position/set-leverage", {"category": "linear", "symbol": symbol,
                                                            "buyLeverage": lev, "sellLeverage": lev})
        except RuntimeError as e:
            if "(110043)" not in str(e):  # 110043: already at this leverage
                raise
        return self.margin_state(symbol)

    def margin_state(self, symbol: str) -> str:
        """The account's margin mode and the symbol's leverage, as Bybit reports them."""
        try:
            mode = str(self.call("GET", "/v5/account/info").get("marginMode") or "")
        except RuntimeError:
            mode = ""
        rows = self.call("GET", "/v5/position/list", {"category": "linear", "symbol": symbol}).get("list") or []
        row = rows[0] if rows else {}
        if not mode:
            mode = "ISOLATED_MARGIN" if str(row.get("tradeMode")) == "1" else "cross"
        mode = {"ISOLATED_MARGIN": "isolated", "REGULAR_MARGIN": "cross", "PORTFOLIO_MARGIN": "portfolio"}.get(mode, mode)
        return f"{mode} margin, {symbol} at {row.get('leverage') or '?'}x"

    def cancel_all(self, symbol: str) -> None:
        """Cancel any resting order on the symbol (the mirror sends market orders, so normally none)."""
        self.call("POST", "/v5/order/cancel-all", {"category": "linear", "symbol": symbol})

    def equity(self) -> str:
        rows = self.call("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}).get("list") or []
        return f"{float(rows[0].get('totalEquity') or 0):,.2f} USD equity" if rows else "equity unknown"

    def check(self) -> str:
        """One signed read at start, so a bad key or hedge mode shows in the logs at once, not on the first fill,
        with the account's equity and margin set-up."""
        rows = self.call("GET", "/v5/position/list", {"category": "linear", "symbol": "BTCUSDT"}).get("list") or []
        hedge = any(int(r.get("positionIdx") or 0) for r in rows)
        sides = {"Buy": 1.0, "Sell": -1.0}
        held = sum(sides.get(r.get("side"), 0.0) * float(r.get("size") or 0) for r in rows)
        mode = ("HEDGE MODE: orders will be rejected until BTCUSDT is switched to one-way mode" if hedge
                else "one-way mode")
        extra = []
        for read in (self.equity, lambda: self.margin_state("BTCUSDT")):
            try:
                extra.append(read())
            except Exception as e:  # noqa: BLE001 - only a report
                extra.append(f"couldn't read: {str(e)[:120]}")
        return f"key accepted, BTCUSDT position {held:+g}, {mode}, " + ", ".join(extra)


def target_for(sleeve) -> str:
    """The demo account a strategy's fills go to: Bybit Demo Trading for a strategy on Binance's USDT
    perpetuals (Binance's own demo is closed to the PM's account), else the Deribit testnet."""
    return "BYBIT" if (sleeve.venue or "").upper() == "BINANCE" else "DERIBIT"


def exact_copy(sleeve) -> str | None:
    """The demo account holding a mirrored strategy's own quantity, by its label, so its page can show the copy
    beside the paper position: Bybit Demo Trading. None for one not mirrored, or copied to the Deribit
    testnet, sized in dollars at each fill's price."""
    return "Bybit Demo Trading" if sleeve.params.get("demo_mirror") and target_for(sleeve) == "BYBIT" else None


def mirrored(store) -> list:
    """The strategies whose params ask for the demo mirror. A run archived by a reset is not one: it trades no
    more, and its risk profile mustn't set the leverage of a symbol it once shared (m13-E5)."""
    archived = store.archived()
    return [s for s in store.sleeves() if s.params.get("demo_mirror") and s.name not in archived]


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
            where = ("its demo account; demo money is not real" if venue is not None else
                     "its demo account, once that is set up on Setup, Accounts")  # no venue names in activity (QA U8)
            store.event(s.name, "info", "mirror_start",
                        f"Demo mirror on: each new fill is copied to {where}, and the paper book stays the record")
            continue
        for f in store.fills_after(s.name, mark):
            if venue is None:
                store.record_mirror(s.name, fill_id=f["id"], status="skipped",
                                    message="the demo account isn't set up")
                continue
            try:
                instrument, amount, why = venue.size(s, f)
            except Exception as e:  # noqa: BLE001 - the demo venue's contract list unreachable: noted, not retried
                instrument, amount, why = None, 0.0, f"couldn't size it: {str(e)[:200]}"
            sign = 1 if f["side"] == "BUY" else -1
            if why:
                store.record_mirror(s.name, fill_id=f["id"], status="skipped", instrument=instrument or "", message=why)
                store.event(s.name, "warning", "mirror_skipped",
                            f"Demo mirror didn't copy the {f['side'].lower()} of {f['qty']:.8g} to the demo account: "
                            f"{why}. The paper book is unaffected.")
                continue
            reset = "Reset strategy" in store.order_reason(f.get("order_id") or "")  # a PM reset's closing fill
            try:
                order_id, price = venue.market(f["side"], instrument, amount,
                                               f"{s.name[:20]}:reset:{f['id']}" if reset else f"{s.name}:{f['id']}")
            except Exception as e:  # noqa: BLE001 - noted, never retried, so it can't trade twice
                store.record_mirror(s.name, fill_id=f["id"], status="error", instrument=instrument,
                                    amount=sign * amount, message=str(e)[:500])
                store.event(s.name, "warning", "mirror_failed",
                            f"Demo mirror couldn't copy the {f['side'].lower()} of {f['qty']:.8g} at {f['price']:,.2f} "
                            f"to the demo account: {str(e)[:200]}. The paper book is unaffected.")
                continue
            store.record_mirror(s.name, fill_id=f["id"], status="filled", instrument=instrument, amount=sign * amount,
                                price=price, order_id=order_id, message="reset" if reset else "")
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
                        f"Demo mirror: the demo account holds {held:+,.6g} {venue.unit} of {instrument}, the mirrored "
                        f"strategies {expected:+,.6g}. The paper book is unaffected.")
        out[instrument] = gap
    return out


_CATCH_UP_FAILED: dict[str, str] = {}  # per strategy, the last catch-up failure warned about


def catch_up(store, targets: dict, seen: dict[str, float]) -> dict[str, float]:
    """Bring Bybit Demo back to the paper position after a copy that failed, was skipped, or came before the
    mirror first saw a strategy. Per mirrored strategy on Bybit, the gap is its paper position less what the
    mirror has put on for it. It trades only a gap seen at the same size on two checks in a row (so a fill
    being copied right now is never counted), and only as far as the account itself is short of the paper
    book in that direction (so an order that errored but went in is never sent again). Returns the gaps seen.
    Deribit is left to the drift warning: it is sized in dollars at each fill's price, so it has no exact
    target."""
    venue, out = targets.get("BYBIT"), {}
    if venue is None:
        return out
    by_symbol: dict[str, list] = {}
    for s in mirrored(store):
        symbol = s.instrument.replace("/", "").upper()
        if target_for(s) == "BYBIT" and symbol in BYBIT_CONTRACTS and store.mirror_watermark(s.name) is not None:
            by_symbol.setdefault(symbol, []).append(s)
    put_on = _put_on(store)
    for symbol, sleeves in by_symbol.items():
        step, min_notional = BYBIT_CONTRACTS[symbol]
        paper = {s.name: store.journal_book(s.name, s.starting_balance) for s in sleeves}
        gaps = {}
        for s in sleeves:
            gap = round(round((paper[s.name]["qty"] - put_on.get(s.name, 0.0)) / step) * step, 8)
            if abs(gap) >= step:
                gaps[s.name] = gap
        out.update(gaps)
        due = {n: g for n, g in gaps.items() if seen.get(n) == g}
        if not due:
            continue
        account_gap = sum(p["qty"] for p in paper.values()) - venue.position(symbol)
        for name, gap in due.items():
            sign = 1 if gap > 0 else -1
            room = max(0.0, sign * account_gap)  # how far the account is short of the paper book this way
            qty = round(int(min(abs(gap), room) / step + 1e-9) * step, 8)
            price = paper[name].get("entry_px") or 0.0
            if qty < step or (price and qty * price < min_notional):
                continue
            side = "BUY" if sign > 0 else "SELL"
            try:
                order_id, filled = venue.market(side, symbol, qty, f"{name}:catch-up")
            except Exception as e:  # noqa: BLE001 - the next check sees the same gap and tries again
                if _CATCH_UP_FAILED.get(name) != str(e):  # warned once per new reason, not every minute
                    _CATCH_UP_FAILED[name] = str(e)
                    store.event(name, "warning", "mirror_failed",
                                f"Demo mirror couldn't catch the demo account up to the paper position ({side.lower()} "
                                f"{qty:g} {symbol}): {str(e)[:200]}. It tries again each minute; the paper book is "
                                "unaffected.")
                continue
            _CATCH_UP_FAILED.pop(name, None)
            store.record_mirror(name, fill_id=store.mirror_watermark(name), status="filled", instrument=symbol,
                                amount=sign * qty, price=filled, order_id=order_id,
                                message="catch-up to the paper position")
            store.event(name, "info", "mirror_catch_up",
                        f"Demo mirror caught the demo account up to the paper position: {side.lower()} {qty:g} "
                        f"{symbol}, a copy it had missed")
            account_gap -= sign * qty
            out.pop(name, None)
    return out


def bybit_leverage(store) -> dict[str, tuple[float, list]]:
    """Per Bybit symbol, the leverage its mirrored strategies run at on paper (their risk profile's cap) and
    the strategies. They share one Bybit position, so when their caps differ the lowest is used."""
    out: dict[str, tuple[float, list]] = {}
    for s in mirrored(store):
        symbol = s.instrument.replace("/", "").upper()
        if target_for(s) != "BYBIT" or symbol not in BYBIT_CONTRACTS or not markets.is_perp(s.params):
            continue
        lev = risk.profile(s.risk_profile).max_leverage
        prev, names = out.get(symbol, (lev, []))
        out[symbol] = (min(prev, lev), names + [s])
    return out


_MARGIN_WARNED: dict[str, str] = {}  # per symbol, the last margin set-up failure warned about
_MARGIN_RESET: set[str] = set()  # symbols whose demo position was closed once to switch margin mode


def prepare_margin(store, targets: dict, done: dict[str, float]) -> dict[str, float]:
    """Before copying, set Bybit Demo to isolated margin at the paper leverage for each mirrored symbol
    (once per change). If Bybit refuses while the demo account holds the symbol, the demo position is closed
    once, the set-up tried again, and the catch-up reopens it at the new leverage. If it still fails, copies
    go on (same quantity, so the same profit and loss) and a warning says the leverage could not be set.
    Returns {symbol: leverage set}."""
    venue, out = targets.get("BYBIT"), dict(done)
    if venue is None:
        return out
    for symbol, (lev, sleeves) in bybit_leverage(store).items():
        if out.get(symbol) == lev:
            continue
        try:
            state = venue.margin_setup(symbol, lev)
        except Exception as first:  # noqa: BLE001
            held = venue.position(symbol)
            state, err = None, first
            if held and symbol not in _MARGIN_RESET:
                _MARGIN_RESET.add(symbol)
                side = "SELL" if held > 0 else "BUY"
                try:
                    order_id, price = venue.market(side, symbol, abs(held), _link(symbol, "margin-switch"))
                    put_on = _put_on(store)
                    for s in sleeves:
                        if put_on.get(s.name):
                            store.record_mirror(s.name, fill_id=store.mirror_watermark(s.name) or 0, status="filled",
                                                instrument=symbol, amount=-put_on[s.name], price=price,
                                                order_id=order_id, message="closed to switch the demo account to isolated "
                                                "margin; the catch-up reopens it at the paper leverage")
                    state = venue.margin_setup(symbol, lev)
                except Exception as e:  # noqa: BLE001
                    err = e
            if state is None:
                if _MARGIN_WARNED.get(symbol) != str(err):
                    _MARGIN_WARNED[symbol] = str(err)
                    for s in sleeves:
                        store.event(s.name, "warning", "mirror_margin",
                                    f"Demo mirror couldn't set the demo account to isolated margin at {lev:g}x for "
                                    f"{symbol}: {str(err)[:200]}. Copies go on at the same quantity, so profit and "
                                    "loss match, but margin and liquidation price on the demo account differ.")
                continue
        out[symbol] = lev
        _MARGIN_WARNED.pop(symbol, None)
        for s in sleeves:
            store.event(s.name, "info", "mirror_margin",
                        f"Demo mirror set the demo account to the paper book's terms for {symbol}: {state} "
                        f"(the {s.risk_profile} profile's {lev:g}x cap)")
    return out


def resync(store, targets: dict, sleeve: str | None, margin: dict[str, float], tag: str = "resync") -> str:
    """The PM's resync button: bring Bybit Demo in line with the paper book now, for the strategy's symbol (all
    strategies sharing it) or every mirrored symbol. Cancels resting orders, re-applies isolated margin at the
    paper leverage (closing the demo position first if Bybit won't switch while it is open), then trades the
    demo position straight to the paper total, with no wait for the catch-up. Every order and row it makes is
    labelled with tag ("resync", or "reset" after a PM reset). Its rows take the strategy's latest fill as their
    watermark: the paper position already counts every fill, so none is copied again afterwards. Returns paper
    against Bybit, before and after. Never touches the paper book."""
    venue = targets.get("BYBIT")
    groups = bybit_leverage(store)
    if sleeve is not None:
        groups = {sym: g for sym, g in groups.items() if any(s.name == sleeve for s in g[1])}
    if venue is None or not groups:
        return ("nothing to resync: only a perpetual strategy copied to the demo account can be resynced (a copy "
                "sized in dollars has no exact target)")
    lines = []
    for symbol, (lev, sleeves) in groups.items():
        step = BYBIT_CONTRACTS[symbol][0]
        paper = {s.name: store.journal_book(s.name, s.starting_balance)["qty"] for s in sleeves}
        target = round(round(sum(paper.values()) / step) * step, 8)
        before = f"{venue.position(symbol):+g}, {venue.margin_state(symbol)}"
        try:
            venue.cancel_all(symbol)
        except Exception:  # noqa: BLE001 - nothing resting is the usual case
            pass
        held, note = venue.position(symbol), ""
        try:
            venue.margin_setup(symbol, lev)
        except Exception as first:  # noqa: BLE001
            note = f" (margin: {str(first)[:120]})"
            if held:
                order_id, price = venue.market("SELL" if held > 0 else "BUY", symbol, abs(held), _link(symbol, tag))
                put_on = _put_on(store)
                for s in sleeves:
                    if put_on.get(s.name):
                        store.record_mirror(s.name, fill_id=store.mirror_watermark(s.name) or 0, status="filled",
                                            instrument=symbol, amount=-put_on[s.name], price=price,
                                            order_id=order_id, message=f"{tag}: closed to switch to isolated margin")
                held = 0.0
                try:
                    venue.margin_setup(symbol, lev)
                    note = ""
                except Exception as again:  # noqa: BLE001
                    note = f" (margin still refused: {str(again)[:120]})"
        delta = round(round((target - held) / step) * step, 8)
        order_id, price = "", None
        if abs(delta) >= step:
            order_id, price = venue.market("BUY" if delta > 0 else "SELL", symbol, abs(delta), _link(symbol, tag))
        put_on = _put_on(store)
        for s in sleeves:  # the mirror's record now says what each strategy holds on paper
            amount = round(paper[s.name] - put_on.get(s.name, 0.0), 8)
            last = store.last_fill_id(s.name)
            if abs(amount) > 1e-12 or last > (store.mirror_watermark(s.name) or 0):
                store.record_mirror(s.name, fill_id=last, status="filled", instrument=symbol, amount=amount,
                                    price=price, order_id=order_id, message=f"{tag} to the paper position")
        if not note:
            margin[symbol] = lev
        after = f"{venue.position(symbol):+g}, {venue.margin_state(symbol)}"
        lines.append(f"{symbol}: paper {target:+g} at {lev:g}x isolated; demo account before {before}; after {after}{note}")
    return "; ".join(lines)


def process_resyncs(store, targets: dict, margin: dict[str, float]) -> int:
    """Act on the PM's waiting resync requests, oldest first; returns how many were handled."""
    done = 0
    for req in store.pending_resyncs():
        try:
            tag = "reset" if req["reason"].startswith("Reset strategy") else "resync"
            result = resync(store, targets, req["sleeve"], margin, tag)
            level = "info"
        except Exception as e:  # noqa: BLE001 - reported on the request and the strategy, not retried
            result, level = f"failed: {str(e)[:300]}", "warning"
        store.finish_resync(req["id"], result)
        names = [req["sleeve"]] if req["sleeve"] else [s.name for g in bybit_leverage(store).values() for s in g[1]]
        for name in names:
            store.event(name, level, "mirror_resync", f"Demo copy resynced ({req['reason']}): {result}")
        done += 1
    return done


_LINKS = itertools.count()


def _link(symbol: str, tag: str) -> str:
    """A label for a mirror order that is no fill's copy: unique, since Bybit refuses a repeated order link id,
    and short enough for its 36 characters."""
    return f"{symbol}:{tag}:{int(time.time() * 1000):x}{next(_LINKS) % 100:02d}"


def _put_on(store) -> dict[str, float]:
    """Per strategy, the net quantity the mirror has put on (its filled rows)."""
    out: dict[str, float] = {}
    for r in store.mirror_rows(limit=100_000):
        if r["status"] == "filled":
            out[r["sleeve"]] = out.get(r["sleeve"], 0.0) + float(r["amount"] or 0.0)
    return out


def run(store, targets: dict, poll: float = POLL_SECONDS, stop=None) -> None:
    drift, checked, gaps, caught, margin = {}, 0.0, {}, 0.0, {}
    while stop is None or not stop():
        try:
            process_resyncs(store, targets, margin)
            margin = prepare_margin(store, targets, margin)
            mirror_once(store, targets)
            if time.monotonic() - caught > CATCH_UP_SECONDS:
                gaps, caught = catch_up(store, targets, gaps), time.monotonic()
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
    for t in targets.values():
        try:
            print(f"demo mirror: {t.label}: {t.check()}", flush=True)
        except Exception as e:  # noqa: BLE001 - reported, and the loop still runs: each fill notes its own error
            print(f"demo mirror: {t.label}: couldn't sign in: {str(e)[:300]}", flush=True)
    run(Store(), targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
