"""The demo mirror: Binance Demo Trading for strategies on Binance, the Deribit testnet for the rest; demo hosts
only, off unless switched on with a demo account's key and secret, copies each new fill of a mirrored strategy
once, and never touches the paper book."""

from pathlib import Path

import pytest

from sleeve_fund import mirror
from sleeve_fund.store import Store

ROOT = Path(__file__).resolve().parent.parent
ON = {"DEMO_MIRROR": "on", "DERIBIT_TESTNET_API_KEY": "k", "DERIBIT_TESTNET_API_SECRET": "s"}


def test_testnet_only():
    from nautilus_trader.adapters.deribit import DeribitEnvironment

    assert mirror.testnet_url() == "https://test.deribit.com"
    with pytest.raises(mirror.MirrorRefused):
        mirror.testnet_url(DeribitEnvironment.MAINNET)


@pytest.mark.parametrize("env, why", [
    ({}, "DEMO_MIRROR is off"),
    ({"DEMO_MIRROR": "off", "DERIBIT_TESTNET_API_KEY": "k", "DERIBIT_TESTNET_API_SECRET": "s"}, "off"),
    ({"DEMO_MIRROR": "on", "DERIBIT_TESTNET_API_KEY": "k"}, "DERIBIT_TESTNET_API_SECRET not set"),
    ({"DEMO_MIRROR": "on", "BINANCE_DEMO_API_SECRET": "s"}, "BINANCE_DEMO_API_KEY not set"),
    ({"DEMO_MIRROR": "on"}, "not set"),
])
def test_off_quietly_without_the_flag_and_both_secrets(env, why):
    creds, reason = mirror.settings(env)
    assert creds == {} and why in reason


def test_on_with_the_flag_and_both_secrets_and_never_prints_them():
    creds, _ = mirror.settings(ON)
    assert list(creds) == ["DERIBIT"] and (creds["DERIBIT"].key, creds["DERIBIT"].secret) == ("k", "s")
    assert "k" not in repr(creds["DERIBIT"]).replace("key", "")
    both, _ = mirror.settings({**ON, "BINANCE_DEMO_API_KEY": "bk", "BINANCE_DEMO_API_SECRET": "bs"})
    assert sorted(both) == ["BINANCE", "DERIBIT"] and both["BINANCE"].key == "bk"


@pytest.mark.parametrize("var", ["DERIBIT_API_KEY", "KRAKEN_SPOT_API_KEY", "DERIBIT_MAINNET_API_SECRET",
                                 "KRAKEN_API_KEY__MAIN", "BINANCE_API_KEY", "BINANCE_API_SECRET",
                                 "BINANCE_API_KEY__MAIN"])
def test_refuses_to_start_beside_any_other_credential(var):
    with pytest.raises(mirror.MirrorRefused):
        mirror.settings({**ON, var: "x"})


def test_private_calls_go_to_the_testnet_with_a_bearer_token():
    calls = []

    def get(url, headers):
        calls.append((url, headers))
        if "public/auth" in url:
            return {"result": {"access_token": "tok", "expires_in": 900}}
        return {"result": {"order": {"order_id": "1", "average_price": 60_000.0}}}

    t = mirror.Testnet(mirror.Settings("k", "s"), get=get)
    t.market("SELL", "BTC-PERPETUAL", 3000, "pp:7")
    (auth_url, _), (url, headers) = calls
    assert auth_url.startswith("https://test.deribit.com/api/v2/public/auth?")
    assert url.startswith("https://test.deribit.com/api/v2/private/sell?") and "amount=3000" in url
    assert "type=market" in url and "access_token" not in url and headers == {"Authorization": "Bearer tok"}


class FakeVenue:
    """The Deribit testnet's sizing, without the network."""

    label, host, unit = "the Deribit testnet", mirror.TESTNET_HOST, "USD"
    size, owns = staticmethod(mirror.Testnet.size), staticmethod(mirror.Testnet.owns)

    def __init__(self, fail=False, held=0.0):
        self.orders, self.fail, self.held = [], fail, held

    def market(self, side, instrument, amount, label):
        if self.fail:
            raise RuntimeError("not enough funds")
        self.orders.append((side, instrument, amount, label))
        return f"d{len(self.orders)}", 60_000.0

    def position(self, instrument):
        return self.held


def _store():
    store = Store.in_memory()
    for name, params in (("pp-ls", {"market": "perp", "allow_short": True, "demo_mirror": True}),
                         ("pp", {})):
        store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, params=params)
    return store


def _fill(store, sleeve, side, qty, n):
    store.record_fill(sleeve, side=side, qty=qty, price=60_000.0, fee=1.5, order_id=f"o{n}", trade_id=f"t{n}")


def test_copies_each_new_fill_once_from_when_it_starts():
    store, venue = _store(), FakeVenue()
    _fill(store, "pp-ls", "BUY", 0.05, 1)  # before the mirror: not copied
    assert mirror.mirror_once(store, {"DERIBIT": venue}) == 0
    assert [e["kind"] for e in store.events("pp-ls")] == ["mirror_start"]
    _fill(store, "pp-ls", "SELL", 0.05, 2)
    _fill(store, "pp-ls", "SELL", 0.0501, 3)
    _fill(store, "pp", "BUY", 0.05, 4)  # not mirrored
    assert mirror.mirror_once(store, {"DERIBIT": venue}) == 2
    assert [o[:3] for o in venue.orders] == [("SELL", "BTC-PERPETUAL", 3000.0), ("SELL", "BTC-PERPETUAL", 3010.0)]
    assert venue.orders[0][3].startswith("pp-ls:")
    assert mirror.mirror_once(store, {"DERIBIT": venue}) == 0  # nothing new
    assert store.mirror_positions() == {"BTC-PERPETUAL": -6010.0}
    assert store.mirror_watermark("pp") is None
    # The paper book is untouched.
    assert store.journal_book("pp-ls", 10_000)["qty"] == pytest.approx(-0.0501)


def test_a_failed_copy_is_noted_and_not_retried():
    store, venue = _store(), FakeVenue(fail=True)
    mirror.mirror_once(store, {"DERIBIT": venue})
    _fill(store, "pp-ls", "BUY", 0.05, 1)
    assert mirror.mirror_once(store, {"DERIBIT": venue}) == 0
    venue.fail = False
    assert mirror.mirror_once(store, {"DERIBIT": venue}) == 0 and venue.orders == []
    (row, _start) = store.mirror_rows("pp-ls")
    assert row["status"] == "error" and "not enough funds" in row["message"]
    assert store.events("pp-ls")[0]["kind"] == "mirror_failed" and store.events("pp-ls")[0]["level"] == "warning"


def test_small_fills_and_unlisted_instruments_are_skipped():
    store, venue = _store(), FakeVenue()
    store.create_sleeve(name="sui-ls", strategy="ping_pong", instrument="SUI/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=1_000, params={"demo_mirror": True})
    mirror.mirror_once(store, {"DERIBIT": venue})
    _fill(store, "pp-ls", "BUY", 0.00001, 1)  # 0.60 USD: under one 10 USD contract
    _fill(store, "sui-ls", "BUY", 10, 2)
    mirror.mirror_once(store, {"DERIBIT": venue})
    assert venue.orders == []
    assert {r["status"] for r in store.mirror_rows() if r["status"] != "start"} == {"skipped"}


def test_drift_warns_once_per_change():
    store, venue = _store(), FakeVenue(held=-2000.0)
    mirror.mirror_once(store, {"DERIBIT": venue})
    _fill(store, "pp-ls", "SELL", 0.05, 1)
    mirror.mirror_once(store, {"DERIBIT": venue})
    gaps = mirror.check_drift(store, {"DERIBIT": venue}, {})
    assert gaps == {"BTC-PERPETUAL": 1000.0}
    mirror.check_drift(store, {"DERIBIT": venue}, gaps)
    assert [e["kind"] for e in store.events(None)].count("mirror_drift") == 1


def test_only_the_mirror_container_gets_the_testnet_key():
    import yaml

    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    for var in ("DERIBIT", "BINANCE_DEMO"):
        holders = {name for name, svc in services.items()
                   if any(var in str(k) for k in (svc.get("environment") or {}))}
        assert holders == {"mirror"}, var
    # The paper processes run under the supervisor, which would refuse to start a sleeve with the key.
    assert "kraken.env" in str(services["supervisor"].get("env_file"))
    assert services["mirror"]["command"] == "python -m sleeve_fund.mirror run"


def test_the_long_short_test_strategies_are_mirrored():
    from sleeve_fund.paper.config import load_sleeve

    for name in ("ping_pong_ls_test", "rsi_bands_ls_test", "ping_pong_ls_binance", "rsi_bands_ls_binance"):
        assert load_sleeve(ROOT / "configs" / "sleeves" / f"{name}.toml").params["demo_mirror"] is True


INFO = {"symbols": [{"symbol": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING", "baseAsset": "BTC",
                     "quoteAsset": "USDT", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                                                       {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                                                       {"filterType": "MIN_NOTIONAL", "notional": "100"}]}]}


def test_binance_demo_only_and_signed_as_binance_signs():
    """Only demo-fapi.binance.com, never the live host; private calls carry the key in a header and an
    HMAC-SHA256 signature of the query, never the secret."""
    import hashlib
    import hmac
    from urllib.parse import urlparse

    for url in ("https://fapi.binance.com", "https://api.binance.com", "http://demo-fapi.binance.com",
                "https://demo-fapi.binance.com.evil.example"):
        with pytest.raises(mirror.MirrorRefused):
            mirror.BinanceDemo(mirror.Settings("k", "s"), url=url)
    calls = []

    def http(method, url, headers):
        calls.append((method, url, headers))
        return {"orderId": 42, "avgPrice": "60010.5", "status": "FILLED"}

    b = mirror.BinanceDemo(mirror.Settings("k", "s3cret"), http=http, clock=lambda: 1_700_000_000)
    assert b.market("SELL", "BTCUSDT", 0.05, "ping-pong-ls-binance:1234567890123456789") == ("42", 60010.5)
    method, url, headers = calls[0]
    u = urlparse(url)
    assert (method, u.scheme, u.hostname, u.path) == ("POST", "https", "demo-fapi.binance.com", "/fapi/v1/order")
    query, sig = u.query.rsplit("&signature=", 1)
    assert sig == hmac.new(b"s3cret", query.encode(), hashlib.sha256).hexdigest()
    assert "side=SELL" in query and "type=MARKET" in query and "quantity=0.05" in query and "timestamp=1700000000000" in query
    assert "newClientOrderId=ping-pong-ls-binance%3A123456789012345" in query  # cut to Binance's 36 characters
    assert headers == {"X-MBX-APIKEY": "k"} and "s3cret" not in url
    fail = mirror.BinanceDemo(mirror.Settings("k", "s"), http=lambda *a: {"code": -2019, "msg": "Margin is insufficient."})
    with pytest.raises(RuntimeError, match="Margin is insufficient"):
        fail.market("BUY", "BTCUSDT", 0.05, "x:1")


def test_strategies_on_binance_go_to_binance_demo_with_their_own_quantity():
    class Demo(mirror.BinanceDemo):
        def __init__(self):
            super().__init__(mirror.Settings("k", "s"), http=self.http)
            self.orders, self.held = [], 0.0

        def http(self, method, url, headers):
            if "exchangeInfo" in url:
                return INFO
            if "positionRisk" in url:
                return [{"symbol": "BTCUSDT", "positionAmt": str(self.held)}]
            self.orders.append(url)
            return {"orderId": len(self.orders), "avgPrice": "60000"}

    store, demo, testnet = _store(), Demo(), FakeVenue()
    store.create_sleeve(name="bn-ls", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "allow_short": True, "demo_mirror": True},
                        venue="binance")
    targets = {"BINANCE": demo, "DERIBIT": testnet}
    mirror.mirror_once(store, targets)
    assert "Binance Demo Trading (demo-fapi.binance.com)" in store.events("bn-ls")[0]["message"]
    _fill(store, "bn-ls", "SELL", 0.05, 1)
    _fill(store, "bn-ls", "BUY", 0.001, 2)  # 60 USDT: under Binance's 100 USDT smallest order
    _fill(store, "pp-ls", "BUY", 0.05, 3)
    assert mirror.mirror_once(store, targets) == 2
    assert len(demo.orders) == 1 and "symbol=BTCUSDT" in demo.orders[0] and "quantity=0.05" in demo.orders[0]
    assert [o[:3] for o in testnet.orders] == [("BUY", "BTC-PERPETUAL", 3000.0)]
    skipped = [r for r in store.mirror_rows("bn-ls") if r["status"] == "skipped"]
    assert len(skipped) == 1 and "under the smallest order" in skipped[0]["message"]
    assert store.mirror_positions() == {"BTCUSDT": -0.05, "BTC-PERPETUAL": 3000.0}
    demo.held, testnet.held = -0.05, 3000.0
    assert mirror.check_drift(store, targets, {}) == {"BTCUSDT": 0.0, "BTC-PERPETUAL": 0.0}
    assert "mirror_drift" not in [e["kind"] for e in store.events(None)]


def test_a_strategy_whose_demo_account_is_not_set_up_is_skipped_not_sent_elsewhere():
    store, testnet = _store(), FakeVenue()
    store.create_sleeve(name="bn-ls", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "demo_mirror": True}, venue="binance")
    mirror.mirror_once(store, {"DERIBIT": testnet})
    assert "BINANCE_DEMO_API_KEY and BINANCE_DEMO_API_SECRET" in store.events("bn-ls")[0]["message"]
    _fill(store, "bn-ls", "BUY", 0.05, 1)
    mirror.mirror_once(store, {"DERIBIT": testnet})
    assert testnet.orders == [] and store.mirror_rows("bn-ls")[0]["message"] == "no demo account set up for Binance"
