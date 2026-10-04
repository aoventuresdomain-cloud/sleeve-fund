"""The Deribit testnet demo mirror: testnet only, off unless switched on with both testnet secrets, copies
each new fill of a mirrored strategy once, and never touches the paper book."""

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
    ({"DEMO_MIRROR": "on"}, "not set"),
])
def test_off_quietly_without_the_flag_and_both_secrets(env, why):
    creds, reason = mirror.settings(env)
    assert creds is None and why in reason


def test_on_with_the_flag_and_both_secrets_and_never_prints_them():
    creds, _ = mirror.settings(ON)
    assert (creds.key, creds.secret) == ("k", "s")
    assert "k" not in repr(creds).replace("key", "")


@pytest.mark.parametrize("var", ["DERIBIT_API_KEY", "KRAKEN_SPOT_API_KEY", "DERIBIT_MAINNET_API_SECRET",
                                 "KRAKEN_API_KEY__MAIN"])
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
    def __init__(self, fail=False, held=0.0):
        self.orders, self.fail, self.held = [], fail, held

    def market(self, side, instrument, amount, label):
        if self.fail:
            raise RuntimeError("not enough funds")
        self.orders.append((side, instrument, amount, label))
        return {"order": {"order_id": f"d{len(self.orders)}", "average_price": 60_000.0}}

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
    assert mirror.mirror_once(store, venue) == 0
    assert [e["kind"] for e in store.events("pp-ls")] == ["mirror_start"]
    _fill(store, "pp-ls", "SELL", 0.05, 2)
    _fill(store, "pp-ls", "SELL", 0.0501, 3)
    _fill(store, "pp", "BUY", 0.05, 4)  # not mirrored
    assert mirror.mirror_once(store, venue) == 2
    assert [o[:3] for o in venue.orders] == [("SELL", "BTC-PERPETUAL", 3000.0), ("SELL", "BTC-PERPETUAL", 3010.0)]
    assert venue.orders[0][3].startswith("pp-ls:")
    assert mirror.mirror_once(store, venue) == 0  # nothing new
    assert store.mirror_positions() == {"BTC-PERPETUAL": -6010.0}
    assert store.mirror_watermark("pp") is None
    # The paper book is untouched.
    assert store.journal_book("pp-ls", 10_000)["qty"] == pytest.approx(-0.0501)


def test_a_failed_copy_is_noted_and_not_retried():
    store, venue = _store(), FakeVenue(fail=True)
    mirror.mirror_once(store, venue)
    _fill(store, "pp-ls", "BUY", 0.05, 1)
    assert mirror.mirror_once(store, venue) == 0
    venue.fail = False
    assert mirror.mirror_once(store, venue) == 0 and venue.orders == []
    (row, _start) = store.mirror_rows("pp-ls")
    assert row["status"] == "error" and "not enough funds" in row["message"]
    assert store.events("pp-ls")[0]["kind"] == "mirror_failed" and store.events("pp-ls")[0]["level"] == "warning"


def test_small_fills_and_unlisted_instruments_are_skipped():
    store, venue = _store(), FakeVenue()
    store.create_sleeve(name="sui-ls", strategy="ping_pong", instrument="SUI/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=1_000, params={"demo_mirror": True})
    mirror.mirror_once(store, venue)
    _fill(store, "pp-ls", "BUY", 0.00001, 1)  # 0.60 USD: under one 10 USD contract
    _fill(store, "sui-ls", "BUY", 10, 2)
    mirror.mirror_once(store, venue)
    assert venue.orders == []
    assert {r["status"] for r in store.mirror_rows() if r["status"] != "start"} == {"skipped"}


def test_drift_warns_once_per_change():
    store, venue = _store(), FakeVenue(held=-2000.0)
    mirror.mirror_once(store, venue)
    _fill(store, "pp-ls", "SELL", 0.05, 1)
    mirror.mirror_once(store, venue)
    gaps = mirror.check_drift(store, venue, {})
    assert gaps == {"BTC-PERPETUAL": 1000.0}
    mirror.check_drift(store, venue, gaps)
    assert [e["kind"] for e in store.events(None)].count("mirror_drift") == 1


def test_only_the_mirror_container_gets_the_testnet_key():
    import yaml

    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    holders = {name for name, svc in services.items()
               if any("DERIBIT" in str(k) for k in (svc.get("environment") or {}))}
    assert holders == {"mirror"}
    # The paper processes run under the supervisor, which would refuse to start a sleeve with the key.
    assert "kraken.env" in str(services["supervisor"].get("env_file"))
    assert services["mirror"]["command"] == "python -m sleeve_fund.mirror run"


def test_the_long_short_test_strategies_are_mirrored():
    from sleeve_fund.paper.config import load_sleeve

    for name in ("ping_pong_ls_test", "rsi_bands_ls_test"):
        assert load_sleeve(ROOT / "configs" / "sleeves" / f"{name}.toml").params["demo_mirror"] is True
