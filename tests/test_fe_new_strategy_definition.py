"""PAPER-FROM-DEFINITION (HoE 16 Nov): the PM-only New Strategy form picks a committed rule-builder definition, by its
file, name and content hash, for one asset or a basket (Store.create_legs). The definition is copied in as a study
copies it, the hash the form showed is checked again on save, and the Strategy is created stopped: the PM starts it
from its page, where every Start check runs. The form also takes an optional leverage cap (B-35 (a))."""

from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from test_halt_ref import T0, store  # noqa: F401

from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import (
    definition_hash,
    load_definition_file,
    to_params,
)

AUTH = ("pm", "pw")
SAME = {"origin": "http://testserver"}
V5 = PREREGISTERED["V5_LO_ADX"]["definition"]
V5_KEY = f"variants/{V5}"
V5_DEFN = load_definition_file(definition_path(V5))
V5_HASH = definition_hash(V5_DEFN)
FORM = {"name": "v5-paper", "strategy": "rules", "definition": f"{V5_KEY}@{V5_HASH}", "instrument": "BTC/USDT",
        "venue": "binance", "market": "perp", "bar_spec": "1-HOUR-LAST-INTERNAL", "risk_profile": "aggressive",
        "warmup_bars": "0", "reason": "the pre-registered paper forward test"}


@pytest.fixture
def c(store, tmp_path, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.dashboard import book

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    monkeypatch.setattr(book, "is_test_account", lambda store, sleeve: False)
    # The definition's daily candles want stored history this test store has none of (its own check, kept and tested
    # by FORM-ASSET-LIST); here it is taken as loaded.
    monkeypatch.setattr(app_mod, "_check_slower_history", lambda *a: None)
    at = T0 - timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="test", rebase=True, now=at), at)
    return TestClient(app_mod.create_app(store))


def _post(c, **over):
    return c.post("/sleeves/new", data={**FORM, **over}, auth=AUTH, headers=SAME, follow_redirects=False)


def _error(r) -> str:
    assert r.status_code == 303 and r.headers["location"].startswith("/sleeves/new?"), r.headers.get("location")
    return parse_qs(urlparse(r.headers["location"]).query)["error"][0]


def test_the_form_offers_each_committed_definition_by_name_and_hash(c):
    from sleeve_fund.dashboard.app import _committed_definitions

    page = c.get("/sleeves/new", auth=AUTH).text
    assert '<optgroup label="Definitions">' in page and 'name="definition"' in page
    assert f'value="{V5_KEY}@{V5_HASH}"' in page and V5_HASH in page
    defs = _committed_definitions()
    assert V5_KEY in defs and "qr01b_trend_multispeed_long" in defs  # variants/ and the top level
    assert all(d["path"].suffix == ".toml" for d in defs.values())
    assert 'name="max_leverage"' in page


def test_the_backtest_page_offers_no_definitions(c):
    assert "Definitions" not in c.get("/backtest", auth=AUTH).text


def test_a_definition_basket_is_created_stopped_with_the_definition_copied_in(c, store):  # noqa: F811
    r = _post(c, assets="ETH/USDT", own_capital="1000", max_leverage="1")
    assert r.status_code == 303 and r.headers["location"] == "/sleeves/v5-paper", r.headers.get("location")
    assert store.legs("v5-paper") == ["v5-paper", "v5-paper~ETH"]
    rows = [store.sleeve(n) for n in store.legs("v5-paper")]
    want = to_params(V5_DEFN, "BINANCE")
    for s in rows:
        assert s.strategy == "rules" and s.risk_profile == "aggressive" and s.starting_balance == 0
        assert s.desired_state == "stopped"  # the PM starts it himself
        assert {k: s.params[k] for k in want} == want
        assert s.params["max_leverage"] == 1 and s.params["own_capital"] == 1000 and s.params["market"] == "perp"
        assert "allow_short" not in s.params  # long only
    assert [d["action"] for d in store.decisions("v5-paper")].count("create") == 1


def test_a_definition_is_refused_when_unknown_or_changed_and_nothing_is_saved(c, store):  # noqa: F811
    assert "committed definitions" in _error(_post(c, definition=f"../../etc/passwd@{V5_HASH}"))
    assert "committed definitions" in _error(_post(c, definition=""))
    assert "has changed since the form showed it" in _error(_post(c, definition=f"{V5_KEY}@0000000000000000"))
    assert "unknown setting" in _error(_post(c, p_rules__fast="5"))
    assert "its own settings" in _error(_post(c, p_rules__definition_version="1"))  # a known one, typed in
    assert not any(s.name.startswith("v5-paper") for s in store.sleeves())


@pytest.mark.parametrize("bad", ["0.5", "1.234", "x"])
def test_a_bad_leverage_cap_is_refused(c, store, bad):  # noqa: F811
    assert "leverage" in _error(_post(c, max_leverage=bad))
    assert not any(s.name.startswith("v5-paper") for s in store.sleeves())


def test_a_definition_that_places_no_stop_is_held_to_1x_on_a_perp_as_any_model_is(c, store):  # noqa: F811
    assert "places no stop" in _error(_post(c, max_leverage="2"))  # check_perp_stop, unchanged
    r = _post(c)  # no cap given: held at 1x, as the form holds every stopless model on a perp
    assert r.headers["location"] == "/sleeves/v5-paper", r.headers.get("location")
    assert store.sleeve("v5-paper").params["max_leverage"] == 1


def test_another_model_ignores_the_definition_field_and_takes_the_leverage_cap(c, store):  # noqa: F811
    r = _post(c, strategy="ping_pong", instrument="BTC/USD", venue="", market="spot", max_leverage="")
    assert r.headers["location"] == "/sleeves/v5-paper", r.headers.get("location")
    s = store.sleeve("v5-paper")
    assert "definition" not in s.params and "max_leverage" not in s.params
    assert s.desired_state != "stopped"
