"""UI v2, items 8 and 9: one pick-list for instruments (every venue's listing, grouped Perpetuals then Spot,
with a history badge, a typed one checked against the venue), reasons as a pick-list that remembers the PM's
own, candle length as chips, and study windows sized to the stored history."""

import json
import re

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.dashboard import development as dev
from sleeve_fund.dashboard import reasons
from sleeve_fund.store import Store
from sleeve_fund.venues import KRAKEN

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}
LISTED = {"kraken": ["BTC/USD", "ETH/USD", "ADA/USD", "PEPE/USD"], "binance": ["BTC/USDT", "ETH/USDT", "PEPE/USDT"]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund import history
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.dashboard import charts

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    monkeypatch.setattr(charts, "instruments", lambda get_json=None, venue=None: LISTED[venue])
    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store, tmp_path


def _coverage(root, venue, pair, first, last):
    d = root / "hist" / venue / pair.replace("/", "-")
    d.mkdir(parents=True)
    (d / "coverage.json").write_text(json.dumps({"first": first, "last": last, "cursor": "x"}))


def _new(c, **over):
    form = {"name": "btc-test", "strategy": "trend_filter", "instrument": "BTC/USD",
            "bar_spec": "1-HOUR-LAST-INTERNAL", "starting_balance": "5000", "risk_profile": "balanced",
            "warmup_bars": "0", "p_trend_filter__fast": "10", "p_trend_filter__slow": "30",
            "reason_pick": "New test", **over}
    return c.post("/sleeves/new", data=form, auth=AUTH, headers=SAME, follow_redirects=False)


# --- instruments ------------------------------------------------------------------------------------------------

def test_every_venues_instruments_come_grouped_with_a_history_badge_and_no_venue_names(client):
    c, _, tmp = client
    _coverage(tmp, "BINANCE", "BTC/USDT", "2021-10-01T00:00:00+00:00", "2099-01-01T00:00:00+00:00")
    opts = c.get("/api/instruments?options=1", auth=AUTH).json()["options"]
    groups = [o["group"] for o in opts]
    assert groups == sorted(groups, key=lambda g: g != "Perpetuals")  # perpetuals first, then spot
    assert {(o["value"], o["venue"]) for o in opts} >= {("PEPE/USDT", "binance"), ("PEPE/USD", "kraken")}
    btc = next(o for o in opts if o["value"] == "BTC/USDT")
    assert btc["label"] == "BTC/USDT perpetual" and btc["stored"] and btc["badge"] == "1 gap · backtests wait until filled"
    eth = next(o for o in opts if o["value"] == "ETH/USD")
    assert eth["label"] == "ETH/USD spot" and eth["badge"] == "not stored yet"
    assert not any(re.search("binance|kraken", o["label"] + o["badge"], re.IGNORECASE) for o in opts)
    # The chart's own dropdown keeps the plain list.
    assert "instruments" in c.get("/api/instruments", auth=AUTH).json()


def test_new_strategy_backtest_and_development_use_the_one_pick_list(client):
    c, _, _ = client
    for path in ("/sleeves/new", "/backtest", "/research"):
        page = c.get(path, auth=AUTH).text
        assert 'role="combobox"' in page and "data-pl-venue" in page and "data-pl-seed" in page, path
        assert '<datalist' not in page and 'name="venue" value="' in page, path
    page = c.get("/sleeves/new", auth=AUTH).text
    assert re.search(r'role="radiogroup" aria-label="Candle length" id="bar_spec">.*?<span>15m</span>', page, re.DOTALL)


def test_a_typed_instrument_the_venue_doesnt_list_is_refused(client, monkeypatch):
    c, store, _ = client

    def check(pair):
        if pair != "PEPE/USD":
            raise ValueError(f"Kraken does not list {pair}")

    monkeypatch.setattr(KRAKEN, "check_listed", check)
    r = _new(c, name="zz-test", instrument="ZZZQ/USD")
    assert "the+venue+doesn%27t+list+ZZZQ%2FUSD" in r.headers["location"] and not store.sleeves()
    assert _new(c, name="pepe-test", instrument="PEPE/USD").headers["location"] == "/sleeves/pepe-test"
    assert _new(c, name="btc-test").headers["location"] == "/sleeves/btc-test"  # a usual one asks nobody

    def down(pair):
        raise ConnectionError("403 Forbidden")

    monkeypatch.setattr(KRAKEN, "check_listed", down)
    r = _new(c, name="down-test", instrument="QQQZ/USD")
    assert "couldn%27t+reach+the+venue" in r.headers["location"]


# --- reasons ----------------------------------------------------------------------------------------------------

def test_a_reason_of_your_own_comes_back_under_recently_and_can_be_picked_again(client):
    c, store, _ = client
    assert _new(c, name="btc-test", reason_pick="Other", reason_note="Exit fix check after deploy").status_code == 303
    assert store.decisions(action="create")[0]["reason"] == "Other: Exit fix check after deploy"
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert page.count("You used recently") >= 3  # every dialog on the page offers it
    assert 'value="Other: Exit fix check after deploy"' in page and "+ Write your own reason" in page
    assert reasons.compose("pause", "Other: Exit fix check after deploy") == "Other: Exit fix check after deploy"
    assert reasons.compose("pause", "Other: Exit fix check", "round two") == "Other: Exit fix check: round two"
    with pytest.raises(ValueError, match="at least 10"):
        reasons.compose("pause", "Other: short")
    with pytest.raises(ValueError, match="isn't one of the reasons"):
        reasons.compose("stop", "Market event ahead")  # still each action's own list
    r = c.post("/sleeves/btc-test/command", data={"command": "pause", "reason_pick": "Other: Exit fix check after deploy"},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and store.decisions(action="pause")[0]["reason"] == "Other: Exit fix check after deploy"


def test_recent_reasons_are_newest_first_and_each_once():
    log = [{"reason": "Other: Second thought here", "action": "pause"}, {"reason": "Market event ahead", "action": "pause"},
           {"reason": "Other: first reason typed", "action": "reset"}, {"reason": "Other: second THOUGHT here", "action": "stop"}]
    assert [r["text"] for r in reasons.recent(log)] == ["Second thought here", "first reason typed"]


# --- Development ------------------------------------------------------------------------------------------------

def test_study_windows_fit_the_stored_history_and_say_so():
    plan = {"train_days": 730, "test_days": 365, "holdout_days": 365}
    windows, said = dev.fit_windows(plan, 851)  # 2 years 4 months
    assert sum(windows.values()) <= 851 and abs(windows["train_days"] - 2 * windows["test_days"]) <= 1
    assert said == ("You have 2 years 4 months of history, so: learn on 1 year 2 months, test on the next 7 months, "
                    "newest 7 months sealed.")
    assert dev.fit_windows(plan, 2000) == ({}, "You have 5 years 6 months of history; these windows fit inside it.")
    assert dev.fit_windows({"train_days": 730, "test_days": 365}, 851)[0]["holdout_days"] == 0
    assert dev.fit_windows(plan, None) == ({}, "")


def test_the_study_opens_with_windows_sized_to_what_is_stored(client):
    c, _, tmp = client
    _coverage(tmp, "BINANCE", "BTC/USDT", "2024-06-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00")
    page = c.get("/research?strategy=rsi_cross", auth=AUTH).text  # its plan: 2 years, 1 and 1 sealed
    assert re.search(r'id="fit-text">You have 2 years 4 months of history, so: learn on 1 year 2 months', page)
    assert 'id="train_days" name="train_days" type="number" min="30" step="1" value="426"' in page
    # A form sent back keeps its own windows.
    back = c.get("/research?strategy=rsi_cross&minutes=15&train_days=200&test_days=60&holdout_days=0", auth=AUTH).text
    assert 'name="train_days" type="number" min="30" step="1" value="200"' in back


def test_development_is_a_searchable_one_line_list_and_data_coverage_is_read_only(client):
    c, _, tmp = client
    _coverage(tmp, "KRAKEN", "ETH/USD", "2016-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00")
    page = c.get("/research", auth=AUTH).text
    assert 'id="plan-search"' in page and 'class="plan-row"' in page and "plan-card" not in page
    assert ">Data coverage<" in page and "/research/history" not in page.split('id="tab-history"')[1]
    assert "venue-seg" not in page  # the venue is part of the instrument pick
