"""FORM-ASSET-LIST (PM 21 Oct 17:56 UK, HoE): the New Strategy form takes a list of assets and creates the Strategy
through Store.create_legs: fund-held, 0 own capital, one quote currency, a leg row for each asset after the first. Every
check the form makes of its one instrument is made of each asset, and any miss saves nothing. Start refuses, naming it,
an asset whose margin tiers aren't kept, and starts nothing. A one-asset create is as before. Rows read back from the
journal, on Postgres too (TEST_DATABASE_URL)."""

import re
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from test_halt_ref import T0, store  # noqa: F401
from test_two_legs_node import built  # noqa: F401 (the fixture)

from sleeve_fund.venues import venue

AUTH = ("pm", "pw")
SAME = {"origin": "http://testserver"}
FORM = {"name": "duo", "strategy": "ping_pong", "instrument": "BTC/USD", "bar_spec": "1-MINUTE-LAST-INTERNAL",
        "risk_profile": "balanced", "warmup_bars": "0", "reason": "two assets, one idea"}


@pytest.fixture
def c(store, tmp_path, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.dashboard import book

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    monkeypatch.setattr(book, "is_test_account", lambda store, sleeve: False)
    at = T0 - timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="test", rebase=True, now=at), at)
    return TestClient(app_mod.create_app(store))


def _post(c, **over):
    return c.post("/sleeves/new", data={**FORM, **over}, auth=AUTH, headers=SAME, follow_redirects=False)


def _error(r) -> str:
    assert r.status_code == 303 and r.headers["location"].startswith("/sleeves/new?"), r.headers.get("location")
    return parse_qs(urlparse(r.headers["location"]).query)["error"][0]


def test_the_form_offers_other_assets_while_the_fund_holds_the_cash(c):
    assert 'name="assets"' in c.get("/sleeves/new", auth=AUTH).text


def test_a_two_asset_create_makes_the_parent_and_a_leg_with_no_capital_of_their_own(c, store):  # noqa: F811
    r = _post(c, assets="ETH/USD")
    assert r.status_code == 303 and r.headers["location"] == "/sleeves/duo", r.headers.get("location")
    assert store.legs("duo") == ["duo", "duo~ETH"]
    rows = [store.sleeve(n) for n in store.legs("duo")]
    assert [s.instrument for s in rows] == ["BTC/USD", "ETH/USD"]
    assert all(s.starting_balance == 0 and s.strategy == "ping_pong" for s in rows)
    assert rows[1].parent == "duo" and rows[0].params["assets"] == ["BTC/USD", "ETH/USD"]
    assert {store.account_of(s.name) for s in rows} == {"paper"}
    assert [d["action"] for d in store.decisions("duo")].count("create") == 1


def test_an_asset_that_fails_a_check_is_named_and_nothing_is_saved(c, store):  # noqa: F811
    assert "ETH/EUR" in _error(_post(c, assets="ETH/EUR"))  # another quote currency than the first asset's
    assert "ETH-USD" in _error(_post(c, assets="ETH-USD"))  # not BASE/QUOTE
    assert "BTC/USD" in _error(_post(c, assets="BTC/USD"))  # listed twice
    assert not any(s.name.startswith("duo") for s in store.sleeves())


def test_a_one_asset_create_is_as_before(c, store):  # noqa: F811
    r = _post(c)
    assert r.headers["location"] == "/sleeves/duo"
    assert store.legs("duo") == ["duo"] and "assets" not in store.sleeve("duo").params


PERP = {"instrument": "BTC/USDT", "venue": "binance", "assets": "SUI/USDT", "market": "perp"}  # SUI/USDT: no tiers kept (TEC-TIERS3-B: was ETH/USDT, whose tiers are kept since 21 Oct)


@pytest.mark.no_maintenance_tables  # the venue as it is: BTC/USDT's tiers kept, SUI/USDT's not
def test_start_refuses_a_perp_asset_with_no_kept_tiers_naming_it_and_starts_nothing(c, store):  # noqa: F811
    # made as if SUI/USDT's tiers were kept (the form now refuses it): Start still checks every asset at the click
    store.create_legs(name="duo", strategy="ping_pong", bar_spec="1-MINUTE-LAST-INTERNAL", assets=["BTC/USDT", "SUI/USDT"],
                      params={"market": "perp", "stop_loss": 0.02}, starting_balance=0, universe_date="2026-10-21",
                      venue="BINANCE", desired_state="stopped")
    r = c.post("/sleeves/duo/command", data={"command": "start", "reason": "go"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert "SUI/USDT" in parse_qs(urlparse(r.headers["location"]).query).get("command_error", [""])[0], r.headers["location"]
    assert all(store.sleeve(n).desired_state == "stopped" for n in store.legs("duo"))


@pytest.mark.no_maintenance_tables
def test_a_create_refuses_an_asset_with_no_kept_tiers(c, store):  # noqa: F811
    assert "SUI/USDT" in _error(_post(c, **PERP, stop_loss_pct="2"))
    assert not any(s.name.startswith("duo") for s in store.sleeves())


def test_the_page_shows_each_asset_and_the_total(c, store):  # noqa: F811
    _post(c, assets="ETH/USD")
    html = c.get("/sleeves/duo", auth=AUTH).text
    rows = set(re.findall(r'<tr data-asset="(\w+)"', html))
    assert rows == {"BTC", "ETH", "_total"}, rows  # F-B5-PAGE's rows: each asset's P&L, and the Strategy's total


def test_a_form_created_strategy_starts_both_legs_on_the_paper_node(c, store, built):  # noqa: F811
    """Done-when 1: the rows the form makes build one node running the model on each asset (paper/node.py legs)."""
    from test_two_legs_node import HUB

    from sleeve_fund.paper import node as node_mod
    from sleeve_fund.paper.config import from_store

    _post(c, assets="ETH/USD")
    node = node_mod.build_legs_node([from_store(store.sleeve(n)) for n in store.legs("duo")], log_level="ERROR",
                                    asset_fetch=dict, hub=HUB)
    assert [str(s._cfg.instrument_id).split(".")[0] for s in built["strategies"]] == ["BTC/USD", "ETH/USD"]
    node.dispose()


def test_every_leg_is_the_parents_strategy_bar_for_its_asset(c, store):  # noqa: F811
    """Advisor: a leg that drifts from its parent is a different Strategy; only the name and the instrument differ."""
    _post(c, assets="ETH/USD, SOL/USD", stop_loss_pct="2")
    rows = [store.sleeve(n) for n in store.legs("duo")]
    keep = ("strategy", "bar_spec", "params", "risk_profile", "warmup_bars", "venue", "starting_balance")
    first = {k: getattr(rows[0], k) for k in keep}
    assert all({k: getattr(r, k) for k in keep} == first for r in rows[1:]), [r.params for r in rows]
    assert [(r.name, r.instrument) for r in rows] == [("duo", "BTC/USD"), ("duo~ETH", "ETH/USD"), ("duo~SOL", "SOL/USD")]


@pytest.mark.no_maintenance_tables
def test_a_refusal_names_every_asset_with_no_kept_tiers(c, store, monkeypatch):  # noqa: F811
    # TEC-TIERS3-B: SUI/USDT is the only listed pair left with no tiers kept, so XRP/USDT's are set aside for this cell
    monkeypatch.delitem(venue("BINANCE").maintenance_tiers, "XRP/USDT")
    why = _error(_post(c, **{**PERP, "assets": "SUI/USDT XRP/USDT"}, stop_loss_pct="2"))
    assert "SUI/USDT" in why and "XRP/USDT" in why and "BTC/USDT" not in why, why


def test_the_page_lists_the_assets_in_the_order_given_the_tie_break(c, store):  # noqa: F811
    _post(c, assets="SOL/USD ETH/USD")
    html = c.get("/sleeves/duo", auth=AUTH).text
    assert re.findall(r'<tr data-asset="(\w+)"', html) == ["BTC", "SOL", "ETH", "_total"]


def test_the_form_says_the_list_is_fixed_and_the_order_breaks_ties(c):
    html = c.get("/sleeves/new", auth=AUTH).text
    assert "The asset list is fixed once created. A different list is a new Strategy." in html
    assert "The first asset is the main one, and the order breaks ties" in html


def test_a_research_only_market_is_refused_as_on_a_one_asset_create(c, store):  # noqa: F811
    """Advisor: the research-only simulated perp is never a Strategy on the fund (_form_params offers MARKETS only)."""
    from sleeve_fund import markets

    why = _error(_post(c, assets="ETH/USD", market=markets.PERP_FUNDING_STRESS))
    assert why == f"market: one of {', '.join(markets.MARKETS)}", why
    assert not any(s.name.startswith("duo") for s in store.sleeves())


# ---------------------------------------------------------------- CR M1: every per-asset check reaches every asset

def _refuses_eth(monkeypatch, attr, fake):
    from sleeve_fund.dashboard import app as app_mod

    real = getattr(app_mod, attr)
    monkeypatch.setattr(app_mod, attr, lambda *a, **k: fake(real, *a, **k))


def _gone(store):  # noqa: F811
    return not any(s.name.startswith("duo") for s in store.sleeves())


def test_a_second_asset_not_listed_on_the_venue_is_refused_named(c, store, monkeypatch):  # noqa: F811
    def listed(real, st, venue, pair):
        if pair == "ETH/USD":
            raise ValueError("instrument: the venue doesn't list it")
        return real(st, venue, pair)

    _refuses_eth(monkeypatch, "_check_listed", listed)
    assert _error(_post(c, assets="ETH/USD")) == "ETH/USD: instrument: the venue doesn't list it" and _gone(store)


def test_a_second_asset_whose_spread_fails_the_settings_is_refused_named(c, store, monkeypatch):  # noqa: F811
    def checked(real, cfg, half_spread):
        if cfg.instrument == "ETH/USD":  # checked as its own instrument (Advisor), not the first asset's
            raise ValueError("stop: inside the spread")
        return real(cfg, half_spread)

    _refuses_eth(monkeypatch, "_check_strategy_params", checked)
    assert _error(_post(c, assets="ETH/USD")) == "ETH/USD: stop: inside the spread" and _gone(store)


def test_a_second_asset_with_no_history_is_refused_named(c, store, monkeypatch):  # noqa: F811
    def gaps(real, venue, instrument, *a, **k):
        return "no history kept" if instrument == "ETH/USD" else real(venue, instrument, *a, **k)

    _refuses_eth(monkeypatch, "_warmup_gaps", gaps)
    assert _error(_post(c, assets="ETH/USD")).startswith("ETH/USD: no history kept") and _gone(store)


def test_a_second_asset_short_of_slower_candles_is_refused_named(c, store, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as app_mod

    def slower(real, venue, instrument, minutes, candles):  # the first asset has its candles
        if instrument == "ETH/USD":
            raise ValueError("1-hour candles: too few kept")

    _refuses_eth(monkeypatch, "_check_slower_history", slower)
    model = app_mod.REGISTRY["ping_pong"][0]
    monkeypatch.setattr(model, "slower_needs", lambda p: {60: 5}, raising=False)
    assert _error(_post(c, assets="ETH/USD")) == "ETH/USD: 1-hour candles: too few kept" and _gone(store)


def test_every_row_is_on_the_account_chosen(c, store):  # noqa: F811
    store.create_account("research", "paper")
    _post(c, assets="ETH/USD SOL/USD", account="research")
    assert {n: store.account_of(n) for n in store.legs("duo")} == dict.fromkeys(["duo", "duo~ETH", "duo~SOL"],
                                                                                 "research")


def test_a_mixed_quote_is_refused_naming_it_before_the_venue_is_asked(c, store, monkeypatch):  # noqa: F811
    """CR F1, F4."""
    asked = []
    _refuses_eth(monkeypatch, "_check_listed", lambda real, st, venue, pair: asked.append(pair))
    assert _error(_post(c, assets="ETH/EUR SOL/USD")) == "ETH/EUR: every asset shares the instrument's quote currency, USD"
    assert asked == [] and _gone(store)


def test_other_assets_on_a_create_from_a_backtest_are_refused(c, store):  # noqa: F811
    """CR F2: a backtest decided on its one instrument."""
    why = _error(_post(c, assets="ETH/USD", **{"from": "backtest"}))
    assert why.startswith("other assets: a backtest decided on its one instrument") and _gone(store)
