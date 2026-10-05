"""A model sized by a target weight (Donchian, trend filter with vol_target) is refused on a perpetual until order
sizing is rebuilt (review round 13, E13-6): a perp entry ignored the weight and opened at the full cap. One check
(strategies.check_perp_sizing), called wherever such a model could start, backtest or be studied."""

import pandas as pd
import pytest

from sleeve_fund.strategies import PERP_WEIGHT_REFUSAL, check_perp_sizing
from test_dashboard import AUTH, SAME, _new, client  # noqa: F401 - client is a fixture
from test_supervisor import FakePopen

PERP = {"market": "perp"}
REFUSED = [("donchian", PERP), ("donchian", {"market": "perp-venue-fees"}), ("trend_filter", {**PERP, "vol_target": 0.4})]
ALLOWED = [("ping_pong", {**PERP, "allow_short": True}), ("rsi_bands", {**PERP, "allow_short": True}),
           ("rsi_cross", {**PERP, "allow_short": True}), ("dip_buy", {**PERP, "allow_short": True}),
           ("trend_filter", PERP), ("donchian", {}), ("trend_filter", {"vol_target": 0.4})]


@pytest.mark.parametrize("strategy, params", REFUSED)
def test_a_weight_sized_model_on_a_perp_is_refused_in_words(strategy, params):
    with pytest.raises(ValueError, match=PERP_WEIGHT_REFUSAL):
        check_perp_sizing(strategy, params)


@pytest.mark.parametrize("strategy, params", ALLOWED)
def test_side_only_models_and_spot_are_unaffected(strategy, params):
    check_perp_sizing(strategy, params)


def _donchian_form(**over):
    return {"strategy": "donchian", "instrument": "BTC/USDT", "venue": "binance", "market": "perp",
            "bar_spec": "1-DAY-LAST-EXTERNAL", **over}


def test_new_strategy_and_clone_refuse_it_and_create_nothing(client):  # noqa: F811
    c, store = client
    for form in (_donchian_form(name="dc-perp"), _donchian_form(name="dc-clone", **{"from": "clone", "source": "x"}),
                 {"name": "tf-perp", "instrument": "BTC/USDT", "venue": "binance", "market": "perp",
                  "p_trend_filter__vol_target": "0.4"}):
        r = _new(c, **form)
        assert r.status_code == 303 and "sized+by+weight" in r.headers["location"], r.headers["location"]
    assert store.sleeves() == []
    ok = _new(c, name="dc-spot", strategy="donchian", bar_spec="1-DAY-LAST-EXTERNAL")  # spot: as before
    assert ok.headers["location"] == "/sleeves/dc-spot"


def test_the_backtest_page_refuses_it(client):  # noqa: F811
    from sleeve_fund.dashboard.app import _backtest_args

    with pytest.raises(ValueError, match=PERP_WEIGHT_REFUSAL):
        _backtest_args(_donchian_form())
    assert _backtest_args({**_donchian_form(), "strategy": "rsi_bands"})["strategy"] == "rsi_bands"


def _existing(store, name="dc-perp", desired_state="stopped"):
    # Written before the guard: the store holds it as it was.
    return store.create_sleeve(name=name, strategy="donchian", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                               starting_balance=10_000, params=PERP, venue="BINANCE", desired_state=desired_state)


def test_start_and_a_settings_edit_refuse_one_already_in_the_store(client):  # noqa: F811
    c, store = client
    _existing(store)
    r = c.post("/sleeves/dc-perp/command", data={"command": "start", "reason": "try it"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert "command_error" in r.headers["location"] and "sized+by+weight" in r.headers["location"]
    assert store.sleeve("dc-perp").desired_state == "stopped"
    r = c.post("/sleeves/dc-perp/settings", data={"risk_profile": "balanced", "reason": "tighter"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert "settings_error" in r.headers["location"] and "sized+by+weight" in r.headers["location"]


def test_the_supervisor_refuses_to_start_or_restart_one_and_says_why(tmp_path, monkeypatch):
    from sleeve_fund import supervisor
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/t.db")
    _existing(store, desired_state="running")  # e.g. running before the guard: the next restart is refused
    started = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: started.append(a) or FakePopen())
    sup = supervisor.Supervisor(store)
    sup.step()
    s = store.sleeve("dc-perp")
    assert started == [] and s.desired_state == "stopped" and PERP_WEIGHT_REFUSAL in s.status_reason
    assert any(e["kind"] == "start_refused" for e in store.events("dc-perp", limit=10))
    sup.step()
    assert started == []  # stopped, so not refused again and again
    # A start that only sells what it holds (the kill switch, a PM close) still goes ahead.
    store.command("dc-perp", "flatten", "Book kill switch: test", actor="PM")
    store.set_desired_state("dc-perp", "running")
    sup.step()
    assert len(started) == 1


def test_a_backtest_and_a_study_refuse_it_before_reading_any_data():
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.donchian import SPEC
    from sleeve_fund.venues import venue

    perp = venue("binance").instrument("BTC", "USDT")
    with pytest.raises(ValueError, match=PERP_WEIGHT_REFUSAL):
        run_backtest("donchian", pd.DataFrame(), perp, PERP)
    with pytest.raises(ValueError, match=PERP_WEIGHT_REFUSAL):  # a perpetual venue holds every run as a perp
        run_study(SPEC, pd.DataFrame(), perp, "test", ledger=None)
