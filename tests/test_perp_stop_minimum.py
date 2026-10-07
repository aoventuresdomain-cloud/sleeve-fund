"""The interim stop minimum for the hand-coded models on a perpetual (Independent Quant Advisor, 6 Oct; QA evidence in
quant-review/v2-p1/legacy-stops.md): a model that places no stop relies on the liquidation price alone, so it is
capped at 1x. Above that it is refused wherever it is created, edited or started (strategies.check_perp_stop), and a
row already in the store is refused at start with the reason, never changed. Backtests and studies still run."""

import pytest

from sleeve_fund.strategies import check_perp_stop, places_stop
from test_dashboard import AUTH, SAME, _new, client  # noqa: F401 - client is a fixture
from test_supervisor import FakePopen

PERP = {"market": "perp", "allow_short": True}
STOPLESS = [("rsi_bands", PERP), ("ping_pong", PERP), ("buy_and_hold", {"market": "perp"}),
            ("rsi_cross", {**PERP, "time_stop_bars": 6}),  # a time stop isn't a price stop
            ("trend_filter", {"market": "perp"}),  # side-only
            ("rsi_pullback", {"market": "perp"})]  # its ATR trail is checked at bar closes only: a gap goes through
STOPPED = [("dip_buy", PERP),  # its model's default ATR stop
           ("rsi_bands", {**PERP, "stop_loss": 0.02}), ("ping_pong", {**PERP, "stop_atr": 2.0}),
           ("rsi_cross", {**PERP, "stop_swing_bars": 10})]


@pytest.mark.parametrize("strategy, params", STOPLESS)
@pytest.mark.parametrize("profile", ["balanced", "aggressive"])
def test_a_stopless_model_on_a_perp_is_refused_above_1x_in_words(strategy, params, profile):
    assert not places_stop(strategy, params)
    with pytest.raises(ValueError, match="places no stop, so it is capped at 1x.*conservative"):
        check_perp_stop(strategy, params, profile)


@pytest.mark.parametrize("strategy, params", STOPLESS)
def test_a_stopless_model_on_a_perp_runs_at_1x(strategy, params):
    check_perp_stop(strategy, params, "conservative")


@pytest.mark.parametrize("strategy, params", STOPPED)
def test_a_model_that_places_a_stop_is_unaffected(strategy, params):
    assert places_stop(strategy, params)
    check_perp_stop(strategy, params, "aggressive")


@pytest.mark.parametrize("strategy", ["rsi_bands", "ping_pong", "buy_and_hold"])
def test_spot_is_unaffected(strategy):
    check_perp_stop(strategy, {}, "aggressive")


def test_new_strategy_and_a_settings_edit_refuse_it_and_change_nothing(client):  # noqa: F811
    c, store = client
    r = _new(c, name="rb-perp", strategy="rsi_bands", instrument="BTC/USDT", venue="binance", market="perp",
             allow_short="1", risk_profile="balanced")
    assert r.status_code == 303 and "places+no+stop" in r.headers["location"], r.headers["location"]
    assert store.sleeves() == []
    ok = _new(c, name="rb-perp", strategy="rsi_bands", instrument="BTC/USDT", venue="binance", market="perp",
              allow_short="1", risk_profile="conservative")
    assert ok.headers["location"] == "/sleeves/rb-perp"
    r = c.post("/sleeves/rb-perp/settings", data={"risk_profile": "aggressive", "reason": "more"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert "settings_error" in r.headers["location"] and "places+no+stop" in r.headers["location"]
    assert store.sleeve("rb-perp").risk_profile == "conservative"
    # With a stop set, the same edit goes through.
    r = c.post("/sleeves/rb-perp/settings", data={"risk_profile": "aggressive", "stop_loss_pct": "2", "reason": "stop"},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert "settings_error" not in r.headers["location"], r.headers["location"]
    assert store.sleeve("rb-perp").risk_profile == "aggressive"


def _existing(store, name="pp-perp", desired_state="stopped"):
    # Written before the minimum: the store holds it as it was, at 2x and with no stop.
    return store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                               starting_balance=10_000, params={"rise": 0.01, "dip": 0.005, **PERP}, venue="BINANCE",
                               risk_profile="balanced", desired_state=desired_state)


def test_a_start_from_the_dashboard_refuses_one_already_in_the_store(client):  # noqa: F811
    c, store = client
    _existing(store)
    r = c.post("/sleeves/pp-perp/command", data={"command": "start", "reason": "try it"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert "command_error" in r.headers["location"] and "places+no+stop" in r.headers["location"]
    s = store.sleeve("pp-perp")
    assert s.desired_state == "stopped" and s.risk_profile == "balanced" and "stop_loss" not in s.params


def test_the_supervisor_refuses_to_start_one_says_why_and_changes_none_of_its_settings(tmp_path, monkeypatch):
    from sleeve_fund import supervisor
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/t.db")
    _existing(store, desired_state="running")
    before = store.sleeve("pp-perp")
    started = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: started.append(a) or FakePopen())
    supervisor.Supervisor(store).step()
    s = store.sleeve("pp-perp")
    assert started == [] and s.desired_state == "stopped" and "places no stop" in s.status_reason
    assert (s.risk_profile, s.params) == (before.risk_profile, before.params)  # refused, not silently changed
    assert any(e["kind"] == "start_refused" and "capped at 1x" in e["message"] for e in store.events("pp-perp", limit=10))


def test_a_backtest_still_runs_so_the_risk_can_be_measured(prices, instrument):
    from sleeve_fund.research.runner import run_backtest

    res = run_backtest("ping_pong", prices.iloc[:60], instrument, PERP, risk_profile="balanced", half_spread=0)
    assert res.equity is not None


def test_a_backtest_of_a_setting_paper_would_refuse_says_so():
    """HoE, 6 Oct: a backtest above 1x of a stopless model runs, but is labelled; so are the entries the open-risk
    limit would have refused on paper."""
    from sleeve_fund.dashboard.preview import _risk
    from sleeve_fund.research.runner import PAPER_REFUSED, paper_refusal

    assert paper_refusal("ping_pong", PERP, "balanced") == PAPER_REFUSED
    assert paper_refusal("ping_pong", PERP, "conservative") is None and paper_refusal("ping_pong", {}, "balanced") is None
    note = _risk([], "balanced", refused=PAPER_REFUSED, binds=3)["note"]
    assert note.startswith("Would be refused on paper (stopless above 1x).")
    assert "Paper would refuse 3 entries (open risk over 5% of the book)." in note
    assert _risk([], "conservative")["note"] == ""
