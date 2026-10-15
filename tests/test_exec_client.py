"""EXEC-CLIENT (HoE plan 21:30, Advisor HEAD3 (1) and DRILL-VENUE-2): the exchange order client ships OFF. One config
reader for demo and live, its start-check guard, and the production rate limiter. No cell reaches the exchange: the
client is never built here, only its settings, its guard and its limiter."""
from datetime import UTC, datetime, timedelta

import pytest

from sleeve_fund.paper import exec_client as ec

DEMO, PROD = "api-demo.bybit.com", "api.bybit.com"
KEYS = {"BYBIT_DEMO_API_KEY": "k", "BYBIT_DEMO_API_SECRET": "s"}  # exec_target.KEYS: the mode's own pair
LIVE_KEYS = {"BYBIT_TRADE_API_KEY": "k", "BYBIT_TRADE_API_SECRET": "s"}


def _lines(env, **kw):
    return ec.guard(ec.exec_settings(env), **{"drill_go": False, "key_permissions": None, **kw})


def _fails(lines):
    return [what for status, what in lines if status == "FAIL"]


@pytest.mark.parametrize("mode", [None, "", "off", "none", " off "])
def test_the_client_is_off_when_the_mode_is_unset_off_or_none(mode):
    env = {**KEYS, "EXEC_HOST": DEMO} | ({} if mode is None else {"EXEC_MODE": mode})
    s = ec.exec_settings(env)
    assert s.mode == "off" and not s.on
    assert _fails(_lines(env)) == [] and any("OFF" in w for _, w in _lines(env))


@pytest.mark.parametrize("mode", ["OFF", "paper", "sandbox", "Demo-ish"])
def test_any_other_mode_is_on_and_fails_closed(mode):
    """exec_target N-KH1 (HoE): only unset, "off" or "none" is off; anything else fails closed, never a quiet OFF."""
    env = {**KEYS, "EXEC_HOST": DEMO, "EXEC_MODE": mode}
    assert ec.exec_settings(env).on and any("unknown mode" in w for w in _fails(_lines(env, drill_go=True)))


def test_demo_with_a_production_host_and_live_with_a_demo_host_are_refused():
    """DRILL-VENUE-2 MUST 4: the start check refuses a production host in demo mode and a demo host in live mode."""
    assert any("production host" in w for w in _fails(_lines({**KEYS, "EXEC_MODE": "demo", "EXEC_HOST": PROD},
                                                              drill_go=True)))
    assert any("demo host" in w for w in _fails(_lines({**LIVE_KEYS, "EXEC_MODE": "live", "EXEC_HOST": DEMO})))
    assert any("isn't a known" in w for w in _fails(_lines({**KEYS, "EXEC_MODE": "demo", "EXEC_HOST": "evil.example"},
                                                           drill_go=True)))


def test_live_is_refused_in_code_until_g2_and_demo_until_the_drill_go_record():
    """HEAD3 (1), EXEC-SPEC MUST 1: ON is gated in code. Live is refused before G2 whatever the host; demo needs the
    PM's journaled drill-go."""
    assert any("G2" in w for w in _fails(_lines({**LIVE_KEYS, "EXEC_MODE": "live", "EXEC_HOST": PROD})))
    assert any("drill-go" in w for w in _fails(_lines({**KEYS, "EXEC_MODE": "demo", "EXEC_HOST": DEMO})))
    assert _fails(_lines({**KEYS, "EXEC_MODE": "demo", "EXEC_HOST": DEMO}, drill_go=True)) == []


@pytest.mark.parametrize("perms, refused", [
    (None, False),  # not connected yet: checked at the first connect
    ({"trade": True, "withdraw": False, "read_only": False}, False),
    ({"trade": True, "withdraw": True, "read_only": False}, True),
    ({"trade": False, "withdraw": False, "read_only": True}, True),
    ({}, True),  # revoked: the venue lists no permission
])
def test_only_a_trade_only_key_passes(perms, refused):
    """DRILL-VENUE MUST 1, EXEC-SPEC Q2: trade-only keys. A key that can withdraw, or one that can't trade (read-only or
    revoked), is a FAIL."""
    env = {**KEYS, "EXEC_MODE": "demo", "EXEC_HOST": DEMO}
    assert bool(_fails(_lines(env, drill_go=True, key_permissions=perms))) is refused


def test_a_missing_key_is_refused_and_no_line_ever_shows_a_key():
    env = {"EXEC_MODE": "demo", "EXEC_HOST": DEMO, "BYBIT_DEMO_API_KEY": "sekrit-key-value"}
    lines = _lines(env, drill_go=True)
    assert any("BYBIT_DEMO_API_SECRET" in w for w in _fails(lines))
    assert not any("sekrit" in w for _, w in lines)


def test_the_adapter_config_is_built_by_one_function_for_demo_and_live_from_the_host_alone():
    """MUST 1: one client for demo and live; host and keys from config only, no demo-only path. The environment is
    read from the mode, the URLs from the host."""
    demo = ec.adapter_kwargs(ec.exec_settings({**KEYS, "EXEC_MODE": "demo", "EXEC_HOST": DEMO}), KEYS)
    live = ec.adapter_kwargs(ec.exec_settings({**LIVE_KEYS, "EXEC_MODE": "live", "EXEC_HOST": PROD}), LIVE_KEYS)
    assert set(demo) == set(live) and demo["api_key"] == live["api_key"] == "k"
    assert demo["base_url_http"] == f"https://{DEMO}" and live["base_url_http"] == f"https://{PROD}"
    assert (demo["environment"], live["environment"]) == ("DEMO", "MAINNET")
    with pytest.raises(ValueError, match="OFF"):
        ec.adapter_kwargs(ec.exec_settings({}), {})
    from sleeve_fund.exec_target import ExecTargetRefused

    with pytest.raises(ExecTargetRefused, match="production host"):  # never built from a mismatched target
        ec.adapter_kwargs(ec.exec_settings({**KEYS, "EXEC_MODE": "demo", "EXEC_HOST": PROD}), KEYS)


class _Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 15, 12, tzinfo=UTC)

    def __call__(self):
        return self.t


def test_the_limiter_holds_production_limits_and_never_holds_back_an_exit_a_stop_or_a_cancel():
    """DRILL-VENUE-2 (b): production per-endpoint limits in our own limiter, demo or not. Over the limit an entry
    waits at most its window and is then refused; an exit, a stop or a cancel always goes."""
    clock = _Clock()
    lim = ec.Limiter({"order/create": (2, 60.0, "https://x")}, now=clock)
    asked = clock.t
    assert lim.take("order/create", opening=True) == "send"
    assert lim.take("order/create", opening=True) == "send"
    assert lim.take("order/create", opening=True, since=asked) == "wait"
    assert lim.take("order/create", opening=False) == "send"  # an exit jumps the queue, over the limit or not
    clock.t += timedelta(seconds=5)
    assert lim.take("order/create", opening=True, since=asked) == "wait"  # 5 s is the default queue, inclusive
    clock.t += timedelta(seconds=1)
    assert lim.take("order/create", opening=True, since=asked) == "refuse"
    clock.t += timedelta(seconds=60)
    assert lim.take("order/create", opening=True) == "send"  # refilled


def test_every_limit_names_its_published_source():
    """HEAD3 (1) c: every venue figure names its documented source (no recorded responses ship tonight: the ON path's
    fixtures are drill prep, and join this list when built)."""
    assert ec.LIMITS and all(src.startswith("https://") for (_n, _per, src) in ec.LIMITS.values())
    assert set(ec.SOURCES) == {"rate limits", "orderLinkId"}
    assert all(src.startswith("https://bybit-exchange.github.io/docs/") for src in ec.SOURCES.values())


def test_the_start_check_shows_the_client_off_by_default_and_reads_the_drills_record_from_the_journal(tmp_path):
    """HEAD3 (1): the start check shows ON/OFF, host and key type; demo ON needs the PM's journaled drills record."""
    from sleeve_fund import start_check
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/j.db")
    off = start_check.check_exec({}, store.engine)
    assert not off.failed and "OFF" in off.lines[0][1]
    env = {**KEYS, "EXEC_MODE": "demo", "EXEC_HOST": DEMO}
    assert start_check.check_exec(env, store.engine).failed
    store.decide("someone", ec.DRILL_GO, "not the PM")
    store.decide("pm", ec.DRILLS_PASSED, "the wrong record: drills-passed is not the go to start them")
    assert start_check.check_exec(env, store.engine).failed
    store.decide("pm", ec.DRILL_GO, "go: demo drills")
    sec = start_check.check_exec(env, store.engine)
    assert not sec.failed and "demo mode on api-demo.bybit.com" in sec.lines[-1][1]


@pytest.mark.parametrize("mode", ["demo", "live"])
def test_a_paper_node_refuses_to_build_with_the_client_on_and_off_builds_only_the_simulated_client(monkeypatch, mode):
    """OFF byte-identical and no call to the exchange: an OFF node has the simulated client only and never touches the
    exchange adapter; ON is refused at the node until the drills (the exchange's market data isn't built yet)."""
    from nautilus_trader.adapters import bybit

    from sleeve_fund.paper import node as node_mod
    from sleeve_fund.paper.config import SleeveConfig
    from sleeve_fund.paper.safety import PaperSafetyError

    touched = []
    for name in ("BybitExecutionClientFactory", "BybitHttpClient", "BybitRawHttpClient"):
        monkeypatch.setattr(bybit, name, lambda *a, _n=name, **k: touched.append(_n))
    sleeve = SleeveConfig(name="t", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                          starting_balance=1000.0)
    monkeypatch.setenv("EXEC_MODE", mode)
    monkeypatch.setenv("EXEC_HOST", DEMO if mode == "demo" else PROD)
    with pytest.raises(PaperSafetyError, match="EXEC_MODE"):
        node_mod.build_node(sleeve, log_level="ERROR", asset_fetch=dict)
    monkeypatch.delenv("EXEC_MODE")
    built = node_mod.build_node(sleeve, log_level="ERROR", asset_fetch=dict)
    try:
        assert touched == [] and built.environment.name == "SANDBOX"
    finally:
        built.dispose()


@pytest.mark.parametrize("raw, want", [(None, 5.0), ("", 5.0), ("0", 0.0), ("2.5", 2.5), ("10", 10.0)])
def test_the_entry_queue_defaults_to_5_s_and_holds_0_to_10(raw, want):
    """EXEC-SPEC Q1: an entry over the limit queues at most entry_queue_max_s (default 5 s, bounded 0-10), then is
    refused and journaled."""
    env = {} if raw is None else {ec.QUEUE_ENV: raw}
    assert ec.entry_queue_max(env) == timedelta(seconds=want)


@pytest.mark.parametrize("raw", ["-1", "10.5", "30", "five"])
def test_an_entry_queue_outside_0_to_10_s_is_refused_at_load(raw):
    with pytest.raises(ValueError, match="0 to 10"):
        ec.entry_queue_max({ec.QUEUE_ENV: raw})


def test_a_queue_of_0_refuses_an_entry_over_the_limit_at_once():
    clock = _Clock()
    lim = ec.Limiter({"order/create": (1, 60.0, "https://x")}, now=clock, queue_max=timedelta(0))
    assert lim.take("order/create", opening=True) == "send"
    assert lim.take("order/create", opening=True, since=clock.t) == "wait"  # asked this instant: not yet waited
    clock.t += timedelta(milliseconds=1)
    assert lim.take("order/create", opening=True, since=clock.t - timedelta(milliseconds=1)) == "refuse"


@pytest.mark.parametrize("perms", [{"trade": False, "read_only": True}, {}])
def test_a_key_that_cannot_trade_while_the_venue_holds_a_position_is_critical(perms):
    """EXEC-SPEC Q2 MUST: ON with a read-only or revoked key while the venue holds a position or an order raises a
    CRITICAL "cannot protect"; with nothing held it is a plain FAIL."""
    env = {**KEYS, "EXEC_MODE": "demo", "EXEC_HOST": DEMO}
    held = _fails(_lines(env, drill_go=True, key_permissions=perms, venue_holds=True))
    assert any(w.startswith("CRITICAL") and "cannot protect" in w for w in held)
    assert ec.cannot_protect(perms, venue_holds=True).startswith("CRITICAL")
    assert not any(w.startswith("CRITICAL") for w in _fails(_lines(env, drill_go=True, key_permissions=perms)))
    assert ec.cannot_protect({"trade": True}, venue_holds=True) is None


@pytest.mark.parametrize("cid, ok", [
    ("a" * 36, True), ("O-20261015-121800-PAPER-001-12", True), ("a" * 37, False), ("orphan.close", False),
    ("orphan close", False), ("", False),
])
def test_an_order_link_id_fits_the_venues_36_characters_letters_digits_dash_and_underscore(cid, ok):
    """EXEC-SPEC MUST 5: the id the venue keeps for our order (its orderLinkId) is at most 36 letters, digits, dashes
    and underscores; the client refuses any other before it is sent."""
    assert ec.link_id_ok(cid) is ok


def test_the_drills_passed_record_is_what_lets_the_funds_paper_strategies_onto_the_demo_client(tmp_path):
    """EXEC-SPEC MUST 1: two records. drill-go lets demo ON (the drills); drills-passed lets the fund's paper Strategies
    onto the demo client. Until the exchange's market data path is built a paper node refuses ON either way."""
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/j.db")
    assert not ec.recorded(store.engine, ec.DRILL_GO) and not ec.recorded(store.engine, ec.DRILLS_PASSED)
    store.decide("pm", ec.DRILL_GO, "go: demo drills")
    assert ec.recorded(store.engine, ec.DRILL_GO) and not ec.recorded(store.engine, ec.DRILLS_PASSED)


@pytest.mark.parametrize("var", ["EXEC_API_KEY", "EXEC_API_SECRET"])
def test_a_paper_node_refuses_the_order_clients_keys_as_any_venue_credential(var):
    """QT3 fx1: paper is keyless, so the client's own key or secret in the environment refuses the node too."""
    from sleeve_fund.paper.safety import PaperSafetyError, assert_keyless

    with pytest.raises(PaperSafetyError, match=var):
        assert_keyless({var: "x"})
    assert_keyless({"EXEC_MODE": "off", "EXEC_HOST": DEMO, "EXEC_ENTRY_QUEUE_MAX_S": "5"})


def test_the_start_check_shows_host_and_key_once_with_the_gates_in_the_order_client_section(tmp_path):
    """One reader (PE4's exec_target): the start check's "Order client" section carries the host and key line and the
    client's gates as its extra lines, so nothing shows twice; check_exec stays as the guard on its own."""
    from sleeve_fund import start_check
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/j.db")
    env = {**KEYS, "EXEC_MODE": "demo", "EXEC_HOST": DEMO}
    sec = start_check.check_exec_target(env, start_check.exec_gates(env, store.engine))
    words = [w for _, w in sec.lines]
    assert sec.failed and sum(DEMO in w for w in words) == 1 and any("drill-go" in w for w in words)
    store.decide("pm", ec.DRILL_GO, "go: demo drills")
    assert not start_check.check_exec_target(env, start_check.exec_gates(env, store.engine)).failed
    assert start_check.exec_gates({}, store.engine) == []
