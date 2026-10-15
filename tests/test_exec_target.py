"""Head 3: the start check's key and host line (test-network / demo), one reader shared with EXEC-CLIENT's guard
(sleeve_fund.exec_target; PE3 agreed the interface 15 Oct ~12:30 UK). It shows the key type (demo or live) and the host,
and refuses a mismatch: a production host in demo mode, a demo host in live mode, a host off the closed list, a key or
secret missing, the other mode's key set alongside. Off by default: the client stays the simulated paper one.

Red first on 47c2e498 (head 2): sleeve_fund.exec_target missing, no "Order client" section."""

from __future__ import annotations

import subprocess
import sys

import pytest

from sleeve_fund import exec_target as et

SECRET_VALUE = "s3cr3t-value-never-printed"
DEMO = {"EXEC_MODE": "demo", "EXEC_HOST": "api-demo.bybit.com", "BYBIT_DEMO_API_KEY": SECRET_VALUE,
        "BYBIT_DEMO_API_SECRET": SECRET_VALUE}
LIVE = {"EXEC_MODE": "live", "EXEC_HOST": "api.bybit.com", "BYBIT_TRADE_API_KEY": SECRET_VALUE,
        "BYBIT_TRADE_API_SECRET": SECRET_VALUE}


# ---- 1. off by default -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env", [{}, {"EXEC_MODE": ""}, {"EXEC_MODE": "  "}])
def test_unset_is_off(env):
    assert et.read(env) is None
    et.assert_ok(env)  # nothing to refuse while off


def test_off_is_a_note_in_its_own_section():
    from sleeve_fund.start_check import NOTE, check_exec_target

    sec = check_exec_target({})
    assert sec.title == "Order client" and sec.lines == [(NOTE, "off (simulated paper client)")] and not sec.failed


# ---- 2. each mismatch fails, named -------------------------------------------------------------------------------------
@pytest.mark.parametrize("env, words", [
    ({**DEMO, "EXEC_HOST": "api.bybit.com"}, "a production host (api.bybit.com) in demo mode"),
    ({**LIVE, "EXEC_HOST": "api-demo.bybit.com"}, "a demo host (api-demo.bybit.com) in live mode"),
    ({**DEMO, "EXEC_HOST": "evil.example.com"}, "host evil.example.com isn't a known BYBIT host"),
    ({**DEMO, "EXEC_HOST": ""}, "no host set (EXEC_HOST)"),
    ({k: v for k, v in DEMO.items() if k != "BYBIT_DEMO_API_KEY"}, "the demo key isn't set (BYBIT_DEMO_API_KEY)"),
    ({k: v for k, v in DEMO.items() if k != "BYBIT_DEMO_API_SECRET"},
     "the demo secret isn't set (BYBIT_DEMO_API_SECRET)"),
    ({**DEMO, "BYBIT_TRADE_API_KEY": "x"}, "a live key is set alongside the demo one (BYBIT_TRADE_API_KEY)"),
    ({**LIVE, "BYBIT_DEMO_API_SECRET": "x"}, "a demo key is set alongside the live one (BYBIT_DEMO_API_SECRET)"),
    ({**DEMO, "EXEC_MODE": "paper"}, "unknown mode 'paper' (EXEC_MODE is demo or live)"),
    ({**DEMO, "EXEC_VENUE": "KRAKEN"}, "unknown venue 'KRAKEN' for the order client"),
])
def test_each_mismatch_is_refused_by_name_and_no_key_value_is_printed(env, words):
    from sleeve_fund.start_check import FAIL, check_exec_target

    t = et.read(env)
    assert words in et.problems(t), et.problems(t)
    sec = check_exec_target(env)
    assert sec.failed and (FAIL, words) in sec.lines, sec.lines
    with pytest.raises(et.ExecTargetRefused, match="refused") as refused:
        et.assert_ok(env)
    assert words in str(refused.value)
    assert SECRET_VALUE not in repr(t) and SECRET_VALUE not in str(sec.lines) and SECRET_VALUE not in str(refused.value)


def test_mode_and_venue_are_read_whatever_their_case_and_spaces():
    t = et.read({**DEMO, "EXEC_MODE": " Demo ", "EXEC_VENUE": "bybit", "EXEC_HOST": " API-DEMO.bybit.com "})
    assert (t.mode, t.venue, t.host) == ("demo", "BYBIT", "api-demo.bybit.com") and et.problems(t) == []


def test_a_host_with_a_scheme_path_or_port_is_not_a_host():
    t = et.read({**DEMO, "EXEC_HOST": "https://api-demo.bybit.com/v5"})
    assert any("isn't a bare host" in p for p in et.problems(t))


# ---- 3. a clean setup --------------------------------------------------------------------------------------------------
def test_a_clean_demo_setup_passes_with_the_key_type_and_host_and_notes_the_scope_unchecked():
    from sleeve_fund.start_check import NOTE, PASS, check_exec_target

    t = et.read(DEMO)
    assert et.problems(t) == [] and et.describe(t) == "demo key, host api-demo.bybit.com"
    et.assert_ok(DEMO)
    sec = check_exec_target(DEMO)
    assert sec.lines == [(PASS, "demo key, host api-demo.bybit.com"), (NOTE, et.SCOPE_UNCHECKED)] and not sec.failed


def test_a_clean_live_setup_reads_as_live_and_the_live_key_check_still_fails_it():
    """The line itself passes a matching live setup; paper-only is the keys section's job, which fails any live key."""
    from sleeve_fund.start_check import PASS, check_exec_target, check_keys

    assert check_exec_target(LIVE).lines[0] == (PASS, "live key, host api.bybit.com")

    class _Engine:  # no live key reported by the supervisor
        def connect(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a):
            return []

    assert check_keys(LIVE, _Engine()).failed


# ---- 4. the reader is the guard's too, and imports nothing that trades -------------------------------------------------
def test_the_module_imports_nothing_that_opens_a_connection_or_places_an_order():
    code = ("import sys; import sleeve_fund.exec_target; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('nautilus_trader', 'urllib3', 'requests', 'httpx', "
            "'websockets', 'aiohttp') or m in ('sleeve_fund.mirror', 'sleeve_fund.paper.node')]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]", out


# ---- 5. in the whole check, both engines ---------------------------------------------------------------------------------
def _journal_url(tmp_path):
    import os

    from sleeve_fund.store import Store, make_engine, metadata

    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        engine = make_engine(url)
        metadata.drop_all(engine)
        Store(engine=engine)
        engine.dispose()
        return url
    url = f"sqlite:///{tmp_path}/j.db"
    Store(url).engine.dispose()
    return url


def test_the_section_is_on_the_page_and_a_mismatch_fails_the_whole_check(tmp_path):
    from sleeve_fund import start_check

    url = _journal_url(tmp_path)
    alerts = {"ALERT_WEBHOOK_URL": "https://hooks.example.com/x", "HEALTHCHECK_PING_URL": "https://hc.example.com/y"}
    ok_off, page_off = start_check.run(url, environ=alerts)
    assert "Order client: PASS" in page_off and "off (simulated paper client)" in page_off
    _, page_bad = start_check.run(url, environ={**alerts, **DEMO, "EXEC_HOST": "api.bybit.com"})
    assert "Order client: FAIL" in page_bad and "a production host (api.bybit.com) in demo mode" in page_bad
    assert "Start check: FAIL" in page_bad
    # off, the section is a note: it never fails the check
    assert not start_check.check_exec_target(alerts).failed and ok_off == ("Start check: PASS" in page_off)


# ---- Advisor KEYHOST MUST (line 5): a FAIL refuses starts and entries only, never exits or stops -----------------------
def test_a_key_host_fail_with_an_open_position_refuses_entries_and_the_stop_still_fires(tmp_path, monkeypatch):
    """As the A7 rule: a start-check FAIL here is a CHOKE cause (entries and starts refused, named), and a Strategy
    holding on a restart still has its stop, exits and a flatten: nothing halts or stops it."""
    from sleeve_fund.paper import runtime
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/j.db")
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, desired_state="running")
    store.record_fill("s1", side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id="in", trade_id="in")
    for k, v in {**DEMO, "EXEC_HOST": "api.bybit.com"}.items():
        monkeypatch.setenv(k, v)
    why = runtime.exec_target_block()
    assert why is not None and why.code == "exec_target" and "a production host (api.bybit.com) in demo mode" in why
    blocked, said = runtime.entry_blocked(store, "s1", starting=True)  # the dashboard's Start reads this too
    assert blocked and "exec_target" in said.codes
    rt = runtime.SleeveRuntime(store, "s1")
    rt.on_start(0.008)
    blocked, said = rt.entry_blocked()
    assert blocked and "exec_target" in said.codes
    assert rt.wiped_out is False and store.sleeve("s1").status != "halted"  # nothing halts or stops it
    # a CHOKE cause, so entries only: CHOKE never gates a stop, a reduce-only order, the PM's close or a liquidation
    assert why.code in runtime.CODES and "Exits and stops still run" in why
    for k in DEMO:
        monkeypatch.delenv(k)
    monkeypatch.delenv("EXEC_HOST", raising=False)
    assert runtime.exec_target_block() is None and not runtime.entry_blocked(store, "s1", starting=True)[0]


def test_demo_keys_with_no_host_variable_at_all_fail_by_name_never_a_default_host():
    """HoQA line 2: keys present, EXEC_HOST missing entirely: FAIL by name, no fallback host."""
    env = {k: v for k, v in DEMO.items() if k != "EXEC_HOST"}
    t = et.read(env)
    assert t.host == "" and "no host set (EXEC_HOST)" in et.problems(t)
    with pytest.raises(et.ExecTargetRefused, match="no host set"):
        et.assert_ok(env)


def test_the_order_clients_own_guard_lines_join_the_section_and_a_fail_among_them_fails_it():
    """PE3's hook: EXEC-CLIENT's guard lines (drill go, the G2 lock) go in this one section, after the key and host."""
    from sleeve_fund.start_check import FAIL, NOTE, PASS, check_exec_target

    sec = check_exec_target(DEMO, extra=[(PASS, "guard line")])
    assert sec.lines[-1] == (PASS, "guard line") and sec.lines[0][1] == "demo key, host api-demo.bybit.com"
    assert not sec.failed
    off = check_exec_target({}, extra=[(FAIL, "guard refused")])
    assert off.lines == [(NOTE, et.OFF), (FAIL, "guard refused")] and off.failed


def test_an_unreadable_target_fails_closed_in_choke(monkeypatch):
    from sleeve_fund.paper import runtime

    def boom(_):
        raise RuntimeError("x")

    monkeypatch.setattr(et, "read", boom)
    why = runtime.exec_target_block({"EXEC_MODE": "demo"})
    assert why is not None and why.code == "exec_target" and "couldn't be read" in why
