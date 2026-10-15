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
